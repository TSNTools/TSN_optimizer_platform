"""
TSN Defragmentation Environment for DRL

Observation (state):
    stream_placement : (N, P) normalised offsets (+1 encoding)
    stream_features  : (N, 5) [offset, duration, period, latency, hop_cost]
    port_metrics     : (P, 3) [utilization, useless_gap_ratio, avg_useful_gap]
    adjacency_matrix : (2, E_max) line-graph edge list

Action:
    (N, P) matrix – action[i,j] = desired_offset / cycle + 1   (0 = unused)
    Decoding supports:
      - multi-move mode: many streams can move in one step,
      - single-move mode: at most one stream move per step.
    In both cases, downstream offsets are auto-adjusted with hop delays.

Reward:
    R = delta_fragmentation  –  move_penalty * (n_moved > 0)
    delta_frag = (gaps_old - gaps_new) / max(1, gaps_old)

    Additional mode: 'paper_simple'
    r_t = w1*DeltaAdmissionFit + w2*DeltaFragScore
          - w3*ShiftCost - w4*MovedStreams - w5*1[invalid]

All internal times are in **nanoseconds** (matching TSNScheduling).
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional

from schedule import Schedule, Slot
from tsn_scheduling_problem import TSNScheduling, MBPS_2_BPNS
from usecase_generator import Usecase


class TSNDefragEnv(gym.Env):
    """Gymnasium environment for TSN schedule defragmentation."""

    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        problem: TSNScheduling,
        usecase: Usecase,
        gap_threshold_ratio: float = 0.01,
        move_penalty: float = 0.01,
        max_edges: int = 200,
        fragmentation_metric: str = "hybrid",
        single_move_per_step: bool = False,
        reward_mode: str = "simple",
        init_from_usecase_schedule: bool = True,
        reward_weights: Optional[Dict[str, float]] = None,
        admission_template_count: int = 8,
    ):
        super().__init__()

        self.problem = problem
        self.usecase = usecase
        self.move_penalty = move_penalty
        self.max_edges = max_edges
        self.fragmentation_metric = fragmentation_metric
        self.single_move_per_step = single_move_per_step
        self.reward_mode = reward_mode
        self.init_from_usecase_schedule = init_from_usecase_schedule
        self.admission_template_count = max(1, int(admission_template_count))

        # Weights for reward_mode='paper_simple'
        default_weights = {
            "w1_admission": 2.0,
            "w2_frag": 1.0,
            "w3_shift": 0.05,
            "w4_moved": 0.2,
            "w5_invalid": 100.0,
        }
        self.reward_weights = dict(default_weights)
        if reward_weights:
            self.reward_weights.update(reward_weights)

        # ── precompute everything inside the instance context ──────────
        with self.problem.use_instance(self.usecase):
            self.n_ports: int = self.problem.n_ports
            self.n_streams: int = self.problem.n_streams

            # macro-cycle = LCM of all stream intervals (ns)
            self.cycle_ns: int = self.problem.macro_cycle

            # gap threshold (ns)
            self.gap_threshold_ns: float = gap_threshold_ratio * self.cycle_ns

            # ── port cache (keyed by 1-based port id) ──────────────────
            self._port_name: Dict[int, str] = {}
            self._port_src_node: Dict[int, str] = {}
            self._port_dst_node: Dict[int, str] = {}
            self._port_speed: Dict[int, float] = {}       # Mbps

            for pid, pinfo in self.problem.ports.items():
                self._port_name[pid] = self.problem.port_id_to_name[pid]
                self._port_src_node[pid] = pinfo.get("sourceNode", "")
                self._port_dst_node[pid] = pinfo.get("destNode", "")
                self._port_speed[pid] = pinfo.get("speed", 1000)

            # ── stream cache (0-based index) ───────────────────────────
            self.stream_info: List[Dict] = []
            enriched_streams = self.problem.streams          # 1-based port IDs

            for s_idx in range(self.n_streams):
                raw = self.usecase.streams[s_idx]            # original dict
                enr = enriched_streams[s_idx]

                # path = list of 1-based port IDs (last hop removed by problem.streams)
                path: List[int] = []
                if "pathLists" in enr and enr["pathLists"]:
                    path = list(enr["pathLists"][0])

                interval_ns = int(raw["interval"])
                frame_bits = TSNScheduling._frame_size(raw)
                max_lat_ns = raw.get("maxLatency", 1_000_000)

                # per-port tx duration (ns)
                port_dur: Dict[int, float] = {}
                for pid in path:
                    spd = self._port_speed.get(pid, 1000)
                    port_dur[pid] = frame_bits / (spd * MBPS_2_BPNS)

                # hop delays between consecutive path ports (ns)
                hop_delays: List[float] = []
                total_hop_cost = 0.0
                for k in range(len(path) - 1):
                    src_nm = self._port_name.get(path[k], "")
                    dst_nm = self._port_name.get(path[k + 1], "")
                    try:
                        delay = self.problem.hop_cost_eg2eg(
                            src_nm, dst_nm, frame_bits
                        )
                    except Exception:
                        delay = port_dur.get(path[k], 0) + 100  # fallback
                    hop_delays.append(delay)
                    total_hop_cost += delay

                self.stream_info.append(
                    {
                        "path": path,
                        "interval_ns": interval_ns,
                        "frame_bits": frame_bits,
                        "port_dur_ns": port_dur,
                        "hop_delays": hop_delays,
                        "total_hop_cost_ns": total_hop_cost,
                        "max_latency_ns": max_lat_ns,
                    }
                )

            # normalisation constants
            self.max_latency_ns: float = max(
                (s["max_latency_ns"] for s in self.stream_info), default=1
            )

            # line-graph adjacency
            self._adjacency = self._build_adjacency()

            # Candidate templates used by lightweight admission-fit estimate.
            self.admission_templates = self._build_admission_templates(
                self.admission_template_count
            )

        # ── Gym spaces ─────────────────────────────────────────────────
        self.action_space = spaces.Box(
            low=0.0, high=2.0,
            shape=(self.n_streams, self.n_ports),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "stream_placement": spaces.Box(
                    0.0, 2.0, (self.n_streams, self.n_ports), np.float32
                ),
                "stream_features": spaces.Box(
                    0.0, np.inf, (self.n_streams, 5), np.float32
                ),
                "port_metrics": spaces.Box(
                    0.0, 1.0, (self.n_ports, 3), np.float32
                ),
                "adjacency_matrix": spaces.Box(
                    -1, self.n_ports, (2, self.max_edges), np.int64
                ),
            }
        )

        # ── mutable state (populated on reset) ────────────────────────
        self.port_schedules: Dict[int, Schedule] = {}           # 0-based idx → Schedule
        self.stream_offsets: Dict[int, Dict[int, float]] = {}   # s_idx → {pid → offset_ns}

    # ================================================================== #
    # Line-graph adjacency
    # ================================================================== #
    def _build_adjacency(self) -> np.ndarray:
        """
        Edge A→B in the line graph iff destNode(A) == sourceNode(B),
        i.e. a frame exiting port A can be forwarded through port B.
        """
        pids = sorted(self._port_src_node.keys())
        edges = []
        for pA in pids:
            dst_a = self._port_dst_node.get(pA)
            if not dst_a:
                continue
            for pB in pids:
                if pA != pB and self._port_src_node.get(pB) == dst_a:
                    edges.append((pA - 1, pB - 1))          # 0-based

        adj = np.full((2, self.max_edges), -1, dtype=np.int64)
        for i, (s, d) in enumerate(edges[: self.max_edges]):
            adj[0, i] = s
            adj[1, i] = d
        return adj

    # ================================================================== #
    # Schedule helpers
    # ================================================================== #
    def _place_stream(self, s_idx: int, first_offset_ns: float) -> None:
        """
        Place stream *s_idx* so that its first port starts at
        *first_offset_ns*.  Downstream port offsets are derived by
        adding cumulative hop delays (constraint enforcement).
        Any existing slots for this stream are removed first.
        """
        sinfo = self.stream_info[s_idx]
        path = sinfo["path"]
        if not path:
            return

        # 1) remove old slots on every port of this stream's path
        for pid in path:
            pidx = pid - 1
            if pidx in self.port_schedules:
                self.port_schedules[pidx] = Schedule(
                    cycle_ns=self.cycle_ns,
                    iterable=[
                        s for s in self.port_schedules[pidx]
                        if s.stream_id != s_idx
                    ],
                )

        # 2) compute per-port offsets with hop-delay propagation
        offsets: Dict[int, float] = {}
        cum_delay = 0.0
        for k, pid in enumerate(path):
            offset = first_offset_ns + cum_delay
            dur = sinfo["port_dur_ns"].get(pid, 0)
            offset = max(0.0, min(float(self.cycle_ns - dur), offset))
            offsets[pid] = offset
            if k < len(sinfo["hop_delays"]):
                cum_delay += sinfo["hop_delays"][k]

        # 3) insert new slots
        for pid, offset in offsets.items():
            pidx = pid - 1
            if pidx not in self.port_schedules:
                self.port_schedules[pidx] = Schedule(cycle_ns=self.cycle_ns)

            dur = sinfo["port_dur_ns"].get(pid, 0)
            slot = Slot(
                start_ns=int(round(offset)),
                end_ns=int(round(offset + dur)),
                period_ns=sinfo["interval_ns"],
                queue_id=pidx,
                stream_id=s_idx,
                port_id=pid,
            )
            self.port_schedules[pidx].append(slot)

        self.stream_offsets[s_idx] = offsets

    def _decode_first_offset_from_action(
        self, s_idx: int, action_row: np.ndarray
    ) -> Optional[float]:
        """
        Decode a stream-level first offset from a per-port action row.

        For every non-zero action on a path port, we back-project the port
        offset to an implied first-hop offset by subtracting the cumulative
        hop delay to that port. The decoded stream move is the average of all
        implied first-hop offsets (clamped to a valid range).
        """
        sinfo = self.stream_info[s_idx]
        path = sinfo["path"]
        if not path:
            return None

        implied_first_offsets: List[float] = []
        cum_delay = 0.0
        for hop_idx, pid in enumerate(path):
            pidx = pid - 1
            if 0 <= pidx < self.n_ports:
                act_val = action_row[pidx]
                if act_val > 0:
                    desired_port_offset = (act_val - 1.0) * self.cycle_ns
                    implied_first_offsets.append(desired_port_offset - cum_delay)

            if hop_idx < len(sinfo["hop_delays"]):
                cum_delay += sinfo["hop_delays"][hop_idx]

        if not implied_first_offsets:
            return None

        first_pid = path[0]
        first_dur = sinfo["port_dur_ns"].get(first_pid, 0)
        first_offset = float(np.mean(implied_first_offsets))
        return max(0.0, min(float(self.cycle_ns - first_dur), first_offset))

    def _decode_first_offset_from_port_action(
        self, s_idx: int, port_idx: int, act_val: float
    ) -> Optional[float]:
        """
        Decode stream first-hop offset from a single (stream, port) action.
        Returns None when the selected port is not on the stream path.
        """
        sinfo = self.stream_info[s_idx]
        path = sinfo["path"]
        pid = port_idx + 1
        if pid not in path or act_val <= 0:
            return None

        hop_idx = path.index(pid)
        cum_delay = float(np.sum(sinfo["hop_delays"][:hop_idx]))
        desired_port_offset = (float(act_val) - 1.0) * self.cycle_ns
        first_pid = path[0]
        first_dur = sinfo["port_dur_ns"].get(first_pid, 0)
        first_offset = desired_port_offset - cum_delay
        return max(0.0, min(float(self.cycle_ns - first_dur), first_offset))

    def _create_initial_schedule(self) -> None:
        """
        Simple initial placement: every stream starts at offset 0 on its
        first port; downstream offsets follow from hop delays.
        """
        self.port_schedules = {}
        self.stream_offsets = {}
        for s_idx in range(self.n_streams):
            self._place_stream(s_idx, first_offset_ns=0.0)

    def _create_initial_schedule_from_usecase(self) -> bool:
        """
        Try to initialize internal state from schedule data already present in usecase.

        Returns True when a schedule was loaded, False when unavailable.
        """
        try:
            port_name_to_id, _ = self.usecase.port_name_id_mapping
            imported_schedule = Schedule.from_usecase(self.usecase, port_name_to_id)
        except Exception:
            return False

        self.port_schedules = {}
        self.stream_offsets = {}

        for slot in imported_schedule:
            if slot.port_id is None:
                continue
            pidx = slot.port_id - 1
            if pidx not in self.port_schedules:
                self.port_schedules[pidx] = Schedule(cycle_ns=self.cycle_ns)
            self.port_schedules[pidx].append(slot)

            if slot.stream_id is not None:
                s_idx = int(slot.stream_id)
                if s_idx not in self.stream_offsets:
                    self.stream_offsets[s_idx] = {}
                # Keep earliest offset per stream/port as canonical placement.
                prev = self.stream_offsets[s_idx].get(slot.port_id)
                if prev is None or slot.start_ns < prev:
                    self.stream_offsets[s_idx][slot.port_id] = float(slot.start_ns)

        return True

    # ================================================================== #
    # Observation builders
    # ================================================================== #
    def _get_stream_placement(self) -> np.ndarray:
        """(N, P)  placement[i,j] = offset / cycle + 1  or 0 if unused."""
        pl = np.zeros((self.n_streams, self.n_ports), dtype=np.float32)
        for s_idx, offsets in self.stream_offsets.items():
            for pid, off in offsets.items():
                j = pid - 1
                if 0 <= j < self.n_ports:
                    pl[s_idx, j] = off / self.cycle_ns + 1.0
        return pl

    def _get_stream_features(self) -> np.ndarray:
        """(N, 5) [offset, duration, period, latency, hop_cost] normalised."""
        feat = np.zeros((self.n_streams, 5), dtype=np.float32)
        for s_idx, sinfo in enumerate(self.stream_info):
            path = sinfo["path"]
            if not path:
                continue
            first_pid = path[0]
            off = self.stream_offsets.get(s_idx, {}).get(first_pid, 0)
            dur = sinfo["port_dur_ns"].get(first_pid, 0)

            feat[s_idx, 0] = off / self.cycle_ns                          # offset
            feat[s_idx, 1] = dur / self.cycle_ns                          # duration
            feat[s_idx, 2] = sinfo["interval_ns"] / self.cycle_ns         # period
            feat[s_idx, 3] = sinfo["max_latency_ns"] / self.max_latency_ns  # latency
            feat[s_idx, 4] = sinfo["total_hop_cost_ns"] / self.cycle_ns   # hop cost
        return feat

    # ── per-port gap analysis ──────────────────────────────────────────
    def _port_gaps(self, pidx: int) -> List[float]:
        """Return gap durations (ns) on 0-based port *pidx*."""
        if pidx not in self.port_schedules or not self.port_schedules[pidx]:
            return []                                          # no streams → no gaps

        occs = []
        for slot in self.port_schedules[pidx]:
            occs.extend(slot.all_occurrences(self.cycle_ns))
        if not occs:
            return []
        occs.sort(key=lambda s: s.start_ns)

        gaps: List[float] = []
        end = 0
        for o in occs:
            if o.start_ns > end:
                gaps.append(o.start_ns - end)
            end = max(end, o.end_ns)
        if end < self.cycle_ns:
            gaps.append(self.cycle_ns - end)
        return gaps

    def _get_port_metrics(self) -> np.ndarray:
        """(P, 3) [utilization, useless_gap_ratio, avg_useful_gap_norm]."""
        met = np.zeros((self.n_ports, 3), dtype=np.float32)
        for pidx in range(self.n_ports):
            if pidx not in self.port_schedules or not self.port_schedules[pidx]:
                continue                                       # nothing on this port

            gaps = self._port_gaps(pidx)
            total_gap = sum(gaps)
            met[pidx, 0] = (self.cycle_ns - total_gap) / self.cycle_ns   # utilisation

            if gaps:
                useless = [g for g in gaps if g < self.gap_threshold_ns]
                met[pidx, 1] = len(useless) / len(gaps)

                useful = [g for g in gaps if g >= self.gap_threshold_ns]
                if useful:
                    met[pidx, 2] = float(np.mean(useful)) / self.cycle_ns
        return met

    def _count_gaps(self) -> int:
        """Total gap count across ports that carry at least one stream."""
        total = 0
        for pidx in range(self.n_ports):
            total += len(self._port_gaps(pidx))
        return total

    def _count_useless_gaps(self) -> int:
        """Count gaps below threshold across all ports."""
        total = 0
        for pidx in range(self.n_ports):
            gaps = self._port_gaps(pidx)
            total += sum(1 for g in gaps if g < self.gap_threshold_ns)
        return total

    def _useless_gap_time_ratio(self) -> float:
        """
        Aggregate ratio of useless-gap time over one cycle across all ports.
        """
        if self.n_ports <= 0 or self.cycle_ns <= 0:
            return 0.0

        ratio_sum = 0.0
        for pidx in range(self.n_ports):
            gaps = self._port_gaps(pidx)
            useless_time = sum(g for g in gaps if g < self.gap_threshold_ns)
            ratio_sum += useless_time / self.cycle_ns
        return ratio_sum / self.n_ports

    def _fragmentation_value(self) -> float:
        """
        Scalar fragmentation objective:
        - 'gap_count': total number of gaps.
        - 'useless_gap_count': count of gaps < threshold.
        - 'hybrid': useless_gap_count + useless_gap_time_ratio.
        """
        if self.fragmentation_metric == "gap_count":
            return float(self._count_gaps())
        if self.fragmentation_metric == "useless_gap_count":
            return float(self._count_useless_gaps())
        if self.fragmentation_metric == "hybrid":
            return float(self._count_useless_gaps()) + self._useless_gap_time_ratio()
        raise ValueError(
            "fragmentation_metric must be one of "
            "{'gap_count', 'useless_gap_count', 'hybrid'}"
        )

    def _simple_fragmentation_value(self) -> float:
        """Simple objective used as default: total gap count only."""
        return float(self._count_gaps())

    def _frag_score(self) -> float:
        """
        FragScore(s) = mean_j[(1-useless_gap_ratio_j) + avg_useful_gap_j].
        """
        met = self._get_port_metrics()
        if met.shape[0] == 0:
            return 0.0
        return float(np.mean((1.0 - met[:, 1]) + met[:, 2]))

    def _build_admission_templates(self, max_templates: int) -> List[Dict]:
        """
        Build a small template set C from existing streams (lightweight proxy).
        """
        candidates = []
        for sinfo in self.stream_info:
            path = list(sinfo.get("path", []))
            if not path:
                continue
            interval_ns = max(1, int(sinfo.get("interval_ns", self.cycle_ns)))
            weight = float(self.cycle_ns) / float(interval_ns)
            candidates.append(
                {
                    "path": path,
                    "port_dur_ns": dict(sinfo.get("port_dur_ns", {})),
                    "weight": weight,
                }
            )

        # Keep the highest-priority templates (shorter intervals -> higher weight).
        candidates.sort(key=lambda x: x["weight"], reverse=True)
        return candidates[:max_templates]

    def _template_fits(self, template: Dict) -> bool:
        """
        Lightweight fit(c, s): template fits if each path port has a gap large
        enough for one transmission window.
        """
        for pid in template.get("path", []):
            pidx = pid - 1
            needed = float(template.get("port_dur_ns", {}).get(pid, 0.0))
            gaps = self._port_gaps(pidx)
            if not any(g >= needed for g in gaps):
                return False
        return True

    def _template_fit_ratio(self, template: Dict) -> float:
        """
        Continuous fit proxy in [0,1]:
        for each path port, compute max_gap/needed (clipped), then take min over
        ports so a single bottleneck port limits admission quality.
        """
        ratios = []
        for pid in template.get("path", []):
            pidx = pid - 1
            needed = max(1e-9, float(template.get("port_dur_ns", {}).get(pid, 0.0)))
            gaps = self._port_gaps(pidx)
            max_gap = max(gaps) if gaps else 0.0
            ratios.append(min(1.0, max_gap / needed))

        if not ratios:
            return 0.0
        return float(min(ratios))

    def _admission_fit_score(self) -> float:
        """
        AdmissionFit(s) = sum_{c in C} w_c * fit(c, s)
        """
        if not getattr(self, "admission_templates", None):
            return 0.0
        total = 0.0
        for tmpl in self.admission_templates:
            total += float(tmpl.get("weight", 1.0)) * self._template_fit_ratio(tmpl)
        return total

    def _has_any_port_overlap(self) -> bool:
        """Return True if any port has overlapping occurrences in [0, cycle)."""
        for pidx, sched in self.port_schedules.items():
            if not sched:
                continue
            occs = []
            for slot in sched:
                occs.extend(slot.all_occurrences(self.cycle_ns))
            if not occs:
                continue
            occs.sort(key=lambda s: s.start_ns)
            prev_end = occs[0].end_ns
            for occ in occs[1:]:
                if occ.start_ns < prev_end:
                    return True
                prev_end = max(prev_end, occ.end_ns)
        return False

    def _violates_latency_constraints(self) -> bool:
        """
        Lightweight latency validation on first occurrences:
        (last_port_start + last_port_duration) - first_port_start <= max_latency.
        """
        for s_idx, sinfo in enumerate(self.stream_info):
            path = sinfo.get("path", [])
            if not path:
                continue
            first_pid = path[0]
            last_pid = path[-1]
            offsets = self.stream_offsets.get(s_idx, {})
            if first_pid not in offsets or last_pid not in offsets:
                continue

            first_start = float(offsets[first_pid])
            last_start = float(offsets[last_pid])
            last_dur = float(sinfo.get("port_dur_ns", {}).get(last_pid, 0.0))
            e2e_latency = (last_start + last_dur) - first_start
            if e2e_latency > float(sinfo.get("max_latency_ns", math.inf)):
                return True
        return False

    def _invalid_schedule_flag(self) -> int:
        """
        1 if schedule violates core constraints, else 0.
        """
        if self._has_any_port_overlap():
            return 1
        if self._violates_latency_constraints():
            return 1
        return 0

    @staticmethod
    def _decode_placement_matrix(placement: np.ndarray) -> np.ndarray:
        """
        Convert encoded placement matrix to normalized offsets in [0,1].
        Encoded value 0 means no placement on that port.
        """
        placement = np.asarray(placement, dtype=np.float32)
        decoded = np.where(placement > 0.0, placement - 1.0, 0.0)
        return decoded

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "stream_placement": self._get_stream_placement(),
            "stream_features": self._get_stream_features(),
            "port_metrics": self._get_port_metrics(),
            "adjacency_matrix": self._adjacency.copy(),
        }

    # ================================================================== #
    # Gym interface
    # ================================================================== #
    def step(self, action):
        """
        Apply *action* (N×P) to rearrange the schedule.

        Multi-move mode (default):
            For each stream, all non-zero action entries on path ports are
            read and projected to an implied first-hop offset.

        Single-move mode:
            At most one stream is moved per step. The selected move is the
            highest action value among valid (stream, path-port) entries.
        """
        action = np.asarray(action, dtype=np.float32).reshape(
            self.n_streams, self.n_ports
        )

        old_placement = self._get_stream_placement()
        old_admission_fit = self._admission_fit_score()
        old_frag_score = self._frag_score()
        old_invalid = int(self._invalid_schedule_flag())

        if self.reward_mode == "simple":
            old_frag = self._simple_fragmentation_value()
        else:
            old_frag = self._fragmentation_value()
        n_moved = 0

        moved_stream = None
        moved_port = None
        moved_actions = []

        if self.single_move_per_step:
            best = None  # (act_val, s_idx, pidx)
            for s_idx, sinfo in enumerate(self.stream_info):
                for pid in sinfo["path"]:
                    pidx = pid - 1
                    act_val = float(action[s_idx, pidx])
                    if act_val <= 0:
                        continue
                    if best is None or act_val > best[0]:
                        best = (act_val, s_idx, pidx)

            if best is not None:
                _, s_idx, pidx = best
                path = self.stream_info[s_idx]["path"]
                first_pid = path[0]
                decoded_offset = self._decode_first_offset_from_port_action(
                    s_idx, pidx, action[s_idx, pidx]
                )
                if decoded_offset is not None:
                    old_offset = self.stream_offsets.get(s_idx, {}).get(first_pid, 0)
                    if abs(decoded_offset - old_offset) > 1:
                        n_moved = 1
                        moved_stream = s_idx
                        moved_port = pidx
                        moved_actions.append(
                            {
                                "stream_id": int(s_idx),
                                "port_id": int(pidx + 1),
                                "old_start_ns": int(round(old_offset)),
                                "new_start_ns": int(round(decoded_offset)),
                                "delta_start_ns": int(round(decoded_offset - old_offset)),
                            }
                        )
                        self._place_stream(s_idx, decoded_offset)
        else:
            for s_idx, sinfo in enumerate(self.stream_info):
                path = sinfo["path"]
                if not path:
                    continue

                decoded_offset = self._decode_first_offset_from_action(
                    s_idx, action[s_idx]
                )
                if decoded_offset is None:
                    continue

                first_pid = path[0]
                old_offset = self.stream_offsets.get(s_idx, {}).get(first_pid, 0)
                if abs(decoded_offset - old_offset) > 1:            # 1 ns tolerance
                    n_moved += 1
                    moved_actions.append(
                        {
                            "stream_id": int(s_idx),
                            "port_id": int(first_pid),
                            "old_start_ns": int(round(old_offset)),
                            "new_start_ns": int(round(decoded_offset)),
                            "delta_start_ns": int(round(decoded_offset - old_offset)),
                        }
                    )
                    self._place_stream(s_idx, decoded_offset)

            if moved_actions:
                moved_stream = moved_actions[0]["stream_id"]
                moved_port = moved_actions[0]["port_id"] - 1

        if self.reward_mode == "simple":
            new_frag = self._simple_fragmentation_value()
        else:
            new_frag = self._fragmentation_value()

        new_placement = self._get_stream_placement()
        new_admission_fit = self._admission_fit_score()
        new_frag_score = self._frag_score()

        x_old = self._decode_placement_matrix(old_placement)
        x_new = self._decode_placement_matrix(new_placement)
        shift_cost = float(np.sum(np.abs(x_new - x_old)))
        moved_streams_matrix = int(np.sum(np.any(np.abs(x_new - x_old) > 1e-9, axis=1)))
        invalid = int(self._invalid_schedule_flag())
        invalid_new = 1 if (invalid == 1 and old_invalid == 0) else 0

        delta = (old_frag - new_frag) / max(1.0, old_frag)
        if self.reward_mode == "paper_simple":
            delta_admission = float(new_admission_fit - old_admission_fit)
            delta_frag_score = float(new_frag_score - old_frag_score)
            w = self.reward_weights
            reward = (
                w["w1_admission"] * delta_admission
                + w["w2_frag"] * delta_frag_score
                - w["w3_shift"] * shift_cost
                - w["w4_moved"] * moved_streams_matrix
                - w["w5_invalid"] * invalid_new
            )
        else:
            delta_admission = float(new_admission_fit - old_admission_fit)
            delta_frag_score = float(new_frag_score - old_frag_score)
            reward = delta - self.move_penalty * n_moved

        info = {
            "old_fragmentation": old_frag,
            "new_fragmentation": new_frag,
            "delta_frag": delta,
            "n_moved": n_moved,
            "all_gaps": self._count_gaps(),
            "useless_gaps": self._count_useless_gaps(),
            "useless_gap_time_ratio": self._useless_gap_time_ratio(),
            "fragmentation_metric": self.fragmentation_metric,
            "reward_mode": self.reward_mode,
            "single_move_per_step": self.single_move_per_step,
            "moved_stream": moved_stream,
            "moved_port": moved_port,
            "moved_actions": moved_actions,
            "old_admission_fit": float(old_admission_fit),
            "new_admission_fit": float(new_admission_fit),
            "delta_admission_fit": float(delta_admission),
            "old_frag_score": float(old_frag_score),
            "new_frag_score": float(new_frag_score),
            "delta_frag_score": float(delta_frag_score),
            "shift_cost": float(shift_cost),
            "moved_streams_matrix": int(moved_streams_matrix),
            "invalid": int(invalid),
            "invalid_old": int(old_invalid),
            "invalid_new": int(invalid_new),
            "reward_weights": dict(self.reward_weights),
        }
        return self._get_obs(), reward, False, False, info

    def reset(self, seed=None, options=None):
        """Reset the schedule to a simple initial placement."""
        super().reset(seed=seed)
        use_uc_schedule = self.init_from_usecase_schedule
        if options and "init_from_usecase_schedule" in options:
            use_uc_schedule = bool(options["init_from_usecase_schedule"])

        loaded = False
        if use_uc_schedule:
            loaded = self._create_initial_schedule_from_usecase()
        if not loaded:
            self._create_initial_schedule()
        return self._get_obs(), {}

    def render(self):
        """Combine all per-port schedules into one and plot."""
        combined = Schedule(cycle_ns=self.cycle_ns)
        for pidx in sorted(self.port_schedules):
            for slot in self.port_schedules[pidx]:
                combined.append(slot)
        return combined.plot()

    # ================================================================== #
    # Convenience
    # ================================================================== #
    def get_combined_schedule(self) -> Schedule:
        """Return a single Schedule object containing all per-port slots."""
        combined = Schedule(cycle_ns=self.cycle_ns)
        for pidx in sorted(self.port_schedules):
            for slot in self.port_schedules[pidx]:
                combined.append(slot)
        return combined
