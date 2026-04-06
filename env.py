"""
V1.1 TSN GCL defragmentation environment.

Action semantics:
- Environment exposes one focus stream in observation.
- Agent chooses only a candidate index for that focus stream.
- Start times are integer nanoseconds.
- Compact/start-time encoding uses start_time + 1, and 0 means inactive.

Reward semantics:
- Fragmentation-improvement reward (gap count + largest hole),
  with penalties for invalid actions and unnecessary moves.

Reward modes:
- weighted_fragmentation: nuanced weighted score for mature experiments.
- gap_improvement_binary: simple signal to test if policy can reduce gap count.
- hole_improvement_binary: simple signal to test if policy can increase largest-hole mean.
"""

import copy
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from evaluator import ScheduleEncodingType, TSNConstraintEvaluator
from heuristic import fragmentation_metrics
from schedule import Schedule, Slot
from tsn_scheduling_problem import MBPS_2_BPNS, TSNScheduling
from usecase_generator import Usecase

BYTE_2_BIT = 8
TOL = 1.1


class TSNDefragEnv(gym.Env):
    """Gymnasium environment for single-stream candidate-based TSN defragmentation."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        problem: TSNScheduling,
        usecase: Usecase,
        gap_threshold_ratio: float = 0.01,
        max_edges: int = 200,
        single_move_per_step: bool = True,
        init_from_usecase_schedule: bool = True,
        reward_weights: Optional[Dict[str, float]] = None,
        reward_frame_sizes: Optional[List[int]] = None,
        reward_mix_percentages: Optional[List[float]] = None,
        reward_incoming_stream_count: int = 100,
        max_candidates: int = 10,
        max_steps: int = 60,
        reward_mode: str = "weighted_fragmentation",
        force_current_start_candidate: bool = False,
        debug_candidate_validation: bool = True,
    ):
        super().__init__()

        self.problem = problem
        self.usecase = usecase
        self.max_edges = int(max_edges)
        self.single_move_per_step = bool(single_move_per_step)
        self.init_from_usecase_schedule = bool(init_from_usecase_schedule)
        self.max_candidates = max(1, int(max_candidates))
        self.max_steps = max(1, int(max_steps))
        self.step_count = 0
        self.current_stream_id = 0
        self.reward_mode = str(reward_mode)
        self.force_current_start_candidate = bool(force_current_start_candidate)
        self.debug_candidate_validation = bool(debug_candidate_validation)
        self.initial_violation_sum = 0
        self.initial_violation_dict: Dict[str, int] = {}

        supported_reward_modes = {
            "weighted_fragmentation",
            "gap_improvement_binary",
            "hole_improvement_binary",
        }
        if self.reward_mode not in supported_reward_modes:
            raise ValueError(
                f"Unsupported reward_mode='{self.reward_mode}'. "
                f"Supported modes: {sorted(supported_reward_modes)}"
            )

        frame_sizes = reward_frame_sizes if reward_frame_sizes is not None else [64, 128, 256]
        mix_percentages = reward_mix_percentages if reward_mix_percentages is not None else [50.0, 25.0, 25.0]
        self.reward_frame_sizes = tuple(int(x) for x in frame_sizes)
        self.reward_mix_percentages = tuple(float(x) for x in mix_percentages)
        self.reward_incoming_stream_count = int(reward_incoming_stream_count)

        default_weights = {
            "w_gap": 1.0,
            "w_hole": 0.5,
            "w_move": 0.01,
            "w_invalid": 1.0,
        }
        self.reward_weights = dict(default_weights)
        if reward_weights:
            self.reward_weights.update(reward_weights)

        with self.problem.use_instance(self.usecase):
            self.n_ports: int = self.problem.n_ports
            self.n_streams: int = self.problem.n_streams
            self.cycle_ns: int = self.problem.macro_cycle
            self.gap_threshold_ns: float = gap_threshold_ratio * self.cycle_ns

            self._port_name: Dict[int, str] = {}
            self._port_src_node: Dict[int, str] = {}
            self._port_dst_node: Dict[int, str] = {}
            self._port_speed: Dict[int, float] = {}

            for pid, pinfo in self.problem.ports.items():
                self._port_name[pid] = self.problem.port_id_to_name[pid]
                self._port_src_node[pid] = pinfo.get("sourceNode", "")
                self._port_dst_node[pid] = pinfo.get("destNode", "")
                self._port_speed[pid] = pinfo.get("speed", 1000)

            self.stream_info: List[Dict] = []
            enriched_streams = self.problem.streams

            for s_idx in range(self.n_streams):
                raw = self.usecase.streams[s_idx]
                enr = enriched_streams[s_idx]

                path: List[int] = []
                if "pathLists" in enr and enr["pathLists"]:
                    path = list(enr["pathLists"][0])

                interval_ns = int(raw["interval"])
                frame_bits = TSNScheduling._frame_size(raw)
                max_lat_ns = int(raw.get("maxLatency", 1_000_000))

                port_dur: Dict[int, float] = {}
                for pid in path:
                    spd = self._port_speed.get(pid, 1000)
                    port_dur[pid] = frame_bits / (spd * MBPS_2_BPNS)

                hop_delays: List[float] = []
                total_hop_cost = 0.0
                for k in range(len(path) - 1):
                    src_nm = self._port_name.get(path[k], "")
                    dst_nm = self._port_name.get(path[k + 1], "")
                    try:
                        delay = self.problem.hop_cost_eg2eg(src_nm, dst_nm, frame_bits)
                    except Exception:
                        delay = port_dur.get(path[k], 0.0) + 100.0
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

            self.max_latency_ns: float = max((s["max_latency_ns"] for s in self.stream_info), default=1)
            self._adjacency = self._build_adjacency()

            self.sched_eval = TSNConstraintEvaluator(
                self.problem,
                check_routing=False,
                check_task_placement=False,
                scheduling_type=ScheduleEncodingType.START_TIME,
            )

        # V1.1 design: the environment chooses the focus stream and the agent
        # chooses only one candidate index for that stream.
        self.action_space = spaces.Discrete(self.max_candidates)
        self.observation_space = spaces.Dict(
            {
                "stream_placement": spaces.Box(0.0, 2.0, (self.n_streams, self.n_ports), np.float32),
                "stream_features": spaces.Box(0.0, np.inf, (self.n_streams, 5), np.float32),
                "port_metrics": spaces.Box(0.0, 1.0, (self.n_ports, 3), np.float32),
                "adjacency_matrix": spaces.Box(-1, self.n_ports, (2, self.max_edges), np.int64),
                "candidate_offsets": spaces.Box(0, self.cycle_ns, (self.max_candidates,), np.int64),
                "candidate_mask": spaces.Box(0, 1, (self.max_candidates,), np.int8),
                "current_stream_id": spaces.Box(0, max(0, self.n_streams - 1), (1,), np.int64),
            }
        )

        self.port_schedules: Dict[int, Schedule] = {}
        self.stream_offsets: Dict[int, Dict[int, float]] = {}

    def _build_adjacency(self) -> np.ndarray:
        """Build line-graph adjacency over ports (0-based indices)."""
        pids = sorted(self._port_src_node.keys())
        edges = []
        for p_a in pids:
            dst_a = self._port_dst_node.get(p_a)
            if not dst_a:
                continue
            for p_b in pids:
                if p_a != p_b and self._port_src_node.get(p_b) == dst_a:
                    edges.append((p_a - 1, p_b - 1))

        adj = np.full((2, self.max_edges), -1, dtype=np.int64)
        for i, (s, d) in enumerate(edges[: self.max_edges]):
            adj[0, i] = s
            adj[1, i] = d
        return adj

    def _place_stream(self, s_idx: int, first_offset_ns: float) -> None:
        """Place one stream from first-hop offset and propagate downstream by hop delays."""
        sinfo = self.stream_info[s_idx]
        path = sinfo["path"]
        if not path:
            return

        for pid in path:
            pidx = pid - 1
            if pidx in self.port_schedules:
                self.port_schedules[pidx] = Schedule(
                    cycle_ns=self.cycle_ns,
                    iterable=[s for s in self.port_schedules[pidx] if s.stream_id != s_idx],
                )

        offsets: Dict[int, float] = {}
        cum_delay = 0.0
        for k, pid in enumerate(path):
            dur = float(sinfo["port_dur_ns"].get(pid, 0.0))
            max_start = max(0.0, float(self.cycle_ns) - dur)
            offset = max(0.0, min(max_start, float(first_offset_ns) + cum_delay))
            offsets[pid] = offset
            if k < len(sinfo["hop_delays"]):
                cum_delay += float(sinfo["hop_delays"][k])

        for pid, offset in offsets.items():
            pidx = pid - 1
            if pidx not in self.port_schedules:
                self.port_schedules[pidx] = Schedule(cycle_ns=self.cycle_ns)

            dur = float(sinfo["port_dur_ns"].get(pid, 0.0))
            slot = Slot(
                start_ns=int(round(offset)),
                end_ns=int(round(offset + dur)),
                period_ns=int(sinfo["interval_ns"]),
                queue_id=pidx,
                stream_id=s_idx,
                port_id=pid,
            )
            self.port_schedules[pidx].append(slot)

        self.stream_offsets[s_idx] = offsets

    def _create_initial_schedule(self) -> None:
        """Simple default initialization: all streams start at 0 on first port."""
        self.port_schedules = {}
        self.stream_offsets = {}
        for s_idx in range(self.n_streams):
            self._place_stream(s_idx, first_offset_ns=0.0)

    def _create_initial_schedule_from_usecase(self) -> bool:
        """Load schedule from usecase when available."""
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

            pidx = int(slot.port_id) - 1
            if pidx not in self.port_schedules:
                self.port_schedules[pidx] = Schedule(cycle_ns=self.cycle_ns)
            self.port_schedules[pidx].append(slot)

            if slot.stream_id is None:
                continue

            s_idx = int(slot.stream_id)
            if s_idx not in self.stream_offsets:
                self.stream_offsets[s_idx] = {}
            prev = self.stream_offsets[s_idx].get(int(slot.port_id))
            if prev is None or slot.start_ns < prev:
                self.stream_offsets[s_idx][int(slot.port_id)] = float(slot.start_ns)

        return True

    def _stream_first_port(self, s_idx: int) -> Optional[int]:
        path = self.stream_info[s_idx]["path"]
        if not path:
            return None
        return int(path[0])

    def _stream_first_duration_ns(self, s_idx: int) -> int:
        first_pid = self._stream_first_port(s_idx)
        if first_pid is None:
            return 0
        dur = self.stream_info[s_idx]["port_dur_ns"].get(first_pid, 0.0)
        return int(np.ceil(float(dur)))

    def _stream_current_first_start_ns(self, s_idx: int) -> int:
        first_pid = self._stream_first_port(s_idx)
        if first_pid is None:
            return 0
        return int(round(float(self.stream_offsets.get(s_idx, {}).get(first_pid, 0.0))))

    @staticmethod
    def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not intervals:
            return []
        ordered = sorted(intervals, key=lambda x: x[0])
        merged: List[Tuple[int, int]] = [ordered[0]]
        for st, en in ordered[1:]:
            last_st, last_en = merged[-1]
            if st <= last_en:
                merged[-1] = (last_st, max(last_en, en))
            else:
                merged.append((st, en))
        return merged

    @staticmethod
    def _overlaps_any(start_ns: int, end_ns: int, occupied: List[Tuple[int, int]]) -> bool:
        for occ_start, occ_end in occupied:
            if start_ns < occ_end and end_ns > occ_start:
                return True
        return False

    def _first_port_occupied_intervals_excluding_stream(self, s_idx: int) -> List[Tuple[int, int]]:
        """Occupied first-port intervals in one cycle, excluding stream s_idx."""
        first_pid = self._stream_first_port(s_idx)
        if first_pid is None:
            return []

        pidx = first_pid - 1
        if pidx not in self.port_schedules or not self.port_schedules[pidx]:
            return []

        intervals: List[Tuple[int, int]] = []
        for slot in self.port_schedules[pidx]:
            if int(slot.stream_id) == int(s_idx):
                continue
            for occ in slot.all_occurrences(self.cycle_ns):
                st = max(0, int(occ.start_ns))
                en = min(int(self.cycle_ns), int(occ.end_ns))
                if en > st:
                    intervals.append((st, en))

        return self._merge_intervals(intervals)

    def _candidate_passes_first_port_filter(self, s_idx: int, candidate_ns: int) -> bool:
        """Local first-port feasibility check used by raw candidate generation."""
        if s_idx < 0 or s_idx >= self.n_streams:
            return False

        first_pid = self._stream_first_port(s_idx)
        if first_pid is None:
            return False

        dur = self._stream_first_duration_ns(s_idx)
        max_start = max(0, int(self.cycle_ns) - int(dur))
        st = int(candidate_ns)
        if st < 0 or st > max_start:
            return False

        en = st + int(dur)
        if en > int(self.cycle_ns):
            return False

        occupied = self._first_port_occupied_intervals_excluding_stream(s_idx)
        return not self._overlaps_any(st, en, occupied)

    def _raw_stream_move_candidates(self, s_idx: int) -> List[int]:
        """Raw first-hop candidates from first-port occupancy and gap boundaries."""
        if s_idx < 0 or s_idx >= self.n_streams:
            return [0]

        first_pid = self._stream_first_port(s_idx)
        if first_pid is None:
            return [0]

        dur = self._stream_first_duration_ns(s_idx)
        max_start = max(0, int(self.cycle_ns) - int(dur))
        current_start = int(np.clip(self._stream_current_first_start_ns(s_idx), 0, max_start))

        occupied = self._first_port_occupied_intervals_excluding_stream(s_idx)
        raw = {0, current_start}
        for _, end_ns in occupied:
            raw.add(int(end_ns))

        # Also derive candidates from explicit free-gap boundaries.
        # For gap [g_start, g_end], both g_start and (g_end - duration) can be useful.
        cursor = 0
        for occ_start, occ_end in occupied:
            if occ_start > cursor:
                g_start, g_end = int(cursor), int(occ_start)
                raw.add(g_start)
                raw.add(g_end - int(dur))
            cursor = max(cursor, int(occ_end))
        if cursor < int(self.cycle_ns):
            g_start, g_end = int(cursor), int(self.cycle_ns)
            raw.add(g_start)
            raw.add(g_end - int(dur))

        candidates: List[int] = []
        for cand in sorted(raw):
            st = int(np.clip(cand, 0, max_start))
            if self._candidate_passes_first_port_filter(s_idx, st):
                candidates.append(st)

        # Stage-1 raw set includes current start boundary candidate by design,
        # but inclusion in final set is decided only after full evaluator checks.
        if self._candidate_passes_first_port_filter(s_idx, int(current_start)):
            candidates.append(int(current_start))
        return sorted(set(int(c) for c in candidates))

    def _simulate_stream_move_compact(self, s_idx: int, first_offset_ns: int) -> np.ndarray:
        """Simulate one stream move and return compact schedule, restoring state afterwards."""
        old_port_schedules = copy.deepcopy(self.port_schedules)
        old_stream_offsets = copy.deepcopy(self.stream_offsets)
        try:
            self._place_stream(int(s_idx), float(int(first_offset_ns)))
            return self._current_compact().copy()
        finally:
            self.port_schedules = old_port_schedules
            self.stream_offsets = old_stream_offsets

    def _candidate_is_fully_feasible(self, s_idx: int, first_offset_ns: int) -> bool:
        """Validate candidate by evaluating full propagated schedule feasibility."""
        # First-port-only checks are not enough: downstream hop propagation can create
        # conflicts. Evaluate the full attempted compact schedule instead.
        compact = self._simulate_stream_move_compact(s_idx, first_offset_ns)
        return bool(self._violation_sum(compact) == 0)

    def _stream_move_candidate_debug(self, s_idx: int) -> List[Dict[str, object]]:
        """Return candidate validation records for one stream."""
        _, debug_records = self._compute_stream_candidates(s_idx, include_debug=True)
        return debug_records

    def _compute_stream_candidates(
        self, s_idx: int, include_debug: bool = False
    ) -> Tuple[List[int], List[Dict[str, object]]]:
        """Candidate pipeline with explicit full-schedule validation stage."""
        current_start = int(self._stream_current_first_start_ns(s_idx))
        raw_candidates = self._raw_stream_move_candidates(s_idx)

        considered = sorted(set(raw_candidates + [int(current_start)]))
        final_candidates: List[int] = []
        debug_records: List[Dict[str, object]] = []

        for cand in considered:
            cand_i = int(cand)
            is_current = bool(cand_i == int(current_start))
            is_raw = bool(cand_i in raw_candidates)
            passes_first = bool(self._candidate_passes_first_port_filter(s_idx, cand_i))
            passes_full = bool(self._candidate_is_fully_feasible(s_idx, cand_i))

            included = False
            fallback_only = False
            if passes_full:
                included = True
            elif is_current and self.force_current_start_candidate:
                # Forcing current start can violate the "all candidates are safe" assumption,
                # so this is explicitly marked as fallback-unsafe in debug metadata.
                included = True
                fallback_only = True

            if included:
                final_candidates.append(cand_i)

            if include_debug or self.debug_candidate_validation:
                debug_records.append(
                    {
                        "candidate_start_ns": int(cand_i),
                        "is_current_start": bool(is_current),
                        "is_raw_candidate": bool(is_raw),
                        "passes_first_port_filter": bool(passes_first),
                        "passes_full_evaluator": bool(passes_full),
                        "included_in_final_candidates": bool(included),
                        "included_only_by_fallback": bool(fallback_only),
                    }
                )

        final_unique = sorted(set(final_candidates))
        if len(final_unique) > self.max_candidates:
            final_unique = final_unique[: self.max_candidates]

        return final_unique, debug_records

    def _stream_move_candidates(self, s_idx: int) -> List[int]:
        """Final candidates exposed to agent after full-schedule validation."""
        final_candidates, _ = self._compute_stream_candidates(s_idx, include_debug=False)
        return final_candidates

    def _active_stream_ids(self) -> List[int]:
        """Streams that have a valid first port and an active placement on that port."""
        active: List[int] = []
        for s_idx in range(self.n_streams):
            first_pid = self._stream_first_port(s_idx)
            if first_pid is None:
                continue
            if first_pid in self.stream_offsets.get(s_idx, {}):
                active.append(int(s_idx))
        return active

    def _advance_current_stream(self) -> None:
        """Move focus stream in deterministic round-robin order over active streams."""
        # Round-robin is intentionally simple and deterministic for easier debugging.
        active = self._active_stream_ids()
        if not active:
            self.current_stream_id = 0
            return

        if self.current_stream_id not in active:
            self.current_stream_id = int(active[0])
            return

        idx = active.index(int(self.current_stream_id))
        self.current_stream_id = int(active[(idx + 1) % len(active)])

    def _candidate_obs_for_stream(self, s_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fixed-size candidate offsets + validity mask for observation."""
        offsets = np.zeros((self.max_candidates,), dtype=np.int64)
        mask = np.zeros((self.max_candidates,), dtype=np.int8)
        cands = self._stream_move_candidates(int(s_idx))
        for i, cand in enumerate(cands[: self.max_candidates]):
            offsets[i] = int(cand)
            mask[i] = 1
        return offsets, mask

    def _first_active_stream_id(self) -> int:
        active = self._active_stream_ids()
        if not active:
            return 0
        return int(active[0])

    def _get_stream_placement(self) -> np.ndarray:
        """(N, P): placement[i,j] = start/cycle + 1 or 0 if stream inactive on that port."""
        placement = np.zeros((self.n_streams, self.n_ports), dtype=np.float32)
        for s_idx, offsets in self.stream_offsets.items():
            for pid, off in offsets.items():
                pidx = int(pid) - 1
                if 0 <= pidx < self.n_ports:
                    placement[s_idx, pidx] = float(off) / float(self.cycle_ns) + 1.0
        return placement

    def _get_stream_features(self) -> np.ndarray:
        """(N, 5) normalized stream descriptors from first hop and metadata."""
        feat = np.zeros((self.n_streams, 5), dtype=np.float32)
        for s_idx, sinfo in enumerate(self.stream_info):
            path = sinfo["path"]
            if not path:
                continue

            first_pid = int(path[0])
            off = float(self.stream_offsets.get(s_idx, {}).get(first_pid, 0.0))
            dur = float(sinfo["port_dur_ns"].get(first_pid, 0.0))

            feat[s_idx, 0] = off / float(self.cycle_ns)
            feat[s_idx, 1] = dur / float(self.cycle_ns)
            feat[s_idx, 2] = float(sinfo["interval_ns"]) / float(self.cycle_ns)
            feat[s_idx, 3] = float(sinfo["max_latency_ns"]) / float(self.max_latency_ns)
            feat[s_idx, 4] = float(sinfo["total_hop_cost_ns"]) / float(self.cycle_ns)
        return feat

    def _port_gaps(self, pidx: int) -> List[float]:
        """Gap durations on one 0-based port for the current cycle."""
        if pidx not in self.port_schedules or not self.port_schedules[pidx]:
            return []

        occs = []
        for slot in self.port_schedules[pidx]:
            occs.extend(slot.all_occurrences(self.cycle_ns))

        if not occs:
            return []

        occs.sort(key=lambda s: s.start_ns)
        gaps: List[float] = []
        cur_end = 0
        for occ in occs:
            if occ.start_ns > cur_end:
                gaps.append(float(occ.start_ns - cur_end))
            cur_end = max(cur_end, int(occ.end_ns))
        if cur_end < self.cycle_ns:
            gaps.append(float(self.cycle_ns - cur_end))
        return gaps

    def _get_port_metrics(self) -> np.ndarray:
        """(P, 3): [utilization, useless_gap_ratio, avg_useful_gap_norm]."""
        met = np.zeros((self.n_ports, 3), dtype=np.float32)
        for pidx in range(self.n_ports):
            if pidx not in self.port_schedules or not self.port_schedules[pidx]:
                continue

            gaps = self._port_gaps(pidx)
            total_gap = float(sum(gaps))
            met[pidx, 0] = (float(self.cycle_ns) - total_gap) / float(self.cycle_ns)

            if gaps:
                useless = [g for g in gaps if g < self.gap_threshold_ns]
                met[pidx, 1] = float(len(useless)) / float(len(gaps))
                useful = [g for g in gaps if g >= self.gap_threshold_ns]
                if useful:
                    met[pidx, 2] = float(np.mean(useful)) / float(self.cycle_ns)
        return met

    def _get_obs(self) -> Dict[str, np.ndarray]:
        stream_id = int(np.clip(self.current_stream_id, 0, max(0, self.n_streams - 1)))
        cand_offsets, cand_mask = self._candidate_obs_for_stream(stream_id)
        return {
            "stream_placement": self._get_stream_placement(),
            "stream_features": self._get_stream_features(),
            "port_metrics": self._get_port_metrics(),
            "adjacency_matrix": self._adjacency.copy(),
            "candidate_offsets": cand_offsets,
            "candidate_mask": cand_mask,
            "current_stream_id": np.asarray([stream_id], dtype=np.int64),
        }

    def _current_compact(self) -> np.ndarray:
        """Current schedule in compact start-time encoding (start+1, 0 means inactive)."""
        return self.get_combined_schedule().to_compact_encoding(self.n_ports, self.n_streams)

    def _violation_row_labels(self, compact: np.ndarray) -> List[str]:
        labels: List[str] = []
        for check_fn in self.sched_eval.active_checks:
            arr = check_fn(compact)
            row_count = int(arr.shape[0])
            name = check_fn.__name__

            if name == "_per_stream_window_sequence_violations" and row_count == 2:
                labels.extend(["stream_window_sequence", "interleaving"])
            elif name == "_per_stream_macrocycle_overlap_violations" and row_count == 1:
                labels.append("macrocycle_overlap")
            elif name == "_per_stream_talker_offset_violations" and row_count == 1:
                labels.append("talker_offset")
            elif name == "_per_stream_latency_violations_start_time" and row_count == 1:
                labels.append("maxlatency_violations")
            else:
                clean = name.replace("_per_stream_", "")
                for idx in range(row_count):
                    labels.append(f"{clean}[{idx}]")
        return labels

    def _violation_row_sums(self, compact: np.ndarray) -> np.ndarray:
        per_stream = self.sched_eval.per_stream_all_violations(compact)
        if per_stream.size == 0:
            return np.zeros((0,), dtype=np.int64)
        return np.asarray(per_stream.sum(axis=1), dtype=np.int64)

    def _violation_dict(self, compact: np.ndarray) -> Dict[str, int]:
        labels = self._violation_row_labels(compact)
        row_sums = self._violation_row_sums(compact)
        out: Dict[str, int] = {}
        for idx, label in enumerate(labels):
            out[label] = int(row_sums[idx]) if idx < row_sums.size else 0
        return out

    def _violation_dict_normalized(self, compact: np.ndarray) -> Dict[str, float]:
        """Category-wise normalized violations for future safe-RL cost design.

        Raw violation categories can differ significantly in scale by construction,
        so this helper normalizes each category independently with a capped map.
        V1 keeps reward simple and only uses a binary invalid penalty.
        """
        raw = self._violation_dict(compact)
        return {k: float(min(float(v) / 20.0, 1.0)) for k, v in raw.items()}

    def _violation_sum(self, compact: np.ndarray) -> int:
        return int(self.sched_eval.per_stream_all_violations(compact).sum())

    def _fragmentation_snapshot(self, compact: np.ndarray) -> Dict[str, float]:
        metrics = fragmentation_metrics(
            compact=compact,
            wind_lengths=np.asarray(self.sched_eval.windLengths),
            cycle_ns=int(self.cycle_ns),
            thresholds=self.reward_frame_sizes,
            reference_speed_mbps=float(np.mean(self.sched_eval.port_speeds)) if self.sched_eval.port_speeds.size else 1000.0,
            incoming_stream_count=int(self.reward_incoming_stream_count),
            frame_mix_percentages=self.reward_mix_percentages,
        )
        return {
            "gap_count_total": float(metrics.get("gap_count_total", 0.0)),
            "largest_hole_mean": float(metrics.get("largest_hole_mean", 0.0)),
            "service_block_ratio": float(metrics.get("service_block_ratio", 1.0)),
        }

    def _admission_rate(self, compact: np.ndarray) -> float:
        metrics = self._fragmentation_snapshot(compact)
        return float(np.clip(1.0 - float(metrics.get("service_block_ratio", 1.0)), 0.0, 1.0))

    def step(self, action):
        """Apply one candidate index to current focus stream and return transition."""
        self.step_count += 1

        # V1.1 design: action is only candidate_index. Focus stream comes from observation.
        # This keeps action meaning fully grounded in observed candidate list.
        selected_candidate_index = int(np.asarray(action).item())
        chosen_stream = int(np.clip(self.current_stream_id, 0, max(0, self.n_streams - 1)))

        old_port_schedules = copy.deepcopy(self.port_schedules)
        old_stream_offsets = copy.deepcopy(self.stream_offsets)

        old_compact = self._current_compact()
        old_metrics = self._fragmentation_snapshot(old_compact)
        old_violation_sum = self._violation_sum(old_compact)
        old_admission_rate = self._admission_rate(old_compact)
        old_first_offset = self._stream_current_first_start_ns(chosen_stream)

        candidates, candidate_debug_records = self._compute_stream_candidates(chosen_stream, include_debug=True)
        valid_candidate_count = len(candidates)

        selected_candidate_was_current_start = False
        selected_candidate_was_fallback_unsafe = False
        selected_candidate_passed_full_evaluator = False

        invalid_selection = selected_candidate_index < 0 or selected_candidate_index >= valid_candidate_count
        selected_first_offset_ns: Optional[int] = None
        applied_first_offset_ns = int(old_first_offset)

        move_attempted = False
        if not invalid_selection:
            move_attempted = True
            selected_first_offset_ns = int(candidates[selected_candidate_index])
            self._place_stream(chosen_stream, float(selected_first_offset_ns))
            applied_first_offset_ns = self._stream_current_first_start_ns(chosen_stream)

            selected_candidate_was_current_start = bool(
                int(selected_first_offset_ns) == int(old_first_offset)
            )
            for rec in candidate_debug_records:
                if int(rec["candidate_start_ns"]) == int(selected_first_offset_ns):
                    selected_candidate_was_fallback_unsafe = bool(rec["included_only_by_fallback"])
                    selected_candidate_passed_full_evaluator = bool(rec["passes_full_evaluator"])
                    break

        attempted_compact = self._current_compact()
        attempted_violation_sum = self._violation_sum(attempted_compact)
        attempted_violation_row_sums = self._violation_row_sums(attempted_compact)
        attempted_violation_dict = self._violation_dict(attempted_compact)
        attempted_violation_dict_normalized = self._violation_dict_normalized(attempted_compact)

        rollback_applied = False
        if move_attempted and attempted_violation_sum > 0:
            rollback_applied = True
            self.port_schedules = old_port_schedules
            self.stream_offsets = old_stream_offsets

        new_compact = self._current_compact()
        new_metrics = self._fragmentation_snapshot(new_compact)
        final_violation_sum = self._violation_sum(new_compact)
        new_admission_rate = self._admission_rate(new_compact)
        applied_first_offset_ns = self._stream_current_first_start_ns(chosen_stream)

        old_gap_count = float(old_metrics.get("gap_count_total", 0.0))
        new_gap_count = float(new_metrics.get("gap_count_total", 0.0))
        old_largest_hole_mean = float(old_metrics.get("largest_hole_mean", 0.0))
        new_largest_hole_mean = float(new_metrics.get("largest_hole_mean", 0.0))

        delta_gap = float(old_gap_count - new_gap_count)
        delta_hole = float(new_largest_hole_mean - old_largest_hole_mean)
        delta_attempted_violation = int(attempted_violation_sum - old_violation_sum)
        delta_final_violation = int(final_violation_sum - old_violation_sum)

        delta_gap_norm = delta_gap / max(1.0, old_gap_count)
        delta_hole_norm = delta_hole / max(1.0, float(self.cycle_ns))

        move_penalty = 1.0 if int(applied_first_offset_ns) != int(old_first_offset) else 0.0
        invalid_penalty = 1.0 if (invalid_selection or rollback_applied) else 0.0

        # Admission is kept as a diagnostic only. Sparse scenarios can saturate
        # admission rate, so reward-mode ablations are useful for learnability checks.
        if self.reward_mode == "weighted_fragmentation":
            reward = (
                float(self.reward_weights.get("w_gap", 1.0)) * float(delta_gap_norm)
                + float(self.reward_weights.get("w_hole", 0.5)) * float(delta_hole_norm)
                - float(self.reward_weights.get("w_move", 0.01)) * float(move_penalty)
                - float(self.reward_weights.get("w_invalid", 1.0)) * float(invalid_penalty)
            )
        elif self.reward_mode == "gap_improvement_binary":
            # Debug mode: direct signal for whether action reduces fragmentation count.
            if invalid_selection or rollback_applied:
                reward = -1.0
            elif delta_gap > 0:
                reward = 1.0
            elif delta_gap == 0:
                reward = 0.0
            else:
                reward = -1.0
        elif self.reward_mode == "hole_improvement_binary":
            # Debug mode: direct signal for whether action increases contiguous free space.
            if invalid_selection or rollback_applied:
                reward = -1.0
            elif delta_hole > 0:
                reward = 1.0
            elif delta_hole == 0:
                reward = 0.0
            else:
                reward = -1.0
        else:
            # Guarded by constructor validation, but keep a safe fallback.
            reward = 0.0

        truncated = bool(self.step_count >= self.max_steps)
        terminated = False

        moved_stream = int(chosen_stream) if move_penalty > 0.0 else None

        acted_stream_id = int(chosen_stream)
        self._advance_current_stream()
        next_stream_id = int(self.current_stream_id)

        info = {
            "old_gap_count": float(old_gap_count),
            "new_gap_count": float(new_gap_count),
            "delta_gap": float(delta_gap),
            "old_largest_hole_mean": float(old_largest_hole_mean),
            "new_largest_hole_mean": float(new_largest_hole_mean),
            "delta_hole": float(delta_hole),
            "old_violation_sum": int(old_violation_sum),
            "final_violation_sum": int(final_violation_sum),
            # Backward-compat alias for existing consumers.
            "new_violation_sum": int(final_violation_sum),
            "attempted_violation_sum": int(attempted_violation_sum),
            "delta_attempted_violation": int(delta_attempted_violation),
            "delta_final_violation": int(delta_final_violation),
            "rollback_applied": bool(rollback_applied),
            "is_feasible": bool(final_violation_sum == 0),
            "moved_stream": moved_stream,
            "acted_stream_id": int(acted_stream_id),
            "next_stream_id": int(next_stream_id),
            "selected_candidate_index": int(selected_candidate_index),
            "selected_first_offset_ns": int(selected_first_offset_ns) if selected_first_offset_ns is not None else None,
            "applied_first_offset_ns": int(applied_first_offset_ns),
            "selected_candidate_was_current_start": bool(selected_candidate_was_current_start),
            "selected_candidate_was_fallback_unsafe": bool(selected_candidate_was_fallback_unsafe),
            "selected_candidate_passed_full_evaluator": bool(selected_candidate_passed_full_evaluator),
            "step_count": int(self.step_count),
            "candidate_debug_records": candidate_debug_records,
            "attempted_violation_row_sums": attempted_violation_row_sums.astype(int).tolist(),
            "attempted_violation_dict": attempted_violation_dict,
            "attempted_violation_dict_normalized": attempted_violation_dict_normalized,
            "old_admission_rate": float(old_admission_rate),
            # Final/post-rollback violations can stay zero even when attempted actions are bad.
            # attempted_violation_sum is the primary diagnostic for unsafe attempted moves.
            "new_admission_rate": float(new_admission_rate),
            "delta_admission_rate": float(new_admission_rate - old_admission_rate),
            "valid_candidate_count": int(valid_candidate_count),
            "invalid_selection": bool(invalid_selection),
            # Initial schedule feasibility is explicitly surfaced for debugging.
            "initial_violation_sum": int(self.initial_violation_sum),
            "initial_violation_dict": dict(self.initial_violation_dict),
            "reward_mode": self.reward_mode,
            "reward_weights": dict(self.reward_weights),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset schedule and focus stream for V1 candidate-based episodes."""
        super().reset(seed=seed)

        use_uc_schedule = self.init_from_usecase_schedule
        if options and "init_from_usecase_schedule" in options:
            use_uc_schedule = bool(options["init_from_usecase_schedule"])

        loaded = False
        if use_uc_schedule:
            loaded = self._create_initial_schedule_from_usecase()
        if not loaded:
            self._create_initial_schedule()

        self.step_count = 0
        self.current_stream_id = self._first_active_stream_id()

        # Initial feasibility matters: attempted violations may come from a
        # pre-existing infeasible schedule, not only from the selected move.
        initial_compact = self._current_compact()
        self.initial_violation_sum = int(self._violation_sum(initial_compact))
        self.initial_violation_dict = self._violation_dict(initial_compact)

        return self._get_obs(), {}

    def render(self):
        """Render combined schedule."""
        combined = Schedule(cycle_ns=self.cycle_ns)
        for pidx in sorted(self.port_schedules):
            for slot in self.port_schedules[pidx]:
                combined.append(slot)
        return combined.plot()

    def get_combined_schedule(self) -> Schedule:
        """Return one schedule object containing all per-port slots."""
        combined = Schedule(cycle_ns=self.cycle_ns)
        for pidx in sorted(self.port_schedules):
            for slot in self.port_schedules[pidx]:
                combined.append(slot)
        return combined

    def get_current_compact(self) -> np.ndarray:
        """Public helper for training/evaluation scripts."""
        return self._current_compact()
