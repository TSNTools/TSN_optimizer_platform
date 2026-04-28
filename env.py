from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from evaluator import ScheduleEncodingType, TSNConstraintEvaluator
from schedule import Schedule, Slot
from tsn_scheduling_problem import MBPS_2_BPNS, TSNScheduling
from usecase_generator import Usecase
from benchmark_utils import (
    build_network_generator_from_usecase,
    generate_one_arrival_stream,
    build_usecase_stream_from_request,
)

BYTE_2_BIT = 8


@dataclass
class StreamDecisionFocus:
    stream_id: int


@dataclass
class StreamMoveCandidate:
    stream_id: int
    offsets: List[int]
    cuts: int = 0
    frss_after: float = 0.0
    fse_after: float = 0.0
    source: str = "generated"
    debug: Dict[str, object] = field(default_factory=dict)


class TSNCandidatePathEnv(gym.Env):
    """
    Event-driven TSN GCL defragmentation env.

    Semantics
    ---------
    - The schedule evolves continuously with:
        * arrivals ~ Poisson(lambda_arrival)
        * departures ~ Binomial(n_active, p_dep) with p_dep = 1 - exp(-mu_departure * dt)
    - A departure creates a *decision point*.
    - During a decision point, the agent may:
        * STOP immediately (skip defrag)
        * start a defrag cycle by choosing one candidate
        * continue reallocating oldest-first until it chooses STOP
    - After STOP (or after all focus streams are exhausted), the decision point ends,
      simulation resumes, and we advance until the next departure.
    - One episode ends after `decision_points_per_episode` departure-triggered decision points.
    - Episode SBR = rejected_arrivals / total_arrivals inside that episode.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        problem: TSNScheduling,
        usecase: Usecase,
        useless_gap_threshold_bytes: int = 1024,
        expected_incoming_bytes: int = 1024,
        invalid_penalty: float = 50.0,
        max_candidates: int = 10,
        max_steps: int = 30,
        max_hops: Optional[int] = None,
        init_from_usecase_schedule: bool = True,
        include_current_offset_candidate: bool = True,
        per_hop_candidate_limit: int = 3,
        beam_width: Optional[int] = None,
        defrag_cycle_penalty: float = 0.05,
        reallocation_penalty: float = 0.01,
        lambda_arrival: float = 1.0,
        mu_departure: float = 0.2,
        dt: float = 1.0,
        decision_points_per_episode: int = 100,
        ed_ed_settings: Optional[dict] = None,
        top_k_oldest_random: int = 10,
        seed: int = 42,
    ):
        super().__init__()

        self.problem = problem
        self.base_usecase = copy.deepcopy(usecase)
        self.usecase = copy.deepcopy(usecase) 

        self.useless_gap_threshold_bytes = int(useless_gap_threshold_bytes)
        self.expected_incoming_bytes = int(expected_incoming_bytes)
        self.invalid_penalty = float(invalid_penalty)

        self.max_candidates = max(1, int(max_candidates))
        self.max_steps = max(1, int(max_steps))
        self.max_hops = max_hops
        self.init_from_usecase_schedule = bool(init_from_usecase_schedule)
        self.include_current_offset_candidate = bool(include_current_offset_candidate)

        self.top_k_oldest_random = int(top_k_oldest_random)

        self.per_hop_candidate_limit = max(1, int(per_hop_candidate_limit))
        self.beam_width = int(beam_width) if beam_width is not None else max(self.max_candidates * 2, 8)

        self.defrag_cycle_penalty = float(defrag_cycle_penalty)
        self.reallocation_penalty = float(reallocation_penalty)

        self.lambda_arrival = float(lambda_arrival)
        self.mu_departure = float(mu_departure)
        self.dt = float(dt)
        self.p_dep = 1.0 - math.exp(-self.mu_departure * self.dt)
        self.decision_points_per_episode = int(decision_points_per_episode)
        self.ed_ed_settings = None
        self.seed_value = int(seed)

        self.rng_py = random.Random(self.seed_value)
        self.rng_np = np.random.default_rng(self.seed_value)

        self.step_count = 0
        self.in_defrag_cycle = False
        self.decision_active = False

        self.focus = StreamDecisionFocus(stream_id=0)
        self.current_candidates: List[StreamMoveCandidate] = []

        self.last_valid_port_schedules: Dict[int, Schedule] = {}
        self.last_valid_stream_offsets: Dict[int, Dict[int, float]] = {}

        self.blocked_requests = 0
        self.total_requests = 1
        self.accepted_requests = 0
        self.current_time = 0.0
        self.stream_arrival_times: Dict[int, float] = {}

        self.episode_invalid_moves = 0
        self.episode_violation_sum = 0
        self.episode_decision_points = 0
        self.sim_steps_between_decisions = 0

        self.decision_stream_order: List[int] = []
        self.decision_pointer = 0
        self.port_intervals: Dict[int, List[Tuple[int, int]]] = {}
        self.network_generator = None

        with self.problem.use_instance(self.usecase):
            self.n_ports = int(self.problem.n_ports)
            self.n_streams = int(self.problem.n_streams)
            self.cycle_ns = int(self.problem.macro_cycle)

            self._port_speed_mbps: Dict[int, float] = {}
            for pid, pinfo in self.problem.ports.items():
                self._port_speed_mbps[int(pid)] = float(pinfo.get("speed", 1000.0))

            self.stream_info: List[Dict[str, object]] = []
            enriched_streams = self.problem.streams

            for s_idx in range(self.n_streams):
                raw = self.usecase.streams[s_idx]
                enr = enriched_streams[s_idx]

                path = list(enr.get("pathLists", [[]])[0]) if enr.get("pathLists") else []
                interval_ns = int(raw.get("interval", self.cycle_ns))
                frame_bits = int(TSNScheduling._frame_size(raw))
                max_lat_ns = int(raw.get("maxLatency", self.cycle_ns))

                port_dur_ns: Dict[int, float] = {}
                for pid in path:
                    spd = self._port_speed_mbps.get(int(pid), 1000.0)
                    port_dur_ns[int(pid)] = frame_bits / (spd * MBPS_2_BPNS)

                total_hop_cost_ns = 0.0
                hop_delays: List[float] = []
                for k in range(len(path) - 1):
                    try:
                        src_nm = self.problem.port_id_to_name[path[k]]
                        dst_nm = self.problem.port_id_to_name[path[k + 1]]
                        delay = float(self.problem.hop_cost_eg2eg(src_nm, dst_nm, frame_bits))
                    except Exception:
                        delay = float(port_dur_ns.get(int(path[k]), 0.0))
                    hop_delays.append(delay)
                    total_hop_cost_ns += delay

                total_required_duration_ns = float(sum(port_dur_ns.get(int(pid), 0.0) for pid in path))

                self.stream_info.append(
                    {
                        "path": path,
                        "interval_ns": interval_ns,
                        "frame_bits": frame_bits,
                        "max_latency_ns": max_lat_ns,
                        "port_dur_ns": port_dur_ns,
                        "total_hop_cost_ns": total_hop_cost_ns,
                        "hop_delays": hop_delays,
                        "total_required_duration_ns": total_required_duration_ns,
                    }
                )

            self.max_latency_ns = max(
                [float(info["max_latency_ns"]) for info in self.stream_info] + [1.0]
            )
            self.max_required_duration_ns = max(
                [float(info["total_required_duration_ns"]) for info in self.stream_info] + [1.0]
            )
            inferred_max_hops = max([len(info["path"]) for info in self.stream_info] + [1])
            self.max_hops = int(max_hops) if max_hops is not None else inferred_max_hops

            self.sched_eval = TSNConstraintEvaluator(
                self.problem,
                check_routing=False,
                check_task_placement=False,
                scheduling_type=ScheduleEncodingType.START_TIME,
            )

        self.observation_space = spaces.Dict(
            {
                "base_features": spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
                "current_offsets": spaces.Box(low=-1.0, high=1.0, shape=(self.max_hops,), dtype=np.float32),
                "candidate_offsets": spaces.Box(
                    low=-1.0, high=1.0, shape=(self.max_candidates, self.max_hops), dtype=np.float32
                ),
                "candidate_features": spaces.Box(
                    low=-1.0, high=1.0, shape=(self.max_candidates, 4), dtype=np.float32
                ),
                "candidate_mask": spaces.Box(
                    low=0, high=1, shape=(self.max_candidates + 1,), dtype=np.int8
                ),
            }
        )
        self.action_space = spaces.Discrete(self.max_candidates + 1)

        self.port_schedules: Dict[int, Schedule] = {}
        self.stream_offsets: Dict[int, Dict[int, float]] = {}

    # ------------------------------------------------------------------
    # Schedule / runtime sync
    # ------------------------------------------------------------------
    def _load_state_from_schedule(self, schedule: Schedule) -> None:
        self.port_schedules = {}
        self.stream_offsets = {}

        for slot in schedule:
            if slot.port_id is None:
                continue

            pidx = int(slot.port_id) - 1
            self.port_schedules.setdefault(pidx, Schedule(cycle_ns=self.cycle_ns)).append(slot)

            if slot.stream_id is not None:
                s_idx = int(slot.stream_id)
                pid = int(slot.port_id)
                self.stream_offsets.setdefault(s_idx, {})[pid] = float(slot.start_ns)

        self._refresh_port_intervals()
        self._sync_evaluator_with_schedule_durations()

    def _refresh_port_intervals(self) -> None:
        combined = self.get_combined_schedule()
        relevant_ports = sorted({int(slot.port_id) for slot in combined if slot.port_id is not None})
        self.port_intervals = {
            pid: combined._occupied_intervals_on_port(pid, combined.cycle_ns)
            for pid in relevant_ports
        }

    def _sync_usecase_schedule(self) -> None:
        self.usecase.schedule = self.get_combined_schedule()
        self._refresh_port_intervals()

    def _active_stream_ids(self) -> List[int]:
        return sorted({int(slot.stream_id) for slot in self.get_combined_schedule() if slot.stream_id is not None})

    def _removed_stream_ids(self, before_ids: List[int], after_ids: List[int]) -> List[int]:
        after_set = set(int(x) for x in after_ids)
        return [int(x) for x in before_ids if int(x) not in after_set]

    def _create_initial_schedule_from_usecase(self) -> bool:
        try:
            port_name_to_id, _ = self.usecase.port_name_id_mapping
            imported_schedule = Schedule.from_usecase(self.usecase, port_name_to_id)
        except Exception:
            return False
        self._load_state_from_schedule(imported_schedule)
        self.usecase.schedule = imported_schedule
        return True

    def _create_fallback_initial_schedule(self) -> None:
        self.port_schedules = {}
        self.stream_offsets = {}

        for s_idx in range(self.n_streams):
            path = list(self.stream_info[s_idx]["path"])
            if not path:
                continue

            offsets: Dict[int, float] = {}
            cum_delay = 0.0
            for k, pid in enumerate(path):
                dur = float(self.stream_info[s_idx]["port_dur_ns"].get(int(pid), 0.0))
                max_start = max(0.0, float(self.cycle_ns) - dur)
                offsets[int(pid)] = max(0.0, min(max_start, cum_delay))
                if k < len(self.stream_info[s_idx]["hop_delays"]):
                    cum_delay += float(self.stream_info[s_idx]["hop_delays"][k])

            self._place_stream_with_offsets(s_idx, offsets)

        self._sync_usecase_schedule()
        self._sync_evaluator_with_schedule_durations()

    def _backup_current_as_last_valid(self) -> None:
        self.last_valid_port_schedules = copy.deepcopy(self.port_schedules)
        self.last_valid_stream_offsets = copy.deepcopy(self.stream_offsets)

    def _restore_last_valid(self) -> None:
        self.port_schedules = copy.deepcopy(self.last_valid_port_schedules)
        self.stream_offsets = copy.deepcopy(self.last_valid_stream_offsets)
        self._sync_usecase_schedule()
        self._sync_evaluator_with_schedule_durations()

    # ------------------------------------------------------------------
    # Schedule utilities
    # ------------------------------------------------------------------
    def _place_stream_with_offsets(self, s_idx: int, offsets: Dict[int, float]) -> None:
        sinfo = self.stream_info[s_idx]
        path = list(sinfo["path"])
        if not path:
            return

        for pid in path:
            pidx = int(pid) - 1
            if pidx in self.port_schedules:
                kept = [slot for slot in self.port_schedules[pidx] if int(slot.stream_id) != int(s_idx)]
                self.port_schedules[pidx] = Schedule(cycle_ns=self.cycle_ns, iterable=kept)

        self.stream_offsets[s_idx] = {}

        for pid in path:
            pid = int(pid)
            dur = float(sinfo["port_dur_ns"].get(pid, 0.0))
            st = float(offsets.get(pid, 0.0))
            st = max(0.0, min(float(self.cycle_ns) - dur, st))
            en = st + dur

            slot = Slot(
                start_ns=int(round(st)),
                end_ns=int(round(en)),
                period_ns=int(sinfo["interval_ns"]),
                queue_id=0,
                stream_id=int(s_idx),
                port_id=pid,
            )

            pidx = pid - 1
            self.port_schedules.setdefault(pidx, Schedule(cycle_ns=self.cycle_ns)).append(slot)
            self.stream_offsets[s_idx][pid] = float(st)

    def _apply_stream_move(self, stream_id: int, offsets: List[int]) -> bool:
        path = self._stream_path(stream_id)
        if not path or len(offsets) != len(path):
            return False

        offsets_dict = {int(pid): float(off) for pid, off in zip(path, offsets)}
        self._place_stream_with_offsets(int(stream_id), offsets_dict)
        self._sync_usecase_schedule()
        self._sync_evaluator_with_schedule_durations()
        return True

    def _sorted_slots_on_port(self, pidx: int) -> List[Slot]:
        sched = self.port_schedules.get(int(pidx), Schedule(cycle_ns=self.cycle_ns))
        return sorted(list(sched), key=lambda s: (int(s.start_ns), int(s.end_ns), int(s.stream_id)))

    def _stream_path(self, s_idx: int) -> List[int]:
        if int(s_idx) < 0 or int(s_idx) >= len(self.stream_info):
            return []
        return list(self.stream_info[int(s_idx)]["path"])

    def _ordered_streams_by_age(self) -> List[int]:
        active = self._active_stream_ids()
        if not active:
            return []

        ordered = sorted(
            active,
            key=lambda sid: (self.stream_arrival_times.get(int(sid), 0.0), int(sid))
        )

        k = min(self.top_k_oldest_random, len(ordered))
        top_k = ordered[:k]

        first = int(self.rng_py.choice(top_k))

        remaining = [sid for sid in ordered if int(sid) != first]

        return [first] + remaining

    def _advance_stream_focus(self) -> None:
        if not self.decision_stream_order:
            self.focus = StreamDecisionFocus(stream_id=0)
            return

        self.decision_pointer += 1
        if self.decision_pointer >= len(self.decision_stream_order):
            self.focus = StreamDecisionFocus(stream_id=self.decision_stream_order[-1])
            return

        self.focus = StreamDecisionFocus(stream_id=int(self.decision_stream_order[self.decision_pointer]))

    # ------------------------------------------------------------------
    # Metrics / fragmentation / gaps
    # ------------------------------------------------------------------
    def _useless_gap_threshold_ns_for_port(self, port_id: int) -> float:
        spd = self._port_speed_mbps.get(int(port_id), 1000.0)
        return (self.useless_gap_threshold_bytes * BYTE_2_BIT) / (spd * MBPS_2_BPNS)

    def _expected_incoming_duration_ns_for_port(self, port_id: int) -> float:
        spd = self._port_speed_mbps.get(int(port_id), 1000.0)
        return (self.expected_incoming_bytes * BYTE_2_BIT) / (spd * MBPS_2_BPNS) + 4000.0

    def _port_gaps(self, pidx: int) -> List[float]:
        slots = self._sorted_slots_on_port(pidx)
        if not slots:
            return [float(self.cycle_ns)]

        gaps: List[float] = []
        cursor = 0.0
        for slot in slots:
            if float(slot.start_ns) > cursor:
                gaps.append(float(slot.start_ns) - cursor)
            cursor = max(cursor, float(slot.end_ns))
        if cursor < float(self.cycle_ns):
            gaps.append(float(self.cycle_ns) - cursor)
        return gaps

    def _fragmentation_rss(self) -> float:
        """
        FRSS based on number of useful gaps, not gap durations.

        A useful gap is a gap whose duration is >= the time needed to transmit
        useless_gap_threshold_bytes on that port.

        Higher value means useful space is concentrated in fewer gaps.
        Lower value means useful gaps are more spread out / fragmented.
        """
        per_port_vals: List[float] = []

        for pidx in range(self.n_ports):
            port_id = pidx + 1
            threshold_ns = self._useless_gap_threshold_ns_for_port(port_id)
            gaps = self._port_gaps(pidx)

            useful_gap_count = sum(1 for g in gaps if float(g) >= float(threshold_ns))

            if useful_gap_count <= 0:
                per_port_vals.append(0.0)
                continue

            # Count-based RSS:
            # vector = [1, 1, ..., 1] over useful gaps
            # sqrt(sum(1^2)) / sum(1) = sqrt(k) / k = 1/sqrt(k)
            frss_count = 1.0 / np.sqrt(float(useful_gap_count))
            per_port_vals.append(float(frss_count))

        return float(np.mean(per_port_vals)) if per_port_vals else 0.0

    def _fragmentation_entropy(self) -> float:
        per_port_vals: List[float] = []
        S = float(self.cycle_ns)
        if S <= 0.0:
            return 0.0
        for pidx in range(self.n_ports):
            gaps = self._port_gaps(pidx)
            if not gaps:
                per_port_vals.append(0.0)
                continue
            arr = np.asarray(gaps, dtype=np.float64) / S
            arr = arr[arr > 0.0]
            per_port_vals.append(float(-np.sum(arr * np.log(arr))) if arr.size else 0.0)
        return float(np.mean(per_port_vals)) if per_port_vals else 0.0

    def _global_metrics(self) -> Dict[str, float]:
        largest_hole_ns = 0.0
        total_useless_gaps = 0.0
        total_gap_count = 0.0
        total_used_time = 0.0
        acceptance_count = 0.0
        total_useful_gaps = 0.0

        for pidx in range(self.n_ports):
            port_id = pidx + 1
            threshold_ns = self._useless_gap_threshold_ns_for_port(port_id)
            expected_dur_ns = self._expected_incoming_duration_ns_for_port(port_id)
            gaps = self._port_gaps(pidx)

            largest_hole_ns = max(largest_hole_ns, max(gaps) if gaps else 0.0)
            total_useless_gaps += sum(1 for g in gaps if g < threshold_ns)
            total_useful_gaps += sum(1 for g in gaps if g >= threshold_ns)
            total_gap_count += len(gaps)
            total_used_time += float(self.cycle_ns) - float(sum(gaps))

            if expected_dur_ns > 0:
                acceptance_count += sum(int(g // expected_dur_ns) for g in gaps)

        mean_util = total_used_time / max(1.0, float(self.n_ports * self.cycle_ns))
        return {
            "largest_hole_ns": float(largest_hole_ns),
            "total_useless_gaps": float(total_useless_gaps),
            "total_useful_gaps": float(total_useful_gaps),
            "total_gap_count": float(total_gap_count),
            "mean_utilization": float(np.clip(mean_util, 0.0, 1.0)),
            "acceptance_count": float(acceptance_count),
            "frss": float(self._fragmentation_rss()),
            "fse": float(self._fragmentation_entropy()),
        }

    def _stream_current_offset_vector(self, stream_id: int) -> List[int]:
        path = self._stream_path(stream_id)
        offs = self.stream_offsets.get(int(stream_id), {})
        return [int(round(float(offs.get(int(pid), 0.0)))) for pid in path]

    def _stream_cut_count(self, stream_id: int, offsets: Optional[List[int]] = None) -> int:
        path = self._stream_path(stream_id)
        if not path:
            return 0
        if offsets is None:
            offsets = self._stream_current_offset_vector(stream_id)
        hop_delays = list(self.stream_info[int(stream_id)]["hop_delays"])
        cuts = 0
        for k in range(len(offsets) - 1):
            expected_next = float(offsets[k]) + float(hop_delays[k])
            if float(offsets[k + 1]) > expected_next:
                cuts += 1
        return int(cuts)

    def _current_sbr(self) -> float:
        return float(self.blocked_requests) / max(1.0, float(self.total_requests))

    def _stop_reward(self) -> float:
        sbr = max(self._current_sbr(), 1.0 / max(1.0, float(self.total_requests)))
        return float(-np.log10(sbr) / 3.0)

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------
    def _occupied_intervals_on_port_excluding_stream(self, port_id: int, stream_id: int) -> List[Tuple[int, int]]:
        pidx = int(port_id) - 1
        intervals: List[Tuple[int, int]] = []
        for slot in self._sorted_slots_on_port(pidx):
            if int(slot.stream_id) == int(stream_id):
                continue
            st = max(0, int(slot.start_ns))
            en = min(int(self.cycle_ns), int(slot.end_ns))
            if en > st:
                intervals.append((st, en))
        if not intervals:
            return []
        intervals.sort(key=lambda x: x[0])
        merged: List[Tuple[int, int]] = [intervals[0]]
        for st, en in intervals[1:]:
            last_st, last_en = merged[-1]
            if st <= last_en:
                merged[-1] = (last_st, max(last_en, en))
            else:
                merged.append((st, en))
        return merged

    def _free_gaps_on_port_excluding_stream(self, port_id: int, stream_id: int) -> List[Tuple[int, int]]:
        occupied = self._occupied_intervals_on_port_excluding_stream(port_id, stream_id)
        if not occupied:
            return [(0, int(self.cycle_ns))]
        free_gaps: List[Tuple[int, int]] = []
        cursor = 0
        for st, en in occupied:
            if st > cursor:
                free_gaps.append((cursor, st))
            cursor = max(cursor, en)
        if cursor < int(self.cycle_ns):
            free_gaps.append((cursor, int(self.cycle_ns)))
        return free_gaps

    def _stream_hop_duration_ns(self, stream_id: int, hop_index: int) -> int:
        path = self._stream_path(stream_id)
        if hop_index < 0 or hop_index >= len(path):
            return 0
        pid = int(path[hop_index])
        dur = float(self.stream_info[int(stream_id)]["port_dur_ns"].get(pid, 0.0))
        return int(np.ceil(dur))

    def _candidate_start_set_for_stream_hop(self, stream_id: int, hop_index: int, min_start_ns: int) -> List[int]:
        path = self._stream_path(stream_id)
        if hop_index < 0 or hop_index >= len(path):
            return []

        port_id = int(path[hop_index])
        dur = int(self._stream_hop_duration_ns(stream_id, hop_index))
        current_offsets = self._stream_current_offset_vector(stream_id)
        cur = int(current_offsets[hop_index]) if hop_index < len(current_offsets) else 0

        if dur <= 0:
            return [max(0, int(min_start_ns))]

        free_gaps = self._free_gaps_on_port_excluding_stream(port_id, stream_id)
        starts = set()
        for gap_start, gap_end in free_gaps:
            start = max(int(gap_start), int(min_start_ns))
            if start + dur > int(gap_end):
                continue
            starts.add(start)
            cursor = start + dur
            added = 1
            while cursor + dur <= int(gap_end) and added < self.per_hop_candidate_limit:
                starts.add(int(cursor))
                cursor += dur
                added += 1

        if self.include_current_offset_candidate and cur >= int(min_start_ns):
            starts.add(cur)
        if not starts and cur >= int(min_start_ns):
            starts.add(cur)
        return sorted(starts)

    def _score_partial_offsets(self, stream_id: int, offsets: List[int]) -> Tuple[int, float]:
        cuts = self._stream_cut_count(stream_id, offsets=offsets)
        span = float(max(offsets) - min(offsets)) if offsets else 0.0
        return cuts, span

    def _evaluate_candidate_metrics(self, stream_id: int, offsets: List[int]) -> Optional[Dict[str, float]]:
        saved_ports = copy.deepcopy(self.port_schedules)
        saved_offsets = copy.deepcopy(self.stream_offsets)

        applied = self._apply_stream_move(stream_id, offsets)
        if not applied:
            self.port_schedules = saved_ports
            self.stream_offsets = saved_offsets
            return None

        violation_sum = self._violation_sum()
        if violation_sum != 0:
            self.port_schedules = saved_ports
            self.stream_offsets = saved_offsets
            self._sync_usecase_schedule()
            self._sync_evaluator_with_schedule_durations()
            return None

        g = self._global_metrics()
        self.port_schedules = saved_ports
        self.stream_offsets = saved_offsets
        self._sync_usecase_schedule()
        self._sync_evaluator_with_schedule_durations()
        return {"frss_after": float(g["frss"]), "fse_after": float(g["fse"])}

    def _candidate_list_for_stream(self, stream_id: int) -> List[StreamMoveCandidate]:
        path = self._stream_path(stream_id)
        if not path:
            return []

        hop_delays = list(self.stream_info[int(stream_id)]["hop_delays"])
        current_offsets = self._stream_current_offset_vector(stream_id)
        partials: List[List[int]] = [[]]

        for hop_idx in range(len(path)):
            next_partials: List[List[int]] = []
            for partial in partials:
                min_start_ns = 0 if hop_idx == 0 else int(round(int(partial[hop_idx - 1]) + float(hop_delays[hop_idx - 1])))
                hop_candidates = self._candidate_start_set_for_stream_hop(stream_id, hop_idx, min_start_ns)
                hop_candidates = hop_candidates[: self.per_hop_candidate_limit]
                for st in hop_candidates:
                    next_partials.append(partial + [int(st)])

            if not next_partials:
                if current_offsets and len(current_offsets) == len(path):
                    next_partials = [list(current_offsets)]
                else:
                    return []

            next_partials = sorted(next_partials, key=lambda vec: self._score_partial_offsets(stream_id, vec))[: self.beam_width]
            partials = next_partials

        if self.include_current_offset_candidate and current_offsets and len(current_offsets) == len(path):
            partials.append(list(current_offsets))

        unique_vecs: List[List[int]] = []
        seen = set()
        for vec in partials:
            key = tuple(int(x) for x in vec)
            if key not in seen:
                seen.add(key)
                unique_vecs.append([int(x) for x in vec])

        candidates: List[StreamMoveCandidate] = []
        for vec in unique_vecs:
            eval_metrics = self._evaluate_candidate_metrics(stream_id, vec)
            if eval_metrics is None:
                continue
            cuts = self._stream_cut_count(stream_id, offsets=vec)
            candidates.append(
                StreamMoveCandidate(
                    stream_id=int(stream_id),
                    offsets=[int(x) for x in vec],
                    cuts=int(cuts),
                    frss_after=float(eval_metrics["frss_after"]),
                    fse_after=float(eval_metrics["fse_after"]),
                    source="generated",
                )
            )

        candidates.sort(
            key=lambda c: (-float(c.frss_after), float(c.fse_after), int(c.cuts), tuple(c.offsets))
        )
        return candidates[: self.max_candidates]

    def _sync_evaluator_with_schedule_durations(self) -> None:
        self.sched_eval.windLengths[:, :] = 0.0
        for pidx, sched in self.port_schedules.items():
            for slot in sched:
                if slot.stream_id is None or slot.port_id is None:
                    continue
                port_idx = int(slot.port_id) - 1
                stream_id = int(slot.stream_id)
                actual_dur = float(slot.end_ns - slot.start_ns)
                self.sched_eval.windLengths[port_idx, stream_id] = actual_dur
                if stream_id < len(self.stream_info):
                    self.stream_info[stream_id]["port_dur_ns"][int(slot.port_id)] = actual_dur

    def _rebuild_runtime_metadata(self) -> None:
        self.problem.usecase = self.usecase
        self.problem.build_metadata()
        with self.problem.use_instance(self.usecase):
            self.n_ports = int(self.problem.n_ports)
            self.n_streams = int(self.problem.n_streams)
            self.cycle_ns = int(self.problem.macro_cycle)

            self._port_speed_mbps = {}
            for pid, pinfo in self.problem.ports.items():
                self._port_speed_mbps[int(pid)] = float(pinfo.get("speed", 1000.0))

            self.stream_info = []
            enriched_streams = self.problem.streams

            for s_idx in range(self.n_streams):
                raw = self.usecase.streams[s_idx]
                enr = enriched_streams[s_idx]

                path = list(enr.get("pathLists", [[]])[0]) if enr.get("pathLists") else []
                interval_ns = int(raw.get("interval", self.cycle_ns))
                frame_bits = int(TSNScheduling._frame_size(raw))
                max_lat_ns = int(raw.get("maxLatency", self.cycle_ns))

                port_dur_ns = {}
                for pid in path:
                    spd = self._port_speed_mbps.get(int(pid), 1000.0)
                    port_dur_ns[int(pid)] = frame_bits / (spd * MBPS_2_BPNS)

                total_hop_cost_ns = 0.0
                hop_delays = []
                for k in range(len(path) - 1):
                    try:
                        src_nm = self.problem.port_id_to_name[path[k]]
                        dst_nm = self.problem.port_id_to_name[path[k + 1]]
                        delay = float(self.problem.hop_cost_eg2eg(src_nm, dst_nm, frame_bits))
                    except Exception:
                        delay = float(port_dur_ns.get(int(path[k]), 0.0))
                    hop_delays.append(delay)
                    total_hop_cost_ns += delay

                total_required_duration_ns = float(
                    sum(port_dur_ns.get(int(pid), 0.0) for pid in path)
                )

                self.stream_info.append(
                    {
                        "path": path,
                        "interval_ns": interval_ns,
                        "frame_bits": frame_bits,
                        "max_latency_ns": max_lat_ns,
                        "port_dur_ns": port_dur_ns,
                        "total_hop_cost_ns": total_hop_cost_ns,
                        "hop_delays": hop_delays,
                        "total_required_duration_ns": total_required_duration_ns,
                    }
                )

            self.max_latency_ns = max(
                [float(info["max_latency_ns"]) for info in self.stream_info] + [1.0]
            )
            self.max_required_duration_ns = max(
                [float(info["total_required_duration_ns"]) for info in self.stream_info] + [1.0]
            )

            inferred_max_hops = max([len(info["path"]) for info in self.stream_info] + [1])
            if self.max_hops is None:
                self.max_hops = inferred_max_hops
            else:
                self.max_hops = max(int(self.max_hops), int(inferred_max_hops))

            self.sched_eval = TSNConstraintEvaluator(
                self.problem,
                check_routing=False,
                check_task_placement=False,
                scheduling_type=ScheduleEncodingType.START_TIME,
            )

            self.sched_eval.stream_pairwise_windowSum = (
                self.sched_eval.windLengths[:, :, None] + self.sched_eval.windLengths[:, None, :]
            )

    # ------------------------------------------------------------------
    # Simulator between decision points
    # ------------------------------------------------------------------
    def _sample_arrivals(self) -> int:
        return int(self.rng_np.poisson(self.lambda_arrival))

    def _process_arrivals(self) -> None:
        sampled_arrivals = self._sample_arrivals()
        if sampled_arrivals <= 0:
            return

        self._sync_usecase_schedule()
        sched = self.usecase.schedule

        for _ in range(sampled_arrivals):
            self.total_requests += 1

            raw_stream = generate_one_arrival_stream(
                self.network_generator,
                ed_ed_settings=self.ed_ed_settings,
            )
            if raw_stream is None:
                self.blocked_requests += 1
                continue

            stream_id = len(self.usecase.streams)

            stream = build_usecase_stream_from_request(
                raw_stream,
                stream_id,
                self.usecase,
                k_paths=1,
            )
            if stream is None:
                self.blocked_requests += 1
                continue

            result = sched.place_stream_first_fit_fast(
                stream=stream,
                usecase=self.usecase,
                problem=self.problem,
                port_intervals=self.port_intervals,
                max_candidates_per_hop=1,
            )

            self._rebuild_runtime_metadata()

            if result.get("accepted", False):
                self.accepted_requests += 1
                self.stream_arrival_times[int(stream_id)] = float(self.current_time)
            else:
                self.blocked_requests += 1

            self._load_state_from_schedule(sched)
    
    def _advance_until_next_decision_point(self) -> bool:
        while self.episode_decision_points < self.decision_points_per_episode:
            self.current_time += self.dt
            self.sim_steps_between_decisions += 1

            self._sync_usecase_schedule()
            sched = self.usecase.schedule
            active_stream_ids = self._active_stream_ids()
            n_active = len(active_stream_ids)

            if n_active > 0:
                n_dep = int(self.rng_np.binomial(n_active, self.p_dep))
                if n_dep > 0:
                    before_ids = list(active_stream_ids)
                    sched.delete_streams(
                        n_remove=n_dep,
                        usecase=self.usecase,
                        rng=self.rng_py,
                        port_intervals=self.port_intervals,
                        stream_arrival_times=self.stream_arrival_times,
                    )
                    self._load_state_from_schedule(sched)
                    after_ids = self._active_stream_ids()
                    for sid in self._removed_stream_ids(before_ids, after_ids):
                        self.stream_arrival_times.pop(int(sid), None)

                    self.episode_decision_points += 1
                    if self._begin_decision_point():
                        return True
                    continue

            self._process_arrivals()

        self.decision_active = False
        self.current_candidates = []
        return False
    
    def _begin_decision_point(self) -> bool:
        self.decision_stream_order = self._ordered_streams_by_age()
        self.decision_pointer = 0
        self.in_defrag_cycle = False

        if not self.decision_stream_order:
            self.focus = StreamDecisionFocus(stream_id=0)
            self.current_candidates = []
            self.decision_active = False
            return False

        self.focus = StreamDecisionFocus(stream_id=int(self.decision_stream_order[0]))
        self.current_candidates = self._candidate_list_for_stream(int(self.focus.stream_id))
        self.decision_active = True
        return True

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        stream_id = int(self.focus.stream_id)
        path = self._stream_path(stream_id)
        current_offsets = self._stream_current_offset_vector(stream_id)

        g = self._global_metrics()
        age = float(self.current_time - self.stream_arrival_times.get(stream_id, self.current_time))
        total_required_duration = float(self.stream_info[stream_id]["total_required_duration_ns"]) if path and stream_id < len(self.stream_info) else 0.0
        hop_count = len(path)
        current_cuts = self._stream_cut_count(stream_id, offsets=current_offsets) if path else 0

        age_norm = np.tanh(age / max(1.0, float(self.current_time + self.dt)))
        duration_norm = float(total_required_duration / max(1.0, self.max_required_duration_ns))
        hop_count_norm = float(hop_count / max(1, self.max_hops))
        cut_norm = float(current_cuts / max(1, self.max_hops - 1))
        frss_norm = np.tanh(float(g["frss"]))
        fse_norm = np.tanh(float(g["fse"]))
        in_cycle_norm = 1.0 if self.in_defrag_cycle else 0.0

        base_features = np.asarray(
            [
                age_norm * 2.0 - 1.0,
                duration_norm * 2.0 - 1.0,
                hop_count_norm * 2.0 - 1.0,
                cut_norm * 2.0 - 1.0,
                frss_norm * 2.0 - 1.0,
                fse_norm * 2.0 - 1.0,
                in_cycle_norm * 2.0 - 1.0,
            ],
            dtype=np.float32,
        )

        current_offsets_arr = np.full((self.max_hops,), -1.0, dtype=np.float32)
        for i, off in enumerate(current_offsets[: self.max_hops]):
            current_offsets_arr[i] = (float(off) / float(self.cycle_ns)) * 2.0 - 1.0

        candidate_offsets = np.full((self.max_candidates, self.max_hops), -1.0, dtype=np.float32)
        candidate_features = np.full((self.max_candidates, 4), -1.0, dtype=np.float32)
        for i, cand in enumerate(self.current_candidates[: self.max_candidates]):
            for h, off in enumerate(cand.offsets[: self.max_hops]):
                candidate_offsets[i, h] = (float(off) / float(self.cycle_ns)) * 2.0 - 1.0
            cuts_norm = float(cand.cuts / max(1, self.max_hops - 1))
            frss_after_norm = np.tanh(float(cand.frss_after))
            fse_after_norm = np.tanh(float(cand.fse_after))
            move_ratio = 0.0
            if current_offsets:
                diff = sum(int(a != b) for a, b in zip(cand.offsets, current_offsets))
                move_ratio = float(diff / max(1, len(current_offsets)))
            candidate_features[i] = np.asarray(
                [
                    cuts_norm * 2.0 - 1.0,
                    frss_after_norm * 2.0 - 1.0,
                    fse_after_norm * 2.0 - 1.0,
                    move_ratio * 2.0 - 1.0,
                ],
                dtype=np.float32,
            )

        candidate_mask = np.zeros((self.max_candidates + 1,), dtype=np.int8)
        candidate_mask[0] = 1
        valid_n = min(len(self.current_candidates), self.max_candidates)
        if valid_n > 0:
            candidate_mask[1 : 1 + valid_n] = 1

        return {
            "base_features": base_features,
            "current_offsets": current_offsets_arr,
            "candidate_offsets": candidate_offsets,
            "candidate_features": candidate_features,
            "candidate_mask": candidate_mask,
        }

    # ------------------------------------------------------------------
    # Validity / reward / stepping
    # ------------------------------------------------------------------
    def get_combined_schedule(self) -> Schedule:
        combined = Schedule(cycle_ns=self.cycle_ns)
        for pidx in sorted(self.port_schedules):
            for slot in self.port_schedules[pidx]:
                combined.append(slot)
        return combined

    def _current_compact(self) -> np.ndarray:
        return self.get_combined_schedule().to_compact_encoding(self.n_ports, self.n_streams)

    def _violation_sum(self) -> int:
        compact = self._current_compact()
        return int(self.sched_eval.per_stream_all_violations(compact).sum())

    def _finish_decision_point_and_continue(self):
        self.decision_active = False
        self.in_defrag_cycle = False
        self.current_candidates = []
        has_next = self._advance_until_next_decision_point()
        terminated = not has_next
        truncated = bool(self.step_count >= self.max_steps)
        return terminated, truncated

    def _terminal_info(self) -> Dict[str, float]:
        sbr = self._current_sbr()
        return {
            "episode_sbr": float(sbr),
            "episode_acceptance": float(self.accepted_requests / max(1, self.total_requests)),
            "episode_arrivals": int(self.total_requests),
            "episode_rejections": int(self.blocked_requests),
            "episode_decision_points": int(self.episode_decision_points),
            "episode_invalid_moves": int(self.episode_invalid_moves),
            "episode_violation_sum": int(self.episode_violation_sum),
            "episode_reallocations": int(self.episode_reallocations),
            "episode_defrag_cycles": int(self.episode_defrag_cycles),
        }

    def step(self, action: int):
        self.step_count += 1
        action = int(np.asarray(action).item())
        forced_to_stop = False

        if not self.decision_active:
            terminated = True
            truncated = False
            info = {"stop_action": True, "forced_to_stop": False, "invalid_move": False, **self._terminal_info()}
            return self._get_obs(), 0.0, terminated, truncated, info

        before_metrics = self._global_metrics()
        current_stream_id = int(self.focus.stream_id)
        in_defrag_cycle_before = bool(self.in_defrag_cycle)

        invalid = False
        stop_action = False
        chosen_candidate: Optional[StreamMoveCandidate] = None
        reward = 0.0


        # If action points to padded / nonexistent candidate, convert it to STOP
        if action > 0:
            cand_idx = action - 1
            if cand_idx < 0 or cand_idx >= len(self.current_candidates):
                action = 0
                forced_to_stop = True

        if action == 0:
            stop_action = True
            terminated, truncated = self._finish_decision_point_and_continue()
            reward = 1 - self._current_sbr()
            info = {
                "stop_action": True,
                "forced_to_stop": bool(forced_to_stop),
                "invalid_move": False,
                "stream_id": current_stream_id,
                "chosen_offsets": None,
                "candidate_count": int(len(self.current_candidates)),
                "sbr": float(self._current_sbr()),
                "frss_before": float(before_metrics["frss"]),
                "frss_after": float(before_metrics["frss"]),
                "fse_before": float(before_metrics["fse"]),
                "fse_after": float(before_metrics["fse"]),
                "in_defrag_cycle_before": in_defrag_cycle_before,
                "in_defrag_cycle_after": bool(self.in_defrag_cycle),
                "step_count": int(self.step_count),
            }
            if terminated or truncated:
                info.update(self._terminal_info())
            return self._get_obs(), float(reward), terminated, truncated, info

        cand_idx = action - 1
        chosen_candidate = self.current_candidates[cand_idx]
        saved_ports = copy.deepcopy(self.port_schedules)
        saved_offsets = copy.deepcopy(self.stream_offsets)
        applied = self._apply_stream_move(chosen_candidate.stream_id, chosen_candidate.offsets)
        if not applied:
            invalid = True
            self.episode_invalid_moves += 1
            self.port_schedules = saved_ports
            self.stream_offsets = saved_offsets
            self._sync_usecase_schedule()
            self._sync_evaluator_with_schedule_durations()
            reward = -float(self.invalid_penalty)
        else:
            violation_sum = self._violation_sum()
            self.episode_violation_sum += int(violation_sum)
            if violation_sum != 0:
                invalid = True
                self.episode_invalid_moves += 1
                self.port_schedules = saved_ports
                self.stream_offsets = saved_offsets
                self._sync_usecase_schedule()
                self._sync_evaluator_with_schedule_durations()
                reward = -float(self.invalid_penalty)
            else:
                self._backup_current_as_last_valid()
                after_metrics = self._global_metrics()
                delta_accept = after_metrics["acceptance_count"] - before_metrics["acceptance_count"]
                delta_useless = (before_metrics["total_useless_gaps"] - after_metrics["total_useless_gaps"]) / 3
                delta_hole = (after_metrics["largest_hole_ns"] - before_metrics["largest_hole_ns"]) / max(1.0, self.cycle_ns)
                delta_frss = after_metrics["frss"] - before_metrics["frss"]
                reward = np.clip(-5 * delta_frss, -1.0, 1.0) - self.reallocation_penalty
                #reward = 2 * delta_hole - 2 * delta_useless
                if not self.in_defrag_cycle:
                    self.in_defrag_cycle = True
                    self.episode_defrag_cycles += 1
                    reward -= float(self.defrag_cycle_penalty)

        terminated = False
        truncated = bool(self.step_count >= self.max_steps)

        if not invalid and chosen_candidate is not None:
            self._advance_stream_focus()
            self.episode_reallocations += 1
            if self.decision_pointer >= len(self.decision_stream_order):
                terminated, truncated = self._finish_decision_point_and_continue()
            elif not truncated:
                self.current_candidates = self._candidate_list_for_stream(int(self.focus.stream_id))
        elif not truncated:
            self.current_candidates = self._candidate_list_for_stream(int(self.focus.stream_id))

        after_metrics = self._global_metrics()
        info = {
            "stop_action": bool(stop_action),
            "invalid_move": bool(invalid),
            "stream_id": int(current_stream_id),
            "chosen_offsets": None if chosen_candidate is None else [int(x) for x in chosen_candidate.offsets],
            "candidate_count": int(len(self.current_candidates)),
            "sbr": float(self._current_sbr()),
            "frss_before": float(before_metrics["frss"]),
            "frss_after": float(after_metrics["frss"]),
            "fse_before": float(before_metrics["fse"]),
            "fse_after": float(after_metrics["fse"]),
            "largest_hole_before_ns": float(before_metrics["largest_hole_ns"]),
            "largest_hole_after_ns": float(after_metrics["largest_hole_ns"]),
            "useless_gaps_before": float(before_metrics["total_useless_gaps"]),
            "useless_gaps_after": float(after_metrics["total_useless_gaps"]),
            "in_defrag_cycle_before": bool(in_defrag_cycle_before),
            "in_defrag_cycle_after": bool(self.in_defrag_cycle),
            "step_count": int(self.step_count),
            "focus_stream_id": int(self.focus.stream_id) if self.decision_stream_order else -1,
            "decision_points_seen": int(self.episode_decision_points),
        }
        if terminated or truncated:
            info.update(self._terminal_info())


        return self._get_obs(), float(reward), terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}

        if seed is not None:
            self.seed_value = int(seed)
        self.rng_py = random.Random(self.seed_value)
        self.rng_np = np.random.default_rng(self.seed_value)

        self.episode_reallocations = 0
        self.episode_defrag_cycles = 0

        self.usecase = copy.deepcopy(self.base_usecase)
        self.problem.usecase = self.usecase
        self.problem.build_metadata()
        self._rebuild_runtime_metadata()
        loaded = False
        if self.init_from_usecase_schedule:
            loaded = self._create_initial_schedule_from_usecase()
        if not loaded:
            self._create_fallback_initial_schedule()

        self._sync_usecase_schedule()
        self._sync_evaluator_with_schedule_durations()
        if self._violation_sum() > 0:
            raise ValueError("Initial schedule is already invalid.")

        self.network_generator = build_network_generator_from_usecase(self.usecase)


        self.blocked_requests = 0
        self.total_requests = 0
        self.accepted_requests = 0
        self.current_time = 0.0

        self.stream_arrival_times = {int(sid): 0.0 for sid in self._active_stream_ids()}

        self.step_count = 0
        self.in_defrag_cycle = False
        self.decision_active = False
        self.episode_invalid_moves = 0
        self.episode_violation_sum = 0
        self.episode_decision_points = 0
        self.sim_steps_between_decisions = 0

        self.decision_stream_order = []
        self.decision_pointer = 0
        self.focus = StreamDecisionFocus(stream_id=0)
        self.current_candidates = []

        self._backup_current_as_last_valid()
        has_decision = self._advance_until_next_decision_point()
        if not has_decision:
            self.decision_active = False
        self._backup_current_as_last_valid()
        return self._get_obs(), {}

    def render(self):
        return self.get_combined_schedule().plot()


TSNCandidateHopEnv = TSNCandidatePathEnv
