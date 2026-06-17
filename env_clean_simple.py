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
    E_after: float = 0.0   
    source: str = "generated"
    debug: Dict[str, object] = field(default_factory=dict)


class TSNCandidatePathEnv(gym.Env):
    """
    Event-driven TSN GCL defragmentation env.

    Semantics
    ---------
    - The schedule evolves in trace-style decision steps:
        * arrivals ~ Poisson(lambda_arrival)
        * each decision step removes exactly one active stream before defrag
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
        useless_gap_threshold_bytes: int = 512,
        expected_incoming_bytes: int = 1024,
        invalid_penalty: float = 50.0,
        max_candidates: int = 10,
        max_steps: int = 30,
        max_hops: Optional[int] = None,
        init_from_usecase_schedule: bool = True,
        include_current_offset_candidate: bool = False,
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
        reward_alpha: float = 0.7,
        min_reallocations_per_cycle: int = 5,
        early_stop_penalty: float = 0.10,
        stream_cooldown_decisions: int = 3,
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

        self.reward_alpha = float(reward_alpha)

        self.min_reallocations_per_cycle = int(min_reallocations_per_cycle)
        self.early_stop_penalty = float(early_stop_penalty)
        self.stream_cooldown_decisions = int(stream_cooldown_decisions)

        self.current_cycle_reallocations = 0
        self.stream_last_moved_decision: Dict[int, int] = {}

        self.per_hop_candidate_limit = max(1, int(per_hop_candidate_limit))
        self.beam_width = int(beam_width) if beam_width is not None else max(self.max_candidates * 2, 8)

        self.defrag_cycle_penalty = float(defrag_cycle_penalty)
        self.reallocation_penalty = float(reallocation_penalty)

        self.lambda_arrival = float(lambda_arrival)
        # Kept for backward-compatible CLI/configs. The current trace-style
        # simulator forces one departure per decision step instead of sampling
        # from mu_departure.
        self.mu_departure = float(mu_departure)
        self.dt = float(dt)
        self.decision_points_per_episode = int(decision_points_per_episode)
        self.ed_ed_settings = ed_ed_settings
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
        self.pending_arrivals_after_decision = 0

        self.episode_requested_departures = 0
        self.episode_executed_departures = 0
        self.episode_skipped_departures = 0

        self.decision_stream_order: List[int] = []
        self.decision_pointer = 0
        self.port_intervals: Dict[int, List[Tuple[int, int]]] = {}
        self.network_generator = None

        self.episode_sum_E_reward = 0.0
        self.episode_sum_frss_reward = 0.0
        self.episode_sum_raw_move_reward = 0.0
        self.episode_sum_stop_reward = 0.0
        self.episode_sum_realloc_penalty = 0.0
        self.episode_sum_cycle_penalty = 0.0
        self.episode_sum_delta_E = 0.0
        self.episode_sum_delta_frss = 0.0

        self.episode_sum_scaled_move_reward = 0.0
        self.episode_terminal_sbr_reward = 0.0

        self.episode_move_actions = 0
        self.episode_stop_actions = 0
        self.episode_forced_stops = 0

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

        # Compact observation for DQN.
        # base_features: current schedule state.
        # candidate_features: per-action candidate quality.
        self.observation_space = spaces.Dict(
            {
                "base_features": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
                "candidate_features": spaces.Box(
                    low=-1.0, high=1.0, shape=(self.max_candidates, 3), dtype=np.float32
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


    # ------------------------------------------------------------------
    # Schedule utilities
    # ------------------------------------------------------------------
    
    def _right_fit_offsets_for_stream(self, stream_id: int) -> Optional[List[int]]:
        """
        Build one true right-fit offset vector for an existing stream.

        This works backwards:
        - place the last hop as late as possible
        - then place previous hops as late as possible while preserving causality

        It only returns offsets. It does not modify the schedule.
        """
        stream_id = int(stream_id)
        path = self._stream_path(stream_id)

        if not path:
            return None

        period = int(self.cycle_ns)
        hop_delays = list(self.stream_info[stream_id]["hop_delays"])

        def merge_intervals(intervals):
            cleaned = []
            for a, b in intervals:
                a = max(0, int(a))
                b = min(period, int(b))
                if b > a:
                    cleaned.append((a, b))

            if not cleaned:
                return []

            cleaned.sort()
            merged = [cleaned[0]]

            for a, b in cleaned[1:]:
                la, lb = merged[-1]
                if a <= lb:
                    merged[-1] = (la, max(lb, b))
                else:
                    merged.append((a, b))

            return merged

        def fits(start, duration, intervals):
            start = int(start)
            end = int(start + duration)

            if start < 0 or end > period:
                return False

            for a, b in intervals:
                if start < b and end > a:
                    return False

            return True

        def rightmost_start_before(latest_end, duration, intervals):
            latest_end = min(period, int(latest_end))
            duration = int(duration)

            if duration <= 0:
                return 0

            intervals = merge_intervals(intervals)

            candidates = set()
            candidates.add(latest_end - duration)

            # Just before every occupied interval is a meaningful right-fit candidate.
            for a, _ in intervals:
                candidates.add(int(a) - duration)

            for s in sorted(candidates, reverse=True):
                if s < 0:
                    continue
                if s + duration > latest_end:
                    continue
                if fits(s, duration, intervals):
                    return int(s)

            return None

        offsets = [None for _ in path]

        # Backward placement
        for hop_idx in reversed(range(len(path))):
            pid = int(path[hop_idx])
            duration = int(self._stream_hop_duration_ns(stream_id, hop_idx))

            if hop_idx == len(path) - 1:
                latest_end = period
            else:
                # Need this hop to finish early enough for the next hop.
                # Relation used in your forward candidate generation:
                # next_start >= this_start + hop_delay
                next_start = int(offsets[hop_idx + 1])
                delay_to_next = float(hop_delays[hop_idx])
                latest_start = int(round(next_start - delay_to_next))
                latest_end = latest_start + duration

            if latest_end <= 0:
                return None

            intervals = self._occupied_intervals_on_port_excluding_stream(
                port_id=pid,
                stream_id=stream_id,
            )

            st = rightmost_start_before(
                latest_end=latest_end,
                duration=duration,
                intervals=intervals,
            )

            if st is None:
                return None

            offsets[hop_idx] = int(st)

        if any(x is None for x in offsets):
            return None

        return [int(x) for x in offsets]



    def select_diverse_candidates(
        self,
        candidates: List[StreamMoveCandidate],
        max_candidates: int,
    ) -> List[StreamMoveCandidate]:
        """Return a small, interpretable candidate set.

        The benchmark inserts new arrivals with first-fit. Therefore we keep
        later/right-side placements visible, but we do not let them destroy the
        action list. The final mix is:
        - best quality candidates by (delta_E, delta_FRSS),
        - a few right-side candidates,
        - one middle/diverse candidate if there is room.
        """
        if not candidates:
            return []

        max_candidates = int(max_candidates)
        if max_candidates <= 0:
            return []

        def key(c: StreamMoveCandidate):
            return tuple(int(x) for x in c.offsets)

        unique: List[StreamMoveCandidate] = []
        seen = set()
        for c in candidates:
            k = key(c)
            if k not in seen:
                seen.add(k)
                unique.append(c)

        if len(unique) <= max_candidates:
            return unique

        def delta_E(c: StreamMoveCandidate) -> float:
            return float(c.debug.get("delta_E", 0.0))

        def delta_frss(c: StreamMoveCandidate) -> float:
            return float(c.debug.get("delta_frss", 0.0))

        def mean_offset(c: StreamMoveCandidate) -> float:
            if not c.offsets:
                return 0.0
            return float(sum(int(x) for x in c.offsets)) / float(len(c.offsets))

        selected: List[StreamMoveCandidate] = []
        selected_keys = set()

        def add(c: StreamMoveCandidate) -> None:
            k = key(c)
            if k not in selected_keys and len(selected) < max_candidates:
                selected.append(c)
                selected_keys.add(k)

        # 1) Quality first: this keeps the reward/action semantics stable.
        quality_sorted = sorted(
            unique,
            key=lambda c: (
                -delta_E(c),
                -delta_frss(c),
                int(c.cuts),
                tuple(int(x) for x in c.offsets),
            ),
        )
        quality_quota = max(1, int(round(0.5 * max_candidates)))
        for c in quality_sorted:
            add(c)
            if len(selected) >= quality_quota:
                break

        # 2) Add right-side alternatives because arrivals use first-fit.
        right_sorted = sorted(
            unique,
            key=lambda c: (
                -mean_offset(c),
                -delta_E(c),
                -delta_frss(c),
                int(c.cuts),
            ),
        )
        right_quota = max(1, int(round(0.3 * max_candidates)))
        right_added = 0
        for c in right_sorted:
            before = len(selected)
            add(c)
            if len(selected) > before:
                right_added += 1
            if right_added >= right_quota:
                break

        # 3) Fill with diverse candidates by first-hop position.
        by_first = sorted(unique, key=lambda c: int(c.offsets[0]) if c.offsets else 0)
        for c in (by_first[-1], by_first[len(by_first) // 2], by_first[0]):
            add(c)

        for c in quality_sorted:
            add(c)
            if len(selected) >= max_candidates:
                break

        return selected[:max_candidates]


    def select_diverse_offset_vectors(
        self,
        vectors: List[List[int]],
        limit: int,
        current_offsets: Optional[List[int]] = None,
    ) -> List[List[int]]:
        """
        Diversity-aware pruning for partial/full offset vectors.

        Keeps different first-hop placements instead of only the lowest-span vectors.
        """
        if not vectors:
            return []

        limit = int(limit)
        if limit <= 0:
            return []

        # Remove duplicates
        unique = []
        seen = set()
        for vec in vectors:
            key = tuple(int(x) for x in vec)
            if key not in seen:
                seen.add(key)
                unique.append([int(x) for x in vec])

        if len(unique) <= limit:
            return unique

        selected = []
        selected_keys = set()

        def add(vec):
            key = tuple(int(x) for x in vec)
            if key not in selected_keys and len(selected) < limit:
                selected.append([int(x) for x in vec])
                selected_keys.add(key)

        # 1) Keep current prefix if available
        if current_offsets:
            prefix_len = len(unique[0])
            cur_prefix = [int(x) for x in current_offsets[:prefix_len]]

            for vec in unique:
                if vec == cur_prefix:
                    add(vec)
                    break

        # 2) Keep left-most, right-most, and middle by first-hop offset
        by_first = sorted(unique, key=lambda vec: int(vec[0]) if vec else 0)

        add(by_first[-1])                        # right
        add(by_first[len(by_first) // 2])        # middle
        add(by_first[0])                         # left

        # 3) Keep low-span alternatives
        by_span = sorted(
            unique,
            key=lambda vec: (
                max(vec) - min(vec) if vec else 0,
                sum(vec),
                tuple(vec),
            ),
        )

        for vec in by_span:
            add(vec)
            if len(selected) >= limit:
                break

        return selected[:limit]
    
    
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

        ordered_all = sorted(
            active,
            key=lambda sid: (
                self.stream_arrival_times.get(int(sid), 0.0),
                int(sid),
            ),
        )

        current_dp = int(self.episode_decision_points)
        cooldown = int(getattr(self, "stream_cooldown_decisions", 0))

        eligible = []
        cooled = []

        for sid in ordered_all:
            sid = int(sid)
            last_moved = self.stream_last_moved_decision.get(sid, -10**9)

            if cooldown > 0 and (current_dp - int(last_moved)) < cooldown:
                cooled.append(sid)
            else:
                eligible.append(sid)

        # Prefer non-cooled streams, but if all are cooled, fall back to all.
        ordered = eligible if eligible else ordered_all

        k = min(self.top_k_oldest_random, len(ordered))
        top_k = ordered[:k]

        first = int(self.rng_py.choice(top_k))
        remaining = [sid for sid in ordered if int(sid) != first]

        # Recently moved streams are still available, but only after eligible ones.
        remaining_set = set(int(sid) for sid in remaining)
        cooled_tail = [
            int(sid)
            for sid in cooled
            if int(sid) != first and int(sid) not in remaining_set
        ]

        return [first] + remaining + cooled_tail

 

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
    
    def _expected_duration_ns_for_port_and_size(self, port_id: int, packet_size_bytes: float,) -> float:
        """
        Expected transmission duration on a port for a future stream profile.
        Includes the same fixed margin previously used in
        _expected_incoming_duration_ns_for_port().
        """
        spd = self._port_speed_mbps.get(int(port_id), 1000.0)
        return (float(packet_size_bytes) * BYTE_2_BIT) / (spd * MBPS_2_BPNS) + 4000.0

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

    
    def _fragmentation_rss2(self) -> float:
        per_port_vals: List[float] = []
        for pidx in range(self.n_ports):
            gaps = self._port_gaps(pidx)
            free_sum = float(sum(gaps))
            if free_sum <= 0.0:
                per_port_vals.append(0.0)
                continue
            rss_e = float(np.sqrt(np.sum(np.square(np.asarray(gaps, dtype=np.float64)))) / free_sum)
            per_port_vals.append(rss_e)
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
        total_useful_gaps = 0.0

        for pidx in range(self.n_ports):
            port_id = pidx + 1
            threshold_ns = self._useless_gap_threshold_ns_for_port(port_id)
            gaps = self._port_gaps(pidx)

            largest_hole_ns = max(largest_hole_ns, max(gaps) if gaps else 0.0)
            total_useless_gaps += sum(1 for g in gaps if g < threshold_ns)
            total_useful_gaps += sum(1 for g in gaps if g >= threshold_ns)
            total_gap_count += len(gaps)
            total_used_time += float(self.cycle_ns) - float(sum(gaps))


        mean_util = total_used_time / max(1.0, float(self.n_ports * self.cycle_ns))
        return {
            "largest_hole_ns": float(largest_hole_ns),
            "total_useless_gaps": float(total_useless_gaps),
            "total_useful_gaps": float(total_useful_gaps),
            "total_gap_count": float(total_gap_count),
            "mean_utilization": float(np.clip(mean_util, 0.0, 1.0)),
            "E": float(self._embeddability()),
            "frss": float(self._fragmentation_rss2()),
            "fse": float(self._fragmentation_entropy()),
        }

    def _merge_intervals(self, intervals):
        cleaned = []

        for a, b in intervals:
            a = max(0.0, float(a))
            b = min(float(self.cycle_ns), float(b))

            if b > a:
                cleaned.append((a, b))

        if not cleaned:
            return []

        cleaned.sort(key=lambda x: x[0])
        merged = [cleaned[0]]

        for a, b in cleaned[1:]:
            last_a, last_b = merged[-1]

            if a <= last_b:
                merged[-1] = (last_a, max(last_b, b))
            else:
                merged.append((a, b))

        return merged

    def _intersect_interval_sets(self, A, B):
        A = self._merge_intervals(A)
        B = self._merge_intervals(B)

        i = 0
        j = 0
        out = []

        while i < len(A) and j < len(B):
            a1, a2 = A[i]
            b1, b2 = B[j]

            left = max(a1, b1)
            right = min(a2, b2)

            if right > left:
                out.append((left, right))

            if a2 < b2:
                i += 1
            else:
                j += 1

        return out


    def _future_stream_profiles(self) -> List[Dict[str, float]]:
        settings = self.ed_ed_settings or {}

        packet_sizes = settings.get("packet_sizes", [self.expected_incoming_bytes])
        periods = settings.get("periods", [self.cycle_ns])

        profiles: List[Dict[str, float]] = []

        for packet_size in packet_sizes:
            for period in periods:
                profiles.append(
                    {
                        "packet_size_bytes": float(packet_size),
                        "period_ns": float(period),
                        "latency_factor": 1.0,
                        "jitter_factor": 0.0,
                    }
                )

        return profiles


    def _future_path_templates(self):
        """
        Return unique path templates used to estimate future embeddability.

        First version:
        - use paths already present in stream_info
        - remove duplicates
        - ignore empty paths

        Later, we can replace this with paths sampled from the network generator.
        """
        paths = []
        seen = set()

        for info in self.stream_info:
            path = tuple(int(p) for p in info.get("path", []))

            if not path:
                continue

            if path not in seen:
                seen.add(path)
                paths.append(list(path))

        return paths


    def _port_embeddability_capacity(
        self,
        port_id: int,
        packet_size_bytes: float,
        period_ns: float,
    ) -> float:
        """
        Fast gap-based embeddability capacity for one port.

        It approximates how many placements a future stream of the given
        size could still have on this port.

        Higher means this port has more useful free space for future arrivals.
        """
        port_id = int(port_id)
        period_ns = max(1.0, float(period_ns))

        duration = float(
            self._expected_duration_ns_for_port_and_size(
                port_id=port_id,
                packet_size_bytes=float(packet_size_bytes),
            )
        )

        if duration <= 0.0:
            return 0.0

        gaps = self._port_gaps(port_id - 1)

        # Gap-local placement count:
        # A gap of size g can contain roughly (g - duration) possible starts.
        # Divide by duration to obtain an approximate number of stream-sized placements.
        capacity = 0.0
        for gap in gaps:
            gap = float(gap)
            capacity += max(0.0, gap - duration) / max(1.0, duration)

        # Shorter periods repeat more often inside the macro-cycle,
        # so they consume more schedule capacity.
        repetition_factor = max(1.0, float(self.cycle_ns) / period_ns)
        capacity /= repetition_factor

        return float(max(0.0, capacity))


    def _path_embeddability_capacity(
        self,
        path: List[int],
        profile: Dict[str, float],
    ) -> float:
        """
        Bottleneck embeddability along a full path.

        The future stream must fit on every port of the path.
        Therefore, the path score is the minimum port capacity.
        """
        path = [int(p) for p in path]

        if not path:
            return 0.0

        packet_size_bytes = float(profile.get("packet_size_bytes", self.expected_incoming_bytes))
        period_ns = float(profile.get("period_ns", self.cycle_ns))

        port_scores = []

        for port_id in path:
            cap = self._port_embeddability_capacity(
                port_id=int(port_id),
                packet_size_bytes=packet_size_bytes,
                period_ns=period_ns,
            )
            port_scores.append(float(cap))

        if not port_scores:
            return 0.0

        return float(min(port_scores))


    def _embeddability(self) -> float:
        """
        E: fast path-aware embeddability score in [0, 1].

        It estimates how many future one-way arrivals can still be embedded
        using the current free gaps.

        Higher E means the schedule is more friendly to future arrivals.
        """
        paths = self._future_path_templates()
        profiles = self._future_stream_profiles()

        if not paths or not profiles:
            return 0.0

        scores: List[float] = []

        for path in paths:
            for profile in profiles:
                cap = self._path_embeddability_capacity(path, profile)


                score = float(cap) / (float(cap) + 20.0)
                scores.append(float(score))

        if not scores:
            return 0.0

        return float(np.clip(np.mean(scores), 0.0, 1.0))



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
        """
        Generate diverse candidate starts for one stream hop.

        Instead of only taking the earliest starts in each free gap, sample:
        - left edge
        - right edge
        - middle
        - current offset, if feasible
        - evenly spaced internal points if per_hop_candidate_limit allows

        This avoids the old left-biased behavior where almost all candidates
        were near 0 in large empty gaps.
        """
        path = self._stream_path(stream_id)
        if hop_index < 0 or hop_index >= len(path):
            return []

        port_id = int(path[hop_index])
        dur = int(self._stream_hop_duration_ns(stream_id, hop_index))
        min_start_ns = max(0, int(min_start_ns))

        current_offsets = self._stream_current_offset_vector(stream_id)
        cur = int(current_offsets[hop_index]) if hop_index < len(current_offsets) else 0

        if dur <= 0:
            return [min_start_ns]

        free_gaps = self._free_gaps_on_port_excluding_stream(port_id, stream_id)

        starts = set()

        for gap_start, gap_end in free_gaps:
            lo = max(int(gap_start), min_start_ns)
            hi = int(gap_end) - dur

            if hi < lo:
                continue

            # Always include left, middle, and right of the feasible region.
            starts.add(int(lo))
            starts.add(int((lo + hi) // 2))
            starts.add(int(hi))

            # Add extra evenly spaced points if requested.
            # Example with extra=2: add approx 1/3 and 2/3 positions.
            extra = max(0, int(self.per_hop_candidate_limit) - 3)
            for i in range(1, extra + 1):
                s = lo + int(round(i * (hi - lo) / (extra + 1)))
                starts.add(int(s))

        # Include current offset if it is feasible after excluding this stream.
        if self.include_current_offset_candidate and cur >= min_start_ns:
            for gap_start, gap_end in free_gaps:
                lo = max(int(gap_start), min_start_ns)
                hi = int(gap_end) - dur
                if lo <= cur <= hi:
                    starts.add(int(cur))
                    break

        return self._right_biased_subset(starts, self.per_hop_candidate_limit)


    def _right_biased_subset(self, starts, limit: int) -> List[int]:
        """Keep a tiny diverse set of starts, biased toward later placements."""
        starts = sorted({int(x) for x in starts})
        if not starts:
            return []

        limit = max(1, int(limit))
        if len(starts) <= limit:
            return starts

        selected: List[int] = []

        def add(x: int) -> None:
            x = int(x)
            if x not in selected and len(selected) < limit:
                selected.append(x)

        # Prefer right and right-middle starts, but keep one middle/left anchor
        # so the model still has contrast and does not lose all quality choices.
        add(starts[-1])
        if len(starts) >= 2:
            add(starts[-2])
        add(starts[len(starts) // 2])

        for x in reversed(starts):
            add(x)
            if len(selected) >= limit - 1:
                break

        # One left candidate only if there is still room.
        add(starts[0])
        return selected[:limit]


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
        return {
            "frss_after": float(g["frss"]),
            "fse_after": float(g["fse"]),
            "E_after": float(g["E"]),
        }

    def _candidate_list_for_stream(self, stream_id: int) -> List[StreamMoveCandidate]:
        """
        Generate final DRL candidates for one active stream.

        Improvements over old version:
        - Generates candidates with richer hop starts from _candidate_start_set_for_stream_hop().
        - Excludes the current/no-move candidate.
            Action 0 already represents STOP / do not reconfigure.
            Therefore, actions 1..K are real reallocation candidates only.
        - Selects diverse remaining candidates, so we avoid exposing five very
            similar left-biased candidates.
        """
        stream_id = int(stream_id)

        path = self._stream_path(stream_id)
        if not path:
            return []

        hop_delays = list(self.stream_info[stream_id]["hop_delays"])
        current_offsets = self._stream_current_offset_vector(stream_id)

        # ------------------------------------------------------------
        # Build full-path candidate offset vectors hop by hop
        # ------------------------------------------------------------
        partials: List[List[int]] = [[]]

        for hop_idx in range(len(path)):
            next_partials: List[List[int]] = []

            for partial in partials:
                if hop_idx == 0:
                    min_start_ns = 0
                else:
                    prev_start = int(partial[hop_idx - 1])
                    prev_delay = float(hop_delays[hop_idx - 1])
                    min_start_ns = int(round(prev_start + prev_delay))

                hop_candidates = self._candidate_start_set_for_stream_hop(
                    stream_id,
                    hop_idx,
                    min_start_ns,
                )

                # _candidate_start_set_for_stream_hop already returns a bounded
                # right-biased diverse set, so late placements are not lost.
                for st in hop_candidates:
                    next_partials.append(partial + [int(st)])

            if not next_partials:
                # Fallback to current offsets if possible.
                if current_offsets and len(current_offsets) == len(path):
                    next_partials = [list(current_offsets)]
                else:
                    return []

            # Beam pruning. Prefer fewer cuts, then shorter path span.
            next_partials = self.select_diverse_offset_vectors(
                vectors=next_partials,
                limit=self.beam_width,
                current_offsets=current_offsets,
            )

            partials = next_partials

    
        right_fit_offsets = self._right_fit_offsets_for_stream(stream_id)

        if right_fit_offsets is not None and len(right_fit_offsets) == len(path):
            partials.append(list(right_fit_offsets))

        # ------------------------------------------------------------
        # Remove duplicate offset vectors
        # ------------------------------------------------------------
        unique_vecs: List[List[int]] = []
        seen = set()

        for vec in partials:
            key = tuple(int(x) for x in vec)

            if key not in seen:
                seen.add(key)
                unique_vecs.append([int(x) for x in vec])

        # ------------------------------------------------------------
        # Evaluate candidates and keep only globally valid moves
        # ------------------------------------------------------------
        candidates: List[StreamMoveCandidate] = []

        current_key = (
            tuple(int(x) for x in current_offsets)
            if current_offsets and len(current_offsets) == len(path)
            else None
        )

        right_fit_key = (
            tuple(int(x) for x in right_fit_offsets)
            if right_fit_offsets is not None and len(right_fit_offsets) == len(path)
            else None
        )

        for vec in unique_vecs:

            vec_key = tuple(int(x) for x in vec)

            if current_key is not None and vec_key == current_key:
                continue

            eval_metrics = self._evaluate_candidate_metrics(stream_id, vec)

            if eval_metrics is None:
                continue

            current_metrics = self._global_metrics()
            delta_E = float(eval_metrics["E_after"]) - float(current_metrics["E"])
            delta_frss = float(eval_metrics["frss_after"]) - float(current_metrics["frss"])

            cuts = self._stream_cut_count(stream_id, offsets=vec)


            if right_fit_key is not None and vec_key == right_fit_key:
                source = "right_fit"
            else:
                source = "generated"

            candidates.append(
                StreamMoveCandidate(
                    stream_id=int(stream_id),
                    offsets=[int(x) for x in vec],
                    cuts=int(cuts),
                    frss_after=float(eval_metrics["frss_after"]),
                    fse_after=float(eval_metrics["fse_after"]),
                    E_after=float(eval_metrics["E_after"]),
                    source=source,
                    debug={
                        "delta_E": float(delta_E),
                        "delta_frss": float(delta_frss),
                    },
                )
            )

        if not candidates:
            return []

        # Stable semantic:
        # action 0 = STOP
        # action 1..K = real reallocation candidates only.
        # Current/no-move vectors were already removed above.
        right_fit_candidates = [c for c in candidates if c.source == "right_fit"]
        other_candidates = [c for c in candidates if c.source != "right_fit"]

        def delta_E(c: StreamMoveCandidate) -> float:
            return float(c.debug.get("delta_E", 0.0))

        def delta_frss(c: StreamMoveCandidate) -> float:
            return float(c.debug.get("delta_frss", 0.0))

        def mean_offset(c: StreamMoveCandidate) -> float:
            if not c.offsets:
                return 0.0
            return float(sum(int(x) for x in c.offsets)) / float(len(c.offsets))

        quality_sorted = sorted(
            other_candidates,
            key=lambda c: (
                -delta_E(c),
                -delta_frss(c),
                int(c.cuts),
                tuple(int(x) for x in c.offsets),
            ),
        )

        right_sorted = sorted(
            other_candidates,
            key=lambda c: (
                -mean_offset(c),
                -delta_E(c),
                -delta_frss(c),
                int(c.cuts),
            ),
        )

        right_fit_candidates.sort(
            key=lambda c: (
                -delta_E(c),
                -delta_frss(c),
                int(c.cuts),
                tuple(int(x) for x in c.offsets),
            )
        )

        final_candidates: List[StreamMoveCandidate] = []
        selected_keys = set()

        def add_candidate(c: Optional[StreamMoveCandidate]) -> None:
            if c is None:
                return
            k = tuple(int(x) for x in c.offsets)
            if k not in selected_keys and len(final_candidates) < self.max_candidates:
                final_candidates.append(c)
                selected_keys.add(k)

        # 1) Always expose one true right-fit candidate if available.
        if right_fit_candidates:
            add_candidate(right_fit_candidates[0])

        # 2) Most choices remain quality-ranked: this avoids the aggressive
        # right-heavy version that hurt benchmark SBR.
        quality_quota = max(1, int(round(0.5 * self.max_candidates)))
        for c in quality_sorted:
            add_candidate(c)
            if len(final_candidates) >= quality_quota:
                break

        # 3) Add a few right-side choices for first-fit-friendly compaction.
        right_quota = max(1, int(round(0.3 * self.max_candidates)))
        right_added = 0
        for c in right_sorted:
            before = len(final_candidates)
            add_candidate(c)
            if len(final_candidates) > before:
                right_added += 1
            if right_added >= right_quota:
                break

        # 4) Fill the rest with diverse/quality candidates.
        remaining = [
            c for c in other_candidates
            if tuple(int(x) for x in c.offsets) not in selected_keys
        ]
        for c in self.select_diverse_candidates(remaining, self.max_candidates - len(final_candidates)):
            add_candidate(c)

        return final_candidates[: self.max_candidates]
    

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

    def _process_n_arrivals(self, n_arrivals: int) -> None:
        n_arrivals = int(n_arrivals)
        if n_arrivals <= 0:
            return

        self._sync_usecase_schedule()
        sched = self.usecase.schedule

        for _ in range(n_arrivals):
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

            self.usecase.streams.append(copy.deepcopy(stream))
            self.problem.usecase = self.usecase
            self.problem.build_metadata()

            result = sched.place_existing_stream_first_fit_fast(
                stream_id=stream_id,
                usecase=self.usecase,
                problem=self.problem,
                queue_id=5,
                port_intervals=self.port_intervals,
                max_candidates_per_hop=2,
                beam_width=2,
            )

            if not result.get("accepted"):
                self.usecase.streams.pop()
                self.problem.usecase = self.usecase
                self.problem.build_metadata()

            self._rebuild_runtime_metadata()

            if result.get("accepted", False):
                self.accepted_requests += 1
                self.stream_arrival_times[int(stream_id)] = float(self.current_time)
            else:
                self.blocked_requests += 1

            self._load_state_from_schedule(sched)

    def _process_arrivals(self) -> None:
        # Trace-style arrival step: sample the number of arrivals once,
        # then process exactly that many arrivals.
        self._process_n_arrivals(self._sample_arrivals())


    def _advance_until_next_decision_point(self) -> bool:
        """
        Fixed decision-step simulator.

        This matches benchmark-style evaluation:
        - decision_points_per_episode is the fixed number of trace/decision steps.
        - Each step samples arrivals once.
        - Each step attempts one departure.
        - If a departure is executed, the agent gets a decision point.
        - If no active stream exists, arrivals are still processed and the step still counts.
        """

        while self.episode_decision_points < self.decision_points_per_episode:
            self.current_time += self.dt
            self.sim_steps_between_decisions += 1

            # Count this benchmark-style decision step immediately.
            self.episode_decision_points += 1
            self.episode_requested_departures += 1

            self._sync_usecase_schedule()
            sched = self.usecase.schedule
            active_stream_ids = self._active_stream_ids()

            # Sample arrivals for this fixed step.
            # They are applied after the defrag decision, like benchmark replay.
            self.pending_arrivals_after_decision = self._sample_arrivals()

            # No active stream: departure request is skipped, but the step still counts.
            if len(active_stream_ids) <= 0:
                self.episode_skipped_departures += 1
                self._process_n_arrivals(self.pending_arrivals_after_decision)
                self.pending_arrivals_after_decision = 0
                continue

            # Exactly one departure attempt per fixed decision step.
            before_ids = list(active_stream_ids)

            sched.delete_streams(
                n_remove=1,
                usecase=self.usecase,
                rng=self.rng_py,
                port_intervals=self.port_intervals,
                stream_arrival_times=self.stream_arrival_times,
            )

            self._load_state_from_schedule(sched)

            after_ids = self._active_stream_ids()
            removed_ids = self._removed_stream_ids(before_ids, after_ids)

            if removed_ids:
                self.episode_executed_departures += len(removed_ids)

            for sid in removed_ids:
                self.stream_arrival_times.pop(int(sid), None)

            # Open DRL decision after the departure.
            if self._begin_decision_point():
                return True

            # If no decision can be opened, still replay arrivals.
            self._process_n_arrivals(self.pending_arrivals_after_decision)
            self.pending_arrivals_after_decision = 0

        self.decision_active = False
        self.current_candidates = []
        return False


    def _begin_decision_point(self) -> bool:
        self.decision_stream_order = self._ordered_streams_by_age()
        self.decision_pointer = 0
        self.in_defrag_cycle = False
        self.current_cycle_reallocations = 0

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
    def _norm_signed(self, value: float, scale: float) -> float:
        """Normalize a signed delta to [-1, 1] using a fixed meaningful scale."""
        return float(np.clip(float(value) / max(float(scale), 1e-12), -1.0, 1.0))

    def _norm_unit(self, value: float) -> float:
        """Map a [0, 1] quantity to [-1, 1]."""
        return float(np.clip(float(value), 0.0, 1.0) * 2.0 - 1.0)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Small observation: global state + candidate deltas.

        base_features = [SBR, FRSS, E, in_cycle, candidate_count]
        candidate_features = [delta_E, delta_FRSS, cuts]
        """
        g = self._global_metrics()
        before_E = float(g["E"])
        before_frss = float(g["frss"])

        base_features = np.asarray(
            [
                self._norm_unit(self._current_sbr()),
                self._norm_unit(before_frss),
                self._norm_unit(before_E),
                1.0 if self.in_defrag_cycle else -1.0,
                self._norm_unit(len(self.current_candidates) / max(1, self.max_candidates)),
            ],
            dtype=np.float32,
        )

        candidate_features = np.full((self.max_candidates, 3), -1.0, dtype=np.float32)
        for i, cand in enumerate(self.current_candidates[: self.max_candidates]):
            delta_E = float(cand.E_after) - before_E
            delta_frss = float(cand.frss_after) - before_frss
            cuts_norm = float(cand.cuts / max(1, self.max_hops - 1))
            candidate_features[i] = np.asarray(
                [
                    self._norm_signed(delta_E, 0.005),
                    self._norm_signed(delta_frss, 0.05),
                    self._norm_unit(cuts_norm),
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

        # Trace-style timing: replay arrivals only after the agent stops
        # or after the current defrag cycle is exhausted.
        self._process_n_arrivals(self.pending_arrivals_after_decision)
        self.pending_arrivals_after_decision = 0

        has_next = self._advance_until_next_decision_point()
        terminated = not has_next
        truncated = bool(self.step_count >= self.max_steps)
        return terminated, truncated


    def _reward_diagnostics_info(self) -> Dict[str, float]:
        move_count = max(1, int(self.episode_move_actions))

        return {
            "sum_E_reward": float(self.episode_sum_E_reward),
            "sum_frss_reward": float(self.episode_sum_frss_reward),
            "sum_raw_move_reward": float(self.episode_sum_raw_move_reward),
            "sum_stop_reward": float(self.episode_sum_stop_reward),
            "sum_realloc_penalty": float(self.episode_sum_realloc_penalty),
            "sum_cycle_penalty": float(self.episode_sum_cycle_penalty),

            "avg_delta_E": float(self.episode_sum_delta_E / move_count),
            "avg_delta_frss": float(self.episode_sum_delta_frss / move_count),

            "move_actions": int(self.episode_move_actions),
            "stop_actions": int(self.episode_stop_actions),
            "forced_stops": int(self.episode_forced_stops),
            "sum_scaled_move_reward": float(self.episode_sum_scaled_move_reward),
            "terminal_sbr_reward": float(self.episode_terminal_sbr_reward),
        }

    def _terminal_info(self) -> Dict[str, float]:
        sbr = self._current_sbr()

        info = {
            "episode_sbr": float(sbr),
            "episode_acceptance": float(self.accepted_requests / max(1, self.total_requests)),
            "episode_arrivals": int(self.total_requests),
            "episode_rejections": int(self.blocked_requests),
            "episode_decision_points": int(self.episode_decision_points),
            "episode_invalid_moves": int(self.episode_invalid_moves),
            "episode_violation_sum": int(self.episode_violation_sum),
            "episode_reallocations": int(self.episode_reallocations),
            "episode_defrag_cycles": int(self.episode_defrag_cycles),
            "episode_requested_departures": int(self.episode_requested_departures),
            "episode_executed_departures": int(self.episode_executed_departures),
            "episode_skipped_departures": int(self.episode_skipped_departures),
            "episode_trace_steps": int(self.sim_steps_between_decisions),
        }

        info.update(self._reward_diagnostics_info())
        return info



    def step(self, action: int):
        self.step_count += 1
        action = int(np.asarray(action).item())

        if not self.decision_active:
            info = {
                "stop_action": True,
                "forced_to_stop": False,
                "invalid_move": False,
                **self._terminal_info(),
            }
            return self._get_obs(), 0.0, True, False, info

        before_metrics = self._global_metrics()
        current_stream_id = int(self.focus.stream_id)
        in_cycle_before = bool(self.in_defrag_cycle)

        # DQN does not consume the mask directly. If it selects a padded action,
        # end the decision point like a forced STOP, but give a clear penalty.
        if action > 0 and action - 1 >= len(self.current_candidates):
            self.episode_forced_stops += 1
            self.episode_invalid_moves += 1
            reward = -float(self.invalid_penalty)

            terminated, truncated = self._finish_decision_point_and_continue()
            info = {
                "stop_action": True,
                "forced_to_stop": True,
                "invalid_move": True,
                "stream_id": current_stream_id,
                "reward": float(reward),
                "candidate_count": int(len(self.current_candidates)),
            }
            if terminated or truncated:
                info.update(self._terminal_info())
            return self._get_obs(), float(reward), terminated, truncated, info

        # STOP: explicit admission feedback. This is intentionally simple:
        # choosing to stop receives the current acceptance signal.
        if action == 0:
            self._backup_current_as_last_valid()
            self.episode_stop_actions += 1

            current_sbr = self._current_sbr()
            if not self.in_defrag_cycle:
                reward = -1 * (1 - current_sbr)
            else:
                reward = -1 * (1 - current_sbr)
            self.episode_sum_stop_reward += float(reward)

            terminated, truncated = self._finish_decision_point_and_continue()
            info = {
                "stop_action": True,
                "forced_to_stop": False,
                "invalid_move": False,
                "sbr": float(current_sbr),
                "reward": float(reward),
                "candidate_count": int(len(self.current_candidates)),
                "cycle_reallocations": int(self.current_cycle_reallocations),
            }
            if terminated or truncated:
                self.episode_terminal_sbr_reward = 0.0
                info.update(self._terminal_info())
                info["terminal_sbr_reward"] = 0.0
            return self._get_obs(), float(reward), terminated, truncated, info

        # MOVE: apply one candidate. First move of a cycle pays cycle penalty;
        # continuation moves only pay reallocation penalty.
        chosen_candidate = self.current_candidates[action - 1]
        saved_ports = copy.deepcopy(self.port_schedules)
        saved_offsets = copy.deepcopy(self.stream_offsets)

        invalid = False
        applied = self._apply_stream_move(chosen_candidate.stream_id, chosen_candidate.offsets)
        if not applied:
            invalid = True
        else:
            violation_sum = self._violation_sum()
            self.episode_violation_sum += int(violation_sum)
            invalid = bool(violation_sum != 0)

        move_after_metrics = before_metrics
        delta_E = 0.0
        delta_frss = 0.0
        E_reward = 0.0
        frss_reward = 0.0
        raw_move_reward = 0.0
        realloc_penalty_applied = 0.0
        cycle_penalty_applied = 0.0

        if invalid:
            self.episode_invalid_moves += 1
            self.port_schedules = saved_ports
            self.stream_offsets = saved_offsets
            self._sync_usecase_schedule()
            self._sync_evaluator_with_schedule_durations()
            reward = -float(self.invalid_penalty)
        else:
            self._backup_current_as_last_valid()
            move_after_metrics = self._global_metrics()

            delta_E = float(move_after_metrics["E"] - before_metrics["E"])
            delta_frss = float(move_after_metrics["frss"] - before_metrics["frss"])

            E_reward = self._norm_signed(delta_E, 0.003)
            frss_reward = self._norm_signed(delta_frss, 0.02)
            alpha = float(self.reward_alpha)

            raw_move_reward = alpha * E_reward + (1.0 - alpha) * frss_reward
            reward = float(raw_move_reward)*3

            realloc_penalty_applied = float(self.reallocation_penalty)
            reward -= realloc_penalty_applied

            if not self.in_defrag_cycle:
                self.in_defrag_cycle = True
                self.episode_defrag_cycles += 1
                cycle_penalty_applied = float(self.defrag_cycle_penalty)
                reward -= cycle_penalty_applied

            self.episode_sum_scaled_move_reward += float(reward)
            self.episode_move_actions += 1
            self.episode_sum_E_reward += float(E_reward)
            self.episode_sum_frss_reward += float(frss_reward)
            self.episode_sum_raw_move_reward += float(raw_move_reward)
            self.episode_sum_realloc_penalty += float(realloc_penalty_applied)
            self.episode_sum_cycle_penalty += float(cycle_penalty_applied)
            self.episode_sum_delta_E += float(delta_E)
            self.episode_sum_delta_frss += float(delta_frss)

        terminated = False
        truncated = bool(self.step_count >= self.max_steps)

        if not invalid:
            self._advance_stream_focus()
            self.episode_reallocations += 1
            self.current_cycle_reallocations += 1
            self.stream_last_moved_decision[int(chosen_candidate.stream_id)] = int(self.episode_decision_points)

            if self.decision_pointer >= len(self.decision_stream_order):
                terminated, truncated = self._finish_decision_point_and_continue()
            elif not truncated:
                self.current_candidates = self._candidate_list_for_stream(int(self.focus.stream_id))
        elif not truncated:
            self.current_candidates = self._candidate_list_for_stream(int(self.focus.stream_id))

        after_metrics = self._global_metrics()
        info = {
            "stop_action": False,
            "forced_to_stop": False,
            "invalid_move": bool(invalid),
            "stream_id": current_stream_id,
            "chosen_offsets": [int(x) for x in chosen_candidate.offsets],
            "chosen_candidate_source": str(chosen_candidate.source),
            "reward": float(reward),
            "raw_move_reward": float(raw_move_reward),
            "E_before": float(before_metrics["E"]),
            "E_after": float(move_after_metrics["E"]),
            "delta_E": float(delta_E),
            "E_reward": float(E_reward),
            "before_frss": float(before_metrics["frss"]),
            "after_frss": float(move_after_metrics["frss"]),
            "delta_frss": float(delta_frss),
            "frss_reward": float(frss_reward),
            "realloc_penalty": float(realloc_penalty_applied),
            "cycle_penalty": float(cycle_penalty_applied),
            "in_defrag_cycle_before": bool(in_cycle_before),
            "in_defrag_cycle_after": bool(self.in_defrag_cycle),
            "cycle_reallocations": int(self.current_cycle_reallocations),
            "candidate_count": int(len(self.current_candidates)),
            "after_frss_global": float(after_metrics["frss"]),
            "after_E_global": float(after_metrics["E"]),
        }

        if terminated or truncated:
            self.episode_terminal_sbr_reward = 0.0
            info.update(self._terminal_info())
            info["terminal_sbr_reward"] = 0.0

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
        self.current_cycle_reallocations = 0
        self.stream_last_moved_decision = {}
        self.pending_arrivals_after_decision = 0

        self.episode_sum_E_reward = 0.0
        self.episode_sum_frss_reward = 0.0
        self.episode_sum_raw_move_reward = 0.0
        self.episode_sum_stop_reward = 0.0
        self.episode_sum_realloc_penalty = 0.0
        self.episode_sum_cycle_penalty = 0.0
        self.episode_sum_delta_E = 0.0
        self.episode_sum_delta_frss = 0.0

        self.episode_move_actions = 0
        self.episode_stop_actions = 0
        self.episode_forced_stops = 0

        self.episode_sum_scaled_move_reward = 0.0
        self.episode_terminal_sbr_reward = 0.0
        

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

        self.episode_requested_departures = 0
        self.episode_executed_departures = 0
        self.episode_skipped_departures = 0

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

