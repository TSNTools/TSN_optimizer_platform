from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from evaluator import ScheduleEncodingType, TSNConstraintEvaluator
from schedule import Schedule, Slot
from tsn_scheduling_problem import MBPS_2_BPNS, TSNScheduling
from usecase_generator import Usecase

BYTE_2_BIT = 8


@dataclass
class StreamHopFocus:
    stream_index: int
    hop_index: int


@dataclass
class HopMoveCandidate:
    stream_id: int
    hop_index: int
    port_id: int
    start_ns: int
    score: float = 0.0
    source: str = "generated"
    debug: Dict[str, object] = field(default_factory=dict)


class TSNCandidateHopEnv(gym.Env):
    """
    Clean candidate-based TSN GCL defragmentation environment.

    Core logic
    ----------
    - Focus one (stream, hop) at a time.
    - Generate candidate start times for the focused hop on its current port.
    - Agent picks a candidate index.
    - Apply the move to that hop only.
    - If schedule becomes invalid -> rollback to last valid schedule.
    - Advance to next hop of same stream; after last hop, advance to next stream.

    Observation
    -----------
    Same reward/state philosophy as env2:
    - same 17-D focus_features vector
    Plus:
    - candidate_offsets: normalized candidate starts for focused hop
    - candidate_mask: fixed-size valid mask for action selection

    Reward
    ------
      reward =
          largest_hole_weight * delta_largest_hole
        + useless_gap_weight * delta_useless_gaps
        - invalid_penalty(if invalid)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        problem: TSNScheduling,
        usecase: Usecase,
        useless_gap_threshold_bytes: int = 1024,
        expected_incoming_bytes: int = 1024,
        invalid_penalty: float = 10_000.0,
        largest_hole_weight: float = 1.0,
        useless_gap_weight: float = 1.0,
        acceptance_weight: float = 1.0,
        max_candidates: int = 10,
        max_steps: int = 60,
        init_from_usecase_schedule: bool = True,
        include_current_start_candidate: bool = True,
    ):
        super().__init__()

        self.problem = problem
        self.usecase = usecase
        self.useless_gap_threshold_bytes = int(useless_gap_threshold_bytes)
        self.expected_incoming_bytes = int(expected_incoming_bytes)
        self.invalid_penalty = float(invalid_penalty)
        self.largest_hole_weight = float(largest_hole_weight)
        self.useless_gap_weight = float(useless_gap_weight)
        self.acceptance_weight = float(acceptance_weight)
        self.max_candidates = max(1, int(max_candidates))
        self.max_steps = max(1, int(max_steps))
        self.init_from_usecase_schedule = bool(init_from_usecase_schedule)
        self.include_current_start_candidate = bool(include_current_start_candidate)

        self.step_count = 0
        self.focus = StreamHopFocus(stream_index=0, hop_index=0)
        self.last_valid_port_schedules: Dict[int, Schedule] = {}
        self.last_valid_stream_offsets: Dict[int, Dict[int, float]] = {}
        self.current_candidates: List[HopMoveCandidate] = []

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

                self.stream_info.append(
                    {
                        "path": path,
                        "interval_ns": interval_ns,
                        "frame_bits": frame_bits,
                        "max_latency_ns": max_lat_ns,
                        "port_dur_ns": port_dur_ns,
                        "total_hop_cost_ns": total_hop_cost_ns,
                        "hop_delays": hop_delays,
                    }
                )

            self.max_latency_ns = max(
                [float(info["max_latency_ns"]) for info in self.stream_info] + [1.0]
            )

            self.sched_eval = TSNConstraintEvaluator(
                self.problem,
                check_routing=False,
                check_task_placement=False,
                scheduling_type=ScheduleEncodingType.START_TIME,
            )

        self.observation_space = spaces.Dict(
            {
                "focus_features": spaces.Box(low=-1.0, high=1.0, shape=(19,), dtype=np.float32),
                "candidate_offsets": spaces.Box(low=-1.0, high=1.0, shape=(self.max_candidates,), dtype=np.float32),
                "candidate_mask": spaces.Box(low=0, high=1, shape=(self.max_candidates,), dtype=np.int8),
            }
        )
        self.action_space = spaces.Discrete(self.max_candidates)

        self.port_schedules: Dict[int, Schedule] = {}
        self.stream_offsets: Dict[int, Dict[int, float]] = {}

    # ------------------------------------------------------------------
    # Schedule loading / backup
    # ------------------------------------------------------------------
    def _create_initial_schedule_from_usecase(self) -> bool:
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
            pid = int(slot.port_id)
            self.stream_offsets.setdefault(s_idx, {})[pid] = float(slot.start_ns)
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

    def _backup_current_as_last_valid(self) -> None:
        self.last_valid_port_schedules = copy.deepcopy(self.port_schedules)
        self.last_valid_stream_offsets = copy.deepcopy(self.stream_offsets)

    def _restore_last_valid(self) -> None:
        self.port_schedules = copy.deepcopy(self.last_valid_port_schedules)
        self.stream_offsets = copy.deepcopy(self.last_valid_stream_offsets)

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

    def _sorted_slots_on_port(self, pidx: int) -> List[Slot]:
        sched = self.port_schedules.get(int(pidx), Schedule(cycle_ns=self.cycle_ns))
        return sorted(list(sched), key=lambda s: (int(s.start_ns), int(s.end_ns), int(s.stream_id)))

    def _active_stream_ids(self) -> List[int]:
        return [s_idx for s_idx, sinfo in enumerate(self.stream_info) if sinfo["path"]]

    def _stream_path(self, s_idx: int) -> List[int]:
        return list(self.stream_info[int(s_idx)]["path"])

    def _current_focus_port_id(self) -> Optional[int]:
        s_idx = int(self.focus.stream_index)
        hop_idx = int(self.focus.hop_index)
        path = self._stream_path(s_idx)
        if not path or hop_idx < 0 or hop_idx >= len(path):
            return None
        return int(path[hop_idx])

    def _current_focus_slot(self) -> Optional[Slot]:
        s_idx = int(self.focus.stream_index)
        pid = self._current_focus_port_id()
        if pid is None:
            return None
        pidx = pid - 1
        for slot in self._sorted_slots_on_port(pidx):
            if int(slot.stream_id) == s_idx and int(slot.port_id) == pid:
                return slot
        return None

    def _advance_focus(self) -> None:
        active_streams = self._active_stream_ids()
        if not active_streams:
            self.focus = StreamHopFocus(stream_index=0, hop_index=0)
            return

        cur_stream = int(self.focus.stream_index)
        cur_hop = int(self.focus.hop_index)

        if cur_stream not in active_streams:
            self.focus = StreamHopFocus(stream_index=active_streams[0], hop_index=0)
            return

        path = self._stream_path(cur_stream)
        if cur_hop + 1 < len(path):
            self.focus = StreamHopFocus(stream_index=cur_stream, hop_index=cur_hop + 1)
            return

        stream_pos = active_streams.index(cur_stream)
        next_stream = active_streams[(stream_pos + 1) % len(active_streams)]
        self.focus = StreamHopFocus(stream_index=next_stream, hop_index=0)

    # ------------------------------------------------------------------
    # Metrics / gaps
    # ------------------------------------------------------------------
    def _useless_gap_threshold_ns_for_port(self, port_id: int) -> float:
        spd = self._port_speed_mbps.get(int(port_id), 1000.0)
        return (self.useless_gap_threshold_bytes * BYTE_2_BIT) / (spd * MBPS_2_BPNS)

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

    def _global_metrics(self) -> Dict[str, float]:
        largest_hole_ns = 0.0
        total_useless_gaps = 0.0
        total_gap_count = 0.0
        total_used_time = 0.0
        acceptance_count = 0.0

        for pidx in range(self.n_ports):
            port_id = pidx + 1
            threshold_ns = self._useless_gap_threshold_ns_for_port(port_id)
            expected_dur_ns = self._expected_incoming_duration_ns_for_port(port_id)
            gaps = self._port_gaps(pidx)

            largest_hole_ns = max(largest_hole_ns, max(gaps) if gaps else 0.0)
            total_useless_gaps += sum(1 for g in gaps if g < threshold_ns)
            total_gap_count += len(gaps)
            total_used_time += float(self.cycle_ns) - float(sum(gaps))

            if expected_dur_ns > 0:
                acceptance_count += sum(int(g // expected_dur_ns) for g in gaps)

        mean_util = total_used_time / max(1.0, float(self.n_ports * self.cycle_ns))
        return {
            "largest_hole_ns": float(largest_hole_ns),
            "total_useless_gaps": float(total_useless_gaps),
            "total_gap_count": float(total_gap_count),
            "mean_utilization": float(np.clip(mean_util, 0.0, 1.0)),
            "acceptance_count": float(acceptance_count),
        }

    def _port_metrics(self, pidx: int) -> Dict[str, float]:
        gaps = self._port_gaps(pidx)
        slots = self._sorted_slots_on_port(pidx)
        port_id = pidx + 1
        threshold_ns = self._useless_gap_threshold_ns_for_port(port_id)
        total_gap = float(sum(gaps))
        util = (float(self.cycle_ns) - total_gap) / float(self.cycle_ns)
        useless_count = float(sum(1 for g in gaps if g < threshold_ns))
        useful_gaps = [g for g in gaps if g >= threshold_ns]
        return {
            "utilization": float(np.clip(util, 0.0, 1.0)),
            "largest_hole_norm": float((max(gaps) if gaps else 0.0) / float(self.cycle_ns)),
            "useless_gap_ratio": float(useless_count / max(1.0, len(gaps))),
            "useless_gap_count_norm": float(min(useless_count, self.n_streams) / max(1.0, self.n_streams)),
            "mean_gap_norm": float(np.mean(gaps) / float(self.cycle_ns)) if gaps else 0.0,
            "mean_useful_gap_norm": float(np.mean(useful_gaps) / float(self.cycle_ns)) if useful_gaps else 0.0,
            "slot_count_norm": float(min(len(slots), self.n_streams) / max(1.0, self.n_streams)),
        }
    
    

    def _current_stream_latency_ns(self, stream_id: int) -> float:
        stream_id = int(stream_id)
        if stream_id < 0 or stream_id >= self.n_streams:
            return 0.0

        offsets = self.stream_offsets.get(stream_id, {})
        path = list(self.stream_info[stream_id]["path"])
        if not path:
            return 0.0

        starts: List[float] = []
        ends: List[float] = []
        for pid in path:
            pid = int(pid)
            if pid not in offsets:
                continue
            st = float(offsets[pid])
            dur = float(self.stream_info[stream_id]["port_dur_ns"].get(pid, 0.0))
            starts.append(st)
            ends.append(st + dur)

        if not starts or not ends:
            return 0.0
        return max(ends) - min(starts)

    def _expected_incoming_duration_ns_for_port(self, port_id: int) -> float:
        spd = self._port_speed_mbps.get(int(port_id), 1000.0)
        return (self.expected_incoming_bytes * BYTE_2_BIT) / (spd * MBPS_2_BPNS)


    def _port_acceptance_count(self, pidx: int) -> float:
        port_id = pidx + 1
        expected_dur_ns = self._expected_incoming_duration_ns_for_port(port_id)
        if expected_dur_ns <= 0:
            return 0.0

        gaps = self._port_gaps(pidx)
        return float(sum(int(g // expected_dur_ns) for g in gaps))

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------
    def _occupied_intervals_on_port_excluding_focus(self, port_id: int, stream_id: int) -> List[Tuple[int, int]]:
        pidx = int(port_id) - 1
        intervals: List[Tuple[int, int]] = []
        for slot in self._sorted_slots_on_port(pidx):
            if int(slot.stream_id) == int(stream_id) and int(slot.port_id) == int(port_id):
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

    def _stream_current_hop_start_ns(self, stream_id: int, hop_index: int) -> int:
        path = self._stream_path(stream_id)
        if hop_index < 0 or hop_index >= len(path):
            return 0
        pid = int(path[hop_index])
        return int(round(float(self.stream_offsets.get(int(stream_id), {}).get(pid, 0.0))))

    def _stream_hop_duration_ns(self, stream_id: int, hop_index: int) -> int:
        path = self._stream_path(stream_id)
        if hop_index < 0 or hop_index >= len(path):
            return 0
        pid = int(path[hop_index])
        dur = float(self.stream_info[int(stream_id)]["port_dur_ns"].get(pid, 0.0))
        return int(np.ceil(dur))

    def _candidate_start_set_for_focus(self) -> List[int]:
        s_idx = int(self.focus.stream_index)
        hop_idx = int(self.focus.hop_index)
        path = self._stream_path(s_idx)
        if hop_idx < 0 or hop_idx >= len(path):
            return [0]

        port_id = int(path[hop_idx])
        dur = int(self._stream_hop_duration_ns(s_idx, hop_idx))
        cur = int(self._stream_current_hop_start_ns(s_idx, hop_idx))

        if dur <= 0:
            return [cur]

        free_gaps = self._free_gaps_on_port_excluding_focus(port_id, s_idx)

        starts = set()

        for gap_start, gap_end in free_gaps:
            gap_len = int(gap_end - gap_start)
            if gap_len < dur:
                continue

            start = int(gap_start)
            while start + dur <= int(gap_end):
                starts.add(start)
                start += dur

        if self.include_current_start_candidate:
            starts.add(cur)

        if not starts:
            starts.add(cur)

        return sorted(starts)

    def _free_gaps_on_port_excluding_focus(self, port_id: int, stream_id: int) -> List[Tuple[int, int]]:
        occupied = self._occupied_intervals_on_port_excluding_focus(port_id, stream_id)

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
    
    def _sync_evaluator_with_schedule_durations(self) -> None:
        # reset all widths
        self.sched_eval.windLengths[:, :] = 0.0

        # copy actual durations from the real schedule object
        for pidx, sched in self.port_schedules.items():
            for slot in sched:
                if slot.stream_id is None or slot.port_id is None:
                    continue

                port_idx = int(slot.port_id) - 1
                stream_id = int(slot.stream_id)
                actual_dur = float(slot.end_ns - slot.start_ns)

                self.sched_eval.windLengths[port_idx, stream_id] = actual_dur
                self.stream_info[stream_id]["port_dur_ns"][int(slot.port_id)] = actual_dur

        # recompute pairwise overlap widths used by evaluator
        self.sched_eval.stream_pairwise_windowSum = (
            self.sched_eval.windLengths[:, :, None] + self.sched_eval.windLengths[:, None, :]
        )

    def _apply_hop_move(self, stream_id: int, hop_index: int, new_start_ns: int) -> bool:
        stream_id = int(stream_id)
        hop_index = int(hop_index)
        path = self._stream_path(stream_id)
        if hop_index < 0 or hop_index >= len(path):
            return False

        port_id = int(path[hop_index])
        pidx = port_id - 1

        # locate current slot for this stream/hop
        old_slot: Optional[Slot] = None
        for slot in self._sorted_slots_on_port(pidx):
            if int(slot.stream_id) == stream_id and int(slot.port_id) == port_id:
                old_slot = slot
                break
        if old_slot is None:
            return False

        dur = int(old_slot.end_ns) - int(old_slot.start_ns)
        st = max(0, min(int(self.cycle_ns) - dur, int(new_start_ns)))
        en = st + dur

        new_slot = Slot(
            start_ns=int(st),
            end_ns=int(en),
            period_ns=int(old_slot.period_ns),
            queue_id=int(old_slot.queue_id) if old_slot.queue_id is not None else 0,
            stream_id=int(stream_id),
            port_id=int(port_id),
        )

        updated_slots: List[Slot] = []
        replaced = False
        for existing in self.port_schedules.get(pidx, Schedule(cycle_ns=self.cycle_ns)):
            same = (
                int(existing.stream_id) == int(stream_id)
                and int(existing.port_id) == int(port_id)
                and int(existing.start_ns) == int(old_slot.start_ns)
                and int(existing.end_ns) == int(old_slot.end_ns)
            )
            if same and not replaced:
                updated_slots.append(new_slot)
                replaced = True
            else:
                updated_slots.append(existing)

        if not replaced:
            return False

        self.port_schedules[pidx] = Schedule(cycle_ns=self.cycle_ns, iterable=updated_slots)
        self.stream_offsets.setdefault(stream_id, {})[port_id] = float(st)
        self._sync_evaluator_with_schedule_durations()

        return True

    def _candidate_list_for_focus(self) -> List[HopMoveCandidate]:
        s_idx = int(self.focus.stream_index)
        hop_idx = int(self.focus.hop_index)
        path = self._stream_path(s_idx)
        if hop_idx < 0 or hop_idx >= len(path):
            return []

        port_id = int(path[hop_idx])
        start_set = self._candidate_start_set_for_focus()

        valid_candidates: List[HopMoveCandidate] = []
        for st in start_set:
            saved_ports = copy.deepcopy(self.port_schedules)
            saved_offsets = copy.deepcopy(self.stream_offsets)

            applied = self._apply_hop_move(s_idx, hop_idx, int(st))
            valid = False
            if applied:
                valid = (self._violation_sum() == 0)

            self.port_schedules = saved_ports
            self.stream_offsets = saved_offsets

            if valid:
                valid_candidates.append(
                    HopMoveCandidate(
                        stream_id=s_idx,
                        hop_index=hop_idx,
                        port_id=port_id,
                        start_ns=int(st),
                        source="generated",
                    )
                )

        # Fallback: current start should be safe if reset state is valid,
        # but we still verify it explicitly.
        if not valid_candidates:
            cur = int(self._stream_current_hop_start_ns(s_idx, hop_idx))

            saved_ports = copy.deepcopy(self.port_schedules)
            saved_offsets = copy.deepcopy(self.stream_offsets)

            applied = self._apply_hop_move(s_idx, hop_idx, cur)
            valid = False
            if applied:
                valid = (self._violation_sum() == 0)

            self.port_schedules = saved_ports
            self.stream_offsets = saved_offsets

            if valid:
                valid_candidates.append(
                    HopMoveCandidate(
                        stream_id=s_idx,
                        hop_index=hop_idx,
                        port_id=port_id,
                        start_ns=cur,
                        source="fallback_current",
                    )
                )

        # If still empty, something is badly inconsistent.
        if not valid_candidates:
            raise RuntimeError(
                f"No valid candidates found for stream={s_idx}, hop={hop_idx}, port={port_id}."
            )

        # Keep deterministic order and truncate first
        valid_candidates.sort(key=lambda c: c.start_ns)
        valid_candidates = valid_candidates[: self.max_candidates]

        # Cyclic Padding.
        orig = valid_candidates.copy()
        i = 0
        while len(valid_candidates) < self.max_candidates:
            base = orig[i % len(orig)]
            valid_candidates.append(
                HopMoveCandidate(
                    stream_id=base.stream_id,
                    hop_index=base.hop_index,
                    port_id=base.port_id,
                    start_ns=base.start_ns,
                    source="padded_cycle",
                )
            )
            i += 1

        return valid_candidates

    def _expected_incoming_duration_ns_for_port(self, port_id: int) -> float:
        spd = self._port_speed_mbps.get(int(port_id), 1000.0)
        return (self.expected_incoming_bytes * BYTE_2_BIT) / (spd * MBPS_2_BPNS) + 4000.0
    
    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _focus_slot_context(self) -> Dict[str, float]:
        slot = self._current_focus_slot()
        if slot is None:
            return {
                "port_index": 0.0,
                "stream_id": 0.0,
                "start_ns": 0.0,
                "end_ns": 0.0,
                "duration_ns": 0.0,
                "prev_gap_ns": 0.0,
                "next_gap_ns": 0.0,
                "prev_duration_ns": 0.0,
                "next_duration_ns": 0.0,
                "rank_norm": 0.0,
                "hop_index_norm": 0.0,
            }

        pidx = int(slot.port_id) - 1
        slots = self._sorted_slots_on_port(pidx)
        idx = 0
        for i, s in enumerate(slots):
            if (
                int(s.stream_id) == int(slot.stream_id)
                and int(s.port_id) == int(slot.port_id)
                and int(s.start_ns) == int(slot.start_ns)
            ):
                idx = i
                break

        prev_slot = slots[idx - 1] if idx > 0 else None
        next_slot = slots[idx + 1] if idx + 1 < len(slots) else None

        prev_gap = float(slot.start_ns - prev_slot.end_ns) if prev_slot is not None else float(slot.start_ns)
        next_gap = float(next_slot.start_ns - slot.end_ns) if next_slot is not None else float(self.cycle_ns - slot.end_ns)

        stream_id = int(slot.stream_id)
        path = self._stream_path(stream_id)
        hop_index = path.index(int(slot.port_id)) if int(slot.port_id) in path else 0
        hop_index_norm = 0.0 if len(path) <= 1 else hop_index / float(len(path) - 1)

        return {
            "port_index": float(pidx),
            "stream_id": float(stream_id),
            "start_ns": float(slot.start_ns),
            "end_ns": float(slot.end_ns),
            "duration_ns": float(slot.end_ns - slot.start_ns),
            "prev_gap_ns": max(0.0, prev_gap),
            "next_gap_ns": max(0.0, next_gap),
            "prev_duration_ns": float(prev_slot.end_ns - prev_slot.start_ns) if prev_slot is not None else 0.0,
            "next_duration_ns": float(next_slot.end_ns - next_slot.start_ns) if next_slot is not None else 0.0,
            "rank_norm": 0.0 if len(slots) <= 1 else idx / float(len(slots) - 1),
            "hop_index_norm": float(hop_index_norm),
        }

    def _get_obs(self) -> Dict[str, np.ndarray]:
        ctx = self._focus_slot_context()
        g = self._global_metrics()
        p = self._port_metrics(int(ctx["port_index"]))
        
        port_acceptance = self._port_acceptance_count(int(ctx["port_index"]))
        global_acceptance_norm = float(
            min(g["acceptance_count"], self.n_ports * self.n_streams)
            / max(1.0, self.n_ports * self.n_streams)
        )
        port_acceptance_norm = float(
            min(port_acceptance, self.n_streams) / max(1.0, self.n_streams)
        )

        stream_id = int(ctx["stream_id"])
        sinfo = self.stream_info[stream_id] if 0 <= stream_id < self.n_streams else {
            "interval_ns": self.cycle_ns,
            "max_latency_ns": self.cycle_ns,
            "path": [],
        }

        current_latency_ns = self._current_stream_latency_ns(stream_id)
        global_useless_norm = float(
            min(g["total_useless_gaps"], self.n_ports * self.n_streams)
            / max(1.0, self.n_ports * self.n_streams)
        )

        focus_features = np.asarray(
            [
                (ctx["start_ns"] / self.cycle_ns) * 2.0 - 1.0,
                (ctx["end_ns"] / self.cycle_ns) * 2.0 - 1.0,
                (ctx["duration_ns"] / self.cycle_ns) * 2.0 - 1.0,
                (ctx["prev_gap_ns"] / self.cycle_ns) * 2.0 - 1.0,
                (ctx["next_gap_ns"] / self.cycle_ns) * 2.0 - 1.0,
                (ctx["prev_duration_ns"] / self.cycle_ns) * 2.0 - 1.0,
                (ctx["next_duration_ns"] / self.cycle_ns) * 2.0 - 1.0,
                (ctx["port_index"] / max(1.0, self.n_ports - 1)) * 2.0 - 1.0,
                ctx["rank_norm"] * 2.0 - 1.0,
                (float(sinfo["interval_ns"]) / self.cycle_ns) * 2.0 - 1.0,
                (float(sinfo["max_latency_ns"]) / self.max_latency_ns) * 2.0 - 1.0,
                float(ctx["hop_index_norm"]) * 2.0 - 1.0,
                (current_latency_ns / self.max_latency_ns) * 2.0 - 1.0,
                p["largest_hole_norm"] * 2.0 - 1.0,
                (g["largest_hole_ns"] / self.cycle_ns) * 2.0 - 1.0,
                p["useless_gap_count_norm"] * 2.0 - 1.0,
                global_useless_norm * 2.0 - 1.0,
                port_acceptance_norm * 2.0 - 1.0,
                global_acceptance_norm * 2.0 - 1.0,
            ],
            dtype=np.float32,
        )

        candidate_offsets = np.full((self.max_candidates,), -1.0, dtype=np.float32)
        candidate_mask = np.zeros((self.max_candidates,), dtype=np.int8)
        for i, cand in enumerate(self.current_candidates[: self.max_candidates]):
            candidate_offsets[i] = (float(cand.start_ns) / float(self.cycle_ns)) * 2.0 - 1.0
            candidate_mask[i] = 1

        return {
            "focus_features": focus_features,
            "candidate_offsets": candidate_offsets,
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

    def step(self, action: int):
        self.step_count += 1
        old_metrics = self._global_metrics()
        old_focus = StreamHopFocus(self.focus.stream_index, self.focus.hop_index)

        action = int(np.asarray(action).item())
        invalid = False
        chosen_start_ns = None

        if action < 0 or action >= self.max_candidates:
            invalid = True
            self._restore_last_valid()
        elif action >= len(self.current_candidates):
            invalid = True
            self._restore_last_valid()
        else:
            cand = self.current_candidates[action]
            chosen_start_ns = int(cand.start_ns)
            applied = self._apply_hop_move(cand.stream_id, cand.hop_index, cand.start_ns)
            if not applied:
                invalid = True
                self._restore_last_valid()
            else:
                violation_sum = self._violation_sum()
                if violation_sum > 0:
                    invalid = True
                    self._restore_last_valid()
                else:
                    self._backup_current_as_last_valid()

        new_metrics = self._global_metrics()

        delta_largest_hole = (new_metrics["largest_hole_ns"] - old_metrics["largest_hole_ns"]) / float(self.cycle_ns)
        delta_useless_gaps = ( old_metrics["total_useless_gaps"] - new_metrics["total_useless_gaps"] ) / max(1.0, self.n_ports * self.n_streams)
        delta_acceptance = ( new_metrics["acceptance_count"] - old_metrics["acceptance_count"] ) / max(1.0, self.n_ports * self.n_streams)


        reward = (
            self.largest_hole_weight * float(delta_largest_hole)
            + self.useless_gap_weight * float(delta_useless_gaps)
            + self.acceptance_weight * float(delta_acceptance)
            - (self.invalid_penalty if invalid else 0.0)
        )

        self.focus = old_focus
        self._advance_focus()
        self.current_candidates = self._candidate_list_for_focus()

        terminated = False
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "invalid_move": bool(invalid),
            "chosen_start_ns": chosen_start_ns,
            "delta_largest_hole": float(delta_largest_hole),
            "delta_useless_gaps": float(delta_useless_gaps),
            "largest_hole_before_ns": float(old_metrics["largest_hole_ns"]),
            "largest_hole_after_ns": float(new_metrics["largest_hole_ns"]),
            "useless_gaps_before": float(old_metrics["total_useless_gaps"]),
            "useless_gaps_after": float(new_metrics["total_useless_gaps"]),
            "step_count": int(self.step_count),
            "focus_stream_index": int(self.focus.stream_index),
            "focus_hop_index": int(self.focus.hop_index),
            "candidate_count": int(len(self.current_candidates)),
            "acceptance_before": float(old_metrics["acceptance_count"]),
            "acceptance_after": float(new_metrics["acceptance_count"]),
            "delta_acceptance": float(delta_acceptance),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        loaded = False
        if self.init_from_usecase_schedule:
            loaded = self._create_initial_schedule_from_usecase()
        if not loaded:
            self._create_fallback_initial_schedule()

        self._sync_evaluator_with_schedule_durations()
        if self._violation_sum() > 0:
            raise ValueError(
                "Initial schedule is already invalid. This env assumes reset() starts from a valid schedule."
            )
        
        self.step_count = 0
        active_streams = self._active_stream_ids()
        self.focus = StreamHopFocus(stream_index=active_streams[0] if active_streams else 0, hop_index=0)
        self._backup_current_as_last_valid()
        self.current_candidates = self._candidate_list_for_focus()
        return self._get_obs(), {}

    def render(self):
        return self.get_combined_schedule().plot()
