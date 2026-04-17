from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from evaluator import ScheduleEncodingType, TSNConstraintEvaluator
from schedule import Schedule, Slot
from tsn_scheduling_problem import MBPS_2_BPNS, TSNScheduling
from usecase_generator import Usecase

BYTE_2_BIT = 8
MICROSECOND_NS = 1_000


@dataclass
class FocusRef:
    port_index: int  # 0-based
    slot_index: int  # index inside chronological order of that port


class TSNDefragStepEnv(gym.Env):
    """
    TSN GCL defragmentation environment for step-by-step local moves.

    Main idea
    ---------
    - The environment focuses on one scheduled hop at a time.
    - The agent can move that hop by exactly -1 us or +1 us.
    - If the move creates an invalid schedule, the environment gives a very
      large negative reward and rolls back to the last valid schedule.
    - Then it advances to the next hop in chronological order on that port,
      then to the next port, and wraps around.

    Action space
    ------------
    Discrete(3)
        0 -> move backward by 1 us
        1 -> keep current position
        2 -> move forward by 1 us


    Observation space
    -----------------
    Dict with one compact vector:
        - focus_features: local slot context + latency + local/global fragmentation signals

    Reward
    ------
      reward = delta_largest_hole + delta_useless_gaps - invalid_penalty

    where:
      delta_largest_hole = (new_global_largest_hole_ns - old_global_largest_hole_ns) / cycle_ns
      delta_useless_gaps = old_total_useless_gaps - new_total_useless_gaps
      invalid_penalty = 10000 if attempted schedule is invalid else 0
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        problem: TSNScheduling,
        usecase: Usecase,
        step_ns: int = MICROSECOND_NS,
        useless_gap_threshold_bytes: int = 1024,
        invalid_penalty: float = 10_000.0,
        expected_incoming_bytes: int = 1024,
        largest_hole_weight: float = 1.0,
        useless_gap_weight: float = 1.0,
        max_steps: int = 2_000_000,
        init_from_usecase_schedule: bool = True,
        reset_mode: str = "never",
        freeze_reset_after_episode: int = 100_000,
        acceptance_weight: float = 1.0,

    ):
        super().__init__()
        self.problem = problem
        self.usecase = usecase
        self.step_ns = int(step_ns)
        self.useless_gap_threshold_bytes = int(useless_gap_threshold_bytes)
        self.invalid_penalty = float(invalid_penalty)
        self.largest_hole_weight = float(largest_hole_weight)
        self.useless_gap_weight = float(useless_gap_weight)
        self.max_steps = int(max_steps)
        self.init_from_usecase_schedule = bool(init_from_usecase_schedule)

        self.expected_incoming_bytes = int(expected_incoming_bytes)
        self.acceptance_weight = float(acceptance_weight)

        self.reset_mode = str(reset_mode)
        self.freeze_reset_after_episode = int(freeze_reset_after_episode)

        if self.reset_mode not in {"always", "never", "freeze_after"}:
            raise ValueError(
                f"Unsupported reset_mode={self.reset_mode}. "
                "Use one of: 'always', 'never', 'freeze_after'."
    )

        self.step_count = 0
        self.focus = FocusRef(port_index=0, slot_index=0)
        self.last_valid_port_schedules: Dict[int, Schedule] = {}
        self.last_valid_stream_offsets: Dict[int, Dict[int, float]] = {}

        self.episode_count = 0
        self.has_built_initial_state = False

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

        # Simple, informative dict observation for DQN MultiInputPolicy.
        self.observation_space = spaces.Dict(
            {
                "focus_features": spaces.Box(low=-1.0, high=1.0, shape=(19,), dtype=np.float32),
            }
        )
        
        self.action_deltas_us = [-5, -1, 0, 1, 5]
        self.action_space = spaces.Discrete(len(self.action_deltas_us))

        self.port_schedules: Dict[int, Schedule] = {}
        self.stream_offsets: Dict[int, Dict[int, float]] = {}

    # ------------------------------------------------------------------
    # Schedule loading and bookkeeping
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

    def _derive_path_offsets_from_first_start(self, s_idx: int, first_offset_ns: float) -> Dict[int, float]:
        sinfo = self.stream_info[s_idx]
        path = list(sinfo["path"])
        offsets: Dict[int, float] = {}
        cum_delay = 0.0
        for k, pid in enumerate(path):
            dur = float(sinfo["port_dur_ns"].get(pid, 0.0))
            max_start = max(0.0, float(self.cycle_ns) - dur)
            offset = max(0.0, min(max_start, float(first_offset_ns) + cum_delay))
            offsets[int(pid)] = offset
            if k < len(sinfo["hop_delays"]):
                cum_delay += float(sinfo["hop_delays"][k])
        return offsets

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
            dur = float(sinfo["port_dur_ns"].get(pid, 0.0))
            st = float(offsets.get(int(pid), 0.0))
            st = max(0.0, min(float(self.cycle_ns) - dur, st))
            en = st + dur
            slot = Slot(
                start_ns=int(round(st)),
                end_ns=int(round(en)),
                period_ns=int(sinfo["interval_ns"]),
                queue_id=0,
                stream_id=int(s_idx),
                port_id=int(pid),
            )
            pidx = int(pid) - 1
            self.port_schedules.setdefault(pidx, Schedule(cycle_ns=self.cycle_ns)).append(slot)
            self.stream_offsets[s_idx][int(pid)] = float(st)

    def _create_fallback_initial_schedule(self) -> None:
        self.port_schedules = {}
        self.stream_offsets = {}
        for s_idx in range(self.n_streams):
            offsets = self._derive_path_offsets_from_first_start(s_idx, 0.0)
            self._place_stream_with_offsets(s_idx, offsets)

    def _backup_current_as_last_valid(self) -> None:
        self.last_valid_port_schedules = copy.deepcopy(self.port_schedules)
        self.last_valid_stream_offsets = copy.deepcopy(self.stream_offsets)

    def _expected_incoming_duration_ns_for_port(self, port_id: int) -> float:
        spd = self._port_speed_mbps.get(int(port_id), 1000.0)
        return (self.expected_incoming_bytes * BYTE_2_BIT) / (spd * MBPS_2_BPNS) + 4000.0

    def _restore_last_valid(self) -> None:
        self.port_schedules = copy.deepcopy(self.last_valid_port_schedules)
        self.stream_offsets = copy.deepcopy(self.last_valid_stream_offsets)

    # ------------------------------------------------------------------
    # Low-level schedule inspection
    # ------------------------------------------------------------------
    def _sorted_slots_on_port(self, pidx: int) -> List[Slot]:
        sched = self.port_schedules.get(int(pidx), Schedule(cycle_ns=self.cycle_ns))
        return sorted(list(sched), key=lambda s: (int(s.start_ns), int(s.end_ns), int(s.stream_id)))

    def _active_ports(self) -> List[int]:
        return [pidx for pidx in range(self.n_ports) if len(self._sorted_slots_on_port(pidx)) > 0]

    def _current_focus_slot(self) -> Optional[Slot]:
        ports = self._active_ports()
        if not ports:
            return None

        pidx = int(np.clip(self.focus.port_index, 0, self.n_ports - 1))
        if pidx not in ports:
            pidx = ports[0]
            self.focus.port_index = pidx
            self.focus.slot_index = 0

        slots = self._sorted_slots_on_port(pidx)
        if not slots:
            return None

        sidx = int(np.clip(self.focus.slot_index, 0, len(slots) - 1))
        self.focus.slot_index = sidx
        return slots[sidx]

    def _advance_focus(self) -> None:
        ports = self._active_ports()
        if not ports:
            self.focus = FocusRef(port_index=0, slot_index=0)
            return

        current_port = self.focus.port_index
        if current_port not in ports:
            self.focus = FocusRef(port_index=ports[0], slot_index=0)
            return

        current_slots = self._sorted_slots_on_port(current_port)
        if self.focus.slot_index + 1 < len(current_slots):
            self.focus = FocusRef(port_index=current_port, slot_index=self.focus.slot_index + 1)
            return

        port_pos = ports.index(current_port)
        next_port = ports[(port_pos + 1) % len(ports)]
        self.focus = FocusRef(port_index=next_port, slot_index=0)

    def _focus_slot_context(self) -> Dict[str, float]:
        slot = self._current_focus_slot()
        if slot is None:
            return {
                "port_index": 0,
                "stream_id": 0,
                "start_ns": 0.0,
                "end_ns": 0.0,
                "duration_ns": 0.0,
                "prev_gap_ns": 0.0,
                "next_gap_ns": 0.0,
                "prev_duration_ns": 0.0,
                "next_duration_ns": 0.0,
                "rank_norm": 0.0,
                "port_slot_count": 0.0,
                "hop_index_norm": 0.0,
            }

        pidx = self.focus.port_index
        slots = self._sorted_slots_on_port(pidx)
        idx = self.focus.slot_index
        prev_slot = slots[idx - 1] if idx > 0 else None
        next_slot = slots[idx + 1] if idx + 1 < len(slots) else None

        prev_gap = float(slot.start_ns - prev_slot.end_ns) if prev_slot is not None else float(slot.start_ns)
        next_gap = float(next_slot.start_ns - slot.end_ns) if next_slot is not None else float(self.cycle_ns - slot.end_ns)
        stream_id = int(slot.stream_id)
        path = list(self.stream_info[stream_id]["path"])
        hop_index = 0
        if int(slot.port_id) in path:
            hop_index = path.index(int(slot.port_id))
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
            "port_slot_count": float(len(slots)),
            "hop_index_norm": float(hop_index_norm),
        }

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

        starts = []
        ends = []

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


    def _port_acceptance_count(self, pidx: int) -> float:
        port_id = pidx + 1
        expected_dur_ns = self._expected_incoming_duration_ns_for_port(port_id)
        if expected_dur_ns <= 0:
            return 0.0

        gaps = self._port_gaps(pidx)
        return float(sum(int(g // expected_dur_ns) for g in gaps))

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
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
            "total_hop_cost_ns": 0.0,
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

        return {
            "focus_features": focus_features,
        }

    # ------------------------------------------------------------------
    # Validity / reward
    # ------------------------------------------------------------------
    def _current_compact(self) -> np.ndarray:
        return self.get_combined_schedule().to_compact_encoding(self.n_ports, self.n_streams)

    def _violation_sum(self) -> int:
        compact = self._current_compact()
        return int(self.sched_eval.per_stream_all_violations(compact).sum())

    def normalize_min_max(self, n, min_orig, max_orig):
        if max_orig == min_orig:
            raise ValueError("max_orig and min_orig cannot be equal.")
        
        return (n - min_orig) / (max_orig - min_orig)

    def _move_current_focus_slot(self, delta_ns: int) -> bool:
        slot = self._current_focus_slot()
        if slot is None:
            return False

        old_start = int(slot.start_ns)
        old_end = int(slot.end_ns)
        new_start = old_start + int(delta_ns)
        new_end = old_end + int(delta_ns)

        if new_start < 0 or new_end > self.cycle_ns:
            return False

        pidx = self.focus.port_index
        pid = int(slot.port_id)
        sid = int(slot.stream_id)

        new_slot = Slot(
            start_ns=int(new_start),
            end_ns=int(new_end),
            period_ns=int(slot.period_ns),
            queue_id=int(slot.queue_id) if slot.queue_id is not None else 0,
            stream_id=sid,
            port_id=pid,
        )

        updated_slots: List[Slot] = []
        replaced = False
        for existing in self.port_schedules.get(pidx, Schedule(cycle_ns=self.cycle_ns)):
            same = (
                int(existing.stream_id) == sid
                and int(existing.port_id) == pid
                and int(existing.start_ns) == old_start
                and int(existing.end_ns) == old_end
            )
            if same and not replaced:
                updated_slots.append(new_slot)
                replaced = True
            else:
                updated_slots.append(existing)

        if not replaced:
            return False

        self.port_schedules[pidx] = Schedule(cycle_ns=self.cycle_ns, iterable=updated_slots)
        self.stream_offsets.setdefault(sid, {})[pid] = float(new_start)
        return True

    def step(self, action: int):
        if action < 0 or action >= len(self.action_deltas_us):
            raise ValueError(f"Invalid action: {action}")

        delta_ns = int(self.action_deltas_us[action]) * self.step_ns
        self.step_count += 1
        old_metrics = self._global_metrics()
        old_focus = FocusRef(self.focus.port_index, self.focus.slot_index)

        local_move_applied = self._move_current_focus_slot(delta_ns)
        invalid = False

        if not local_move_applied:
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
        #largest_hole = self.normalize_min_max(new_metrics["largest_hole_ns"], min_orig=0, max_orig=self.cycle_ns)
        #useless_gaps = self.normalize_min_max(new_metrics["total_useless_gaps"], min_orig=0, max_orig=self.cycle_ns*self.n_ports//(self.useless_gap_threshold_bytes*8+4000))
        #acceptance = self.normalize_min_max(new_metrics["acceptance_count"], min_orig=0, max_orig=self.cycle_ns*self.n_ports//(self.expected_incoming_bytes*8+4000))

        reward = (
            self.largest_hole_weight * float(delta_largest_hole)
            + self.useless_gap_weight * float(delta_useless_gaps)
            + self.acceptance_weight * float(delta_acceptance)
            - (self.invalid_penalty if invalid else 0.0)
        )

        self.focus = old_focus
        self._advance_focus()

        terminated = False
        truncated = bool(self.step_count >= self.max_steps)
        info = {
            "invalid_move": bool(invalid),
            "attempted_delta_ns": int(delta_ns),
            "delta_largest_hole": float(delta_largest_hole),
            "delta_useless_gaps": float(delta_useless_gaps),
            "largest_hole_before_ns": float(old_metrics["largest_hole_ns"]),
            "largest_hole_after_ns": float(new_metrics["largest_hole_ns"]),
            "useless_gaps_before": float(old_metrics["total_useless_gaps"]),
            "useless_gaps_after": float(new_metrics["total_useless_gaps"]),
            "step_count": int(self.step_count),
            "focus_port_index": int(self.focus.port_index),
            "focus_slot_index": int(self.focus.slot_index),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self.reset_mode == "always":
            should_rebuild = True
        elif self.reset_mode == "never":
            should_rebuild = not self.has_built_initial_state
        elif self.reset_mode == "freeze_after":
            should_rebuild = (not self.has_built_initial_state) or (
                self.episode_count < self.freeze_reset_after_episode
            )
        else:
            raise ValueError(f"Unsupported reset_mode={self.reset_mode}")

        if should_rebuild:
            loaded = False
            if self.init_from_usecase_schedule:
                loaded = self._create_initial_schedule_from_usecase()
            if not loaded:
                self._create_fallback_initial_schedule()

            if self._violation_sum() > 0:
                raise ValueError(
                    "Initial schedule is already invalid. This env assumes reset() starts from a valid schedule."
                )

            self.has_built_initial_state = True
            self._backup_current_as_last_valid()
        else:
            # Continue from the last valid schedule reached previously
            self._restore_last_valid()

        self.step_count = 0
        active_ports = self._active_ports()
        self.focus = FocusRef(port_index=active_ports[0] if active_ports else 0, slot_index=0)

        self.episode_count += 1
        return self._get_obs(), {}

    def get_combined_schedule(self) -> Schedule:
        combined = Schedule(cycle_ns=self.cycle_ns)
        for pidx in sorted(self.port_schedules):
            for slot in self.port_schedules[pidx]:
                combined.append(slot)
        return combined

    def render(self):
        return self.get_combined_schedule().plot()
