from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import defaultdict
from itertools import count
import math
    
import numpy as np

import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import matplotlib.cm as cm

from dataclasses_json import DataClassJsonMixin

MS_TO_NS = 1_000_000

@dataclass
class Slot:
    """
    A periodic slot definition.

    One Slot represents a repeating transmission window of a cyclic stream.
    Occurrences within a schedule cycle can be generated
    using `all_occurrences()`.
    
    N.B. If a slot is defined for a set of streams sent back to back, but having different periods 
    then repeating the slot and checking overlap during schedule cycle becomes harder.
    """
    # shared class-level counter
    _counter = count()
    
    start_ns: int
    end_ns: int
    period_ns: float
    stream_id: Optional[int] = None
    port_id: Optional[int] = None
    queue_id: Optional[int] = None
    # unique slot definition id
    index: int = field(default_factory=lambda: next(Slot._counter))

    # -----------------------------
    # Derived properties
    # -----------------------------
    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    # -----------------------------
    # Ordering logic
    # -----------------------------
    def __lt__(self, other: "Slot") -> bool:
        return self.end_ns <= other.start_ns

    # -----------------------------
    # Conflict logic
    # -----------------------------
    def conflicts_with(self, other: "Slot", touching_allowed: bool = True) -> bool:
        if touching_allowed:
            return self.start_ns < other.end_ns and self.end_ns > other.start_ns
        else:
            return self.start_ns <= other.end_ns and self.end_ns >= other.start_ns
            
    # -----------------------------
    # Occurrence expansion
    # -----------------------------
    def all_occurrences(self, span_ns: int) -> List["Slot"]:
        """
        Expand this periodic slot into all occurrences
        within the span.

        All occurrences share the same index.
        """

        if self.period_ns <= 0:
            return []

        occurrences: List[Slot] = []

        k_max = math.floor(
            (span_ns - self.start_ns) // self.period_ns
        )

        for k in range(max(0, k_max) + 1):
            start = int(self.start_ns + k * self.period_ns)
            end = int(self.end_ns + k * self.period_ns)
            if start >= span_ns: 
                break  
            occurrences.append(
                Slot(
                    index=self.index,          # preserve same definition id
                    start_ns=start,
                    end_ns=end,
                    period_ns=self.period_ns,
                    queue_id=self.queue_id,
                    stream_id=self.stream_id,
                    port_id=self.port_id,
                )
            )

            if end > span_ns:
                break
        return occurrences

        
class Schedule(list):
    """
    A schedule of periodic slots.
    Inherits from list, so you can append/remove Slot objects directly.
    Stores schedule metadata and provides validation and compilation utilities.
    """

    # ----------------------------
    # Internal TransmissionWindow
    # ----------------------------
    @dataclass
    class TransmissionWindow(DataClassJsonMixin):
        """A concrete, non-periodic transmission window that spams all the queues"""
        start: int
        duration: int
        state: str     
        stream_sequence: Optional[List[int]] = None  # tracks all contributing streams

        @property
        def end(self):
            return self.start+self.duration

    # ----------------------------
    # Schedule constructor
    # ----------------------------
    def __init__(self, cycle_ns: int, basetime: Optional[int] = None,
                 capacity: Optional[int] = None, iterable: Optional[List] = None):
        """
        Initialize the Schedule.
        :param cycle_ns: schedule cycle in ms
        :param basetime: optional reference start time
        :param capacity: optional max number of concurrent slots
        :param port_id: optional port identifier
        :param iterable: optional initial list of Slot objects
        """
        super().__init__(iterable or [])
        self.cycle_ns = cycle_ns
        self.basetime = basetime
        self.capacity = capacity


    # ----------------------------
    # Utility properties
    # ----------------------------
    @property
    def total_occupied_time(self) -> Dict[int, int]:
        """Return total time occupied by slots for each port_id (without merging)"""
        result = defaultdict(int)
        
        for slot in self:
            # Get all occurrences of this periodic slot
            occurrences = slot.all_occurrences(self.cycle_ns)
            
            for occ in occurrences:
                # Add duration of each occurrence
                result[occ.port_id] += (occ.end_ns - occ.start_ns)
        
        return dict(result)

    @property
    def utilization(self) -> Dict[int, float]:
        """Return fraction of cycle occupied per port_id"""
        # Get the total occupied time for all port_ids
        occupied_times = self.total_occupied_time
    
        if not self.cycle_ns:
            # Return 0.0 for each port if cycle_ns is 0 (avoiding division by zero)
            return {port_id: 0.0 for port_id in occupied_times}
    
        # Calculate utilization for each port_id
        return {port_id: occupied_time / self.cycle_ns for port_id, occupied_time in occupied_times.items()}

    # ----------------------------
    # helper properties
    # ----------------------------
    @property
    def get_n_streams(self) -> int:
        """Number of unique streams in the schedule"""
        return len({slot.stream_id for slot in self if slot.stream_id is not None})

    def remove_stream(self, stream_ids):
        """Remove all slots associated with the given stream IDs."""
        for slot in self[:]:  # iterate over a copy since we may modify the list
            if slot.stream_id in stream_ids:
                self.remove(slot)
        print(f"Removed stream(s) {stream_ids} from schedule successfully.")   

    def reallocate_existing_stream_first_fit_keep_old_if_fail(
        self,
        stream_id,
        usecase,
        problem,
        queue_id=5,
        port_intervals=None,
    ):
        """
        Reallocate one existing stream hop-by-hop using first fit.

        For each hop:
        - search earliest non-overlapping candidate from the hop lower bound
        - validate that hop immediately together with already accepted hops
        - if a valid new placement is found, accept it
        - otherwise keep the old slot for that hop
        Continue until all hops are handled.

        Returns
        -------
        bool
            True if at least one hop moved earlier and the mixed placement
            was committed, False otherwise.
        """
        import copy
        import numpy as np
        from evaluator import TSNConstraintEvaluator, ScheduleEncodingType

        if not (0 <= stream_id < len(usecase.streams)):
            return False

        stream = usecase.streams[stream_id]
        if "pathLists" not in stream or not stream["pathLists"]:
            return False

        old_slots = [copy.copy(slot) for slot in self if slot.stream_id == stream_id]
        if not old_slots:
            return False

        old_slot_by_port = {}
        for slot in old_slots:
            if slot.port_id is not None:
                old_slot_by_port[int(slot.port_id)] = copy.copy(slot)

        old_port_intervals = None
        if port_intervals is not None:
            old_port_intervals = {pid: list(v) for pid, v in port_intervals.items()}

        # Remove the stream temporarily so it does not block itself
        self[:] = [slot for slot in self if slot.stream_id != stream_id]

        if port_intervals is not None:
            affected_ports = sorted({
                int(slot.port_id) for slot in old_slots
                if slot.port_id is not None
            })
            for pid in affected_ports:
                port_intervals[pid] = self._occupied_intervals_on_port(pid, self.cycle_ns)

        problem.usecase = usecase
        problem.build_metadata()

        evaluator = TSNConstraintEvaluator(
            problem,
            check_routing=False,
            check_task_placement=False,
            scheduling_type=ScheduleEncodingType.START_TIME,
        )

        port_name_to_id, _ = usecase.port_name_id_mapping
        path = stream["pathLists"][0]
        period = int(stream["interval"])
        hop_port_names = path[:-1]
        hop_port_ids = [int(port_name_to_id[p]) for p in hop_port_names]

        accepted_slots = []
        hop_starts = []
        moved_any = False

        for hop_idx, pid in enumerate(hop_port_ids):
            occupied_ns = int(max(1, round(evaluator.occupiedLengths[pid - 1, stream_id])))

            if hop_idx == 0:
                min_start = 0
            else:
                prev_pid = hop_port_ids[hop_idx - 1]
                prev_start = hop_starts[-1]
                min_start = prev_start + int(round(evaluator.all_hops_costs[prev_pid - 1, stream_id]))

            if port_intervals is not None:
                current_intervals = list(port_intervals.get(pid, []))
            else:
                current_intervals = self._occupied_intervals_on_port(pid, self.cycle_ns)

            for s in accepted_slots:
                if int(s.port_id) == int(pid):
                    current_intervals.append((int(s.start_ns), int(s.end_ns)))
            current_intervals.sort()

            old_slot = old_slot_by_port.get(int(pid))
            if old_slot is None:
                # restore full old placement
                self.extend(old_slots)
                self.sort()
                if port_intervals is not None and old_port_intervals is not None:
                    port_intervals.clear()
                    port_intervals.update(old_port_intervals)
                return False

            old_start = int(old_slot.start_ns)
            chosen_slot = None

            candidate_start = self._first_non_overlapping_start(
                current_intervals,
                min_start,
                occupied_ns,
                period,
            )

            # Only search earlier-or-equal placements; if none works, keep old hop
            while candidate_start is not None and int(candidate_start) <= old_start:
                trial_slot = Slot(
                    start_ns=int(candidate_start),
                    end_ns=int(candidate_start + occupied_ns),
                    period_ns=int(period),
                    stream_id=int(stream_id),
                    port_id=int(pid),
                    queue_id=queue_id,
                )

                tmp_schedule = Schedule(
                    cycle_ns=self.cycle_ns,
                    basetime=self.basetime,
                    iterable=list(self) + list(accepted_slots) + [trial_slot],
                )

                try:
                    compact = tmp_schedule.to_compact_encoding(problem.n_ports, problem.n_streams)
                    violations = evaluator.per_stream_all_violations(compact)
                    stream_v = violations[:, int(stream_id)]
                    if np.all(stream_v == 0):
                        chosen_slot = trial_slot
                        break
                except ValueError:
                    pass

                next_start = self._first_non_overlapping_start(
                    current_intervals,
                    int(candidate_start) + 1,
                    occupied_ns,
                    period,
                )
                if next_start is None or int(next_start) <= int(candidate_start):
                    candidate_start = None
                else:
                    candidate_start = int(next_start)

            if chosen_slot is None:
                # fallback: keep old slot for this hop and continue
                chosen_slot = Slot(
                    start_ns=int(old_slot.start_ns),
                    end_ns=int(old_slot.end_ns),
                    period_ns=int(old_slot.period_ns),
                    stream_id=int(stream_id),
                    port_id=int(old_slot.port_id),
                    queue_id=old_slot.queue_id if old_slot.queue_id is not None else queue_id,
                )
            else:
                if int(chosen_slot.start_ns) < int(old_slot.start_ns):
                    moved_any = True

            accepted_slots.append(chosen_slot)
            hop_starts.append(int(chosen_slot.start_ns))

        # Commit mixed placement
        self.extend(accepted_slots)
        self.sort()

        usecase.streams[stream_id]["offset"] = int(hop_starts[0])

        if port_intervals is not None:
            affected_ports = sorted({
                int(slot.port_id) for slot in accepted_slots
                if slot.port_id is not None
            })
            for pid in affected_ports:
                port_intervals[pid] = self._occupied_intervals_on_port(pid, self.cycle_ns)

        usecase.removeSchedule()
        _, port_id_to_name = usecase.port_name_id_mapping
        usecase.schedule = self
        self.update_graph(usecase.graph, port_id_to_name)

        return moved_any
     
    def banana(self):
        print("banana")


    def place_existing_stream_right_fit_backward(
        self,
        stream_id,
        usecase,
        problem,
        queue_id=5,
        port_intervals=None,
        max_candidates_per_hop=3,
        beam_width=5,
    ):
        """
        True right-fit placement for an existing stream, preserving stream_id.

        Unlike left-to-right placement, this builds the path backwards:
        - place last hop as late as possible
        - then place previous hops as late as possible while preserving causality

        Assumes the stream's old slots have already been removed.
        """
        import numpy as np
        from evaluator import TSNConstraintEvaluator, ScheduleEncodingType

        max_candidates_per_hop = max(1, int(max_candidates_per_hop))
        beam_width = max(1, int(beam_width))

        stream_id = int(stream_id)

        if not (0 <= stream_id < len(usecase.streams)):
            return {"accepted": [], "rejected": [stream_id]}

        stream = usecase.streams[stream_id]

        if "pathLists" not in stream or not stream["pathLists"]:
            return {"accepted": [], "rejected": [stream_id]}

        def merge_intervals(intervals, period_ns):
            cleaned = []
            for a, b in intervals:
                a = max(0, int(a))
                b = min(int(period_ns), int(b))
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

        def fits_interval(start, duration, intervals, period_ns):
            start = int(start)
            end = int(start + duration)

            if start < 0 or end > int(period_ns):
                return False

            for a, b in intervals:
                if start < int(b) and end > int(a):
                    return False

            return True

        def right_candidates_with_latest_end(
            intervals,
            latest_end,
            duration,
            period_ns,
            limit,
        ):
            """
            Return valid starts from right to left, with start+duration <= latest_end.
            """
            latest_end = min(int(latest_end), int(period_ns))
            duration = int(duration)
            period_ns = int(period_ns)

            if duration <= 0:
                return [0]

            merged = merge_intervals(intervals, period_ns)

            candidates = set()

            # Latest possible start before latest_end
            candidates.add(latest_end - duration)

            # Just before occupied intervals
            for a, _ in merged:
                candidates.add(int(a) - duration)

            valid = []

            for s in sorted(candidates, reverse=True):
                if s < 0:
                    continue
                if s + duration > latest_end:
                    continue

                if fits_interval(s, duration, merged, period_ns):
                    valid.append(int(s))

                if len(valid) >= int(limit):
                    break

            return valid

        def score_backward(starts_by_hop):
            """
            Lower is better.
            Prefer larger starts, especially first-hop offset.
            """
            starts = [starts_by_hop[i] for i in sorted(starts_by_hop.keys())]
            first_start = starts[0]
            total = sum(starts)
            span = starts[-1] - starts[0]
            return -10 * first_start - total + span

        problem.usecase = usecase
        problem.build_metadata()

        port_name_to_id, _ = usecase.port_name_id_mapping

        evaluator = TSNConstraintEvaluator(
            problem,
            check_routing=False,
            check_task_placement=False,
            scheduling_type=ScheduleEncodingType.START_TIME,
        )

        path = stream["pathLists"][0]
        period = int(stream["interval"])

        hop_port_names = path[:-1]
        hop_port_ids = [int(port_name_to_id[p]) for p in hop_port_names]

        if not hop_port_ids:
            return {"accepted": [], "rejected": [stream_id]}

        n_hops = len(hop_port_ids)

        # Beam state:
        #   starts_by_hop: dict hop_idx -> start
        #   slots_by_hop: dict hop_idx -> Slot
        #   score
        beam = [({}, {}, 0.0)]

        # Build from last hop to first hop
        for hop_idx in reversed(range(n_hops)):
            pid = hop_port_ids[hop_idx]
            duration = int(max(1, round(evaluator.occupiedLengths[pid - 1, stream_id])))

            new_beam = []

            for starts_by_hop, slots_by_hop, _ in beam:
                # For the last hop, latest end is the period end.
                # For earlier hops, it must finish early enough before next hop.
                if hop_idx == n_hops - 1:
                    latest_end = period
                else:
                    next_pid = hop_port_ids[hop_idx]
                    next_start = starts_by_hop[hop_idx + 1]

                    # Same relation used forward:
                    # next_start >= this_start + all_hops_costs[this_pid-1, stream_id]
                    # therefore this_start <= next_start - cost
                    cost_to_next = int(
                        round(evaluator.all_hops_costs[pid - 1, stream_id])
                    )
                    latest_start = int(next_start - cost_to_next)
                    latest_end = latest_start + duration

                if latest_end <= 0:
                    continue

                if port_intervals is not None:
                    current_intervals = list(port_intervals.get(pid, []))
                else:
                    current_intervals = self._occupied_intervals_on_port(pid, self.cycle_ns)

                # Same-stream slots already selected in this backward candidate
                for s in slots_by_hop.values():
                    if int(s.port_id) == int(pid):
                        current_intervals.append((int(s.start_ns), int(s.end_ns)))

                starts = right_candidates_with_latest_end(
                    intervals=current_intervals,
                    latest_end=latest_end,
                    duration=duration,
                    period_ns=period,
                    limit=max_candidates_per_hop,
                )

                for st in starts:
                    slot = Slot(
                        start_ns=int(st),
                        end_ns=int(st + duration),
                        period_ns=int(period),
                        stream_id=int(stream_id),
                        port_id=int(pid),
                        queue_id=queue_id,
                    )

                    new_starts = dict(starts_by_hop)
                    new_slots = dict(slots_by_hop)

                    new_starts[hop_idx] = int(st)
                    new_slots[hop_idx] = slot

                    # Optional local ordering check for already known next hop
                    if hop_idx < n_hops - 1:
                        cost_to_next = int(
                            round(evaluator.all_hops_costs[pid - 1, stream_id])
                        )
                        if new_starts[hop_idx] + cost_to_next > new_starts[hop_idx + 1]:
                            continue

                    sc = score_backward(new_starts)
                    new_beam.append((new_starts, new_slots, sc))

            if not new_beam:
                return {"accepted": [], "rejected": [stream_id]}

            new_beam.sort(key=lambda x: x[2])
            beam = new_beam[:beam_width]

        base_slots = list(self)

        accepted_slots = None
        accepted_starts = None

        for starts_by_hop, slots_by_hop, _ in beam:
            partial_slots = [
                slots_by_hop[i]
                for i in range(n_hops)
            ]

            hop_starts = [
                starts_by_hop[i]
                for i in range(n_hops)
            ]

            tmp_schedule = Schedule(
                cycle_ns=self.cycle_ns,
                basetime=self.basetime,
                iterable=base_slots + list(partial_slots),
            )

            try:
                compact = tmp_schedule.to_compact_encoding(
                    problem.n_ports,
                    problem.n_streams,
                )
            except ValueError:
                continue

            violations = evaluator.per_stream_all_violations(compact)
            stream_v = violations[:, stream_id]

            if np.all(stream_v == 0):
                accepted_slots = partial_slots
                accepted_starts = hop_starts
                break

        if accepted_slots is None:
            return {"accepted": [], "rejected": [stream_id]}

        for slot in accepted_slots:
            self.append(slot)

        self.sort()

        usecase.streams[stream_id]["offset"] = int(accepted_starts[0])

        if port_intervals is not None:
            affected_ports = sorted({
                int(slot.port_id)
                for slot in accepted_slots
                if slot.port_id is not None
            })

            for pid in affected_ports:
                port_intervals[pid] = self._occupied_intervals_on_port(
                    pid,
                    self.cycle_ns,
                )

        usecase.removeSchedule()
        _, port_id_to_name = usecase.port_name_id_mapping
        usecase.schedule = self
        self.update_graph(usecase.graph, port_id_to_name)

        return {
            "accepted": [stream_id],
            "rejected": [],
            "starts": accepted_starts,
        }

    
    def place_existing_stream_first_fit_fast(
        self,
        stream_id,
        usecase,
        problem,
        queue_id=5,
        port_intervals=None,
        max_candidates_per_hop=3,
        beam_width=5,
    ):
        """
        Place an already existing stream definition into the current schedule
        while preserving its original stream_id.

        This is the preserve-ID version of place_stream_first_fit_fast().
        It assumes the stream's old slots have already been removed from the schedule.

        Returns
        -------
        dict
            {"accepted": [stream_id], "rejected": [], "starts": [...]} on success
            {"accepted": [], "rejected": [stream_id]} on failure
        """
        import numpy as np
        from evaluator import TSNConstraintEvaluator, ScheduleEncodingType

        max_candidates_per_hop = max(1, int(max_candidates_per_hop))
        beam_width = max(1, int(beam_width))

        if not (0 <= int(stream_id) < len(usecase.streams)):
            return {"accepted": [], "rejected": [int(stream_id)]}

        stream_id = int(stream_id)
        stream = usecase.streams[stream_id]

        if "pathLists" not in stream or not stream["pathLists"]:
            return {"accepted": [], "rejected": [stream_id]}

        def candidate_starts_for_hop(intervals, min_start, duration, period_ns, limit):
            """Return first `limit` non-overlapping starts >= min_start."""
            starts = []

            cand = self._first_non_overlapping_start(
                intervals=intervals,
                min_start=int(min_start),
                duration=int(duration),
                period_ns=int(period_ns),
            )

            while cand is not None and len(starts) < limit:
                starts.append(int(cand))

                cand = self._first_non_overlapping_start(
                    intervals=intervals,
                    min_start=int(cand) + 1,
                    duration=int(duration),
                    period_ns=int(period_ns),
                )

            return starts

        def score_partial(slots, starts):
            """
            Lower is better.
            Prefer left compaction and avoid unnecessarily spread paths.
            """
            if not starts:
                return 10**18

            first_start = int(starts[0])
            last_start = int(starts[-1])
            span = last_start - first_start

            return 10 * first_start + sum(starts) + span

        # Important: DO NOT append to usecase.streams.
        # This function preserves the existing stream_id.
        problem.usecase = usecase
        problem.build_metadata()

        port_name_to_id, _ = usecase.port_name_id_mapping

        evaluator = TSNConstraintEvaluator(
            problem,
            check_routing=False,
            check_task_placement=False,
            scheduling_type=ScheduleEncodingType.START_TIME,
        )

        path = stream["pathLists"][0]
        period = int(stream["interval"])

        hop_port_names = path[:-1]
        hop_port_ids = [int(port_name_to_id[p]) for p in hop_port_names]

        if not hop_port_ids:
            return {"accepted": [], "rejected": [stream_id]}

        # Beam state: (partial_slots, hop_starts, score)
        beam = [([], [], 0.0)]

        for hop_idx, pid in enumerate(hop_port_ids):
            occupied_ns = int(max(1, round(evaluator.occupiedLengths[pid - 1, stream_id])))

            new_beam = []

            for partial_slots, hop_starts, _ in beam:
                if hop_idx == 0:
                    min_start = 0
                else:
                    prev_pid = hop_port_ids[hop_idx - 1]
                    prev_start = hop_starts[-1]
                    min_start = prev_start + int(
                        round(evaluator.all_hops_costs[prev_pid - 1, stream_id])
                    )

                if port_intervals is not None:
                    current_intervals = list(port_intervals.get(pid, []))
                else:
                    current_intervals = self._occupied_intervals_on_port(pid, self.cycle_ns)

                # Same stream slots already selected in this candidate
                for s in partial_slots:
                    if int(s.port_id) == int(pid):
                        s_start = int(s.start_ns)
                        s_duration = int(s.end_ns) - int(s.start_ns)
                        s_period = int(s.period_ns)

                        k = 0
                        while s_start + k * s_period < int(self.cycle_ns):
                            occ_start = s_start + k * s_period
                            occ_end = occ_start + s_duration

                            if occ_end <= int(self.cycle_ns):
                                current_intervals.append((occ_start, occ_end))

                            k += 1

                current_intervals.sort()

                starts = candidate_starts_for_hop(
                    intervals=current_intervals,
                    min_start=min_start,
                    duration=occupied_ns,
                    period_ns=period,
                    limit=max_candidates_per_hop,
                )

                for st in starts:
                    slot = Slot(
                        start_ns=int(st),
                        end_ns=int(st + occupied_ns),
                        period_ns=int(period),
                        stream_id=int(stream_id),
                        port_id=int(pid),
                        queue_id=queue_id,
                    )

                    new_slots = partial_slots + [slot]
                    new_starts = hop_starts + [int(st)]
                    sc = score_partial(new_slots, new_starts)

                    new_beam.append((new_slots, new_starts, sc))

            if not new_beam:
                return {"accepted": [], "rejected": [stream_id]}

            new_beam.sort(key=lambda x: x[2])
            beam = new_beam[:beam_width]

        # Final validation only on complete candidates
        base_slots = list(self)

        accepted_slots = None
        accepted_starts = None

        for partial_slots, hop_starts, _ in beam:
            tmp_schedule = Schedule(
                cycle_ns=self.cycle_ns,
                basetime=self.basetime,
                iterable=base_slots + list(partial_slots),
            )

            try:
                compact = tmp_schedule.to_compact_encoding(
                    problem.n_ports,
                    problem.n_streams,
                )
            except ValueError:
                continue

            violations = evaluator.per_stream_all_violations(compact)
            stream_v = violations[:, stream_id]

            if np.all(stream_v == 0):
                accepted_slots = partial_slots
                accepted_starts = hop_starts
                break

        if accepted_slots is None:
            return {"accepted": [], "rejected": [stream_id]}

        # Commit
        for slot in accepted_slots:
            self.append(slot)

        self.sort()

        usecase.streams[stream_id]["offset"] = int(accepted_starts[0])

        if port_intervals is not None:
            affected_ports = sorted({
                int(slot.port_id)
                for slot in accepted_slots
                if slot.port_id is not None
            })

            for pid in affected_ports:
                port_intervals[pid] = self._occupied_intervals_on_port(
                    pid,
                    self.cycle_ns,
                )

        usecase.removeSchedule()
        _, port_id_to_name = usecase.port_name_id_mapping
        usecase.schedule = self
        self.update_graph(usecase.graph, port_id_to_name)

        return {
            "accepted": [stream_id],
            "rejected": [],
            "starts": accepted_starts,
        }

 
    def oldest_first_first_fit_defrag(
        self,
        usecase,
        problem,
        p,
        n,
        departure_counter,
        queue_id=5,
        port_intervals=None,
        stream_arrival_times=None,
    ):
        """
        OF-FF(p,n):
        every p departures, pick the n oldest active streams and try to
        reallocate them one by one.

        Oldest ordering:
        1) smallest arrival time
        2) smallest first-hop offset
        3) smallest stream id

        If stream_arrival_times is None, ordering falls back effectively to:
        1) smallest first-hop offset
        2) smallest stream id
        """
        if p <= 0 or n <= 0:
            return {
                "triggered": False,
                "selected_stream_ids": [],
                "replaced_count": 0,
            }

        if departure_counter == 0 or (departure_counter % p) != 0:
            return {
                "triggered": False,
                "selected_stream_ids": [],
                "replaced_count": 0,
            }

        if stream_arrival_times is None:
            stream_arrival_times = {}

        active_stream_ids = sorted({
            int(slot.stream_id) for slot in self
            if slot.stream_id is not None
        })

        if not active_stream_ids:
            return {
                "triggered": True,
                "selected_stream_ids": [],
                "replaced_count": 0,
            }

        # first-hop offset proxy = earliest scheduled start among this stream's slots
        stream_first_offset = {}
        for sid in active_stream_ids:
            starts = [
                int(slot.start_ns)
                for slot in self
                if slot.stream_id is not None and int(slot.stream_id) == int(sid)
            ]
            stream_first_offset[sid] = min(starts) if starts else float("inf")

        ordered_stream_ids = sorted(
            active_stream_ids,
            key=lambda sid: (
                float(stream_arrival_times.get(int(sid), float("inf"))),
                float(stream_first_offset.get(int(sid), float("inf"))),
                int(sid),
            ),
        )

        selected_stream_ids = ordered_stream_ids[:min(int(n), len(ordered_stream_ids))]

        replaced_count = 0
        moved_stream_ids = []

        for sid in selected_stream_ids:
            moved = self.reallocate_existing_stream_first_fit_keep_old_if_fail(
                stream_id=int(sid),
                usecase=usecase,
                problem=problem,
                queue_id=queue_id,
                port_intervals=port_intervals,
            )
            if moved:
                replaced_count += 1
                moved_stream_ids.append(int(sid))

        if usecase is not None:
            usecase.removeSchedule()
            _, port_id_to_name = usecase.port_name_id_mapping
            usecase.schedule = self
            self.update_graph(usecase.graph, port_id_to_name)

        return {
            "triggered": True,
            "selected_stream_ids": selected_stream_ids,
            "moved_stream_ids": moved_stream_ids,
            "replaced_count": replaced_count,
        }

    def delete_streams(
        self,
        n_remove,
        usecase=None,
        save_json=False,
        jsnPath=None,
        ucFileName=None,
        rng=None,
        port_intervals=None,
        stream_arrival_times=None,
    ):
        """
        Remove n_remove random streams from the currently scheduled streams.

        Parameters
        ----------
        n_remove : int
            Number of distinct scheduled streams to remove.
        usecase : Usecase, optional
            If provided, the usecase graph/streams will be refreshed to reflect
            the new schedule. Required if save_json=True.
        save_json : bool
            If True, export the updated usecase to JSON.
        jsnPath : str, optional
            Output folder passed to usecase.to_file().
        ucFileName : str, optional
            Output file name passed to usecase.to_file().
        rng : random.Random or numpy.random.Generator, optional
            Random generator for reproducibility.
        port_intervals : dict, optional
            Cached {port_id: [(start,end), ...]} intervals to update incrementally.
        stream_arrival_times : dict, optional
            Kept for backward compatibility. Not used in random deletion.

        Returns
        -------
        dict
            {
                "removed_stream_ids": [...],
                "removed_slot_count": int
            }
        """
        import random
        import numpy as np

        if rng is None:
            rng = random.Random()

        if not isinstance(n_remove, int) or n_remove < 0:
            raise ValueError("n_remove must be a non-negative integer")

        scheduled_stream_ids = sorted({
            int(slot.stream_id)
            for slot in self
            if slot.stream_id is not None
        })

        if n_remove > len(scheduled_stream_ids):
            raise ValueError(
                f"Requested removal of {n_remove} streams, "
                f"but only {len(scheduled_stream_ids)} scheduled streams exist."
            )

        if n_remove == 0:
            return {
                "removed_stream_ids": [],
                "removed_slot_count": 0,
            }

        # Randomly choose distinct active/scheduled streams.
        # Supports both Python random.Random and numpy Generator/RandomState.
        if hasattr(rng, "sample"):
            selected_stream_ids = rng.sample(scheduled_stream_ids, n_remove)
        elif hasattr(rng, "choice"):
            selected_stream_ids = rng.choice(
                scheduled_stream_ids,
                size=n_remove,
                replace=False,
            ).tolist()
        else:
            raise TypeError(
                "rng must be either random.Random-like with .sample() "
                "or numpy Generator/RandomState-like with .choice()."
            )

        stream_ids = set(int(sid) for sid in selected_stream_ids)

        # Ports affected before removal
        affected_ports = sorted({
            int(slot.port_id)
            for slot in self
            if slot.stream_id is not None
            and int(slot.stream_id) in stream_ids
            and slot.port_id is not None
        })

        before = len(self)
        self[:] = [
            slot for slot in self
            if slot.stream_id is None or int(slot.stream_id) not in stream_ids
        ]
        removed_slots = before - len(self)

        # Refresh cached intervals only on affected ports
        if port_intervals is not None:
            for pid in affected_ports:
                port_intervals[pid] = self._occupied_intervals_on_port(
                    pid,
                    self.cycle_ns,
                )

        if usecase is None and not save_json:
            return {
                "removed_stream_ids": sorted(stream_ids),
                "removed_slot_count": removed_slots,
            }

        if save_json and usecase is None:
            raise ValueError("usecase must be provided when save_json=True")

        if usecase is not None:
            usecase.removeSchedule()

            # Clear schedule-derived fields only for removed streams.
            # Keep stream definitions to preserve stable stream IDs.
            for sid in stream_ids:
                if 0 <= int(sid) < len(usecase.streams):
                    stream = usecase.streams[int(sid)]
                    for key in ("offset", "accLatency", "pathLists"):
                        if key in stream:
                            del stream[key]

            _, port_id_to_name = usecase.port_name_id_mapping
            usecase.schedule = self
            self.update_graph(usecase.graph, port_id_to_name)

        if save_json:
            usecase.to_file(jsnPath=jsnPath, ucFileName=ucFileName)

        return {
            "removed_stream_ids": sorted(stream_ids),
            "removed_slot_count": removed_slots,
        }


    def _occupied_intervals_on_port(self, port_id, span_ns = None):
        """
        Return sorted occupied intervals [start, end) on a port within one cycle.
        """
        if span_ns is None:
            span_ns = self.cycle_ns
        span_ns = int(span_ns)
        intervals = []
        for slot in self:
            if slot.port_id != port_id:
                continue
            # current use case is one-cycle scheduling, but keep this safe
            for occ in slot.all_occurrences(span_ns):
                s = int(occ.start_ns)
                e = int(occ.end_ns)
                if s < span_ns and e > 0:
                    intervals.append((max(0, s), min(span_ns, e)))
        intervals.sort()
        return intervals

    def _first_non_overlapping_start(self, intervals, min_start, duration, period_ns):
        """
        Return earliest start >= min_start such that all periodic occurrences of
        [start, start+duration) within the schedule macrocycle do not overlap
        existing intervals on this port.

        Example:
        cycle_ns = 300k, period_ns = 100k
        candidate start = 10k, duration = 5k

        We check:
            [10k, 15k)
            [110k, 115k)
            [210k, 215k)

        The returned start is an offset within one period, so:
        0 <= start and start + duration <= period_ns
        """
        start = int(min_start)
        duration = int(duration)
        period_ns = int(period_ns)
        cycle_ns = int(self.cycle_ns)

        if duration <= 0 or period_ns <= 0 or cycle_ns <= 0:
            return None

        # Start must fit inside one period.
        latest_start = period_ns - duration
        if latest_start < 0:
            return None

        start = max(0, start)

        # Normalize intervals and keep only valid ones.
        intervals = sorted(
            (int(a), int(b))
            for a, b in intervals
            if int(b) > int(a)
        )

        def occurrences_for_start(st):
            occ = []
            k = 0
            while st + k * period_ns < cycle_ns:
                a = st + k * period_ns
                b = a + duration

                # Candidate occurrence must stay inside macrocycle.
                if b > cycle_ns:
                    return None

                occ.append((a, b))
                k += 1

            return occ

        while start <= latest_start:
            occs = occurrences_for_start(start)
            if occs is None:
                return None

            jumped = False

            for ca, cb in occs:
                for a, b in intervals:
                    if ca < b and cb > a:
                        # Conflict for occurrence ca = start + k*period.
                        # Move base start so this occurrence begins at b.
                        k = (ca - start) // period_ns
                        new_start = int(b) - int(k) * period_ns

                        # Ensure progress.
                        if new_start <= start:
                            new_start = start + 1

                        start = new_start
                        jumped = True
                        break

                if jumped:
                    break

            if not jumped:
                return int(start)

        return None
        
    def generate_and_place_line_streams_first_fit(
        self,
        usecase,
        problem,
        n_streams=1,
        frame_size=1024,
        period=100000,
        latency_factors=(0.6, 0.8),
        queue_id=5,
        save_json=False,
        jsnPath=None,
        ucFileName=None,
        rng=None,
        port_intervals=None,
    ):
        """
        Generate streams for the fixed 2-host / 2-switch line topology and place them
        with sequential hop-by-hop first-fit.

        Half are assigned forward path and half reverse path (randomized order),
        without fallback to the other direction.

        Optimized behavior:
        - place hop 1 greedily by first-fit on its port
        - place hop 2 greedily by first-fit from its sequencing lower bound
        - place hop 3 greedily by first-fit from its sequencing lower bound
        - run evaluator only ONCE for the full 3-hop candidate
        - update cached port intervals incrementally if provided
        """
        import random
        import numpy as np
        from evaluator import TSNConstraintEvaluator, ScheduleEncodingType

        rng = random.Random() if rng is None else rng

        port_name_to_id, _ = usecase.port_name_id_mapping

        forward_path = [
            "0-ED1.Port1",
            "0-SW1.P2",
            "0-SW2.P1",
            "0-ED2.Port1",
        ]
        reverse_path = [
            "0-ED2.Port1",
            "0-SW2.P2",
            "0-SW1.P1",
            "0-ED1.Port1",
        ]

        accepted = []
        rejected = []

        n_forward = n_streams // 2
        n_reverse = n_streams - n_forward

        direction_plan = (
            [("0-ED1", "0-ED2", forward_path)] * n_forward +
            [("0-ED2", "0-ED1", reverse_path)] * n_reverse
        )
        rng.shuffle(direction_plan)

        for talker, listener, path in direction_plan:
            stream_id = len(usecase.streams)
            factor = rng.choice(list(latency_factors))
            max_latency = int(round(factor * period))

            stream = {
                "name": f"stream-{stream_id}",
                "type": "isochronous",
                "interval": int(period),
                "maxFramesPerInterval": 1,
                "maxFrameSize": int(frame_size),
                "maxLatency": int(max_latency),
                "tlkrPssblPlcmnt": [talker],
                "lstnsrPssblPlcmnt": [[listener]],
                "earliest-transmit-offset": 0,
                "latest-transmit-offset": int(period - 1),
                "jitter": 0,
                "pathLists": [path],
                "originalStreamId": stream_id,
            }

            # Temporarily append so problem/evaluator dimensions include the new stream
            usecase.streams.append(stream)
            problem.usecase = usecase
            problem.build_metadata()

            evaluator = TSNConstraintEvaluator(
                problem,
                check_routing=False,
                check_task_placement=False,
                scheduling_type=ScheduleEncodingType.START_TIME,
            )

            hop_port_names = path[:-1]   # transmitting hops only
            hop_port_ids = [port_name_to_id[p] for p in hop_port_names]

            partial_slots = []
            hop_starts = []
            path_failed = False

            for hop_idx, pid in enumerate(hop_port_ids):
                occupied_ns = int(max(1, round(evaluator.occupiedLengths[pid - 1, stream_id])))

                if hop_idx == 0:
                    min_start = 0
                else:
                    prev_pid = hop_port_ids[hop_idx - 1]
                    prev_start = hop_starts[-1]
                    min_start = prev_start + int(round(evaluator.all_hops_costs[prev_pid - 1, stream_id]))

                # use cached intervals if available, otherwise build them on the fly
                if port_intervals is not None:
                    current_intervals = list(port_intervals.get(pid, []))
                else:
                    current_intervals = self._occupied_intervals_on_port(pid, self.cycle_ns)

                # add any already chosen candidate slots on the same port
                for s in partial_slots:
                    if s.port_id == pid:
                        current_intervals.append((int(s.start_ns), int(s.end_ns)))
                current_intervals.sort()

                start_ns = self._first_non_overlapping_start(
                    current_intervals,
                    min_start,
                    occupied_ns,
                    period,
                )

                if start_ns is None:
                    path_failed = True
                    break

                candidate_slot = Slot(
                    start_ns=int(start_ns),
                    end_ns=int(start_ns + occupied_ns),
                    period_ns=int(period),
                    stream_id=stream_id,
                    port_id=pid,
                    queue_id=queue_id,
                )

                partial_slots.append(candidate_slot)
                hop_starts.append(int(start_ns))

            if path_failed:
                usecase.streams.pop()
                problem.usecase = usecase
                problem.build_metadata()
                rejected.append(stream_id)
                continue

            # Final evaluator check ONCE for the whole 3-hop candidate
            tmp_schedule = Schedule(
                cycle_ns=self.cycle_ns,
                basetime=self.basetime,
                iterable=list(self) + list(partial_slots),
            )

            try:
                compact = tmp_schedule.to_compact_encoding(problem.n_ports, problem.n_streams)
            except ValueError:
                usecase.streams.pop()
                problem.usecase = usecase
                problem.build_metadata()
                rejected.append(stream_id)
                continue

            violations = evaluator.per_stream_all_violations(compact)
            stream_v = violations[:, stream_id]

            if not np.all(stream_v == 0):
                usecase.streams.pop()
                problem.usecase = usecase
                problem.build_metadata()
                rejected.append(stream_id)
                continue

            # Commit accepted slots
            for slot in partial_slots:
                self.append(slot)

            if hop_starts:
                usecase.streams[stream_id]["offset"] = int(hop_starts[0])

            # update cached intervals incrementally
            if port_intervals is not None:
                for slot in partial_slots:
                    pid = slot.port_id
                    port_intervals.setdefault(pid, [])
                    port_intervals[pid].append((int(slot.start_ns), int(slot.end_ns)))
                    port_intervals[pid].sort()

            accepted.append(stream_id)

        # Refresh only graph port schedules; keep stream definitions/pathLists untouched
        usecase.removeSchedule()
        _, port_id_to_name = usecase.port_name_id_mapping
        usecase.schedule = self
        self.update_graph(usecase.graph, port_id_to_name)

        if save_json:
            out = usecase.to_file(jsnPath=jsnPath, ucFileName=ucFileName)
            return {"accepted": accepted, "rejected": rejected, "output": out}

        return {"accepted": accepted, "rejected": rejected}
        
    # ----------------------------
    # Conflict checking
    # ----------------------------
    def validate_conflicts(self, touching_allowed: bool = True) -> List[tuple]:
        """
        Check all occurrences of all slots for conflicts.
        Returns a list of tuples of conflicting Slot objects.
        """
        conflicts = []
        occurrences = defaultdict(list)

        # Collect occurrences by port_id
        for slot in self:
            occurrences[slot.port_id].extend(slot.all_occurrences(self.cycle_ns))

        for port_id, port_slots in occurrences.items():
            # Sort by start time
            port_slots.sort(key=lambda s: s.start_ns)
            # Check pairwise for conflicts
            for i in range(len(port_slots) - 1):
                if port_slots[i].conflicts_with(port_slots[i+1], touching_allowed):
                    conflicts.append((port_slots[i], port_slots[i+1]))

        return conflicts

    # ----------------------------
    # Compile schedule
    # ----------------------------  

    @classmethod
    def from_usecase(cls, usecase, port_name_to_id):
        original_to_index = {}
        for idx, stream in enumerate(usecase.streams):
            original_id = stream.get("originalStreamId")
            if original_id is not None:
                original_to_index[int(original_id)] = idx
    
        def transmission_window_to_slots(tw, port_name):
            slots = []
            start = int(tw["start"])
            open_queues = [i for i, ch in enumerate(tw["state"]) if ch == "o"]
            stream_ids = [int(s) for s in tw["stream_sequence"]]
            for stream_id in stream_ids:
                stream_idx = original_to_index.get(stream_id, stream_id)
                if stream_idx < 0 or stream_idx >= len(usecase.streams):
                    continue

                period_ns = usecase.streams[stream_idx]['interval'] #* MS_TO_NS
                duration_ns = tw.get("duration")
                if duration_ns is None:
                    duration_ns = usecase.window_length(stream_idx, port_name)
                duration_ns = max(1, int(duration_ns))
                for qid in open_queues:
                    slots.append(
                        Slot(
                            start_ns=start,
                            end_ns=start + duration_ns,
                            period_ns=period_ns,
                            stream_id=stream_id,
                            port_id=port_id,
                            queue_id=qid,
                        )
                    )
            return slots
        
        all_slots = defaultdict(list)
        cycle_ns = None
        for node, nodeData in usecase.graph.nodes(data=True):
            for port, portData in nodeData["ports"].items():
                port_name = f"{node}.{port}"
                port_id = port_name_to_id[port_name]
                cycle_ns = portData.get("scheduleCycle", cycle_ns)
                schedule = portData.get("schedule", [])
                for tw in schedule:
                    slots = transmission_window_to_slots(
                        tw,
                        port_name
                    )
                    for s in slots:
                        s.port_id = port_id
                        offset = s.start_ns % s.period_ns if s.period_ns else s.start_ns
                        key = (port_id, s.stream_id, s.queue_id, offset)
                        all_slots[key].append(s)

        schedule = cls(cycle_ns)
        for slots in all_slots.values():
            base = min(slots, key=lambda s: s.start_ns)
            schedule.append(base)

        # Persist port metadata on the schedule to keep plotting order consistent.
        schedule.port_id_to_name = {
            pid: name for name, pid in port_name_to_id.items() if pid != 0
        }
        schedule.port_sort_key = {}
        for pid, pname in schedule.port_id_to_name.items():
            pmeta = usecase.ports.get(pname, {})
            src = pmeta.get("sourceNode")
            dst = pmeta.get("destNode")
            if src is not None and dst is not None:
                schedule.port_sort_key[pid] = (str(src), str(dst))
            else:
                schedule.port_sort_key[pid] = (str(pname), "")
    
        return schedule
        
    def update_graph(self, graph, port_id_to_name):
        """
        Expand all periodic slots into their occurrences and merge overlapping/touching windows.
        Returns a list of TransmissionWindow objects (non-periodic).
        """
        occurrences = []
    
        for slot in self:
            occurrences.extend(slot.all_occurrences(self.cycle_ns))
    
        occurrences.sort(key=lambda s: s.start_ns)
    
        compiled: Dict[int, List[Schedule.TransmissionWindow]] = defaultdict(list)
    
        for occ in occurrences:
            node_name, port = port_id_to_name[occ.port_id].split('.')
            port_data = graph.nodes[node_name]['ports'][port]
            queueCount = port_data.get("queueCount", 8)
            windows = compiled[occ.port_id]
    
            # Build state string for this window
            state_list = list('C' * queueCount)
            if occ.queue_id is not None:
                state_list[occ.queue_id] = 'o'
            state = ''.join(state_list)
    
            # Create new window
            new_window = self.TransmissionWindow(
                start=int(occ.start_ns),
                duration=int(occ.end_ns - occ.start_ns),
                state=state,
                stream_sequence=[occ.stream_id] if occ.stream_id is not None else []
            )
    
            # Merge if overlapping and same state
            if windows and (occ.start_ns <= windows[-1].start + windows[-1].duration) and state == windows[-1].state:
                last = windows[-1]
                # Extend duration
                end_time = max(last.start + last.duration, new_window.start + new_window.duration)
                last.duration = end_time - last.start
                # Merge stream_sequence
                if new_window.stream_sequence:
                    if last.stream_sequence is None:
                        last.stream_sequence = new_window.stream_sequence
                    else:
                        for sid in new_window.stream_sequence:
                            if sid not in last.stream_sequence:
                                last.stream_sequence.append(sid)
            else:
                # Append window
                windows.append(new_window)
    
            # Always update the graph port
            port_data['basetime'] = self.basetime
            port_data['scheduleCycle'] = self.cycle_ns
            port_data.setdefault('schedule', [])
            # Only append **new windows** if not already in the list
            if new_window.to_dict() not in port_data['schedule']:
                port_data['schedule'].append(new_window.to_dict())
    
    def print_banana(self):
        for slot in self:
            if slot.port_id == 10:
                print("banana")


    def update_streams(self, streams, ports, port_id_to_name):
        """
        Add schedule-derived data to streams.
        Only streams that still have slots in the schedule are updated.
        Streams with no slots are left untouched.
        """
        for s_idx, stream in enumerate(streams):
            stream_slots = self.slots_by_stream(s_idx)
            if not stream_slots:
                # This stream is no longer scheduled (e.g. deleted from schedule).
                continue

            n_listeners = len(stream['lstnsrPssblPlcmnt'])
            stream_slots.sort()

            # talker data from first slot
            talker_slot = stream_slots[0]
            talker_port = port_id_to_name[talker_slot.port_id]
            talker_node = talker_port.split('.')[0]

            stream['offset'] = talker_slot.start_ns
            stream['tlkrPssblPlcmnt'] = [talker_node]
            stream['accLatency'] = [[] for _ in range(n_listeners)]
            stream['pathLists'] = [[talker_port] for _ in range(n_listeners)]

            for slot in stream_slots[1:]:
                port = port_id_to_name[slot.port_id]
                node_name = port.split('.')[0]
                pathLists = stream['pathLists']

                for l_idx in range(n_listeners):
                    pathList = pathLists[l_idx]
                    last = pathList[-1]
                    if ports[last]['destNode'] == node_name:
                        pathList.append(port)
                        stream['accLatency'][l_idx] = slot.end_ns - stream['offset']

            # finalize listener placement from each path
            for l_idx in range(n_listeners):
                pathList = stream['pathLists'][l_idx]
                last_port = pathList[-1]
                stream['lstnsrPssblPlcmnt'][l_idx] = [ports[last_port]['destNode']]
    
    @classmethod
    def from_compact_encoding(
        cls,
        compact_encoding,
        windLengths, stream_intervals,
        cycle_ns: Optional[int] = None,
        basetime:Optional[int] = 0,
        tt_queue_id = 0
    ):
        """
        Build a Schedule instance from a compact_encoding array.
    
        Parameters
        ----------
        compact_encoding : 2D numpy array (n_ports , n_streams)
            Nonzero value = start_time + 1
        windLengths : 2D numpy array (n_ports , n_streams)
            streams[i] = (period_ns, duration_ns)
        cycle_ns : int
            Schedule cycle
        """
        n_ports, n_streams = compact_encoding.shape

        schedule = cls(cycle_ns=cycle_ns, basetime=basetime)
        
        for port_idx in range(n_ports):
            for stream_id in range(n_streams):
                value = compact_encoding[port_idx, stream_id]
                if value != 0:
                    start_time = int(value) - 1
                    period_ns = stream_intervals[stream_id]
                    duration_ns = windLengths[port_idx, stream_id]
    
                    slot = Slot(
                        start_ns=start_time,
                        end_ns=int(start_time + duration_ns),
                        period_ns=int(period_ns),
                        stream_id=stream_id,
                        port_id=port_idx+1,
                        queue_id=tt_queue_id
                    )
                    schedule.append(slot)
        return schedule

    def to_compact_encoding(self, n_ports, n_streams):
        """
        Convert Schedule into compact_encoding array.
    
        Parameters
        ----------
        streams : list of tuples
            streams[i] = (period_ns, duration_ns)
    
        Returns
        -------
        compact_encoding : np.ndarray (n_ports x n_streams)
        """
        if not self:
            raise ValueError("Schedule is empty")
    
        compact = np.zeros((n_ports, n_streams), dtype=int)
    
        for slot in self:
            if slot.stream_id is None or slot.port_id is None: # unknown stream or port
                continue

            port_idx = slot.port_id -1
            stream_id = slot.stream_id
    
            if compact[port_idx, stream_id] != 0:
                raise ValueError(
                    f"Multiple slots found for port {port_idx}, stream {stream_id}. "
                    "Compact encoding allows only one."
                )
    
            compact[port_idx, stream_id] = int(slot.start_ns) + 1
    
        return compact

    def slots_by_port(self, port_id: int) -> List[Slot]:
        """
        Return all Slot definitions belonging to a given port.
        """
        return [slot for slot in self if slot.port_id == port_id]

    def slots_by_stream(self, stream_id: int) -> List[Slot]:
        """
        Return all Slot definitions belonging to a given stream.
        """
        return [slot for slot in self if slot.stream_id == stream_id]

    def plot(self, tick: float = 1.0, span: Optional[int] = None, label: bool = False,
             port_ids: Optional[List[int]] = None,
             stream_ids: Optional[List[int]] = None,
             cycles: int = 1,
             by_cycle: bool = False):
        """
        Visualize schedule occurrences with one axis per port.
    
        Parameters
        ----------
        tick : float
            Time unit scaling.
        span : int
            Max x-axis span in ticks. If None, uses cycle_ns * cycles.
        label : bool
            Annotate bars with stream ids.
        port_id : int or None
            If provided, plot only this port.
        cycles : int
            Number of schedule cycles to render when span is None.
        by_cycle : bool
            If True, wrap x-axis to one cycle and stack occurrences per cycle index.
            This makes delayed first appearances (e.g., on downstream ports) visible
            in the exact cycle they first occur.
    
        Returns
        -------
        fig
        axs
        """
        if self.validate_conflicts():
            print("Warning!!! schedule has conflicts, plot may be inaccurate.")

        if cycles < 1:
            raise ValueError("cycles must be >= 1")

        if span is None:
            span_ns = int(self.cycle_ns * cycles)
            span_ticks = int(span_ns / tick)
        else:
            # Backward-compatible behavior: span is interpreted as x-axis ticks.
            span_ticks = int(span)
            span_ns = int(span_ticks * tick)

        # Expand occurrences over the requested time span.
        occurrences = []
        for slot in self:
            occurrences.extend(slot.all_occurrences(span_ns))

         # Filter port if requested
        if port_ids is not None:
            occurrences = [o for o in occurrences if o.port_id in port_ids]
            
        # Filter streams if requested
        if stream_ids is not None:
            stream_ids = set(stream_ids)
            occurrences = [o for o in occurrences if o.stream_id in stream_ids]
            
        if not occurrences:
            fig, ax = plt.subplots()
            ax.set_title("Empty schedule")
            return fig, [ax]
    
        # Group occurrences by port.
        # Prefer ILP-like source->dest ordering when metadata is available.
        ports = {o.port_id for o in occurrences if o.port_id is not None}
        if hasattr(self, "port_sort_key"):
            ports = sorted(ports, key=lambda p: self.port_sort_key.get(p, (str(p), "")))
        elif hasattr(self, "port_id_to_name"):
            ports = sorted(ports, key=lambda p: self.port_id_to_name.get(p, str(p)))
        else:
            ports = sorted(ports)
    
        port_occ = {p: [] for p in ports}
        for o in occurrences:
            port_occ[o.port_id].append(o)
    
        n_ports = len(ports)
    
        fig, axs = plt.subplots(
            n_ports,
            1,
            figsize=(20, max(3, n_ports * 2)),
            sharex=True
        )
    
        if n_ports == 1:
            axs = [axs]
    
        # Stream coloring
        all_stream_ids = sorted({o.stream_id for o in occurrences if o.stream_id is not None})
        cmap = plt.get_cmap("plasma", len(all_stream_ids))
        stream_to_color = {sid: i for i, sid in enumerate(all_stream_ids)}
    
        for ax, port in zip(axs, ports):

            if not by_cycle:
                for s in port_occ[port]:
                    start_ticks = s.start_ns / tick
                    width_ticks = (s.end_ns - s.start_ns) / tick

                    color_idx = stream_to_color.get(s.stream_id, 0)
                    color = cmap(color_idx / max(1, len(all_stream_ids)-1))

                    ax.barh(
                        y=0,
                        width=width_ticks,
                        left=start_ticks,
                        height=0.8,
                        color=color,
                        edgecolor="white"
                    )

                    if label and s.stream_id is not None:
                        ax.text(
                            start_ticks + width_ticks/2,
                            0,
                            str(s.stream_id),
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=9
                        )

                if hasattr(self, "port_id_to_name"):
                    ax.set_ylabel(self.port_id_to_name.get(port, f"Port {port}"))
                else:
                    ax.set_ylabel(f"Port {port}")
                ax.set_yticks([])
            else:
                # Wrap to cycle-local x and stack bars on the cycle row where they occur.
                for s in port_occ[port]:
                    seg_start = int(s.start_ns)
                    seg_end = int(s.end_ns)

                    while seg_start < seg_end and seg_start < span_ns:
                        cyc_idx = int(seg_start // self.cycle_ns)
                        if cyc_idx >= cycles:
                            break

                        cyc_start = cyc_idx * self.cycle_ns
                        cyc_end = cyc_start + self.cycle_ns

                        part_start = max(seg_start, cyc_start)
                        part_end = min(seg_end, cyc_end)
                        if part_end <= part_start:
                            break

                        start_ticks = (part_start - cyc_start) / tick
                        width_ticks = (part_end - part_start) / tick

                        color_idx = stream_to_color.get(s.stream_id, 0)
                        color = cmap(color_idx / max(1, len(all_stream_ids)-1))

                        ax.barh(
                            y=cyc_idx,
                            width=width_ticks,
                            left=start_ticks,
                            height=0.8,
                            color=color,
                            edgecolor="black"
                        )

                        if label and s.stream_id is not None:
                            ax.text(
                                start_ticks + width_ticks/2,
                                cyc_idx,
                                str(s.stream_id),
                                ha="center",
                                va="center",
                                color="black",
                                fontsize=8
                            )

                        seg_start = part_end

                if hasattr(self, "port_id_to_name"):
                    ax.set_ylabel(self.port_id_to_name.get(port, f"Port {port}"))
                else:
                    ax.set_ylabel(f"Port {port}")
                ax.set_yticks(list(range(cycles)))
                ax.set_yticklabels([f"C{c+1}" for c in range(cycles)])

        # legend
        legend_handles = []
        
        for sid in all_stream_ids:
            color_idx = stream_to_color[sid]
            color = cmap(color_idx / max(1, len(all_stream_ids)-1))
        
            legend_handles.append(
                Patch(facecolor=color, edgecolor="black", label=f"Stream {sid}")
            )
        
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            title="Streams"
        )
    
        axs[-1].set_xlabel(f"Time (tick={tick})")
        if by_cycle:
            axs[-1].set_xlim(0, int(self.cycle_ns / tick))
        else:
            axs[-1].set_xlim(0, span_ticks)
    
        fig.suptitle(f"Schedule visualization (cycle={self.cycle_ns}, cycles={cycles})")
    
        plt.tight_layout()
        #plt.close(fig)
        plt.show()
    
        return fig, axs