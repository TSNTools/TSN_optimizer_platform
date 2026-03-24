from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluator import ScheduleEncodingType, TSNConstraintEvaluator
from schedule import Schedule


@dataclass
class HeuristicRunResult:
    """Result bundle for OF-FF execution."""

    before_compact: np.ndarray
    after_compact: np.ndarray
    before_schedule: Schedule
    after_schedule: Schedule
    departed_stream_ids: List[int]
    considered_stream_ids: List[int]
    moved_stream_ids: List[int]
    scheduler_violation_sum_before: int
    scheduler_violation_sum_after: int
    metrics_before: Dict[str, object]
    metrics_after: Dict[str, object]
    metrics_delta: Dict[str, object]
    debug_reports: List[Dict[str, object]]


def build_scheduling_only_evaluator(problem) -> TSNConstraintEvaluator:
    """Create evaluator with only scheduling checks in START_TIME encoding."""
    return TSNConstraintEvaluator(
        problem,
        check_routing=False,
        check_task_placement=False,
        scheduling_type=ScheduleEncodingType.START_TIME,
    )


def scheduler_violation_sum(sched_eval: TSNConstraintEvaluator, compact: np.ndarray) -> int:
    """Aggregate scheduling-only violations from evaluator.py checks."""
    per_stream = sched_eval.per_stream_all_violations(compact)
    return int(per_stream.sum())


def _gaps_per_port(compact: np.ndarray, wind_lengths: np.ndarray, cycle_ns: int) -> List[List[int]]:
    """Return free-gap lengths per port for one cycle."""
    n_ports, n_streams = compact.shape
    all_gaps: List[List[int]] = []

    for p in range(n_ports):
        intervals: List[Tuple[int, int]] = []
        for s in range(n_streams):
            v = int(compact[p, s])
            if v <= 0:
                continue
            start = v - 1
            dur = int(np.ceil(float(wind_lengths[p, s])))
            end = min(int(cycle_ns), start + max(0, dur))
            if end > start:
                intervals.append((start, end))

        if not intervals:
            all_gaps.append([int(cycle_ns)])
            continue

        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for st, en in intervals[1:]:
            last_st, last_en = merged[-1]
            if st <= last_en:
                merged[-1] = (last_st, max(last_en, en))
            else:
                merged.append((st, en))

        gaps: List[int] = []
        cur = 0
        for st, en in merged:
            if st > cur:
                gaps.append(int(st - cur))
            cur = max(cur, en)
        if cur < cycle_ns:
            gaps.append(int(cycle_ns - cur))

        all_gaps.append(gaps)

    return all_gaps


def fragmentation_metrics(
    compact: np.ndarray,
    wind_lengths: np.ndarray,
    cycle_ns: int,
    thresholds: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """
    Defragmentation-oriented metrics from compact encoding.

    Core metrics:
    - largest hole per port (+mean, +max)
    - count of holes >= threshold c (how many c-sized streams can be placed conservatively)
    - c-capacity by floor(hole/c)
    """
    gaps = _gaps_per_port(compact=compact, wind_lengths=wind_lengths, cycle_ns=int(cycle_ns))
    largest_per_port = np.asarray([max(gs) if gs else 0 for gs in gaps], dtype=int)
    flat_gaps = [g for gs in gaps for g in gs]

    out: Dict[str, object] = {
        "largest_hole_per_port": largest_per_port,
        "largest_hole_mean": float(np.mean(largest_per_port)) if largest_per_port.size else 0.0,
        "largest_hole_max": int(np.max(largest_per_port)) if largest_per_port.size else 0,
        "gap_count_total": int(len(flat_gaps)),
    }

    if thresholds:
        holes_ge_c_per_port: Dict[int, np.ndarray] = {}
        holes_ge_c_total: Dict[int, int] = {}
        c_capacity_total: Dict[int, int] = {}

        for c in thresholds:
            cc = int(c)
            per_port = np.asarray([sum(1 for g in gs if g >= cc) for gs in gaps], dtype=int)
            holes_ge_c_per_port[cc] = per_port
            holes_ge_c_total[cc] = int(np.sum(per_port))
            c_capacity_total[cc] = int(sum(sum(g // cc for g in gs) for gs in gaps))

        out["holes_ge_c_per_port"] = holes_ge_c_per_port
        out["holes_ge_c_total"] = holes_ge_c_total
        out["c_capacity_total"] = c_capacity_total

    return out


def _metrics_delta(before: Dict[str, object], after: Dict[str, object]) -> Dict[str, object]:
    """Numeric metrics delta: after - before."""
    delta: Dict[str, object] = {}
    for key in before.keys():
        if key not in after:
            continue
        b = before[key]
        a = after[key]

        if isinstance(b, (int, float, np.integer, np.floating)) and isinstance(a, (int, float, np.integer, np.floating)):
            delta[key] = float(a) - float(b)
        elif isinstance(b, np.ndarray) and isinstance(a, np.ndarray):
            delta[key] = a - b
        elif isinstance(b, dict) and isinstance(a, dict):
            sub: Dict[object, object] = {}
            for sk in b.keys():
                if sk not in a:
                    continue
                bv = b[sk]
                av = a[sk]
                if isinstance(bv, (int, float, np.integer, np.floating)) and isinstance(av, (int, float, np.integer, np.floating)):
                    sub[sk] = float(av) - float(bv)
                elif isinstance(bv, np.ndarray) and isinstance(av, np.ndarray):
                    sub[sk] = av - bv
            delta[key] = sub
    return delta


def infer_stream_ages_from_compact(compact: np.ndarray) -> np.ndarray:
    """
    Infer stream ages from compact encoding using earliest active start.

    Rule requested by user:
    - For each stream, take the minimum non-zero compact start over all ports.
    - Streams with lower earliest start are considered older.

    Returned ages follow this convention used by OF-FF selection:
    - Higher age value means older stream.
    """
    n_ports, n_streams = compact.shape
    earliest = np.full((n_streams,), np.inf, dtype=float)

    for s in range(n_streams):
        vals: List[int] = []
        for p in range(n_ports):
            v = int(compact[p, s])
            if v > 0:
                vals.append(v - 1)  # decode compact start
        if vals:
            earliest[s] = float(min(vals))

    # Rank streams by earliest start ascending (lower start => older => larger age).
    active_streams = [s for s in range(n_streams) if np.isfinite(earliest[s])]
    active_streams.sort(key=lambda s: earliest[s])

    ages = np.zeros((n_streams,), dtype=float)
    rank = len(active_streams)
    for s in active_streams:
        ages[s] = float(rank)
        rank -= 1

    return ages


def _row_labels_for_active_checks(sched_eval: TSNConstraintEvaluator, compact: np.ndarray) -> List[str]:
    """Build row labels matching per_stream_all_violations row order."""
    labels: List[str] = []
    for check_function in sched_eval.active_checks:
        arr = check_function(compact)
        row_count = int(arr.shape[0])
        name = check_function.__name__

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


def _violation_row_sums(sched_eval: TSNConstraintEvaluator, compact: np.ndarray) -> np.ndarray:
    """Row-wise sums of per-stream violations."""
    return sched_eval.per_stream_all_violations(compact).sum(axis=1)


def stream_windows_from_compact(
    compact: np.ndarray,
    instance,
    stream_id: int,
    n_ports: int,
    decode_compact_start: bool = True,
) -> List[Dict[str, int]]:
    """
    Gather per-port window information for one stream from compact encoding.
    
    """
    current: List[Dict[str, int]] = []
    j = int(stream_id)

    for i in range(0, int(n_ports)):
        if int(compact[i][j]) == 0:
            continue

        duration = 0
        for slot in instance.schedule:
            if slot.port_id == i + 1 and slot.stream_id == j:
                duration = int(slot.duration_ns)
                break

        start_time = int(compact[i][j])
        if decode_compact_start:
            start_time -= 1

        current.append(
            {
                "stream": j,
                "port": i,
                "start_time": start_time,
                "duration": duration,
            }
        )

    current.sort(key=lambda x: x["start_time"])
    return current


def _port_intervals_excluding_stream(
    compact: np.ndarray,
    wind_lengths: np.ndarray,
    cycle_ns: int,
    port_idx: int,
    stream_idx: int,
) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    n_streams = compact.shape[1]

    for s in range(n_streams):
        if s == stream_idx:
            continue
        encoded = int(compact[port_idx, s])
        if encoded <= 0:
            continue

        start = encoded - 1
        duration = int(np.ceil(float(wind_lengths[port_idx, s])))
        end = min(int(cycle_ns), start + max(0, duration))
        if start >= cycle_ns or end <= start:
            continue
        intervals.append((max(0, start), end))

    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged: List[Tuple[int, int]] = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _fits_interval(start: int, end: int, occupied: List[Tuple[int, int]]) -> bool:
    for occ_start, occ_end in occupied:
        if start < occ_end and end > occ_start:
            return False
    return True


def _port_candidate_starts(
    compact: np.ndarray,
    wind_lengths: np.ndarray,
    cycle_ns: int,
    stream_idx: int,
    port_idx: int,
    current_start: int,
    lower_bound: int,
) -> List[int]:
    """All feasible non-overlapping candidates in [lower_bound, current_start], sorted earliest first."""
    duration = int(np.ceil(float(wind_lengths[port_idx, stream_idx])))
    if duration <= 0:
        return [max(0, int(lower_bound))]

    lb = max(0, int(lower_bound))
    ub = int(current_start)
    if lb > ub:
        return [current_start]

    occupied = _port_intervals_excluding_stream(
        compact=compact,
        wind_lengths=wind_lengths,
        cycle_ns=cycle_ns,
        port_idx=port_idx,
        stream_idx=stream_idx,
    )

    candidates = {lb}
    for _, end in occupied:
        if lb <= end <= ub:
            candidates.add(int(end))

    valid: List[int] = []
    for cand in sorted(candidates):
        end = cand + duration
        if end > cycle_ns:
            continue
        if _fits_interval(cand, end, occupied):
            valid.append(int(cand))

    if not valid:
        return [current_start]
    if current_start not in valid:
        valid.append(int(current_start))
    return sorted(set(valid))


def _move_stream_first_fit(
    compact: np.ndarray,
    sched_eval: TSNConstraintEvaluator,
    wind_lengths: np.ndarray,
    cycle_ns: int,
    stream_idx: int,
    baseline_violation_sum: int,
    all_hops_costs: np.ndarray,
    upstream_ports: np.ndarray,
    downstream_ports: np.ndarray,
) -> Tuple[np.ndarray, bool, Dict[str, object], np.ndarray]:
    """
    Try first-fit move for one stream.

    For each active port of this stream, place the stream window at the earliest
    feasible hole in [0, current_start] on that port. Accept only non-worsening
    scheduling violation sum.
    """
    path_ports = np.where(compact[:, stream_idx] > 0)[0]
    report: Dict[str, object] = {
        "stream": int(stream_idx),
        "port_attempts": [],
        "candidate_violation_sum": int(baseline_violation_sum),
        "accepted": False,
        "reason": "no_active_ports",
    }
    if path_ports.size == 0:
        return compact, False, report, compact

    candidate = compact.copy()
    moved = False

    path_set = set(int(p) for p in path_ports.tolist())
    preds: Dict[int, List[Tuple[int, int]]] = {int(p): [] for p in path_ports.tolist()}
    for up, down in zip(upstream_ports.tolist(), downstream_ports.tolist()):
        if up in path_set and down in path_set:
            gap = int(np.ceil(float(all_hops_costs[int(up), int(stream_idx)])))
            preds[int(down)].append((int(up), max(0, gap)))

    # Topological-like order; fallback to current-time sort.
    ports = [int(p) for p in path_ports.tolist()]
    indegree = {p: 0 for p in ports}
    edges: Dict[int, List[int]] = {p: [] for p in ports}
    for down, p_list in preds.items():
        for up, _ in p_list:
            edges[up].append(down)
            indegree[down] += 1

    ready = sorted([p for p in ports if indegree[p] == 0], key=lambda p: int(candidate[p, stream_idx]) - 1)
    ordered_ports: List[int] = []
    while ready:
        p = ready.pop(0)
        ordered_ports.append(p)
        for nxt in edges[p]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
                ready.sort(key=lambda x: int(candidate[x, stream_idx]) - 1)
    if len(ordered_ports) < len(ports):
        rest = [p for p in ports if p not in set(ordered_ports)]
        rest.sort(key=lambda p: int(candidate[p, stream_idx]) - 1)
        ordered_ports.extend(rest)

    new_starts: Dict[int, int] = {}

    for p in ordered_ports:
        cur_start = int(candidate[p, stream_idx]) - 1
        if cur_start <= 0:
            report["port_attempts"].append(
                {
                    "port": int(p),
                    "current_start": int(cur_start),
                    "first_fit_start": int(cur_start),
                    "lower_bound": int(cur_start),
                    "had_earlier_hole": False,
                }
            )
            new_starts[int(p)] = int(cur_start)
            continue

        lb = 0
        for up, min_gap in preds.get(int(p), []):
            up_start = new_starts.get(int(up), int(candidate[int(up), stream_idx]) - 1)
            lb = max(lb, int(up_start + min_gap))

        candidates = _port_candidate_starts(
            compact=candidate,
            wind_lengths=wind_lengths,
            cycle_ns=cycle_ns,
            stream_idx=stream_idx,
            port_idx=int(p),
            current_start=cur_start,
            lower_bound=lb,
        )

        # First possible fit: earliest candidate that keeps violations non-worsening.
        new_start = cur_start
        checked: List[int] = []
        for cand in candidates:
            checked.append(int(cand))
            trial = candidate.copy()
            trial[int(p), int(stream_idx)] = int(cand + 1)
            if scheduler_violation_sum(sched_eval, trial) <= baseline_violation_sum:
                new_start = int(cand)
                break

        report["port_attempts"].append(
            {
                "port": int(p),
                "current_start": int(cur_start),
                "first_fit_start": int(new_start),
                "lower_bound": int(lb),
                "had_earlier_hole": bool(new_start < cur_start),
                "checked_candidates": checked,
            }
        )
        if new_start < cur_start:
            candidate[p, stream_idx] = int(new_start + 1)
            moved = True
        new_starts[int(p)] = int(new_start)

    if not moved:
        report["candidate_violation_sum"] = int(baseline_violation_sum)
        report["accepted"] = False
        report["reason"] = "no_earlier_hole"
        return compact, False, report, compact

    cand_viols = scheduler_violation_sum(sched_eval, candidate)
    report["candidate_violation_sum"] = int(cand_viols)
    if cand_viols <= baseline_violation_sum:
        report["accepted"] = True
        report["reason"] = "accepted"
        return candidate, True, report, candidate
    report["accepted"] = False
    report["reason"] = "violations_worsened"
    return compact, False, report, candidate


def choose_departures(
    compact: np.ndarray,
    p: int,
    stream_ages: Optional[Sequence[float]] = None,
    departed_stream_ids: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Choose stream IDs to depart before OF-FF.

    Priority:
    1) Use provided departed_stream_ids (filtered to active streams)
    2) Else select oldest active streams from stream_ages
    """
    active = np.where(np.any(compact > 0, axis=0))[0].tolist()
    active_set = set(active)

    if p <= 0 or not active:
        return []

    if departed_stream_ids is not None:
        selected = [int(s) for s in departed_stream_ids if int(s) in active_set]
        return selected[: int(p)]

    if stream_ages is None:
        # Default deterministic order: larger stream index treated as older.
        stream_ages = np.arange(compact.shape[1], dtype=float)

    ordered = sorted(active, key=lambda s: float(stream_ages[s]), reverse=True)
    return ordered[: int(p)]


def compact_after_departures(
    compact: np.ndarray,
    wind_lengths: np.ndarray,
    stream_intervals: np.ndarray,
    cycle_ns: int,
    departure_stream_ids: Sequence[int],
) -> np.ndarray:
    """Apply departures using Schedule.remove_stream and re-export compact."""
    schedule = Schedule.from_compact_encoding(
        compact_encoding=np.asarray(compact, dtype=int),
        windLengths=np.asarray(wind_lengths),
        stream_intervals=np.asarray(stream_intervals, dtype=int),
        cycle_ns=int(cycle_ns),
    )
    if departure_stream_ids:
        schedule.remove_stream([int(s) for s in departure_stream_ids])
    return schedule.to_compact_encoding(compact.shape[0], compact.shape[1])


def of_ff(
    problem,
    instance,
    p: int,
    n: int,
    stream_ages: Optional[Sequence[float]] = None,
    departed_stream_ids: Optional[Sequence[int]] = None,
    frame_size_thresholds: Optional[Sequence[int]] = (500, 1000, 2000, 5000),
) -> HeuristicRunResult:
    """
    Run OF-FF(p,n) heuristic end-to-end:
    - Build compact encoding from instance.schedule
    - Apply p departures (Schedule.remove_stream)
    - Move oldest n streams with first-fit if scheduling does not worsen
    - Convert resulting compact encoding back to Schedule
    """
    with problem.use_instance(instance):
        n_ports = int(problem.n_ports)
        n_streams = int(problem.n_streams)
        cycle_ns = int(instance.schedule.cycle_ns)

        compact = instance.schedule.to_compact_encoding(n_ports, n_streams)

        if stream_ages is None:
            stream_ages_arr = infer_stream_ages_from_compact(compact)
        else:
            stream_ages_arr = np.asarray(stream_ages, dtype=float)

        sched_eval = build_scheduling_only_evaluator(problem)

        before = compact.copy()

        # Step 1: departures.
        selected_departures = choose_departures(
            compact=before,
            p=int(p),
            stream_ages=stream_ages_arr,
            departed_stream_ids=departed_stream_ids,
        )

        work = compact_after_departures(
            compact=before,
            wind_lengths=sched_eval.windLengths,
            stream_intervals=sched_eval.stream_intervals,
            cycle_ns=cycle_ns,
            departure_stream_ids=selected_departures,
        )

        # Step 2: oldest n active streams after departures.
        active_after_departure = np.where(np.any(work > 0, axis=0))[0]
        ordered_oldest = sorted(
            active_after_departure.tolist(),
            key=lambda s: float(stream_ages_arr[s]),
            reverse=True,
        )
        considered = ordered_oldest[: max(0, int(n))]

        # Step 3: first-fit move with scheduling safety gate.
        moved_streams: List[int] = []
        baseline = scheduler_violation_sum(sched_eval, work)
        debug_reports: List[Dict[str, object]] = []
        row_labels = _row_labels_for_active_checks(sched_eval, work)
        baseline_row_sums = _violation_row_sums(sched_eval, work)

        for stream_idx in considered:
            candidate, moved, report, attempted_candidate = _move_stream_first_fit(
                compact=work,
                sched_eval=sched_eval,
                wind_lengths=np.asarray(sched_eval.windLengths),
                cycle_ns=cycle_ns,
                stream_idx=int(stream_idx),
                baseline_violation_sum=baseline,
                all_hops_costs=np.asarray(sched_eval.all_hops_costs),
                upstream_ports=np.asarray(sched_eval._upstream_ports, dtype=int),
                downstream_ports=np.asarray(sched_eval._downstream_ports, dtype=int),
            )

            candidate_row_sums = _violation_row_sums(sched_eval, attempted_candidate)
            deltas = candidate_row_sums - baseline_row_sums
            report["per_check_violation_deltas"] = {
                label: int(delta) for label, delta in zip(row_labels, deltas.tolist())
            }
            report["baseline_violation_sum"] = int(baseline)
            report["candidate_violation_sum"] = int(scheduler_violation_sum(sched_eval, attempted_candidate))
            debug_reports.append(report)

            if moved:
                work = candidate
                baseline = scheduler_violation_sum(sched_eval, work)
                baseline_row_sums = _violation_row_sums(sched_eval, work)
                moved_streams.append(int(stream_idx))

        before_viol = scheduler_violation_sum(sched_eval, before)
        after_viol = scheduler_violation_sum(sched_eval, work)

        metrics_before = fragmentation_metrics(
            compact=before,
            wind_lengths=np.asarray(sched_eval.windLengths),
            cycle_ns=cycle_ns,
            thresholds=frame_size_thresholds,
        )
        metrics_after = fragmentation_metrics(
            compact=work,
            wind_lengths=np.asarray(sched_eval.windLengths),
            cycle_ns=cycle_ns,
            thresholds=frame_size_thresholds,
        )
        metrics_delta = _metrics_delta(metrics_before, metrics_after)

        before_schedule = Schedule.from_compact_encoding(
            compact_encoding=before,
            windLengths=sched_eval.windLengths,
            stream_intervals=sched_eval.stream_intervals,
            cycle_ns=cycle_ns,
        )
        after_schedule = Schedule.from_compact_encoding(
            compact_encoding=work,
            windLengths=sched_eval.windLengths,
            stream_intervals=sched_eval.stream_intervals,
            cycle_ns=cycle_ns,
        )

    return HeuristicRunResult(
        before_compact=before,
        after_compact=work,
        before_schedule=before_schedule,
        after_schedule=after_schedule,
        departed_stream_ids=[int(s) for s in selected_departures],
        considered_stream_ids=[int(s) for s in considered],
        moved_stream_ids=moved_streams,
        scheduler_violation_sum_before=before_viol,
        scheduler_violation_sum_after=after_viol,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        metrics_delta=metrics_delta,
        debug_reports=debug_reports,
    )


def of_ff_with_current_info(
    problem,
    instance,
    p: int,
    n: int,
    stream_ages: Optional[Sequence[float]] = None,
    departed_stream_ids: Optional[Sequence[int]] = None,
    frame_size_thresholds: Optional[Sequence[int]] = (500, 1000, 2000, 5000),
) -> Tuple[HeuristicRunResult, List[List[Dict[str, int]]]]:
    """
    Run OF-FF and also return "current" per-stream info for selected streams.

    The returned list contains one item per considered stream, where each item
    is the list of dicts in the same style as the notebook snippet.
    """
    run_result = of_ff(
        problem=problem,
        instance=instance,
        p=p,
        n=n,
        stream_ages=stream_ages,
        departed_stream_ids=departed_stream_ids,
        frame_size_thresholds=frame_size_thresholds,
    )

    current_info: List[List[Dict[str, int]]] = []

    with problem.use_instance(instance):
        n_ports = int(problem.n_ports)
        compact = run_result.before_compact
        for stream_id in run_result.considered_stream_ids:
            current_info.append(
                stream_windows_from_compact(
                    compact=compact,
                    instance=instance,
                    stream_id=int(stream_id),
                    n_ports=n_ports,
                    decode_compact_start=True,
                )
            )

    return run_result, current_info


def print_run_summary(result: HeuristicRunResult, max_thresholds: int = 10) -> None:
    """Pretty-print a compact summary of OF-FF run outcomes and metrics."""
    print("=== OF-FF Summary ===")
    print("Departed streams:", result.departed_stream_ids)
    print("Considered streams:", result.considered_stream_ids)
    print("Moved streams:", result.moved_stream_ids)
    print(
        "Violation sum before -> after:",
        result.scheduler_violation_sum_before,
        "->",
        result.scheduler_violation_sum_after,
    )

    mb = result.metrics_before
    ma = result.metrics_after
    md = result.metrics_delta

    if "largest_hole_mean" in mb and "largest_hole_mean" in ma:
        print(
            "Largest hole mean before -> after:",
            mb["largest_hole_mean"],
            "->",
            ma["largest_hole_mean"],
            "(delta:",
            md.get("largest_hole_mean", "n/a"),
            ")",
        )
    if "largest_hole_max" in mb and "largest_hole_max" in ma:
        print(
            "Largest hole max before -> after:",
            mb["largest_hole_max"],
            "->",
            ma["largest_hole_max"],
            "(delta:",
            md.get("largest_hole_max", "n/a"),
            ")",
        )

    holes_before = mb.get("holes_ge_c_total", {})
    holes_after = ma.get("holes_ge_c_total", {})
    cap_before = mb.get("c_capacity_total", {})
    cap_after = ma.get("c_capacity_total", {})

    if holes_before and holes_after:
        print("\nHoles >= c (before -> after):")
        thresholds = sorted(list(holes_before.keys()))[: max(0, int(max_thresholds))]
        for c in thresholds:
            print(f"  c={c}: {holes_before[c]} -> {holes_after.get(c, 'n/a')}")

    if cap_before and cap_after:
        print("\nC-capacity total (before -> after):")
        thresholds = sorted(list(cap_before.keys()))[: max(0, int(max_thresholds))]
        for c in thresholds:
            print(f"  c={c}: {cap_before[c]} -> {cap_after.get(c, 'n/a')}")
