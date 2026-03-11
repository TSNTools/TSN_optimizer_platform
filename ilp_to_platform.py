"""
Convert ILP scheduling outputs into platform Schedule objects.

This module is intentionally independent from the rctsn package internals.
It only relies on exported artifacts (pickles) and the platform's
`schedule.py` definitions.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from schedule import Schedule, Slot


PortTuple = Tuple[str, str]


@dataclass
class ILPArtifacts:
    """Container for raw ILP artifacts loaded from one run folder."""

    variables_a: Dict[int, float]
    variables_z: Dict[Tuple[int, int], float]
    variables_x: Dict[Tuple[int, int, int, PortTuple], float]
    stream_dict: Dict[int, Dict[int, Dict[str, Any]]]


@dataclass
class ILPConversionResult:
    """Result object returned by ILP->Schedule conversion."""

    schedule: Schedule
    cycle_ns: int
    active_streams: List[int]
    selected_paths: Dict[int, int]
    port_tuple_to_port_id: Dict[PortTuple, int]
    port_id_to_port_tuple: Dict[int, PortTuple]
    dropped_entries: int
    wrapped_entries: int


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_ilp_artifacts(run_dir: str | Path) -> ILPArtifacts:
    """
    Load ILP output pickles from a run output folder.

    Expected files:
    - variables_a.pkl
    - variables_z.pkl
    - variables_x.pkl
    - stream_dict.pkl
    """
    base = Path(run_dir)
    return ILPArtifacts(
        variables_a=_load_pickle(base / "variables_a.pkl"),
        variables_z=_load_pickle(base / "variables_z.pkl"),
        variables_x=_load_pickle(base / "variables_x.pkl"),
        stream_dict=_load_pickle(base / "stream_dict.pkl"),
    )


def _lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // math.gcd(a, b)


def _lcmm(values: List[int]) -> int:
    if not values:
        return 1
    out = values[0]
    for v in values[1:]:
        out = _lcm(out, v)
    return max(1, out)


def extract_active_streams(
    variables_a: Dict[int, float], activation_threshold: float = 0.5
) -> List[int]:
    """Return sorted list of streams with a[s] > threshold."""
    active = [int(s) for s, val in variables_a.items() if float(val) > activation_threshold]
    return sorted(active)


def extract_selected_paths(
    variables_z: Dict[Tuple[int, int], float],
    active_streams: List[int],
    activation_threshold: float = 0.5,
    strict: bool = True,
) -> Dict[int, int]:
    """
    Return selected path per active stream from z[(s,u)].

    In strict mode, each active stream must have exactly one selected path.
    """
    selected: Dict[int, int] = {}
    for s in active_streams:
        paths = [u for (ss, u), val in variables_z.items() if int(ss) == s and float(val) > activation_threshold]
        if len(paths) == 1:
            selected[s] = int(paths[0])
            continue
        if strict:
            raise ValueError(
                f"Stream {s} has {len(paths)} selected paths in z (expected 1)."
            )
        if len(paths) > 1:
            selected[s] = int(sorted(paths)[0])
    return selected


def _path_period_ns(
    stream_dict: Dict[int, Dict[int, Dict[str, Any]]],
    stream_id: int,
    path_id: int,
    time_scale_to_ns: int,
) -> int:
    period = stream_dict[stream_id][path_id]["period"]
    return int(round(float(period) * time_scale_to_ns))


def _duration_ns_for_port(
    path_data: Dict[str, Any],
    port: PortTuple,
    duration_mode: str,
    time_scale_to_ns: int,
) -> int:
    times = path_data.get("times", {}).get(port, {})
    deviations = path_data.get("deviations", {}).get(port, {})
    gamma = float(path_data.get("gamma", 1.0))

    base = float(sum(times.values()))
    dev = float(sum(deviations.values()))

    if duration_mode == "best_case":
        duration = base
    elif duration_mode == "worst_case":
        duration = base + dev
    elif duration_mode == "robust":
        duration = base + gamma * dev
    else:
        raise ValueError("duration_mode must be one of {'best_case', 'worst_case', 'robust'}.")

    return max(1, int(round(duration * time_scale_to_ns)))


def _build_port_mapping(selected_port_tuples: List[PortTuple]) -> Tuple[Dict[PortTuple, int], Dict[int, PortTuple]]:
    unique_ports = sorted(set(selected_port_tuples), key=lambda x: (str(x[0]), str(x[1])))
    port_tuple_to_port_id = {p: i + 1 for i, p in enumerate(unique_ports)}
    port_id_to_port_tuple = {v: k for k, v in port_tuple_to_port_id.items()}
    return port_tuple_to_port_id, port_id_to_port_tuple


def _infer_cycle_ns(
    stream_dict: Dict[int, Dict[int, Dict[str, Any]]],
    selected_paths: Dict[int, int],
    time_scale_to_ns: int,
) -> int:
    periods = [
        _path_period_ns(stream_dict, s, u, time_scale_to_ns)
        for s, u in selected_paths.items()
    ]
    return _lcmm(periods)


def convert_ilp_artifacts_to_schedule(
    artifacts: ILPArtifacts,
    time_scale_to_ns: int = 1000,
    duration_mode: str = "robust",
    strict: bool = True,
    activation_threshold: float = 0.5,
    queue_id: int = 0,
    cycle_ns: Optional[int] = None,
    compact_periodic: bool = False,
) -> ILPConversionResult:
    """
    Convert ILP pickles to a platform Schedule object.

    Parameters
    ----------
    time_scale_to_ns:
        Multiplicative factor to convert ILP time unit into platform ns.
        Use 1000 when ILP times are in microseconds.
    duration_mode:
        {'best_case', 'worst_case', 'robust'}.
    compact_periodic:
        If True, attempts one periodic Slot per (stream, path, port) by
        checking repetition consistency. If False, keeps explicit repetition
        slots with period=cycle_ns.
    """
    variables_a = artifacts.variables_a
    variables_z = artifacts.variables_z
    variables_x = artifacts.variables_x
    stream_dict = artifacts.stream_dict

    active_streams = extract_active_streams(
        variables_a, activation_threshold=activation_threshold
    )
    selected_paths = extract_selected_paths(
        variables_z,
        active_streams=active_streams,
        activation_threshold=activation_threshold,
        strict=strict,
    )

    if cycle_ns is None:
        cycle_ns = _infer_cycle_ns(
            stream_dict=stream_dict,
            selected_paths=selected_paths,
            time_scale_to_ns=time_scale_to_ns,
        )

    # Gather relevant rows (only active + selected path).
    rows = []
    dropped_entries = 0
    all_ports: List[PortTuple] = []
    for key, start_val in variables_x.items():
        if not isinstance(key, tuple) or len(key) != 4:
            dropped_entries += 1
            continue
        s, u, rep, port = key
        s = int(s)
        u = int(u)
        rep = int(rep)
        if s not in selected_paths:
            continue
        if selected_paths[s] != u:
            continue
        if s not in stream_dict or u not in stream_dict[s]:
            dropped_entries += 1
            continue

        path_data = stream_dict[s][u]
        if port not in path_data.get("ports", []):
            # x variable not aligned with stream_dict path definition.
            dropped_entries += 1
            continue

        start_ns = int(round(float(start_val) * time_scale_to_ns))
        dur_ns = _duration_ns_for_port(
            path_data=path_data,
            port=port,
            duration_mode=duration_mode,
            time_scale_to_ns=time_scale_to_ns,
        )
        period_ns = _path_period_ns(
            stream_dict=stream_dict,
            stream_id=s,
            path_id=u,
            time_scale_to_ns=time_scale_to_ns,
        )
        rows.append((s, u, rep, port, start_ns, dur_ns, period_ns))
        all_ports.append(port)

    port_tuple_to_port_id, port_id_to_port_tuple = _build_port_mapping(all_ports)

    schedule = Schedule(cycle_ns=int(cycle_ns))
    wrapped_entries = 0

    if compact_periodic:
        # Group by (stream, path, port): try compressing repetitions to one periodic slot.
        grouped: Dict[Tuple[int, int, PortTuple], List[Tuple[int, int, int, int]]] = {}
        for s, u, rep, port, start_ns, dur_ns, period_ns in rows:
            grouped.setdefault((s, u, port), []).append((rep, start_ns, dur_ns, period_ns))

        for (s, _u, port), vals in grouped.items():
            vals.sort(key=lambda x: x[0])  # by repetition index
            reps = [v[0] for v in vals]
            starts = [v[1] for v in vals]
            dur = vals[0][2]
            period = vals[0][3]

            rep0 = reps[0]
            start0 = starts[0]
            periodic_ok = True
            for rep_i, start_i in zip(reps, starts):
                expected = start0 + (rep_i - rep0) * period
                if abs(start_i - expected) > 1:
                    periodic_ok = False
                    break

            if periodic_ok:
                wrapped_start = start0 % cycle_ns
                if wrapped_start != start0:
                    wrapped_entries += 1
                schedule.append(
                    Slot(
                        start_ns=int(wrapped_start),
                        end_ns=int(wrapped_start + dur),
                        period_ns=int(period),
                        stream_id=s,
                        port_id=port_tuple_to_port_id[port],
                        queue_id=queue_id,
                    )
                )
            else:
                # Fallback to explicit repetition slots for this (stream,port).
                for _rep, start_i, dur_i, _period in vals:
                    wrapped_start = start_i % cycle_ns
                    if wrapped_start != start_i:
                        wrapped_entries += 1
                    schedule.append(
                        Slot(
                            start_ns=int(wrapped_start),
                            end_ns=int(wrapped_start + dur_i),
                            period_ns=int(cycle_ns),
                            stream_id=s,
                            port_id=port_tuple_to_port_id[port],
                            queue_id=queue_id,
                        )
                    )
    else:
        # Exact per-repetition representation (one slot per x-row).
        for s, _u, _rep, port, start_ns, dur_ns, _period_ns in rows:
            wrapped_start = start_ns % cycle_ns
            if wrapped_start != start_ns:
                wrapped_entries += 1
            schedule.append(
                Slot(
                    start_ns=int(wrapped_start),
                    end_ns=int(wrapped_start + dur_ns),
                    period_ns=int(cycle_ns),
                    stream_id=s,
                    port_id=port_tuple_to_port_id[port],
                    queue_id=queue_id,
                )
            )

    return ILPConversionResult(
        schedule=schedule,
        cycle_ns=int(cycle_ns),
        active_streams=active_streams,
        selected_paths=selected_paths,
        port_tuple_to_port_id=port_tuple_to_port_id,
        port_id_to_port_tuple=port_id_to_port_tuple,
        dropped_entries=dropped_entries,
        wrapped_entries=wrapped_entries,
    )


def convert_ilp_run_to_schedule(
    run_dir: str | Path,
    **kwargs,
) -> ILPConversionResult:
    """
    Convenience wrapper: load pickles from run_dir and convert.
    """
    artifacts = load_ilp_artifacts(run_dir)
    return convert_ilp_artifacts_to_schedule(artifacts, **kwargs)


def summarize_conversion(result: ILPConversionResult) -> Dict[str, int]:
    """
    Small summary useful for notebook/testing output.
    """
    return {
        "n_slots": len(result.schedule),
        "n_active_streams": len(result.active_streams),
        "n_selected_paths": len(result.selected_paths),
        "n_ports_used": len(result.port_tuple_to_port_id),
        "cycle_ns": int(result.cycle_ns),
        "dropped_entries": int(result.dropped_entries),
        "wrapped_entries": int(result.wrapped_entries),
    }
