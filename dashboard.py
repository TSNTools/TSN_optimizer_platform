from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from heuristic import (
    infer_stream_ages_from_compact,
    of_ff_with_current_info,
)
from tsn_scheduling_problem import TSNScheduling


def parse_thresholds(text: str) -> Tuple[int, ...]:
    values: List[int] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        values.append(int(p))
    if not values:
        return (64, 128, 256)
    return tuple(values)


def parse_percentages(text: str, n_items: int) -> Tuple[float, ...]:
    values: List[float] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        values.append(float(p))
    if not values:
        if n_items <= 0:
            return tuple()
        return tuple([100.0 / n_items] * n_items)
    if len(values) != n_items:
        if n_items <= 0:
            return tuple()
        return tuple([100.0 / n_items] * n_items)
    return tuple(values)


def parse_int_list(text: str) -> List[int]:
    if not text.strip():
        return []
    out: List[int] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def per_stream_start_map(compact: np.ndarray) -> Dict[int, Dict[int, int]]:
    out: Dict[int, Dict[int, int]] = {}
    n_ports, n_streams = compact.shape
    for s in range(n_streams):
        per_port: Dict[int, int] = {}
        for p in range(n_ports):
            v = int(compact[p, s])
            if v > 0:
                per_port[p + 1] = v - 1
        if per_port:
            out[s] = per_port
    return out


def moved_stream_changes_df(before_compact: np.ndarray, after_compact: np.ndarray, moved: List[int]) -> pd.DataFrame:
    before = per_stream_start_map(before_compact)
    after = per_stream_start_map(after_compact)
    rows: List[Dict[str, int]] = []

    for s in moved:
        ports = sorted(set(before.get(s, {}).keys()) | set(after.get(s, {}).keys()))
        for p in ports:
            b = before.get(s, {}).get(p)
            a = after.get(s, {}).get(p)
            if b is None or a is None:
                continue
            rows.append(
                {
                    "stream": int(s),
                    "port": int(p),
                    "start_before_ns": int(b),
                    "start_after_ns": int(a),
                    "shift_ns": int(a - b),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["stream", "port", "start_before_ns", "start_after_ns", "shift_ns"])
    return pd.DataFrame(rows).sort_values(["stream", "port"]).reset_index(drop=True)


def thresholds_table(result) -> pd.DataFrame:
    before = result.metrics_before.get("usable_holes_total", result.metrics_before.get("holes_ge_c_total", {}))
    after = result.metrics_after.get("usable_holes_total", result.metrics_after.get("holes_ge_c_total", {}))

    rows: List[Dict[str, float]] = []
    for c in sorted(before.keys()):
        rows.append(
            {
                "frame_size_B": int(c),
                "usable_holes_before": int(before[c]),
                "usable_holes_after": int(after.get(c, 0)),
                "usable_holes_delta": int(after.get(c, 0)) - int(before[c]),
            }
        )
    return pd.DataFrame(rows)



def main() -> None:
    st.set_page_config(page_title="OF-FF Heuristic Dashboard", layout="wide")
    st.markdown(
        """
        <style>
        .big-note {font-size: 1.02rem;}
        [data-testid="stDataFrame"] div {font-size: 15px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("OF-FF Heuristic Dashboard")
    st.markdown(
        '<div class="big-note">'
        "Run Older-First First-Possible-Fit defragmentation: remove <b>p</b> streams first, then try to move the oldest <b>n</b> streams while keeping scheduling feasible."
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption("Indexing note: stream IDs are 0-based, port IDs are 1-based.")

    with st.sidebar:
        st.header("Inputs")
        default_cfg = "uc_config.cfg"
        config_path = st.text_input("Config file", value=default_cfg)

        json_files = sorted([p.name for p in Path(".").glob("*.json")])
        default_json = "instance2.json" if "instance2.json" in json_files else (json_files[0] if json_files else "")
        selected_json = st.selectbox("Usecase JSON", options=json_files, index=json_files.index(default_json) if default_json in json_files else 0)

        p = st.number_input("Departures p", min_value=0, value=10, step=1)
        n = st.number_input("Oldest streams n", min_value=0, value=5, step=1)
        thr_text = st.text_input("Frame sizes (bytes, comma-separated)", value="64,128,256")
        mix_text = st.text_input("Frame mix percentages (same order, comma-separated)", value="50,25,25")
        incoming_stream_count = st.number_input("Incoming stream count", min_value=1, value=100, step=1)

        run_clicked = st.button("Run OF-FF", type="primary")

    if "last_run" not in st.session_state:
        st.session_state["last_run"] = None

    if run_clicked:
        thresholds = parse_thresholds(thr_text)
        mix_percentages = parse_percentages(mix_text, len(thresholds))

        try:
            problem = TSNScheduling(config_path)
            instance = problem.import_scheduling_usecase(selected_json)

            result, current_info = of_ff_with_current_info(
                problem=problem,
                instance=instance,
                p=int(p),
                n=int(n),
                stream_ages=None,
                departed_stream_ids=None,
                frame_size_thresholds=thresholds,
                frame_mix_percentages=mix_percentages,
                incoming_stream_count=int(incoming_stream_count),
            )

            with problem.use_instance(instance):
                compact_now = instance.schedule.to_compact_encoding(problem.n_ports, problem.n_streams)
                inferred_ages = infer_stream_ages_from_compact(compact_now)

            st.session_state["last_run"] = {
                "result": result,
                "current_info": current_info,
                "inferred_ages": inferred_ages,
                "selected_json": selected_json,
                "p": int(p),
                "n": int(n),
                "frame_sizes": thresholds,
                "mix_percentages": mix_percentages,
                "incoming_stream_count": int(incoming_stream_count),
            }
        except Exception as exc:
            st.error(f"Run failed: {exc}")
            st.exception(exc)

    run_data = st.session_state.get("last_run")
    if run_data is None:
        st.info("Choose parameters on the left, then click 'Run OF-FF'.")
        return

    result = run_data["result"]
    current_info = run_data["current_info"]
    inferred_ages = run_data["inferred_ages"]

    st.caption(
        "Showing last run: "
        f"json={run_data['selected_json']}, p={run_data['p']}, n={run_data['n']}, "
        f"frame_sizes={list(run_data['frame_sizes'])}, mix={list(run_data['mix_percentages'])}, "
        f"incoming={run_data['incoming_stream_count']}"
    )
    tabs = st.tabs(["Run Summary", "Metrics", "Moved Changes & Ages", "Plots", "Debug & Current Info"])

    with tabs[0]:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Moved Streams", len(result.moved_stream_ids))
            c2.metric("Violations (before)", result.scheduler_violation_sum_before)
            c3.metric("Violations (after)", result.scheduler_violation_sum_after)
            c4.metric(
                "Largest Hole Mean Delta",
                f"{result.metrics_delta.get('largest_hole_mean', 0.0):.2f}",
            )
            c5, c6 = st.columns(2)
            sbr_before = float(result.metrics_before.get("service_block_ratio", np.nan))
            sbr_after = float(result.metrics_after.get("service_block_ratio", np.nan))
            if np.isfinite(sbr_before):
                c5.metric("Service Block Ratio (before)", f"{100.0 * sbr_before:.2f}%")
            if np.isfinite(sbr_after):
                c6.metric("Service Block Ratio (after)", f"{100.0 * sbr_after:.2f}%")
            st.write("Departed streams:", result.departed_stream_ids)
            st.write("Considered streams:", result.considered_stream_ids)
            st.write("Moved streams:", result.moved_stream_ids)

    metrics_table = pd.DataFrame(
            [
                {
                    "metric": "largest_hole_mean",
                    "before": result.metrics_before.get("largest_hole_mean", np.nan),
                    "after": result.metrics_after.get("largest_hole_mean", np.nan),
                    "delta": result.metrics_delta.get("largest_hole_mean", np.nan),
                },
                {
                    "metric": "largest_hole_max",
                    "before": result.metrics_before.get("largest_hole_max", np.nan),
                    "after": result.metrics_after.get("largest_hole_max", np.nan),
                    "delta": result.metrics_delta.get("largest_hole_max", np.nan),
                },
                {
                    "metric": "gap_count_total",
                    "before": result.metrics_before.get("gap_count_total", np.nan),
                    "after": result.metrics_after.get("gap_count_total", np.nan),
                    "delta": result.metrics_delta.get("gap_count_total", np.nan),
                },
                {
                    "metric": "usable_space_total_ns",
                    "before": result.metrics_before.get("usable_space_total_ns", np.nan),
                    "after": result.metrics_after.get("usable_space_total_ns", np.nan),
                    "delta": result.metrics_delta.get("usable_space_total_ns", np.nan),
                },
                {
                    "metric": "service_block_ratio",
                    "before": result.metrics_before.get("service_block_ratio", np.nan),
                    "after": result.metrics_after.get("service_block_ratio", np.nan),
                    "delta": result.metrics_delta.get("service_block_ratio", np.nan),
                },
            ]
        )

    moved_df = moved_stream_changes_df(result.before_compact, result.after_compact, result.moved_stream_ids)
    age_df = pd.DataFrame({"stream": np.arange(len(inferred_ages)), "age": inferred_ages})
    age_df = age_df[age_df["age"] > 0].sort_values("age", ascending=False)

    with tabs[1]:
            st.dataframe(metrics_table, use_container_width=False)
            st.write("Largest hole per port (before)")
            st.dataframe(
                pd.DataFrame({"largest_hole_per_port_before": result.metrics_before["largest_hole_per_port"]}),
                use_container_width=False,
            )
            st.write("Largest hole per port (after)")
            st.dataframe(
                pd.DataFrame({"largest_hole_per_port_after": result.metrics_after["largest_hole_per_port"]}),
                use_container_width=False,
            )
            st.write("Frame-size threshold metrics")
            st.dataframe(thresholds_table(result), use_container_width=False)

    with tabs[2]:
            st.write("Moved stream changes (stream IDs 0-based, port IDs 1-based)")
            st.dataframe(moved_df, use_container_width=False)
            st.write("Inferred stream ages")
            st.dataframe(age_df, use_container_width=False)

    with tabs[3]:
            st.write("Choose plotting filters, then click Plot.")

            stream_mode = st.radio(
                "Stream selection",
                options=["Moved streams", "All active streams", "Custom stream IDs"],
                horizontal=True,
            )

            custom_stream_input = ""
            if stream_mode == "Custom stream IDs":
                custom_stream_input = st.text_input("Stream IDs (comma-separated, 0-based)", value="")

            port_mode = st.radio(
                "Port selection",
                options=["All ports", "Custom port IDs"],
                horizontal=True,
            )

            custom_port_input = ""
            if port_mode == "Custom port IDs":
                custom_port_input = st.text_input("Port IDs (comma-separated, 1-based)", value="")

            do_plot = st.button("Plot before/after", key="plot_before_after")

            if do_plot:
                stream_ids: List[int] | None
                port_ids: List[int] | None

                if stream_mode == "Moved streams":
                    stream_ids = list(result.moved_stream_ids)
                    if not stream_ids:
                        st.warning("No moved streams found for this run.")
                elif stream_mode == "All active streams":
                    stream_ids = None
                else:
                    try:
                        parsed = parse_int_list(custom_stream_input)
                    except ValueError:
                        st.error("Invalid stream ID list. Use comma-separated integers.")
                        parsed = []
                    stream_ids = parsed if parsed else None

                if port_mode == "All ports":
                    port_ids = None
                else:
                    try:
                        parsed_ports = parse_int_list(custom_port_input)
                    except ValueError:
                        st.error("Invalid port ID list. Use comma-separated integers.")
                        parsed_ports = []
                    if any(pid < 1 for pid in parsed_ports):
                        st.error("Port IDs must start from 1.")
                        parsed_ports = [pid for pid in parsed_ports if pid >= 1]
                    port_ids = parsed_ports if parsed_ports else None

                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("Before (post-departure baseline)")
                    fig_b, _ = result.before_schedule.plot(label=True, stream_ids=stream_ids, port_ids=port_ids)
                    st.pyplot(fig_b)

                with col_b:
                    st.write("After (post-defrag)")
                    fig_a, _ = result.after_schedule.plot(label=True, stream_ids=stream_ids, port_ids=port_ids)
                    st.pyplot(fig_a)
            else:
                st.info("Press 'Plot before/after' to render the charts with the selected filters.")

    with tabs[4]:
            st.subheader("Debug Reports")
            for rep in result.debug_reports:
                with st.expander(f"Stream {rep['stream']} | accepted={rep['accepted']} | reason={rep['reason']}"):
                    st.write("baseline_violation_sum:", rep.get("baseline_violation_sum"))
                    st.write("candidate_violation_sum:", rep.get("candidate_violation_sum"))
                    st.write("per_check_violation_deltas:", rep.get("per_check_violation_deltas", {}))
                    st.dataframe(pd.DataFrame(rep.get("port_attempts", [])), use_container_width=False)

            st.subheader("Current Info (Selected Streams)")
            for idx, info in enumerate(current_info):
                st.write(f"Selected stream #{idx + 1}")
                st.dataframe(pd.DataFrame(info), use_container_width=False)


if __name__ == "__main__":
    main()
