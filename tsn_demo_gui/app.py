from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from src.json_loader import (
    get_links_df,
    get_nodes_df,
    get_scheduled_ports,
    get_schedule_records,
    get_stream,
    get_stream_details,
    get_stream_options,
    get_streams_df,
    get_summary_stats,
    load_json_from_bytes,
    load_json_from_path,
    validate_usecase_json,
)
from src.schedule_viz import create_gcl_timeline, create_port_utilization_figure
from src.topology_viz import create_topology_figure
from src.training_monitor import (
    DEFAULT_SAMPLE_RUN,
    create_actions_figure,
    create_metric_figure,
    create_overview_figure,
    create_reward_band_figure,
    create_reward_components_figure,
    load_json_file,
    load_monitor_csv,
    load_monitor_from_upload,
    log_until_episode,
    read_text_if_exists,
    run_paths,
    summarize_run,
    tail_lines,
)
from src.training_parser import (
    ParsedTrainingCommand,
    build_train_command,
    parse_training_shell,
    parsed_to_dataframe_rows,
)
from src.training_runner import process_status, start_training_process, stop_process


APP_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = APP_DIR / "examples" / "finals.json"
DEFAULT_RUN_SH = APP_DIR / "examples" / "run.sh"


st.set_page_config(
    page_title="TSN Defrag",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    :root {
        --tsn-soft: #F6F5FF;
        --tsn-primary: #6C63FF;
        --tsn-text: #1e293b;
        --tsn-muted: #64748b;
        --tsn-border: #e5e7eb;
        --tsn-green: #16a34a;
        --tsn-orange: #f97316;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: #ffffff;
        color: var(--tsn-text);
    }

    [data-testid="stSidebar"] {
        background: var(--tsn-soft);
        border-right: 1px solid var(--tsn-border);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] label {
        color: var(--tsn-text);
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.15rem 1.35rem;
        border-radius: 1.25rem;
        background: linear-gradient(120deg, #F6F5FF 0%, #ffffff 65%, #F6F5FF 100%);
        border: 1px solid var(--tsn-border);
        margin-bottom: 1rem;
        box-shadow: 0 10px 28px rgba(108, 99, 255, 0.08);
    }

    .hero h1 {
        margin: 0;
        font-size: 2rem;
        letter-spacing: -0.02em;
    }

    .hero p {
        margin: 0.35rem 0 0 0;
        color: var(--tsn-muted);
    }

    .soft-card {
        border: 1px solid var(--tsn-border);
        border-radius: 1rem;
        padding: 1rem;
        background: var(--tsn-soft);
    }

    .small-muted {
        color: var(--tsn-muted);
        font-size: 0.9rem;
    }

    div[data-testid="stMetric"] {
        background: var(--tsn-soft);
        border: 1px solid var(--tsn-border);
        border-radius: 1rem;
        padding: 0.7rem 0.85rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.45rem;
    }

    .section-nav {
        padding: 0.85rem 1rem 0.45rem 1rem;
        border: 1px solid var(--tsn-border);
        border-radius: 1rem;
        background: #ffffff;
        margin: 1rem 0 1rem 0;
    }

    .section-nav-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: var(--tsn-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: -0.2rem;
    }

    div[role="radiogroup"] {
        gap: 0.6rem;
    }

    div[role="radiogroup"] label {
        background: var(--tsn-soft);
        border: 1px solid var(--tsn-border);
        border-radius: 999px;
        padding: 0.35rem 0.75rem;
    }

    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background: #ffffff;
        border: 1px solid var(--tsn-border);
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_default_data():
    return load_json_from_path(str(DEFAULT_JSON))


@st.cache_data(show_spinner=False)
def load_uploaded_data(raw: bytes):
    return load_json_from_bytes(raw)


@st.cache_data(show_spinner=False)
def load_sample_monitor(path: str):
    return load_monitor_csv(path)


def render_header():
    st.markdown(
        """
<div class="hero">
    <h1>TSN Defrag</h1>
    <p>Light interface for topology inspection, GCL schedule visualization, DRL training, and benchmarking.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_data_loader():
    st.sidebar.header("Input")
    uploaded = st.sidebar.file_uploader(
        "Upload use-case JSON",
        type=["json"],
        help="Upload a JSON generated/exported from your TSN scheduling use case.",
    )

    if uploaded is not None:
        data = load_uploaded_data(uploaded.getvalue())
        st.sidebar.success(f"Loaded: {uploaded.name}")
        return data, uploaded.name

    data = load_default_data()
    st.sidebar.info("Using bundled example: examples/finals.json")
    return data, "examples/finals.json"


def render_stats(data: dict):
    stats = get_summary_stats(data)
    columns = st.columns(6)

    columns[0].metric("Nodes", stats["nodes"])
    columns[1].metric("Links", stats["links"])
    columns[2].metric("Streams", stats["streams"])
    columns[3].metric("Scheduled ports", stats["scheduled_ports"])
    columns[4].metric("GCL windows", stats["gcl_windows"])
    columns[5].metric("Cycle", f'{stats["cycle_ns"] or "-"} ns')


def render_visualization(data: dict):
    warnings = validate_usecase_json(data)
    if warnings:
        st.warning("The uploaded JSON may be incomplete:\n\n" + "\n".join(f"- {w}" for w in warnings))
        if any("Missing top-level" in w for w in warnings):
            return

    stream_options = get_stream_options(data)
    stream_label_to_index = dict(stream_options)

    st.subheader("Topology")

    top_controls = st.columns([1, 2])
    selected_stream_label = top_controls[0].selectbox(
        "Highlight stream path",
        ["None"] + [label for label, _ in stream_options],
        index=0,
    )

    selected_stream = None
    selected_stream_index = None
    if selected_stream_label != "None":
        selected_stream_index = stream_label_to_index[selected_stream_label]
        selected_stream = get_stream(data, selected_stream_index)

    fig = create_topology_figure(
        data=data,
        selected_stream=selected_stream,
    )
    st.plotly_chart(fig, use_container_width=True)

    if selected_stream is not None and selected_stream_index is not None:
        with st.expander("Selected stream details", expanded=True):
            details = get_stream_details(selected_stream, selected_stream_index)
            st.dataframe(pd.DataFrame([details]), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("GCL schedule")

    scheduled_ports = get_scheduled_ports(data)
    if not scheduled_ports:
        st.info("No scheduled ports were found in this JSON.")
        return

    scheduled_df = pd.DataFrame(scheduled_ports)

    gcl_controls = st.columns([1, 1, 1])
    node_options = ["All scheduled nodes"] + sorted(scheduled_df["node"].unique().tolist())
    selected_node = gcl_controls[0].selectbox("Node", node_options)

    if selected_node == "All scheduled nodes":
        port_options = ["All scheduled ports"]
    else:
        node_ports = scheduled_df.loc[scheduled_df["node"] == selected_node, "port"].tolist()
        port_options = ["All scheduled ports"] + sorted(node_ports)

    selected_port = gcl_controls[1].selectbox("Port", port_options)

    only_stream_windows = gcl_controls[2].checkbox(
        "Only selected stream windows",
        value=False,
        disabled=selected_stream is None,
    )

    node_filter = None if selected_node == "All scheduled nodes" else selected_node
    port_filter = None if selected_port == "All scheduled ports" else selected_port

    schedule_df = get_schedule_records(data, node_name=node_filter, port_name=port_filter)

    if only_stream_windows and selected_stream_index is not None and not schedule_df.empty:
        sid = str(selected_stream_index)
        schedule_df = schedule_df[
            schedule_df["stream_sequence"].astype(str).apply(
                lambda text: sid in [part.strip() for part in text.split(",") if part.strip()]
            )
        ]

    title_parts = []
    if node_filter:
        title_parts.append(node_filter)
    if port_filter:
        title_parts.append(port_filter)
    title = "GCL timeline" if not title_parts else "GCL timeline · " + ".".join(title_parts)

    st.plotly_chart(create_gcl_timeline(schedule_df, title=title), use_container_width=True)

    with st.expander("Schedule table", expanded=False):
        if schedule_df.empty:
            st.info("No windows match the current selection.")
        else:
            st.dataframe(schedule_df, use_container_width=True, hide_index=True)

    with st.expander("Port utilization", expanded=False):
        all_schedule_df = get_schedule_records(data)
        st.plotly_chart(create_port_utilization_figure(all_schedule_df), use_container_width=True)

    st.divider()
    st.subheader("Raw inspection tables")

    tabs = st.tabs(["Nodes", "Links", "Streams", "Scheduled ports"])
    with tabs[0]:
        st.dataframe(get_nodes_df(data), use_container_width=True, hide_index=True)
    with tabs[1]:
        st.dataframe(get_links_df(data), use_container_width=True, hide_index=True)
    with tabs[2]:
        st.dataframe(get_streams_df(data), use_container_width=True, hide_index=True)
    with tabs[3]:
        st.dataframe(scheduled_df, use_container_width=True, hide_index=True)


ENVIRONMENT_KEYS = [
    "config",
    "instance",
    "out_dir",
    "train_episodes",
    "timesteps",
    "decision_points_per_episode",
    "lambda_arrival",
    "mu_departure",
    "dt",
    "max_candidates",
    "max_steps",
    "per_hop_candidate_limit",
    "beam_width",
    "top_k_oldest_random",
]
REWARD_KEYS = [
    "reward_alpha",
    "invalid_penalty",
    "defrag_cycle_penalty",
    "reallocation_penalty",
    "min_reallocations_per_cycle",
    "early_stop_penalty",
    "stream_cooldown_decisions",
]
DQN_KEYS = [
    "learning_rate",
    "buffer_size",
    "batch_size",
    "learning_starts",
    "target_update_interval",
    "train_freq",
    "gradient_steps",
    "exploration_fraction",
    "exploration_initial_eps",
    "exploration_final_eps",
    "gamma",
    "seed",
    "resume_from",
]
BOOL_KEYS = ["no_usecase_schedule", "no_current_offset_candidate"]


def _format_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if pd.isna(value):
            return "-"
        return f"{value:.4f}"
    return str(value)


def _input_for_arg(key: str, value: Any, container, namespace: str) -> Any:
    label = key.replace("_", "-")
    state_key = f"{namespace}_{key}"
    if isinstance(value, bool):
        return container.checkbox(label, value=bool(value), key=state_key)
    if isinstance(value, int) and not isinstance(value, bool):
        return container.number_input(label, value=int(value), step=1, key=state_key)
    if isinstance(value, float):
        return container.number_input(label, value=float(value), format="%.8f", key=state_key)
    if value is None:
        raw = container.text_input(label, value="", key=state_key)
        return None if raw.strip() == "" else raw
    return container.text_input(label, value=str(value), key=state_key)


def render_arg_group(title: str, keys: list[str], args: Dict[str, Any], namespace: str) -> Dict[str, Any]:
    edited: Dict[str, Any] = {}
    with st.expander(title, expanded=title == "Environment"):
        cols = st.columns(3)
        for idx, key in enumerate(keys):
            if key not in args:
                continue
            edited[key] = _input_for_arg(key, args[key], cols[idx % 3], namespace)
    return edited


def get_shell_text_from_ui() -> Tuple[str, str]:
    uploaded = st.file_uploader(
        "Upload training shell file",
        type=["sh"],
        help="Upload the .sh file that contains your train.py command and hyperparameters.",
    )
    if uploaded is None:
        return "", "No file uploaded"
    return uploaded.getvalue().decode("utf-8", errors="replace"), uploaded.name

def show_command_summary(parsed: ParsedTrainingCommand):
    cols = st.columns(5)
    cols[0].metric("Output dir", parsed.out_dir or "-")
    cols[1].metric("Train script", Path(parsed.train_py).name if parsed.train_py else "-")
    cols[2].metric("Python", parsed.python_executable or "python")
    cols[3].metric("CPU pin", parsed.taskset_cpu or "-")
    cols[4].metric("Parsed args", len(parsed.args))

    with st.expander("Parsed parameters table", expanded=False):
        st.dataframe(pd.DataFrame(parsed_to_dataframe_rows(parsed.args)), use_container_width=True, hide_index=True)


def render_training_launch():
    st.subheader("Training setup")
    st.caption("Upload your `.sh` training file, verify the parsed parameters, edit them from the interface, then launch your existing `train.py` on the SSH/server machine.")

    shell_text, shell_source = get_shell_text_from_ui()
    if not shell_text.strip():
        st.info("Upload a `.sh` file to parse training parameters.")
        return

    parsed = parse_training_shell(shell_text)
    st.session_state["training_parsed"] = parsed
    st.success(f"Parsed command from {shell_source}")
    show_command_summary(parsed)

    st.markdown("#### Editable training parameters")
    args = dict(parsed.args)

    quick = st.checkbox("Quick test override", value=False, help="Useful to test the GUI without launching a long training run.")
    if quick:
        args["train_episodes"] = 2
        args["timesteps"] = 20
        args["decision_points_per_episode"] = 5

    edited: Dict[str, Any] = {}
    edited.update(render_arg_group("Environment", ENVIRONMENT_KEYS, args, "env"))
    edited.update(render_arg_group("Reward", REWARD_KEYS, args, "rew"))
    edited.update(render_arg_group("DQN", DQN_KEYS, args, "dqn"))
    edited.update(render_arg_group("Boolean flags", BOOL_KEYS, args, "flags"))

    # Preserve non-displayed args.
    for key, value in args.items():
        edited.setdefault(key, value)

    st.session_state["training_args"] = edited

    st.markdown("#### Launch controls")
    launch_cols = st.columns([1.2, 1.2, 1, 1])
    default_cwd = str(Path(parsed.train_py).expanduser().parent) if parsed.train_py and Path(parsed.train_py).is_absolute() else ""
    cwd = launch_cols[0].text_input("Working directory", value=default_cwd, help="Usually your TSN_optimizer_platform folder on the SSH server.")
    use_taskset = launch_cols[1].checkbox("Use taskset CPU pinning", value=bool(parsed.taskset_cpu))
    out_dir = str(edited.get("out_dir", parsed.out_dir or ""))
    log_path = launch_cols[2].text_input("Log path", value=str(Path(out_dir) / "train.log") if out_dir else parsed.log_path)
    launch_cols[3].write("")
    launch_cols[3].write("")

    try:
        command, extra_env = build_train_command(parsed, edited, use_taskset=use_taskset)
        with st.expander("Final command preview", expanded=False):
            st.code(" ".join(command), language="bash")
            st.json(extra_env or {})
    except Exception as exc:
        st.error(f"Could not build command: {exc}")
        return

    status = process_status(st.session_state.get("training_process"))
    st.markdown(f'Process status: <span class="status-pill">{status}</span>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 4])
    if c1.button("Start training", type="primary", disabled=status == "running"):
        try:
            proc = start_training_process(command, log_path=log_path, cwd=cwd or None, extra_env=extra_env)
            st.session_state["training_process"] = proc
            st.session_state["active_run_dir"] = str(Path(out_dir).expanduser()) if out_dir else ""
            st.success(f"Training started. PID: {proc.pid}")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to start training: {exc}")

    if c2.button("Stop", disabled=status != "running"):
        if stop_process(st.session_state.get("training_process")):
            st.warning("Stop signal sent.")
            st.rerun()

    if Path(log_path).expanduser().exists():
        with st.expander("Training log tail", expanded=True):
            st.code(tail_lines(read_text_if_exists(log_path, max_chars=20000), 60), language="text")


def load_run_from_choice(prefix: str = "run") -> Tuple[Optional[pd.DataFrame], Dict[str, Any], str, str]:
    uploaded = st.file_uploader(
        "Upload monitor.csv, run_args.json, and optionally train.log",
        type=["csv", "json", "log", "txt"],
        accept_multiple_files=True,
        key=f"{prefix}_uploads",
        help="For playback, upload at least monitor.csv. Add run_args.json and train.log for richer details.",
    )
    if not uploaded:
        return None, {}, "", "uploaded files"

    df: Optional[pd.DataFrame] = None
    args: Dict[str, Any] = {}
    log_text = ""
    for file in uploaded:
        name = file.name.lower()
        if name.endswith(".csv") or "monitor" in name:
            df = load_monitor_from_upload(file)
        elif name.endswith(".json") or "args" in name:
            args = json.loads(file.getvalue().decode("utf-8", errors="replace"))
        elif name.endswith(".log") or "log" in name:
            log_text = file.getvalue().decode("utf-8", errors="replace")
    return df, args, log_text, "uploaded files"

def render_training_dashboard(df: pd.DataFrame, run_args: Dict[str, Any], log_text: str = "", run_label: str = "", end_episode: Optional[int] = None):
    if df is None or df.empty:
        st.info("No monitor data available yet.")
        return

    if end_episode is not None:
        df = df.iloc[: max(1, int(end_episode))].copy()

    summary = summarize_run(df, run_args)
    progress = min(1.0, summary.get("episodes", 0) / max(1, summary.get("target_episodes", summary.get("episodes", 1))))
    st.progress(progress, text=f"{summary.get('episodes', 0)} / {summary.get('target_episodes', summary.get('episodes', 0))} episodes")

    cols = st.columns(6)
    cols[0].metric("Latest reward", _format_number(summary.get("latest_reward")))
    cols[1].metric("Best reward", _format_number(summary.get("best_reward")), f"ep {summary.get('best_reward_episode', '-')}")
    cols[2].metric("Latest SBR", _format_number(summary.get("latest_sbr")))
    cols[3].metric("Acceptance", _format_number(summary.get("latest_acceptance")))
    cols[4].metric("Reallocations", _format_number(summary.get("latest_reallocations")))
    cols[5].metric("Cycles", _format_number(summary.get("latest_defrag_cycles")))

    if run_args:
        with st.expander("Run arguments", expanded=False):
            st.json(run_args)

    st.plotly_chart(create_reward_band_figure(df, window=10), use_container_width=True)

    chart_tabs = st.tabs(["Overview", "SBR", "Overhead", "Reward components", "Actions", "Data/log"])
    with chart_tabs[0]:
        st.plotly_chart(create_overview_figure(df, window=10), use_container_width=True)
    with chart_tabs[1]:
        st.plotly_chart(create_metric_figure(df, "episode_sbr", window=10), use_container_width=True)
        if "episode_acceptance" in df.columns:
            st.plotly_chart(create_metric_figure(df, "episode_acceptance", window=10), use_container_width=True)
    with chart_tabs[2]:
        st.plotly_chart(create_metric_figure(df, "episode_reallocations", window=10), use_container_width=True)
        st.plotly_chart(create_metric_figure(df, "episode_defrag_cycles", window=10), use_container_width=True)
        st.plotly_chart(create_metric_figure(df, "l", title="Episode length", window=10), use_container_width=True)
        if "episode_violation_sum" in df.columns:
            st.plotly_chart(create_metric_figure(df, "episode_violation_sum", window=10), use_container_width=True)
    with chart_tabs[3]:
        st.plotly_chart(create_reward_components_figure(df), use_container_width=True)
    with chart_tabs[4]:
        st.plotly_chart(create_actions_figure(df, window=10), use_container_width=True)
    with chart_tabs[5]:
        st.dataframe(df.tail(25), use_container_width=True, hide_index=True)
        if log_text:
            episode = int(df["episode"].iloc[-1]) if "episode" in df.columns else len(df)
            st.code(log_until_episode(log_text, episode, tail=45), language="text")


def render_training_playback_frame(df: pd.DataFrame, run_args: Dict[str, Any], log_text: str = "", end_episode: Optional[int] = None):
    if df is None or df.empty:
        st.info("No monitor data available yet.")
        return

    if end_episode is not None:
        df = df.iloc[: max(1, int(end_episode))].copy()

    summary = summarize_run(df, run_args)
    progress = min(1.0, summary.get("episodes", 0) / max(1, summary.get("target_episodes", summary.get("episodes", 1))))
    st.progress(progress, text=f"Playback · {summary.get('episodes', 0)} / {summary.get('target_episodes', summary.get('episodes', 0))} episodes")

    cols = st.columns(5)
    cols[0].metric("Reward", _format_number(summary.get("latest_reward")))
    cols[1].metric("Best", _format_number(summary.get("best_reward")), f"ep {summary.get('best_reward_episode', '-')}")
    cols[2].metric("SBR", _format_number(summary.get("latest_sbr")))
    cols[3].metric("Realloc", _format_number(summary.get("latest_reallocations")))
    cols[4].metric("Cycles", _format_number(summary.get("latest_defrag_cycles")))

    st.plotly_chart(create_reward_band_figure(df, window=10), use_container_width=True)
    st.plotly_chart(create_overview_figure(df, window=10), use_container_width=True)

    if log_text:
        episode = int(df["episode"].iloc[-1]) if "episode" in df.columns else len(df)
        with st.expander("Live-like log tail", expanded=False):
            st.code(log_until_episode(log_text, episode, tail=28), language="text")


def render_training_playback():
    st.subheader("Completed run playback")
    st.caption("Upload a completed run (`monitor.csv`, optionally `run_args.json` and `train.log`) and replay it as if training was progressing live.")
    df, args, log_text, run_label = load_run_from_choice(prefix="playback")
    if df is None:
        return

    controls = st.columns([1, 1, 1, 2])
    duration = controls[0].number_input("Playback duration (s)", min_value=5, max_value=180, value=45, step=5)
    step = controls[1].number_input("Episode step", min_value=1, max_value=max(1, len(df)), value=max(1, len(df) // 40), step=1)
    manual_episode = controls[2].slider("Manual episode", min_value=1, max_value=len(df), value=len(df), step=1)

    placeholder = st.empty()
    play = controls[3].button("▶ Play completed run", type="primary")

    if play:
        total_steps = max(1, len(range(1, len(df) + 1, int(step))))
        sleep_s = max(0.02, float(duration) / float(total_steps))
        for ep in range(1, len(df) + 1, int(step)):
            with placeholder.container():
                render_training_playback_frame(df, args, log_text, end_episode=ep)
            time.sleep(sleep_s)
        with placeholder.container():
            render_training_playback_frame(df, args, log_text, end_episode=len(df))
    else:
        with placeholder.container():
            render_training_playback_frame(df, args, log_text, end_episode=manual_episode)


def render_training():
    st.subheader("Training")
    tabs = st.tabs(["Launch", "Playback"])
    with tabs[0]:
        render_training_launch()
    with tabs[1]:
        render_training_playback()


def render_benchmark_placeholder():
    st.subheader("Benchmarking")
    st.info(
        "This section is intentionally disabled for now. "
        "Later it will run No-Defrag, OF-FF, OF-LF, Full-Reconfig, and DRL benchmarks."
    )
    st.selectbox("Method", ["No-Defrag", "OF-FF", "OF-LF", "Full-Reconfig", "DRL"], disabled=True)
    st.button("Run benchmark", disabled=True)


def render_section_navigation() -> str:
    st.markdown(
        '<div class="section-nav"><div class="section-nav-title">Section</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Section",
        ["Visualization", "Training", "Benchmarking"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return page


def main():
    render_header()

    data, data_name = render_data_loader()
    st.sidebar.caption(f"Current data: `{data_name}`")

    render_stats(data)
    page = render_section_navigation()

    if page == "Visualization":
        render_visualization(data)
    elif page == "Training":
        render_training()
    else:
        render_benchmark_placeholder()


if __name__ == "__main__":
    main()
