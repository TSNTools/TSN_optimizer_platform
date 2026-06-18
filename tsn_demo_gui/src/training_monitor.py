from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_SAMPLE_RUN = Path(__file__).resolve().parents[1] / "examples" / "sample_run"


PRETTY = {
    "r": "Reward",
    "l": "Episode length",
    "episode_sbr": "SBR",
    "episode_acceptance": "Acceptance",
    "episode_arrivals": "Arrivals",
    "episode_rejections": "Rejections",
    "episode_invalid_moves": "Invalid moves",
    "episode_violation_sum": "Violation sum",
    "episode_reallocations": "Reallocations",
    "episode_defrag_cycles": "Defrag cycles",
    "sum_E_reward": "E reward",
    "sum_frss_reward": "FRSS reward",
    "sum_raw_move_reward": "Raw move reward",
    "sum_stop_reward": "Stop reward",
    "sum_realloc_penalty": "Reallocation penalty",
    "sum_cycle_penalty": "Cycle penalty",
    "move_actions": "Move actions",
    "stop_actions": "Stop actions",
    "forced_stops": "Forced stops",
}


def load_monitor_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"monitor.csv not found: {p}")
    df = pd.read_csv(p, comment="#")
    if len(df) == 0:
        raise ValueError(f"monitor.csv is empty: {p}")
    df = df.reset_index(drop=True)
    df["episode"] = np.arange(1, len(df) + 1)
    return df


def load_monitor_from_upload(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file, comment="#")
    if len(df) == 0:
        raise ValueError("Uploaded monitor.csv is empty.")
    df = df.reset_index(drop=True)
    df["episode"] = np.arange(1, len(df) + 1)
    return df


def load_json_file(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text_if_exists(path: str | Path, max_chars: Optional[int] = None) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    if max_chars is not None and len(text) > max_chars:
        return text[-max_chars:]
    return text


def run_paths(run_dir: str | Path) -> Dict[str, Path]:
    p = Path(run_dir).expanduser()
    return {
        "run_dir": p,
        "monitor": p / "monitor.csv",
        "args": p / "run_args.json",
        "log": p / "train.log",
        "best_model": p / "best_model.zip",
        "final_model": p / "final_model.zip",
    }


def summarize_run(df: pd.DataFrame, run_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if df.empty:
        return {}
    last = df.iloc[-1]
    best_idx = int(df["r"].idxmax()) if "r" in df.columns else int(len(df) - 1)
    best = df.iloc[best_idx]
    return {
        "episodes": int(len(df)),
        "target_episodes": int((run_args or {}).get("train_episodes", len(df)) or len(df)),
        "latest_reward": float(last.get("r", np.nan)),
        "best_reward": float(best.get("r", np.nan)),
        "best_reward_episode": int(best.get("episode", best_idx + 1)),
        "latest_sbr": float(last.get("episode_sbr", np.nan)),
        "latest_acceptance": float(last.get("episode_acceptance", np.nan)),
        "latest_invalid_moves": float(last.get("episode_invalid_moves", np.nan)),
        "latest_reallocations": float(last.get("episode_reallocations", np.nan)),
        "latest_defrag_cycles": float(last.get("episode_defrag_cycles", np.nan)),
        "latest_violation_sum": float(last.get("episode_violation_sum", np.nan)),
        "elapsed_seconds": float(last.get("t", np.nan)),
    }


def _rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(max(1, int(window)), min_periods=1).mean()


def create_metric_figure(
    df: pd.DataFrame,
    metric: str,
    title: Optional[str] = None,
    window: int = 10,
    raw_alpha: float = 0.25,
) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty or metric not in df.columns:
        fig.update_layout(title=title or metric, height=320)
        return fig

    tmp = df.copy()
    y = pd.to_numeric(tmp[metric], errors="coerce")
    name = PRETTY.get(metric, metric)
    title = title or f"{name} per episode"

    fig.add_trace(
        go.Scatter(
            x=tmp["episode"],
            y=y,
            mode="lines",
            line=dict(width=1.5, color="rgba(108, 99, 255, 0.30)"),
            name=f"episode {name.lower()}",
            hovertemplate="Episode %{x}<br>Value %{y:.4f}<extra></extra>",
        )
    )

    roll = _rolling(y, window)
    fig.add_trace(
        go.Scatter(
            x=tmp["episode"],
            y=roll,
            mode="lines",
            line=dict(width=3, color="#6C63FF"),
            name=f"rolling mean ({window})",
            hovertemplate="Episode %{x}<br>Rolling %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=20, r=20, t=55, b=35),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Episode",
        yaxis_title=name,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_reward_band_figure(df: pd.DataFrame, window: int = 10) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty or "r" not in df.columns:
        fig.update_layout(title="Episode reward", height=360)
        return fig

    tmp = df.copy()
    y = pd.to_numeric(tmp["r"], errors="coerce")
    mean = _rolling(y, window)
    std = y.rolling(max(1, int(window)), min_periods=1).std().fillna(0.0)
    upper = mean + std
    lower = mean - std

    fig.add_trace(
        go.Scatter(
            x=tmp["episode"],
            y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tmp["episode"],
            y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(108, 99, 255, 0.18)",
            name="rolling std band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tmp["episode"],
            y=mean,
            mode="lines",
            line=dict(width=3, color="#6C63FF"),
            name="rolling mean reward",
            hovertemplate="Episode %{x}<br>Reward %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tmp["episode"],
            y=y,
            mode="lines",
            line=dict(width=1, color="rgba(30, 41, 59, 0.25)"),
            name="episode reward",
            hovertemplate="Episode %{x}<br>Reward %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Episode reward",
        height=390,
        margin=dict(l=20, r=20, t=55, b=35),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Episode",
        yaxis_title="Reward",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_overview_figure(df: pd.DataFrame, window: int = 10) -> go.Figure:
    metrics = [
        ("r", "Reward"),
        ("episode_sbr", "SBR"),
        ("episode_reallocations", "Reallocations"),
        ("episode_defrag_cycles", "Defrag cycles"),
    ]
    present = [(m, t) for m, t in metrics if m in df.columns]
    if not present:
        return go.Figure()

    fig = make_subplots(rows=len(present), cols=1, shared_xaxes=True, subplot_titles=[t for _, t in present])
    for row, (metric, title) in enumerate(present, start=1):
        y = pd.to_numeric(df[metric], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=df["episode"],
                y=y,
                mode="lines",
                line=dict(width=1, color="rgba(108, 99, 255, 0.25)"),
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["episode"],
                y=_rolling(y, window),
                mode="lines",
                line=dict(width=2.5, color="#6C63FF"),
                showlegend=False,
            ),
            row=row,
            col=1,
        )
    fig.update_layout(
        height=max(560, 200 * len(present)),
        margin=dict(l=20, r=20, t=65, b=35),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(title_text="Episode", row=len(present), col=1)
    return fig


def create_reward_components_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    cols = [
        "sum_E_reward",
        "sum_frss_reward",
        "sum_raw_move_reward",
        "sum_stop_reward",
        "sum_realloc_penalty",
        "sum_cycle_penalty",
    ]
    for col in cols:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["episode"],
                    y=pd.to_numeric(df[col], errors="coerce"),
                    mode="lines",
                    name=PRETTY.get(col, col),
                )
            )
    fig.update_layout(
        title="Reward components",
        height=390,
        margin=dict(l=20, r=20, t=55, b=35),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Episode",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_actions_figure(df: pd.DataFrame, window: int = 10) -> go.Figure:
    fig = go.Figure()
    for col in ["move_actions", "stop_actions", "forced_stops"]:
        if col in df.columns:
            y = _rolling(pd.to_numeric(df[col], errors="coerce"), window)
            fig.add_trace(go.Scatter(x=df["episode"], y=y, mode="lines", name=PRETTY.get(col, col)))
    fig.update_layout(
        title=f"Actions per episode · rolling mean ({window})",
        height=360,
        margin=dict(l=20, r=20, t=55, b=35),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Episode",
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def tail_lines(text: str, n: int = 35) -> str:
    if not text:
        return ""
    return "\n".join(text.splitlines()[-max(1, int(n)):])


def log_until_episode(log_text: str, episode: int, tail: int = 28) -> str:
    if not log_text:
        return ""
    episode = int(episode)
    lines = []
    pattern = re.compile(r"episode\s+(\d+)/(\d+)")
    for line in log_text.splitlines():
        match = pattern.search(line)
        if match and int(match.group(1)) > episode:
            break
        lines.append(line)
    return "\n".join(lines[-tail:])
