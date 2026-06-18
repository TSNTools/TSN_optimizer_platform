from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_gcl_timeline(schedule_df: pd.DataFrame, title: str = "GCL timeline") -> go.Figure:
    """
    Create a compact Gantt-like GCL timeline.

    The function expects flattened windows with columns:
    full_port, start, duration, end, state, stream_sequence.
    """
    if schedule_df is None or schedule_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No schedule found for this selection",
            height=260,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return fig

    df = schedule_df.copy()
    df["stream_label"] = df["stream_sequence"].replace("", "no stream")
    df["hover"] = (
        "<b>" + df["full_port"].astype(str) + "</b><br>"
        + "Start: " + df["start"].astype(str) + " ns<br>"
        + "Duration: " + df["duration"].astype(str) + " ns<br>"
        + "End: " + df["end"].astype(str) + " ns<br>"
        + "State: " + df["state"].astype(str) + "<br>"
        + "Stream(s): " + df["stream_sequence"].astype(str)
    )

    fig = go.Figure()

    for stream_label, group in df.groupby("stream_label", sort=False):
        fig.add_trace(
            go.Bar(
                x=group["duration"],
                y=group["full_port"],
                base=group["start"],
                orientation="h",
                name=f"Stream {stream_label}",
                customdata=group["hover"],
                hovertemplate="%{customdata}<extra></extra>",
                text=group["stream_sequence"],
                textposition="inside",
            )
        )

    max_cycle = df["cycle"].dropna().max() if "cycle" in df else None
    max_end = int(df["end"].max())
    x_max = int(max(max_end, max_cycle or max_end))

    fig.update_layout(
        title=title,
        barmode="overlay",
        height=max(320, 90 + 42 * df["full_port"].nunique()),
        margin=dict(l=20, r=20, t=55, b=30),
        xaxis_title="Time within cycle (ns)",
        yaxis_title="Port",
        legend_title="GCL windows",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(range=[0, x_max]),
    )

    return fig


def create_port_utilization_figure(schedule_df: pd.DataFrame) -> go.Figure:
    if schedule_df is None or schedule_df.empty:
        return go.Figure()

    df = schedule_df.copy()
    grouped = (
        df.groupby(["node", "port", "full_port", "cycle"], dropna=False)["duration"]
        .sum()
        .reset_index()
    )
    grouped["cycle"] = grouped["cycle"].fillna(grouped["duration"].max())
    grouped["utilization_pct"] = grouped["duration"] / grouped["cycle"] * 100

    fig = px.bar(
        grouped,
        x="full_port",
        y="utilization_pct",
        hover_data=["duration", "cycle"],
        labels={
            "full_port": "Port",
            "utilization_pct": "Occupied time (%)",
            "duration": "Occupied time (ns)",
            "cycle": "Cycle (ns)",
        },
        title="Scheduled-port utilization",
    )

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=55, b=80),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig
