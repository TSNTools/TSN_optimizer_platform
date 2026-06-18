from __future__ import annotations

from typing import Dict, Optional, Tuple

import networkx as nx
import plotly.graph_objects as go

from .json_loader import build_graph, get_stream_path_edges


def _node_color(node_type: str) -> str:
    text = str(node_type).lower()
    if text in {"switch", "sw"}:
        return "#6C63FF"
    if text in {"host", "ed", "wed"}:
        return "#20C997"
    return "#94A3B8"


def _node_symbol(node_type: str) -> str:
    text = str(node_type).lower()
    if text in {"switch", "sw"}:
        return "square"
    return "circle"


def create_topology_figure(
    data: dict,
    selected_stream: Optional[dict] = None,
    layout_name: str = "kamada-kawai",
) -> go.Figure:
    graph = build_graph(data)

    if graph.number_of_nodes() == 0:
        return go.Figure()

    if layout_name == "circular":
        pos = nx.circular_layout(graph)
    elif layout_name == "spring":
        pos = nx.spring_layout(graph, seed=7, k=1.1)
    else:
        try:
            pos = nx.kamada_kawai_layout(graph)
        except Exception:
            pos = nx.spring_layout(graph, seed=7, k=1.1)

    selected_edges = {
        tuple(sorted(edge))
        for edge in get_stream_path_edges(selected_stream)
    }

    normal_x = []
    normal_y = []
    highlight_x = []
    highlight_y = []

    for src, dst in graph.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        target = highlight_x if tuple(sorted((src, dst))) in selected_edges else normal_x
        target_y = highlight_y if tuple(sorted((src, dst))) in selected_edges else normal_y
        target.extend([x0, x1, None])
        target_y.extend([y0, y1, None])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=normal_x,
            y=normal_y,
            mode="lines",
            line=dict(width=1.5, color="#CBD5E1"),
            hoverinfo="none",
            showlegend=False,
        )
    )

    if highlight_x:
        fig.add_trace(
            go.Scatter(
                x=highlight_x,
                y=highlight_y,
                mode="lines",
                line=dict(width=5, color="#F97316"),
                hoverinfo="none",
                name="Selected stream path",
            )
        )

    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_color = []
    node_symbol = []
    node_size = []

    for node_name, attrs in graph.nodes(data=True):
        x, y = pos[node_name]
        scheduled_ports = int(attrs.get("scheduled_ports", 0) or 0)
        ports = int(attrs.get("ports", 0) or 0)
        node_type = attrs.get("type", "")

        node_x.append(x)
        node_y.append(y)
        node_text.append(node_name)
        node_hover.append(
            f"<b>{node_name}</b><br>"
            f"Type: {node_type}<br>"
            f"Ports: {ports}<br>"
            f"Scheduled ports: {scheduled_ports}"
        )
        node_color.append(_node_color(node_type))
        node_symbol.append(_node_symbol(node_type))
        node_size.append(22 + scheduled_ports * 3)

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hovertext=node_hover,
            hoverinfo="text",
            marker=dict(
                size=node_size,
                color=node_color,
                symbol=node_symbol,
                line=dict(width=2, color="white"),
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=520,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode="closest",
    )

    return fig
