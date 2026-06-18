from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import pandas as pd


JsonDict = Dict[str, Any]


def load_json_from_bytes(raw: bytes) -> JsonDict:
    """Load a TSN use-case JSON from uploaded bytes."""
    return json.loads(raw.decode("utf-8"))


def load_json_from_path(path: str) -> JsonDict:
    """Load a TSN use-case JSON from a local path."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_usecase_json(data: JsonDict) -> List[str]:
    """Return human-readable validation warnings. Empty list means good enough for visualization."""
    warnings: List[str] = []

    for key in ["nodes", "links", "streams"]:
        if key not in data:
            warnings.append(f"Missing top-level key: {key}")

    if "nodes" in data and not isinstance(data["nodes"], list):
        warnings.append("`nodes` should be a list.")

    if "links" in data and not isinstance(data["links"], list):
        warnings.append("`links` should be a list.")

    if "streams" in data and not isinstance(data["streams"], list):
        warnings.append("`streams` should be a list.")

    return warnings


def iter_nodes(data: JsonDict) -> Iterable[JsonDict]:
    return data.get("nodes", []) or []


def iter_links(data: JsonDict) -> Iterable[JsonDict]:
    return data.get("links", []) or []


def iter_streams(data: JsonDict) -> Iterable[JsonDict]:
    return data.get("streams", []) or []


def node_by_name(data: JsonDict) -> Dict[str, JsonDict]:
    return {str(node.get("name")): node for node in iter_nodes(data)}


def get_nodes_df(data: JsonDict) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for node in iter_nodes(data):
        ports = node.get("ports", {}) or {}
        scheduled_ports = [
            port_name
            for port_name, port in ports.items()
            if port.get("schedule")
        ]

        rows.append(
            {
                "node": node.get("name", ""),
                "type": node.get("type", ""),
                "ports": len(ports),
                "scheduled_ports": len(scheduled_ports),
                "scheduled_port_names": ", ".join(scheduled_ports),
                "procDelay": node.get("procDelay", ""),
            }
        )

    return pd.DataFrame(rows)


def get_links_df(data: JsonDict) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for idx, link in enumerate(iter_links(data), start=1):
        rows.append(
            {
                "id": idx,
                "sourceNode": link.get("sourceNode", ""),
                "destNode": link.get("destNode", ""),
                "sourcePort": link.get("sourcePort", ""),
                "destPort": link.get("destPort", ""),
                "propDelay": link.get("propDelay", ""),
                "linkBw": link.get("linkBw", ""),
            }
        )

    return pd.DataFrame(rows)


def get_streams_df(data: JsonDict) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for idx, stream in enumerate(iter_streams(data)):
        path_lists = stream.get("pathLists") or []
        first_path = path_lists[0] if path_lists else []

        rows.append(
            {
                "id": idx,
                "name": stream.get("name", f"stream-{idx}"),
                "interval": stream.get("interval", ""),
                "maxLatency": stream.get("maxLatency", ""),
                "maxFrameSize": stream.get("maxFrameSize", ""),
                "maxFramesPerInterval": stream.get("maxFramesPerInterval", ""),
                "talker": ", ".join(stream.get("tlkrPssblPlcmnt", []) or []),
                "listener": ", ".join(
                    item for group in (stream.get("lstnsrPssblPlcmnt", []) or []) for item in group
                ),
                "path_hops": len(first_path),
                "path": " → ".join(first_path),
            }
        )

    return pd.DataFrame(rows)


def parse_port_ref(port_ref: str) -> Tuple[str, str]:
    """
    Split a port reference such as `0-SW1.P4` into (`0-SW1`, `P4`).

    If the value does not contain a dot, the full value is returned as node and
    the port part is empty.
    """
    text = str(port_ref)
    if "." not in text:
        return text, ""
    node, port = text.rsplit(".", 1)
    return node, port


def get_scheduled_ports(data: JsonDict) -> List[Dict[str, Any]]:
    scheduled: List[Dict[str, Any]] = []

    for node in iter_nodes(data):
        node_name = str(node.get("name", ""))
        for port_name, port in (node.get("ports", {}) or {}).items():
            schedule = port.get("schedule") or []
            if schedule:
                scheduled.append(
                    {
                        "node": node_name,
                        "port": str(port_name),
                        "full_port": f"{node_name}.{port_name}",
                        "cycle": port.get("scheduleCycle"),
                        "windows": len(schedule),
                    }
                )

    return scheduled


def get_schedule_records(
    data: JsonDict,
    node_name: Optional[str] = None,
    port_name: Optional[str] = None,
) -> pd.DataFrame:
    """Return flattened GCL windows for one port or all scheduled ports."""
    rows: List[Dict[str, Any]] = []

    for node in iter_nodes(data):
        current_node = str(node.get("name", ""))
        if node_name is not None and current_node != node_name:
            continue

        for current_port, port in (node.get("ports", {}) or {}).items():
            current_port = str(current_port)
            if port_name is not None and current_port != port_name:
                continue

            schedule = port.get("schedule") or []
            for idx, window in enumerate(schedule, start=1):
                start = int(window.get("start", 0))
                duration = int(window.get("duration", 0))
                seq = window.get("stream_sequence", []) or []
                rows.append(
                    {
                        "node": current_node,
                        "port": current_port,
                        "full_port": f"{current_node}.{current_port}",
                        "window": idx,
                        "start": start,
                        "duration": duration,
                        "end": start + duration,
                        "state": window.get("state", ""),
                        "stream_sequence": ", ".join(map(str, seq)),
                        "cycle": port.get("scheduleCycle"),
                    }
                )

    return pd.DataFrame(rows)


def get_schedule_cycle(data: JsonDict) -> Optional[int]:
    cycles = [
        item.get("cycle")
        for item in get_scheduled_ports(data)
        if item.get("cycle") is not None
    ]
    if not cycles:
        return None

    # Usually all ports share the same cycle. Use the most frequent value.
    return max(set(cycles), key=cycles.count)


def get_summary_stats(data: JsonDict) -> Dict[str, Any]:
    scheduled_ports = get_scheduled_ports(data)
    schedule_df = get_schedule_records(data)

    return {
        "nodes": len(list(iter_nodes(data))),
        "links": len(list(iter_links(data))),
        "streams": len(list(iter_streams(data))),
        "scheduled_ports": len(scheduled_ports),
        "gcl_windows": int(len(schedule_df)),
        "cycle_ns": get_schedule_cycle(data),
    }


def build_graph(data: JsonDict) -> nx.Graph:
    graph = nx.Graph()

    for node in iter_nodes(data):
        name = str(node.get("name", ""))
        graph.add_node(
            name,
            type=node.get("type", ""),
            ports=len(node.get("ports", {}) or {}),
            scheduled_ports=sum(
                1 for _, port in (node.get("ports", {}) or {}).items() if port.get("schedule")
            ),
        )

    for link in iter_links(data):
        src = str(link.get("sourceNode", ""))
        dst = str(link.get("destNode", ""))
        if not src or not dst:
            continue

        graph.add_edge(
            src,
            dst,
            sourcePort=link.get("sourcePort", ""),
            destPort=link.get("destPort", ""),
            propDelay=link.get("propDelay", ""),
            linkBw=link.get("linkBw", ""),
        )

    return graph


def get_stream_options(data: JsonDict) -> List[Tuple[str, int]]:
    options: List[Tuple[str, int]] = []

    for idx, stream in enumerate(iter_streams(data)):
        label = f"{idx} · {stream.get('name', f'stream-{idx}')}"
        options.append((label, idx))

    return options


def get_stream(data: JsonDict, stream_index: int) -> Optional[JsonDict]:
    streams = list(iter_streams(data))
    if 0 <= int(stream_index) < len(streams):
        return streams[int(stream_index)]
    return None


def get_stream_path_ports(stream: Optional[JsonDict]) -> List[str]:
    if not stream:
        return []

    path_lists = stream.get("pathLists") or []
    if not path_lists:
        return []

    return [str(item) for item in path_lists[0]]


def get_stream_path_edges(stream: Optional[JsonDict]) -> List[Tuple[str, str]]:
    ports = get_stream_path_ports(stream)
    nodes = [parse_port_ref(port)[0] for port in ports]

    edges: List[Tuple[str, str]] = []
    for a, b in zip(nodes, nodes[1:]):
        if a != b:
            edges.append((a, b))

    return edges


def get_stream_details(stream: Optional[JsonDict], stream_index: int) -> Dict[str, Any]:
    if not stream:
        return {}

    path_ports = get_stream_path_ports(stream)

    return {
        "id": stream_index,
        "name": stream.get("name", f"stream-{stream_index}"),
        "interval_ns": stream.get("interval", ""),
        "max_latency_ns": stream.get("maxLatency", ""),
        "max_frame_size_bytes": stream.get("maxFrameSize", ""),
        "frames_per_interval": stream.get("maxFramesPerInterval", ""),
        "talker": ", ".join(stream.get("tlkrPssblPlcmnt", []) or []),
        "listener": ", ".join(
            item for group in (stream.get("lstnsrPssblPlcmnt", []) or []) for item in group
        ),
        "path": " → ".join(path_ports),
    }
