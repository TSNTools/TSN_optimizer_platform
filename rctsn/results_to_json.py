import argparse
import csv
import json
import os
import pickle
import sys
import math
from collections import defaultdict

import networkx as nx


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ILP output pickles to TSN Usecase JSON."
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="ILP run directory (contains variables_*.pkl). If omitted, latest run is used.",
    )
    parser.add_argument(
        "--runs-root",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "output")),
        help="Directory containing run_* folders used to resolve latest run.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Force using latest run folder from --runs-root.",
    )
    parser.add_argument(
        "--graphml",
        default=None,
        help="Optional graphml path. Defaults to <run-dir>/network.graphml or sibling input/network.graphml.",
    )
    parser.add_argument(
        "--streams-csv",
        default=None,
        help="Optional streams.csv path. Defaults to <run-dir>/streams.csv or sibling input/streams.csv.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <run-dir>/ilp_schedule.json.",
    )
    parser.add_argument(
        "--link-config",
        default=None,
        help="Optional link_config.yaml path. Defaults to <run-dir>/link_config.yaml or sibling input/link_config.yaml.",
    )
    parser.add_argument(
        "--default-link-speed",
        type=int,
        default=100,
        help="Default link speed in Mbps if attributes are missing.",
    )
    parser.add_argument(
        "--default-queue-count",
        type=int,
        default=6,
        help="Default queue count per port.",
    )
    parser.add_argument(
        "--basetime",
        type=int,
        default=0,
        help="Basetime value to inject for all ports.",
    )
    parser.add_argument(
        "--default-proc-delay-ns",
        type=int,
        default=1000,
        help="Default processing delay for switches in ns.",
    )
    parser.add_argument(
        "--ilp-time-unit",
        choices=["ms", "us", "ns"],
        default="us",
        help="Time unit used by ILP inputs (streams.csv periods/latency/jitter and x pickle start times). Default is us.",
    )
    parser.add_argument(
        "--duration-source",
        choices=["link", "ilp-best", "ilp-worst", "ilp-robust"],
        default="ilp-robust",
        help="How to compute per-port window duration. 'ilp-robust' uses stream_dict times/deviations/gamma.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate output by loading it through Usecase.from_json.",
    )
    return parser.parse_args()


def _pick_existing_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _find_latest_run_dir(runs_root):
    if not runs_root or not os.path.isdir(runs_root):
        return None

    run_dirs = []
    for name in os.listdir(runs_root):
        path = os.path.join(runs_root, name)
        if os.path.isdir(path) and name.startswith("run_"):
            run_dirs.append(path)

    if not run_dirs:
        return None

    # Prefer newest modification time to avoid relying on lexicographic order.
    run_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return run_dirs[0]


def _is_host_name(name):
    lname = str(name).lower()
    return ("ed" in lname) or lname.startswith("host")


def _is_switch_name(name):
    lname = str(name).lower()
    return ("sw" in lname) or lname.startswith("switch")


def _strip_parallel_suffix(node_name):
    return str(node_name).split("#")[0]


def _int_or_default(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _time_to_ns(value, unit):
    v = float(value)
    if unit == "ns":
        return int(round(v))
    if unit == "us":
        return int(round(v * 1_000))
    # ms
    return int(round(v * 1_000_000))


def _normalize_node_names(graph):
    # Keep names unchanged when they already follow host/switch style.
    if all(_is_host_name(n) or _is_switch_name(n) for n in graph.nodes()):
        node_types = {
            n: ("host" if _is_host_name(n) else "switch")
            for n in graph.nodes()
        }
        same_map = {n: n for n in graph.nodes()}
        return graph.copy(), same_map, node_types

    host_count = 0
    switch_count = 0
    name_map = {}
    node_types = {}

    for old_name in sorted(graph.nodes()):
        if _is_host_name(old_name):
            host_count += 1
            new_name = f"host-{host_count}"
            node_types[new_name] = "host"
        else:
            switch_count += 1
            new_name = f"switch-{switch_count}"
            node_types[new_name] = "switch"
        name_map[old_name] = new_name

    normalized = nx.relabel_nodes(graph, name_map, copy=True)
    return normalized, name_map, node_types


def _load_link_config_defaults(link_config_path, default_link_speed_mbps, default_proc_delay_ns):
    defaults = {
        "link_speed_mbps": default_link_speed_mbps,
        "prop_delay_ns": 10,
        "proc_delay_ns": default_proc_delay_ns,
    }
    if not link_config_path or not os.path.exists(link_config_path):
        return defaults

    # Minimal parser for the expected link_config structure.
    # We only need link_types -> '0' -> best_case_speed/processing_time/propagation_time.
    in_link_types = False
    in_type_0 = False
    best_case_speed = None
    propagation_time = None
    processing_time = None

    with open(link_config_path, "r") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue

            stripped = line.strip()
            if stripped.startswith("link_types:"):
                in_link_types = True
                in_type_0 = False
                continue

            if not in_link_types:
                continue

            # Enter/leave type-0 block by indentation and label.
            if stripped.startswith("'0':") or stripped.startswith('"0":') or stripped.startswith("0:"):
                in_type_0 = True
                continue
            if stripped.startswith("'") and stripped.endswith(":") and not (
                stripped.startswith("'0':")
            ):
                if in_type_0:
                    break
                continue

            if not in_type_0:
                continue

            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not value:
                continue

            if key == "best_case_speed":
                best_case_speed = _int_or_default(value, None)
            elif key == "propagation_time":
                propagation_time = _int_or_default(value, None)
            elif key == "processing_time":
                processing_time = _int_or_default(value, None)

    defaults["link_speed_mbps"] = _int_or_default(best_case_speed, defaults["link_speed_mbps"])

    # link_config type 0 times are in microseconds.
    prop_us = _int_or_default(propagation_time, 10)
    proc_us = _int_or_default(processing_time, defaults["proc_delay_ns"] // 1_000)
    defaults["prop_delay_ns"] = prop_us * 1_000
    defaults["proc_delay_ns"] = proc_us * 1_000
    return defaults


def _build_port_maps(graph, node_types, link_speed_mbps, default_queue_count, default_proc_delay_ns):
    undirected = graph.to_undirected()
    node_ports = {}
    egress_port_by_neighbor = defaultdict(dict)

    for node in sorted(graph.nodes()):
        if node_types[node] == "host":
            node_ports[node] = {
                "Port1": {
                    "name": "Port1",
                    "speed": link_speed_mbps,
                    "queueCount": default_queue_count,
                    "basetime": 0,
                    "scheduleCycle": 0,
                    "schedule": [],
                }
            }
            continue

        ports = {}
        neighbors = sorted(undirected.neighbors(node))
        for i, nb in enumerate(neighbors, start=1):
            pname = f"P{i}"
            egress_port_by_neighbor[node][nb] = pname
            ports[pname] = {
                "name": pname,
                "speed": link_speed_mbps,
                "queueCount": default_queue_count,
                "basetime": 0,
                "scheduleCycle": 0,
                "schedule": [],
            }
        node_ports[node] = ports

    nodes_json = []
    for node in sorted(graph.nodes()):
        entry = {
            "type": node_types[node],
            "name": node,
            "ports": node_ports[node],
        }
        if node_types[node] == "switch":
            entry["procDelay"] = _int_or_default(
                graph.nodes[node].get("procDelay"),
                default_proc_delay_ns,
            )
        nodes_json.append(entry)

    return nodes_json, egress_port_by_neighbor


def _make_fq_port(node, local_port):
    return f"{node}.{local_port}"


def _build_links_json(graph, node_types, egress_port_by_neighbor, global_link_speed_mbps):
    links_json = []
    directed = graph if graph.is_directed() else graph.to_directed()

    for src, dst, attrs in directed.edges(data=True):
        if node_types[src] == "host":
            src_local = "Port1"
        else:
            src_local = egress_port_by_neighbor[src][dst]

        if node_types[dst] == "host":
            dst_local = "Port1"
        else:
            dst_local = egress_port_by_neighbor[dst][src]

        prop_delay = _int_or_default(attrs.get("propDelay"), 10)
        link_bw = _int_or_default(attrs.get("linkBw"), global_link_speed_mbps)

        links_json.append(
            {
                "sourceNode": src,
                "sourcePort": _make_fq_port(src, src_local),
                "destNode": dst,
                "destPort": _make_fq_port(dst, dst_local),
                "propDelay": prop_delay,
                "linkBw": link_bw,
            }
        )

    return links_json


def _override_link_delays(links_json, prop_delay_ns, link_speed_mbps):
    for link in links_json:
        link["propDelay"] = prop_delay_ns
        link["linkBw"] = link_speed_mbps


def _load_streams_info(streams_csv_path, ilp_time_unit):
    streams_info = {}
    with open(streams_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for ordinal, row in enumerate(reader):
            stream_idx = _int_or_default(row.get("index"), ordinal)
            streams_info[stream_idx] = {
                "src": str(row["src"]),
                "dst": str(row["dst"]),
                "packet_size": _int_or_default(row.get("packet_size"), 64),
                "period": _time_to_ns(_int_or_default(row.get("period"), 1000), ilp_time_unit),
                "max_latency": _time_to_ns(_int_or_default(row.get("max_latency"), 1000), ilp_time_unit),
                "priority": _int_or_default(row.get("priority"), 0),
                "gamma": float(row.get("gamma", 1.0)),
                "max_jitter": _time_to_ns(_int_or_default(row.get("max_jitter"), 0), ilp_time_unit),
            }
    return streams_info


def _extract_selected_paths(z):
    selected_path = {}
    for key, val in z.items():
        if val != 1:
            continue
        if not isinstance(key, tuple):
            continue
        if len(key) == 2:
            s, u = key
            selected_path[s] = u
        elif len(key) == 3:
            s, u, _ = key
            selected_path[s] = u
    return selected_path

def _build_scheduled_stream_id_map(a, streams_info):
    """
    Build a dense remapping for active/scheduled streams only.

    Returns
    -------
    dict
        old_stream_id -> new_dense_stream_id
    """
    active_old_ids = [
        s for s in sorted(streams_info.keys())
        if _is_active_stream(a, s)
    ]
    return {old_id: new_id for new_id, old_id in enumerate(active_old_ids)}


def _extract_first_rep_hops(x, old_to_new_name):
    path_hops = defaultdict(list)
    for key, time_us in x.items():
        if not isinstance(key, tuple) or len(key) != 4:
            continue
        s, u, r, edge = key
        if r != 1:
            continue
        old_src, old_dst = edge
        old_src = _strip_parallel_suffix(old_src)
        old_dst = _strip_parallel_suffix(old_dst)
        src = old_to_new_name[old_src]
        dst = old_to_new_name[old_dst]
        path_hops[(s, u)].append((src, dst, time_us, old_src, old_dst))
    return path_hops


def _duration_from_stream_dict_ns(stream_dict, stream_id, path_id, old_src, old_dst, mode, ilp_time_unit):
    if stream_dict is None:
        return None

    path_data = (stream_dict.get(stream_id, {}) or {}).get(path_id, {}) or {}
    edge = (old_src, old_dst)
    times = (path_data.get("times", {}) or {}).get(edge, {}) or {}
    deviations = (path_data.get("deviations", {}) or {}).get(edge, {}) or {}
    if not times and not deviations:
        return None

    base = float(sum(times.values()))
    dev = float(sum(deviations.values()))
    gamma = float(path_data.get("gamma", 1.0))

    if mode == "ilp-best":
        duration_ilp = base
    elif mode == "ilp-worst":
        duration_ilp = base + dev
    else:
        duration_ilp = base + gamma * dev

    return max(1, _time_to_ns(duration_ilp, ilp_time_unit))


def _calc_duration_ns(packet_size_bytes, link_speed_mbps):
    speed_bits_per_ns = max(link_speed_mbps / 1000.0, 1e-12)
    return int((packet_size_bytes * 8) / speed_bits_per_ns)

def _calc_occupied_duration_ns(
    src_node,
    node_types,
    packet_size_bytes,
    link_speed_mbps,
    prop_delay_ns,
    proc_delay_ns,
):
    """
    Occupied GCL window model:
    - talker host egress: tx + prop
    - switch egress: tx + prop + proc
    """
    tx_ns = _calc_duration_ns(packet_size_bytes, link_speed_mbps)

    if node_types[src_node] == "host":
        return max(1, tx_ns + prop_delay_ns)

    return max(1, tx_ns + prop_delay_ns + proc_delay_ns)


def _is_active_stream(a, stream_id):
    direct = a.get(stream_id)
    if direct is not None:
        return direct == 1
    tuple_key = a.get((stream_id,))
    if tuple_key is not None:
        return tuple_key == 1
    return False


def _estimate_schedule_cycle(streams_info):
    periods = [max(1, int(info.get("period", 1))) for info in streams_info.values()]
    if not periods:
        return 100000

    cycle = 1
    for p in periods:
        cycle = math.lcm(cycle, p)
        if cycle > 1_000_000_000:
            return max(periods)
    return cycle


def _apply_port_defaults(nodes_json, basetime, schedule_cycle):
    for node in nodes_json:
        for _, port_data in node["ports"].items():
            port_data["basetime"] = basetime
            port_data["scheduleCycle"] = schedule_cycle
            port_data.setdefault("schedule", [])


def _build_streams_json(
    a,
    selected_path,
    path_hops,
    streams_info,
    old_to_new_name,
    node_types,
    egress_port_by_neighbor,
    link_speed_mbps,
    prop_delay_ns,
    proc_delay_ns,
    ilp_time_unit,
    stream_dict,
    duration_source,
    scheduled_id_map,
):
    schedule_entries = defaultdict(list)
    streams_json = []

    for s, info in streams_info.items():
        if not _is_active_stream(a, s):
            continue

        if s not in selected_path:
            candidates = [u for (ss, u) in path_hops if ss == s]
            if len(candidates) == 1:
                u = candidates[0]
            else:
                print(f"Warning: cannot determine selected path for stream {s}; skipping")
                continue
        else:
            u = selected_path[s]

        if (s, u) not in path_hops:
            print(f"Warning: no hops for stream {s}, path {u}; skipping")
            continue

        hops = list(path_hops[(s, u)])
        hops.sort(key=lambda t: t[2])

        talker = old_to_new_name[_strip_parallel_suffix(info["src"])]
        listener = old_to_new_name[_strip_parallel_suffix(info["dst"])]

        if s not in scheduled_id_map:
            continue
        schedule_stream_id = int(scheduled_id_map[s])
        path_list = [_make_fq_port(talker, "Port1")]

        first_hop_duration_ns = None

        for i, (src_node, dst_node, time_us, old_src, old_dst) in enumerate(hops):
            if node_types[src_node] == "switch":
                src_local = egress_port_by_neighbor[src_node][dst_node]
                fq_src = _make_fq_port(src_node, src_local)
                path_list.append(fq_src)
            else:
                # Hosts have a single egress port.
                src_local = "Port1"

            start_ns = _time_to_ns(float(time_us), ilp_time_unit)
            duration_ns = _calc_occupied_duration_ns(
                src_node,
                node_types,
                info["packet_size"],
                link_speed_mbps,
                prop_delay_ns,
                proc_delay_ns,
            )

            schedule_entries[(src_node, src_local)].append((start_ns, duration_ns, schedule_stream_id))
            if i == 0:
                first_hop_duration_ns = duration_ns

            # Keep last element as listener ingress for compatibility with TSN parser.
            if i == len(hops) - 1:
                path_list.append(_make_fq_port(listener, "Port1"))

        offset_ns = _time_to_ns(float(hops[0][2]), ilp_time_unit)
        duration_ns = _calc_duration_ns(info["packet_size"], link_speed_mbps)
        talker_duration_ns = first_hop_duration_ns or _calc_occupied_duration_ns(
            talker,
            node_types,
            info["packet_size"],
            link_speed_mbps,
            prop_delay_ns,
            proc_delay_ns,
        )

        latest_offset = max(
            0,
            info["period"] - talker_duration_ns,
        )

        stream_json = {
            "name": f"stream-{schedule_stream_id}",
            "type": "isochronous",
            "interval": info["period"],
            "maxFramesPerInterval": 1,
            "maxFrameSize": info["packet_size"],
            "maxLatency": info["max_latency"],
            "tlkrPssblPlcmnt": [talker],
            "lstnsrPssblPlcmnt": [[listener]],
            "earliest-transmit-offset": 0,
            "latest-transmit-offset": latest_offset,
            "jitter": info["max_jitter"],
            "offset": offset_ns,
            "accLatency": 0,
            "pathLists": [path_list],
            "originalStreamId": s,
        }
        streams_json.append(stream_json)

    return streams_json, schedule_entries


def _inject_port_schedules(nodes_json, schedule_entries, queue_count):
    if queue_count <= 1:
        state_value = "o"
    else:
        state_value = ("C" * (queue_count - 1)) + "o"

    for node_obj in nodes_json:
        for port_name, port_info in node_obj["ports"].items():
            port_info["schedule"] = []
            entries = schedule_entries.get((node_obj["name"], port_name), [])
            entries.sort(key=lambda e: e[0])

            for start_ns, duration_ns, stream_id in entries:
                port_info["schedule"].append(
                    {
                        "start": start_ns,
                        "duration": duration_ns,
                        "state": state_value,
                        "stream_sequence": [stream_id],
                    }
                )


def _resolve_run_dir(args):
    if args.latest or not args.run_dir:
        latest = _find_latest_run_dir(os.path.abspath(args.runs_root))
        if latest is None:
            raise FileNotFoundError(
                f"No run_* folders found under runs root: {os.path.abspath(args.runs_root)}"
            )
        return os.path.abspath(latest)

    return os.path.abspath(args.run_dir)


def _validate_json(output_json):
    # Local import for optional validation without making this script hard-dependent.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from usecase_generator import Usecase

    uc = Usecase.from_json(output_json)
    print(
        "Validation OK: "
        f"{len(uc.graph.nodes)} nodes, {len(uc.graph.edges)} links, {len(uc.streams)} streams"
    )


def main():
    args = parse_args()

    run_dir = _resolve_run_dir(args)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print(f"Using run directory: {run_dir}")

    variables_a_path = os.path.join(run_dir, "variables_a.pkl")
    variables_z_path = os.path.join(run_dir, "variables_z.pkl")
    variables_x_path = os.path.join(run_dir, "variables_x.pkl")
    stream_dict_path = os.path.join(run_dir, "stream_dict.pkl")

    graphml_path = _pick_existing_path(
        [
            args.graphml,
            os.path.join(run_dir, "network.graphml"),
            os.path.abspath(os.path.join(run_dir, "..", "..", "input", "network.graphml")),
        ]
    )
    streams_csv_path = _pick_existing_path(
        [
            args.streams_csv,
            os.path.join(run_dir, "streams.csv"),
            os.path.abspath(os.path.join(run_dir, "..", "..", "input", "streams.csv")),
        ]
    )
    link_config_path = _pick_existing_path(
        [
            args.link_config,
            os.path.join(run_dir, "link_config.yaml"),
            os.path.abspath(os.path.join(run_dir, "..", "..", "input", "link_config.yaml")),
        ]
    )
    output_path = args.output or os.path.join(run_dir, "ilp_schedule.json")

    if graphml_path is None:
        raise FileNotFoundError("Could not find network.graphml. Use --graphml.")
    if streams_csv_path is None:
        raise FileNotFoundError("Could not find streams.csv. Use --streams-csv.")

    cfg_defaults = _load_link_config_defaults(
        link_config_path,
        args.default_link_speed,
        args.default_proc_delay_ns,
    )

    with open(variables_a_path, "rb") as f:
        a = pickle.load(f)
    with open(variables_z_path, "rb") as f:
        z = pickle.load(f)
    with open(variables_x_path, "rb") as f:
        x = pickle.load(f)
    stream_dict = None
    if os.path.exists(stream_dict_path):
        with open(stream_dict_path, "rb") as f:
            stream_dict = pickle.load(f)

    raw_graph = nx.read_graphml(graphml_path)
    graph, old_to_new_name, node_types = _normalize_node_names(raw_graph)
    link_speed_mbps = cfg_defaults["link_speed_mbps"]

    streams_info = _load_streams_info(streams_csv_path, args.ilp_time_unit)
    schedule_cycle = _estimate_schedule_cycle(streams_info)

    nodes_json, egress_port_by_neighbor = _build_port_maps(
        graph,
        node_types,
        link_speed_mbps,
        args.default_queue_count,
        cfg_defaults["proc_delay_ns"],
    )
    _apply_port_defaults(nodes_json, args.basetime, schedule_cycle)

    links_json = _build_links_json(
        graph,
        node_types,
        egress_port_by_neighbor,
        link_speed_mbps,
    )
    _override_link_delays(links_json, cfg_defaults["prop_delay_ns"], link_speed_mbps)

    selected_path = _extract_selected_paths(z)
    path_hops = _extract_first_rep_hops(x, old_to_new_name)
    scheduled_id_map = _build_scheduled_stream_id_map(a, streams_info)
    streams_json, schedule_entries = _build_streams_json(
        a,
        selected_path,
        path_hops,
        streams_info,
        old_to_new_name,
        node_types,
        egress_port_by_neighbor,
        link_speed_mbps,
        cfg_defaults["prop_delay_ns"],
        cfg_defaults["proc_delay_ns"],
        args.ilp_time_unit,
        stream_dict,
        args.duration_source,
        scheduled_id_map,
    )
    _inject_port_schedules(nodes_json, schedule_entries, args.default_queue_count)

    output_json = {
        "nodes": nodes_json,
        "links": links_json,
        "streams": streams_json,
        "index": None,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=2)

    print(f"Converted {len(streams_json)} active streams")
    print(f"JSON saved to {output_path}")

    if args.validate:
        _validate_json(output_json)


if __name__ == "__main__":
    main()