import copy
import networkx as nx
import random
import io
import contextlib

from rctsn.scenario_creator.network_generator import NetworkGenerator
from rctsn.utils.path_calculator import k_shortest_paths


def build_network_generator_from_usecase(usecase):
    """
    Build a lightweight NetworkGenerator adapter from the current usecase topology.
    This lets us reuse the same stream-generation logic as the ILP scenario creator.
    """
    ng = NetworkGenerator(subnet_id="0")
    ng.network = nx.Graph()

    # copy nodes
    for node in usecase.graph.nodes():
        if "WED" in node:
            node_type = "WED"
        elif "WN" in node:
            node_type = "WN"
        elif "ED" in node:
            node_type = "ED"
        else:
            node_type = "SW"

        ng.network.add_node(node, subnet=node.split("-")[0], type=node_type)

        if node_type in ["WED", "ED"]:
            ng._device_cache[node_type].add(node)

    # copy edges
    for u, v, data in usecase.graph.edges(data=True):
        ng.network.add_edge(u, v)
        ng.network[u][v]["link_type"] = str(data.get("link_type", "0"))

    return ng

def _get_switches_with_attached_eds(ng):
    """
    Return switches that have at least one directly attached ED.
    For the medium star, this should return only the edge switches
    if the center has no EDs.
    """
    switches = []
    for node in ng.network.nodes():
        if ng._get_core_type(node) != "SW":
            continue

        attached_eds = [
            n for n in ng.network.neighbors(node)
            if ng._get_core_type(n) == "ED"
        ]

        if attached_eds:
            switches.append(node)

    return switches


def _is_classic_host_node(ng, node):
    """
    Classic fat-tree format:
        host-1-p0, host-2-p0, ...
    """
    data_type = ng.network.nodes[node].get("type", "")
    return data_type == "host" or str(node).startswith("host-")


def _is_classic_switch_node(ng, node):
    """
    Classic fat-tree format:
        switch-c1, switch-p0-e1, ...
    """
    data_type = ng.network.nodes[node].get("type", "")
    return data_type == "switch" or str(node).startswith("switch-")


def _access_switch_of_classic_host(ng, host):
    """
    Return the switch directly attached to a classic host.
    """
    neighbors = list(ng.network.neighbors(host))

    switches = [
        n for n in neighbors
        if _is_classic_switch_node(ng, n)
    ]

    if not switches:
        return None

    # Prefer edge switches if layer metadata exists
    edge_switches = [
        n for n in switches
        if ng.network.nodes[n].get("layer") == "edge"
    ]

    if edge_switches:
        return edge_switches[0]

    return switches[0]


def _pod_of(node):
    """
    Examples:
        host-1-p0      -> 0
        switch-p2-e1   -> 2
        switch-p1-a2   -> 1
    """
    for part in str(node).split("-"):
        if part.startswith("p") and part[1:].isdigit():
            return int(part[1:])
    return None


def _generate_one_classic_host_arrival(
    ng,
    ed_ed_settings,
    cross_pod_only=False,
):
    """
    Generate one raw arrival stream for fat-tree / classic names:
        host-* -> host-*

    Returns a raw stream dict compatible with build_usecase_stream_from_request().
    """
    edge_to_hosts = {}

    for node in ng.network.nodes():
        if not _is_classic_host_node(ng, node):
            continue

        sw = _access_switch_of_classic_host(ng, node)
        if sw is None:
            continue

        edge_to_hosts.setdefault(sw, []).append(node)

    edge_switches = sorted(edge_to_hosts.keys())

    if len(edge_switches) < 2:
        raise ValueError(
            f"Need at least two host-attached switches, found: {edge_switches}"
        )

    for _ in range(1000):
        src_sw, dst_sw = random.sample(edge_switches, 2)

        if cross_pod_only and _pod_of(src_sw) == _pod_of(dst_sw):
            continue

        src = random.choice(edge_to_hosts[src_sw])
        dst = random.choice(edge_to_hosts[dst_sw])

        if src == dst:
            continue

        packet_size = random.choice(ed_ed_settings["packet_sizes"])
        period = random.choice(ed_ed_settings["periods"])
        latency_factor = random.choice(ed_ed_settings["latency_factors"])
        priority = random.choice(ed_ed_settings["priorities"])
        gamma = random.choice(ed_ed_settings["gammas"])
        jitter_factor = random.choice(ed_ed_settings.get("jitter_factors", [0.0]))

        return {
            "src": src,
            "dst": dst,
            "packet_size": int(packet_size),
            "period": int(period),
            "max_latency": int(period * latency_factor),
            "priority": int(priority),
            "gamma": float(gamma),
            "max_jitter": int(period * jitter_factor),
        }

    raise RuntimeError("Could not generate one classic host arrival.")


def _get_switches_with_attached_eds(ng):
    """
    Return switches with directly attached EDs.
    Used for our custom 0-SW / 0-ED line and star scenarios.
    """
    switches = []

    for node in ng.network.nodes():
        try:
            if ng._get_core_type(node) != "SW":
                continue
        except ValueError:
            continue

        attached_eds = [
            n for n in ng.network.neighbors(node)
            if ng._get_core_type(n) == "ED"
        ]

        if attached_eds:
            switches.append(node)

    return switches


def generate_one_arrival_stream(
    ng,
    ed_ed_settings=None,
    wed_wed_settings=None,
    wed_ed_settings=None,
    debug=False,
):
    """
    Generate exactly one logical arrival request.

    Supports two topology naming formats:

    1) Custom NetworkGenerator format:
        0-ED1, 0-ED2, 0-SW1, ...

       In this case we use generate_one_way_ed_streams_between_switches()
       between two different ED-attached switches.

    2) Classic fat-tree usecase format:
        host-1-p0, switch-p0-e1, switch-c1, ...

       In this case we directly sample one host-* source and one host-* destination.
    """

    if ed_ed_settings is None:
        ed_ed_settings = {
            "packet_sizes": [64, 128, 256, 512],
            "periods": [500_000, 1_000_000],
            "latency_factors": [0.7, 0.8, 0.9, 1.0],
            "priorities": [1],
            "gammas": [1.0],
            "jitter_factors": [0.0],
        }

    # ------------------------------------------------------------
    # Case 1: classic fat-tree names: host-* / switch-*
    # ------------------------------------------------------------
    if any(str(n).startswith("host-") for n in ng.network.nodes()):
        arrival_stream = _generate_one_classic_host_arrival(
            ng=ng,
            ed_ed_settings=ed_ed_settings,
            cross_pod_only=False,
        )

        if debug:
            print("[DEBUG] generated classic host arrival:")
            print(arrival_stream)

        return arrival_stream

    # ------------------------------------------------------------
    # Case 2: our NetworkGenerator names: 0-ED / 0-SW
    # ------------------------------------------------------------
    before = len(ng.streams)

    edge_switches = _get_switches_with_attached_eds(ng)

    if len(edge_switches) < 2:
        raise ValueError(
            f"Need at least two switches with attached EDs, found: {edge_switches}"
        )

    src_switch, dst_switch = random.sample(edge_switches, 2)

    with contextlib.redirect_stdout(io.StringIO()):
        ng.generate_one_way_ed_streams_between_switches(
            src_switch=src_switch,
            dst_switch=dst_switch,
            count=1,
            settings=ed_ed_settings,
            shuffle_streams=False,
        )

    after = len(ng.streams)
    added = ng.streams[before:after]

    if debug:
        print(f"[DEBUG] before={before}, after={after}, added_count={len(added)}")
        print(f"[DEBUG] src_switch={src_switch}, dst_switch={dst_switch}")
        for i, s in enumerate(added):
            print(f"[DEBUG] added[{i}] = {s}")

    if not added:
        return None

    arrival_stream = copy.deepcopy(added[0])

    # Remove temporary generated stream from the generator object.
    del ng.streams[before:after]

    if debug:
        print(f"[DEBUG] kept one stream: {arrival_stream}")
        print(f"[DEBUG] ng.streams length after cleanup = {len(ng.streams)}")

    return arrival_stream


def build_usecase_stream_from_request(raw_stream, stream_id, usecase, k_paths=1, debug=False):
    """
    Convert one raw generated stream request into the usecase stream format.
    """
    src = raw_stream["src"]
    dst = raw_stream["dst"]

    graph_nx = usecase.graph.to_undirected()

    with contextlib.redirect_stdout(io.StringIO()):
        kpaths = k_shortest_paths(graph_nx, src, dst, k=k_paths)

    if debug:
        print(f"[DEBUG] src={src}, dst={dst}, kpaths={kpaths}")

    if not kpaths:
        return None

    edge_path = kpaths[0]

    edge_to_port = {}
    for port_name, pinfo in usecase.ports.items():
        edge_to_port[(pinfo["sourceNode"], pinfo["destNode"])] = port_name

    path_list = []
    for u, v in edge_path:
        if (u, v) not in edge_to_port:
            raise KeyError(f"No egress port found in usecase.ports for edge ({u}, {v})")
        path_list.append(edge_to_port[(u, v)])

    custom_graph = nx.DiGraph()
    for i in range(len(path_list) - 1):
        custom_graph.add_edge(path_list[i], path_list[i + 1])

    stream = {
        "name": f"stream-{stream_id}",
        "type": "isochronous",
        "interval": int(raw_stream["period"]),
        "maxFramesPerInterval": 1,
        "maxFrameSize": int(raw_stream["packet_size"]),
        "maxLatency": int(raw_stream.get("max_latency", raw_stream["period"])),
        "tlkrPssblPlcmnt": [src],
        "lstnsrPssblPlcmnt": [[dst]],
        "earliest-transmit-offset": 0,
        "latest-transmit-offset": int(raw_stream["period"] - 1),
        "jitter": int(raw_stream.get("max_jitter", 0)),
        "pathLists": [path_list],
        "originalStreamId": stream_id,
        "priority": int(raw_stream.get("priority", 1)),
        "gamma": float(raw_stream.get("gamma", 1.0)),
        "customGraph": custom_graph,
    }

    if debug:
        print(f"[DEBUG] built stream={stream}")

    return stream

