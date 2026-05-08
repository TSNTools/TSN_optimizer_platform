import copy
import networkx as nx
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


def generate_one_arrival_stream(
    ng,
    ed_ed_settings=None,
    wed_wed_settings=None,
    wed_ed_settings=None,
    debug=False,
):
    """
    Generate exactly one logical arrival request.

    NetworkGenerator internally adds a bidirectional pair.
    We keep only the first newly added stream as the actual arrival
    and discard the reverse companion so one call = one arrival.
    """
    if ed_ed_settings is None:
        ed_ed_settings = {
            "packet_sizes": [256],
            "periods": [150000, 300000],
            "latency_factors": [0.7, 0.8, 0.9, 1.0],
            "priorities": [1],
            "gammas": [1.0],
            "jitter_factors": [0.0],
        }

    before = len(ng.streams)

    with contextlib.redirect_stdout(io.StringIO()):
        ng.generate_streams_with_dict_settings(
            wed_to_wed=0,
            ed_to_ed=1,
            wed_to_ed=0,
            mode="any",
            wed_wed_settings=wed_wed_settings or {
                "packet_sizes": [1024],
                "periods": [100000],
                "latency_factors": [0.6, 0.8],
                "priorities": [1],
                "gammas": [1.0],
                "jitter_factors": [0.0],
            },
            ed_ed_settings=ed_ed_settings,
            wed_ed_settings=wed_ed_settings or {
                "packet_sizes": [1024],
                "periods": [100000],
                "latency_factors": [0.6, 0.8],
                "priorities": [1],
                "gammas": [1.0],
                "jitter_factors": [0.0],
            },
            shuffle_streams=False,
        )

    after = len(ng.streams)
    added = ng.streams[before:after]

    if debug:
        print(f"[DEBUG] before={before}, after={after}, added_count={len(added)}")
        for i, s in enumerate(added):
            print(f"[DEBUG] added[{i}] = {s}")

    if not added:
        return None

    arrival_stream = copy.deepcopy(added[0])

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