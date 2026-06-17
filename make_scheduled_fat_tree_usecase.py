# make_scheduled_fat_tree_usecase.py

import argparse
import json
import random
from pathlib import Path

import networkx as nx

from usecase_generator import UsecaseGenerator
from tsn_scheduling_problem import TSNScheduling
from schedule import Schedule


ED_ED_SETTINGS = {
    # periods are in microseconds here, then converted to ns
    "packet_sizes": [64, 128, 256, 512],
    "periods_us": [500, 1000],
    "latency_factors": [0.7, 0.8, 0.9, 1.0],
    "priorities": [1],
    "gammas": [1.0],
    "jitter_factors": [0.0],
}


def pod_of(node_name: str):
    parts = node_name.split("-")
    for p in parts:
        if p.startswith("p") and p[1:].isdigit():
            return int(p[1:])
    return None


def edge_switch_of_host(usecase, host: str):
    g = usecase.graph

    neighbors = set(g.successors(host)) | set(g.predecessors(host))

    for n in neighbors:
        data = g.nodes[n]
        if data.get("type") == "switch" and data.get("layer") == "edge":
            return n

    return None


def hosts_grouped_by_edge_switch(usecase):
    result = {}

    for n, data in usecase.graph.nodes(data=True):
        if data.get("type") != "host":
            continue

        edge_sw = edge_switch_of_host(usecase, n)
        if edge_sw is not None:
            result.setdefault(edge_sw, []).append(n)

    return result


def shortest_port_path(usecase, src_host: str, dst_host: str):
    """
    Create scheduler-compatible pathLists entry:
        [egress_port_1, egress_port_2, ..., dst_host]

    This follows the same format as UsecaseGenerator.shortestPath().
    """
    node_path = nx.shortest_path(usecase.graph, src_host, dst_host)

    port_path = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        edge_data = usecase.graph.get_edge_data(u, v)
        port_path.append(edge_data["sourcePort"])

    port_path.append(dst_host)
    return port_path


def make_stream(stream_id: int, src: str, dst: str, path_list, settings):
    period_us = random.choice(settings["periods_us"])
    period_ns = int(period_us * 1000)

    packet_size = random.choice(settings["packet_sizes"])
    latency_factor = random.choice(settings["latency_factors"])
    priority = random.choice(settings["priorities"])
    gamma = random.choice(settings["gammas"])
    jitter_factor = random.choice(settings["jitter_factors"])

    return {
        "name": f"stream-{stream_id}",
        "type": "isochronous",

        # ns
        "interval": int(period_ns),

        "maxFramesPerInterval": 1,
        "maxFrameSize": int(packet_size),

        # ns
        "maxLatency": int(period_ns * latency_factor),

        "tlkrPssblPlcmnt": [src],
        "lstnsrPssblPlcmnt": [[dst]],

        "earliest-transmit-offset": 0,
        "latest-transmit-offset": int(period_ns - 1),

        "jitter": int(period_ns * jitter_factor),

        # fixed one-path scheduling
        "pathLists": [path_list],

        "priority": int(priority),
        "gamma": float(gamma),
    }


def generate_candidate_stream(usecase, edge_to_hosts, edge_switches, stream_id, cross_pod_only=True):
    for _ in range(1000):
        src_edge, dst_edge = random.sample(edge_switches, 2)

        if cross_pod_only and pod_of(src_edge) == pod_of(dst_edge):
            continue

        src = random.choice(edge_to_hosts[src_edge])
        dst = random.choice(edge_to_hosts[dst_edge])

        if src == dst:
            continue

        path_list = shortest_port_path(usecase, src, dst)

        return make_stream(
            stream_id=stream_id,
            src=src,
            dst=dst,
            path_list=path_list,
            settings=ED_ED_SETTINGS,
        )

    raise RuntimeError("Could not generate a valid candidate stream.")


def make_scheduled_fat_tree_usecase(
    config_path: str,
    output_path: str,
    target_streams: int = 72,
    seed: int = 2004,
    cross_pod_only: bool = True,
    max_attempts: int = 5000,
    max_candidates_per_hop: int = 5,
    beam_width: int = 30,
):
    random.seed(seed)

    usecase = UsecaseGenerator.empty_uc(config_path)

    # Force scheduling-only assumptions
    usecase.config.set("streams_config", "numofstreamsperinstance", str(target_streams))
    usecase.config.set("streams_config", "numplacestalker", "1")
    usecase.config.set("streams_config", "numplaceslistener", "1")
    usecase.config.set("streams_config", "numlistenersperstr", "1")
    usecase.config.set(
        "streams_config",
        "routing",
        "{'fixPath': True, 'shortestPath': True}",
    )

    edge_to_hosts = hosts_grouped_by_edge_switch(usecase)
    edge_switches = sorted(edge_to_hosts.keys())

    if len(edge_switches) < 2:
        raise RuntimeError("Need at least two edge switches with hosts.")

    problem = TSNScheduling(config_path)
    problem.usecase = usecase

    # Macrocycle for periods [500000, 1000000] ns is 1000000 ns.
    # We compute it generally from the settings.
    cycle_ns = 1
    for p_us in ED_ED_SETTINGS["periods_us"]:
        p_ns = int(p_us * 1000)
        import math
        cycle_ns = cycle_ns * p_ns // math.gcd(cycle_ns, p_ns)

    schedule = Schedule(cycle_ns=cycle_ns, basetime=0)

    usecase.streams = []
    accepted = 0
    rejected = 0
    attempts = 0

    while accepted < target_streams and attempts < max_attempts:
        attempts += 1

        stream_id = len(usecase.streams)

        try:
            stream = generate_candidate_stream(
                usecase=usecase,
                edge_to_hosts=edge_to_hosts,
                edge_switches=edge_switches,
                stream_id=stream_id,
                cross_pod_only=cross_pod_only,
            )
        except Exception:
            rejected += 1
            continue

        usecase.streams.append(stream)

        result = schedule.place_existing_stream_first_fit_fast(
            stream_id=stream_id,
            usecase=usecase,
            problem=problem,
            queue_id=5,
            port_intervals=None,
            max_candidates_per_hop=max_candidates_per_hop,
            beam_width=beam_width,
        )

        if result.get("accepted"):
            accepted += 1
            print(f"[ACCEPT] {accepted}/{target_streams} stream_id={stream_id} starts={result.get('starts')}")
        else:
            # Remove failed stream definition and keep stream IDs compact.
            usecase.streams.pop()
            rejected += 1
            print(f"[REJECT] attempt={attempts}, accepted={accepted}, rejected={rejected}")

    if accepted < target_streams:
        raise RuntimeError(
            f"Only scheduled {accepted}/{target_streams} streams after {attempts} attempts."
        )

    # Make sure final schedule is attached and graph ports are updated.
    usecase.schedule = schedule
    _, port_id_to_name = usecase.port_name_id_mapping
    schedule.update_graph(usecase.graph, port_id_to_name)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    usecase.to_file(
        jsnPath=str(output_path.parent) + "/",
        ucFileName=output_path.name,
    )

    hosts = [n for n, d in usecase.graph.nodes(data=True) if d.get("type") == "host"]
    switches = [n for n, d in usecase.graph.nodes(data=True) if d.get("type") == "switch"]

    print("\nDone.")
    print("Saved:", output_path)
    print("Switches:", len(switches))
    print("Hosts:", len(hosts))
    print("Links:", usecase.graph.number_of_edges())
    print("Scheduled streams:", len(usecase.streams))
    print("Schedule slots:", len(schedule))
    print("Cycle ns:", schedule.cycle_ns)
    print("Rejected generation/placement attempts:", rejected)
    print("Periods ns:", sorted({s["interval"] for s in usecase.streams}))
    print("Packet sizes:", sorted({s["maxFrameSize"] for s in usecase.streams}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="uc_config.cfg")
    parser.add_argument("--out", default="finals_fattree_k4_72_scheduled.json")
    parser.add_argument("--n-streams", type=int, default=72)
    parser.add_argument("--seed", type=int, default=2004)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--max-candidates-per-hop", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=30)

    parser.add_argument(
        "--allow-same-pod",
        action="store_true",
        help="Allow same-pod traffic. Default is cross-pod only.",
    )

    args = parser.parse_args()

    make_scheduled_fat_tree_usecase(
        config_path=args.config,
        output_path=args.out,
        target_streams=args.n_streams,
        seed=args.seed,
        cross_pod_only=not args.allow_same_pod,
        max_attempts=args.max_attempts,
        max_candidates_per_hop=args.max_candidates_per_hop,
        beam_width=args.beam_width,
    )