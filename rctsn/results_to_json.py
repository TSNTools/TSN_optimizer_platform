import pickle
import csv
import json
import networkx as nx
from collections import defaultdict

# Load pickle files
with open('/home/ws/rctsn/data/output/run_20260225_102229_46c86ade/variables_a.pkl', 'rb') as f:
    a = pickle.load(f)          # stream activation: {stream_idx: 1 if scheduled}

with open('/home/ws/rctsn/data/output/run_20260225_102229_46c86ade/variables_z.pkl', 'rb') as f:
    z = pickle.load(f)          # path selection: {(stream, path): 1 if used}

with open('/home/ws/rctsn/data/output/run_20260225_102229_46c86ade/variables_x.pkl', 'rb') as f:
    x = pickle.load(f)          # start times: {(stream, path, rep, (src, dst)): time (µs)}

# Load network graph from .graphml
G = nx.read_graphml('/home/ws/rctsn/data/input/network.graphml')
# Determine link speed from graph attributes (default 100 Mbps)
# Look for a 'bandwidth' or 'linkBw' attribute on edges
link_speed_mbps = 100  # default from link_config.yaml
# Attempt to get from first edge attribute
if G.number_of_edges() > 0:
    first_edge = list(G.edges(data=True))[0]
    if 'bandwidth' in first_edge[2]:
        link_speed_mbps = int(first_edge[2]['bandwidth'])
    elif 'linkBw' in first_edge[2]:
        link_speed_mbps = int(first_edge[2]['linkBw'])

# Conversion: duration_ns = packet_size_bytes * 8 / (speed_bits_per_ns)
# speed_bits_per_ns = link_speed_mbps / 1000? Actually 1 Mbps = 1e6 bits/s = 0.001 bits/ns
# So bits per ns = link_speed_mbps * 1e6 / 1e9 = link_speed_mbps / 1000.
# Thus duration_ns = (packet_size * 8) / (link_speed_mbps / 1000) = packet_size * 8 * 1000 / link_speed_mbps
# With link_speed_mbps = 100, duration_ns = packet_size * 8 * 1000 / 100 = packet_size * 80
def calc_duration(packet_size_bytes):
    return int(packet_size_bytes * 8 * 1000 / link_speed_mbps)

# Build nodes list and port mappings
nodes_json = []
node_name_to_obj = {}
switch_ports = defaultdict(dict)   # node -> {neighbor: port_name}
node_neighbors = {node: [] for node in G.nodes()}

# First pass: collect neighbors for each node
for u, v in G.edges():
    node_neighbors[u].append(v)
    node_neighbors[v].append(u)

# Build nodes and assign ports
for node, neighbors in node_neighbors.items():
    node_type = 'host' if 'ED' in node else 'switch'
    ports = {}
    if node_type == 'host':
        ports['Port1'] = {"name": "Port1", "speed": link_speed_mbps, "queueCount": 4}
    else:
        # Sort neighbors for consistent port naming
        sorted_neighbors = sorted(neighbors)
        for i, neighbor in enumerate(sorted_neighbors):
            port_name = f'P{i+1}'
            ports[port_name] = {"name": port_name, "speed": link_speed_mbps, "queueCount": 4}
            switch_ports[node][neighbor] = port_name
    node_entry = {
        "type": node_type,
        "name": node,
        "ports": ports
    }
    nodes_json.append(node_entry)
    node_name_to_obj[node] = node_entry

# Build links list
links_json = []
for u, v, attrs in G.edges(data=True):
    prop_delay = int(attrs.get('propDelay', 10))  # default 10 ns
    link_bw = int(attrs.get('linkBw', link_speed_mbps))
    # Determine ports based on node types
    if 'ED' in u:
        src_port = 'Port1'
    else:
        src_port = switch_ports[u][v]
    if 'ED' in v:
        dst_port = 'Port1'
    else:
        dst_port = switch_ports[v][u]
    links_json.append({
        "sourceNode": u,
        "sourcePort": src_port,
        "destNode": v,
        "destPort": dst_port,
        "propDelay": prop_delay,
        "linkBw": link_bw
    })

# Read streams.csv to get stream parameters
streams_info = {}
with open('/home/ws/rctsn/data/input/streams.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        idx = int(row['index'])
        streams_info[idx] = {
            'src': row['src'],
            'dst': row['dst'],
            'packet_size': int(row['packet_size']),
            'period': int(row['period']),
            'max_latency': int(row['max_latency']),
            'priority': int(row['priority']),
            'gamma': float(row['gamma']),
            'max_jitter': int(row['max_jitter'])
        }

# Determine selected path for each stream from z
selected_path = {}
for key, val in z.items():
    if val == 1:
        if isinstance(key, tuple):
            if len(key) == 2:
                s, u = key
                selected_path[s] = u
            elif len(key) == 3:
                s, u, _ = key
                selected_path[s] = u

# Group x entries by (stream, path) for first repetition (r=1)
path_hops = defaultdict(list)
for (s, u, r, (src, dst)), time_us in x.items():
    if r == 1:
        path_hops[(s, u)].append((src, dst, time_us))

# Prepare schedule entries for each switch port
schedule_entries = defaultdict(list)  # key: (node, port) -> list of (start_ns, duration, stream_name)

# Build streams list and collect schedule data
streams_json = []
for s in streams_info:
    if a.get(s, 0) == 0:
        continue
    if s not in selected_path:
        # If no path in z, try to infer from path_hops (should be only one)
        candidates = [u for (ss, u) in path_hops if ss == s]
        if len(candidates) == 1:
            u = candidates[0]
        else:
            print(f"Warning: cannot determine path for stream {s}, skipping")
            continue
    else:
        u = selected_path[s]

    if (s, u) not in path_hops:
        print(f"Warning: no hops found for stream {s} path {u}, skipping")
        continue

    hops = path_hops[(s, u)]
    hops.sort(key=lambda x: x[2])   # sort by start time (µs)

    # Build path list in "Node.Port" format
    talker = streams_info[s]['src']
    listener = streams_info[s]['dst']
    path_list = [f"{talker}.Port1"]

    # Process each hop to collect schedule info and build path
    for i, (src_node, dst_node, time_us) in enumerate(hops):
        # Find egress port for src_node (must be a switch)
        if 'SW' in src_node:
            port_name = switch_ports[src_node][dst_node]
            # Convert time to nanoseconds
            start_ns = int(time_us * 1000)
            duration_ns = calc_duration(streams_info[s]['packet_size'])
            schedule_entries[(src_node, port_name)].append((start_ns, duration_ns, f"Stream-{s}"))
        else:
            # src_node is host? should not happen except first hop where src is talker
            # But we already added talker port separately
            pass

        # Add to path list if this hop involves a switch egress (skip last hop to listener)
        if i < len(hops) - 1:
            if 'SW' in src_node:
                path_list.append(f"{src_node}.{port_name}")
        else:
            path_list.append(f"{listener}.Port1")

    offset_ns = int(hops[0][2] * 1000)  # first hop time in ns

    info = streams_info[s]
    stream_json = {
        "name": f"Stream-{s}",
        "type": "isochronous",
        "interval": info['period'],
        "maxFramesPerInterval": 1,
        "maxFrameSize": info['packet_size'],
        "maxLatency": info['max_latency'],
        "tlkrPssblPlcmnt": [talker],
        "lstnsrPssblPlcmnt": [[listener]],
        "earliest-transmit-offset": 0,
        "latest-transmit-offset": info['period'] - (info['packet_size'] * 8 * 1000 // link_speed_mbps),
        "jitter": info['max_jitter'],
        "offset": offset_ns,
        "accLatency": 0,
        "pathLists": [path_list]
    }
    streams_json.append(stream_json)

# Now populate the schedule arrays in the nodes
for node_obj in nodes_json:
    if node_obj['type'] == 'switch':
        for port_name, port_info in node_obj['ports'].items():
            port_info['schedule'] = []
            # Add schedule entries for this port
            entries = schedule_entries.get((node_obj['name'], port_name), [])
            # Sort by start time
            entries.sort(key=lambda x: x[0])
            for start_ns, duration_ns, stream_name in entries:
                port_info['schedule'].append({
                    "start": start_ns,
                    "duration": duration_ns,
                    "state": "CoCC"
                })
            # If no entries, schedule can be empty list (optional)
    else:
        # hosts don't have schedules
        pass

# Assemble final JSON
output_json = {
    "nodes": nodes_json,
    "links": links_json,
    "streams": streams_json,
    "index": None
}

# Save to file
with open('ilp_schedule.json', 'w') as f:
    json.dump(output_json, f, indent=2)

print(f"Converted {len(streams_json)} streams. JSON saved to ilp_schedule.json")