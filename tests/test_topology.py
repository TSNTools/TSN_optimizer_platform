import networkx as nx


def test_line_topology_node_count(baseline_topology_graph):
    graph = baseline_topology_graph

    hosts = [node for node in graph.nodes if node.startswith("host-")]
    switches = [node for node in graph.nodes if node.startswith("switch-c")]

    assert len(hosts) == 10
    assert len(switches) == 5

    assert set(hosts) == {f"host-{i}" for i in range(10)}
    assert set(switches) == {f"switch-c{i}" for i in range(5)}


def test_line_topology_switch_chain(baseline_topology_graph):
    graph = baseline_topology_graph
    switches = [f"switch-c{i}" for i in range(5)]

    switch_subgraph = nx.Graph(graph.subgraph(switches))
    assert nx.is_connected(switch_subgraph)

    expected_edges = {(f"switch-c{i}", f"switch-c{i+1}") for i in range(4)}
    actual_edges = {tuple(sorted(edge)) for edge in switch_subgraph.edges()}
    assert actual_edges == expected_edges


def test_line_topology_host_attachment(baseline_topology_graph):
    graph = baseline_topology_graph

    for host_index in range(10):
        host_name = f"host-{host_index}"
        assert host_name in graph

        neighbors = list(graph.neighbors(host_name))
        assert len(neighbors) >= 1
        assert all(neighbor.startswith("switch-c") for neighbor in neighbors)

        node_data = graph.nodes[host_name]
        assert node_data["type"] == "host"
        assert "ingress_ports" in node_data
        assert "egress_ports" in node_data


def test_line_topology_port_metadata(baseline_topology_graph):
    graph = baseline_topology_graph

    for node, _, edge_data in graph.edges(data=True):
        assert edge_data["sourceNode"] == node
        assert "destNode" in edge_data
        assert "sourcePort" in edge_data
        assert "destPort" in edge_data
        assert edge_data["propDelay"] == 10
        assert edge_data["linkBw"] == 1000

    for node, node_data in graph.nodes(data=True):
        assert "ingress_ports" in node_data
        assert "egress_ports" in node_data

        if node.startswith("switch-c"):
            assert node_data["type"] == "switch"
            assert node_data["procDelay"] == 1000