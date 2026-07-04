"""
Test stream generation, placement, and routing for the baseline 5-stream line usecase.
"""

def test_baseline_usecase_generates_five_streams(baseline_usecase):
    assert baseline_usecase.n_streams == 5
    assert len(baseline_usecase.streams) == 5


def test_stream_fields_are_valid(baseline_usecase):
    valid_latencies = {
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000
    }

    for stream in baseline_usecase.streams:
        assert stream.interval > 0
        assert 1 <= stream.maxFramesPerInterval <= 2
        assert 64 <= stream.maxFrameSize <= 255
        assert stream.maxLatency in valid_latencies

        assert isinstance(stream.tlkrPssblPlcmnt, list)
        assert len(stream.tlkrPssblPlcmnt) == 1
        assert isinstance(stream.lstnsrPssblPlcmnt, list)
        assert len(stream.lstnsrPssblPlcmnt) == 1
        assert len(stream.lstnsrPssblPlcmnt[0]) == 1

        assert stream.is_routed
        assert stream.customGraph is not None


def test_stream_endpoints_exist_in_graph(baseline_usecase):
    graph = baseline_usecase.graph

    for stream in baseline_usecase.streams:
        talker = stream.tlkrPssblPlcmnt[0]
        listener = stream.lstnsrPssblPlcmnt[0][0]

        assert talker in graph
        assert listener in graph
        assert talker != listener


def test_stream_paths_are_valid(baseline_usecase):
    graph = baseline_usecase.graph

    for stream in baseline_usecase.streams:
        assert stream.pathLists, "Each stream must have computed pathLists"

        for path in stream.pathLists:
            assert isinstance(path, list)
            assert path[-1].startswith("host-")

            # Every hop before the final listener must be a valid egress port
            for hop in path[:-1]:
                assert "." in hop
                node, port = hop.split(".")
                assert node in graph
                assert port in graph.nodes[node]["egress_ports"]

            # Verify the last egress hop points to the listener node
            final_hop = path[-2]
            node, port = final_hop.split(".")
            port_data = graph.nodes[node]["egress_ports"][port]
            assert port_data["destNode"] == path[-1]


def test_stream_generate_uses_config_ranges(baseline_config):
    from stream import Stream

    stream = Stream.generate(baseline_config)
    assert stream.interval % 1000 == 0
    assert 1 <= stream.maxFramesPerInterval <= 2
    assert 64 <= stream.maxFrameSize <= 255
    assert stream.maxLatency in {
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000
    }