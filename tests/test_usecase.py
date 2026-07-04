import pytest


def test_usecase_instantiates_from_specification(baseline_config):
    from usecase import Usecase

    uc = Usecase(specification=baseline_config)

    assert uc.specification is baseline_config
    assert uc.graph is not None
    assert uc.n_streams == 5
    assert len(uc.streams) == 5
    assert uc.index > 0


def test_usecase_properties(baseline_usecase):
    uc = baseline_usecase

    assert uc.n_streams == 5
    assert len(uc.hosts) == 10
    assert len(uc.switches) == 5
    assert uc.macro_cycle > 0
    assert isinstance(uc.ports, dict)
    assert len(uc.ports) > 0

    assert set(uc.hosts) == {f"host-{i}" for i in range(10)}
    assert set(uc.switches) == {f"switch-c{i}" for i in range(5)}


def test_usecase_from_file(baseline_config):
    from usecase import Usecase
    from pathlib import Path

    config_path = Path(__file__).parent / "test_data" / "baseline_config.ini"
    uc = Usecase(specsFile=str(config_path))

    assert uc.n_streams == 5
    assert len(uc.hosts) == 10
    assert len(uc.switches) == 5
    assert uc.specification.graph_config["typ"] == "line"


def test_usecase_graph_and_stream_consistency(baseline_usecase):
    uc = baseline_usecase
    graph = uc.graph

    for stream in uc.streams:
        talker = stream.tlkrPssblPlcmnt[0]
        listener = stream.lstnsrPssblPlcmnt[0][0]

        assert talker in graph
        assert listener in graph
        assert talker != listener

        assert stream.is_routed
        assert stream.pathLists is not None
        assert len(stream.pathLists) >= 1

        for path in stream.pathLists:
            assert path[-1].startswith("host-")
            assert "." in path[-2]
            node, port = path[-2].split(".")
            assert port in graph.nodes[node]["egress_ports"]
            assert graph.nodes[node]["egress_ports"][port]["destNode"] == path[-1]