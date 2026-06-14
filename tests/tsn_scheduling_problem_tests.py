import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest

from tsn_scheduling_problem import (
    BYTE_2_BIT,
    MBPS_2_BPNS,
    TOL,
    InstanceContext,
    TSNScheduling,
    require_instance,
)


# ------------------------------------------------------------------
# Helpers / Fixtures
# ------------------------------------------------------------------


class DummyConfig:
    def getint(self, section, key):
        values = {
            ("streams_config", "numplacesTalker"): 1,
            ("streams_config", "numplaceslistener"): 1,
        }
        return values[(section, key)]

    def get(self, section, key):
        if (section, key) == ("streams_config", "routing"):
            return "{'fixPath': True}"
        raise KeyError(key)


class DummyUsecase:
    def __init__(self):
        self.config = DummyConfig()

        self.ports = {
            "host1.Port1": {
                "destPort": "switch1.Port1",
                "sourceNode": "host1",
                "destNode": "switch1",
                "speed": 1000,
            },
            "switch1.Port1": {
                "destPort": "host2.Port1",
                "sourceNode": "switch1",
                "destNode": "host2",
                "speed": 1000,
            },
        }

        self.links = [
            {
                "sourcePort": "host1.Port1",
                "destPort": "switch1.Port1",
            }
        ]

        self.hosts = ["host1", "host2"]

        self.streams = [
            {
                "interval": 1000,
                "maxFramesPerInterval": 1,
                "maxFrameSize": 100,
                "maxLatency": 100000,
                "tlkrPssblPlcmnt": ["host1"],
                "lstnsrPssblPlcmnt": [["host2"]],
            }
        ]

        self.graph = nx.DiGraph()
        self.graph.add_node("host1", procDelay=0)
        self.graph.add_node("switch1", procDelay=5)
        self.graph.add_node("host2", procDelay=0)

        self.graph.add_edge(
            "host1",
            "switch1",
            sourcePort="host1.Port1",
            destPort="switch1.Port1",
            propDelay=10,
        )

        self.graph.add_edge(
            "switch1",
            "host2",
            sourcePort="switch1.Port1",
            destPort="host2.Port1",
            propDelay=10,
        )

    def get_config(self):
        return {"dummy": True}


@pytest.fixture
def problem(monkeypatch):
    monkeypatch.setattr(
        "tsn_scheduling_problem.UsecaseGenerator",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "tsn_scheduling_problem.InstancesLoader",
        lambda: MagicMock(),
    )

    p = TSNScheduling("dummy.json")
    p.usecase = DummyUsecase()
    return p


# ------------------------------------------------------------------
# Decorator Tests
# ------------------------------------------------------------------


def test_require_instance_raises_when_no_usecase():
    class Dummy:
        usecase = None

        def no_instance_assigned(self):
            raise ValueError("No instance")

        @require_instance
        def method(self):
            return True

    with pytest.raises(ValueError, match="No instance"):
        Dummy().method()


# ------------------------------------------------------------------
# Context Manager Tests
# ------------------------------------------------------------------


def test_instance_context_sets_and_clears_usecase(problem):
    uc = DummyUsecase()

    problem.build_metadata = MagicMock()

    with problem.use_instance(uc):
        assert problem.usecase is uc

    assert problem.usecase is None



def test_instance_context_builds_metadata_once(problem):
    uc = DummyUsecase()

    problem.build_metadata = MagicMock()

    with problem.use_instance(uc):
        pass

    with problem.use_instance(uc):
        pass

    assert problem.build_metadata.call_count == 1


# ------------------------------------------------------------------
# Property Tests
# ------------------------------------------------------------------


def test_n_ports(problem):
    assert problem.n_ports == 2



def test_n_streams(problem):
    assert problem.n_streams == 1



def test_port_name_to_id(problem):
    mapping = problem.port_name_to_id

    assert mapping["host1.Port1"] == 1
    assert mapping["switch1.Port1"] == 2
    assert mapping[""] == 0



def test_port_id_to_name(problem):
    mapping = problem.port_id_to_name

    assert mapping[1] == "host1.Port1"
    assert mapping[2] == "switch1.Port1"
    assert mapping[0] == ""



def test_macro_cycle(problem):
    assert problem.macro_cycle == 1000



def test_streams_property_maps_port_ids(problem):
    streams = problem.streams
    stream = streams[0]

    assert stream["tlkrPssblPlcmnt"] == [1]
    assert stream["lstnsrPssblPlcmnt"] == [[2]]


# ------------------------------------------------------------------
# Static / Utility Tests
# ------------------------------------------------------------------


def test_frame_size():
    stream = {
        "maxFramesPerInterval": 2,
        "maxFrameSize": 100,
    }

    result = TSNScheduling._frame_size(stream)

    expected = 2 * 100 * BYTE_2_BIT * TOL

    assert result == expected



def test_hops_costs(problem):
    stream = problem.usecase.streams[0]
    frame_size = TSNScheduling._frame_size(stream)

    costs = problem.hops_costs(frame_size)

    assert isinstance(costs, np.ndarray)
    assert len(costs) == 2
    assert np.all(costs > 0)



def test_hop_cost_eg2eg(problem):
    stream = problem.usecase.streams[0]
    frame_size = TSNScheduling._frame_size(stream)

    result = problem.hop_cost_eg2eg(
        "host1.Port1",
        "switch1.Port1",
        frame_size,
    )

    expected_trans_delay = frame_size / (1000 * MBPS_2_BPNS)
    expected = expected_trans_delay + 10 + 5

    assert math.isclose(result, expected)


# ------------------------------------------------------------------
# Graph / BFS Tests
# ------------------------------------------------------------------


def test_network_borders(problem):
    borders = problem.network_borders(problem.usecase.graph)

    assert "switch1.Port1" in borders



def test_bfs_forward(problem):
    stream = problem.usecase.streams[0]

    result = problem.bfs_forward(
        problem.usecase.graph,
        "host1.Port1",
        stream,
    )

    assert isinstance(result, dict)
    assert "switch1.Port1" in result



def test_bfs_backward(problem):
    stream = problem.usecase.streams[0]

    result = problem.bfs_backward(
        problem.usecase.graph,
        "host2.Port1",
        stream,
    )

    assert isinstance(result, dict)


# ------------------------------------------------------------------
# Configuration Tests
# ------------------------------------------------------------------


def test_get_task(problem):
    routing_fixed, placement_fixed = problem.get_task()

    assert routing_fixed is True
    assert placement_fixed is True



def test_get_concurrent_ports_invalid_type(problem):
    with pytest.raises(ValueError):
        problem.get_concurrent_ports(["host1.Port1"], typ="invalid")



def test_configure_stream_ports(problem):
    ports_in = {
        "talker": ["host1.Port1"],
        "listener": [],
        "switch": [],
    }

    ports_out = {
        "talker": [],
        "listener": [],
        "switch": [],
    }

    config_column, custom_graph = problem.configure_stream_ports(
        ports_in,
        ports_out,
    )

    assert isinstance(config_column, np.ndarray)
    assert config_column.shape[0] == problem.n_ports
    assert isinstance(custom_graph, nx.DiGraph)



def test_make_default_stream_config(problem):
    stream = problem.usecase.streams[0]

    config_column, custom_graph = problem.make_default_stream_config(stream)

    assert isinstance(config_column, np.ndarray)
    assert isinstance(custom_graph, nx.DiGraph)



def test_make_stream_config(problem):
    stream = problem.usecase.streams[0]

    config_column, custom_graph = problem.make_stream_config(
        stream,
        always_in=[0],
        always_out=[],
    )

    assert isinstance(config_column, np.ndarray)
    assert isinstance(custom_graph, nx.DiGraph)


# ------------------------------------------------------------------
# Metadata / Initialization Tests
# ------------------------------------------------------------------


def test_build_metadata(problem, monkeypatch):
    mock_config = np.array([1, 0])
    mock_graph = nx.DiGraph()

    monkeypatch.setattr(
        problem,
        "make_default_stream_config",
        lambda stream: (mock_config, mock_graph),
    )

    problem.initialize_problem = MagicMock()

    problem.build_metadata()

    assert problem.config_matrix.shape == (2, 1)
    assert "customGraph" in problem.usecase.streams[0]
    problem.initialize_problem.assert_called_once()



def test_build_metadata_invalid_column_size(problem, monkeypatch):
    monkeypatch.setattr(
        problem,
        "make_default_stream_config",
        lambda stream: (np.array([1]), nx.DiGraph()),
    )

    with pytest.raises(ValueError, match="config column length"):
        problem.build_metadata()



def test_initialize_problem(problem, monkeypatch):
    evaluator_mock = MagicMock()

    monkeypatch.setattr(
        "tsn_scheduling_problem.TSNConstraintEvaluator",
        evaluator_mock,
        raising=False,
    )

    problem.initialize_problem()

    evaluator_mock.assert_called_once()