import json

import pytest
import random
from pathlib import Path
from usecase_config import UsecaseConfig
from usecase import Usecase
from topology_handler import TopologyHandler


@pytest.fixture(scope="session")
def baseline_config():
    """Load baseline UsecaseConfig from test_data/baseline_config.ini
    
    Configuration details:
    - Topology: line graph with 10 hosts and 5 switches
    - Streams: 5 auto-generated streams
    - Periods: 100μs, 200μs, 300μs (from period_range)
    - Frame sizes: 64-256 bytes
    - Max latencies: [0.5, 1, 2, 5, 10] ms
    """
    # Find the test_data directory relative to this conftest.py file
    test_dir = Path(__file__).parent
    config_path = test_dir / "test_data" / "baseline_config.ini"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Baseline config not found at {config_path}. "
            f"Please ensure test_data/baseline_config.ini exists in the project root."
        )
    
    return UsecaseConfig.from_file(str(config_path))


@pytest.fixture
def baseline_fresh_usecase(baseline_config):
    """Create Usecase with baseline config (5 streams, line topology)
    
    This is the primary test subject: a usecase with:
    - 5 streams auto-generated with deterministic parameters
    - Line topology (5 switches, 10 hosts)
    - Shortest path routing enabled
    """
    random.seed(0)
    return Usecase(specification=baseline_config)

@pytest.fixture
def baseline_usecase():
    file_path = "tests/test_data/test_usecase.json"
    imp_uc = json.load(open(file_path))
    return Usecase.from_json(imp_uc)


@pytest.fixture
def baseline_topology_graph(baseline_config):
    """Generate topology graph (no streams)
    
    Useful for testing topology generation in isolation.
    """
    random.seed(0)
    return TopologyHandler.makeTopology(baseline_config)


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for round-trip tests
    
    Usage:
        def test_roundtrip(baseline_config, temp_config_file):
            baseline_config.to_file(temp_config_file)
            reloaded = UsecaseConfig.from_file(temp_config_file)
    """
    cfg_path = tmp_path / "test_config.ini"
    yield str(cfg_path)
    # Cleanup handled by tmp_path fixture


@pytest.fixture
def temp_json_file(tmp_path):
    """Create a temporary JSON file for serialization tests"""
    json_path = tmp_path / "test_data.json"
    yield str(json_path)
    # Cleanup handled by tmp_path fixture