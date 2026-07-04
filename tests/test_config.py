"""
Test configuration loading, parsing, and serialization.

Tests focus on UsecaseConfig:
- Loading from INI files
- Round-trip consistency (INI → object → INI)
- JSON serialization
- Error handling
"""

import pytest
from pathlib import Path
from usecase_config import UsecaseConfig


class TestConfigLoading:
    """Test UsecaseConfig.from_file() and configuration parsing"""
    
    def test_load_baseline_config(self, baseline_config):
        """Verify baseline config loads correctly with expected values"""
        assert baseline_config.num_hosts == 10
        assert baseline_config.num_switches == 5
        assert baseline_config.port_speed == 1000
        assert baseline_config.proc_delay == 1
        assert baseline_config.prop_delay == 10
        assert baseline_config.link_bw == 1000
        assert baseline_config.queue_count_per_port == 6
        
    def test_baseline_graph_config(self, baseline_config):
        """Verify topology graph config is correctly parsed"""
        assert baseline_config.graph_config is not None
        assert baseline_config.graph_config.get("typ") == "line"
        
    def test_baseline_streams_config(self, baseline_config):
        """Verify stream generation config is correctly parsed"""
        assert baseline_config.num_of_streams_per_instance == 5
        assert baseline_config.num_places_talker == 1
        assert baseline_config.num_places_listener == 1
        assert baseline_config.num_listeners_per_stream == 1
        
    def test_period_range_parsing(self, baseline_config):
        """Verify period range is parsed as dict with correct keys"""
        period = baseline_config.period_range
        assert isinstance(period, dict)
        assert period["start"] == 100
        assert period["stop"] == 300
        assert period["step"] == 200
        
    def test_datasize_range_parsing(self, baseline_config):
        """Verify data size range is correctly parsed"""
        datasize = baseline_config.datasize_range
        assert isinstance(datasize, dict)
        assert datasize["start"] == 64
        assert datasize["stop"] == 256
        
    def test_max_latency_range_as_list(self, baseline_config):
        """Verify max latency range is parsed as list of floats"""
        latency = baseline_config.max_latency_range
        assert isinstance(latency, list)
        assert len(latency) == 5
        assert latency == [0.5, 1.0, 2.0, 5.0, 10.0]
        
    def test_routing_config_parsing(self, baseline_config):
        """Verify routing config dict is correctly parsed"""
        routing = baseline_config.routing
        assert isinstance(routing, dict)
        assert routing["fixPath"] is True
        assert routing["shortestPath"] is True
        
    def test_stride_config_parsing(self, baseline_config):
        """Verify stride placement strategy is correctly parsed"""
        stride = baseline_config.stride
        assert stride["active"] is True
        assert stride["step"] == 5
        assert stride["increase_factor"] == 50
        
    def test_staggered_config_parsing(self, baseline_config):
        """Verify staggered placement strategy is correctly parsed"""
        staggered = baseline_config.staggered
        assert staggered["active"] is False
        
    def test_hotspot_config_parsing(self, baseline_config):
        """Verify hotspot placement strategy is correctly parsed"""
        hotspot = baseline_config.hotspot
        assert hotspot["active"] is False

class TestConfigRoundTrip:
    """Test INI round-trip consistency: INI → object → INI"""
    
    def test_ini_roundtrip_consistency(self, baseline_config, temp_config_file):
        """Load config, save to temp file, reload, verify equality"""
        # Save to temp file
        baseline_config.to_file(temp_config_file)
        
        # Reload from temp file
        reloaded_config = UsecaseConfig.from_file(temp_config_file)
        
        # Compare all topology fields
        assert reloaded_config.num_hosts == baseline_config.num_hosts
        assert reloaded_config.num_switches == baseline_config.num_switches
        assert reloaded_config.port_speed == baseline_config.port_speed
        assert reloaded_config.proc_delay == baseline_config.proc_delay
        assert reloaded_config.prop_delay == baseline_config.prop_delay
        assert reloaded_config.link_bw == baseline_config.link_bw
        assert reloaded_config.queue_count_per_port == baseline_config.queue_count_per_port
        
    def test_ini_roundtrip_streams_config(self, baseline_config, temp_config_file):
        """Verify stream config survives INI round-trip"""
        baseline_config.to_file(temp_config_file)
        reloaded = UsecaseConfig.from_file(temp_config_file)
        
        assert reloaded.num_of_streams_per_instance == baseline_config.num_of_streams_per_instance
        assert reloaded.period_range == baseline_config.period_range
        assert reloaded.numberframes_range == baseline_config.numberframes_range
        assert reloaded.datasize_range == baseline_config.datasize_range
        assert reloaded.num_places_talker == baseline_config.num_places_talker
        assert reloaded.num_places_listener == baseline_config.num_places_listener
        
    def test_ini_roundtrip_routing_config(self, baseline_config, temp_config_file):
        """Verify routing and placement strategy configs survive round-trip"""
        baseline_config.to_file(temp_config_file)
        reloaded = UsecaseConfig.from_file(temp_config_file)
        
        assert reloaded.routing == baseline_config.routing
        assert reloaded.stride == baseline_config.stride
        assert reloaded.staggered == baseline_config.staggered
        assert reloaded.hotspot == baseline_config.hotspot


class TestConfigSerialization:
    """Test JSON serialization and deserialization"""
    
    def test_json_roundtrip_topology(self, baseline_config, temp_json_file):
        """Serialize config to JSON, deserialize, verify topology fields"""
        json_str = baseline_config.to_json()
        reloaded = UsecaseConfig.from_json(json_str)
        
        assert reloaded.num_hosts == baseline_config.num_hosts
        assert reloaded.num_switches == baseline_config.num_switches
        assert reloaded.port_speed == baseline_config.port_speed
        assert reloaded.graph_config == baseline_config.graph_config
        
    def test_json_roundtrip_streams(self, baseline_config):
        """Verify stream config survives JSON round-trip"""
        json_str = baseline_config.to_json()
        reloaded = UsecaseConfig.from_json(json_str)
        
        assert reloaded.num_of_streams_per_instance == baseline_config.num_of_streams_per_instance
        assert reloaded.period_range == baseline_config.period_range
        assert reloaded.max_latency_range == baseline_config.max_latency_range
        
    def test_json_roundtrip_nested_dicts(self, baseline_config):
        """Verify nested dict structures survive JSON serialization"""
        json_str = baseline_config.to_json()
        reloaded = UsecaseConfig.from_json(json_str)
        
        # Check nested dictionaries
        assert reloaded.routing == baseline_config.routing
        assert reloaded.stride == baseline_config.stride
        assert reloaded.expiration == baseline_config.expiration
        assert reloaded.generation == baseline_config.generation


class TestConfigValidation:
    """Test configuration validation and error handling"""
    
    def test_config_missing_file_raises_error(self):
        """Verify FileNotFoundError when config file doesn't exist"""
        with pytest.raises(FileNotFoundError):
            UsecaseConfig.from_file("/nonexistent/path/config.ini")
            
    def test_config_all_fields_populated(self, baseline_config):
        """Verify all required fields are populated in baseline config"""
        # Topology fields
        assert baseline_config.num_hosts > 0
        assert baseline_config.num_switches > 0
        assert baseline_config.port_speed > 0
        assert baseline_config.proc_delay >= 0
        assert baseline_config.prop_delay >= 0
        assert baseline_config.link_bw > 0
        
        # Streams fields
        assert baseline_config.num_of_streams_per_instance > 0
        assert baseline_config.num_places_talker > 0
        assert baseline_config.num_places_listener > 0
        assert len(baseline_config.period_range) > 0
        assert len(baseline_config.numberframes_range) > 0
        assert len(baseline_config.datasize_range) > 0
        assert len(baseline_config.max_latency_range) > 0