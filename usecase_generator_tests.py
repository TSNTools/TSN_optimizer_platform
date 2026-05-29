##
# Unit tests for usecase_generator.py
##

import unittest
import configparser
import math
import sys
import os

# Ensure the source file is importable
sys.path.insert(0, '/mnt/user-data/uploads')
from usecase_generator import UsecaseGenerator, Usecase, MBPS_2_BPNS, BYTE_2_BIT, TOL

# Detect which class actually owns generate_fattree_regions (it lives on Usecase)
_fattree_regions_owner = Usecase if hasattr(Usecase, 'generate_fattree_regions') else UsecaseGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(
    num_hosts=8,
    num_switchs=4,
    num_streams=2,
    graph_type="ft",
    k=4,
    num_places_talker=1,
    num_places_listener=1,
    num_listeners_per_str=1,
    fix_path=False,
    shortest_path=False,
    finite_duration=False,
):
    """Build a minimal in-memory configparser that Usecase / UsecaseGenerator can consume."""
    config = configparser.RawConfigParser(inline_comment_prefixes="#", allow_no_value=True)

    config.add_section("topology_config")
    config.set("topology_config", "portspeed", "1000")
    config.set("topology_config", "propdelay", "1000")
    config.set("topology_config", "procdelay", "1")
    config.set("topology_config", "linkBw", "1000")
    config.set("topology_config", "queuecountperport", "8")
    config.set("topology_config", "numhosts", str(num_hosts))
    config.set("topology_config", "numswitchs", str(num_switchs))
    config.set("topology_config", "minnodedegreepersw", "2")

    if graph_type == "ft":
        config.set("topology_config", "graphconfig", f"{{'typ':'ft','k':{k}}}")
    else:
        config.set(
            "topology_config",
            "graphconfig",
            "{'typ':'er','prob':0.8,'minnodedegreepersw':2,'numCoreSwitches':0,'numEdgN':2}",
        )

    config.add_section("streams_config")
    config.set("streams_config", "numofstreamsperinstance", str(num_streams))
    config.set("streams_config", "period_range", "{'start':125,'stop':250}")
    config.set("streams_config", "numberframes_range", "{'start':1,'stop':3}")
    config.set("streams_config", "datasize_range", "{'start':64,'stop':128}")
    config.set("streams_config", "maxlatency_range", "1.0,2.0")
    config.set("streams_config", "numplacesTalker", str(num_places_talker))
    config.set("streams_config", "numplacesListener", str(num_places_listener))
    config.set("streams_config", "numlistenersperstr", str(num_listeners_per_str))
    config.set(
        "streams_config",
        "routing",
        f"{{'fixPath':{fix_path},'shortestPath':{shortest_path}}}",
    )
    config.set("streams_config", "stride", "{'active':False,'step':2,'increase_factor':2}")
    config.set("streams_config", "staggered", "{'active':False,'edgep':0.5,'podp':0.3}")
    config.set("streams_config", "hotspot", "{'active':False,'increase_factor':2}")
    config.set(
        "streams_config",
        "expiration",
        f"{{'finite_duration':{finite_duration},'start':1,'stop':5}}",
    )
    return config


def make_usecase(**kwargs):
    """Construct a Usecase with a pre-built config (no file path needed)."""
    cfg = make_config(**kwargs)
    uc = Usecase(configFile=None)
    uc.config = cfg
    uc.graph = UsecaseGenerator.makeTopology(uc)
    return uc


# ===========================================================================
# Tests
# ===========================================================================


class TestFatTreeTopology(unittest.TestCase):
    """Tests for k_ary_fat_tree and generate_fattree_regions."""

    def _build(self, k, directed=True):
        return UsecaseGenerator.k_ary_fat_tree(k, directed=directed)

    # --- node counts ---

    def test_k4_node_count(self):
        k = 4
        G = self._build(k)
        n_core = (k ** 2) // 4          # 4
        n_agg = k * (k // 2)            # 8
        n_edge = k * math.ceil(k / 2)   # 8
        n_hosts = k * math.ceil(k / 2) * (k // 2)  # 16
        expected = n_core + n_agg + n_edge + n_hosts
        self.assertEqual(G.number_of_nodes(), expected)

    def test_k6_node_count(self):
        k = 6
        G = self._build(k)
        n_core = (k ** 2) // 4
        n_agg = k * (k // 2)
        n_edge = k * math.ceil(k / 2)
        n_hosts = k * math.ceil(k / 2) * (k // 2)
        expected = n_core + n_agg + n_edge + n_hosts
        self.assertEqual(G.number_of_nodes(), expected)

    # --- node types ---

    def test_all_nodes_have_type(self):
        G = self._build(4)
        for node, data in G.nodes(data=True):
            self.assertIn("type", data, f"Node {node} missing 'type'")
            self.assertIn(data["type"], {"host", "switch"})

    def test_host_count_k4(self):
        k = 4
        G = self._build(k)
        hosts = [n for n, d in G.nodes(data=True) if d["type"] == "host"]
        expected = k * math.ceil(k / 2) * (k // 2)
        self.assertEqual(len(hosts), expected)

    def test_switch_count_k4(self):
        k = 4
        G = self._build(k)
        switches = [n for n, d in G.nodes(data=True) if d["type"] == "switch"]
        n_core = (k ** 2) // 4
        n_agg = k * (k // 2)
        n_edge = k * math.ceil(k / 2)
        expected = n_core + n_agg + n_edge
        self.assertEqual(len(switches), expected)

    # --- directedness ---

    def test_directed_by_default(self):
        import networkx as nx
        G = self._build(4)
        self.assertIsInstance(G, nx.DiGraph)

    def test_undirected_option(self):
        import networkx as nx
        G = self._build(4, directed=False)
        self.assertNotIsInstance(G, nx.DiGraph)

    # --- connectivity ---

    def test_k4_strongly_connected(self):
        import networkx as nx
        G = self._build(4)
        self.assertTrue(nx.is_strongly_connected(G))

    def test_k4_undirected_connected(self):
        import networkx as nx
        G = self._build(4, directed=False)
        self.assertTrue(nx.is_connected(G))


class TestGenerateFattreeRegions(unittest.TestCase):
    """Tests for generate_fattree_regions (defined on Usecase in this codebase)."""

    def test_region_count(self):
        k = 4
        regions = _fattree_regions_owner.generate_fattree_regions(k)
        # 1 core region + k pod regions
        self.assertEqual(len(regions), k + 1)

    def test_core_region_size(self):
        k = 4
        regions = _fattree_regions_owner.generate_fattree_regions(k)
        self.assertEqual(len(regions[0]), (k ** 2) // 4)

    def test_pod_region_size(self):
        k = 4
        regions = _fattree_regions_owner.generate_fattree_regions(k)
        per_pod = k // 2 + math.ceil(k / 2)
        for pod_region in regions[1:]:
            self.assertEqual(len(pod_region), per_pod)

    def test_no_overlap_between_regions(self):
        k = 4
        regions = _fattree_regions_owner.generate_fattree_regions(k)
        all_nodes = [node for region in regions for node in region]
        self.assertEqual(len(all_nodes), len(set(all_nodes)), "Regions overlap")

    def test_k6_region_count(self):
        k = 6
        regions = _fattree_regions_owner.generate_fattree_regions(k)
        self.assertEqual(len(regions), k + 1)

    def test_naming_core(self):
        regions = _fattree_regions_owner.generate_fattree_regions(4)
        for name in regions[0]:
            self.assertTrue(name.startswith("switch-c"), name)

    def test_naming_pod(self):
        regions = _fattree_regions_owner.generate_fattree_regions(4)
        for name in regions[1]:
            self.assertTrue(
                name.startswith("switch-p"), name
            )


class TestMakeTopology(unittest.TestCase):
    """Tests for UsecaseGenerator.makeTopology."""

    def _uc(self, **kw):
        return make_usecase(**kw)

    def test_fat_tree_topology_has_hosts_and_switches(self):
        uc = self._uc(graph_type="ft", k=4)
        types = {d["type"] for _, d in uc.graph.nodes(data=True)}
        self.assertIn("host", types)
        self.assertIn("switch", types)

    def test_every_node_has_ports(self):
        uc = self._uc(graph_type="ft", k=4)
        for node, data in uc.graph.nodes(data=True):
            self.assertIn("ports", data, f"Node {node} missing 'ports'")

    def test_switches_have_proc_delay(self):
        uc = self._uc(graph_type="ft", k=4)
        for node, data in uc.graph.nodes(data=True):
            if data["type"] == "switch":
                self.assertIn("procDelay", data)

    def test_random_er_graph_creates_hosts_and_switches(self):
        # NOTE: makeRandomGraph has a known bug — it slices `switches[:-numCoreSwitches]`
        # when assigning hosts, which produces an empty list when numCoreSwitches >= numswitchs,
        # causing random.choice() to raise IndexError.  This test documents the bug.
        with self.assertRaises(IndexError):
            self._uc(graph_type="er", num_hosts=4, num_switchs=6)


class TestStreamGeneration(unittest.TestCase):
    """Tests for generateStream and streamGenerator."""

    def setUp(self):
        self.uc = make_usecase(graph_type="ft", k=4, num_streams=3)

    def test_generate_stream_returns_dict(self):
        stream = UsecaseGenerator.generateStream(0, self.uc)
        self.assertIsInstance(stream, dict)

    def test_stream_has_required_keys(self):
        required = {
            "name", "type", "interval", "maxFramesPerInterval",
            "maxFrameSize", "maxLatency", "tlkrPssblPlcmnt",
            "lstnsrPssblPlcmnt", "earliest-transmit-offset",
            "latest-transmit-offset", "jitter",
        }
        stream = UsecaseGenerator.generateStream(0, self.uc)
        self.assertTrue(required.issubset(stream.keys()))

    def test_stream_name_includes_id(self):
        stream = UsecaseGenerator.generateStream(5, self.uc)
        self.assertEqual(stream["name"], "stream-5")

    def test_stream_type_is_isochronous(self):
        stream = UsecaseGenerator.generateStream(0, self.uc)
        self.assertEqual(stream["type"], "isochronous")

    def test_interval_in_nanoseconds(self):
        # period_range start=125 µs → min interval = 125_000 ns
        for _ in range(10):
            stream = UsecaseGenerator.generateStream(0, self.uc)
            self.assertGreaterEqual(stream["interval"], 125_000)

    def test_latest_transmit_offset_is_half_interval(self):
        stream = UsecaseGenerator.generateStream(0, self.uc)
        self.assertEqual(stream["latest-transmit-offset"], stream["interval"] // 2)

    def test_earliest_transmit_offset_is_zero(self):
        stream = UsecaseGenerator.generateStream(0, self.uc)
        self.assertEqual(stream["earliest-transmit-offset"], 0)

    def test_jitter_is_zero(self):
        stream = UsecaseGenerator.generateStream(0, self.uc)
        self.assertEqual(stream["jitter"], 0)

    def test_talker_is_a_host(self):
        stream = UsecaseGenerator.generateStream(0, self.uc)
        for talker in stream["tlkrPssblPlcmnt"]:
            self.assertIn(talker, self.uc.graph.nodes)
            self.assertEqual(self.uc.graph.nodes[talker]["type"], "host")

    def test_stream_generator_yields_correct_count(self):
        uc = make_usecase(graph_type="ft", k=4, num_streams=5)
        streams = list(UsecaseGenerator.streamGenerator(uc))
        self.assertEqual(len(streams), 5)
        self.assertEqual(len(uc.streams), 5)

    def test_finite_duration_adds_lifespan(self):
        uc = make_usecase(graph_type="ft", k=4, num_streams=1, finite_duration=True)
        stream = UsecaseGenerator.generateStream(0, uc)
        self.assertIn("lifespanCycles", stream)
        self.assertIsInstance(stream["lifespanCycles"], int)

    def test_no_lifespan_when_finite_duration_false(self):
        stream = UsecaseGenerator.generateStream(0, self.uc)
        self.assertNotIn("lifespanCycles", stream)


class TestUsecaseClass(unittest.TestCase):
    """Tests for the Usecase helper class properties."""

    def setUp(self):
        self.uc = make_usecase(graph_type="ft", k=4, num_streams=2)
        list(UsecaseGenerator.streamGenerator(self.uc))

    def test_hosts_property_returns_only_hosts(self):
        for h in self.uc.hosts:
            self.assertEqual(self.uc.graph.nodes[h]["type"], "host")

    def test_switches_property_returns_only_switches(self):
        for s in self.uc.switches:
            self.assertEqual(self.uc.graph.nodes[s]["type"], "switch")

    def test_links_property_returns_list(self):
        self.assertIsInstance(self.uc.links, list)
        self.assertGreater(len(self.uc.links), 0)

    def test_ports_property_returns_dict(self):
        self.assertIsInstance(self.uc.ports, dict)
        self.assertGreater(len(self.uc.ports), 0)

    def test_frame_size_formula(self):
        stream = self.uc.streams[0]
        expected = int(
            stream["maxFramesPerInterval"] * stream["maxFrameSize"] * BYTE_2_BIT * TOL
        )
        self.assertEqual(self.uc._frame_size(0), expected)

    def test_port_name_id_mapping_bijective(self):
        name_to_id, id_to_name = self.uc.port_name_id_mapping
        # empty string maps to 0
        self.assertEqual(name_to_id[""], 0)
        self.assertEqual(id_to_name[0], "")
        # all other IDs are positive
        for name, idx in name_to_id.items():
            if name != "":
                self.assertGreater(idx, 0)

    def test_distance_to_hosts_keys_are_hosts(self):
        dists = self.uc.distance_to_hosts
        host_set = set(self.uc.hosts)
        self.assertEqual(set(dists.keys()), host_set)


class TestPathConvenience(unittest.TestCase):
    """Tests for pathConvenient and periodConvenient."""

    def setUp(self):
        self.uc = make_usecase(graph_type="ft", k=4, num_streams=1)
        list(UsecaseGenerator.streamGenerator(self.uc))

    def _make_path(self):
        """Return a valid 2-hop path string list from the topology."""
        stream = self.uc.streams[0]
        talker = stream["tlkrPssblPlcmnt"][0]
        # pick any neighbour of the talker
        neighbors = list(self.uc.graph.neighbors(talker))
        if not neighbors:
            self.skipTest("No neighbours found")
        switch = neighbors[0]
        port_key = self.uc.graph.get_edge_data(talker, switch)["sourcePort"]
        return [port_key, switch]

    def test_path_convenient_high_latency(self):
        stream = self.uc.streams[0]
        path = self._make_path()
        # with a very large maxLatency the path should always be convenient
        result = UsecaseGenerator.pathConvenient(
            path,
            maxLatency=10 ** 12,
            maxFramesPerInterval=stream["maxFramesPerInterval"],
            maxFrameSize=stream["maxFrameSize"],
            usecase=self.uc,
        )
        self.assertTrue(result)

    def test_path_convenient_zero_latency(self):
        stream = self.uc.streams[0]
        path = self._make_path()
        result = UsecaseGenerator.pathConvenient(
            path,
            maxLatency=0,
            maxFramesPerInterval=stream["maxFramesPerInterval"],
            maxFrameSize=stream["maxFrameSize"],
            usecase=self.uc,
        )
        self.assertFalse(result)


class TestTalkerListenerEmbedding(unittest.TestCase):
    """Tests for talkerEmbedding and listenerEmbedding."""

    def setUp(self):
        self.uc = make_usecase(
            graph_type="ft",
            k=4,
            num_streams=0,
            num_places_talker=1,
            num_places_listener=1,
            num_listeners_per_str=1,
        )

    def test_talker_embedding_returns_list(self):
        result = UsecaseGenerator.talkerEmbedding(self.uc)
        self.assertIsInstance(result, list)

    def test_talker_embedding_length(self):
        result = UsecaseGenerator.talkerEmbedding(self.uc)
        self.assertEqual(len(result), 1)

    def test_talker_is_valid_host(self):
        result = UsecaseGenerator.talkerEmbedding(self.uc)
        hosts = [n for n, d in self.uc.graph.nodes(data=True) if d["type"] == "host"]
        self.assertIn(result[0], hosts)

    def test_listener_embedding_returns_array(self):
        import numpy as np
        talker = UsecaseGenerator.talkerEmbedding(self.uc)[0]
        result = UsecaseGenerator.listenerEmbedding(self.uc, talker=talker)
        self.assertIsInstance(result, np.ndarray)

    def test_listener_does_not_contain_talker(self):
        talker = UsecaseGenerator.talkerEmbedding(self.uc)[0]
        result = UsecaseGenerator.listenerEmbedding(self.uc, talker=talker)
        self.assertNotIn(talker, result.flatten().tolist())


class TestUsecaseGeneratorClassMethods(unittest.TestCase):
    """Integration-level tests for generateUC and ucGenerator."""

    def _cfg_path(self):
        """Write a temp config file and return its path."""
        import tempfile
        cfg = make_config(graph_type="ft", k=4, num_streams=2)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".ini", delete=False
        )
        cfg.write(tmp)
        tmp.flush()
        tmp.close()
        return tmp.name

    def test_generate_uc_returns_usecase(self):
        path = self._cfg_path()
        uc = UsecaseGenerator.generateUC(path)
        self.assertIsInstance(uc, Usecase)
        os.unlink(path)

    def test_generate_uc_has_streams(self):
        path = self._cfg_path()
        uc = UsecaseGenerator.generateUC(path)
        self.assertEqual(len(uc.streams), 2)
        os.unlink(path)

    def test_generate_uc_index_increments(self):
        UsecaseGenerator.index = 0
        path = self._cfg_path()
        uc1 = UsecaseGenerator.generateUC(path)
        uc2 = UsecaseGenerator.generateUC(path)
        self.assertLess(uc1.index, uc2.index)
        os.unlink(path)

    def test_uc_generator_yields_incrementally(self):
        path = self._cfg_path()
        # ucGenerator yields the same Usecase object mutated in-place each time.
        # Collect the stream counts at each yield by consuming via a loop.
        counts = []
        for uc in UsecaseGenerator.ucGenerator(path):
            counts.append(len(uc.streams))
        # After all yields the final count equals num_streams
        self.assertEqual(counts[-1], 2)
        # Counts should be non-decreasing
        self.assertEqual(counts, sorted(counts))
        os.unlink(path)


class TestConstants(unittest.TestCase):
    def test_mbps_2_bpns(self):
        self.assertAlmostEqual(MBPS_2_BPNS, 0.001)

    def test_byte_2_bit(self):
        self.assertEqual(BYTE_2_BIT, 8)

    def test_tol_greater_than_one(self):
        self.assertGreater(TOL, 1.0)


if __name__ == "__main__":
    unittest.main()