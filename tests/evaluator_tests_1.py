"""
Unit tests for TSNConstraintEvaluator (evaluator.py)

Run with:
    pytest test_evaluator.py -v

Notes
-----
* evaluator.py intentionally omits 'import numpy as np' and 'MBPS_2_BPNS'
  because they are supplied by the surrounding package.  conftest.py injects
  them before the module is loaded.
* All tests use a small but fully-wired 4-node / 8-port linear topology so
  that every code-path in __init__ succeeds without a real problem object.

Topology
--------
  host_A ──p1──► sw_1 ──p3──► sw_2 ──p5──► host_B
         ◄──p2──      ◄──p4──      ◄──p6──
  host_A ──p7──► sw_1  (secondary link, keeps every node's egress non-empty)
         ◄──p8──

  Ports (1-based):
   1  host_A egress → sw_1  ingress (destNode=sw_1,   destPort=2)
   2  sw_1  egress → host_A ingress (destNode=host_A,  destPort=1)
   3  sw_1  egress → sw_2   ingress (destNode=sw_2,   destPort=4)
   4  sw_2  egress → sw_1   ingress (destNode=sw_1,   destPort=3)
   5  sw_2  egress → host_B ingress (destNode=host_B, destPort=6)
   6  host_B egress → sw_2  ingress (destNode=sw_2,   destPort=5)
   7  host_A egress → sw_1  ingress (destNode=sw_1,   destPort=8)
   8  sw_1  egress → host_A ingress (destNode=host_A,  destPort=7)

  Hosts: host_A, host_B   Switches: sw_1, sw_2
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from evaluator import TSNConstraintEvaluator, ScheduleEncodingType

# ---------------------------------------------------------------------------
# Topology constants
# ---------------------------------------------------------------------------

NODES = {
    "host_A": (0, None),
    "sw_1":   (1, None),
    "sw_2":   (2, None),
    "host_B": (3, None),
}

PORTS = {
    1: {"speed": 100, "destNode": "sw_1",   "destPort": 2},
    2: {"speed": 100, "destNode": "host_A", "destPort": 1},
    3: {"speed": 100, "destNode": "sw_2",   "destPort": 4},
    4: {"speed": 100, "destNode": "sw_1",   "destPort": 3},
    5: {"speed": 100, "destNode": "host_B", "destPort": 6},
    6: {"speed": 100, "destNode": "sw_2",   "destPort": 5},
    7: {"speed": 100, "destNode": "sw_1",   "destPort": 8},
    8: {"speed": 100, "destNode": "host_A", "destPort": 7},
}

PORT_ID_TO_NODE = {
    1: "host_A", 2: "sw_1",  3: "sw_1",  4: "sw_2",
    5: "sw_2",   6: "host_B", 7: "host_A", 8: "sw_1",
}

EGRESS_MAP  = {"host_A": [1, 7], "sw_1": [2, 3, 8], "sw_2": [4, 5], "host_B": [6]}
INGRESS_MAP = {"host_A": [2, 8], "sw_1": [1, 4, 7], "sw_2": [3, 6], "host_B": [5]}
ADJACENT_MAP= {"host_A": [2, 8], "sw_1": [1, 4, 7], "sw_2": [3, 6], "host_B": [5]}
HOSTS       = ["host_A", "host_B"]

N_PORTS   = len(PORTS)    # 8
N_NODES   = len(NODES)    # 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_graph(n_edges=2):
    g = MagicMock()
    g.edges.return_value = list(range(n_edges))
    return g


def _make_problem(n_streams=1, max_latency=1000, interval=100, frame_size=64):
    """Build a minimal mock problem for the 8-port linear topology."""
    stream_uc = {
        "tlkrPssblPlcmnt":   ["host_A"],
        "lstnsrPssblPlcmnt": [["host_B"]],
        "maxLatency": max_latency,
        "interval":   interval,
    }
    stream_prob = {
        "maxLatency": max_latency,
        "interval":   interval,
        "customGraph": _fake_graph(2),
    }

    uc = MagicMock()
    uc.macro_cycle = 1000
    uc.hosts = HOSTS
    uc.streams = [stream_uc] * n_streams
    uc._frame_size = lambda s_idx: frame_size

    prob = MagicMock()
    prob.n_nodes   = N_NODES
    prob.n_ports   = N_PORTS
    prob.n_streams = n_streams
    prob.usecase   = uc
    prob.streams   = [stream_prob] * n_streams
    prob.ports     = PORTS
    prob.nodes     = NODES
    prob.port_id_to_node = PORT_ID_TO_NODE
    prob.hops_costs = lambda fs: fs / np.maximum(
        np.array([PORTS[pid]["speed"] for pid in sorted(PORTS)]) * 0.125, 1e-9
    )
    prob.node_to_port_id = (EGRESS_MAP, INGRESS_MAP, ADJACENT_MAP)
    return prob


def _make_ev(n_streams=1, max_latency=1000, interval=100,
             scheduling_type=ScheduleEncodingType.PRIORITY,
             check_routing=True, check_task_placement=True):
    prob = _make_problem(n_streams=n_streams,
                         max_latency=max_latency, interval=interval)
    return TSNConstraintEvaluator(
        prob,
        scheduling_type=scheduling_type,
        check_routing=check_routing,
        check_task_placement=check_task_placement,
    )


def _zeros(n_streams=1):
    return np.zeros((N_PORTS, n_streams), dtype=int)


# ===========================================================================
# ScheduleEncodingType
# ===========================================================================

class TestScheduleEncodingType:
    def test_priority_value(self):
        assert ScheduleEncodingType.PRIORITY.value == 1

    def test_start_time_value(self):
        assert ScheduleEncodingType.START_TIME.value == 2

    def test_distinct_values(self):
        assert ScheduleEncodingType.PRIORITY != ScheduleEncodingType.START_TIME


# ===========================================================================
# UnionFind
# ===========================================================================

class TestUnionFind:
    def _uf(self, elements):
        return TSNConstraintEvaluator.UnionFind(elements)

    def test_find_self(self):
        uf = self._uf([1, 2, 3])
        assert uf.find(1) == 1

    def test_union_connects_two(self):
        uf = self._uf([1, 2, 3])
        uf.union(1, 2)
        assert uf.find(1) == uf.find(2)

    def test_union_transitivity(self):
        uf = self._uf([1, 2, 3])
        uf.union(1, 2)
        uf.union(2, 3)
        assert uf.find(1) == uf.find(3)

    def test_unrelated_distinct(self):
        uf = self._uf([1, 2, 3])
        uf.union(1, 2)
        assert uf.find(1) != uf.find(3)

    def test_path_compression(self):
        uf = self._uf([1, 2, 3, 4])
        uf.union(1, 2); uf.union(2, 3); uf.union(3, 4)
        root = uf.find(1)
        assert uf.find(4) == root   # find triggers path compression


# ===========================================================================
# Constructor — weight & flag handling
# ===========================================================================

class TestConstructor:
    def test_default_priority_weights(self):
        ev = _make_ev()
        assert "priority_conflicts" in ev.weights
        assert "maxlatency_violations" in ev.weights

    def test_default_start_time_weights(self):
        ev = _make_ev(scheduling_type=ScheduleEncodingType.START_TIME)
        for key in ("stream_window_sequence", "macrocycle_overlap", "talker_offset",
                    "maxlatency_violations"):
            assert key in ev.weights

    def test_custom_weight_overrides_default(self):
        prob = _make_problem()
        ev = TSNConstraintEvaluator(prob, weights={"maxlatency_violations": 42})
        assert ev.weights["maxlatency_violations"] == 42

    def test_n_constr_equals_violation_key_count(self):
        ev = _make_ev()
        assert ev.n_constr == len(ev.violations_keys)

    def test_no_routing_excludes_routing_keys(self):
        ev = _make_ev(check_routing=False)
        for key in ("backwarding", "flow_conservation", "looping"):
            assert key not in ev.weights

    def test_no_task_placement_excludes_placement_keys(self):
        ev = _make_ev(check_task_placement=False)
        for key in ("talker_not_fixed", "stray_talker", "stray_listener"):
            assert key not in ev.weights

    def test_n_streams_stored(self):
        ev = _make_ev(n_streams=2)
        assert ev.n_streams == 2

    def test_n_ports_stored(self):
        ev = _make_ev()
        assert ev.n_ports == N_PORTS

    def test_macro_cycle_stored(self):
        ev = _make_ev()
        assert ev.macro_cycle == 1000

    def test_stream_intervals_shape(self):
        ev = _make_ev(n_streams=3)
        assert ev.stream_intervals.shape == (3,)

    def test_windLengths_nonnegative(self):
        ev = _make_ev()
        assert np.all(ev.windLengths >= 0)

    def test_talker_mask_shape(self):
        ev = _make_ev(n_streams=2)
        assert ev.talker_mask.shape == (N_PORTS, 2)

    def test_port_speeds_positive(self):
        ev = _make_ev()
        assert np.all(ev.port_speeds > 0)


# ===========================================================================
# group_connected_ports
# ===========================================================================

class TestGroupConnectedPorts:
    def setup_method(self):
        self.ev = _make_ev()

    def test_empty_chromosome_no_groups(self):
        assert self.ev.group_connected_ports(_zeros(), 0) == []

    def test_single_active_port_one_group(self):
        chrom = _zeros()
        chrom[0, 0] = 1          # port 1
        groups = self.ev.group_connected_ports(chrom, 0)
        assert len(groups) == 1
        assert 1 in groups[0]

    def test_two_adjacent_ports_one_group(self):
        chrom = _zeros()
        chrom[0, 0] = 1          # port 1
        chrom[2, 0] = 1          # port 3 (sw_1 egress → sw_2, next hop of port 1)
        groups = self.ev.group_connected_ports(chrom, 0)
        # port 1 and port 3 are connected via next_hop_list
        assert len(groups) == 1

    def test_disconnected_ports_two_groups(self):
        chrom = _zeros()
        chrom[0, 0] = 1          # port 1 (host_A egress)
        chrom[4, 0] = 1          # port 5 (sw_2 egress, not reachable from port 1 in one hop)
        groups = self.ev.group_connected_ports(chrom, 0)
        assert len(groups) == 2

    def test_returns_list_of_sets(self):
        chrom = _zeros()
        chrom[0, 0] = 1
        groups = self.ev.group_connected_ports(chrom, 0)
        assert all(isinstance(g, set) for g in groups)


# ===========================================================================
# talker_listener_groups
# ===========================================================================

class TestTalkerListenerGroups:
    def setup_method(self):
        self.ev = _make_ev()

    def test_same_group_returns_same_object(self):
        groups = [{1, 3, 5}]
        tg, lg = self.ev.talker_listener_groups(groups, [1], [5])
        assert tg is lg

    def test_talker_absent_returns_none(self):
        groups = [{3, 5}]
        tg, lg = self.ev.talker_listener_groups(groups, [1], [5])
        assert tg is None
        assert lg is not None

    def test_listener_absent_returns_none(self):
        groups = [{1, 3}]
        tg, lg = self.ev.talker_listener_groups(groups, [1], [5])
        assert tg is not None
        assert lg is None

    def test_separate_groups_different_objects(self):
        groups = [{1}, {5}]
        tg, lg = self.ev.talker_listener_groups(groups, [1], [5])
        assert tg is not lg

    def test_both_absent_both_none(self):
        tg, lg = self.ev.talker_listener_groups([], [1], [5])
        assert tg is None and lg is None


# ===========================================================================
# group_penalty
# ===========================================================================

class TestGroupPenalty:
    def setup_method(self):
        self.ev = _make_ev()

    def test_none_group_returns_k(self):
        assert self.ev.group_penalty(None, 7) == 7

    def test_penalty_nonnegative(self):
        group = {1, 3}       # 2 ports → 2 distinct nodes
        for k in range(6):
            assert self.ev.group_penalty(group, k) >= 0

    def test_perfect_path_zero_penalty(self):
        # 3 ports, 3 distinct nodes → raw = k - 3 + (3-3)/2
        # choose k=3 → penalty = 0
        group = {1, 3, 5}    # port_id_to_node: host_A, sw_1, sw_2 → 3 distinct
        assert self.ev.group_penalty(group, 3) == 0

    def test_extra_potential_positive_penalty(self):
        group = {1, 3, 5}    # 3 nodes
        # k=6 → raw = 6-3 + 0 = 3
        assert self.ev.group_penalty(group, 6) == 3


# ===========================================================================
# _per_stream_host_role_violations
# ===========================================================================

class TestHostRoleViolations:
    def setup_method(self):
        self.ev = _make_ev(check_routing=False)

    def test_output_shape_5_rows(self):
        v = self.ev._per_stream_host_role_violations(_zeros())
        assert v.shape == (5, 1)

    def test_no_active_ports_missing_talker_listener(self):
        v = self.ev._per_stream_host_role_violations(_zeros())
        assert v[0, 0] == 1   # talker_not_fixed: |0-1|=1
        assert v[1, 0] == 1   # listener_not_fixed: |0-1|=1

    def test_correct_talker(self):
        # port 1 is host_A egress = talker port for stream 0
        chrom = _zeros()
        chrom[0, 0] = 1
        v = self.ev._per_stream_host_role_violations(chrom)
        assert v[0, 0] == 0   # exactly 1 talker

    def test_correct_listener(self):
        # port 5 (index 4) is sw_2 egress → host_B; adjacent_map[host_B]=[5]
        chrom = _zeros()
        chrom[4, 0] = 1      # port 5 is in adjacent_map["host_B"]
        v = self.ev._per_stream_host_role_violations(chrom)
        assert v[1, 0] == 0   # exactly 1 listener

    def test_all_violations_nonnegative(self):
        chrom = np.random.randint(0, 2, size=(N_PORTS, 1))
        v = self.ev._per_stream_host_role_violations(chrom)
        assert np.all(v >= 0)


# ===========================================================================
# _per_stream_flow_conservation_violations
# ===========================================================================

class TestFlowConservationViolations:
    def setup_method(self):
        self.ev = _make_ev(check_task_placement=False)

    def test_output_has_three_rows(self):
        v = self.ev._per_stream_flow_conservation_violations(_zeros())
        assert v.shape[0] == 3

    def test_all_zeros_no_violations(self):
        v = self.ev._per_stream_flow_conservation_violations(_zeros())
        assert np.all(v == 0)

    def test_forward_path_no_looping(self):
        # ports 1→3→5 (0-indexed: 0,2,4) are the forward egress chain
        chrom = _zeros()
        for p in [0, 2, 4]:
            chrom[p, 0] = 1
        v = self.ev._per_stream_flow_conservation_violations(chrom)
        assert v[2, 0] == 0   # looping row

    def test_violations_nonnegative(self):
        chrom = np.random.randint(0, 2, size=(N_PORTS, 1))
        v = self.ev._per_stream_flow_conservation_violations(chrom)
        assert np.all(v >= 0)


# ===========================================================================
# _per_stream_priority_conflicts
# ===========================================================================

class TestPriorityConflicts:
    def setup_method(self):
        self.ev = _make_ev(n_streams=2, check_routing=False,
                           check_task_placement=False)

    def test_unique_priorities_no_conflict(self):
        # stream 0 gets priority 1, stream 1 gets priority 2 on every active port
        chrom = np.zeros((N_PORTS, 2), dtype=int)
        chrom[0, 0] = 1
        chrom[0, 1] = 2   # same port, different priorities
        v = self.ev._per_stream_priority_conflicts(chrom)
        assert np.all(v == 0)

    def test_duplicate_priority_same_port_conflict(self):
        chrom = np.zeros((N_PORTS, 2), dtype=int)
        chrom[0, 0] = 1
        chrom[0, 1] = 1   # same port, same priority → conflict
        v = self.ev._per_stream_priority_conflicts(chrom)
        assert v.sum() > 0

    def test_inactive_zero_no_conflict(self):
        chrom = _zeros(n_streams=2)
        v = self.ev._per_stream_priority_conflicts(chrom)
        assert np.all(v == 0)

    def test_output_shape(self):
        v = self.ev._per_stream_priority_conflicts(np.zeros((N_PORTS, 2), dtype=int))
        assert v.shape == (1, 2)


# ===========================================================================
# _per_stream_latency_violations_priority
# ===========================================================================

class TestLatencyViolationsPriority:
    def test_inactive_no_violation(self):
        ev = _make_ev()
        v = ev._per_stream_latency_violations_priority(_zeros())
        assert v[0, 0] == 0

    def test_within_budget_no_violation(self):
        ev = _make_ev(max_latency=10_000)
        chrom = _zeros()
        chrom[0, 0] = 1   # one active hop
        v = ev._per_stream_latency_violations_priority(chrom)
        assert v[0, 0] == 0

    def test_exceeds_budget_violation(self):
        ev = _make_ev(max_latency=1e-10)
        chrom = _zeros()
        chrom[0, 0] = 1
        v = ev._per_stream_latency_violations_priority(chrom)
        assert v[0, 0] == 1

    def test_output_shape(self):
        ev = _make_ev()
        v = ev._per_stream_latency_violations_priority(_zeros())
        assert v.shape == (1, 1)

    def test_result_binary(self):
        ev = _make_ev()
        chrom = np.random.randint(0, 3, size=(N_PORTS, 1))
        v = ev._per_stream_latency_violations_priority(chrom)
        assert set(v.flatten()).issubset({0, 1})


# ===========================================================================
# _per_stream_latency_violations_start_time
# ===========================================================================

class TestLatencyViolationsStartTime:
    def _ev(self, max_latency=10_000):
        return _make_ev(max_latency=max_latency,
                        scheduling_type=ScheduleEncodingType.START_TIME,
                        check_task_placement=False, check_routing=False)

    def test_inactive_no_violation(self):
        v = self._ev()._per_stream_latency_violations_start_time(_zeros())
        assert v[0, 0] == 0

    def test_single_port_within_budget(self):
        ev = self._ev(max_latency=10_000)
        chrom = _zeros()
        chrom[0, 0] = 10
        v = ev._per_stream_latency_violations_start_time(chrom)
        assert v[0, 0] == 0

    def test_large_spread_violates_tight_budget(self):
        ev = self._ev(max_latency=1e-10)
        chrom = _zeros()
        chrom[0, 0] = 100
        chrom[2, 0] = 200   # spread = 100 ns >> 1e-10
        v = ev._per_stream_latency_violations_start_time(chrom)
        assert v[0, 0] == 1

    def test_output_shape(self):
        v = self._ev()._per_stream_latency_violations_start_time(_zeros())
        assert v.shape == (1, 1)

    def test_result_binary(self):
        ev = self._ev()
        chrom = np.random.randint(0, 50, size=(N_PORTS, 1))
        v = ev._per_stream_latency_violations_start_time(chrom)
        assert set(v.flatten()).issubset({0, 1})


# ===========================================================================
# _per_stream_talker_offset_violations
# ===========================================================================

class TestTalkerOffsetViolations:
    def setup_method(self):
        self.ev = _make_ev(interval=100,
                           scheduling_type=ScheduleEncodingType.START_TIME,
                           check_task_placement=False, check_routing=False)

    def test_talker_before_interval_no_violation(self):
        chrom = _zeros()
        chrom[0, 0] = 50    # port 1 = host_A egress = talker; 50 < 100
        v = self.ev._per_stream_talker_offset_violations(chrom)
        assert v[0, 0] == 0

    def test_talker_at_interval_violation(self):
        chrom = _zeros()
        chrom[0, 0] = 100   # equals interval → violation (>=)
        v = self.ev._per_stream_talker_offset_violations(chrom)
        assert v[0, 0] == 1

    def test_talker_beyond_interval_violation(self):
        chrom = _zeros()
        chrom[0, 0] = 150
        v = self.ev._per_stream_talker_offset_violations(chrom)
        assert v[0, 0] == 1

    def test_inactive_no_violation(self):
        v = self.ev._per_stream_talker_offset_violations(_zeros())
        assert v[0, 0] == 0

    def test_output_shape(self):
        v = self.ev._per_stream_talker_offset_violations(_zeros())
        assert v.shape == (1, 1)


# ===========================================================================
# _per_stream_macrocycle_overlap_violations
# ===========================================================================

class TestMacrocycleOverlapViolations:
    def _ev2(self):
        return _make_ev(n_streams=2,
                        scheduling_type=ScheduleEncodingType.START_TIME,
                        check_task_placement=False, check_routing=False)

    def test_single_stream_no_self_overlap(self):
        ev = _make_ev(scheduling_type=ScheduleEncodingType.START_TIME,
                      check_task_placement=False, check_routing=False)
        chrom = _zeros()
        chrom[0, 0] = 10
        v = ev._per_stream_macrocycle_overlap_violations(chrom)
        assert np.all(v == 0)

    def test_same_start_time_causes_overlap(self):
        ev = self._ev2()
        chrom = np.zeros((N_PORTS, 2), dtype=int)
        chrom[0, 0] = 5
        chrom[0, 1] = 5    # identical start → always overlapping
        v = ev._per_stream_macrocycle_overlap_violations(chrom)
        assert v.sum() > 0

    def test_output_shape_two_streams(self):
        ev = self._ev2()
        v = ev._per_stream_macrocycle_overlap_violations(np.zeros((N_PORTS, 2), dtype=int))
        assert v.shape == (1, 2)

    def test_violations_nonnegative(self):
        ev = self._ev2()
        chrom = np.random.randint(0, 50, size=(N_PORTS, 2))
        v = ev._per_stream_macrocycle_overlap_violations(chrom)
        assert np.all(v >= 0)


# ===========================================================================
# _per_stream_window_sequence_violations
# ===========================================================================

class TestWindowSequenceViolations:
    def setup_method(self):
        self.ev = _make_ev(n_streams=2,
                           scheduling_type=ScheduleEncodingType.START_TIME,
                           check_task_placement=False, check_routing=False)

    def test_all_zero_no_violation(self):
        v = self.ev._per_stream_window_sequence_violations(np.zeros((N_PORTS, 2), dtype=int))
        # shape: (2, n_streams) — seq and fifo rows
        assert np.all(v >= 0)

    def test_output_rows(self):
        chrom = np.zeros((N_PORTS, 2), dtype=int)
        v = self.ev._per_stream_window_sequence_violations(chrom)
        # Expect 2 rows (seq_v, fifo_v)
        assert v.shape[0] == 2

    def test_violations_nonnegative(self):
        chrom = np.random.randint(0, 100, size=(N_PORTS, 2))
        v = self.ev._per_stream_window_sequence_violations(chrom)
        assert np.all(v >= 0)


# ===========================================================================
# per_stream_all_violations
# ===========================================================================

class TestPerStreamAllViolations:
    def setup_method(self):
        self.ev = _make_ev()

    def test_returns_2d_array(self):
        v = self.ev.per_stream_all_violations(_zeros())
        assert v.ndim == 2

    def test_second_dim_equals_n_streams(self):
        v = self.ev.per_stream_all_violations(_zeros())
        assert v.shape[1] == 1

    def test_result_is_cached(self):
        chrom = _zeros()
        v1 = self.ev.per_stream_all_violations(chrom)
        v2 = self.ev.per_stream_all_violations(chrom)
        assert v1 is v2

    def test_different_chromosomes_not_same_object(self):
        c1, c2 = _zeros(), np.ones((N_PORTS, 1), dtype=int)
        assert self.ev.per_stream_all_violations(c1) is not \
               self.ev.per_stream_all_violations(c2)

    def test_violations_nonnegative(self):
        chrom = np.random.randint(0, 3, size=(N_PORTS, 1))
        assert np.all(self.ev.per_stream_all_violations(chrom) >= 0)

    def test_first_dim_equals_n_constr(self):
        v = self.ev.per_stream_all_violations(_zeros())
        assert v.shape[0] == self.ev.n_constr


# ===========================================================================
# feasible_streams
# ===========================================================================

class TestFeasibleStreams:
    def setup_method(self):
        self.ev = _make_ev()

    def test_returns_bool_array(self):
        f = self.ev.feasible_streams(_zeros())
        assert f.dtype == bool

    def test_shape_equals_n_streams(self):
        assert self.ev.feasible_streams(_zeros()).shape == (1,)

    def test_multi_stream_shape(self):
        ev = _make_ev(n_streams=3)
        assert ev.feasible_streams(np.zeros((N_PORTS, 3), dtype=int)).shape == (3,)

    def test_feasible_is_complement_of_infeasible(self):
        chrom = _zeros()
        f = self.ev.feasible_streams(chrom)
        v = self.ev.per_stream_all_violations(chrom)
        infeasible = np.any(v > 0, axis=0)
        np.testing.assert_array_equal(f, ~infeasible)


# ===========================================================================
# compute_metrics
# ===========================================================================

class TestComputeMetrics:
    def setup_method(self):
        self.ev = _make_ev()

    def test_returns_six_element_array(self):
        assert self.ev.compute_metrics(_zeros()).shape == (6,)

    def test_result_is_cached(self):
        chrom = _zeros()
        assert self.ev.compute_metrics(chrom) is self.ev.compute_metrics(chrom)

    def test_congestion_in_unit_interval(self):
        m = self.ev.compute_metrics(np.ones((N_PORTS, 1), dtype=int))
        assert 0.0 <= m[0] <= 1.0

    def test_avg_util_in_unit_interval(self):
        m = self.ev.compute_metrics(np.ones((N_PORTS, 1), dtype=int))
        assert 0.0 <= m[1] <= 1.0

    def test_load_imbalance_in_unit_interval(self):
        m = self.ev.compute_metrics(np.ones((N_PORTS, 1), dtype=int))
        assert 0.0 <= m[3] <= 1.0

    def test_zero_chromosome_load_imbalance_is_one(self):
        m = self.ev.compute_metrics(_zeros())
        assert m[3] == 1.0

    def test_path_len_score_in_unit_interval(self):
        m = self.ev.compute_metrics(_zeros())
        assert 0.0 <= m[4] <= 1.0

    def test_avg_path_len_nonnegative(self):
        m = self.ev.compute_metrics(_zeros())
        assert m[5] >= 0

    def test_custom_alpha_accepted(self):
        m = self.ev.compute_metrics(_zeros(), alpha=0.3)
        assert m.shape == (6,)


# ===========================================================================
# evaluate
# ===========================================================================

class TestEvaluate:
    def setup_method(self):
        self.ev = _make_ev()

    def test_returns_two_element_tuple(self):
        result = self.ev.evaluate(_zeros())
        assert isinstance(result, tuple) and len(result) == 2

    def test_violations_1d(self):
        violations, _ = self.ev.evaluate(_zeros())
        assert violations.ndim == 1

    def test_metrics_shape(self):
        _, metrics = self.ev.evaluate(_zeros())
        assert metrics.shape == (6,)

    def test_violations_nonnegative(self):
        violations, _ = self.ev.evaluate(np.random.randint(0, 3, (N_PORTS, 1)))
        assert np.all(violations >= 0)

    def test_violations_length_equals_n_constr(self):
        violations, _ = self.ev.evaluate(_zeros())
        assert len(violations) == self.ev.n_constr


# ===========================================================================
# suggest_corrections — smoke tests
# ===========================================================================

class TestSuggestCorrections:
    def test_priority_returns_three_arrays(self):
        ev = _make_ev()
        result = ev.suggest_corrections(np.zeros((3, N_PORTS, 1), dtype=int))
        assert len(result) == 3

    def test_start_time_returns_three_arrays(self):
        ev = _make_ev(scheduling_type=ScheduleEncodingType.START_TIME)
        result = ev.suggest_corrections(np.zeros((2, N_PORTS, 1), dtype=int))
        assert len(result) == 3

    def test_remove_bias_batch_dimension(self):
        ev = _make_ev()
        remove_bias, _, _ = ev.suggest_corrections(np.zeros((4, N_PORTS, 1), dtype=int))
        assert remove_bias.shape[0] == 4

    def test_add_bias_nonnegative(self):
        ev = _make_ev()
        _, add_bias, _ = ev.suggest_corrections(np.zeros((2, N_PORTS, 1), dtype=int))
        assert np.all(add_bias >= 0)

    def test_gene_value_bias_shape_priority(self):
        ev = _make_ev(n_streams=2)
        Y = np.zeros((3, N_PORTS, 2), dtype=int)
        _, _, gene_bias = ev.suggest_corrections(Y)
        # gene_value_bias is multiplied by expanded_act → same spatial shape
        assert gene_bias.shape[0] == 3
