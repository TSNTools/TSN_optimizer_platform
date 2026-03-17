from enum import Enum
import numpy as np

class ScheduleEncodingType(Enum):
    PRIORITY = 1
    START_TIME = 2
    #QUEUING_DELAY = 3

MBPS_2_BPNS = 0.001

from scipy.sparse import csr_matrix
class TSNConstraintEvaluator:
    def __init__(self, problem, weights=None, check_routing=True, check_task_placement=True, 
                 scheduling_type: ScheduleEncodingType = ScheduleEncodingType.PRIORITY):
        # Default penalty weights
        default_weights = {
            # task placement
            "talker_not_fixed": 1,
            "listener_not_fixed": 1,
            "stray_talker": 1,
            "stray_listener": 1,
            "talker_listener_overlap": 1,
            # routing
            "backwarding": 1,
            "flow_conservation": 1,
            "looping": 1,
            "disconnected_path": 1,
            # scheduling
            "priority_conflicts": 1,
            "maxlatency_violations": 1,
            "stream_window_sequence":1,
            "interleaving":1,
            "macrocycle_overlap":1,
            "talker_offset":1
        }

        # Merge with user-supplied weights
        merged_weights = {**default_weights, **(weights or {})}

        # Determine which violation functions are active
        self.scheduling_type = scheduling_type
        self.active_checks = []
        relevant_weight_keys = set()

        if check_task_placement:
            self.active_checks.append(self._per_stream_host_role_violations)
            relevant_weight_keys.update([
                "talker_not_fixed",
                "listener_not_fixed",
                "stray_talker",
                "stray_listener",
                "talker_listener_overlap"
            ])

        if check_routing:
            self.active_checks.extend([
                self._per_stream_flow_conservation_violations,
                self._per_stream_flow_continuity_violations
            ])
            relevant_weight_keys.update([
                "backwarding",
                "flow_conservation",
                "looping",
                "disconnected_path"
            ])

        # Always check scheduling
        if self.scheduling_type.value == ScheduleEncodingType.PRIORITY.value:
            self.active_checks.extend([
                self._per_stream_priority_conflicts,
                self._per_stream_latency_violations_priority
            ])
            relevant_weight_keys.update([
                "maxlatency_violations",
                "priority_conflicts"
            ])
        if self.scheduling_type.value == ScheduleEncodingType.START_TIME.value:
            self.active_checks.extend([
                self._per_stream_window_sequence_violations,
                self._per_stream_macrocycle_overlap_violations,
                self._per_stream_talker_offset_violations,
                self._per_stream_latency_violations_start_time
            ])
            relevant_weight_keys.update([
                "maxlatency_violations",
                "stream_window_sequence",
                "interleaving",
                "macrocycle_overlap",
                "talker_offset"
            ])

        # Keep only relevant weights
        self.weights = {k: merged_weights[k] for k in relevant_weight_keys}
        self.n_constr = len(relevant_weight_keys)
        self.violations_keys = relevant_weight_keys
        
        #############################################################
        self._violation_evaluation_cache = {}
        self._metrics_evaluation_cache = {}
        ##############################################################
        
        n_nodes = problem.n_nodes
        n_ports = problem.n_ports
        n_streams = problem.n_streams

        self.n_streams =n_streams
        self.n_ports = n_ports

        self.macro_cycle = getattr(problem.usecase, "macro_cycle", 0)  # fallback if missing
        ports = problem.ports                        # dict keyed by port id (1-based)
        streams = problem.streams
        self.hosts = getattr(problem.usecase, "hosts", [])
        self.nodes = problem.nodes

        self.port_id_to_node = problem.port_id_to_node
        self.egress_map, self.ingress_map, self.adjacent_map = problem.node_to_port_id 

        ############### precompute ports/streams data ################
        self.port_speeds = np.zeros(n_ports, dtype=float)
        for p_idx in range(n_ports):
            pid = p_idx + 1
            pdata = ports[pid]
            self.port_speeds[p_idx] = pdata["speed"]
            
        self.stream_maxLatencies = [] 
        stream_intervals = []
        self.windLengths = np.zeros((n_ports, n_streams), dtype=float)
        self.all_hops_costs = np.zeros((n_ports, n_streams))
        stream_potential = []
        eps = 1e-9
        for s_idx, stream in enumerate(streams):
            stream_potential.append(len(stream["customGraph"].edges())) # total potential
            self.stream_maxLatencies.append(stream["maxLatency"])
            stream_intervals.append(stream["interval"])
            
            # precompute hops costs for latency evaluation
            frame_size = problem.usecase._frame_size(s_idx)
            self.all_hops_costs[:, s_idx] = problem.hops_costs(frame_size)
            # precompute windLengths
            for p_idx in range(n_ports):
                self.windLengths[p_idx, s_idx] = (frame_size / max(self.port_speeds[p_idx] * MBPS_2_BPNS, eps))

        self.stream_potential = stream_potential
        self.stream_intervals = np.asarray(stream_intervals, dtype=int)
        streams_pairwise_gcl = np.gcd(self.stream_intervals[:, None], self.stream_intervals[None, :])
        streams_pairwise_gcl[streams_pairwise_gcl == 0] = 1   # safety
        self.streams_pairwise_gcl = streams_pairwise_gcl # (S,S)
        self.streams_pairwise_gcl_inv = 1.0 / self.streams_pairwise_gcl[None, None, :, :]

        stream_pairwise_windowSum = (self.windLengths[:, :, None] + self.windLengths[:, None, :]) # (P,S,S)
        self.stream_pairwise_windowSum = stream_pairwise_windowSum
        # used to filter self-comparison for stream overlap
        self.pair_mask = ~np.eye(n_streams, dtype=bool)

        ################# nodes masks #####################
        node_index = {name: nid for name, (nid, _) in self.nodes.items()}
        
        node_egress_mask  = np.zeros((n_nodes, n_ports), dtype=int)
        node_ingress_mask  = np.zeros((n_nodes, n_ports), dtype=int)
        egress_to_node_mask  = np.zeros((n_nodes, n_ports), dtype=int)
        
        for name, n_idx in node_index.items():
            node_egress_mask[n_idx, np.array(self.egress_map[name])-1]=1
            node_ingress_mask[n_idx, np.array(self.ingress_map[name])-1]=1
            egress_to_node_mask[n_idx, np.array(self.adjacent_map[name])-1]=1
        
        host_mask = np.zeros((n_nodes), dtype=int)
        for host in self.hosts:
            host_mask[node_index[host]] = 1

        self.host_mask = host_mask
        self.switch_mask = 1-host_mask      

        self.egress_mask = node_egress_mask
        self.ingress_mask = node_ingress_mask
        self.to_node_mask = egress_to_node_mask

        ################# ports masks #####################
        adjacency_mask = np.zeros((n_ports, n_ports), dtype=int)
        next_hop_matrix = np.zeros((n_ports, n_ports), dtype=int)
        
        next_hop_list = [[] for _ in range(n_ports)]
        
        
        for pid in range(1, n_ports + 1):
            p_idx = pid - 1
            pdata = ports[pid]
            dst_node = pdata["destNode"]
            dst_port = pdata["destPort"]

            adjacency_mask[pid - 1, dst_port-1] = 1  
            
            next_ports = self.egress_map.get(dst_node, []) if dst_node not in self.hosts else []
            next_hop_list[p_idx] = list(dict.fromkeys(next_ports))

            next_eg_indices = [p-1 for p in next_ports]
            next_hop_matrix[p_idx, next_eg_indices] = 1
        
        self.adjacency_mask = adjacency_mask #
        self.next_hop_list = next_hop_list
        next_hop_matrix = csr_matrix(next_hop_matrix)

        self.next_hop_matrix = next_hop_matrix
        ################ CSR connectivity ################
        self._upstream_ports = np.repeat(np.arange(n_ports), np.diff(next_hop_matrix.indptr))
        self._downstream_ports = next_hop_matrix.indices
        
        ################# streams masks #####################
        talker_mask        = np.zeros((n_ports, n_streams), dtype=int)
        listener_mask      = np.zeros((n_ports, n_streams), dtype=int)
        listener_adj_mask  = np.zeros((n_ports, n_streams), dtype=int)
        
        self.stream_talker_ports = [] ##########
        self.stream_listener_adj_ports = [] ########## 
        
        for s_idx, stream in enumerate(problem.usecase.streams):
            
            # 1) Talker ports (1-based)
            t_nodes = stream.get("tlkrPssblPlcmnt", [])
            tps = []
            for t_node in t_nodes:
                t_ports = self.egress_map[t_node]
                tps +=t_ports
                talker_mask[np.array(t_ports) - 1, s_idx] = 1
            self.stream_talker_ports.append(np.array(tps, dtype=int) if tps else np.array([], dtype=int))

            # 2) Listener ports (1-based, stored as list of lists)
            lst_lists = stream.get("lstnsrPssblPlcmnt", [])
            lst_nodes = [l for sub in lst_lists for l in sub]
            lps = []
            for lst_node in lst_nodes:
                lst_ports = self.ingress_map[lst_node]
                listener_mask[np.array(lst_ports) - 1, s_idx] = 1
                
                lst_adj_ports = self.adjacent_map[lst_node]
                lps += lst_adj_ports
                listener_adj_mask[np.array(lst_adj_ports) - 1, s_idx] = 1
            self.stream_listener_adj_ports.append(np.array(lps, dtype=int) if lps else np.array([], dtype=int))

        # Save masks
        self.talker_mask = talker_mask # ports that can be talker
        self.listener_mask = listener_mask # ports that can be listener
        self.listener_adj_mask = listener_adj_mask # ports that reach to listener

        hosts_port_mask = node_egress_mask[host_mask==1].sum(0)
        hosts_adj_port_mask = egress_to_node_mask[host_mask==1].sum(0)
        self.stray_talker_mask = (1 - talker_mask) * hosts_port_mask[:,None]
        self.stray_listener_mask = (1 - listener_adj_mask) * hosts_adj_port_mask[:,None]
        
    def _per_stream_host_role_violations(self, chromosome):
        act = chromosome > 0          # bool (no astype)
    
        # Boolean masking (no integer multiply)
        active_talkers   = act & self.talker_mask
        active_listeners = act & self.listener_adj_mask
    
        # Per-stream counts
        talker_count   = active_talkers.sum(axis=0)
        listener_count = active_listeners.sum(axis=0)
    
        # Stray per-stream (reduce directly)
        stray_talker   = (act & self.stray_talker_mask).sum(axis=0)
        stray_listener = (act & self.stray_listener_mask).sum(axis=0)
    
        # Matrix multiply (dominant cost)
        downstream = self.adjacency_mask @ active_listeners
    
        # Overlap per-stream
        overlap = (active_talkers & downstream).sum(axis=0)

        return np.vstack([
            np.abs(talker_count - 1),      # talker_not_fixed
            np.abs(listener_count - 1),    # listener_not_fixed
            overlap,                       # talker_listener_overlap
            stray_talker,                  # stray_talker
            stray_listener                 # stray_listener
        ])

    # -------------------------
    # Connectivity helpers (Union-Find)
    # -------------------------
    class UnionFind:
        def __init__(self, elements):
            self.parent = {e: e for e in elements}
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx != ry:
                self.parent[ry] = rx

    def group_connected_ports(self, chromosome, s_idx):
        """Group connected active ports using the adjacency precomputed in __init__."""
        active_ports = [p + 1 for p in np.nonzero(chromosome[:, s_idx])[0]]
        if not active_ports:
            return []
        pid_set = set(active_ports)
        uf = self.UnionFind(active_ports)
        for pid in active_ports:
            for nb in self.next_hop_list[pid - 1]:
                if nb in pid_set:
                    uf.union(pid, nb)
        groups = {}
        for pid in active_ports:
            root = uf.find(pid)
            groups.setdefault(root, set()).add(int(pid))
        return list(groups.values())

    def talker_listener_groups(self, groups, talker_ids, listener_ids):
        """True if any group contains at least one talker port and one listener port."""
        talker_group = None
        listener_group = None
        
        for group in groups:
            if group & set(talker_ids):
                talker_group = group
            if group & set(listener_ids):
                listener_group = group
        return talker_group, listener_group
            
    def group_penalty(self, group, k):
        if group is None:
            return k  # completely disconnected group
    
        active_nodes = {self.port_id_to_node[pid] for pid in group}
        
        n_nodes = len(active_nodes)
        n_edges = len(group)
    
        raw = k - n_nodes + (n_edges - n_nodes) / 2
        return int(max(0, raw))

    def _per_stream_flow_continuity_violations(self, chromosome):
        n = self.n_streams
        v = np.zeros(n, dtype=int)  # 1D is faster to write into
    
        # Local bindings (avoids repeated attribute lookup)
        group_connected_ports = self.group_connected_ports
        talker_listener_groups = self.talker_listener_groups
        group_penalty = self.group_penalty
    
        stream_talker_ports = self.stream_talker_ports
        stream_listener_adj_ports = self.stream_listener_adj_ports
        stream_potential = self.stream_potential
    
        for s_idx in range(n):
            groups = group_connected_ports(chromosome, s_idx)
    
            talker_group, listener_group = talker_listener_groups(
                groups,
                stream_talker_ports[s_idx],
                stream_listener_adj_ports[s_idx],
            )
    
            if (talker_group is not None and listener_group is not None and talker_group is listener_group):
                continue
    
            k = stream_potential[s_idx]
            v[s_idx] = (group_penalty(talker_group, k) + group_penalty(listener_group, k))
    
        return v[np.newaxis, :]

    def _per_stream_flow_conservation_violations(self, chromosome):
        # Local bindings (faster attribute access)
        egress_mask = self.egress_mask
        ingress_mask = self.ingress_mask
        adjacency_mask = self.adjacency_mask
        switch_mask = self.switch_mask.astype(bool)
    
        # Binary activation
        act = (chromosome > 0).astype(int)  # (P, S)
    
        # Compute adjacency once
        adj_act = adjacency_mask @ act     # (P, S)
    
        # Broadcast masks
        active_egp = egress_mask[:, :, None] * act[None, :, :]      # (N,P,S)
        active_ing = ingress_mask[:, :, None] * adj_act[None, :, :] # (N,P,S)
    
        # Degrees (avoid building active_ieg_per_node)
        in_deg  = active_ing.sum(axis=1)   # (N,S)
        out_deg = active_egp.sum(axis=1)   # (N,S)
    
        # ---- Violations ----
    
        # backwarding (intersection count without explicit temp array)
        backwarding = (active_egp & active_ing).sum(axis=(0, 1))
    
        # Only switch nodes
        in_sw  = in_deg[switch_mask]
        out_sw = out_deg[switch_mask]
    
        flow_conservation = np.abs(in_sw - out_sw).sum(axis=0)
        looping = np.maximum(0, in_sw - 1).sum(axis=0)
    
        # Stack into (3, S)
        return np.vstack((
            backwarding,
            flow_conservation,
            looping
        ))

###############################################################################
    def _per_stream_priority_conflicts(self, chromosome):
        # Flatten ports × streams
        flat = chromosome.reshape(-1, self.n_streams)  # shape (n_ports, n_streams)
        
        # Indices of active entries
        rows, cols = np.nonzero(flat > 0)
        
        # Corresponding values (priority labels)
        labels = flat[rows, cols]
        
        # Linearize (row, label) to detect duplicates
        lin = rows * (self.n_streams + 1) + labels  # +1 to avoid zero conflicts
        
        # Count occurrences
        counts = np.bincount(lin, minlength=flat.shape[0] * (self.n_streams + 1))
        
        # Duplicates = counts > 1
        mask_active = counts[lin] > 1
        
        # Allocate output
        v = np.zeros(self.n_streams, dtype=int)
        
        # Sum conflicts per stream
        np.add.at(v, cols[mask_active], 1)
        
        return v[np.newaxis, :]

    def _per_stream_latency_violations_priority(self, chromosome):
        n_ports, n_streams = self.n_ports, self.n_streams
    
        active_mask = chromosome > 0
        big_num = 1_000_000
    
        # Replace inactive entries with big number so they sort to the end
        priorities_for_sort = np.where(active_mask, chromosome, big_num)
    
        # Sort priorities along each port
        sorted_indices = np.argsort(priorities_for_sort, axis=1)
        row_idx = np.arange(n_ports)[:, None]
    
        # Reorder hops and active_mask according to sorted priorities
        sorted_hops = np.take_along_axis(self.all_hops_costs, sorted_indices, axis=1)
        sorted_mask = np.take_along_axis(active_mask, sorted_indices, axis=1)
    
        # Zero out inactive entries
        sorted_hops *= sorted_mask
    
        # Cumulative sum along each port
        cumsum_delays = np.cumsum(sorted_hops, axis=1)
    
        # Map cumulative delays back to original stream order
        cumulative_delays = np.zeros_like(chromosome, dtype=float)
        cumulative_delays[row_idx, sorted_indices] = cumsum_delays
    
        # Sum per stream
        total_latency = (cumulative_delays * active_mask).sum(axis=0)
    
        # Compare against max latency
        violations = (total_latency > self.stream_maxLatencies).astype(int)
    
        return violations[np.newaxis, :]

###############################################################################
    def _per_stream_latency_violations_start_time(self, chromosome):
        """
        chromosome[p, s] = transmission start time (0 = inactive)
        """
    
        active = chromosome > 0
    
        # Replace inactive entries so they don't affect min/max
        start_min = np.where(active, chromosome, np.inf)
        start_max = np.where(active, chromosome, -np.inf)
    
        # Per-stream earliest and latest transmission
        t_min = start_min.min(axis=0)
        t_max = start_max.max(axis=0)
    
        # Transmission window (use worst-case hop window)
        # windLengths already = transmission duration per port/stream
        max_window = np.where(active, self.windLengths, 0).max(axis=0)
    
        # Accumulated latency
        latency = (t_max - t_min) + max_window
    
        # Streams with no activity → no violation
        latency = np.where(np.isfinite(t_min), latency, 0)
    
        violations = (latency > self.stream_maxLatencies).astype(int)
    
        return violations[np.newaxis, :]

    def _per_stream_talker_offset_violations(self, chromosome):
        """
        chromosome[p, s] = start time (0 = inactive)
        """
    
        active = chromosome > 0
    
        # Active talker entries
        talker_active = self.talker_mask & active
    
        # Extract talker start time
        talker_times = np.where(talker_active, chromosome, 0)
        talker_offset = talker_times.max(axis=0)
    
        intervals = self.stream_intervals
    
        violations = (
            (talker_offset >= intervals)
        ).astype(int)
    
        return violations[np.newaxis, :]

    def _per_stream_macrocycle_overlap_violations(self, chromosome):
        """
        Detect periodic window overlaps over the macro-cycle.
        Implements the periodic-interval collision test: 
            (phase1 - phase2) mod gcd(interval1, interval2) < window1 + window2; where phase = start % interval
        chromosome[p,s] = start time (0 = inactive)
        """
    
        active = chromosome > 0  # (P,S)

        # --- normalize start times into stream phase
        phase = chromosome % self.stream_intervals[None, :]
    
        # --- pairwise start-time differences (P,S,S) without abs
        diff = phase[:, :, None] - phase[:, None, :]
    
        # --- periodic phase difference using precomputed gcd
        delta = np.mod(diff, self.streams_pairwise_gcl)
    
        # --- overlap condition
        overlap = delta < self.stream_pairwise_windowSum
    
        # remove self comparisons
        overlap &= self.pair_mask
    
        # require both streams active on same port
        both_active = active[:, :, None] & active[:, None, :]
        overlap &= both_active
    
        # --- count violations per stream
        # (sum over ports and partner streams)
        v = overlap.sum(axis=(0, 2))
    
        return v[np.newaxis, :]
        
    def _per_stream_window_sequence_violations(self, chromosome):
        """
        chromosome[p, s] = start time (0 = inactive)
        
        Enforces two scheduling constraints:
        -------------------------------------------------------
        1) INTRA-STREAM SEQUENCING (causality)
    
            Constraint per connection (u → d): chromosome[d, s] >= chromosome[u, s] + hop_cost[u, s]
    
        Prevents a frame leaving a downstream port before it
        arrives from upstream. Violations are marked at the downstream port (d).
        -------------------------------------------------------
        2) INTER-STREAM FIFO (slot stealing / interleaving)
    
        For two streams s1,s2 on the same port:
            (t11 - t21)*(t12 - t22) >= 0
        meaning arrival order must equal departure order.
        Prevents one stream stealing another stream’s slot.
        """
        n_ports, n_streams = self.n_ports, self.n_streams
        upstream_ports = self._upstream_ports
        downstream_ports = self._downstream_ports
    
        if upstream_ports.size == 0:
            return np.zeros((n_streams,), dtype=int)
        
        upstream_start = chromosome[upstream_ports]          # (connections, S)
        downstream_start = chromosome[downstream_ports]      # (connections, S)
    
        upstream_active = upstream_start > 0
        downstream_active = downstream_start > 0
        # Only check connections where stream exists at both ends
        valid = upstream_active & downstream_active
        # Required arrival time at downstream
        enqueuing_start = upstream_start + self.all_hops_costs[upstream_ports] # per connection

        # -------------------------------------------------
        # PART 1 — INTRA-STREAM SEQUENCING
        # -------------------------------------------------
        connection_violations = valid & (downstream_start < enqueuing_start)
        # Accumulate violations at downstream ports
        violations = np.zeros((n_ports, n_streams), dtype=bool)
        np.maximum.at(
            violations,
            (downstream_ports[:, None], np.arange(n_streams)),
            connection_violations
        )
        # Sum violations per stream
        seq_v  = violations.sum(axis=0)
        # -------------------------------------------------
        # PART 2 — INTER-STREAM FIFO (slot stealing)
        # -------------------------------------------------
        active = chromosome > 0
        
        eg_queue_arrival = np.zeros((n_ports, n_streams), dtype=float)           
        np.maximum.at(
            eg_queue_arrival,
            (downstream_ports[:, None], np.arange(n_streams)),
            enqueuing_start
        )

        arr_diff = eg_queue_arrival[:, :, None] - eg_queue_arrival[:, None, :]
        dep_diff = chromosome[:, :, None] - chromosome[:, None, :]
        
        interleave = (arr_diff * dep_diff) < 0
        interleave &= self.pair_mask
        both_active = active[:, :, None] & active[:, None, :]
        interleave &= both_active
        
        fifo_v = interleave.sum(axis=(0,2))

        return np.vstack((seq_v[np.newaxis, :], fifo_v[np.newaxis, :]))
###################################################################################
    
    def per_stream_all_violations(self, chromosome):
        key = chromosome.tobytes()
        # get violations
        cached = self._violation_evaluation_cache.get(key)
        if cached is not None:
            per_stream_violations = cached
        else:
            # List to hold violation arrays
            all_v = []

            # Collect results as (1, S) arrays and append to list
            for check_function in self.active_checks:
                all_v.append(check_function(chromosome))               

            per_stream_violations = np.vstack(all_v)

            self._violation_evaluation_cache[key] = per_stream_violations
        # Concatenate all violation arrays along the first axis (rows)
        return per_stream_violations

    def feasible_streams(self, chromosome):
        violations_array = self.per_stream_all_violations(chromosome)
        infeasible_streams = np.any(violations_array > 0, axis=0)
        return ~infeasible_streams

###################################################################################

    def compute_metrics(self, chromosome, alpha=0.9, eps=1e-9):
        """Vectorized performance metrics (congestion, imbalance, path length)."""
        key = chromosome.tobytes()
        cached = self._metrics_evaluation_cache.get(key)
        if cached is not None:
            metrics = cached
        else:
            load_per_port = np.sum((chromosome > 0).astype(float) * self.windLengths, axis=1)
            util = load_per_port / np.maximum(self.macro_cycle * self.port_speeds, eps)
            
            avg_util = float(np.clip(np.nanmean(util), 0, 1))
            std_util = float(np.clip(np.nanstd(util), 0, 1))
            congestion_score = float(np.clip(alpha * avg_util + (1 - alpha) * std_util, 0, 1))
            
            mean_load = float(np.mean(load_per_port))
            if mean_load <= eps:
                load_imbalance = 1.0
            else:
                imbalance = float(np.std(load_per_port) / mean_load)
                load_imbalance = float(min(imbalance, 1.0) if np.isfinite(imbalance) else 1.0)
            
            active_per_stream = np.count_nonzero(chromosome > 0, axis=0)
            avg_path_len = float(np.mean(active_per_stream))
            path_len_score = float(np.clip(avg_path_len / (self.n_ports + eps), 0, 1))

            metrics = np.array([congestion_score, avg_util, std_util, load_imbalance, path_len_score, avg_path_len])
            self._metrics_evaluation_cache[key] = metrics
        # Return the metrics as an array
        return metrics

###################################################################################

    def evaluate(self, chromosome):
        key = chromosome.tobytes()
        # get violations
        per_stream_violations = self.per_stream_all_violations(chromosome)

        # get metrics
        metrics = self.compute_metrics(chromosome)
        
        return per_stream_violations.sum(axis=1), metrics

###################################################################################
    
    def suggest_corrections(self, Y):

        adj_port_mask= self.to_node_mask[None, :, :, None]
        egress_mask = self.egress_mask[None, :, :, None]
        switch_mask = self.switch_mask[None, :, None]
        talker_mask   = self.talker_mask[None,:,:]
        listener_adj_mask = self.listener_adj_mask[None,:,:]
        adjacency_mask = self.adjacency_mask
        stream_intervals = self.stream_intervals[None, None, :]

        act = Y > 0
        adj_act = (adjacency_mask @ act)
        expanded_act = act[:, None, :, :]  # (I,N, P,S)
        expanded_adj_act = adj_act[:, None, :, :]

        active_egp = egress_mask & expanded_act
        active_adj_egp = adj_port_mask & expanded_act
        
        out_deg = active_egp.sum(axis=2) 
        in_deg = active_adj_egp.sum(axis=2)

        # switches state
        active_switch_egp = out_deg * switch_mask
        active_switch_adj_egp = in_deg * switch_mask

        active_hosts_egp = out_deg * (1-switch_mask)
        active_hosts_adj_egp = in_deg * (1-switch_mask)

        # hosts (talkers and listenres) state
        active_talker_egp = active_egp.sum(axis=1) * talker_mask
        active_listener_adj_ing = active_adj_egp.sum(axis=1) * listener_adj_mask
        
        n_talkers = active_talker_egp.sum(axis=1) 
        n_listeners = active_listener_adj_ing.sum(axis=1)

        forwarding_hosts = ((active_hosts_egp == 1) & (active_hosts_adj_egp == 1))[:,:,None,:]

        # switches grouping
        overloaded_switches = (active_switch_adj_egp > 1)[:,:,None,:]
        sourced_switches = ((active_switch_adj_egp == 0) & (active_switch_egp > 0))[:,:,None,:] 
        sinked_switches = ((active_switch_adj_egp > 0) & (active_switch_egp==0))[:,:,None,:]

        # hosts grouping
        too_many_talkers = n_talkers > 1
        missing_talker = n_talkers < 1
        too_many_listeners = n_listeners > 1
        missing_listener = n_listeners < 1

        # removal pressure
        remove_bias = (
            2 * too_many_talkers[:, None, :] * active_talker_egp # remove extra talkers
            + 2 * too_many_listeners[:, None, :] * active_listener_adj_ing # remove extra listeners
            + 5 * ((active_adj_egp + active_egp) & forwarding_hosts).sum(axis=1) # Encourage moving talker and listener elsewhere
            + (active_adj_egp & overloaded_switches).sum(axis=1) # remove looping
            + (active_egp & sourced_switches).sum(axis=1) # remove blankstart switches
            + (active_adj_egp & sinked_switches).sum(axis=1) # remove deadend switches
            + (expanded_adj_act & expanded_act).sum(axis=1) # remove backwarding
        )

        # addition pressure
        add_bias = (
             missing_talker[:, None, :] * talker_mask # activate talker
            +missing_listener[:, None, :] * listener_adj_mask # activate listener
            +(egress_mask & sinked_switches).sum(axis=1) # extend deadend switches
            + (adj_port_mask & sourced_switches).sum(axis=1) # extend blankstart switches
        )
        
        add_bias*=(1-self.stray_talker_mask)*(1-self.stray_listener_mask) # avoid useless activations

        # priority pressure
        if self.scheduling_type.value == ScheduleEncodingType.PRIORITY.value:
            counts = (Y[:, :, :, None] == Y[:, :, None, :]).sum(axis=2)
            conflict_mask = (counts > 1)[:, None, :, :] #& act
            gene_value_bias = conflict_mask.astype(int) #change conflicting priority
        elif self.scheduling_type.value == ScheduleEncodingType.START_TIME.value:
            n_chromosomes, n_ports, n_streams = Y.shape
        
            upstream_ports = self._upstream_ports
            downstream_ports = self._downstream_ports
            all_hops_costs = self.all_hops_costs

            stream_pairwise_windowSum = self.stream_pairwise_windowSum[None]
            streams_pairwise_gcl = self.streams_pairwise_gcl[None, None, :, :]
            pair_mask = self.pair_mask[None, None]
        
            # Gather per-connection start times
            upstream_start = Y[:, upstream_ports, :]       # (I, connections, S)
            downstream_start = Y[:, downstream_ports, :]   # (I, connections, S)
    
            upstream_active = upstream_start > 0
            downstream_active = downstream_start > 0
            valid = upstream_active & downstream_active
    
            # Required arrival time at downstream (broadcast correctly)
            enqueuing_start = upstream_start + all_hops_costs[upstream_ports][None, :, :]
    
            connection_bias = np.where(valid & (downstream_start < enqueuing_start),
                               enqueuing_start - downstream_start,
                               0)
    
            sequencing_bias_downstream = np.zeros_like(Y, dtype=int) # (I, P, S)
            sequencing_bias_upstream = np.zeros_like(Y, dtype=int) # (I, P, S)
            eg_queue_arrival = np.zeros((n_chromosomes, n_ports, n_streams), dtype=float) # !!!!
            for i in range(n_chromosomes):
                np.maximum.at(
                    sequencing_bias_downstream[i],
                    (downstream_ports[:, None], np.arange(n_streams)),
                    connection_bias[i]
                )
                np.maximum.at(
                    sequencing_bias_upstream[i],
                    (upstream_ports[:, None], np.arange(n_streams)),
                    connection_bias[i]
                )
            # ---  interleaving bias ---
                np.maximum.at(                                             # !!!!
                    eg_queue_arrival[i],
                    (downstream_ports[:, None], np.arange(n_streams)),
                    enqueuing_start[i]
                )

            arr_diff = eg_queue_arrival[:, :, :, None] - eg_queue_arrival[:, :, None, :]
            dep_diff = Y[:, :, :, None] - Y[:, :, None, :]
            
            interleave = (arr_diff * dep_diff) < 0
            interleave &= pair_mask
            act = Y > 0
            both_active = act[:, :, :, None] & act[:, :, None, :]  # require both streams active
            interleave &= both_active
            
            sign_upstream = np.sign(arr_diff)
            sign_downstream = np.sign(dep_diff)

            pressure_down = sign_upstream * np.abs(dep_diff) * interleave
            pressure_up = sign_downstream * np.abs(arr_diff) * interleave
            
            downstream_bias = np.sum(pressure_down, axis=3)
            
            upstream_bias_tmp = np.sum(pressure_up, axis=3)
            upstream_bias = np.zeros_like(Y)
            for i in range(n_chromosomes):
                np.add.at(
                    upstream_bias[i],
                    (upstream_ports[:, None], np.arange(n_streams)),
                    upstream_bias_tmp[i, upstream_ports, :]
                )
            interleaving_pressure = upstream_bias + downstream_bias
            # ---  talker offset bias ---
            # Compute per-stream latest talker start
            talker_times = np.where(active_talker_egp > 0, Y, 0)               # (I, P, S)
            talker_offset = talker_times.max(axis=1, keepdims=True)            # (I, 1, S)
            # violation = talker too late
            talker_bias = np.maximum(0, talker_offset - stream_intervals)  # (I, 1, S)
            
            # ---  maxlatency bias ---
            t_max = np.where(act, Y, -np.iinfo(Y.dtype).max).max(axis=1, keepdims=True)
            latency_limit = talker_offset + np.asarray(self.stream_maxLatencies)[None,None,:]
            end_time = Y + self.windLengths[None,:,:]
            is_latest_active = (Y == t_max) & act
            latency_bias = (latency_limit - end_time) * (end_time > latency_limit)  * is_latest_active

            # --- macrocycle_bias overlap bias ---
            phase = Y % stream_intervals
            diff = phase[:, :, :, None] - phase[:, :, None, :]
            delta = diff - np.floor(diff * self.streams_pairwise_gcl_inv) * streams_pairwise_gcl
            
            # diff = Y[:, :, :, None] - Y[:, :, None, :] # (I,P,S,S) # pairwise start differences
            # delta = diff - streams_pairwise_gcl * np.floor(diff / streams_pairwise_gcl)
            overlap = delta < stream_pairwise_windowSum # overlap condition
            overlap &= pair_mask                  # remove self comparisons
            overlap &= both_active
            # signed cyclic direction
            half_g = streams_pairwise_gcl / 2
            signed_delta = np.where(delta <= half_g, delta, delta - streams_pairwise_gcl)
            # overlap magnitude
            overlap_depth = stream_pairwise_windowSum - delta
            overlap_depth = np.maximum(overlap_depth, 0)
            overlap_depth *= overlap
            
            gene_value_bias =  np.stack([
                sequencing_bias_downstream # (sequencing) move forward early downstream windows at switches
                , - sequencing_bias_upstream # (sequencing) move backward late upstream windows
                , - talker_bias  * talker_mask # (talker offset) talker out of 1st period
                , + (np.sign(signed_delta) * overlap_depth).sum(axis=3) # (macrocycle overlap) overlapping slots repel each other
                , + np.maximum(upstream_bias, 0) # (interleaving) move forward owner at upstream
                , + np.minimum(upstream_bias, 0) # (interleaving) move backward thief at uptream 
                , + np.maximum(downstream_bias, 0) # (interleaving) move forward thief at downstream
                , + np.minimum(downstream_bias, 0) # (interleaving) move backward owner at downstream
                , + latency_bias # (maxlatency) move latest window bakward
            ], axis=1).astype(int)

        else:
            gene_value_bias = expanded_act
                    
        #####################################
        remove_bias *= act
        add_bias    *= (~act) & (adj_act == 0) # avoid backward connection and talker/listener overlap
        gene_value_bias  = gene_value_bias * expanded_act

        return remove_bias, add_bias, gene_value_bias    