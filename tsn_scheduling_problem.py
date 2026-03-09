import math
import time
import json
import ast
from collections import deque

import numpy as np
import networkx as nx

from usecase_generator import UsecaseGenerator, Usecase

import matplotlib.pyplot as plt



# constants
BYTE_2_BIT = 8
MBPS_2_BPNS = 0.001
TOL = 1.1  # tolerance is 0.1 of the frame length

def require_instance(func):
    """Decorator: call self.no_instance_assigned() if no usecase is active."""
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "usecase", None):
            return self.no_instance_assigned()
        return func(self, *args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper


class InstanceContext:
    """
    Context manager to temporarily set the current usecase (instance).
    It will call parent.build_metadata() once per usecase (by id).
    """
    def __init__(self, parent, instance):
        self.parent = parent
        self.instance = instance

    def __enter__(self):
        self.parent.usecase = self.instance

        # create set to track built usecases (store ids to avoid relying on hashability)
        if not hasattr(self.parent, "_metadata_built_for"):
            self.parent._metadata_built_for = set()

        uid = id(self.instance)
        if uid not in self.parent._metadata_built_for:
            self.parent.build_metadata()
            self.parent._metadata_built_for.add(uid)

        return self.parent

    def __exit__(self, exc_type, exc_value, traceback):
        self.parent.usecase = None
        # do NOT remove from _metadata_built_for; metadata is cached


class TSNScheduling():
    """
    Cleaned TSN scheduling problem class (step 1).
    - uses lists/dicts for streams/ports
    - metadata is built lazily and cached per-instance (usecase)
    """

    def __init__(self, problem_definition, ContextHandler=InstanceContext, **kwargs):
        # core helpers
        self.uc_gen = UsecaseGenerator()
        self.config_file = problem_definition

        # instance & context
        self.usecase = None
        self.ContextHandler = ContextHandler
        self.instances = []

        # metadata (initialized per-instance in build_metadata)
        self.config_matrix = None

    # -------------------------
    # metadata
    # -------------------------
    def build_metadata(self):
        """
        Build config_matrix and attach customGraph to each stream dict.
        - Expects self.usecase to be set.
        - Uses make_default_stream_config(stream) which returns (config_column, custom_graph).
        """
        if not self.usecase:
            return self.no_instance_assigned()

        # allocate matrix
        self.config_matrix = np.zeros((self.n_ports, self.n_streams), dtype=int)

        # iterate streams in the raw usecase (list of dicts)
        for j, stream in enumerate(self.usecase.streams):
            config_column, custom_graph = self.make_default_stream_config(stream)

            # sanity check: config_column must match number of ports
            if len(config_column) != self.n_ports:
                raise ValueError(
                    f"config column length ({len(config_column)}) != n_ports ({self.n_ports})"
                )

            self.config_matrix[:, j] = config_column
            # attach custom graph to the stream dict so downstream code can use it
            stream["customGraph"] = custom_graph
            
        # After metadata is built, initialize Problem
        self.initialize_problem()

    def initialize_problem(self):
        n_var = self.n_ports * self.n_streams
        fr, ftp = self.get_task()
        check_routing = not (fr and ftp)
    
    # -------------------------
    # context helper
    # -------------------------
    def use_instance(self, instance):
        """Return a context manager that sets .usecase to instance inside a with-block."""
        return self.ContextHandler(self, instance)

    def no_instance_assigned(self):
        raise ValueError("No current instance is set! Use a 'with problem.use_instance(instance):' block.")

    # -------------------------
    # instantiate / import helpers
    # -------------------------
    def instantiate(self, export=False):
        usecase = self.generate_scheduling_usecase(self.config_file, export=export)
        self.instances.append(usecase)
        return usecase

    def import_scheduling_usecase(self, file_path, remove_routing=False, fix_task_placement=False):
        imp_uc = json.load(open(file_path))
        uc = Usecase.from_json(imp_uc)
        if remove_routing:
            uc.removeRouting()
        if fix_task_placement:
            uc.fixTalkers()
            uc.fixListeners()
        return uc

    def generate_scheduling_usecase(self, config_file, export=False, usecases_folder='./problem_instances/'):
        uc = self.uc_gen.generateUC(config_file)
        if export:
            uc.to_file(jsnPath=usecases_folder)
        return uc

    def generate_incremental_usecases(self, config_file, export=False, usecases_folder='./problem_instances/'):
        ucs = list(self.uc_gen.ucGenerator(config_file))
        if export:
            for uc in ucs: 
                uc.to_file(jsnPath=usecases_folder)
        return ucs
        

    @require_instance
    def get_config(self):
        return self.usecase.get_config()

    def get_task(self):
        numplacesTalker = self.usecase.config.getint('streams_config', 'numplacesTalker')
        numplaceslistener = self.usecase.config.getint('streams_config', 'numplaceslistener')
        routing = ast.literal_eval(str(self.usecase.config.get('streams_config', 'routing')))
        return routing['fixPath'], numplacesTalker==numplaceslistener==1
        
    # -------------------------
    # properties (usecase-dependent)
    # -------------------------
    @property
    @require_instance
    def n_ports(self):
        return len(self.usecase.ports)

    @property
    @require_instance
    def n_streams(self):
        return len(self.usecase.streams)

    @property
    @require_instance
    def port_id_to_name(self):
        """1-based port id -> name mapping. includes 0 -> '' as before."""
        mapping = dict(enumerate(self.usecase.ports.keys(), start=1))
        mapping[0] = ''
        return mapping

    @property
    @require_instance
    def port_name_to_id(self):
        """name -> 1-based port id mapping. includes '' -> 0."""
        mapping = {key: index for index, key in enumerate(self.usecase.ports.keys(), start=1)}
        mapping[''] = 0
        return mapping

    @property
    @require_instance
    def ports(self):
        """Return ports as a dict keyed by 1-based port id (value is the port data), 
           with destPort modified using self.port_name_to_id."""
        return {
            self.port_name_to_id[key]: {**value, 'destPort': self.port_name_to_id.get(value['destPort'], value['destPort'])}
            for key, value in self.usecase.ports.items()
        }

    @property
    @require_instance
    def port_id_to_node(self):
        return {pId: self.ports[pId]['sourceNode'] for pId in self.ports}

    @property
    @require_instance
    def nodes(self):
        # return {node:idx for idx, node in enumerate(self.usecase.graph.nodes)}
        return {node[0]:(idx,node[1].get('procDelay', 0)) 
                for idx, node in enumerate(self.usecase.graph.nodes(data=True))}
        
    @property
    @require_instance
    def n_nodes(self):
        return len(self.usecase.graph.nodes)
        
    @property
    @require_instance
    def node_to_port_id(self):
        egress_map = {}
        ingress_map = {}
        adjacent_egress_map = {}
        
        for src, dest, einfo in self.usecase.graph.edges(data=True):
            # src = einfo.get('sourceNode')
            # dest= einfo.get('destNode')
            srcP = self.port_name_to_id[einfo.get('sourcePort')]
            destP = self.port_name_to_id[einfo.get('destPort')]
        
            egress_map.setdefault(src, []).append(srcP)
            ingress_map.setdefault(dest, []).append(destP)
            adjacent_egress_map.setdefault(dest, []).append(srcP)
        
        return egress_map, ingress_map, adjacent_egress_map

    @property
    @require_instance
    def next_prev_hops(self, hop):
        """ hop= the egress port of the current hop"""
        hop_src = ports[hop]['sourceNode']
        hop_dest = ports[hop]['destNode']
        ports= self.ports

        next_hops = [pid for pid, info in ports.items() if info['sourceNode'] == hop_dest]
        prev_hops = [pid for pid, info in ports.items() if info['destNode'] == hop_src]


    @property
    @require_instance
    def macro_cycle(self):
        """LCM of stream intervals."""
        macro_cycle = 1
        for stream in self.usecase.streams:
            i = int(stream['interval'])
            macro_cycle = macro_cycle * i // math.gcd(macro_cycle, i)
        return macro_cycle

    @property
    @require_instance
    def streams(self):
        """
        Return streams enriched with port ids where needed.
        This property will create a new list each call (shallow copies).
        """
        streams = []
        for s in self.usecase.streams:
            stream = s.copy()
            # replace talker/listener possible placements with 1-based port ids
            stream['tlkrPssblPlcmnt'] = [self.port_name_to_id[node + '.Port1'] for node in s['tlkrPssblPlcmnt']]
            stream['lstnsrPssblPlcmnt'] = [[self.port_name_to_id[node + '.Port1'] for node in nodes] for nodes in s['lstnsrPssblPlcmnt']]
            if 'pathLists' in stream:
                # convert hops to port ids but keep the same shape as original (omit last hop if that was a host owner)
                stream['pathLists'] = [[self.port_name_to_id[hop] for hop in path[:-1]] for path in s['pathLists']]
            streams.append(stream)
        return streams

    @property
    @require_instance
    def links(self):
        links = []
        for l in self.usecase.links:
            link = l.copy()
            link['sourcePort'] = self.port_name_to_id[l['sourcePort']]
            link['destPort'] = self.port_name_to_id[l['destPort']]
            links.append(link)
        return links

    @classmethod
    def _frame_size(cls, stream):
        """
        Compute the frame size in bits *with tolerance*.
        """
        return (
            stream["maxFramesPerInterval"]
            * stream["maxFrameSize"]
            * BYTE_2_BIT
            * TOL
        )
        
    @require_instance
    def hops_costs(self, frame_size):
        """
        Cost of forwarding a frame from one egress port.
        """
        hcost = np.zeros(self.n_ports)
        for pId, egport in self.ports.items():
            destNode = egport['destNode']
            sourceNode = egport['sourceNode']
            
            trans_delay = frame_size / (
                egport["speed"] * MBPS_2_BPNS
            )
            prop_delay = self.usecase.graph[sourceNode][destNode]["propDelay"]
            proc_delay = self.usecase.graph.nodes[destNode].get("procDelay",0)
    
            hcost[pId-1] = trans_delay + prop_delay + proc_delay
    
        return hcost
        
    # -------------------------
    # graph helpers
    # -------------------------
    @require_instance
    def network_borders(self, graph):
        """Return egress ports of switches that are connected to listeners (ports on switch->host edges)."""
        return [edge[2]['sourcePort'] for edge in graph.edges(data=True) if 'host' in edge[2]['destPort']]

    # -------------------------
    # stream configuration helpers
    # -------------------------
    @require_instance
    def configure_stream_ports(self, ports_in, ports_out):
        """
        After instantiating the problem, the streams can be independently configured using this function to force using some ports and excluding some other ports.
                                        ------------------------------     ------------
                                        | customGraph | config_column|      seedGraph
                             -----------------------------------------     ------------
                            always on   |    exist    |       1      |        exist
                             -----------------------------------------     ------------
                            always off/ |  removed    |              |       exist in        
                            concurent of|  from the   |       0      |      some seeds            
                            an always on|   graph     |              |      but not in            
                             -----------------------------------------      some others
                            free to use |    exist    |       0      |        seeds
                            ------------------------------------------     ------------ 
                            
        Build config_column and a pruned custom_graph for the stream based on ports_in / ports_out.
        - ports_in: dict with keys 'talker','listener','switch' listing egress-port names to force active
        - ports_out: dict with keys 'talker','listener','switch' listing ports to force inactive
        Returns (config_column (np.array len n_ports), custom_graph (networkx DiGraph)).
        """
        # ----- 1) always-active ports (talker/switch are egress ports; listener requires the incoming sourcePort)
        always_active_ports = []
        for typ, plist in ports_in.items():
            if typ != 'listener':
                always_active_ports.extend(plist)
            else:
                # for each listener egress-port name we need its incoming sourcePort in the global usecase graph
                for listener_port in plist:
                    node = listener_port.split('.')[0]
                    # take first incoming edge (should exist) and get its sourcePort
                    in_edges = list(self.usecase.graph.in_edges(node, data=True))
                    if in_edges:
                        src_port = in_edges[0][2]['sourcePort']
                        always_active_ports.append(src_port)
                    else:
                        # no incoming edge -> skip (defensive)
                        continue

        # map to 0-based indices for config matrix
        always_active_ids = [self.port_name_to_id[p] - 1 for p in always_active_ports if p in self.port_name_to_id]
        config_column = np.zeros(self.n_ports, dtype=int)
        if always_active_ids:
            config_column[always_active_ids] = 1

        # ----- 2) determine edges to remove from the global graph
        edges_to_remove = []

        # concurrent ports for forced-ins (these are edges to remove because they are "concurrent" with forced-ins)
        for typ, egports in ports_in.items():
            edges_to_remove.extend(self.get_concurrent_ports(egports, typ=typ))

        # ports explicitly forced out: remove edges touching those ports
        ports_out_listener = set(ports_out.get('listener', []))
        ports_out_talker = set(ports_out.get('talker', []))
        ports_out_switch = set(ports_out.get('switch', []))

        for u, v, data in self.usecase.graph.edges(data=True):
            if data.get('destPort') in ports_out_listener or data.get('sourcePort') in ports_out_talker or data.get('sourcePort') in ports_out_switch:
                edges_to_remove.append((u, v, data))

        # ensure we do not remove edges that are needed for always-active ports
        filtered_edges_to_remove = []
        for edge in edges_to_remove:
            src_port = edge[2].get('sourcePort')
            if src_port is None:
                continue
            src_id = self.port_name_to_id.get(src_port, None)
            if src_id is None:
                continue
            if (src_id - 1) in always_active_ids:
                # keep this edge (do not remove)
                continue
            filtered_edges_to_remove.append(edge)

        # ----- 3) build the custom graph (edge-subgraph then copy + prune isolated nodes)
        # build a set of edge keys (u,v) to keep
        keep_edges = [(u, v) for u, v, data in self.usecase.graph.edges(data=True) if (u, v, data) not in filtered_edges_to_remove]
        custom_graph = self.usecase.graph.edge_subgraph(keep_edges).copy()

        # prune isolated nodes
        isolated_nodes = [n for n, d in custom_graph.degree() if d < 1]
        if isolated_nodes:
            custom_graph.remove_nodes_from(isolated_nodes)

        return config_column, custom_graph

    @require_instance
    def get_concurrent_ports(self, egports, typ=None):
        """
        Return a list of edges (u,v,data) that are "concurrent" with the given egress ports.
        - typ in {'switch','talker','listener'}.
        """
        if not egports:
            return []

        nodes_names = [p.split('.')[0] for p in egports]

        if typ == 'switch':
            # other egress ports of the same switch (exclude the provided ones)
            return [
                edge for node in nodes_names
                for edge in self.usecase.graph.out_edges(node, data=True)
                if edge[2]['sourcePort'] not in egports
            ]
        elif typ == 'talker':
            # all egress ports of other hosts
            return [
                edge for node in self.usecase.graph.nodes
                if node not in nodes_names and node.startswith('host')
                for edge in self.usecase.graph.out_edges(node, data=True)
            ]
        elif typ == 'listener':
            # all ingress ports of other hosts
            return [
                edge for node in self.usecase.graph.nodes
                if node not in nodes_names and node.startswith('host')
                for edge in self.usecase.graph.in_edges(node, data=True)
            ]
        else:
            raise ValueError("check arguments of get_concurrent_ports(egports, typ)!!")

    # -------------------------
    # default stream config builders (wrappers)
    # -------------------------
    @require_instance
    def make_default_stream_config(self, stream):
        """
        [generate a subgraph by removing all deactivated nodes and edges (i.e. that are not allowed to be used)]
        Build default (config_column, custom_graph) for a stream dict.
        stream is a dict with keys like 'tlkrPssblPlcmnt', 'lstnsrPssblPlcmnt', optionally 'pathLists'.
        """
        ports_in = {
            'talker': [stream['tlkrPssblPlcmnt'][0] + '.Port1'] if len(stream['tlkrPssblPlcmnt'])==1 else [],      # by default, talkers are free to use unless fixed
            'listener': [listener[0] for listener in stream['lstnsrPssblPlcmnt'] if len(listener) == 1],    # listeners free to use unless fixed
        }

        ports_out = {
            'talker': [host + '.Port1' for host in self.usecase.hosts if host not in stream['tlkrPssblPlcmnt']],
            'listener': [host + '.Port1' for host in self.usecase.hosts if host not in [h for listener in stream['lstnsrPssblPlcmnt'] for h in listener]],
            'switch': []
        }

        if 'pathLists' in stream:
            # all internal switch egress ports in provided paths are forced-in
            ports_in['switch'] = [egressport for path in stream['pathLists'] for egressport in path[1:-1]]
            # switches not in the path are out
            ports_out['switch'] = [port for port in self.usecase.ports if (port not in ports_in['switch']) and ('switch' in port)]

        return self.configure_stream_ports(ports_in, ports_out)

    @require_instance
    def make_stream_config(self, stream, always_in, always_out, overwrite=False, as_idx=True):
        """
        Build (config_column, custom_graph) from explicit always_in / always_out lists.
        - always_in/out are either indices (0-based) or port names depending on as_idx flag.
        """
        if as_idx:
            # convert 0-based indices to 1-based id->name then to names
            always_in = [self.port_id_to_name[pidx + 1] for pidx in always_in]
            always_out = [self.port_id_to_name[pidx + 1] for pidx in always_out]

        ports_in = {
            'talker': [port for port in always_in if 'host' in port],
            'listener': [],
            'switch': [port for port in always_in if 'switch' in port and port not in self.network_borders(self.usecase.graph)]
        }

        ports_out = {
            'talker': [port for port in always_out if 'host' in port],
            'listener': [],
            'switch': [port for port in always_out if 'switch' in port]
        }

        if not overwrite:
            # respect original talker/listener placements from input usecase
            original_out_talkers = [host + '.Port1' for host in self.usecase.hosts if host not in stream['tlkrPssblPlcmnt']]
            ports_out['talker'] += original_out_talkers
            ports_out['listener'] += [host + '.Port1' for host in self.usecase.hosts if host not in [h for listener in stream['lstnsrPssblPlcmnt'] for h in listener]]
            ports_in['talker'] = [p for p in ports_in['talker'] if p not in original_out_talkers]

        return self.configure_stream_ports(ports_in, ports_out)

# ---------------- BFS and hop-cost utilities ---------------- #

    @require_instance
    def hop_cost_eg2eg(self, cur_egp, next_egp, frame_size):
        """
        Cost of forwarding a frame from one egress port to the egress port at next hop.

        Parameters
        ----------
        cur_egp : str
            Current egress port name.
        next_egp : str
            Next egress port name.
        frame_size : float
            Frame size in bits (with tolerance applied).

        Returns
        -------
        float
            Delay in nanoseconds.
        """
        trans_delay = frame_size / (
            self.usecase.ports[cur_egp]["speed"] * MBPS_2_BPNS
        )
        next_node = next_egp.split(".")[0]
        prop_delay = self.usecase.graph[cur_egp.split(".")[0]][next_node]["propDelay"]
        proc_delay = self.usecase.graph.nodes[next_node]["procDelay"]

        return trans_delay + prop_delay + proc_delay

    @require_instance
    def bfs_forward(self, graph, start_port, stream):
        """
        Forward BFS from a talker port to compute minimum delay budgets.

        Parameters
        ----------
        graph : networkx.DiGraph
            The usecase graph.
        start_port : str
            Egress port name of the talker host.
        stream : dict
            Stream definition.

        Returns
        -------
        dict
            Mapping of next-egress-port -> {current-egress-port: min_delay}
        """
        frame_size = TSNScheduling._frame_size(stream)

        queue = deque([(start_port, 0)])  # (current egress port, cumulative delay)
        visited = set()
        ss_min = {}

        while queue:
            cep, delay = queue.popleft()

            for _, next_node, data in graph.out_edges(
                self.usecase.ports[cep]["destNode"], data=True
            ):
                nip = data["destPort"]
                nep = data["sourcePort"]

                if nip == cep:  # avoid going back
                    continue

                if (cep, nep) not in visited:
                    visited.add((cep, nep))
                    new_delay = delay + self.hop_cost_eg2eg(cep, nep, frame_size)

                    if nep not in ss_min:
                        ss_min[nep] = {}
                    ss_min[nep][cep] = new_delay

                    queue.append((nep, new_delay))

        return ss_min

    @require_instance
    def bfs_backward(self, graph, end_port, stream):
        """
        Backward BFS from a listener ingress port to compute maximum
        remaining delay budgets.

        Parameters
        ----------
        graph : networkx.DiGraph
            The usecase graph.
        end_port : str
            Ingress port name of the listener host.
        stream : dict
            Stream definition.

        Returns
        -------
        dict
            Mapping of current-egress-port -> {previous-egress-port: max_delay}
        """
        ml_ns = stream["maxLatency"]
        frame_size = TSNScheduling._frame_size(stream)

        queue = deque([(end_port, ml_ns)])  # (current egress port, remaining delay)
        visited = set()
        ss_max = {}

        while queue:
            cep, delay = queue.popleft()

            for _, _, data in graph.in_edges(cep.split(".")[0], data=True):
                nip = data["destPort"]
                nep = data["sourcePort"]

                if nip == cep:  # avoid going back
                    continue

                if (cep, nep) not in visited:
                    visited.add((cep, nep))
                    new_delay = delay - self.hop_cost_eg2eg(nep, cep, frame_size)

                    if cep not in ss_max:
                        ss_max[cep] = {}
                    ss_max[cep][nep] = new_delay

                    queue.append((nep, new_delay))

        return ss_max

    @staticmethod
    def drawFattree(graph, ax=None, node_size=300, font_size=12, **kwds):
        """
        Draw a fat-tree style layout for the given graph.
    
        Parameters
        ----------
        graph : networkx.Graph
            The graph to draw.
        ax : matplotlib.axes.Axes or None
            Axis to draw into. If None, a new figure is created.
        node_size : int
            Size of nodes.
        font_size : int
            Size of labels.
        kwds : dict
            Extra keyword arguments passed to nx.draw.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(20, 8), dpi=60)
    
        positions = {}
        layers = {'core': 3, 'aggregation': 2, 'edge': 1, 'end': 0}
        xx = {'core': 115, 'aggregation': 75, 'edge': 75, 'end': 0}
    
        for node in graph.nodes():
            if '-c' in node:
                layer = 'core'
            elif '-a' in node:
                layer = 'aggregation'
            elif '-e' in node:
                layer = 'edge'
            else:
                layer = 'end'
            graph.nodes[node]['layer'] = layer
            y = layers[layer]
            x = xx[layer]
            xx[layer] += 20
            positions[node] = (x, y)
    
        pos = nx.multipartite_layout(graph, subset_key='layer', align='horizontal')
        nx.draw(graph, pos=positions, ax=ax, node_size=node_size, font_size=font_size, **kwds)
    
        return ax


