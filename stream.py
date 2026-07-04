from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin, config
from typing import List, Dict, Optional
import networkx as nx
import numpy as np
import random
import itertools

from usecase_config import UsecaseConfig
from topology_handler import TopologyHandler
from utils import requires_attribute, MBPS_TO_BPNS, BYTE_2_BIT

@dataclass
class Stream(DataClassJsonMixin):
    name: str = field(default_factory=lambda: f"stream-{Stream._next_id()}")
    type: str = "isochronous"

    interval: int = 0  # [ns]
    maxFramesPerInterval: int = 0
    maxFrameSize: int = 0  # [bytes]
    maxLatency: int = 0  # [ns]

    tlkrPssblPlcmnt: List[str] = field(default_factory=list)
    lstnsrPssblPlcmnt: List[List[str]] = field(default_factory=list)

    earliest_transmit_offset: int = 0
    latest_transmit_offset: int = 0
    jitter: int = 0

    # optional / computed / runtime fields
    offset: Optional[int] = None
    accLatency: Optional[int] = None
    pathLists: Optional[List[List[str]]] = None
    lifespanCycles: Optional[int] = None

    customGraph: Optional[nx.DiGraph] = field(default=None, repr=False, compare=False, metadata=config(exclude=lambda x: True))
    # tolerance factor for frame size and window length calculations, to account for overhead and variability
    tolerance: float = field(default=1.1, repr=False, metadata=config(exclude=lambda x: True))
    # class-level counter
    _counter = 0
    @classmethod
    def _next_id(cls):
        cls._counter += 1
        return cls._counter
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def to_dict(self): # remove None values for cleaner JSON output
        d = super().to_dict()
        return {k: v for k, v in d.items() if v is not None}
    # -------------------------
    # convenience methods
    # -------------------------
    @property
    def frame_size_bits(self) -> int:
        return int(
            self.maxFramesPerInterval
            * self.maxFrameSize
            * BYTE_2_BIT
            * self.tolerance
        )
    
    def window_length(self, port_speed_mbps: float, minLength: int=53) -> int:
        return max(self.frame_size_bits / port_speed_mbps*MBPS_TO_BPNS,minLength)

    @property
    def is_routed(self) -> bool:
        return self.pathLists is not None

    def build_stream_graph(self, graph: nx.DiGraph):
        """
        Simple pruning rule:

        IF pathLists exists:
            keep only ports in pathLists (and their edges)
            → everything else is implicitly removed

        ELSE:
            remove concurrent ports of:
                - tlkrPssblPlcmnt (talkers)
                - lstnsrPssblPlcmnt (listeners)
        """

        # --------------------------------------------------
        # CASE 1: path-based pruning (STRICT MODE)
        # --------------------------------------------------
        if self.pathLists:

            active_ports = set()

            # keep only ports in pathLists (exclude last element = listener node)
            for path in self.pathLists:
                for hop in path[:-1]:
                    active_ports.add(hop)

            # remove everything not in active_ports
            pruned = nx.DiGraph()
            pruned.add_nodes_from(graph.nodes(data=True))
            for u, v, data in graph.edges(data=True):
                src_port = data.get("sourcePort")
                if src_port in active_ports:
                    pruned.add_edge(u, v, **data)

            isolated_nodes = [n for n, d in pruned.degree() if d < 1]
            if isolated_nodes:
                pruned.remove_nodes_from(isolated_nodes)

            return pruned

        # --------------------------------------------------
        # CASE 2: free routing mode (concurrency pruning)
        # --------------------------------------------------
        edges_to_remove = []

        # talker concurrency removal
        edges_to_remove += TopologyHandler.get_concurrent_ports(
            graph,
            self.tlkrPssblPlcmnt,
            typ="talker"
        )

        # listener concurrency removal
        edges_to_remove += TopologyHandler.get_concurrent_ports(
            graph,
            [l for sub in self.lstnsrPssblPlcmnt for l in sub],
            typ="listener"
        )

        # rebuild graph
        if not edges_to_remove:
            return None  # no pruning needed
        pruned = nx.DiGraph()
        pruned.add_nodes_from(graph.nodes(data=True))

        for u, v, data in graph.edges(data=True):
            if (u, v, data) in edges_to_remove:
                continue
            pruned.add_edge(u, v, **data)

        isolated_nodes = [n for n, d in pruned.degree() if d < 1]
        if isolated_nodes:
            pruned.remove_nodes_from(isolated_nodes)

        return pruned
    
    @property
    @requires_attribute('customGraph', "distance_to_hosts requires stream.customGraph.")
    def distance_to_hosts(self):
        """
        Returns:
            dict -> {("from"| "to", host): {node: distance}}
        """
        G = self.customGraph

        distances = {}

        for host in self.tlkrPssblPlcmnt:
            if host not in G:
                continue

            dist = nx.single_source_dijkstra_path_length(
                G,
                host,
                weight='hop_cost'
            )

            distances[("from", host)] = dist

        for listener in self.lstnsrPssblPlcmnt:
            for host in listener:
                if host not in G:
                    continue

                dist = nx.single_source_dijkstra_path_length(
                    G.reverse(),
                    host,
                    weight='hop_cost'
                )

                distances[("to", host)] = dist

        return distances
    
    @classmethod
    def generate(cls, config: UsecaseConfig) -> 'Stream':
        """
        Generates a Stream object based on the UsecaseConfig.

        Args:
            streamId: Unique stream identifier
            config: UsecaseConfig instance
            talker: Optional list of talker placements
            listeners: Optional list of listener placements
        """
        # Select stream parameters randomly within the configured ranges
        period_ns = int(random.randrange(*config.period_range.values()))*1000  # Convert μs → ns
        maxFramesPerInterval = random.randrange(*config.numberframes_range.values())
        maxFrameSize = random.randrange(*config.datasize_range.values())
        maxLatency = int(random.choice(config.max_latency_range) * 1_000_000)  # ms → ns

        # Determine lifespan if finite duration
        lifespanCycles = None
        if config.expiration.get('finite_duration', False):
            start_sec = config.expiration.get('start', 0) * 60
            stop_sec = config.expiration.get('stop', 0) * 60
            if stop_sec > start_sec:
                duration_ns = random.randint(start_sec, stop_sec) * 1_000_000_000
                lifespanCycles = duration_ns // period_ns

        # Create the Stream object
        stream = Stream(
            interval=period_ns,
            maxFramesPerInterval=maxFramesPerInterval,
            maxFrameSize=maxFrameSize,
            maxLatency=maxLatency,
            earliest_transmit_offset=0,
            latest_transmit_offset=period_ns // 2,
            jitter=0,
            lifespanCycles=lifespanCycles
        )

        return stream



class TrafficHandler:
    """Handler for stream and path computations in a network graph."""
    
    @classmethod
    def embed_to_topology(cls, stream:Stream, config: UsecaseConfig, graph: nx.DiGraph, network_load: list[Stream]):
        stream.tlkrPssblPlcmnt = cls.talkerEmbedding(config, graph, network_load)  # Should return List[str]
        stream.lstnsrPssblPlcmnt = cls.listenerEmbedding(config, graph, network_load, talker=stream.tlkrPssblPlcmnt[0])  # Should return List[List[str]]

        routing = config.routing
        if routing.get('fixPath', False) and config.num_places_talker==1:
            if routing.get('shortestPath', False):
                # Compute shortest paths (one per listener)
                stream = cls.shortestPath(stream, graph, network_load)
            else:
                # Compute random paths (one per listener)
                stream = cls.randomPath(stream, graph, network_load)

        # build custom graph for this stream based on its pathLists or placements, needed for the distance_to_hosts property of stream
        stream.customGraph = stream.build_stream_graph(graph)
        return stream
    

    ########################### Host Embedding ###############################
    @classmethod
    def talkerEmbedding(cls, config: UsecaseConfig, graph: nx.Graph, prev_streams: list[Stream]):
        '''
        configure the probability of choosing a host as talker based on the needed options, and 
        choose the actual placements based on the configured probabilities
        '''    
        
        stride = config.stride
        staggered = config.staggered
        hotspot = config.hotspot
        numpossplace = config.num_places_talker
    
        allhosts = [node for node, data in graph.nodes(data=True) if data['type']=='host']
        allProbs = np.full(len(allhosts) , 1/len(allhosts))
        
        # choose talker
        if hotspot['active']:
            prevchosen = set(int(plc.split('-')[1])-1 for strm in prev_streams for plc in strm['tlkrPssblPlcmnt'])
            allProbs[list(prevchosen)] *= hotspot['increase_factor']
            
        probabilities = allProbs/np.sum(allProbs) # to normalize
        
        '''does striding between chosen_placement and placements to be selected'''
        if stride['active'] and numpossplace>1:
            i = stride['step']
            
            chosen_placement = np.random.choice(allhosts, size=1, p=probabilities, replace=False)[0] # choose 1st placement of the talker
            x = int(chosen_placement.split('-')[1])-1
            for _ in range(numpossplace):
                x = (i+x)%len(allhosts)
                allProbs[x] *= stride['increase_factor']
            
            x = int(chosen_placement.split('-')[1])-1   
            tmp_hosts = allhosts[:x]+allhosts[x+1:]
            tmp_probabilities = allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]] / np.sum(allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]]) # to normalize
            return np.append(np.random.choice(tmp_hosts, size=numpossplace-1, p=tmp_probabilities, replace=False),chosen_placement).tolist()
        
            '''does staggering between chosen_placement and elements to be selected'''
        elif staggered['active'] and numpossplace>1: 
            edgep = staggered['edgep']
            podp = staggered['podp']
            
            chosen_placement = np.random.choice(allhosts, size=1, p=probabilities, replace=False)[0] # choose 1st placement of the talker
            pod = chosen_placement.split('-')[2]

            neighbors_in_pod = [node for node in allhosts if pod in node]        
            for host in neighbors_in_pod:
                x = int(host.split('-')[1])-1
                allProbs[x] = podp

            neighbors_in_edge = [node for node in graph.neighbors(list(graph.neighbors(chosen_placement))[0]) if (node in allhosts) and (node!=chosen_placement)]
            for host in neighbors_in_edge:
                x = int(host.split('-')[1])-1
                allProbs[x] = edgep

            x = int(chosen_placement.split('-')[1])-1
            if hotspot['active']:# in case hotspoting is combined with staggering
                allProbs[list(prevchosen)] *= hotspot['increase_factor']
               
            tmp_hosts = allhosts[:x]+allhosts[x+1:]
            tmp_probabilities = allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]] / np.sum(allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]]) # to normalize
            return np.append(np.random.choice(tmp_hosts, size=numpossplace-1, p=tmp_probabilities, replace=False),chosen_placement).tolist()
        
            '''choose randomly, uniformly'''
        else: 
            return np.random.choice(allhosts, size=numpossplace, p=probabilities, replace=False).tolist()
        
    @classmethod
    def listenerEmbedding(cls, config: UsecaseConfig, graph: nx.Graph, prev_streams: list[Stream], talker: str, listeners: list[str] = None):
        '''
        configure the probability of choosing a host based on the needed options, and 
        choose the actual placements based on the configured probabilities
        
        * the function is used to select listeners, the talker should be:
            - provided, 
            - filtered from hosts, and
            - removed from the returned list of elements if needed (don't forget to increase numpossplace to count for the given talker)
        ''' 
        if listeners is None:
            listeners = []
        
        stride = config.stride
        staggered = config.staggered
        hotspot = config.hotspot
        numpossplace = config.num_places_listener
        numlistenersperstr = config.num_listeners_per_stream
        # All host nodes
        allhosts = [node for node, data in graph.nodes(data=True) if data.get('type') == 'host']
        allProbs = np.full(len(allhosts), 1 / len(allhosts))


        if hotspot['active']:
            prevchosen = [plc for strm in prev_streams for plcs in strm['lstnsrPssblPlcmnt'] for plc in plcs]+listeners
            prevchosen = set(int(plc.split('-')[1])-1 for plc in prevchosen if plc in allhosts)
            allProbs[list(prevchosen)] *= hotspot['increase_factor']
            
            
        # if talker is None:
        #     probabilities = allProbs/np.sum(allProbs) # to normalize
        #     talker = np.random.choice(allhosts, size=1, p=probabilities, replace=False)[0]
            
            '''does striding between talker and elements to be selected'''
        if stride['active']:
            i = stride['step']
            
            x = int(talker.split('-')[1])-1
            for _ in range(numpossplace*numlistenersperstr):
                x = (i+x)%len(allhosts)
                allProbs[x] *= stride['increase_factor']
            x = int(talker.split('-')[1])-1   

            tmp_hosts = allhosts[:x]+allhosts[x+1:]
            tmp_probabilities = allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]] / np.sum(allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]]) # to normalize
            return np.random.choice(tmp_hosts, size=numpossplace*numlistenersperstr, p=tmp_probabilities, replace=False).reshape(numlistenersperstr, numpossplace).tolist()
        
            '''does staggering between talker and elements to be selected'''
        elif staggered['active']: 
            edgep = staggered['edgep']
            podp = staggered['podp']
            pod = talker.split('-')[2]

            neighbors_in_pod = [node for node in allhosts if pod in node]        
            for host in neighbors_in_pod:
                x = int(host.split('-')[1])-1
                allProbs[x] = podp

            neighbors_in_edge = [node for node in graph.neighbors(list(graph.neighbors(talker))[0]) if (node in allhosts) and (node!=talker)]
            for host in neighbors_in_edge:
                x = int(host.split('-')[1])-1
                allProbs[x] = edgep

            x = int(talker.split('-')[1])-1
            if hotspot['active']:# in case hotspoting is combined with staggering
                allProbs[list(prevchosen)] *= hotspot['increase_factor']
               
            tmp_hosts = allhosts[:x]+allhosts[x+1:]
            tmp_probabilities = allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]] / np.sum(allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]]) # to normalize
            return np.random.choice(tmp_hosts, size=numpossplace*numlistenersperstr, p=tmp_probabilities, replace=False).reshape(numlistenersperstr, numpossplace).tolist()
        
            '''choose randomly, uniformly'''
        else: 
            x = int(talker.split('-')[1])-1
            tmp_hosts = allhosts[:x]+allhosts[x+1:]
            tmp_probabilities = allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]] / np.sum(allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]]) # to normalize
            return np.random.choice(tmp_hosts, size=numpossplace*numlistenersperstr, p=tmp_probabilities, replace=False).reshape(numlistenersperstr, numpossplace).tolist()
        
    ######################## Path Computation ##############################
    @classmethod
    def shortestPath(cls, stream: Stream, graph: nx.DiGraph, network_load: list[Stream]):
        paths = []
        assigned_listeners = set()

        flat_listeners = [item for sublist in stream.lstnsrPssblPlcmnt for item in sublist]

        # Compute all shortest paths (talker -> listener)
        for talker, listener in itertools.product(stream.tlkrPssblPlcmnt, flat_listeners):
            if listener in assigned_listeners:
                continue

            path = nx.shortest_path(graph, talker, listener)

            # Convert path to port-level
            for hop in range(len(path) - 1):
                if graph.is_directed():
                    path[hop] = graph.get_edge_data(path[hop], path[hop + 1])['sourcePort']
                else:
                    path[hop] = path[hop] + '.' + graph._adj[path[hop]][path[hop + 1]][path[hop]]

            # Validate path
            if cls.validate_path(path[:-1], graph, stream, network_load):
                paths.append(path)
                assigned_listeners.add(listener)

        stream.pathLists = paths
        stream.lstnsrPssblPlcmnt = [[l[-1] for l in paths]]  # keep listeners in same structure
        return stream

    @classmethod
    def randomPath(cls, stream: Stream, graph: nx.DiGraph, network_load: list[Stream]):
        paths = []
        assigned_listeners = set()

        flat_listeners = [item for sublist in stream.lstnsrPssblPlcmnt for item in sublist]

        # Shuffle all listener-talker pairs for randomness
        listener_talker_pairs = list(itertools.product(stream.tlkrPssblPlcmnt, flat_listeners))
        random.shuffle(listener_talker_pairs)

        for talker, listener in listener_talker_pairs:
            if listener in assigned_listeners:
                continue

            # Get all simple paths and shuffle
            candidate_paths = list(nx.all_simple_paths(graph, talker, listener))
            random.shuffle(candidate_paths)

            # Pick the first valid path
            for path in candidate_paths:
                # Convert path to port-level
                for hop in range(len(path) - 1):
                    if graph.is_directed():
                        path[hop] = graph.get_edge_data(path[hop], path[hop + 1])['sourcePort']
                    else:
                        path[hop] = path[hop] + '.' + graph._adj[path[hop]][path[hop + 1]][path[hop]]

                # Validate path
                if cls.validate_path(path[:-1], graph, stream, network_load):
                    paths.append(path)
                    assigned_listeners.add(listener)
                    break

            if len(assigned_listeners) >= len(flat_listeners):
                break

        stream.pathLists = paths
        stream.lstnsrPssblPlcmnt = [[l[-1] for l in paths]]  # same structure as before
        return stream

    @staticmethod
    def validate_path(path: List[str], graph: nx.DiGraph, stream: Stream, network_load: list[Stream]) -> bool:
        """
        Validates a single path for this stream.

        Checks:
        1. Transmission time per hop ≤ interval (bandwidth check)
        2. Total path latency ≤ maxLatency (short-circuit if exceeded)
        3. (TODO) Bandwidth availability considering other streams on the same ports
        4. (TODO) check that the pairwise gcd of two streams in the same port is greater than the sum of window length, otherwise they are going to overlap anyway
        """
        total_latency = 0
        
        for idx, hop in enumerate(path[:-1]):
            node, port = hop.split('.')
            next_node, _ = path[idx + 1].split('.')
            
            # Transmission delay on this port
            port_speed = graph.nodes[node]['egress_ports'][port]['speed']  # [bits/ns]
            trans_delay = stream.window_length(port_speed)  # [ns]
            
            # Bandwidth / cycle check
            if trans_delay > stream.interval:
                return False
            
            # Edge propagation + node processing
            edge_data = graph.get_edge_data(node, next_node, default={})
            prop_delay = edge_data.get('propDelay', 0)
            proc_delay = graph.nodes[node].get('procDelay', 0)
            
            # Accumulate latency and short-circuit if exceeded
            total_latency += prop_delay + proc_delay + trans_delay
            if total_latency > stream.maxLatency:
                return False
        
        return True
    
    @staticmethod
    def find_all_paths(graph: nx.DiGraph,
                       talkers: Optional[List[str]] = None,
                       listeners: Optional[List[List[str]]] = None
                       ) -> Dict[tuple, any]:
        """
        Returns a dictionary of path generators indexed by (src, dst) pairs.
        """
        all_paths = {}

        # default eligible hosts
        eligible_talkers = [node for node in graph.nodes if node.startswith('host')]
        eligible_listeners = [node for node in graph.nodes if node.startswith('host')]

        # filter by provided talkers/listeners
        if talkers:
            eligible_talkers = [node for node in talkers if node in eligible_talkers]
        if listeners:
            flat_listeners = [item for sublist in listeners for item in sublist]
            eligible_listeners = [node for node in flat_listeners if node in eligible_listeners]

        # generate paths
        for src, dst in itertools.product(eligible_talkers, eligible_listeners):
            all_paths[(src, dst)] = nx.all_simple_paths(graph, src, dst)

        return all_paths
    