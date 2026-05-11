from dataclasses import dataclass, field
import json
from dataclasses_json import DataClassJsonMixin, config
from typing import Dict, List, Optional, Union
import math

from scipy.fftpack import dst

from usecase_config import UsecaseConfig
from topology_handler import TopologyHandler
from stream import Stream, TrafficHandler

from schedule import Schedule

import networkx as nx


@dataclass
class Usecase(DataClassJsonMixin):
    """Refactored Usecase dataclass with JSON serialization."""
    index: int = field(default_factory=lambda: Usecase._next_id())

    # config is NOT serialized to JSON, it's only used at runtime for internal purposes
    specsFile: str = field(default="uc_config.ini", repr=False, compare=False, metadata=config(exclude=lambda x: True))
    specification: Optional[UsecaseConfig] = field(default=None, repr=False, compare=False, metadata=config(exclude=lambda x: True))
    
    graph: nx.DiGraph = field(
        init=False, compare=False,
        metadata=config(
            encoder=TopologyHandler.graph_encoder,
            decoder=TopologyHandler.graph_decoder
        )
    )
    streams: List[Stream] = field(
        default_factory=list, 
        metadata=config(
            encoder=lambda s: [stream.to_dict() for stream in s], 
            decoder=lambda d: [Stream.from_dict(s) for s in d]
        )
    )
    
    # class-level counter
    _counter = 0
    @classmethod
    def _next_id(cls):
        cls._counter += 1
        return cls._counter

    def __post_init__(self):
        # Generate UsecaseConfig object from file
        if self.specification is None:
            self.specification = UsecaseConfig.from_file(self.specsFile)

        # Generate the graph from the config
        if not hasattr(self, "graph") or self.graph is None:
            self.graph = TopologyHandler.makeTopology(self.specification)

        # Auto-generate streams if not provided
        if not self.streams:
            for _ in range(self.specification.num_of_streams_per_instance):
                self.add_stream()

    # ---------------- Properties ----------------
    @property
    def n_streams(self):
        return len(self.streams)
        
    @property
    def n_ports(self):
        return len(self.ports)
    
    @property
    def hosts(self) -> List[str]:
        return [n for n in self.graph.nodes if "host" in n]

    @property
    def switches(self) -> List[str]:
        return [n for n in self.graph.nodes if "switch" in n]

    @property
    def links(self):
        return [d for _, _, d in self.graph.edges(data=True)]
    
    @property
    def macro_cycle(self) -> int:
        mc = 1
        for s in self.streams:
            i = int(s.interval)
            mc = mc * i // math.gcd(mc, i)
        return mc
    
    @property
    def ports(self):
        """
        returns egress and ingress ports
        """
        def map_data(d):
            out = d.copy()  # keep all existing fields
            out['name'] = f"{d['sourceNode']}.{d['sourcePort']}"
            del out['sourcePort']
            out['destPort'] = f"{d['destNode']}.{d['destPort']}"
            return out
            
        ports = {}
        for node, data in self.graph.nodes(data=True):
            for pname, pdata in data['ingress_ports'].items():
                ports[node+'.'+pname]={
                    'name':node+'.'+pname,
                    'sourceNode':node,
                    'destPort':node+'.'+pname, 
                    'destNode':node,
                    'speed':1e-9, 
                    'duplex':pdata['duplex']
                }
            for pname, pdata in data['egress_ports'].items():
                ports[node+'.'+pname]= map_data(pdata)
        return ports
    
    # scheduling hook (kept minimal)
    @property
    def port_name_id_mapping(self):
        """name -> 1-based port id mapping. includes '' -> 0."""
        port_name_to_id = {key: index for index, key in enumerate(self.ports.keys(), start=1)}
        port_name_to_id[''] = 0
        port_id_to_name = dict(enumerate(self.ports.keys(), start=1))
        port_id_to_name[0] = ''
        return port_name_to_id, port_id_to_name
    
    @property
    def streams_by_source_port(self) -> Dict[str, List[Stream]]:
        """Organize streams by their source port name."""
        mapping: Dict[str, List[Stream]] = {}
        for stream in self.streams:
            if len(stream.tlkrPssblPlcmnt)>0:
                mapping.setdefault(stream.tlkrPssblPlcmnt[0], []).append(stream)
        return mapping
    
    @property
    def task(self):
        return self.specification.routing['fixPath'], self.specification.num_places_talker==self.specification.num_places_listener==1

    @property
    def schedule(self):
        port_name_to_id, _ = self.port_name_id_mapping
        return Schedule.from_usecase(self, port_name_to_id)    
    
    @property
    def utilization(self):
        utilization = {}

        hosts = set(self.hosts)

        for idx, port in self.ports.items():

            if port['sourceNode'] in hosts:
                continue

            if 'schedule' in port:
                utilization[idx] = (
                    sum(slot['duration'] for slot in port['schedule']) / port['scheduleCycle']
                )
            else:
                utilization[idx] = 0

        values = utilization.values()
        return sum(values) / len(utilization) if utilization else 0, utilization
        
    @property
    def distance_to_hosts(self):
        """
        Returns: dict -> {host: {node: distance}}
        """
        all_distances = {}
    
        for host in self.hosts:
            if host not in self.graph:
                continue
            dist = nx.single_source_dijkstra_path_length(self.graph, host)
            all_distances[host] = dist
        return all_distances
    # ---------------- Methods ----------------

    #def window_length(self, stream_idx, port_name, eps=1e-9):
    #    stream = self.streams[stream_idx]
    #    return stream.window_length(
    #        self.ports[port_name]['speed'],
    #        eps
    #    )

    #def distance_to_hosts_per_stream(self, s_idx):
    #    return self.streams[s_idx].distance_to_hosts()

    def addSchedule(self, schedule):
        # remove existing schedule first
        self.removeSchedule()

        _, port_id_to_name = self.port_name_id_mapping

        schedule.update_graph(self.graph, port_id_to_name)
        schedule.update_streams(
            self.streams,
            self.ports,
            port_id_to_name
        )

    def removeSchedule(self):

        for _, node_data in self.graph.nodes(data=True):

            for _, port_data in node_data['egress_ports'].items():

                port_data.pop('schedule', None)
                port_data.pop('basetime', None)
                port_data.pop('scheduleCycle', None)

        for stream in self.streams:

            stream.offset = None
            stream.accLatency = None

    def remove_streams(self, stream_ids):
        for stream_id in sorted(stream_ids, reverse=True):
            del self.streams[stream_id]

    def add_stream(self):
        stream = Stream.generate(self.specification)
        self.streams.append(
            TrafficHandler.embed_to_topology(stream, self.specification, self.graph, network_load=self.streams)
        )

    def remove_link(self, src, dst):
        TopologyHandler.remove_link(self.graph, self.streams, src, dst)  

    def add_link(self, src, dst):
        TopologyHandler.add_link(self.graph, self.specification, src, dst)

    def to_file(self, ucFileName=None, jsnPath="./"):

        if ucFileName is None:
            ucFileName = f"{self.specification.graph_config.get('typ','uc')}.json"

        data = self.to_json()

        with open(jsnPath + ucFileName, "w") as f:
            f.write(data)

        return ucFileName
    
    @classmethod
    def from_json(cls, data: dict):
        graph = TopologyHandler.graph_decoder(data["graph"])

        streams = [Stream.from_dict(s) for s in data["streams"]]

        uc = cls(
            specsFile="uc_config.ini",
            index=data.get("index"),
            streams=streams
        )

        uc.graph = graph  # override post_init graph generation

        # restore stream graph binding (important)
        for s in uc.streams:
            if s.customGraph is None:
                s.customGraph = uc.graph

        return uc