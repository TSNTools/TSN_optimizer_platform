from typing import Optional
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin

from topology import makeRandomGraph, k_ary_fat_tree, makeLineGraph
from usecase_config import UsecaseConfig
import networkx as nx

@dataclass
class Port(DataClassJsonMixin):
    """Represents a network port on a node."""
    name: str
    node_name: str
    speed: int = 0  # in Mbps or Gbps
    queue_count: int = 1

    # Optional info about connected nodes
    source_node: Optional[str] = None
    dest_node: Optional[str] = None
    source_port: Optional[str] = None
    dest_port: Optional[str] = None

    def is_egress(self) -> bool:
        """Return True if this port is an egress port."""
        return self.source_node is not None and self.dest_node is not None

    def is_ingress(self) -> bool:
        """Return True if this port is an ingress port."""
        return self.dest_node is not None

class TopologyHandler:
    """Handler to manage a networkx graph."""

    #def __init__(self, usecase: Usecase):
    #    self.usecase = usecase

    @classmethod
    def makeTopology(cls, config: UsecaseConfig):
        # read default topology config
        portSpeed  = config.port_speed  # mbps
        propDelay = config.prop_delay
        procDelay = config.proc_delay  # us
        linkBw = config.link_bw  # mbps
        queueCountPerPort = config.queue_count_per_port

        graphConfig = config.graph_config
        graphType = graphConfig['typ'].replace("'", '')
        
        ## generate graph
        if graphType == 'ft':
            graph = k_ary_fat_tree(graphConfig['k'])
        elif graphType == 'line':
            graph = makeLineGraph(config)
        else:
            graph = makeRandomGraph(config)

        # add port information to graph
        for node, successors in graph._adj.items():
            port_index = 1 
            graph.nodes[node]['type'] = 'host'
            for succ in successors:
                cls.set_link_data(graph, node, succ, 
                                               propDelay, linkBw, portSpeed, queueCountPerPort)
                if port_index > 1:
                    graph.nodes[node]['type'] = 'switch'
                    graph.nodes[node]['procDelay'] = procDelay * 1000  # in [ns]

                port_index += 1

        return graph

    @classmethod
    def set_link_data(cls, graph, node, dst, propDelay, linkBw, portSpeed, queueCountPerPort):
        def create_new_port(node):
            all_ports = list(graph.nodes[node].get('egress_ports', {}).keys()) + \
                        list(graph.nodes[node].get('ingress_ports', {}).keys())
            indices = [int(p.replace('Port','')) for p in all_ports] if all_ports else [0]
            return max(indices) + 1
        # initialize ports dicts
        for n in [node, dst]:
            graph.nodes[n].setdefault('egress_ports', {})
            graph.nodes[n].setdefault('ingress_ports', {})

        # initialize ports names
        src_port_name = None
        dst_port_name = None
        # reuse existing ports
        try:
            src_port_name = graph._adj[dst][node]['destPort'].split('.')[1]
            dst_port_name = graph._adj[dst][node]['sourcePort'].split('.')[1]
        except:
            pass
        # create new ports
        if src_port_name is None:
            src_port_name = 'Port'+str(create_new_port(node))
        if dst_port_name is None:
            dst_port_name = 'Port'+str(create_new_port(dst))


        # --- adjacency dicts ---
        graph._adj.setdefault(node, {})[dst] = graph._adj.get(node, {}).get(dst, {})
        #graph._adj.setdefault(dst, {})[node] = graph._adj.get(dst, {}).get(node, {})
        # --- register ports at nodes ---
        graph.nodes[node]['egress_ports'][src_port_name]= {
            'sourceNode': node,
            'sourcePort':src_port_name,
            'destPort':dst_port_name,
            'destNode':dst,
            'speed':portSpeed, 
            'queueCount':queueCountPerPort,
            'duplex':'tx'
        }
        if src_port_name in graph.nodes[node]['ingress_ports']:
            graph.nodes[node]['ingress_ports'][src_port_name]['duplex']='full'
            graph.nodes[node]['egress_ports'][src_port_name]['duplex']='full'
            
        graph.nodes[dst]['ingress_ports'][dst_port_name]= {
            'sourceNode':node, 
            'sourcePort':src_port_name,
            'duplex':'rx'
        }
        if dst_port_name in graph.nodes[dst]['egress_ports']:
            graph.nodes[dst]['ingress_ports'][dst_port_name]['duplex']='full'
            graph.nodes[dst]['egress_ports'][dst_port_name]['duplex']='full'
            
        # --- update adjacency ---
        graph._adj[node][dst].update({
            'sourceNode': node,
            'sourcePort': node + '.' + src_port_name,
            'destNode': dst,
            'destPort': dst + '.' + dst_port_name,
            'propDelay': propDelay,
            'linkBw': linkBw
        })
    # ---------------- Node/Port Operations ----------------

    # ---------------- Link Operations ----------------
    @classmethod
    def remove_link(cls, graph, streams, src, dst):
        if not graph.has_edge(src, dst):
            return

        edge_data = graph[src][dst]
        port_to_remove = edge_data['sourcePort']

        graph.remove_edge(src, dst)

        # remove ingress port at destination
        dst_ingress = graph.nodes(data=True)[dst]['ingress_ports']
        dst_port = edge_data['destPort'].split('.')[1]
        dst_ingress.pop(dst_port, None)

        # remove egress port at source
        src_egress = graph.nodes[src]['egress_ports']
        src_port = edge_data['sourcePort'].split('.')[1]
        src_egress.pop(src_port, None)

        # clean stream paths
        for stream in streams:
            if stream.pathLists:
                stream.pathLists = [
                    [p for p in path if p != port_to_remove]
                    for path in stream.pathLists
                ]

    @classmethod
    def add_link(cls, graph, config, src, dst):

        graph.add_edge(src, dst)

        cls.set_link_data(
            graph,
            src,
            dst,
            config.prop_delay,
            config.link_bw,
            config.port_speed,
            config.queue_count_per_port
        )

    @classmethod
    def get_concurrent_ports(cls, graph: nx.DiGraph, egports: list[str], typ: str = None):
        """
        Return edges that are "concurrent" with given ports.

        Rules:
        - switch   → other egress ports of same switch (exclude input ports)
        - talker   → all egress ports of other hosts
        - listener → all ingress ports of other hosts
        """

        if not egports:
            return []

        nodes_names = [p.split('.')[0] for p in egports]

        # --------------------------------------------------
        # SWITCH: other egress ports of same node
        # --------------------------------------------------
        if typ == 'switch':
            return [
                edge
                for node in nodes_names
                for edge in graph.out_edges(node, data=True)
                if edge[2].get('sourcePort') not in egports
            ]

        # --------------------------------------------------
        # TALKER: all egress ports of other hosts
        # --------------------------------------------------
        if typ == 'talker':
            return [
                edge
                for node in graph.nodes
                if node not in nodes_names and node.startswith('host')
                for edge in graph.out_edges(node, data=True)
            ]

        # --------------------------------------------------
        # LISTENER: all ingress ports of other hosts
        # --------------------------------------------------
        if typ == 'listener':
            return [
                edge
                for node in graph.nodes
                if node not in nodes_names and node.startswith('host')
                for edge in graph.in_edges(node, data=True)
            ]

        raise ValueError(
            "Invalid typ for get_concurrent_ports. Must be one of: switch, talker, listener"
        )
    
    # ---------------- Serialization Helpers ----------------
    @staticmethod
    def graph_encoder(graph: nx.DiGraph, source="sourcePort", target="destPort", name="name",
                    linkSym="links", nodeSym="nodes") -> dict:
        """
        Custom encoder for networkx DiGraph that adds port information to nodes.
        Mimics the behavior of `topology2json` but does not require self.ports.
        Returns a dict ready for JSON serialization.
        """

        def map_ports(node, data):
            # compute ports from graph directly
            ingress_ports = data.get("ingress_ports", {})
            egress_ports = data.get("egress_ports", {})
            common_keys = set(ingress_ports) & set(egress_ports)
            ports = []
            for port in common_keys:
                pdata = egress_ports[port].copy()  # use egress port as base
                pdata['name'] = pdata['sourcePort']  # only the port name
                # remove unneeded fields
                for k in ['sourceNode', 'sourcePort', 'destNode', 'destPort']:
                    pdata.pop(k, None)
                if pdata.get('duplex') == 'rx':
                    pdata.pop('speed', None)
                ports.append(pdata)
            return ports

        if len({source, target, name}) < 3:
            raise nx.NetworkXError("Attribute names are not unique.")

        node_list = []
        for n, data in graph.nodes(data=True):
            node_data = {
                'type': data.get('type'),
                name: n,
                'ports': map_ports(n, data)
            }
            if data.get('type') == 'switch':
                node_data['procDelay'] = data.get('procDelay')
            node_list.append(node_data)

        edge_list = [d for _, _, d in graph.edges(data=True)]

        return {
            nodeSym: node_list,
            linkSym: edge_list
        }
    
    @staticmethod
    def graph_decoder(impUC: dict) -> nx.DiGraph:
        """Decode a JSON string into a networkx DiGraph with proper port dictionaries."""

        G = nx.DiGraph()

        def map_ports(node):
            """Reconstruct ingress_ports and egress_ports from node['ports']."""
            node_data = {
                'type': node.get('type', 'host')
            }
            if node_data['type'] == 'switch':
                node_data['procDelay'] = node.get('procDelay', 0)

            node_data['ingress_ports'] = {}
            node_data['egress_ports'] = {}

            for port in node.get('ports', []):
                duplex = port['duplex']
                name = port['name']

                data = {
                    'sourceNode': node['name'],
                    'sourcePort': name,
                    'duplex': duplex
                }

                # reconstruct egress_ports and ingress_ports
                if duplex == 'rx':
                    node_data['ingress_ports'][name] = data.copy()
                elif duplex == 'tx':
                    data['speed'] = port.get('speed', 0)
                    data['queueCount'] = port.get('queueCount', 0)
                    node_data['egress_ports'][name] = data.copy()
                elif duplex == 'full':
                    node_data['ingress_ports'][name] = data.copy()
                    data['speed'] = port.get('speed', 0)
                    data['queueCount'] = port.get('queueCount', 0)
                    node_data['egress_ports'][name] = data.copy()

            return node_data

        # Add nodes
        for node in impUC.get('nodes', []):
            G.add_node(node['name'], **map_ports(node))

        # Add edges
        for link in impUC.get('links', []):
            src = link['sourceNode']
            dst = link['destNode']
            G.add_edge(src, dst, **link)

            # Update the egress_ports with destNode/destPort
            port_name = link['sourcePort'].split('.')[1]
            if port_name in G.nodes[src]['egress_ports']:
                G.nodes[src]['egress_ports'][port_name].update({
                    'destNode': dst,
                    'destPort': link['destPort'].split('.')[1]
                })

        return G
    
