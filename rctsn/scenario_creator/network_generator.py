import csv
import networkx as nx
import matplotlib.pyplot as plt
import os
import random
from typing import List, Tuple, Dict, Optional, Any
import pickle


def save_network_generator(factory, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(factory, f)


def load_network_generator(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


class NetworkGenerator:
    MAX_ATTEMPTS = 1000  # Prevent infinite loops

    def __init__(self, subnet_id: str = "0"):
        """
        Initialize the NetworkGenerator.

        :param subnet_id: Identifier for the network's subnet. "0" denotes the main network.
        """
        self.subnet_id: str = subnet_id
        self.network: nx.Graph = nx.Graph()
        # Global counters for each type; used for unique numbering.
        self.counters: Dict[str, int] = {"SW": 0, "WN": 0, "WED": 0, "ED": 0}
        self.streams: List[Dict[str, Any]] = []
        # Mapping of subnetwork names to list of node names.
        self.subnetworks: Dict[str, List[str]] = {}
        self._device_cache: Dict[str, set] = {"WED": set(), "ED": set()}

    # --- Helper methods ---
    @staticmethod
    def _get_core_type(node_name: str) -> str:
        """
        Extract the core type from a node name.
        For example, '1-SW3' returns 'SW' (ignoring the subnet prefix and number).

        :param node_name: Full node name.
        :return: Core node type.
        """
        core = node_name.split("-", 1)[-1]
        for t in ["WED", "ED", "WN", "SW"]:
            if core.startswith(t):
                return t
        raise ValueError(f"Unknown node type for node: {node_name}")

    def _update_counter(self, node_type: str) -> int:
        """
        Update and return the counter for a given node type.
        """
        self.counters[node_type] += 1
        return self.counters[node_type]

    def _add_node(self, prefix: str, subnet_id: Optional[str] = None) -> str:
        """
        Create and add a node to the network with a given prefix and subnet_id.
        Also set node attributes for later filtering.

        :param prefix: One of "SW", "WN", "WED", or "ED".
        :param subnet_id: The subnet identifier to use (if None, uses self.subnet_id).
        :return: The full node name.
        """
        effective_subnet = subnet_id if subnet_id is not None else self.subnet_id
        num = self._update_counter(prefix)
        node_name = f"{effective_subnet}-{prefix}{num}"
        self.network.add_node(node_name, subnet=effective_subnet, type=prefix)
        # If the node is a device, cache it.
        if prefix in ["WED", "ED"]:
            self._device_cache[prefix].add(node_name)
        return node_name

    def _add_edge(self, a: str, b: str) -> None:
        """
        Add an edge between nodes a and b and assign uplink/downlink speeds.

        :param a: Source node.
        :param b: Destination node.
        """
        self.network.add_edge(a, b)
        link_type = self._map_link_type(a, b)
        self.network[a][b]["link_type"] = link_type

    def _map_link_type(self, a: str, b: str) -> Tuple[int, int]:
        """
        Determine link type between two nodes based on their core types.

        :param a: Node name.
        :param b: Node name.
        :return: Tuple of (uplink_speed, downlink_speed)
        """
        type_map = {
            ("WED", "WN"): "1",
            ("WN", "SW"): "0",
            ("ED", "SW"): "0",
            ("SW", "SW"): "0",
        }
        a_type = self._get_core_type(a)
        b_type = self._get_core_type(b)
        for (t1, t2), type in type_map.items():
            if (a_type == t1 and b_type == t2) or (a_type == t2 and b_type == t1):
                return type
        return "type_1"  # default type

    # --- Topology creation methods ---
    def create_line(self, num_nodes: int) -> List[str]:
        """Create a line (chain) topology of switches."""
        nodes = [self._add_node("SW") for _ in range(num_nodes)]
        for i in range(len(nodes) - 1):
            self._add_edge(nodes[i], nodes[i + 1])
        return nodes

    def create_star(self, num_nodes: int) -> str:
        """Create a star topology; return the central switch node."""
        central = self._add_node("SW")
        for _ in range(num_nodes - 1):
            peripheral = self._add_node("SW")
            self._add_edge(central, peripheral)
        return central

    def create_ring(self, num_nodes: int) -> List[str]:
        """Create a ring topology of switches."""
        nodes = [self._add_node("SW") for _ in range(num_nodes)]
        for i in range(num_nodes):
            self._add_edge(nodes[i], nodes[(i + 1) % num_nodes])
        return nodes

    def create_mesh(self, num_nodes: int) -> List[str]:
        """Create a fully connected mesh topology of switches."""
        nodes = [self._add_node("SW") for _ in range(num_nodes)]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                self._add_edge(nodes[i], nodes[j])
        return nodes

    # --- Subnetwork merging ---
    def connect_subnetwork(
        self,
        subnet: "NetworkGenerator",
        connect_to: Optional[str] = None,
        connect_from: Optional[str] = None,
        subnetwork_name: Optional[str] = None,
        link_type: Optional[str] = None,
    ) -> str:
        """
        Merge a separate NetworkGenerator as a subnetwork. All nodes in the subnet will be
        renamed with the new subnetwork identifier. You may provide a custom subnetwork name.

        :param subnet: The NetworkGenerator instance to merge.
        :param connect_to: Node in the main network to which the subnetwork will connect.
        :param connect_from: Node in the subnet to use as the connection point. If None, the first switch is used.
        :param subnetwork_name: Optional custom name for the subnetwork.
        :param link_type: Optional custom link type for the connection (e.g., "2").
        :return: The name of the subnetwork's root switch.
        """
        if not self.network.nodes:
            raise ValueError("Main network is empty; create it first.")

        # Determine the new subnet identifier.
        new_subnet_id = (
            subnetwork_name
            if subnetwork_name is not None
            else str(len(self.subnetworks) + 1)
        )
        subnet.subnet_id = new_subnet_id

        # Save the original name of the connect_from node before renaming
        connect_from_original = connect_from
        
        rename_map: Dict[str, str] = {}
        # Remove any old subnet prefix and assign the new one.
        for node in subnet.network.nodes():
            core = node.split("-", 1)[-1]
            node_type = None
            for t in ["WED", "ED", "WN", "SW"]:
                if core.startswith(t):
                    node_type = t
                    break
            if node_type is None:
                raise ValueError(f"Unknown node type for node: {node}")
            new_number = self._update_counter(node_type)
            new_name = f"{new_subnet_id}-{node_type}{new_number}"
            rename_map[node] = new_name
            
            # Update connect_from if needed
            if node == connect_from_original:
                connect_from = new_name

        # Relabel the subnet's nodes.
        subnet_network = nx.relabel_nodes(subnet.network, rename_map)

        # Update the subnet attribute for all nodes in the merged subnet.
        for node in subnet_network.nodes():
            subnet_network.nodes[node]["subnet"] = new_subnet_id

        # Merge the relabeled subnet into the main network.
        self.network = nx.compose(self.network, subnet_network)

        # Record this subnetwork.
        self.subnetworks[new_subnet_id] = list(subnet_network.nodes)

        # Identify a switch in the subnet to serve as the connection point if not specified
        if connect_from is None:
            sub_switches = [
                n for n in subnet_network.nodes if self._get_core_type(n) in ["SW", "WN"]
            ]
            if not sub_switches:
                raise ValueError("Subnetwork has no switches.")
            connect_from = sub_switches[0]
        elif connect_from not in self.network.nodes:
            raise ValueError(f"Connection node {connect_from} not found in subnet network.")

        # Connect the subnetwork to the main network.
        connect_to = connect_to or next(iter(self.network.nodes))
        
        # Add the edge with custom link type if provided
        self.network.add_edge(connect_to, connect_from)
        
        # Assign the link type (custom or mapped)
        if link_type is not None:
            self.network[connect_to][connect_from]["link_type"] = link_type
        else:
            self.network[connect_to][connect_from]["link_type"] = self._map_link_type(connect_to, connect_from)

        # Update device caches.
        for node in subnet_network.nodes():
            if self._get_core_type(node) in ["WED", "ED"]:
                self._device_cache[self._get_core_type(node)].add(node)

        return connect_from

    # --- Device management ---
    def add_wireless_devices(self, wn_count: int, wed_per_wn: int) -> None:
        """
        Add wireless devices. For each randomly selected switch, add a wireless node (WN)
        and then attach a specified number of wireless endpoints (WED) to it.
        New nodes inherit the subnet prefix from their parent switch.
        """
        sw_nodes = [n for n in self.network.nodes if self._get_core_type(n) == "SW"]
        selected = random.sample(sw_nodes, min(wn_count, len(sw_nodes)))
        for sw in selected:
            sw_subnet = self.network.nodes[sw]["subnet"]
            wn = self._add_node("WN", subnet_id=sw_subnet)
            self._add_edge(sw, wn)
            for _ in range(wed_per_wn):
                wn_subnet = self.network.nodes[wn]["subnet"]
                wed = self._add_node("WED", subnet_id=wn_subnet)
                self._add_edge(wn, wed)

    def add_wired_devices(self, devices_per_switch: int) -> None:
        """
        Add wired devices to each switch. Each wired device (ED) inherits the subnet
        of its parent switch.
        """
        for sw in [n for n in self.network.nodes if self._get_core_type(n) == "SW"]:
            sw_subnet = self.network.nodes[sw]["subnet"]
            for _ in range(devices_per_switch):
                ed = self._add_node("ED", subnet_id=sw_subnet)
                self._add_edge(sw, ed)

    # --- Stream generation ---
    def generate_streams_with_dict_settings(
        self,
        wed_to_wed: int = 0,
        ed_to_ed: int = 0,
        wed_to_ed: int = 0,
        mode: str = "any",
        wed_wed_settings: Dict[str, List] = {
            "packet_sizes": [32, 64, 128, 256, 512, 1024, 1420],
            "periods": [8000],
            "latency_factors": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "priorities": [1, 2, 3],
            "gammas": [1.0],
            "jitter_factors": [0.1, 0.2, 0.3, 0.4, 0.5]
        },
        ed_ed_settings: Dict[str, List] = {
            "packet_sizes": [32, 64, 128, 256, 512, 1024, 1420],
            "periods": [500, 1000, 2000, 4000, 8000],
            "latency_factors": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "priorities": [1, 2, 3],
            "gammas": [1.0],
            "jitter_factors": [0.1, 0.2, 0.3, 0.4, 0.5]
        },
        wed_ed_settings: Dict[str, List] = {
            "packet_sizes": [32, 64, 128, 256, 512, 1024, 1420],
            "periods": [4000, 8000],
            "latency_factors": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "priorities": [1, 2, 3],
            "gammas": [1.0],
            "jitter_factors": [0.1, 0.2, 0.3, 0.4, 0.5]
        },
        shuffle_streams: bool = True,
        shuffle_scope: str = "all",  # "all" | "batch" | "none"
    ) -> int:
        """
        Generate bidirectional streams. Safe to call multiple times; streams are appended.
        shuffle_scope:
          - "all": shuffle the entire list (backward compatible default)
          - "batch": shuffle only the newly added streams
          - "none": do not shuffle
        Returns: number of streams added in this call.
        """
        start_len = len(self.streams)
        wed_devices = list(self._device_cache["WED"])
        ed_devices = list(self._device_cache["ED"])

        self._validate_stream_requests(
            wed_devices, ed_devices, wed_to_wed, ed_to_ed, wed_to_ed
        )

        if mode == "subnet":
            device_groups = self._get_subnet_device_groups()
            if not device_groups:
                raise ValueError("No devices found in any subnetwork.")
        else:
            device_groups = [wed_devices + ed_devices]

        self._generate_stream_pairs_dict(
            wed_devices, wed_devices, wed_to_wed, device_groups, wed_wed_settings
        )
        self._generate_stream_pairs_dict(
            ed_devices, ed_devices, ed_to_ed, device_groups, ed_ed_settings
        )
        self._generate_stream_pairs_dict(
            wed_devices, ed_devices, wed_to_ed, device_groups, wed_ed_settings
        )
        
        added = len(self.streams) - start_len

        if shuffle_streams and added > 0:
            if shuffle_scope == "all":
                random.shuffle(self.streams)
            elif shuffle_scope == "batch":
                batch = self.streams[start_len:]
                random.shuffle(batch)
                self.streams[start_len:] = batch
            # "none" does nothing

            print(f"Streams shuffled (scope={shuffle_scope}, total={len(self.streams)})")
        else:
            print(f"Streams added without shuffling (added={added}, total={len(self.streams)})")

        return added

    def _generate_stream_pairs_dict(
        self,
        src_pool: List[str],
        dst_pool: List[str],
        count: int,
        groups: List[List[str]],
        settings: Dict[str, List],
    ) -> None:
        """Generate bidirectional streams between pairs of devices using dictionary settings."""
        if not groups:
            raise ValueError("No valid device groups for stream generation.")

        # Extract settings from the dictionary
        packet_size_options = settings["packet_sizes"]
        period_options = settings["periods"]
        latency_factors = settings["latency_factors"]
        priority_options = settings["priorities"]
        gamma_options = settings["gammas"]
        # Get jitter factors with a default of [0.0] if not specified
        jitter_factors = settings.get("jitter_factors", [0.0])
                
        valid_groups = [g for g in groups if len(g) >= 2]
        if not valid_groups:
            raise ValueError("No non-empty device groups found.")

        generated = 0
        attempts = 0
        while generated < count and attempts < self.MAX_ATTEMPTS:
            group = random.choice(valid_groups)
            valid_src = [d for d in group if d in src_pool]
            valid_dst = [d for d in group if d in dst_pool]
            if valid_src and valid_dst:
                src = random.choice(valid_src)
                dst = random.choice(valid_dst)
                if src != dst:
                    # Generate forward stream parameters
                    if isinstance(packet_size_options, tuple):
                        packet_size = random.randint(*packet_size_options)
                    else:
                        packet_size = random.choice(packet_size_options)
                    period = random.choice(period_options)
                    
                    # Generate max_latency based on latency factors
                    latency_factor = random.choice(latency_factors) 
                    max_latency = int(period * latency_factor)
                    
                    # Generate max_jitter based on jitter factors
                    jitter_factor = random.choice(jitter_factors)
                    max_jitter = int(period * jitter_factor)
                    
                    # Generate priority
                    priority = random.choice(priority_options)
                    
                    # Generate gamma
                    gamma = random.choice(gamma_options)
                    
                    # Add bidirectional streams
                    self._add_bidirectional_stream_dict(
                        src, dst, packet_size, period, max_latency, priority, gamma, max_jitter, settings
                    )
                    generated += 1
            attempts += 1

    def _validate_stream_requests(
        self,
        wed_devices: List[str],
        ed_devices: List[str],
        wed_to_wed: int,
        ed_to_ed: int,
        wed_to_ed: int,
    ) -> None:
        """Ensure there are enough devices to generate the requested streams."""
        if len(wed_devices) < 2 and wed_to_wed > 0:
            raise ValueError("Not enough WED devices for WED-WED streams.")
        if len(ed_devices) < 2 and ed_to_ed > 0:
            raise ValueError("Not enough ED devices for ED-ED streams.")
        if (not wed_devices or not ed_devices) and wed_to_ed > 0:
            raise ValueError("Missing devices for WED-ED streams.")

    def _get_subnet_device_groups(self) -> List[List[str]]:
        """
        Get groups of devices (WED/ED) for each subnetwork.
        Only groups with at least one switch and connected devices are returned.
        """
        groups = []
        for sub_nodes in self.subnetworks.values():
            wed_devices = []
            ed_devices = []
            for node in sub_nodes:
                node_core_type = self._get_core_type(node)
                if node_core_type in {"SW", "WN"}:
                    for neighbor in self.network.neighbors(node):
                        neighbor_core_type = self._get_core_type(neighbor)
                        if neighbor_core_type == "WED":
                            wed_devices.append(neighbor)
                        elif neighbor_core_type == "ED":
                            ed_devices.append(neighbor)
            if wed_devices or ed_devices:
                groups.append(wed_devices + ed_devices)
        return groups

    def _add_bidirectional_stream_dict(
        self, 
        src: str, 
        dst: str, 
        packet_size: int, 
        period: int, 
        max_latency: int,
        priority: int,
        gamma: float,
        max_jitter: int,
        settings: Dict[str, List],
    ) -> None:
        """
        Create and add a bidirectional stream (both directions) with independent parameters.
        The reverse stream gets its own randomly-generated parameters from the same settings.
        Uses dictionary-based settings for clearer parameter specification.
        """
        # Forward stream
        stream = {
            "src": src,
            "dst": dst,
            "packet_size": packet_size,
            "period": period,
            "max_latency": max_latency,
            "priority": priority,
            "gamma": gamma,
            "max_jitter": max_jitter,  # Add max_jitter field
        }
        self.streams.append(stream)
        
        # Extract settings from the dictionary
        packet_size_options = settings["packet_sizes"]
        period_options = settings["periods"]
        latency_factors = settings["latency_factors"]
        priority_options = settings["priorities"]
        gamma_options = settings["gammas"]
        jitter_factors = settings.get("jitter_factors", [0.0])  # Default to [0.0] if not specified
        
        # Generate reverse stream parameters
        if isinstance(packet_size_options, tuple):
            reverse_packet_size = random.randint(*packet_size_options)
        else:
            reverse_packet_size = random.choice(packet_size_options)
        
        reverse_period = random.choice(period_options)
        
        # Generate max_latency based on latency factors
        latency_factor = random.choice(latency_factors)
        reverse_max_latency = int(reverse_period * latency_factor)
        
        # Generate max_jitter based on jitter factors
        jitter_factor = random.choice(jitter_factors)
        reverse_max_jitter = int(reverse_period * jitter_factor)
        
        # Generate priority for reverse stream
        reverse_priority = random.choice(priority_options)
        
        # Generate gamma for reverse stream
        reverse_gamma = random.choice(gamma_options)
        
        # Create reverse stream with independently generated parameters
        reverse_stream = {
            "src": dst,
            "dst": src,
            "packet_size": reverse_packet_size,
            "period": reverse_period,
            "max_latency": reverse_max_latency,
            "priority": reverse_priority,
            "gamma": reverse_gamma,
            "max_jitter": reverse_max_jitter,  # Add max_jitter field
        }
        self.streams.append(reverse_stream)

    # --- Export and Visualization ---
    def export(self, output_dir: str = "output") -> None:
        """
        Export the network as GraphML.
        """
        os.makedirs(output_dir, exist_ok=True)
        di_graph = nx.DiGraph(self.network)
        nx.write_graphml(di_graph, os.path.join(output_dir, "network.graphml"))

    def export_streams(self, csv_file: str = "output/streams.csv") -> None:
        """
        Export the generated streams to a CSV file.
        Each row represents a stream with its parameters.
        """
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        with open(csv_file, "w", newline="") as csvfile:
            fieldnames = ["index", "src", "dst", "packet_size", "period", "max_latency", "priority", "gamma", "max_jitter"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for index, stream in enumerate(self.streams):
                stream_with_index = {"index": index, **stream}
                writer.writerow(stream_with_index)
        print(f"Streams exported to {csv_file}")

    def visualize(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the network topology with a legend and link types.
        Uses curved edges to prevent overlapping and improves layout for better readability.
        """
        color_map = {
            "SW": "lightblue",
            "WN": "orange",
            "WED": "yellow",
            "ED": "lightgreen",
        }
        colors = []
        for node in self.network.nodes():
            node_type = self._get_core_type(node)
            colors.append(color_map.get(node_type, "gray"))

        plt.figure(figsize=(16, 12))
        
        # Use Kamada-Kawai layout for better distribution of nodes (less edge crossings)
        # This algorithm produces layouts with less overlapping edges than spring_layout
        try:
            pos = nx.kamada_kawai_layout(self.network)
        except:
            # Fall back to spring layout if kamada_kawai fails
            pos = nx.spring_layout(self.network, k=0.4, iterations=200, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.network,
            pos,
            node_color=colors,
            node_size=300,
            alpha=0.8
        )
        
        # Draw node labels
        nx.draw_networkx_labels(
            self.network,
            pos,
            font_size=6,
            font_weight="bold"
        )
        
        # Draw edges with curved lines to reduce overlap
        # This will create curved edges between nodes that have multiple connections
        edge_list = list(self.network.edges())
        
        # Group edges by node pairs to detect parallel edges
        edge_groups = {}
        for u, v in edge_list:
            if (u, v) not in edge_groups and (v, u) not in edge_groups:
                edge_groups[(u, v)] = []
            if (u, v) in edge_groups:
                edge_groups[(u, v)].append((u, v))
            else:
                edge_groups[(v, u)].append((u, v))
        
        # Draw edges with different curve styles
        for (u, v), edges in edge_groups.items():
            # Use curved edges with connectionstyle
            nx.draw_networkx_edges(
                self.network,
                pos,
                edgelist=[edges[0]],
                width=0.8,
                alpha=0.7,
                edge_color='gray',
                connectionstyle=f"arc3,rad={0.2}"
            )
        
        # Draw edge labels with smaller font
        edge_labels = nx.get_edge_attributes(self.network, "link_type")
        # Adjust label positions slightly to avoid overlap with curved edges
        nx.draw_networkx_edge_labels(
            self.network, 
            pos, 
            edge_labels=edge_labels, 
            font_size=5,
            label_pos=0.4,  # Position labels closer to edges' middle
            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7)  # Add background to labels
        )

        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Switch (SW)",
                markerfacecolor="lightblue",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Wireless Node (WN)",
                markerfacecolor="orange",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Wireless End-Device (WED)",
                markerfacecolor="yellow",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Wired End-Device (ED)",
                markerfacecolor="lightgreen",
                markersize=8,
            ),
        ]
        # Move legend outside the main plot
        plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=8)
        plt.title("Network Topology")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def save_network_stats(self, output_file: str, header: str, extra_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Save network statistics to a text file.
        
        :param output_file: Path to the output text file.
        :param header: Header text for the file.
        :param extra_info: Optional dictionary of extra information to append (e.g., jitter settings).
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Calculate node counts by type
        node_types = {"SW": 0, "WN": 0, "WED": 0, "ED": 0}
        for node in self.network.nodes():
            node_type = self._get_core_type(node)
            if node_type in node_types:
                node_types[node_type] += 1
                
        # Count links by type
        link_types = {}
        for u, v, data in self.network.edges(data=True):
            link_type = data.get("link_type", "unknown")
            link_types[link_type] = link_types.get(link_type, 0) + 1
            
        with open(output_file, "w") as f:
            f.write(f"{header}\n")
            f.write("=" * len(header) + "\n\n")
            
            f.write("Nodes:\n")
            f.write(f"  Total nodes: {self.network.number_of_nodes()}\n")
            for node_type, count in node_types.items():
                f.write(f"  {node_type} nodes: {count}\n")
            
            f.write("\nLinks:\n")
            f.write(f"  Total links: {self.network.number_of_edges()}\n")
            for link_type, count in link_types.items():
                f.write(f"  {link_type} links: {count}\n")
            
            f.write("\nStreams:\n")
            f.write(f"  Total streams: {len(self.streams)}\n")
            
            if extra_info:
                f.write("\nAdditional Information:\n")
                for key, value in extra_info.items():
                    f.write(f"  {key}: {value}\n")
        
        print(f"Network statistics saved to {output_file}")
