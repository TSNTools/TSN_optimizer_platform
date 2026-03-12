import math
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from utils.path_calculator import k_shortest_paths
from utils.helper import calculate_lcm, calculate_repetitions_list


class DataFactory:
    """
    Provides functions for working with streams and their associated ports.
    Handles processing of streams data for optimization.
    """

    def __init__(self, link_config: Dict, graph_nx: nx.Graph, raw_streams: List[Dict],
                 k_path_count: int = 3, lcm_rep: int = 1, stream_limit: Optional[int] = None):
        """
        Initialize DataFactory with pre-loaded data.

        Args:
            link_config (Dict): The configuration dictionary.
            graph_nx (nx.Graph): The network graph.
            raw_streams (List[Dict]): List of stream dictionaries.
            k_path_count (int): Number of k-shortest paths to calculate for each stream.
            lcm_rep (int): LCM repetition factor.
            stream_limit (Optional[int]): Maximum number of streams to process. If None, process all.
        """
        self.link_config = link_config
        self.graph_nx = graph_nx
        
        # Apply stream limit if specified
        if stream_limit is not None and stream_limit < len(raw_streams):
            self.raw_streams = raw_streams[:stream_limit]
            print(f"Limiting streams from {len(raw_streams)} to {stream_limit}")
        else:
            self.raw_streams = raw_streams
            
        self.k_path_count = k_path_count
        self.lcm_rep = lcm_rep
        self.M = 9999999
        self.lcm = 1  # Initialize LCM to 1

        # Initialize the stream dictionary
        self.stream_dict = {}
        # Initialize ports set
        self.ports = set()
        
        # Process streams
        self.process_streams()
        
        # Set streams to be the keys of stream_dict
        self.streams = list(self.stream_dict.keys())

    def process_streams(self, start_index: int = 0, limit: Optional[int] = None):
        """
        Process streams from raw_streams starting at start_index up to limit.
        If limit is None, process all remaining streams from start_index.
        
        Args:
            start_index (int): Index to start processing from
            limit (Optional[int]): Maximum number of streams to process from start_index
            
        Returns:
            int: Number of streams processed
        """
        # Calculate end index
        end_index = len(self.raw_streams) if limit is None else min(start_index + limit, len(self.raw_streams))
        
        # Process streams in range
        count = 0
        for stream_idx in range(start_index, end_index):
            self._process_single_stream(stream_idx, self.raw_streams[stream_idx])
            count += 1
            
        # Recalculate global values after processing streams
        self._calculate_global_lcm()
        self._calculate_repetitions_for_all_streams()
        self._extract_ports()
        
        # Update streams list
        self.streams = list(self.stream_dict.keys())
        
        return count

    def add_stream(self, stream_data: Dict[str, Any], stream_id: Optional[int] = None) -> int:
        """
        Add a new stream to the stream_dict and process all necessary calculations.

        Args:
            stream_data (Dict[str, Any]): The stream data with required fields:
                - src (str): Source node
                - dst (str): Destination node
                - packet_size (int): Size of the packet in bytes
                - period (int): Period of the stream
                Optional fields:
                - max_latency (int): Maximum allowed latency
                - priority (int): Stream priority
                - gamma (float): Gamma value for calculations (default: 1.0)
                - max_jitter (int): Maximum allowed jitter
            stream_id (Optional[int]): Custom stream ID to use. If None, next available ID is used.

        Returns:
            int: The ID of the added stream.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        # Validate required fields
        required_fields = ["src", "dst", "packet_size", "period"]
        for field in required_fields:
            if field not in stream_data:
                raise ValueError(f"Missing required field '{field}' in stream_data")
        
        # Validate and convert data types
        processed_data = {
            "src": str(stream_data["src"]),
            "dst": str(stream_data["dst"]),
            "packet_size": int(stream_data["packet_size"]),
            "period": int(stream_data["period"]),
            "gamma": float(stream_data["gamma"]) # Ensure gamma is float, default 1.0
        }
        
        # Handle optional fields
        if "max_latency" in stream_data and stream_data["max_latency"] is not None:
            processed_data["max_latency"] = int(stream_data["max_latency"])
        if "priority" in stream_data and stream_data["priority"] is not None:
            processed_data["priority"] = int(stream_data["priority"])
        if "max_jitter" in stream_data and stream_data["max_jitter"] is not None:
            processed_data["max_jitter"] = int(stream_data["max_jitter"])
        
        # Determine stream ID
        next_id = max(self.stream_dict.keys() or [-1]) + 1
        actual_stream_id = stream_id if stream_id is not None else next_id
        
        if actual_stream_id in self.stream_dict:
            raise ValueError(f"Stream ID {actual_stream_id} already exists")
        
        # Process the new stream
        self._process_single_stream(actual_stream_id, processed_data)
        
        # Recalculate LCM if necessary
        if self.stream_dict[actual_stream_id]:  # If stream has valid paths
            periods = [
                self.stream_dict[actual_stream_id][path]["period"]
                for path in self.stream_dict[actual_stream_id]
                if isinstance(self.stream_dict[actual_stream_id][path], dict) and
                "period" in self.stream_dict[actual_stream_id][path]
            ]
            if periods:
                new_lcm = calculate_lcm(periods + [self.lcm])
                if new_lcm != self.lcm:
                    self.lcm = new_lcm
                    # Recalculate repetitions for all streams if LCM changed
                    self._calculate_repetitions_for_all_streams()
                else:
                    # Calculate repetitions just for the new stream
                    self._calculate_repetitions_for_stream(actual_stream_id)
            
            # Update ports set with new stream's ports
            self._extract_ports()
            
            # Update streams list
            if actual_stream_id not in self.streams:
                self.streams = list(self.stream_dict.keys())
        
        return actual_stream_id

    def _process_single_stream(self, stream_idx: int, stream_data: Dict[str, Any]):
        """
        Process a single stream and populate the stream dictionary.

        Args:
            stream_idx (int): The stream index/ID.
            stream_data (Dict[str, Any]): The stream data dictionary.
        """
        # Initialize the stream in stream_dict
        self.stream_dict[stream_idx] = {}
        
        try:
            # Get shortest paths
            shortest_paths = k_shortest_paths(
                G=self.graph_nx, 
                source=stream_data["src"], 
                target=stream_data["dst"], 
                k=self.k_path_count
            )
            
            if not shortest_paths:
                print(f"Warning: No paths found for stream {stream_idx} from {stream_data['src']} to {stream_data['dst']}")
                self.stream_dict[stream_idx]["error"] = "No paths found"
                return
            
            path_counter = 0
            
            for base_path in shortest_paths:
                link_parallel_paths = {}
                
                # Check which links in the path need multiple parallel paths
                for i, port_tuple in enumerate(base_path):
                    sender, receiver = port_tuple
                    try:
                        link_type = self._get_link_type(sender, receiver)
                        # Get number of parallel paths from config
                        link_type_str = str(link_type)
                        num_parallel = self.link_config["link_types"][link_type_str].get("parallel_paths", 1)
                        
                        if num_parallel > 1:
                            link_parallel_paths[(i, port_tuple)] = (link_type, num_parallel)
                    except Exception as e:
                        print(f"Error processing link {port_tuple} in stream {stream_idx}: {e}")
                        continue
                
                if link_parallel_paths:
                    # Calculate total number of path variants
                    max_duplicates = max(num_parallel for _, (_, num_parallel) in link_parallel_paths.items())
                    
                    for duplicate_idx in range(max_duplicates):
                        # Create a modified path with port identifiers
                        modified_path = []
                        for i, port_tuple in enumerate(base_path):
                            sender, receiver = port_tuple
                            link_type = self._get_link_type(sender, receiver)
                            link_type_str = str(link_type)
                            num_parallel = self.link_config["link_types"][link_type_str].get("parallel_paths", 1)
                            
                            if num_parallel > 1:
                                # Use modulo to distribute across available parallel paths
                                path_idx = duplicate_idx % num_parallel + 1
                                # Add port identifier to both nodes
                                modified_port = (f"{sender}#{path_idx}", f"{receiver}#{path_idx}")
                            else:
                                modified_port = port_tuple
                            
                            modified_path.append(modified_port)
                        
                        # Process the modified path
                        path_data = self._process_path(stream_data, modified_path)
                        self.stream_dict[stream_idx][path_counter] = path_data
                        path_counter += 1
                else:
                    # No parallel links needed, process the original path
                    path_data = self._process_path(stream_data, base_path)
                    self.stream_dict[stream_idx][path_counter] = path_data
                    path_counter += 1
        except Exception as e:
            print(f"Error processing stream {stream_idx}: {e}")
            self.stream_dict[stream_idx]["error"] = f"Processing error: {str(e)}"

    def _get_link_type(self, sender: str, receiver: str) -> int:
        """
        Get the link type between sender and receiver nodes.

        Args:
            sender (str): Sender node.
            receiver (str): Receiver node.

        Returns:
            int: Link type.

        Raises:
            ValueError: If link type cannot be determined.
        """
        # Remove any port identifier (part after #)
        sender_base = sender.split('#')[0] if '#' in sender else sender
        receiver_base = receiver.split('#')[0] if '#' in receiver else receiver
        
        try:
            return int(self.graph_nx[sender_base][receiver_base]["link_type"])
        except KeyError:
            raise ValueError(f"Link type not found for edge ({sender_base}, {receiver_base})")

    def _calculate_transmission_time(self, byte_size: int, link_type: int, sender: str, receiver: str) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Calculate transmission times and deviations for frames.
        For wireless links, considers uplink/downlink based on sender/receiver names.

        Args:
            byte_size (int): Size of data in bytes.
            link_type (int): Link type identifier.
            sender (str): Sender node name (can include port identifier).
            receiver (str): Receiver node name (can include port identifier).

        Returns:
            Tuple[Dict[int, float], Dict[int, float]]: Dictionaries mapping frame indices to times and deviations.
        """
        link_type_str = str(link_type)
        link_conf_for_type = self.link_config["link_types"][link_type_str]
        times_dict = {}
        deviations_dict = {}
        
        if link_type_str == "0":  # Wired link
            link_speed = link_conf_for_type["best_case_speed"]
            worst_case_speed = link_conf_for_type["worst_case_speed"]
            processing_time = link_conf_for_type["processing_time"]
            propagation_time = link_conf_for_type["propagation_time"]
            
            num_frames = math.ceil(byte_size / 1500) if byte_size > 0 else 1
            frame_size = math.ceil(byte_size / num_frames) if num_frames > 0 else 0
            
            for frame_index in range(num_frames):
                best_case_time = (
                    math.ceil((frame_size * 8) / link_speed) +
                    processing_time +
                    propagation_time
                )
                worst_case_time = (
                    math.ceil((frame_size * 8) / worst_case_speed) +
                    processing_time +
                    propagation_time
                )
                
                times_dict[frame_index] = best_case_time
                deviations_dict[frame_index] = worst_case_time - best_case_time
        else:  # Wireless link (or other types using packet_sizes)
            # Determine direction for wireless links based on node names
            sender_base = sender.split('#')[0]
            receiver_base = receiver.split('#')[0]
            direction = None
            if "WED" in sender_base and "WN" in receiver_base:
                direction = "uplink"
            elif "WN" in sender_base and "WED" in receiver_base:
                direction = "downlink"

            available_sizes = sorted(map(int, link_conf_for_type["packet_sizes"].keys()), reverse=True)
            remaining_size = byte_size
            frame_index = 0
            
            while remaining_size > 0:
                found_fit = False
                for size in available_sizes:
                    if size <= remaining_size:
                        size_key = str(size)
                        packet_size_entry = link_conf_for_type["packet_sizes"][size_key]
                        
                        times_data_source = None
                        if direction and direction in packet_size_entry:
                            times_data_source = packet_size_entry[direction]
                        elif "best_case_time" in packet_size_entry: # Fallback if no direction or direction not in keys
                            times_data_source = packet_size_entry
                            if direction and direction not in packet_size_entry:
                                print(f"Warning: Link type {link_type_str}, packet size {size_key}: Direction '{direction}' not found. Using direct time fields for {sender_base}->{receiver_base}.")
                        else:
                            raise KeyError(
                                f"Config for link_type {link_type_str}, packet_size {size_key} "
                                f"is missing '{direction}' specific times or direct best_case_time/worst_case_time fields for {sender_base}->{receiver_base}."
                            )

                        times_dict[frame_index] = times_data_source["best_case_time"]
                        deviations_dict[frame_index] = times_data_source["worst_case_time"] - times_dict[frame_index]
                        remaining_size -= size
                        frame_index += 1
                        found_fit = True
                        break
                
                if not found_fit: # If no packet size fits (e.g. remaining_size < smallest available_size)
                    if available_sizes:
                        smallest_size_str = str(available_sizes[-1])
                        packet_size_entry = link_conf_for_type["packet_sizes"][smallest_size_str]
                        
                        times_data_source = None
                        if direction and direction in packet_size_entry:
                            times_data_source = packet_size_entry[direction]
                        elif "best_case_time" in packet_size_entry:
                            times_data_source = packet_size_entry
                            if direction and direction not in packet_size_entry:
                                 print(f"Warning: Link type {link_type_str}, packet size {smallest_size_str} (smallest): Direction '{direction}' not found. Using direct time fields for {sender_base}->{receiver_base}.")
                        else:
                            raise KeyError(
                                f"Config for link_type {link_type_str}, packet_size {smallest_size_str} (smallest) "
                                f"is missing '{direction}' specific times or direct best_case_time/worst_case_time fields for {sender_base}->{receiver_base}."
                            )

                        times_dict[frame_index] = times_data_source["best_case_time"]
                        deviations_dict[frame_index] = times_data_source["worst_case_time"] - times_dict[frame_index]
                        remaining_size = 0  # Assume the smallest packet handles the rest
                        frame_index += 1
                    else:
                        print(f"Error: No packet sizes defined for link type {link_type_str} or remaining_size {remaining_size} cannot be handled.")
                        break # Avoid infinite loop
        
        return times_dict, deviations_dict

    def _calculate_ns(self, path_data: Dict, current_port: Tuple[str, str], port_index: int) -> int:
        """
        Calculate the NS (Network Segment) value for a path segment.

        Args:
            path_data (Dict): Path data.
            current_port (Tuple[str, str]): Current port tuple.
            port_index (int): Index of the port in the path.

        Returns:
            int: NS value.
        """
        if port_index >= len(path_data["ports"]) - 1:
            return 0  # Last port has NS=0
        
        # Ensure times for the current port exist and are not empty
        times_current = path_data["times"].get(current_port)
        if not times_current: # Checks for None or empty dict
            print(f"Warning: No time data for current_port {current_port} in _calculate_ns. Path: {path_data.get('ports')}. Stream packet_size: {path_data.get('packet_size')}. Returning NS=0.")
            return 0
        
        num_frames = len(times_current)
        if num_frames == 0: # Should be caught by 'if not times_current' but explicit check is fine
            print(f"Warning: num_frames is 0 for current_port {current_port} in _calculate_ns. Path: {path_data.get('ports')}. Returning NS=0.")
            return 0

        # Get the single gamma value for the path. Assumed to exist from _process_path.
        path_gamma = path_data["gamma"]

        deviations_current_dict = path_data["deviations"].get(current_port, {})
        
        # Get time and deviation for the first frame, with defaults if missing (though should exist if num_frames > 0)
        time_first = times_current.get(0)
        dev_first = deviations_current_dict.get(0, 0.0) # Default to 0.0 if deviation for frame 0 is missing

        if time_first is None:
            print(f"Warning: Frame 0 time data missing for current_port {current_port} (num_frames: {num_frames}) in _calculate_ns. Path: {path_data.get('ports')}. Returning NS=0.")
            return 0

        dev_total = sum(deviations_current_dict.values())

        if num_frames > 1:
            # Ensure next_port exists before trying to access its data in _maximize_sum
            if port_index + 1 < len(path_data["ports"]):
                next_port = path_data["ports"][port_index + 1]
                max_sum_value = self._maximize_sum(num_frames, path_data, current_port, next_port)
            else:
                # This case should ideally be prevented by the initial check, but as a safeguard:
                print(f"Warning: Attempting to calculate multi-frame NS but next_port is out of bounds for {current_port} in _calculate_ns. Path: {path_data.get('ports')}. Defaulting max_sum_value to 0.")
                max_sum_value = 0.0 # Or handle as an error, but this prevents crash

            calculated_first_frame_value = time_first + min(path_gamma * dev_total, dev_first)
            
            return math.ceil(max(calculated_first_frame_value, max_sum_value))
        elif num_frames == 1:
            return math.ceil(time_first + path_gamma * dev_total)
        else: 
            # This case (num_frames < 1) should have been caught earlier.
            print(f"Warning: Unexpected num_frames ({num_frames}) at end of _calculate_ns for {current_port}. Path: {path_data.get('ports')}. Returning NS=0.")
            return 0

    def _maximize_sum(self, num_frames: int, path_data: Dict, 
                     current_port: Tuple[str, str], next_port: Tuple[str, str]) -> float:
        """
        Calculate the maximum sum for NS calculation.

        Args:
            num_frames (int): Number of frames.
            path_data (Dict): Path data.
            current_port (Tuple[str, str]): Current port tuple.
            next_port (Tuple[str, str]): Next port tuple.

        Returns:
            float: Maximum sum value.
        """
        max_value = float("-inf")
        
        # Get the single gamma value for the path (used for both current and next port context within this path)
        path_gamma = path_data.get("gamma", 1.0) # Safe get with default
        
        dev_current_all = path_data["deviations"].get(current_port, {})
        # Ensure next_port exists in deviations, default to empty dict if not
        dev_next_all = path_data["deviations"].get(next_port, {})
        
        times_current_all = path_data["times"].get(current_port, {})
        # Ensure next_port exists in times, default to empty dict if not
        times_next_all = path_data["times"].get(next_port, {})
        
        dev_current_sum = sum(dev_current_all.values())
        dev_next_sum = sum(dev_next_all.values())
        
        for i in range(1, num_frames): # Iterates from frame 1 up to num_frames-1
            # Sum of times for current port up to frame i (frames 0 to i)
            sum_times_current = sum(times_current_all.get(k, 0.0) for k in range(i + 1))
            
            # Sum of deviations for current port up to frame i
            sum_dev_current = sum(dev_current_all.get(k, 0.0) for k in range(i + 1))
            
            # Sum of times for next port up to frame i-1 (frames 0 to i-1)
            sum_times_next = sum(times_next_all.get(k, 0.0) for k in range(i))
            
            # Sum of deviations for next port up to frame i-1
            sum_dev_next = sum(dev_next_all.get(k, 0.0) for k in range(i))
            
            # Calculate expression using the single path_gamma
            value = (
                sum_times_current + 
                min(path_gamma * dev_current_sum, sum_dev_current) -
                sum_times_next -
                min(path_gamma * dev_next_sum, sum_dev_next) 
            )
            
            max_value = max(max_value, value)
        
        return max_value if max_value > float("-inf") else 0.0

    def _process_path(self, stream_data: Dict, path_ports: List[Tuple[str, str]]) -> Dict:
        """
        Process a path to generate path data including times, deviations, etc.

        Args:
            stream_data (Dict): Stream data.
            path_ports (List[Tuple[str, str]]): List of port tuples.

        Returns:
            Dict: Path data.
        """
        path_data = {
            "ports": path_ports,
            "period": int(stream_data["period"]),
            "packet_size": int(stream_data["packet_size"]),
            "times": {},
            "deviations": {},
            "gamma": stream_data.get("gamma", 1.0),  # Store single gamma for the path
            "ns": {},
            "link_types": {} 
        }
        
        # Add optional fields if they exist in stream_data
        if "max_latency" in stream_data:
            path_data["max_latency"] = int(stream_data["max_latency"])
        if "priority" in stream_data:
            path_data["priority"] = int(stream_data["priority"])
        if "max_jitter" in stream_data:
            path_data["max_jitter"] = int(stream_data["max_jitter"])
        
        # Calculate times and deviations for each port
        for port_tuple in path_ports:
            try:
                sender_node, receiver_node = port_tuple
                link_type = self._get_link_type(sender_node, receiver_node)
                
                path_data["link_types"][port_tuple] = str(link_type)

                times, deviations = self._calculate_transmission_time(
                    path_data["packet_size"], 
                    link_type,
                    sender_node, 
                    receiver_node 
                )
                
                path_data["times"][port_tuple] = times
                path_data["deviations"][port_tuple] = deviations
                # Gamma is already stored at path_data level, no need for gamma_map per port
            except Exception as e:
                print(f"Error calculating times for port {port_tuple} for stream {stream_data.get('id', 'unknown')}: {e}")
                return {"error": f"Error processing port {port_tuple} for stream {stream_data.get('id', 'unknown')}: {str(e)}"}
        
        # Calculate NS for each port
        for port_index, port_tuple in enumerate(path_ports):
            try:
                if port_index < len(path_ports) - 1:  # Not the last port
                    ns_value = self._calculate_ns(path_data, port_tuple, port_index)
                    path_data["ns"][port_tuple] = ns_value
                else:  # Last port
                    path_data["ns"][port_tuple] = 0
            except Exception as e:
                print(f"Error calculating NS for port {port_tuple}: {e}")
    
        return path_data

    def _calculate_global_lcm(self):
        """
        Calculate the LCM for all periods in the stream dictionary.
        Sets self.lcm to the calculated value.
        """
        periods = []
        for stream_id in self.stream_dict:
            for path_id in self.stream_dict[stream_id]:
                if isinstance(self.stream_dict[stream_id][path_id], dict) and "period" in self.stream_dict[stream_id][path_id]:
                    periods.append(self.stream_dict[stream_id][path_id]["period"])
        
        if periods:
            self.lcm = calculate_lcm(periods)
        else:
            self.lcm = 1  # Default if no periods are found

    def _calculate_repetitions_for_all_streams(self):
        """
        Calculate repetition lists for all streams in the stream dictionary.
        """
        for stream_id in self.stream_dict:
            self._calculate_repetitions_for_stream(stream_id)

    def _calculate_repetitions_for_stream(self, stream_id):
        """
        Calculate repetition list for a specific stream.

        Args:
            stream_id: The ID of the stream to calculate repetitions for.
        """
        if stream_id in self.stream_dict:
            for path_id in self.stream_dict[stream_id]:
                if isinstance(self.stream_dict[stream_id][path_id], dict) and "period" in self.stream_dict[stream_id][path_id]:
                    period = self.stream_dict[stream_id][path_id]["period"]
                    try:
                        last_rep_index, repetitions = calculate_repetitions_list(
                            period, self.lcm, self.lcm_rep
                        )
                        self.stream_dict[stream_id][path_id]["last_rep_index"] = last_rep_index
                        self.stream_dict[stream_id][path_id]["repetitions"] = repetitions
                    except ValueError as e:
                        print(f"Error calculating repetitions for stream {stream_id}, path {path_id}: {e}")
                        self.stream_dict[stream_id][path_id]["last_rep_index"] = 0
                        self.stream_dict[stream_id][path_id]["repetitions"] = []

    def _extract_ports(self):
        """
        Extract all unique ports from the stream dictionary.
        Sets self.ports to the set of all ports.
        """
        self.ports = set()
        for stream_id in self.stream_dict:
            for path_id in self.stream_dict[stream_id]:
                if isinstance(self.stream_dict[stream_id][path_id], dict) and "ports" in self.stream_dict[stream_id][path_id]:
                    self.ports.update(self.stream_dict[stream_id][path_id]["ports"])

    # Methods for accessing stream data
    def get_next_port(self, stream, path, current_port):
        """
        Get the next port in the path for a given stream.

        Args:
            stream: Stream ID.
            path: Path ID.
            current_port: Current port tuple.

        Returns:
            Tuple[str, str]: Next port tuple, or None if current_port is the last port.
        """
        try:
            ports = self.stream_dict[stream][path]["ports"]
            index = ports.index(current_port)
            
            if index < len(ports) - 1:
                return ports[index + 1]
            return None
        except (KeyError, ValueError):
            return None

    def get_previous_port(self, stream, path, current_port):
        """
        Get the previous port in the path for a given stream.

        Args:
            stream: Stream ID.
            path: Path ID.
            current_port: Current port tuple.

        Returns:
            Tuple[str, str]: Previous port tuple, or None if current_port is the first port.
        """
        try:
            ports = self.stream_dict[stream][path]["ports"]
            index = ports.index(current_port)
            
            if index > 0:
                return ports[index - 1]
            return None
        except (KeyError, ValueError):
            return None

    def get_streams_for_port(self, port):
        """
        Get all streams that use a given port.

        Args:
            port: Port tuple.

        Returns:
            List: List of stream IDs that use the port.
        """
        streams_with_port = []
        for stream in self.stream_dict:
            for path in self.stream_dict[stream]:
                if isinstance(self.stream_dict[stream][path], dict) and "ports" in self.stream_dict[stream][path]:
                    if port in self.stream_dict[stream][path]["ports"]:
                        streams_with_port.append(stream)
                        break
        return streams_with_port

    def get_paths_for_port_stream(self, stream, port):
        """
        Get all paths of a stream that use a given port.

        Args:
            stream: Stream ID.
            port: Port tuple.

        Returns:
            List: List of path IDs.
        """
        paths_with_port = []
        if stream in self.stream_dict:
            for path in self.stream_dict[stream]:
                if isinstance(self.stream_dict[stream][path], dict) and "ports" in self.stream_dict[stream][path]:
                    if port in self.stream_dict[stream][path]["ports"]:
                        paths_with_port.append(path)
        return paths_with_port

    def get_time_for_stream_port(self, stream, path, port):
        """
        Get the total time for a specific port in a stream path.

        Args:
            stream: Stream ID.
            path: Path ID.
            port: Port tuple.

        Returns:
            float: Sum of times for the port.
        """
        try:
            path_data = self.stream_dict[stream][path]
            if "times" in path_data and port in path_data["times"]:
                return sum(path_data["times"][port].values())
            return 0
        except KeyError:
            return 0

    def get_deviation_for_stream_port(self, stream, path, port):
        """
        Get the total deviation for a specific port in a stream path.

        Args:
            stream: Stream ID.
            path: Path ID.
            port: Port tuple.

        Returns:
            float: Sum of deviations for the port.
        """
        try:
            path_data = self.stream_dict[stream][path]
            if "deviations" in path_data and port in path_data["deviations"]:
                return sum(path_data["deviations"][port].values())
            return 0
        except KeyError:
            return 0

    def get_repetitions_for_stream(self, stream):
        """
        Get repetitions for a stream (first path).

        Args:
            stream: Stream ID.

        Returns:
            List: List of repetition indices, or None if stream not found.
        """
        if stream in self.stream_dict and 0 in self.stream_dict[stream]:
            path_data = self.stream_dict[stream][0]
            if "repetitions" in path_data:
                return path_data["repetitions"]
        return None
        
    def get_repetitions_for_stream_path(self, stream, path):
        """
        Get repetitions for a specific stream path.

        Args:
            stream: Stream ID.
            path: Path ID.

        Returns:
            List: List of repetition indices, or empty list if path not found.
        """
        try:
            return self.stream_dict[stream][path]["repetitions"]
        except KeyError:
            return []

    def get_gamma_for_stream_path(self, stream: int, path: int) -> float:
        """
        Get the gamma value for a specific stream and path.

        Args:
            stream: Stream ID.
            path: Path ID.

        Returns:
            float: Gamma value. Raises KeyError if not found.
        """
        return self.stream_dict[stream][path]["gamma"]
