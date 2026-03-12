import pandas as pd
import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from math import gcd
from functools import reduce
from collections import defaultdict
from matplotlib.patches import Rectangle  # Import Rectangle for legend
from utils.path_manager import PathManager
from utils import helper
import yaml
import warnings
import traceback 

# Suppress specific warnings if needed (e.g., Matplotlib future warnings)
warnings.filterwarnings("ignore", category=FutureWarning)


class OutputManager:
    """
    Manages loading of simulation/optimization results, processing them,
    and generating scheduling chart output.
    """

    # Use type hints for better readability and maintainability
    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
        self.variables_x: dict | None = None  # {(stream, path, rep, port): open_time}
        self.variables_z: dict | None = (
            None  # {(stream, path): 1 if scheduled, 0 otherwise}
        )
        self.variables_a: dict | None = (
            None  # {(stream,): 1 if scheduled, 0 otherwise} - Note the key format
        )
        self.stream_dict: dict | None = None  # Detailed stream path information
        self.graph_nx: nx.Graph | None = None  # Network topology
        self.streams_df: pd.DataFrame | None = (
            None  # Original stream definitions (period, size, etc.)
        )
        self.histogram_config: dict | None = (
            None  # Configuration for miss probability calculation
        )

        # Derived data
        self.activated_streams_list: list[tuple] | None = (
            None  # List of (stream_id, path_id) tuples for scheduled streams
        )
        self.ports_set: set[tuple] | None = (
            None  # Set of unique (sender, receiver) ports used by activated streams
        )
        self.all_nodes: list[str] | None = None  # Sorted list of unique node names
        self.lcm_period: int | None = None  # Least Common Multiple of stream periods

        # Centralize output directory creation
        self.processed_results_dir = self.path_manager.processed_results_dir

        try:
            os.makedirs(self.processed_results_dir, exist_ok=True)
            print(
                f"Ensured processed results directory exists: {self.processed_results_dir}"
            )
        except OSError as e:
            print(
                f"ERROR: Could not create processed results directory {self.processed_results_dir}: {e}"
            )

    def load_files(self):
        """Loads all necessary input and output files from the simulation/optimization."""
        print("--- Loading Input Files ---")

        # Load primary results (pickle files)
        self.stream_dict = helper.load_pickle(self.path_manager.stream_dict_output_dir)
        self.variables_x = helper.load_pickle(self.path_manager.variables_x_dir)
        self.variables_z = helper.load_pickle(self.path_manager.variables_z_dir)
        self.variables_a = helper.load_pickle(self.path_manager.variables_a_dir)

        # Load topology graph
        try:
            graph_path = self.path_manager.network_graphml_output_dir
            self.graph_nx = nx.read_graphml(graph_path)
            print(f"Loaded graph_nx (GraphML) successfully from {graph_path}.")
        except Exception as e:
            print(f"Error loading topology graph (GraphML): {e}")
            self.graph_nx = None

        try:
            streams_output_path = self.path_manager.streams_output_dir
            print(f"Attempting to load streams (CSV) from: {streams_output_path}")
            # Assuming stream ID is the first column or index
            self.streams_df = pd.read_csv(streams_output_path, index_col=0)
            print(f"Loaded streams (CSV) successfully.")
        except FileNotFoundError:
            print(f"ERROR: Streams CSV not found: {streams_output_path}")
            self.streams_df = None
        except Exception as e:
            print(f"Error loading streams (CSV) from {streams_output_path}: {e}")
            self.streams_df = None

        # Load histogram config (try output path first, then input path)
        self.histogram_config = None

        if self.path_manager.histogram_config_output_dir:
            print(
                f"Attempting to load histogram config from output path: {self.path_manager.histogram_config_output_dir}"
            )
            self.histogram_config = helper.load_yaml_file(
                self.path_manager.histogram_config_output_dir
            )

        if self.path_manager.histogram_config_input_dir:
            print(
                f"Attempting to load histogram config from input path: {self.path_manager.histogram_config_input_dir}"
            )
            self.histogram_config = helper.load_yaml_file(
                self.path_manager.histogram_config_input_dir
            )

        print("--- File Loading Complete ---")

    def prepare_common_data(self) -> bool:
        """
        Prepares data structures derived from loaded files, used by multiple methods.
        Calculates LCM, identifies activated streams, ports, and nodes.

        Returns:
            bool: True if preparation was successful, False otherwise.
        """
        print("--- Preparing Common Data ---")
        # Check prerequisites
        if self.variables_z is None:
            print(
                "ERROR: variables_z is not loaded. Cannot identify activated streams."
            )
            return False
        if self.stream_dict is None:
            print("ERROR: stream_dict is not loaded. Cannot get path information.")
            return False
        if self.streams_df is None:
            print("ERROR: streams_df is not loaded. Cannot calculate LCM period.")
            return False
        if "period" not in self.streams_df.columns:
            print("ERROR: 'period' column missing in streams_df. Cannot calculate LCM.")
            return False

        # Get activated streams (where Z variable is 1)
        try:
            self.activated_streams_list = [
                stream_path
                for stream_path, value in self.variables_z.items()
                if isinstance(stream_path, tuple)
                and len(stream_path) == 2
                and (value == 1.0 or value == 1)
            ]
            if not self.activated_streams_list:
                print(
                    "WARNING: No scheduled streams found (variables_z has no entries with value 1). Processing will continue with empty results."
                )
                self.ports_set = set()
                self.all_nodes = []
            else:
                print(
                    f"Identified {len(self.activated_streams_list)} activated stream paths."
                )
        except Exception as e:
            print(
                f"ERROR: Failed to process variables_z to find activated streams: {e}"
            )
            return False

        # Get unique ports and nodes involved in activated streams
        self.ports_set = set()
        all_nodes_temp = set()
        for stream, path in self.activated_streams_list:
            try:
                path_info = self.stream_dict.get(stream, {}).get(path, {})
                current_ports = path_info.get("ports", [])
                if not current_ports:
                    print(
                        f"WARNING: No ports found in stream_dict for activated stream/path ({stream}, {path})."
                    )
                    continue
                self.ports_set.update(current_ports)
                for port in current_ports:
                    if isinstance(port, tuple) and len(port) == 2:
                        all_nodes_temp.add(port[0])  # Sender
                        all_nodes_temp.add(port[1])  # Receiver
                    else:
                        print(
                            f"WARNING: Invalid port format {port} found for stream {stream}, path {path}."
                        )
            except Exception as e:
                print(
                    f"ERROR: Failed to process ports for stream {stream}, path {path}: {e}"
                )
                # Continue processing other streams if possible

        self.all_nodes = sorted(list(all_nodes_temp))
        print(
            f"Identified {len(self.ports_set)} unique active ports and {len(self.all_nodes)} unique active nodes."
        )

        # Calculate LCM period from the original streams_df
        try:
            # Ensure periods are integers and handle potential NaNs or non-numeric values
            periods = (
                pd.to_numeric(self.streams_df["period"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
            if not periods:
                print(
                    "WARNING: No valid periods found in streams_df to calculate LCM. Defaulting LCM to 1."
                )
                self.lcm_period = 1
            else:
                self.lcm_period = self._calculate_lcm_from_list(periods)
                print(f"Calculated LCM period: {self.lcm_period}")
                if self.lcm_period <= 0:
                    print(
                        "ERROR: Calculated LCM period is zero or negative. Defaulting to 1."
                    )
                    self.lcm_period = 1
        except Exception as e:
            print(f"ERROR: Failed to calculate LCM period: {e}")
            return False

        print("--- Common Data Preparation Complete ---")
        return True

    @staticmethod
    def _calculate_lcm_from_list(numbers: list[int]) -> int:
        """Calculates the least common multiple (LCM) of a list of positive integers."""
        if not numbers:
            return 1  # LCM of empty set is 1

        # Filter out non-positive numbers and ensure integer type
        valid_numbers = [
            n for n in numbers if isinstance(n, (int, np.integer)) and n > 0
        ]
        if not valid_numbers:
            return 1  # Only zeros, negative numbers, or empty list provided

        def _lcm(a, b):
            # gcd requires non-zero inputs
            if a == 0 or b == 0:
                return 0
            return abs(a * b) // gcd(a, b)

        result = valid_numbers[0]
        for i in range(1, len(valid_numbers)):
            result = _lcm(result, valid_numbers[i])
            if result == 0:  # Should not happen with positive inputs, but as safeguard
                print("WARNING: LCM calculation resulted in 0 unexpectedly.")
                return 1

        return result if result > 0 else 1

    def _calculate_single_link_miss_prob(
        self, link_type: str, packet_size: int, duration_scheduled: float, sender: str, receiver: str
    ) -> float:
        """
        Calculates the probability that the actual transmission time on a link
        exceeds the scheduled duration, based on histogram data.
        Considers uplink/downlink for wireless links if specified in histogram_config.
        """
        print(f"\n--- DEBUG: _calculate_single_link_miss_prob ---")
        print(f"    Args: link_type='{link_type}', packet_size={packet_size}, duration_scheduled={duration_scheduled:.4f}, sender='{sender}', receiver='{receiver}'")

        if self.histogram_config is None:
            print(f"    DEBUG_MISS_PROB: self.histogram_config is None. Returning 0.0 miss probability.")
            return 0.0

        packet_size_str = str(packet_size)
        print(f"    DEBUG_MISS_PROB: packet_size_str='{packet_size_str}'")

        try:
            link_type_level_config = self.histogram_config['link_types'][link_type]
            if not link_type_level_config:
                print(f"    DEBUG_MISS_PROB: No histogram config found for link_type='{link_type}'. Returning 0.0 miss probability.")
                return 0.0
            print(f"    DEBUG_MISS_PROB: Found link_type_level_config for '{link_type}'. Keys: {list(link_type_level_config.keys())}")

            packet_size_level_data = link_type_level_config.get("packet_sizes", {}).get(packet_size_str, {})
            if not packet_size_level_data:
                print(f"    DEBUG_MISS_PROB: No histogram config for link_type='{link_type}', packet_size='{packet_size_str}'. Returning 0.0 miss probability.")
                return 0.0
            print(f"    DEBUG_MISS_PROB: Found packet_size_level_data for '{packet_size_str}'. Keys: {list(packet_size_level_data.keys())}")

            # Determine direction for wireless links
            sender_base = sender.split('#')[0]
            receiver_base = receiver.split('#')[0]
            direction = None
            if link_type != "0":
                if "WED" in sender_base and "WN" in receiver_base:
                    direction = "uplink"
                elif "WN" in sender_base and "WED" in receiver_base:
                    direction = "downlink"
            print(f"    DEBUG_MISS_PROB: Determined direction: {direction} (sender_base='{sender_base}', receiver_base='{receiver_base}')")

            data_source_for_hist = packet_size_level_data
            if direction and direction in packet_size_level_data:
                data_source_for_hist = packet_size_level_data[direction]
                print(f"    DEBUG_MISS_PROB: Using directional data_source_for_hist for '{direction}'. Keys: {list(data_source_for_hist.keys())}")
            elif direction:
                print(f"    DEBUG_MISS_PROB: Direction '{direction}' determined but not found in packet_size_level_data. Will use non-directional if available.")
                # Fallback to non-directional if direct keys exist
            else:
                print(f"    DEBUG_MISS_PROB: No direction determined or not applicable. Using packet_size_level_data as data_source_for_hist. Keys: {list(data_source_for_hist.keys())}")


            histogram_values = data_source_for_hist.get("histogram_data")
            bin_boundaries = data_source_for_hist.get("bin_boundaries")

            if histogram_values is None or bin_boundaries is None:
                print(f"    DEBUG_MISS_PROB: Missing 'histogram_data' or 'bin_boundaries' in data_source_for_hist (keys: {list(data_source_for_hist.keys())}). Returning 0.0 miss probability.")
                return 0.0
            print(f"    DEBUG_MISS_PROB: histogram_values (len {len(histogram_values)}): {histogram_values}")
            print(f"    DEBUG_MISS_PROB: bin_boundaries (len {len(bin_boundaries)}): {bin_boundaries}")
            
            if not isinstance(histogram_values, list) or not isinstance(bin_boundaries, list) or len(histogram_values) != len(bin_boundaries):
                print(f"    WARNING_MISS_PROB: Malformed histogram data. Lengths: hist={len(histogram_values if isinstance(histogram_values, list) else [])}, bins={len(bin_boundaries if isinstance(bin_boundaries, list) else [])}. Returning 0.0 miss probability.")
                return 0.0

            if not histogram_values:
                print(f"    DEBUG_MISS_PROB: histogram_values is empty. Returning 0.0 miss probability.")
                return 0.0

            total_frequency = sum(histogram_values)
            if total_frequency == 0:
                print(f"    DEBUG_MISS_PROB: total_frequency is 0. Returning 0.0 miss probability.")
                return 0.0
            print(f"    DEBUG_MISS_PROB: total_frequency: {total_frequency}")

            miss_frequency = 0
            for i, (freq, actual_duration) in enumerate(zip(histogram_values, bin_boundaries)):
                # print(f"      DEBUG_MISS_PROB: Bin {i}: freq={freq}, actual_duration={actual_duration}, duration_scheduled={duration_scheduled:.4f}")
                if actual_duration > duration_scheduled + 1e-9: # Epsilon for float comparison
                    miss_frequency += freq
                    # print(f"        DEBUG_MISS_PROB: Bin {i} MISSED. current miss_frequency={miss_frequency}")
            
            miss_probability = miss_frequency / total_frequency
            print(f"    DEBUG_MISS_PROB: miss_frequency: {miss_frequency}, miss_probability: {miss_probability:.4f}")
            print(f"--- END DEBUG: _calculate_single_link_miss_prob ---\n")
            return miss_probability

        except Exception as e:
            print(
                f"    ERROR_MISS_PROB: Unexpected error for link_type='{link_type}', packet_size='{packet_size_str}', sender='{sender}', receiver='{receiver}': {e}"
            )
            traceback.print_exc()
            print(f"--- END DEBUG: _calculate_single_link_miss_prob (EXCEPTION) ---\n")
            return 0.0

    def calculate_success_probabilities(self) -> dict:
        """
        Calculates the end-to-end success probability for each scheduled stream.
        This considers the miss probability of each link in the stream's path.
        """
        print(f"\n--- DEBUG: calculate_success_probabilities ---")
        if (
            not self.activated_streams_list
            or not self.stream_dict
            or self.streams_df is None
        ):
            print(
                "    DEBUG_SUCCESS_PROB: Missing data (activated_streams_list, stream_dict, or streams_df). Returning empty dict."
            )
            return {}
        
        if self.histogram_config is None:
            print("    DEBUG_SUCCESS_PROB: self.histogram_config is None. Individual link miss probs will be 0. Returning empty dict for now (will default to 1.0 later).")
            # If histogram_config is None, _calculate_single_link_miss_prob will return 0, leading to 1.0 success.
            # To make it explicit that we can't calculate, we could return {} and let update_streams_csv handle default.
            # For now, let it proceed, _calculate_single_link_miss_prob will handle the None case.
            # return {} # This would cause it to fill with 1.0 later.

        stream_success_probabilities = {}
        print(f"    DEBUG_SUCCESS_PROB: Processing {len(self.activated_streams_list)} activated streams.")

        for stream_id, path_id in self.activated_streams_list:
            print(f"\n    DEBUG_SUCCESS_PROB: Stream ID: {stream_id}, Path ID: {path_id}")
            try:
                path_info = self.stream_dict.get(stream_id, {}).get(path_id, {})
                if not path_info:
                    print(f"        DEBUG_SUCCESS_PROB: No path_info found for stream {stream_id}, path {path_id}. Skipping.")
                    continue

                stream_ports = path_info.get("ports", [])
                stream_times = path_info.get("times", {})
                stream_devs = path_info.get("deviations", {})
                stream_gamma = path_info.get("gamma", 0.0) # Get single gamma for the path
                
                # Get packet_size for the stream from streams_df
                # Assuming stream_id is the index or can be used to look up in streams_df
                try:
                    # Check if stream_id is a direct index
                    if stream_id in self.streams_df.index:
                         packet_size = int(self.streams_df.loc[stream_id, "packet_size"])
                    # Fallback: if stream_id is an integer that might correspond to iloc
                    elif isinstance(stream_id, int) and stream_id < len(self.streams_df):
                        packet_size = int(self.streams_df.iloc[stream_id]["packet_size"])
                    else:
                        print(f"        WARNING_SUCCESS_PROB: Could not determine packet_size for stream {stream_id} from streams_df. Skipping stream.")
                        continue
                except (KeyError, IndexError, ValueError) as e:
                    print(f"        WARNING_SUCCESS_PROB: Error getting packet_size for stream {stream_id}: {e}. Skipping stream.")
                    continue
                
                print(f"        DEBUG_SUCCESS_PROB: Packet Size: {packet_size}")
                print(f"        DEBUG_SUCCESS_PROB: Stream Ports: {stream_ports}")

                if not stream_ports:
                    print(f"        DEBUG_SUCCESS_PROB: No ports for stream {stream_id}, path {path_id}. Assuming 1.0 success prob for stream.")
                    stream_success_probabilities[stream_id] = 1.0
                    continue

                link_success_probabilities = []

                for port_idx, port in enumerate(stream_ports):
                    sender, receiver = port
                    link_type_from_graph = str(path_info["link_types"][port])
                    deviation_on_link = sum(stream_devs.get(port, {}).values()) # Sum of deviations for all frames on this link

                    print(f"            DEBUG_SUCCESS_PROB: Port {port_idx+1}/{len(stream_ports)}: {sender}->{receiver}, LinkType: {link_type_from_graph}, DeviationOnLink: {deviation_on_link:.4f}")

                    if link_type_from_graph != "0" and deviation_on_link > 1e-9 : # Use epsilon for float comparison
                        base_time_on_link = sum(stream_times.get(port, {}).values())
                        # Use the single gamma for the path
                        
                        print(f"                DEBUG_SUCCESS_PROB: Wireless/Variable Link. BaseTime: {base_time_on_link:.4f}, Gamma (for path): {stream_gamma:.4f}")

                        prob_miss_this_link = 0.0
                        if stream_gamma >= 1.0 - 1e-9: # Epsilon for float comparison
                            print(f"                DEBUG_SUCCESS_PROB: Gamma >= 1.0. Miss probability = 0.0")
                            prob_miss_this_link = 0.0
                        elif stream_gamma < 0:
                            print(f"                DEBUG_SUCCESS_PROB: Gamma < 0. Using base_time for scheduled duration.")
                            duration_scheduled = base_time_on_link
                            prob_miss_this_link = self._calculate_single_link_miss_prob(
                                link_type_from_graph, packet_size, duration_scheduled, sender, receiver
                            )
                        else:
                            duration_scheduled = base_time_on_link + (deviation_on_link * stream_gamma)
                            print(f"                DEBUG_SUCCESS_PROB: Calculating miss prob. DurationScheduled: {duration_scheduled:.4f}")
                            prob_miss_this_link = self._calculate_single_link_miss_prob(
                                link_type_from_graph, packet_size, duration_scheduled, sender, receiver
                            )
                        
                        current_link_success_prob = 1.0 - prob_miss_this_link
                        link_success_probabilities.append(current_link_success_prob)
                        print(f"                DEBUG_SUCCESS_PROB: MissProbThisLink: {prob_miss_this_link:.4f}, SuccessProbThisLink: {current_link_success_prob:.4f}")
                    else:
                        print(f"            DEBUG_SUCCESS_PROB: Wired link or zero deviation. Assuming 1.0 success for this link.")
                        link_success_probabilities.append(1.0) # Wired or no deviation link

                overall_success_prob = 1.0
                if link_success_probabilities: # If list is not empty
                    for prob in link_success_probabilities:
                        overall_success_prob *= prob
                
                stream_success_probabilities[stream_id] = overall_success_prob
                print(f"        DEBUG_SUCCESS_PROB: All link success probs for stream {stream_id}: {link_success_probabilities}")
                print(f"        DEBUG_SUCCESS_PROB: Overall success probability for stream {stream_id}: {overall_success_prob:.4f}")

            except Exception as e:
                print(f"        ERROR_SUCCESS_PROB: Error processing stream {stream_id}, path {path_id}: {e}")
                traceback.print_exc()
                stream_success_probabilities[stream_id] = 0.0 # Default to 0 on error for safety

        print(f"--- END DEBUG: calculate_success_probabilities ---\n")
        return stream_success_probabilities

    def _add_text_label(
        self, ax, stream_id, start, duration, y_pos, num_ports, is_blocked=False
    ):
        """Helper method to add text labels (stream ID) to bars on scheduling charts."""
        # Only add label if the bar is visually wide enough (adjust threshold as needed)
        # Threshold relative to LCM period might be better? For now, absolute.
        label_threshold = max(0.5, self.lcm_period / 500 if self.lcm_period else 1.0)
        if duration > label_threshold:
            label = str(stream_id) if stream_id != -1 else "?"  # Handle placeholder ID

            # Dynamic font size calculation
            # Base size scales inversely with number of ports, capped for readability
            base_size = 60 / max(10, num_ports)  # Avoid division by zero
            # Scale with duration, but cap min/max size
            dynamic_fontsize = max(
                1.0, min(duration * 0.8, base_size, 7)
            )  # Adjusted caps

            # Use slightly lower alpha for blocked periods or very short durations
            alpha = min(
                0.7 if is_blocked else 1.0, duration / label_threshold, 1.0
            )  # Cap alpha at 1

            try:
                ax.text(
                    start + duration / 2,  # Center horizontally
                    y_pos,  # Center vertically within the bar's y-range
                    label,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=dynamic_fontsize,
                    rotation=0,  # Keep horizontal
                    clip_on=True,  # Prevent text spilling outside axes boundaries
                    alpha=alpha,
                )
            except Exception as e:
                print(
                    f"Warning: Failed to add text label for stream {stream_id} at ({start}, {y_pos}): {e}"
                )

    def draw_scheduling_chart(self, highlight_stream_index=None):
        """
        Draws the UNWRAPPED scheduling chart over multiple LCM periods (e.g., 3).
        Shows transmission windows and blocked periods.
        """
        print("--- Generating Unwrapped Scheduling Chart ---")
        # Check prerequisites
        if (
            not self.activated_streams_list
            or not self.ports_set
            or self.lcm_period is None
            or self.variables_x is None
            or self.stream_dict is None
        ):
            print("WARNING: Missing data for unwrapped scheduling chart. Skipping.")
            return

        # Sort ports for consistent Y-axis ordering
        sorted_ports = sorted(list(self.ports_set))
        port_to_y = {port: i for i, port in enumerate(sorted_ports)}
        num_ports = len(sorted_ports)
        if num_ports == 0:
            print("WARNING: No active ports found. Skipping unwrapped chart.")
            return

        # Determine plot range (e.g., 3 LCM periods)
        num_lcm_periods_to_plot = 3
        plot_end_time = self.lcm_period * num_lcm_periods_to_plot

        # --- Prepare data for plotting ---
        plot_data = []  # Store tuples: (y_pos, start, duration, stream_id, is_blocked)

        for stream, path in self.activated_streams_list:
            try:
                path_info = self.stream_dict.get(stream, {}).get(path, {})
                if not path_info:
                    continue

                stream_ports = path_info.get("ports", [])
                stream_times = path_info.get("times", {})
                stream_devs = path_info.get("deviations", {})
                stream_gamma = path_info.get("gamma", 0.0)  # Get single gamma for the path
                stream_bp = path_info.get("bp", {})  # Blocked period per PREVIOUS port

                # Iterate through repetitions relevant to the plot duration
                # Note: This assumes repetitions in stream_dict cover the required range
                for repetition in path_info.get("repetitions", []):
                    # Estimate if this repetition could start within the plot window
                    # This is an approximation; a more precise check might be needed if periods vary wildly
                    approx_rep_start = repetition * self.stream_dict.get(
                        stream, {}
                    ).get("period", self.lcm_period)
                    if approx_rep_start > plot_end_time:
                        continue  # Optimization: Skip repetitions clearly outside the plot range

                    for port_idx, port in enumerate(stream_ports):
                        if port not in self.ports_set:
                            continue  # Skip if port wasn't activated/included

                        y_pos = port_to_y[port]
                        open_time_key = (stream, path, repetition, port)

                        if open_time_key not in self.variables_x:
                            # print(f"WARNING: Missing timing data (variables_x) for {open_time_key}")
                            continue  # Skip this port/repetition if timing is missing

                        open_time = self.variables_x[open_time_key]

                        # Skip if the window starts after the plot ends
                        if open_time >= plot_end_time:
                            continue

                        # Calculate transmission duration (base + gamma * deviation)
                        time = sum(stream_times.get(port, {}).values())
                        deviation = sum(stream_devs.get(port, {}).values())
                        # Use the single path gamma
                        duration = time + (deviation * stream_gamma)
                        close_time = open_time + duration

                        # Add transmission window if it overlaps with the plot range
                        if duration > 0:
                            plot_start = max(
                                0, open_time
                            )  # Ensure start is not negative
                            plot_duration = min(
                                duration, plot_end_time - plot_start
                            )  # Clip duration at plot end
                            if plot_duration > 1e-9:  # Use epsilon for float comparison
                                plot_data.append(
                                    (y_pos, plot_start, plot_duration, stream, False)
                                )  # False = not blocked

                        # Calculate and add blocked period (applies to current port, caused by previous)
                        if (
                            port_idx > 0
                        ):  # Blocked period depends on the *previous* port in the path
                            previous_port = stream_ports[port_idx - 1]
                            # BP duration is associated with the previous port in stream_dict
                            blocked_period_duration = stream_bp.get(previous_port, 0)

                            if blocked_period_duration > 0:
                                bp_start_abs = (
                                    open_time - blocked_period_duration
                                )  # BP ends when current transmission starts
                                # Add blocked period if it overlaps with the plot range
                                if bp_start_abs < plot_end_time:
                                    plot_bp_start = max(0, bp_start_abs)
                                    # Duration is clipped by plot end AND by the start of the transmission window
                                    plot_bp_duration = min(
                                        blocked_period_duration,
                                        plot_end_time - plot_bp_start,
                                        open_time - plot_bp_start,
                                    )
                                    if plot_bp_duration > 1e-9:
                                        plot_data.append(
                                            (
                                                y_pos,
                                                plot_bp_start,
                                                plot_bp_duration,
                                                stream,
                                                True,
                                            )
                                        )  # True = blocked

            except Exception as e:
                print(
                    f"ERROR: Failed processing stream {stream}, path {path} for unwrapped chart: {e}"
                )

        # --- Create the plot ---
        if not plot_data:
            print(
                "WARNING: No scheduling data found within the plot range. Creating empty chart."
            )
            fig, ax = plt.subplots(
                figsize=(12, max(6, num_ports * 0.3))
            )  # Basic empty plot
            ax.set_yticks(range(num_ports))
            ax.set_yticklabels(
                sorted_ports, fontsize=max(2, min(6, 60 / max(10, num_ports)))
            )
            ax.set_xlabel("Time", fontsize=8)
            ax.set_ylabel("Port Gates", fontsize=8)
            ax.set_title("Unwrapped Scheduling Windows (No Data)", fontsize=10)
            ax.set_xlim(0, plot_end_time)
            ax.invert_yaxis()
            plt.tight_layout()
        else:
            # Adaptive sizing
            ytick_fontsize = max(2, min(6, 60 / max(10, num_ports)))
            label_fontsize = max(4, min(8, 80 / max(10, num_ports)))
            title_fontsize = max(6, min(10, 100 / max(10, num_ports)))
            legend_fontsize = max(3, min(6, 60 / max(10, num_ports)))
            fig_height = max(
                6, min(25, num_ports * 0.35)
            )  # Adjusted height factor, capped max height
            fig, ax = plt.subplots(figsize=(15, fig_height))  # Wider figure

            # Plot bars
            for y_pos, start, duration, stream_id, is_blocked in plot_data:
                # Determine color based on blocked status and highlight
                if is_blocked:
                    color = "lightcoral"
                elif stream_id == highlight_stream_index:
                    color = "blue"
                else:
                    color = "skyblue"

                ax.broken_barh(
                    [(start, duration)],
                    (y_pos - 0.4, 0.8),  # Y position and height of bars
                    facecolors=color,
                    edgecolor="gray",  # Add subtle edge
                    linewidth=0.2,
                )
                # Add text label inside the bar
                self._add_text_label(
                    ax, stream_id, start, duration, y_pos, num_ports, is_blocked
                )

            # Draw LCM period marker lines
            for lcm_multiple in range(
                self.lcm_period, int(plot_end_time) + 1, self.lcm_period
            ):
                ax.axvline(
                    x=lcm_multiple,
                    color="gray",
                    linestyle="--",
                    linewidth=0.7,
                    alpha=0.8,
                )

            # Configure axes and labels
            ax.set_xlim(0, plot_end_time)
            ax.set_ylim(-0.5, num_ports - 0.5)
            ax.set_xlabel("Time", fontsize=label_fontsize)
            ax.set_ylabel("Port Gates", fontsize=label_fontsize)
            ax.set_yticks(range(num_ports))
            ax.set_yticklabels(sorted_ports, fontsize=ytick_fontsize)
            ax.set_title(
                "Unwrapped Scheduling Windows for Port Gates", fontsize=title_fontsize
            )
            ax.invert_yaxis()  # Ports often shown top-down

            # Create Legend
            handles = [
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    fc="skyblue",
                    ec="gray",
                    lw=0.2,
                    label="Transmission Window",
                ),
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    fc="lightcoral",
                    ec="gray",
                    lw=0.2,
                    label="Blocked Period",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    color="gray",
                    linestyle="--",
                    lw=0.7,
                    label=f"LCM Period Marker ({self.lcm_period})",
                ),
            ]
            if highlight_stream_index is not None:
                handles.insert(
                    1,
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        fc="blue",
                        ec="gray",
                        lw=0.2,
                        label=f"Stream {highlight_stream_index}",
                    ),
                )

            ax.legend(handles=handles, loc="upper right", fontsize=legend_fontsize)
            plt.tight_layout(
                rect=[0, 0.03, 1, 0.97]
            )  # Adjust layout to prevent title overlap

        # Save the plot
        chart_path = os.path.join(
            self.processed_results_dir, "port_schedules_unwrapped.png"
        )
        try:
            plt.savefig(chart_path, format="png", dpi=400)  # Adjusted DPI
            print(f"Unwrapped scheduling chart saved as {chart_path}")
        except Exception as e:
            print(f"ERROR: Failed to save unwrapped scheduling chart: {e}")
        finally:
            plt.close(fig)  # Ensure figure is closed

        print("--- Unwrapped Scheduling Chart Generation Complete ---")

    def _get_wrapped_port_intervals(
        self, port_filter=None
    ) -> dict[tuple, list[tuple[str, float, float, int]]]:
        """
        Calculates 'used' and 'blocked' intervals for each port, wrapped within one LCM period.
        Merges overlapping intervals.

        Args:
            port_filter (optional): A specific port tuple (sender, receiver) to calculate for.
                                     If None, calculates for all activated ports.

        Returns:
            A dictionary where keys are port tuples (sender, receiver) and
            values are lists of merged interval tuples:
            [(status, start, end, representative_stream_id), ...].
            'status' is 'used' or 'blocked'. 'start' and 'end' are times within [0, lcm_period).
            'representative_stream_id' is one of the stream IDs causing the interval (-1 if unknown).
        """
        # Check prerequisites
        if (
            self.lcm_period is None
            or self.lcm_period <= 0
            or self.variables_x is None
            or self.stream_dict is None
            or self.activated_streams_list is None
            or self.ports_set is None
        ):
            print(
                "ERROR: Missing data needed for interval calculation (_get_wrapped_port_intervals)."
            )
            return {}

        raw_intervals = defaultdict(
            list
        )  # Store (status, start, end, stream_id) before merging
        ports_to_process = {port_filter} if port_filter else self.ports_set

        for stream, path in self.activated_streams_list:
            try:
                path_info = self.stream_dict.get(stream, {}).get(path, {})
                if not path_info:
                    continue

                stream_ports = path_info.get("ports", [])
                stream_times = path_info.get("times", {})
                stream_devs = path_info.get("deviations", {})
                stream_gamma = path_info.get("gamma", 0.0)  # Get single gamma for the path
                stream_bp = path_info.get("bp", {})

                for repetition in path_info.get("repetitions", []):
                    for port_idx, port in enumerate(stream_ports):
                        if port not in ports_to_process:
                            continue

                        open_time_key = (stream, path, repetition, port)
                        if open_time_key not in self.variables_x:
                            continue

                        open_time = self.variables_x[open_time_key]

                        # Calculate transmission duration
                        time = sum(stream_times.get(port, {}).values())
                        deviation = sum(stream_devs.get(port, {}).values())
                        # Use the single path gamma
                        duration = time + (deviation * stream_gamma)

                        if duration <= 1e-9:
                            continue  # Skip negligible/zero duration

                        # --- Add transmission interval(s) with wrapping ---
                        norm_start = open_time % self.lcm_period
                        norm_end = (open_time + duration) % self.lcm_period

                        if (
                            norm_start < norm_end or abs(norm_start - norm_end) < 1e-9
                        ):  # Doesn't wrap (or exactly LCM period)
                            raw_intervals[port].append(
                                (
                                    "used",
                                    norm_start,
                                    (
                                        norm_end
                                        if norm_end > norm_start
                                        else self.lcm_period
                                    ),
                                    stream,
                                )
                            )
                        elif duration < self.lcm_period:  # Wraps around LCM boundary
                            raw_intervals[port].append(
                                ("used", norm_start, self.lcm_period, stream)
                            )
                            raw_intervals[port].append(("used", 0.0, norm_end, stream))
                        else:  # Duration is >= LCM period, covers the whole period
                            raw_intervals[port].append(
                                ("used", 0.0, self.lcm_period, stream)
                            )

                        # --- Calculate and add blocked period interval(s) with wrapping ---
                        if port_idx > 0:
                            previous_port = stream_ports[port_idx - 1]
                            bp_duration = stream_bp.get(previous_port, 0)

                            if bp_duration > 1e-9:
                                bp_start_abs = open_time - bp_duration
                                bp_norm_start = bp_start_abs % self.lcm_period
                                bp_norm_end = (
                                    open_time % self.lcm_period
                                )  # BP ends when transmission starts

                                if (
                                    bp_norm_start < bp_norm_end
                                    or abs(bp_norm_start - bp_norm_end) < 1e-9
                                ):  # Doesn't wrap
                                    raw_intervals[port].append(
                                        ("blocked", bp_norm_start, bp_norm_end, stream)
                                    )
                                elif bp_duration < self.lcm_period:  # Wraps
                                    raw_intervals[port].append(
                                        (
                                            "blocked",
                                            bp_norm_start,
                                            self.lcm_period,
                                            stream,
                                        )
                                    )
                                    raw_intervals[port].append(
                                        ("blocked", 0.0, bp_norm_end, stream)
                                    )
                                else:  # BP duration >= LCM
                                    raw_intervals[port].append(
                                        ("blocked", 0.0, self.lcm_period, stream)
                                    )
            except Exception as e:
                print(
                    f"ERROR: Failed processing stream {stream}, path {path} for wrapped intervals: {e}"
                )

        # --- Merge overlapping intervals for each port ---
        merged_intervals = {}
        for port, intervals in raw_intervals.items():
            if not intervals:
                merged_intervals[port] = []
                continue

            # Sort by start time, then end time (helps merging)
            intervals.sort(key=lambda x: (x[1], x[2]))

            merged = []
            if not intervals:
                continue

            # Initialize with the first interval
            current_status, current_start, current_end, current_stream = intervals[0]

            for i in range(1, len(intervals)):
                next_status, next_start, next_end, next_stream = intervals[i]

                # Check for overlap or adjacency (within epsilon)
                if next_start < current_end + 1e-9:
                    # Merge: Extend the end time
                    current_end = max(current_end, next_end)
                    # 'used' status takes precedence over 'blocked' if they overlap
                    if current_status == "blocked" and next_status == "used":
                        current_status = "used"
                        current_stream = next_stream  # Update representative stream if status changes
                    # If statuses are the same, keep the stream ID from the interval that started earlier (already current_stream)
                else:
                    # No overlap, finish the previous merged interval
                    if (
                        current_end - current_start > 1e-9
                    ):  # Add only if duration is significant
                        merged.append(
                            (current_status, current_start, current_end, current_stream)
                        )
                    # Start a new interval
                    current_status, current_start, current_end, current_stream = (
                        next_status,
                        next_start,
                        next_end,
                        next_stream,
                    )

            # Add the last merged interval
            if current_end - current_start > 1e-9:
                merged.append(
                    (current_status, current_start, current_end, current_stream)
                )

            merged_intervals[port] = merged

        return merged_intervals

    def draw_wrapped_scheduling_chart(self, highlight_stream_index=None):
        """
        Draws the scheduling chart WRAPPED within a single LCM period.
        Shows merged transmission ('used') and 'blocked' intervals.
        """
        print("--- Generating Wrapped Scheduling Chart ---")
        # Check prerequisites
        if not self.ports_set or self.lcm_period is None or self.lcm_period <= 0:
            print(
                "WARNING: Missing data (ports, LCM) for wrapped scheduling chart. Skipping."
            )
            return

        # Get merged intervals for all ports
        port_intervals = self._get_wrapped_port_intervals()

        if not port_intervals:
            print(
                "WARNING: No scheduling intervals calculated. Skipping wrapped chart."
            )
            return

        # Sort ports for consistent Y-axis
        # Use only ports that actually have intervals calculated
        sorted_ports = sorted(
            [port for port in self.ports_set if port in port_intervals]
        )
        if not sorted_ports:
            print(
                "WARNING: No intervals found for any active ports. Skipping wrapped chart."
            )
            return

        port_to_y = {port: i for i, port in enumerate(sorted_ports)}
        num_ports = len(sorted_ports)

        # --- Create the plot ---
        # Adaptive sizing
        ytick_fontsize = max(2, min(6, 60 / max(10, num_ports)))
        label_fontsize = max(4, min(8, 80 / max(10, num_ports)))
        title_fontsize = max(6, min(10, 100 / max(10, num_ports)))
        legend_fontsize = max(3, min(6, 60 / max(10, num_ports)))
        fig_height = max(6, min(25, num_ports * 0.35))  # Adjusted height factor
        fig, ax = plt.subplots(figsize=(15, fig_height))  # Wider figure

        # Plot bars from merged intervals
        for port, intervals in port_intervals.items():
            if port not in port_to_y:
                continue  # Skip if port wasn't included in sorted_ports
            y_pos = port_to_y[port]
            for status, start, end, stream_id in intervals:
                duration = end - start
                if duration <= 1e-9:
                    continue  # Skip negligible duration intervals

                is_blocked = status == "blocked"
                # Determine color
                if is_blocked:
                    color = "lightcoral"
                elif stream_id == highlight_stream_index:
                    color = "blue"
                else:
                    color = "skyblue"

                ax.broken_barh(
                    [(start, duration)],
                    (y_pos - 0.4, 0.8),  # Bar position and height
                    facecolors=color,
                    edgecolor="gray",
                    linewidth=0.2,
                )
                # Add text label (using the representative stream_id)
                self._add_text_label(
                    ax, stream_id, start, duration, y_pos, num_ports, is_blocked
                )

        # Configure axes and labels
        ax.set_xlim(0, self.lcm_period)
        ax.set_ylim(-0.5, num_ports - 0.5)
        ax.set_xlabel(f"Time", fontsize=label_fontsize)
        ax.set_ylabel("Port Gates", fontsize=label_fontsize)
        ax.set_yticks(range(num_ports))
        ax.set_yticklabels(sorted_ports, fontsize=ytick_fontsize)
        ax.set_title(
            "Scheduling Windows for Port Gates",
            fontsize=title_fontsize,
        )
        ax.invert_yaxis()  # Show ports top-down

        # Create Legend
        handles = [
            Rectangle(
                (0, 0),
                1,
                1,
                fc="skyblue",
                ec="gray",
                lw=0.2,
                label="Transmission Window",
            )
        ]
        if highlight_stream_index is not None:
            handles.insert(
                1,
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    fc="blue",
                    ec="gray",
                    lw=0.2,
                    label=f"Stream {highlight_stream_index}",
                ),
            )

        ax.legend(handles=handles, loc="upper right", fontsize=legend_fontsize)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # Save the plot
        chart_path = os.path.join(
            self.processed_results_dir, "port_schedules_wrapped.png"
        )
        try:
            plt.savefig(chart_path, format="png", dpi=400)  # Adjusted DPI
            print(f"Wrapped scheduling chart saved as {chart_path}")
        except Exception as e:
            print(f"ERROR: Failed to save wrapped scheduling chart: {e}")
        finally:
            plt.close(fig)  # Ensure figure is closed

        print("--- Wrapped Scheduling Chart Generation Complete ---")

    def calculate_port_utilization_metrics(
        self, min_blank_size=100
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Calculates utilization, blocked periods, utilizable free space, and non-utilizable free space
        for each link based on the wrapped intervals within the LCM period.

        Args:
            min_blank_size (float): Minimum consecutive blank time units for space to be "utilizable".

        Returns:
            A tuple containing four DataFrames (indexed by source, columns by destination):
            1. utilization_df: Percentage of time the link is used for transmission.
            2. blocked_df: Percentage of time the link is blocked (by other streams in same queue).
            3. utilizable_df: Percentage of time the link is free in utilizable blocks (>= min_blank_size).
            4. non_utilizable_df: Percentage of time the link is free in non-utilizable blocks (< min_blank_size).
        """
        print(
            f"--- Calculating Port Utilization Metrics (min_blank_size={min_blank_size}) ---"
        )
        # Check prerequisites
        if (
            self.lcm_period is None
            or self.lcm_period <= 0
            or not self.all_nodes
            or self.ports_set is None
        ):
            print(
                "WARNING: Missing data (LCM, nodes, ports) for utilization calculation. Returning empty results."
            )
            empty_df = pd.DataFrame(
                index=self.all_nodes or [], columns=self.all_nodes or [], dtype=float
            )
            return empty_df, empty_df.copy(), empty_df.copy(), empty_df.copy()

        # Get the merged wrapped intervals (includes 'used' and 'blocked')
        port_intervals = self._get_wrapped_port_intervals()

        num_nodes = len(self.all_nodes)
        node_to_idx = {node: i for i, node in enumerate(self.all_nodes)}

        # Initialize matrices with NaN (indicates no direct link or not calculated)
        util_matrix = np.full((num_nodes, num_nodes), np.nan)
        blocked_matrix = np.full((num_nodes, num_nodes), np.nan)
        utilizable_matrix = np.full((num_nodes, num_nodes), np.nan)
        non_utilizable_matrix = np.full((num_nodes, num_nodes), np.nan)

        # --- Calculate metrics for each activated port ---
        for port in self.ports_set:
            sender, receiver = port
            sender_idx = node_to_idx.get(sender)
            receiver_idx = node_to_idx.get(receiver)
            if sender_idx is None or receiver_idx is None:
                continue  # Should not happen

            intervals = port_intervals.get(
                port, []
            )  # Get merged intervals for this port

            # Separate intervals by status
            used_intervals = sorted(
                [
                    (start, end)
                    for status, start, end, _ in intervals
                    if status == "used"
                ]
            )
            blocked_intervals = sorted(
                [
                    (start, end)
                    for status, start, end, _ in intervals
                    if status == "blocked"
                ]
            )

            # Calculate total used and blocked time from merged intervals
            total_used_time = sum(max(0, end - start) for start, end in used_intervals)
            total_blocked_time = sum(
                max(0, end - start) for start, end in blocked_intervals
            )

            # Calculate utilization percentages
            utilization_percent = min(
                100.0, (total_used_time / self.lcm_period) * 100.0
            )
            blocked_percent = min(100.0, (total_blocked_time / self.lcm_period) * 100.0)

            util_matrix[sender_idx, receiver_idx] = utilization_percent
            blocked_matrix[sender_idx, receiver_idx] = blocked_percent

            # --- Calculate Free Space (Utilizable vs Non-Utilizable) ---
            # Free space is the time NOT covered by 'used' intervals.
            utilizable_time = 0.0
            non_utilizable_time = 0.0
            current_time = 0.0

            # Iterate through the gaps BETWEEN 'used' intervals
            for start, end in used_intervals:
                free_duration = start - current_time
                if free_duration > 1e-9:  # If there's a gap
                    if free_duration >= min_blank_size:
                        utilizable_time += free_duration
                    else:
                        non_utilizable_time += free_duration
                # Move current time marker to the end of the 'used' interval
                current_time = max(current_time, end)

            # Check free space after the last 'used' interval up to the LCM period end
            final_free_duration = self.lcm_period - current_time
            if final_free_duration > 1e-9:
                if final_free_duration >= min_blank_size:
                    utilizable_time += final_free_duration
                else:
                    non_utilizable_time += final_free_duration

            # Calculate percentages for free space
            utilizable_percent = min(100.0, (utilizable_time / self.lcm_period) * 100.0)
            non_utilizable_percent = min(
                100.0, (non_utilizable_time / self.lcm_period) * 100.0
            )

            # --- Normalization Check ---
            # Ensure the sum of used + utilizable + non-utilizable is ~100%
            # Blocked time overlaps with free time conceptually, so it's not part of this sum.
            total_free_calc = (
                utilization_percent + utilizable_percent + non_utilizable_percent
            )
            if (
                abs(total_free_calc - 100.0) > 0.1
            ):  # Allow small tolerance for float errors
                # print(f"Warning: Utilization calculation for port {port} sums to {total_free_calc:.2f}%. Adjusting non-utilizable.")
                # Adjust non-utilizable space primarily
                non_utilizable_percent = max(
                    0.0, 100.0 - utilization_percent - utilizable_percent
                )

            utilizable_matrix[sender_idx, receiver_idx] = utilizable_percent
            non_utilizable_matrix[sender_idx, receiver_idx] = non_utilizable_percent

        # Convert numpy arrays to DataFrames
        index_cols = self.all_nodes
        utilization_df = pd.DataFrame(util_matrix, index=index_cols, columns=index_cols)
        blocked_df = pd.DataFrame(blocked_matrix, index=index_cols, columns=index_cols)
        utilizable_df = pd.DataFrame(
            utilizable_matrix, index=index_cols, columns=index_cols
        )
        non_utilizable_df = pd.DataFrame(
            non_utilizable_matrix, index=index_cols, columns=index_cols
        )

        print("--- Port Utilization Metrics Calculation Complete ---")
        return utilization_df, blocked_df, utilizable_df, non_utilizable_df

    def update_streams_csv(self):
        """
        Calculates utilization metrics and draws heatmaps for:
        1. Link Utilization (%)
        2. Blocked Periods (%)
        3. Utilizable Free Space (%)
        4. Non-Utilizable Free Space (%)
        Also saves metrics to CSV and generates a summary bar chart by node type.
        """
        print(
            f"--- Generating Utilization Heatmaps & Summary (min_blank_size={min_blank_size}) ---"
        )

        # Calculate all metrics first
        utilization_df, blocked_df, utilizable_df, non_utilizable_df = (
            self.calculate_port_utilization_metrics(min_blank_size)
        )

        # Check if DataFrames are valid
        if (
            utilization_df.empty
            or blocked_df.empty
            or utilizable_df.empty
            or non_utilizable_df.empty
        ):
            print(
                "WARNING: Calculated utilization data is empty. Skipping heatmap generation."
            )
            return

        nodes = utilization_df.index.tolist()
        num_nodes = len(nodes)
        if num_nodes == 0:
            print(
                "WARNING: No nodes found in utilization data. Skipping heatmap generation."
            )
            return

        # --- Calculate node type statistics ---
        node_types = {}
        for node in nodes:
            try:
                parts = str(node).split("-")
                type_str = (
                    "".join(c for c in parts[1] if c.isalpha())
                    if len(parts) > 1
                    else "".join(c for c in str(node) if c.isalpha())
                )
                node_types[node] = type_str if type_str else "UNKNOWN"
            except Exception:
                node_types[node] = "UNKNOWN"

        # Define type combinations to analyze
        type_combinations = sorted(
            list(
                set(
                    (node_types.get(s, "UNKNOWN"), node_types.get(r, "UNKNOWN"))
                    # Corrected check for NaN using np.isnan()
                    for s in nodes
                    for r in nodes
                    if not np.isnan(
                        utilization_df.loc[s, r]
                    )  # Only consider existing links
                )
            )
        )

        # Calculate averages for each metric by node type combination
        metrics_avg = defaultdict(
            lambda: defaultdict(list)
        )  # {metric_name: {combo: [values]}}
        for sender in nodes:
            for receiver in nodes:
                combo = (
                    node_types.get(sender, "UNKNOWN"),
                    node_types.get(receiver, "UNKNOWN"),
                )
                # Use np.isnan() for checking individual values
                util_val = utilization_df.loc[sender, receiver]
                if not np.isnan(util_val):
                    metrics_avg["util"][combo].append(util_val)

                blocked_val = blocked_df.loc[sender, receiver]
                if not np.isnan(blocked_val):
                    metrics_avg["blocked"][combo].append(blocked_val)

                utilizable_val = utilizable_df.loc[sender, receiver]
                if not np.isnan(utilizable_val):
                    metrics_avg["utilizable"][combo].append(utilizable_val)

                non_utilizable_val = non_utilizable_df.loc[sender, receiver]
                if not np.isnan(non_utilizable_val):
                    metrics_avg["non_utilizable"][combo].append(non_utilizable_val)

        # Compute final averages
        final_averages = defaultdict(dict)  # {metric_name: {combo: avg_value}}
        overall_averages = {}  # {metric_name: overall_avg_value}
        for metric, combo_data in metrics_avg.items():
            all_vals_metric = []
            for combo, vals in combo_data.items():
                avg = np.mean(vals) if vals else np.nan
                final_averages[metric][combo] = avg
                if not np.isnan(avg):
                    all_vals_metric.extend(vals)  # Collect valid values for overall avg
            overall_averages[metric] = (
                np.mean(all_vals_metric) if all_vals_metric else np.nan
            )

        # --- Create and Save Summary CSV ---
        summary_data = {
            "Link Type": [f"{c[0]}->{c[1]}" for c in type_combinations] + ["Overall"]
        }
        summary_data["Utilization (%)"] = [
            final_averages["util"].get(c, np.nan) for c in type_combinations
        ] + [overall_averages.get("util", np.nan)]
        summary_data["Blocked (%)"] = [
            final_averages["blocked"].get(c, np.nan) for c in type_combinations
        ] + [overall_averages.get("blocked", np.nan)]
        summary_data[f"Utilizable Free (>= {min_blank_size}) (%)"] = [
            final_averages["utilizable"].get(c, np.nan) for c in type_combinations
        ] + [overall_averages.get("utilizable", np.nan)]
        summary_data[f"Non-Utilizable Free (< {min_blank_size}) (%)"] = [
            final_averages["non_utilizable"].get(c, np.nan) for c in type_combinations
        ] + [overall_averages.get("non_utilizable", np.nan)]

        avg_summary_df = pd.DataFrame(summary_data)
        helper.save_dataframe_as_csv(  # Updated call
            avg_summary_df,
            self.processed_results_dir,
            f"utilization_summary_{min_blank_size}.csv",
            index=False,
            float_format="%.2f",
        )

        # --- Plot Summary Bar Chart ---
        try:
            plt.figure(figsize=(max(12, len(type_combinations) * 0.8), 7))
            link_labels = summary_data["Link Type"]
            x = np.arange(len(link_labels))
            width = 0.2

            plt.bar(
                x - 1.5 * width,
                summary_data["Utilization (%)"],
                width,
                label="Utilization",
                color="crimson",
                alpha=0.8,
            )
            plt.bar(
                x - 0.5 * width,
                summary_data["Blocked (%)"],
                width,
                label="Blocked",
                color="mediumpurple",
                alpha=0.8,
            )
            plt.bar(
                x + 0.5 * width,
                summary_data[f"Utilizable Free (>= {min_blank_size}) (%)"],
                width,
                label=f"Utilizable Free (≥{min_blank_size})",
                color="royalblue",
                alpha=0.8,
            )
            plt.bar(
                x + 1.5 * width,
                summary_data[f"Non-Utilizable Free (< {min_blank_size}) (%)"],
                width,
                label=f"Non-Utilizable Free (<{min_blank_size})",
                color="darkorange",
                alpha=0.8,
            )

            plt.xlabel("Link Type (Source -> Destination)", fontsize=11)
            plt.ylabel("Percentage (%)", fontsize=11)
            plt.title(
                f"Average Link Metrics by Node Type (min_blank_size={min_blank_size})",
                fontsize=13,
            )
            plt.xticks(x, link_labels, rotation=45, ha="right", fontsize=9)
            plt.yticks(fontsize=9)
            plt.legend(
                fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2
            )
            plt.grid(axis="y", linestyle="--", alpha=0.6)
            plt.ylim(0, 105)  # Set Y limit slightly above 100
            plt.tight_layout(rect=[0, 0.1, 1, 0.96])  # Adjust layout

            summary_chart_path = os.path.join(
                self.processed_results_dir,
                f"utilization_by_node_type_{min_blank_size}.png",
            )
            plt.savefig(summary_chart_path, format="png", dpi=300)
            print(f"Node type utilization summary chart saved as {summary_chart_path}")
        except Exception as e:
            print(f"ERROR: Failed to save node type summary chart: {e}")
        finally:
            plt.close()

        # --- Plotting Heatmaps ---
        fig_size = max(8, num_nodes * 0.5)  # Dynamic figure size based on node count
        label_fontsize = max(6, min(10, fig_size * 1.0))
        title_fontsize = max(8, min(12, fig_size * 1.2))
        annot_kws = {"size": max(3, min(7, fig_size * 0.6))}  # Dynamic annotation size

        heatmap_common_args = dict(
            annot=True,
            fmt=".1f",
            linewidths=0.3,
            linecolor="lightgray",
            cbar=True,
            square=True,
            vmin=0,
            vmax=100,
            annot_kws=annot_kws,
            xticklabels=nodes,
            yticklabels=nodes,
        )

    def update_streams_csv(self):
        """
        Adds 'scheduled' and 'success_probability' columns to the streams DataFrame
        and saves it as 'streams_updated.csv'.
        """
        print("--- Updating Streams CSV ---")
        if self.streams_df is None:
            print("WARNING: streams_df is None. Cannot update streams CSV.")
            return

        streams_with_updates = self.streams_df.copy()

        # --- Add 'scheduled' column based on variables_a ---
        # ... (this part remains the same) ...
        scheduled_col_added = False
        if self.variables_a is not None:
            try:
                scheduled_mapping = {}
                for key, value in self.variables_a.items():
                    try:
                        stream_index = int(
                            key[0] if isinstance(key, tuple) and key else key
                        )
                        scheduled_mapping[stream_index] = int(value)
                    except (IndexError, TypeError, ValueError):
                        print(f"WARNING: Could not process key {key} from variables_a")

                if scheduled_mapping:
                    a_series = pd.Series(scheduled_mapping, dtype=int)
                    try:
                        streams_with_updates["scheduled"] = (
                            streams_with_updates.index.map(a_series)
                        )
                    except TypeError:
                        try:
                            streams_with_updates["scheduled"] = (
                                streams_with_updates.index.astype(int).map(a_series)
                            )
                        except Exception as e_map:
                            print(
                                f"ERROR: Failed to map scheduled values using index: {e_map}. Setting 'scheduled' to 0."
                            )
                            streams_with_updates["scheduled"] = 0

                    streams_with_updates["scheduled"] = (
                        streams_with_updates["scheduled"].fillna(0).astype(int)
                    )
                    print("Added 'scheduled' column based on variables_a.")
                    scheduled_col_added = True
                else:
                    print("WARNING: No valid stream indices found in variables_a.")

            except Exception as e:
                print(
                    f"ERROR: Failed to process variables_a for 'scheduled' column: {e}"
                )

        if not scheduled_col_added:
            print("WARNING: 'scheduled' column not added. Setting to default 0.")
            streams_with_updates["scheduled"] = 0

        # --- Calculate and add 'success_probability' column --- # Renamed column
        success_prob_col_added = False  # Renamed variable
        if self.histogram_config is not None:
            # Call the renamed method
            success_probabilities = (
                self.calculate_success_probabilities()
            )  # Renamed method call
            if success_probabilities:  # Check if the dictionary is not empty
                prob_series = pd.Series(success_probabilities, dtype=float)
                try:
                    # Map using index
                    streams_with_updates["success_probability"] = (  # Renamed column
                        streams_with_updates.index.map(prob_series)
                    )
                except TypeError:
                    try:
                        # Try converting index if needed
                        streams_with_updates["success_probability"] = (
                            streams_with_updates.index.astype(  # Renamed column
                                int
                            ).map(prob_series)
                        )
                    except Exception as e_map_prob:
                        print(
                            f"ERROR: Failed to map success probabilities using index: {e_map_prob}. Setting 'success_probability' to 1.0."
                        )
                        streams_with_updates["success_probability"] = (
                            1.0  # Default to success
                        )

                # Fill NaN values (e.g., unscheduled streams) with 1.0 (success)
                streams_with_updates["success_probability"] = streams_with_updates[
                    "success_probability"  # Renamed column
                ].fillna(1.0)
                print("Added 'success_probability' column.")  # Updated message
                success_prob_col_added = True  # Renamed variable
            else:
                print(
                    "WARNING: Success probability calculation returned empty results."
                )
        else:
            print(
                "WARNING: Histogram config not loaded. Skipping success probability calculation."
            )

        if not success_prob_col_added:  # Renamed variable
            print(
                "WARNING: 'success_probability' column not added. Setting to default 1.0."  # Updated message
            )
            streams_with_updates["success_probability"] = (
                1.0  # Renamed column, default to success
            )

        # --- Save the updated streams CSV ---
        helper.save_dataframe_as_csv(  # Updated call
            streams_with_updates,
            self.processed_results_dir,
            "streams_updated.csv",
            index=True,
            float_format="%.6g",
        )

        print("--- Streams CSV Update Complete ---")

    def process(self, highlight_stream_index=None, min_blank_size=100):
        """
        Main processing pipeline: Loads data, prepares common structures,
        generates intermediate CSVs, creates plots, calculates metrics,
        and saves the final updated streams CSV.
        """
        print("=" * 20 + " Starting Output Processing " + "=" * 20)
        self.load_files()

        # Prepare common data needed for subsequent steps (LCM, activated streams, etc.)
        if not self.prepare_common_data():
            print(
                "ERROR: Halting processing due to errors during common data preparation."
            )
            return

        # Generate scheduling charts
        self.draw_scheduling_chart(highlight_stream_index)
        self.draw_wrapped_scheduling_chart(highlight_stream_index)

        # Update streams CSV (calculates miss probability internally now)
        self.update_streams_csv()

        print("=" * 20 + " Output Processing Finished " + "=" * 20)


# Example Usage Block
if __name__ == "__main__":
    print("Running OutputManager example...")

    # --- Configuration ---
    # Adapt these paths based on your project structure
    script_dir = os.path.dirname(__file__)

    FOLDER_UUID = "ba38de57-3380-40c3-af6e-cc6b02951205"  # Example UUID
    # For single run output generator only output dir exist
    input_dir = os.path.join(script_dir, "data", "output", FOLDER_UUID)
    output_dir = os.path.join(script_dir, "data", "output", FOLDER_UUID)

    # Parameters for processing
    highlight_stream = (
        None  # Set to an integer stream ID to highlight in charts, or None
    )
    min_blank_space_for_utilization = (
        100  # Threshold for utilizable free space calculation
    )

    # --- Setup ---
    # Create PathManager instance pointing to the specific results folder
    # PathManager expects the base data directory (e.g., ../data) and the UUID
    try:
        path_manager = PathManager(input_dir=input_dir, output_dir=output_dir)
        print(f"PathManager Input Path: {path_manager.input_dir}")
        print(f"PathManager Output Path: {path_manager.output_dir}")
    except NameError:
        print(
            "ERROR: PathManager class not found. Make sure it's imported correctly from utils.helper."
        )
        exit()
    except Exception as e:
        print(f"ERROR: Failed to initialize PathManager: {e}")
        exit()

    # Create OutputManager instance
    output_manager = OutputManager(path_manager)

    # --- Run Processing ---
    try:
        output_manager.process(
            highlight_stream_index=highlight_stream,
            min_blank_size=min_blank_space_for_utilization,
        )
        print("\nOutput processing script finished successfully!")
        print(f"Check results in: {output_manager.processed_results_dir}")
    except Exception as e:
        print(f"\nERROR: An error occurred during output processing: {e}")
        import traceback

        traceback.print_exc()  # Print detailed traceback for debugging
