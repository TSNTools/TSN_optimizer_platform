import os
import pickle
import csv
import networkx as nx
import pandas as pd  
import yaml
import math
from typing import Dict, List, Tuple, Any, Optional

def save_as_pkl(data, folder_path, name): 
    """Saves data to a pickle file."""
    pickle_path = os.path.join(folder_path, name)
    try:
        with open(pickle_path, "wb") as pickle_file:
            pickle.dump(data, pickle_file)
        print(f"Data saved to {pickle_path} using pickle")
    except Exception as e:
        print(f"ERROR: Failed to save pickle file {pickle_path}: {e}")

def save_dataframe_as_csv(df, folder_path, filename, index=True, float_format=None):
    """Saves a pandas DataFrame to a CSV file."""
    if df is None:
        print(f"Skipping CSV save for {filename} as DataFrame is None.")
        return
        
    csv_path = os.path.join(folder_path, filename)
    try:
        df.to_csv(csv_path, index=index, float_format=float_format)
        print(f"DataFrame saved to {csv_path}")
    except Exception as e:
        print(f"ERROR: Failed to save DataFrame to CSV {csv_path}: {e}")

def save_as_csv(data_dict, folder_path, filename):
    """Saves a dictionary to a CSV file."""
    csv_path = os.path.join(folder_path, filename)
    try:
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Variable", "Value"])
            for name, value in data_dict.items():
                writer.writerow([name, value])
        print(f"CSV file saved to {csv_path}")
    except Exception as e:
        print(f"ERROR: Failed to save CSV file {csv_path}: {e}")

def save_as_graphml(graph, folder_path, filename):
    """Saves a NetworkX graph to a GraphML file."""
    graphml_path = os.path.join(folder_path, filename)
    try:
        nx.write_graphml(graph, graphml_path)
        print(f"GraphML file saved to {graphml_path}")
    except Exception as e:
        print(f"ERROR: Failed to save GraphML file {graphml_path}: {e}")

def load_pickle(file_path):
    """Load a pickle file and return its content, with error handling."""
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return None
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except PermissionError:
        print(f"Error: Permission denied to read the file '{file_path}'.")
    except pickle.UnpicklingError:
        print(f"Error: The file '{file_path}' is not a valid pickle file or is corrupted.")
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
    return None

def load_yaml_file(file_path: str):
    """Loads a YAML file and returns its content, with error handling."""
    if not isinstance(file_path, str) or not file_path:
        print(f"Error: Invalid or empty YAML file path provided (got type {type(file_path)}): '{file_path}'")
        return None
    if not os.path.isfile(file_path):
        print(f"Error: The YAML file '{file_path}' does not exist.")
        return None
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            # print(f"Successfully loaded YAML file: {file_path}") # Optional: if you want success message from helper
            return data
    except FileNotFoundError: # Should be caught by isfile, but good for robustness
        print(f"Error: The YAML file '{file_path}' was not found.")
    except PermissionError:
        print(f"Error: Permission denied to read the YAML file '{file_path}'.")
    except yaml.YAMLError as e:
        print(f"Error: The file '{file_path}' is not a valid YAML file or is corrupted: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading YAML file {file_path}: {e}")
    return None

def load_graphml(file_path):
    """Load a GraphML file and return it as a NetworkX graph, with error handling."""
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return None
    try:
        graph = nx.read_graphml(file_path)
        return graph
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except PermissionError:
        print(f"Error: Permission denied to read the file '{file_path}'.")
    except nx.NetworkXError as e:
        print(f"Error: The file '{file_path}' is not a valid GraphML file or is corrupted.")
        print(f"NetworkXError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

def load_streams_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load streams from a CSV file with error handling.
    
    Args:
        csv_path (str): Path to the CSV file with stream data
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries with stream data
    """
    streams = []
    try:
        with open(csv_path, mode="r") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                try:
                    # Convert appropriate fields to correct types
                    processed_row = {
                        "src": str(row["src"]),
                        "dst": str(row["dst"]),
                        "packet_size": int(row["packet_size"]),
                        "period": int(row["period"]),
                    }
                    
                    # Handle optional fields
                    if "max_latency" in row and row["max_latency"]:
                        processed_row["max_latency"] = int(row["max_latency"])
                    
                    if "priority" in row and row["priority"]:
                        processed_row["priority"] = int(row["priority"])
                    
                    if "gamma" in row and row["gamma"]:
                        processed_row["gamma"] = float(row["gamma"])
                    else:
                        processed_row["gamma"] = 1.0  # Default gamma value
                    
                    if "max_jitter" in row and row["max_jitter"]:
                        processed_row["max_jitter"] = int(row["max_jitter"])
                    
                    streams.append(processed_row)
                except ValueError as e:
                    print(f"Error processing row in CSV: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: Stream CSV file not found at {csv_path}")
    except Exception as e:
        print(f"Error loading streams from CSV: {e}")
    
    return streams

def calculate_lcm(numbers: list):
    """
    Calculate the least common multiple (LCM) of a list of numbers.

    Args:
        numbers (list): List of numbers.

    Returns:
        int: LCM of the input numbers.
    """
    lcm_result = 1
    for num in numbers:
        if isinstance(num, int) and num > 0:
            lcm_result = math.lcm(lcm_result, num)
    
    return lcm_result if lcm_result > 0 else 1

def calculate_repetitions_list(period_s: int, lcm: int, lcm_rep: int):
    """
    Calculate the repetition list for a given period.

    Args:
        period_s (int): Period of the stream.
        lcm (int): LCM of all periods.
        lcm_rep (int): LCM repetition factor.

    Returns:
        tuple: (last_rep_index, repetitions_list)

    Raises:
        ValueError: If repetition length is not an integer.
    """
    if period_s <= 0:
        raise ValueError(f"Period must be positive: {period_s}")
    
    repetition_length = lcm / period_s
    
    # Check if repetition_length is close to an integer
    if abs(repetition_length - round(repetition_length)) > 1e-10:
        # For non-integer values, round to nearest
        repetition_length = round(repetition_length)
        print(f"Warning: Non-integer repetition length rounded to {repetition_length}")
    else:
        repetition_length = int(repetition_length)
        
    last_rep_index = repetition_length - 1
    virtual_repetition_length = lcm_rep * repetition_length

    return last_rep_index, [x for x in range(1, int(virtual_repetition_length) + 1)]

def get_link_speed(graph, sender, receiver):
    """
    Retrieves the link speed for a directed edge in the graph.

    Args:
        graph (networkx.Graph): The network graph.
        sender (str): The sender node.
        receiver (str): The receiver node.

    Returns:
        float: The link speed between the sender and receiver.
    """
    return graph[sender][receiver][str((sender, receiver))]

