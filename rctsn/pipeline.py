from models.data_factory import DataFactory
from models.output_generator import OutputManager
from utils.helper import (
    load_graphml,
    load_streams_from_csv,
    load_yaml_file,
    save_as_graphml,
    save_as_pkl,
)
from utils.path_manager import PathManager
from models.batches.main import solve_with_batches, save_timing_info
from utils.memory_tracker import MemoryTracker  

import os
import uuid
import shutil
import argparse
from datetime import datetime
import traceback
from typing import List, Dict
import networkx as nx


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run TSN Scheduling Pipeline")
    parser.add_argument('--input', type=str, help='Path to input directory',
                       default=None)  # Will default to data/input in main()
    parser.add_argument('--output', type=str, help='Path to output directory',
                       default=None)  # Will default to data/output in main()
    parser.add_argument('--config', type=str, help='Path to config file',
                       default=None)  # Will try input_dir/link_config.yaml by default
    parser.add_argument('--streams', type=int, help='Number of streams to process',
                       default=None) # Changed default to None to distinguish from not set
    parser.add_argument('--batch-size', type=int, help='Batch size for processing',
                       default=10)
    parser.add_argument('--time-limit', type=int, help='Time limit for solver in seconds',
                       default=60)
    parser.add_argument('--k-paths', type=int, help='Number of paths per stream',
                       default=3)
    parser.add_argument('--highlight', type=int, help='Stream ID to highlight',
                       default=None)
    parser.add_argument('--memory-interval', type=int, help='Memory sampling interval in seconds',
                       default=5)
    parser.add_argument('--gamma', type=float, help='Global gamma value to override for all streams',
                       default=None)
    return parser.parse_args()


def create_batches(stream_ids, batch_size):
    """Create batches for optimization from a list of stream IDs"""
    batches = []
    for i in range(0, len(stream_ids), batch_size):
        batch = stream_ids[i:i + batch_size]
        batches.append(batch)
    return batches


def create_priority_batches(streams_df, stream_ids, batch_size, priority_column="priority"):
    """
    Create batches for optimization based on stream priorities.
    Higher priority streams (higher numerical values) are placed in earlier batches.
    
    Args:
        streams_df: DataFrame containing stream information with priority column. 
                    Assumes stream_id can be used for iloc if streams_df is indexed 0..N-1.
        stream_ids: List of valid stream IDs to include in batches. These IDs should align with
                    how DataFactory assigns them (typically 0 to N-1 based on processed streams).
        batch_size: Maximum number of streams per batch
        priority_column: Column name for priority in the DataFrame (default: 'priority')
        
    Returns:
        List of batches, where each batch is a list of stream IDs
    """
    batches = []
    
    # Check if priority column exists and streams_df is valid
    if streams_df is not None and not streams_df.empty and priority_column in streams_df.columns:
        print("Creating batches based on stream priority (higher priority value = higher priority)")
        
        # Filter streams_df to only include relevant stream_ids that are present in its index
        # This assumes stream_ids from DataFactory (0 to N-1) correspond to DataFrame iloc positions
        # or that streams_df is already indexed by these stream_ids.
        # For safety, we'll map stream_ids to priorities if they exist in the DataFrame's index.
        
        # Create a mapping of stream IDs to priorities
        priorities = {}
        # Ensure streams_df is indexed by the 'index' column if it exists, or default 0..N-1
        df_for_priority = streams_df
        if 'index' in df_for_priority.columns and df_for_priority.index.name != 'index':
             # If 'index' column exists and is not the actual index, try to set it.
             # This is a common case if streams.csv has an 'index' column but it's not auto-loaded as index.
            try:
                df_for_priority = streams_df.set_index('index', drop=False) # Keep original index column if needed elsewhere
            except KeyError: # 'index' column might not exist
                pass


        for stream_id in stream_ids: # stream_ids are 0 to N-1 from DataFactory
            try:
                # If stream_id is a direct index in df_for_priority (e.g. after set_index)
                if stream_id in df_for_priority.index:
                    priority = df_for_priority.loc[stream_id, priority_column]
                # Else, if stream_id refers to an iloc position (original assumption)
                elif stream_id < len(df_for_priority): # Check bounds for iloc
                    priority = df_for_priority.iloc[stream_id][priority_column]
                else:
                    priority = 0 # Default priority if stream_id not found
                priorities[stream_id] = priority
            except (KeyError, IndexError):
                # print(f"Warning: Stream ID {stream_id} not found in streams_df for priority batching. Assigning default priority 0.")
                priorities[stream_id] = 0  # Default priority if lookup fails
        
        # Sort stream IDs by priority (HIGHER value = higher priority)
        sorted_ids = sorted(priorities.keys(), key=lambda x: priorities.get(x, 0), reverse=True)
        
        # Create batches based on priority order
        for i in range(0, len(sorted_ids), batch_size):
            batch = sorted_ids[i:i + batch_size]
            batches.append(batch)
            
        # Print priority distribution
        if priority_column in streams_df.columns:
            priority_counts = streams_df[priority_column].value_counts().sort_index()
            print(f"Priority distribution: {dict(priority_counts)}")
    else:
        if streams_df is None or streams_df.empty:
            print("Streams DataFrame is empty or not loaded. Creating batches sequentially.")
        elif priority_column not in streams_df.columns:
            print(f"Priority column '{priority_column}' not found in streams.csv. Creating batches sequentially.")
        batches = create_batches(stream_ids, batch_size)
    
    return batches


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up paths
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    # Default input_dir if not provided by args
    default_input_name = "input" # Default folder name within data_dir
    input_dir = args.input if args.input else os.path.join(data_dir, default_input_name)
    
    # Create folder name with format: run_YYYYMMDD_HHMMSS_uuid-part
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uuid_part = str(uuid.uuid4())[:8]  # Take just the first 8 characters of the UUID
    folder_name = f"run_{timestamp}_{uuid_part}"
    
    # Default output_dir if not provided by args
    output_dir = args.output if args.output else os.path.join(data_dir, "output", folder_name)

    # Print start information
    print("=== Starting TSN Scheduling Pipeline ===")
    start_time = datetime.now()
    print(f"Start time: {start_time}")
    
    # Initialize PathManager AFTER output_dir is finalized
    path_manager = PathManager(input_dir=input_dir, output_dir=output_dir)
    
    print(f"Input directory: {path_manager.input_dir}") # Now reflects actual input_dir used
    print(f"Output folder: {path_manager.output_dir}")
    
    # Start memory tracking
    tracker = None
    try:
        tracker = MemoryTracker(output_folder=path_manager.memory_dir, interval=args.memory_interval)
        tracker.start()
    except Exception as e:
        print(f"WARNING: Failed to initialize memory tracker: {e}")
        # Continue without tracking
    
    streams_df_for_priority_batching = None # This will hold the df used for batching
    
    try:
        # Copy input files, with special handling for streams.csv
        print("Copying input files...")
        if not os.path.exists(path_manager.input_dir):
            print(f"ERROR: Source input directory does not exist: {path_manager.input_dir}")
            return # Exit if source input directory is missing
            
        os.makedirs(path_manager.output_dir, exist_ok=True) # Ensure output base dir exists for copying

        for item in os.listdir(path_manager.input_dir):
            s = os.path.join(path_manager.input_dir, item)
            d = os.path.join(path_manager.output_dir, item) # Target path in the run's output folder
            
            if item == "streams.csv":
                try:
                    import pandas as pd
                    # Read the original CSV file
                    temp_df = pd.read_csv(s)
                    original_count = len(temp_df)
                    
                    # Apply global gamma override if provided
                    if args.gamma is not None:
                        print(f"Applying global gamma override: {args.gamma}")
                        if 'gamma' not in temp_df.columns:
                            print(f"Warning: 'gamma' column not found in {s}. Adding it.")
                        temp_df['gamma'] = args.gamma
                    
                    # Limit the number of rows if args.streams is set
                    if args.streams is not None and args.streams < original_count:
                        print(f"Limiting streams.csv from {original_count} to {args.streams} rows")
                        streams_df_to_save = temp_df.iloc[:args.streams].copy()
                    else:
                        streams_df_to_save = temp_df.copy()
                    
                    # Save the (potentially modified and limited) dataframe to the output directory
                    streams_df_to_save.to_csv(d, index=False)
                    streams_df_for_priority_batching = streams_df_to_save # Use this for batching
                    print(f"Processed and copied streams.csv to {d}")

                except Exception as e:
                    print(f"WARNING: Error processing streams.csv for gamma/limiting: {e}. Falling back to direct copy.")
                    shutil.copy2(s, d)
                    try: # Try to load the copied df for batching, it won't have overrides if copy failed before save
                        streams_df_for_priority_batching = pd.read_csv(d)
                    except Exception as e_load:
                        print(f"WARNING: Could not load copied streams.csv for batching: {e_load}")
                        streams_df_for_priority_batching = None
            elif os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
                
        print("Input files copied.")
        
        # Load data (streams will be loaded from the modified streams.csv in output_dir)
        print("\n--- Loading Data ---")
        # path_manager.streams_output_dir now points to the streams.csv inside the run's output folder
        streams = load_streams_from_csv(path_manager.streams_output_dir) 
        network_graph = load_graphml(path_manager.network_graphml_input_dir) # From original input
        
        # Determine link_config path
        # Default to input_dir/link_config.yaml if args.config is not provided
        link_config_path = args.config or os.path.join(path_manager.input_dir, "link_config.yaml")
        if not os.path.exists(link_config_path):
             # Fallback to output_dir/link_config.yaml if not in input_dir (e.g. if it was copied)
            link_config_path_alt = os.path.join(path_manager.output_dir, "link_config.yaml")
            if os.path.exists(link_config_path_alt):
                link_config_path = link_config_path_alt
            else:
                print(f"Warning: Link config not found at {link_config_path} or {link_config_path_alt}. Trying default name in input_dir.")
                # Try a common default name if specific path fails
                link_config_path = os.path.join(path_manager.input_dir, "link_config.yaml")


        print(f"Attempting to load link configuration from: {link_config_path}")
        link_config = load_yaml_file(link_config_path)
        
        if not streams:
            raise ValueError(f"Failed to load streams from CSV at {path_manager.streams_output_dir}")
        if not network_graph:
            raise ValueError(f"Failed to load network graph from {path_manager.network_graphml_input_dir}")
        if not link_config:
            raise ValueError(f"Failed to load link configuration from {link_config_path}")
            
        print(f"Loaded {len(streams)} streams (from {path_manager.streams_output_dir}), "
              f"{len(network_graph.nodes)} nodes, and {len(network_graph.edges)} edges.")
        
        # Create DataFactory
        # stream_limit is implicitly handled as `streams` list is already limited.
        # DataFactory will use the gamma from the `streams` list of dicts.
        print(f"\n--- Initializing DataFactory with k_paths={args.k_paths} ---")
        print(streams)
        data_factory = DataFactory(
            link_config=link_config,
            graph_nx=network_graph,
            raw_streams=streams, 
            k_path_count=args.k_paths,
            lcm_rep=1
            # stream_limit is not needed here as `streams` is already the limited & gamma-modified list
        )
        
        print(f"DataFactory initialized with {len(data_factory.stream_dict)} streams and {len(data_factory.ports)} ports")
        
        # Save the processed network graph (original graph is used, not modified by DataFactory init)
        save_as_graphml(network_graph, path_manager.output_dir, "network_topology_used.graphml")
        print(data_factory.stream_dict)
        # Determine number of streams to process
        stream_ids = list(data_factory.stream_dict.keys()) # These are 0 to N-1
        total_count = len(stream_ids)
        batch_size = args.batch_size
        print(f"Processing {total_count} streams in batches of {batch_size}")
        
        # Create batches based on priority if available
        # Use streams_df_for_priority_batching which has the limited and gamma-modified data
        batches = create_priority_batches(streams_df_for_priority_batching, stream_ids, batch_size)
        
        # Print batch information
        print(f"Created {len(batches)} batches:")
        for i, batch in enumerate(batches):
            if streams_df_for_priority_batching is not None and 'priority' in streams_df_for_priority_batching.columns:
                # Get priorities for this batch if we have that information
                batch_priorities = []
                for stream_id in batch:
                    # Ensure stream_id is valid for iloc access if df is 0-indexed and stream_id is within bounds
                    # Or, if df is indexed by 'index' column, use .loc
                    priority_val = '?'
                    if 'index' in streams_df_for_priority_batching.columns and streams_df_for_priority_batching.index.name == 'index':
                        if stream_id in streams_df_for_priority_batching.index:
                            priority_val = streams_df_for_priority_batching.loc[stream_id, 'priority']
                    elif stream_id < len(streams_df_for_priority_batching):
                         priority_val = streams_df_for_priority_batching.iloc[stream_id]['priority']
                    
                    batch_priorities.append(priority_val)
                print(f"  Batch {i+1}: {len(batch)} streams, IDs: {batch}, priorities: {batch_priorities}")
            else:
                print(f"  Batch {i+1}: {len(batch)} streams, IDs: {batch}")
        
        # Run optimization
        print(f"\n--- Running Optimization (time_limit={args.time_limit} seconds) ---")
        (
            status,
            variables_x,
            variables_z,
            variables_a,
            timing_info,
        ) = solve_with_batches(
            data_factory=data_factory,  # Pass DataFactory instance directly
            output_path=path_manager.output_dir,
            batches=batches,
            time_limit=args.time_limit,
        )
        print(f"Optimization status: {status}")
        
        # Save timing information
        save_timing_info(timing_info, path_manager.output_dir)
        
        # Save results
        print("\n--- Saving Results ---")
        save_as_pkl(data_factory.stream_dict, path_manager.output_dir, "stream_dict.pkl")
        save_as_pkl(variables_x, path_manager.output_dir, "variables_x.pkl")
        save_as_pkl(variables_z, path_manager.output_dir, "variables_z.pkl")
        save_as_pkl(variables_a, path_manager.output_dir, "variables_a.pkl")
        print(f"Results saved.")
        
        # Generate output visualization
        print(f"\n--- Generating Output Visualization (highlighting stream {args.highlight}) ---")
        output_manager = OutputManager(path_manager)
        output_manager.process(highlight_stream_index=args.highlight)
            
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
    finally:
        # Stop memory tracking
        if tracker:
            try:
                tracker.stop()
                # Ensure memory directory exists before plotting
                os.makedirs(path_manager.memory_dir, exist_ok=True)
                tracker.plot_usage(path_manager.memory_dir)
            except Exception as e:
                print(f"WARNING: Error stopping or plotting memory tracker: {e}")
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\n=== Pipeline Complete ===")
        print(f"Total execution time: {duration}")
        print(f"Results saved in folder: {folder_name}")
        print(f"Full path: {path_manager.output_dir}")


if __name__ == "__main__":
    main()
