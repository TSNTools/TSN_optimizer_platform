import os
import uuid
import shutil
import time
import csv
import networkx as nx
from datetime import datetime
from models.batches.optimization_model import OptimizationModel
from models.data_factory import DataFactory
from utils.helper import (
    load_streams_from_csv,
    save_as_pkl,
    load_yaml_file,
    load_graphml,
)
from utils.path_manager import PathManager
from models.output_generator import OutputManager
from typing import List, Dict
from models.data_factory import DataFactory


def solve_with_batches(
    data_factory, output_path, batches: List[List[int]], time_limit=3600
):
    """
    Solve the scheduling problem iteratively with exactly the specified batches.
    Batches should not have overlapping streams.

    Args:
        data_factory: DataFactory instance to create the optimization model
        output_path: Path to save outputs
        batches: List of batches, where each batch is a list of stream indices
        time_limit: Maximum solver time in seconds per iteration

    Returns:
        tuple: Status, all decision variables, and timing information
    """
    # Create logs directory
    solver_output_dir = os.path.join(output_path, "solver_files")
    os.makedirs(solver_output_dir, exist_ok=True)

    # Initialize timing tracking
    timing_info = {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "iterations": [],
        "total_time": 0,
    }

    model = OptimizationModel(
        data_factory=data_factory,
        output_path=solver_output_dir,
        time_limit=time_limit,
    )

    print(f"Starting optimization with {len(batches)} batches...")
    all_scheduled = []
    last_status = None
    total_start_time = time.time()

    # Iterate over each batch
    for iteration, batch in enumerate(batches, 1):
        iter_start_time = time.time()
        print(f"\n=== ITERATION {iteration} ===")

        model.set_variable_streams(batch)

        model.update_variable_variable_constraints()
        model.update_variable_fixed_constraints()

        # Solve the model
        status = model.solve(iteration=iteration)
        last_status = status

        # Calculate iteration time
        iter_end_time = time.time()
        iter_duration = iter_end_time - iter_start_time

        print(f"Iteration {iteration} status: {status}")
        print(f"Iteration {iteration} time: {iter_duration:.2f} seconds")

        # Store timing information
        timing_info["iterations"].append(
            {
                "iteration": iteration,
                "batch": batch,
                "status": status,
                "duration_seconds": iter_duration,
                "streams_in_batch": len(batch),
                "streams_scheduled": 0,  # Will update below
            }
        )

        # Check which streams were scheduled in this iteration
        new_scheduled_streams = model.get_newly_scheduled_streams()
        all_scheduled.extend(new_scheduled_streams)

        # Update timing info with scheduled streams
        timing_info["iterations"][-1]["streams_scheduled"] = len(new_scheduled_streams)

        print(f"Scheduled in iteration {iteration}: {new_scheduled_streams}")

        # Fix variables for the newly scheduled streams
        model.fix_scheduled_stream_variables(streams_to_fix=new_scheduled_streams)

    # Calculate total time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    timing_info["total_time"] = total_duration
    timing_info["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get final results
    print("\n=== FINAL RESULTS ===")
    variables_x, variables_z, variables_a = model.get_solution_variables()

    print(f"Total scheduled streams: {len(all_scheduled)} out of {len(data_factory.stream_dict)}")
    print(f"Scheduled stream IDs: {sorted(all_scheduled)}")
    print(f"Total optimization time: {total_duration:.2f} seconds")

    return (
        last_status,
        variables_x,
        variables_z,
        variables_a,
        timing_info,
    )


def save_timing_info(timing_info: Dict, output_folder: str):
    """Save timing information to CSV and summary files"""
    # Save detailed timing CSV
    timing_csv_path = os.path.join(output_folder, "timing_metrics.csv")
    with open(timing_csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "iteration",
            "batch",
            "status",
            "duration_seconds",
            "streams_in_batch",
            "streams_scheduled",
            "success_rate",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for iter_info in timing_info["iterations"]:
            # Calculate success rate
            streams_in_batch = iter_info["streams_in_batch"]
            streams_scheduled = iter_info["streams_scheduled"]
            success_rate = (
                streams_scheduled / streams_in_batch if streams_in_batch > 0 else 0
            )
            iter_info["success_rate"] = success_rate
            writer.writerow(iter_info)

    # Save summary file
    summary_path = os.path.join(output_folder, "timing_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Optimization Timing Summary\n")
        f.write(f"==========================\n\n")
        f.write(f"Start time: {timing_info['start_time']}\n")
        f.write(f"End time: {timing_info['end_time']}\n")
        f.write(f"Total duration: {timing_info['total_time']:.2f} seconds\n\n")

        f.write(f"Iterations: {len(timing_info['iterations'])}\n")
        f.write(
            f"Average iteration time: {sum(i['duration_seconds'] for i in timing_info['iterations']) / len(timing_info['iterations']):.2f} seconds\n"
        )

        total_scheduled = sum(i["streams_scheduled"] for i in timing_info["iterations"])
        total_attempted = sum(i["streams_in_batch"] for i in timing_info["iterations"])
        f.write(
            f"Total streams scheduled: {total_scheduled} out of {total_attempted} attempted\n"
        )
        f.write(f"Overall success rate: {total_scheduled / total_attempted:.2f}\n")
        f.write(
            f"Scheduling efficiency: {total_scheduled / len(timing_info['iterations']):.2f} streams per iteration\n"
        )
        f.write(
            f"Processing speed: {total_scheduled / timing_info['total_time']:.2f} streams per second\n\n"
        )

        f.write("Iteration Details:\n")
        for i, iter_info in enumerate(timing_info["iterations"], 1):
            f.write(f"  Iteration {i}: {iter_info['duration_seconds']:.2f}s - ")
            f.write(
                f"Scheduled {iter_info['streams_scheduled']}/{iter_info['streams_in_batch']} streams - "
            )
            f.write(f"Status: {iter_info['status']}\n")

    print(f"Timing information saved to {output_folder}")


def main():
    """Main function to run the optimization"""
    # Setup paths
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "data")
    folder_name = str(uuid.uuid4())
    input_folder = os.path.join(data_path, "input")
    output_folder = os.path.join(data_path, "output", folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # Load data
    path_manager = PathManager(input_dir=input_folder, output_dir=output_folder)
    streams: List = load_streams_from_csv(path_manager.stream_dict_output_dir)
    network_graph: nx.graph = load_graphml(path_manager.network_graphml_input_dir)
    link_config: Dict = load_yaml_file(path_manager.link_config_input_dir)
    data_factory = DataFactory(
        link_config=link_config,
        network_graph=network_graph,
        raw_streams=streams,
        k_path_count=3,
        lcm_rep=1,
    )

    # Copy input files to output directory
    input_folder = os.path.join(data_path, "input")
    for item in os.listdir(input_folder):
        s = os.path.join(input_folder, item)
        d = os.path.join(output_folder, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    def create_batches(total_count, batch_size):
        batches = []
        start = 0
        while start < total_count:
            end = min(start + batch_size, total_count)
            batches.append([x for x in range(start, end)])
            start = end
        return batches

    total_count = 60
    batch_size = 10

    batches = create_batches(total_count, batch_size)

    # Run the optimization with specified batches
    (
        status,
        variables_x,
        variables_z,
        variables_a,
        timing_info,
    ) = solve_with_batches(
        data_factory==data_factory,
        output_path=path_manager.output_dir,
        batches=batches,
        time_limit=60,
    )

    # Save timing information
    save_timing_info(timing_info, output_folder)

    # Save results
    save_as_pkl(data_factory.stream_dict, output_folder, "stream_dict.pkl")
    save_as_pkl(variables_x, output_folder, "variables_x.pkl")
    save_as_pkl(variables_z, output_folder, "variables_z.pkl")
    save_as_pkl(variables_a, output_folder, "variables_a.pkl")

    print(f"Results saved to: {folder_name}")

    # Generate output visualization
    output_manager = OutputManager(path_manager)
    output_manager.process(highlight_stream_index=None)


if __name__ == "__main__":
    main()
