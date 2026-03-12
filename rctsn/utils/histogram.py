import numpy as np
import matplotlib.pyplot as plt
import yaml  # Added for YAML loading
import os    # Added for path joining

def load_histogram_config(filepath):
    """Loads histogram configuration from a YAML file."""
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Histogram config file not found at {filepath}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {filepath}: {e}")
        return None

def generate_random_from_histogram(histogram, bin_values, num_samples=1):
    """
    Generates random numbers by sampling from discrete values weighted by histogram frequencies.

    Args:
        histogram (array-like): The frequency counts for each discrete value.
        bin_values (array-like): The discrete values corresponding to each frequency count.
                                 Must have the same length as histogram.
        num_samples (int, optional): The number of random samples to generate. Defaults to 1.

    Returns:
        numpy.ndarray: An array of random numbers sampled from the provided discrete values,
                       or None if inputs are invalid.
    """
    # Input validation
    if not isinstance(histogram, (list, np.ndarray)) or not isinstance(bin_values, (list, np.ndarray)):
         print("Error: histogram and bin_values must be array-like.")
         return None
    if len(histogram) == 0:
        print("Error: Histogram data is empty.")
        return None
    # --- User Requirement: Lengths must be equal ---
    if len(histogram) != len(bin_values):
        print(f"Error: Mismatch between histogram length ({len(histogram)}) and bin_values length ({len(bin_values)}). Lengths must be equal.")
        return None
    # Ensure histogram contains numeric types and sum is positive
    try:
        hist_sum = np.sum(histogram)
        if hist_sum <= 0:
            print("Error: Sum of histogram frequencies must be positive.")
            return None
    except (TypeError, ValueError):
         print("Error: Histogram data must contain numeric values.")
         return None

    # Convert inputs to NumPy arrays
    hist = np.array(histogram)
    values = np.array(bin_values)

    # Calculate the probability of selecting each value
    probabilities = hist / hist_sum

    # Ensure probabilities sum to 1 (handle potential floating point inaccuracies)
    probabilities /= probabilities.sum()

    # Use NumPy's choice function to randomly select VALUES based on their probabilities
    random_samples = np.random.choice(values, size=num_samples, p=probabilities)

    return random_samples

if __name__ == '__main__':
    # --- Configuration Loading ---
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, 'data/histogram_config.yaml')
    histogram_config = load_histogram_config(config_path)

    if histogram_config:
        # --- Select Link Type and Packet Size ---
        link_type_to_test = '1'  # Choose link type '1'
        packet_size_to_test = '32' # Choose packet size '32', '1420' (as strings)

        print(f"Attempting to test Link Type: {link_type_to_test}, Packet Size: {packet_size_to_test}")

        # --- Access Nested Data ---
        if ('link_types' in histogram_config and
                link_type_to_test in histogram_config['link_types'] and
                'packet_sizes' in histogram_config['link_types'][link_type_to_test] and
                packet_size_to_test in histogram_config['link_types'][link_type_to_test]['packet_sizes']):

            packet_size_data = histogram_config['link_types'][link_type_to_test]['packet_sizes'][packet_size_to_test]

            # Rename 'bin_boundaries' to 'bin_values' for clarity in this context
            if 'histogram_data' in packet_size_data and 'bin_boundaries' in packet_size_data:
                histogram_data = packet_size_data['histogram_data']
                bin_values = packet_size_data['bin_boundaries'] # Treat as representative values
                description = packet_size_data.get('description', f'Link Type {link_type_to_test} - {packet_size_to_test} Bytes')

                print(f"Using histogram for: {description}")
                print(f"Histogram Data: {histogram_data}")
                print(f"Bin Values: {bin_values}") # Changed print label

                # Generate 1000 random samples from the loaded distribution
                # Pass bin_values instead of bin_boundaries
                random_output = generate_random_from_histogram(histogram_data, bin_values, num_samples=1000)

                if random_output is not None:
                    # --- Visualization ---
                    plt.figure(figsize=(12, 6))

                    # --- Plot 1: Histogram of Generated Samples ---
                    plt.subplot(1, 2, 1) # Create subplot 1
                    # Determine appropriate bins for visualizing the generated discrete samples
                    # One way is to center bins around the unique values generated
                    unique_vals = np.unique(random_output)
                    if len(unique_vals) > 1:
                        # Create edges halfway between unique values
                        bin_edges_vis = (unique_vals[:-1] + unique_vals[1:]) / 2
                        # Add edges at the start and end based on spacing
                        start_edge = unique_vals[0] - (bin_edges_vis[0] - unique_vals[0]) if len(bin_edges_vis) > 0 else unique_vals[0] - 0.5
                        end_edge = unique_vals[-1] + (unique_vals[-1] - bin_edges_vis[-1]) if len(bin_edges_vis) > 0 else unique_vals[-1] + 0.5
                        vis_bins = np.concatenate(([start_edge], bin_edges_vis, [end_edge]))
                    elif len(unique_vals) == 1:
                        # If only one value, create a small bin around it
                        vis_bins = [unique_vals[0] - 0.5, unique_vals[0] + 0.5]
                    else:
                        vis_bins = 10 # Default fallback

                    plt.hist(random_output, bins=vis_bins, density=True, alpha=0.7, label='Generated Random Data (Density)')
                    plt.xlabel('Value (e.g., Time in microseconds)')
                    plt.ylabel('Density')
                    plt.title('Histogram of Generated Samples')
                    plt.legend()
                    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

                    # --- Plot 2: Original Probability Mass Function (PMF) ---
                    plt.subplot(1, 2, 2) # Create subplot 2
                    if len(histogram_data) > 0 and len(bin_values) == len(histogram_data):
                        # Calculate probabilities
                        probabilities = np.array(histogram_data) / np.sum(histogram_data)
                        # Use stem plot to show probability at each discrete value
                        plt.stem(bin_values, probabilities, basefmt=" ", label='Original PMF', linefmt='C1-', markerfmt='C1o')
                        plt.xlabel('Value (e.g., Time in microseconds)')
                        plt.ylabel('Probability')
                        plt.title('Original Probability Distribution')
                        plt.legend()
                        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                    else:
                        print("Warning: Cannot plot original PMF due to invalid data/values.")

                    plt.suptitle(f'Distribution Analysis ({description})') # Overall title
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
                    plt.show()
                else:
                    print("Failed to generate random samples.")
            else:
                # Adjusted error message key
                print(f"Error: 'histogram_data' or 'bin_boundaries' (interpreted as bin_values) missing for link type '{link_type_to_test}', packet size '{packet_size_to_test}' in {config_path}")
        else:
            print(f"Error: Link type '{link_type_to_test}' or packet size '{packet_size_to_test}' not found or structure incorrect in {config_path}")
    else:
        print("Could not load histogram configuration.")