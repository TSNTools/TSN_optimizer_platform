from network_generator import (
    NetworkGenerator,
    save_network_generator,
    load_network_generator,
)
import os
import uuid


# --- Example Usage ---
if __name__ == "__main__":

    # Define relative paths for the input files and output directory
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, "output")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    folder_name = str(uuid.uuid4())

    # Create a single network with a ring topology (5 switches)
    factory = NetworkGenerator(subnet_id="0")

    # Create a ring with 5 switches
    ring_switches = factory.create_line(2)
    
    # Add wireless and wired devices to the network
    # - More wireless nodes with end devices
    factory.add_wireless_devices(wn_count=0, wed_per_wn=0)
    # - More wired devices to the switches
    factory.add_wired_devices(devices_per_switch=1)

    # Define stream settings using dictionaries for better clarity
    # We explicitly specify the latency factors, gamma values, and jitter factors
    # NOTE: This small scenario is wired-only (no WN/WED devices).
    wed_to_wed_count = 0
    ed_to_ed_count = 10
    wed_to_ed_count = 0

    wed_wed_settings = {
        "packet_sizes": [32, 64, 128, 256, 512, 1024, 1420],
        "periods": [8000],
        "latency_factors": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "priorities": [1, 2, 3],
        "gammas": [1.0],
        "jitter_factors": [0.1, 0.2, 0.3, 0.4, 0.5]  # New parameter for jitter as a ratio of period
    }
    
    ed_ed_settings = {
        "packet_sizes": [1024],
        "periods": [180],
        "latency_factors": [0.5, 0.8],
        "priorities": [1],
        "gammas": [1.0],
        "jitter_factors": [0]  # New parameter for jitter as a ratio of period
    }
    
    wed_ed_settings = {
        "packet_sizes": [32, 64, 128, 256, 512, 1024, 1420],
        "periods": [4000, 8000],
        "latency_factors": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "priorities": [1, 2, 3],
        "gammas": [1.0], 
        "jitter_factors": [0.1, 0.2, 0.3, 0.4, 0.5] 
    }

    # Generate streams across the entire network
    try:
        factory.generate_streams_with_dict_settings(
            wed_to_wed=wed_to_wed_count,
            ed_to_ed=ed_to_ed_count,
            wed_to_ed=wed_to_ed_count,
            mode="any",
            wed_wed_settings=wed_wed_settings,
            ed_ed_settings=ed_ed_settings,
            wed_ed_settings=wed_ed_settings,
            shuffle_streams=True
        )
    except ValueError as e:
        print(f"Stream generation error: {e}")

    # Output number of streams to the console
    print(f"Generated {len(factory.streams)} streams")

    # Export streams to CSV
    factory.export_streams(
        csv_file=os.path.join(output_dir, folder_name, "streams.csv")
    )

    # Export and visualize the network
    factory.export(output_dir=os.path.join(output_dir, folder_name))
    factory.visualize(save_path=os.path.join(output_dir, folder_name, "network.png"))

    # Save network statistics to a text file
    stats_file = os.path.join(output_dir, folder_name, "network_stats.txt")
    extra_info = {
        "WED-WED stream count": wed_to_wed_count,
        "ED-ED stream count": ed_to_ed_count,
        "WED-ED stream count": wed_to_ed_count,
        "WED-WED jitter factors": wed_wed_settings['jitter_factors'],
        "ED-ED jitter factors": ed_ed_settings['jitter_factors'],
        "WED-ED jitter factors": wed_ed_settings['jitter_factors']
    }
    factory.save_network_stats(stats_file, "Ring Topology Network Statistics", extra_info)

    # Save the NetworkGenerator object
    save_network_generator(
        factory, os.path.join(output_dir, folder_name, "network_generator.pkl")
    )

    # Print final statistics
    print(f"Network exported to {output_dir}/{folder_name}")
    print(f"Number of nodes: {factory.network.number_of_nodes()}")
    print(f"Number of links: {factory.network.number_of_edges()}")
    print(f"Number of streams: {len(factory.streams)}")
    print(f"Network statistics saved to {stats_file}")
    
