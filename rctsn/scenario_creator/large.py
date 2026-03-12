from network_generator import (
    NetworkGenerator,
    save_network_generator,
    load_network_generator,
)
import os
import uuid
import random


def create_and_connect_subnet(main_factory, subnet_id, topology_type, size, available_switches, subnet_wn_count, subnet_wed_per_wn, subnet_ed_per_sw=1):
    """Create a subnet with the specified topology and connect it to the main network.

    Args:
        main_factory: The main NetworkGenerator instance
        subnet_id: Numerical ID for the subnet
        topology_type: Type of topology ('ring', 'star', 'mesh', 'line')
        size: Number of switches in the subnet
        available_switches: List of available main network switches to connect to
        subnet_wn_count: Number of Wireless Nodes (WN) to add to the subnet.
        subnet_wed_per_wn: Number of Wireless End Devices (WED) per WN in the subnet.

    Returns:
        Tuple of (subnet object, list of subnet switches, connection point, used main switch)
    """
    print(f"Creating {topology_type} subnet with ID {subnet_id}...")
    subnet = NetworkGenerator(subnet_id=str(subnet_id))

    # Create the specified topology
    if topology_type == "ring":
        switches = subnet.create_ring(size)
        connect_from = switches[0]
    elif topology_type == "star":
        central_switch = subnet.create_star(size)
        switches = [central_switch]  # Only return the central switch for star topology
        connect_from = central_switch
    elif topology_type == "mesh":
        switches = subnet.create_mesh(size)
        connect_from = switches[0]
    elif topology_type == "line":
        switches = subnet.create_line(size)
        connect_from = switches[0]
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")

    # Add devices to the subnet using the provided counts
    subnet.add_wireless_devices(wn_count=subnet_wn_count, wed_per_wn=subnet_wed_per_wn)
    subnet.add_wired_devices(devices_per_switch=subnet_ed_per_sw)

    # Connect to main network - randomly select a main switch
    if not available_switches:
        raise ValueError("No available switches in main network to connect to")

    connect_to = random.choice(available_switches)
    main_factory.connect_subnetwork(
        subnet=subnet,
        connect_to=connect_to,
        connect_from=connect_from,
        subnetwork_name=str(subnet_id),
        link_type="0"
    )

    return subnet, switches, connect_from, connect_to


# --- Example Usage ---
if __name__ == "__main__":
    # Define relative paths for the input files and output directory
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, "output")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    folder_name = str(uuid.uuid4())

    print("Creating hybrid network topology...")
    
    # Create the main network with a ring topology (8 switches)
    main_factory = NetworkGenerator(subnet_id="0")
    main_switches = main_factory.create_ring(20)
    
    # Add wireless and wired devices to the main network
    main_factory.add_wireless_devices(wn_count=20, wed_per_wn=5)
    main_factory.add_wired_devices(devices_per_switch=5)
    
    print(f"Created main backbone ring with {len(main_switches)} switches")
    
    # Create and connect different types of subnetworks
    subnets = []
    
    # Create 4 different subnets with varying topologies
    subnet_configs = [
        {"id": 1, "type": "star", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
        {"id": 2, "type": "mesh", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
        {"id": 3, "type": "line", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
        {"id": 4, "type": "ring", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
        {"id": 5, "type": "star", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
        {"id": 6, "type": "mesh", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
        {"id": 7, "type": "line", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
        {"id": 8, "type": "ring", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
        {"id": 9, "type": "star", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
        {"id": 10, "type": "mesh", "size": 10, "wn_count": 10, "wed_per_wn": 5 , "ed_per_sw": 5},
        {"id": 11, "type": "line", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
        {"id": 12, "type": "ring", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
        {"id": 13, "type": "star", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
        {"id": 14, "type": "mesh", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
        {"id": 15, "type": "line", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
        {"id": 16, "type": "ring", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
        {"id": 17, "type": "star", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
        {"id": 18, "type": "mesh", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
        {"id": 19, "type": "line", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
        {"id": 20, "type": "star", "size": 10, "wn_count": 10, "wed_per_wn": 5, "ed_per_sw": 5},
    ]
    
    # Keep track of which main switches are already used
    # We'll use a copy of the main_switches list to ensure we don't connect
    # multiple subnets to the same main switch unless necessary
    used_main_switches = []
    
    for config in subnet_configs:
        # Prioritize unused main switches, but fall back to all if needed
        available_switches = [s for s in main_switches if s not in used_main_switches]
        if not available_switches:
            available_switches = main_switches  # Use all if all are already used once
        
        subnet, switches, connect_point, used_switch = create_and_connect_subnet(
            main_factory, 
            config["id"], 
            config["type"], 
            config["size"], 
            available_switches,
            config["wn_count"],
            config["wed_per_wn"],
            config.get("ed_per_sw", 1),  # new: per-subnet EDs-per-switch
        )
        
        subnets.append((subnet, switches, connect_point, config["id"], config["type"]))
        used_main_switches.append(used_switch)
        print(f"Connected subnet {config['id']} to main switch {used_switch}")
    
    # Add some redundant connections between subnets
    print("Adding inter-subnet connections for redundancy...")
    
    # Find switches in different subnets to connect
    for i in range(len(subnets)):
        for j in range(i+1, len(subnets)):
            if random.random() < 0.5:  # 50% chance to add a connection between subnets
                subnet_i_id = subnets[i][3]
                subnet_j_id = subnets[j][3]
                
                # Find a switch in each subnet that isn't the connection point
                switch_i = next((n for n in main_factory.network.nodes() 
                               if f"{subnet_i_id}-SW" in n and n != subnets[i][2]), None)
                switch_j = next((n for n in main_factory.network.nodes() 
                               if f"{subnet_j_id}-SW" in n and n != subnets[j][2]), None)
                
                if switch_i and switch_j:
                    main_factory._add_edge(switch_i, switch_j)
                    print(f"Added connection between subnet {subnet_i_id} and subnet {subnet_j_id}")
    
    # Add additional connections between main network and subnets for redundancy
    print("Adding additional connections between main network and subnets...")
    
    for subnet_info in subnets:
        subnet_id = subnet_info[3]
        subnet_switches = [n for n in main_factory.network.nodes() 
                        if f"{subnet_id}-SW" in n and n != subnet_info[2]]
        
        # Try to find an unused main switch for the additional connection
        available_main = [s for s in main_switches if s not in used_main_switches]
        if not available_main:
            available_main = main_switches  # Use any if all are already used
            
        if subnet_switches and available_main and random.random() < 0.5:
            additional_main = random.choice(available_main)
            additional_subnet = random.choice(subnet_switches)
            main_factory._add_edge(additional_main, additional_subnet)
            used_main_switches.append(additional_main)
            print(f"Added redundant connection between main network and subnet {subnet_id}")
    
    print(f"Network topology created with {main_factory.network.number_of_nodes()} nodes and {main_factory.network.number_of_edges()} links")
    
    # Define stream settings using dictionaries for better clarity
    # We explicitly specify the latency factors, gamma values, and jitter factors
    wed_wed_settings = {
        "packet_sizes": [32, 64, 128, 256, 512, 1024, 1420],
        "periods": [8000],
        "latency_factors": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "priorities": [1, 2, 3],
        "gammas": [1.0],
        "jitter_factors": [0.1, 0.2, 0.3, 0.4, 0.5]  # New parameter for jitter as a ratio of period
    }
    
    ed_ed_settings = {
        "packet_sizes": [32, 64, 128, 256, 512, 1024, 1420],
        "periods": [500, 1000, 2000, 4000, 8000],
        "latency_factors": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "priorities": [1, 2, 3],
        "gammas": [1.0],
        "jitter_factors": [0.1, 0.2, 0.3, 0.4, 0.5]  # New parameter for jitter as a ratio of period
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
    print("Generating streams...")
    try:


        # 1) Within-subnet streams
        main_factory.generate_streams_with_dict_settings(
            wed_to_wed=5000,
            ed_to_ed=5000,
            wed_to_ed=5000,
            mode="subnet",
            wed_wed_settings=wed_wed_settings,
            ed_ed_settings=ed_ed_settings,
            wed_ed_settings=wed_ed_settings,
            shuffle_streams=True,
            shuffle_scope="all",
        )
        # 2) Cross-subnet streams
        main_factory.generate_streams_with_dict_settings(
            wed_to_wed=1000,
            ed_to_ed=1000,
            wed_to_ed=1000,
            mode="any",
            wed_wed_settings=wed_wed_settings,
            ed_ed_settings=ed_ed_settings,
            wed_ed_settings=wed_ed_settings,
            shuffle_streams=True,
            shuffle_scope="all",  
        )
    except ValueError as e:
        print(f"Stream generation error: {e}")

    # Output number of streams to the console
    print(f"Generated {len(main_factory.streams)} streams")

    # Create output folder
    output_folder = os.path.join(output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # Export streams to CSV
    main_factory.export_streams(
        csv_file=os.path.join(output_folder, "streams.csv")
    )

    # Export network to GraphML
    main_factory.export(output_dir=output_folder)
    
    # Calculate statistics
    node_types = {"SW": 0, "WN": 0, "WED": 0, "ED": 0}
    for node in main_factory.network.nodes():
        node_type = main_factory._get_core_type(node)
        if node_type in node_types:
            node_types[node_type] += 1
            
    # Count links by type
    link_types = {}
    for u, v, data in main_factory.network.edges(data=True):
        link_type = data.get("link_type", "unknown")
        link_types[link_type] = link_types.get(link_type, 0) + 1
    
    # Count nodes by subnet
    subnet_counts = {}
    for node, attrs in main_factory.network.nodes(data=True):
        subnet = attrs.get('subnet', 'unknown')
        if subnet not in subnet_counts:
            subnet_counts[subnet] = 0
        subnet_counts[subnet] += 1

    # Save network statistics to a text file
    stats_file = os.path.join(output_folder, "network_stats.txt")
    with open(stats_file, "w") as f:
        f.write("Hybrid Network Topology Statistics\n")
        f.write("================================\n\n")
        
        f.write("Network Structure:\n")
        f.write("- Main Backbone: Ring topology (8 switches) - Subnet 0\n")
        for i, config in enumerate(subnet_configs):
            f.write(f"- Subnet {config['id']}: {config['size']} switches in {config['type']} configuration\n")
        
        f.write("\nConnectivity:\n")
        for i, (_, _, _, subnet_id, topology) in enumerate(subnets):
            f.write(f"- Subnet {subnet_id} ({topology}) connected to main backbone\n")
            
        f.write("\nNodes:\n")
        f.write(f"  Total nodes: {main_factory.network.number_of_nodes()}\n")
        for node_type, count in node_types.items():
            f.write(f"  {node_type} nodes: {count}\n")
        
        f.write("\nSubnetworks:\n")
        for subnet, count in subnet_counts.items():
            f.write(f"  Subnet {subnet}: {count} nodes\n")
        
        f.write("\nLinks:\n")
        f.write(f"  Total links: {main_factory.network.number_of_edges()}\n")
        for link_type, count in link_types.items():
            f.write(f"  {link_type} links: {count}\n")
        
        f.write("\nStreams:\n")
        f.write(f"  Total streams: {len(main_factory.streams)}\n")
        
        # Add jitter settings info
        f.write("\nJitter Settings:\n")
        f.write(f"  WED-WED jitter factors: {wed_wed_settings['jitter_factors']}\n")
        f.write(f"  ED-ED jitter factors: {ed_ed_settings['jitter_factors']}\n")
        f.write(f"  WED-ED jitter factors: {wed_ed_settings['jitter_factors']}\n")

    # Save the NetworkGenerator object
    save_network_generator(
        main_factory, os.path.join(output_folder, "network_generator.pkl")
    )
    
    # Visualize the network
    print("Visualizing network topology...")
    main_factory.visualize(save_path=os.path.join(output_folder, "network.png"))

    # Print final statistics
    print(f"Network exported to {output_dir}/{folder_name}")
    print(f"Number of nodes: {main_factory.network.number_of_nodes()}")
    print(f"Number of links: {main_factory.network.number_of_edges()}")
    print(f"Number of streams: {len(main_factory.streams)}")
    print(f"Network statistics saved to {stats_file}")
