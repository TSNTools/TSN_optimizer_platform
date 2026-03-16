from utils.helper import (
    load_graphml,
    load_streams_from_csv,
    load_yaml_file,
    save_as_graphml,
    save_as_pkl,
)
path_streams="/home/ws/TSN_optimizer_platform/rctsn/data/input/streams.csv"
path_graphml="/home/ws/TSN_optimizer_platform/rctsn/data/input/network.graphml"




if __name__ == "__main__":
    streams = load_streams_from_csv(path_streams) 
    network_graph = load_graphml(path_graphml)
    print(type(network_graph))
    print(type(streams))