from itertools import islice
import networkx as nx


def k_shortest_paths(G, source, target, k, weight=None):

    path_list = []
    for path in list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    ):
        edges_list = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

        path_list.append(edges_list)
        print(edges_list)
    return path_list
