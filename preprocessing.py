from itertools import combinations
from typing import List, Tuple

import numpy as np

def process_dataset(path: str, k: int) -> Tuple[int, List[List], List[List]]:
    """
    Takes the path to the network dataset and returns hyperedges of a k-uniform.
    Nodes form a k-hyperedge is there are at least k connections between them

    Parameters
    ----------
    path: path to the dataset file
    k: numer of nodes in a hyperedge

    Returns
    -------
    N: the humber of nodes in the hypergraph
    2 lists of lists containing hyperedges and nodes of a k-uniform hypergraph and 
    the complement hypergraph
    """
    with open(path) as file:
        edges = [tuple(map(int, line.rstrip()[2:].split(' '))) for line in file if line[0] == 'e']

    old_node_index = list({i for edge in edges for i in edge})
    old_node_index.sort()
    n = len(old_node_index)
    new_node_index = np.arange(n)
    old_to_new = dict(zip(old_node_index, new_node_index))

    def inner_map(edge, dict):
        l = [dict[node] for node in edge]
        l.sort()
        return tuple(l)

    diadic_connections = {inner_map(edge, old_to_new) for edge in edges}
    if k == 2:
        complement_edges = []
        for comb in combinations(np.arange(n).tolist(), 2):
            if comb in diadic_connections:
                pass
            else:
                complement_edges.append(list(comb))
        return n, list(diadic_connections), complement_edges
    else:
        hyperedges = []
        complement_hyperedges = []
        for comb in combinations(np.arange(n).tolist(), k):
            connection_count = 0
            for connection in combinations(comb, 2):
                if connection in diadic_connections:
                    connection_count += 1
            if connection_count >= k:
                hyperedges.append(comb)
            else:
                complement_hyperedges.append(comb)
        return n, hyperedges, complement_hyperedges