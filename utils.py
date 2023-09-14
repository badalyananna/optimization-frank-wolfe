from itertools import combinations
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    
def plot_hisotry(df: pd.DataFrame, seed: int, ssc: Optional[bool]=False):
    """The function to plot the history of the specific run of the algorithm based on seed.
    The plots the ssc=True plot the dashed line for the run with SSC procedure."""
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex='col', sharey='row')
    dfs = df[df.seed==seed]
    assert len(dfs) > 0, "The dataframe is empty. Probably the seed doesn't exist"
    for i in range(2):
        for j in range(2):
            ax = axes[j, i]
            x = 'cpu_time' if i == 0 else 'iteration'
            y = 'of_value' if j == 0 else 'duality_gap'
            x_name = 'CPU time' if i == 0 else 'Iterations'
            y_name = 'O.F. value' if j == 0 else 'Duality Gap'
            if ssc:
                sns.lineplot(data=dfs, x=x, y=y, hue='Variant', style='SSC', ax=ax)
            else:
                sns.lineplot(data=dfs, x=x, y=y, hue='Variant', ax=ax)
            ax.set_yscale('log')
            ax.spines[['right', 'top']].set_visible(False)
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            ax.set_ylabel(y_name)
            ax.set_xlabel(x_name)
    if ssc:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
        fig.suptitle('Performance of FW variants with and without SSC procedure', fontsize=14, y=1)
    else:
        fig.legend(handles, labels, title='Variants', loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
        fig.suptitle('Performance of FW variants', fontsize=14, y=1)

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()