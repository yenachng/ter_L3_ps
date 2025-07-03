import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.spatial.distance import euclidean

# get a color vector for nodes based on their cluster labels
def get_color_vector(node_order, label_dict, cmap_name='tab20'):
    cmap = cm.get_cmap(cmap_name, max(label_dict.values()) + 1)
    return [cmap(label_dict.get(node, 0)) if node in label_dict else "lightgray" for node in node_order]

# build a layout based on cluster memberships or hard labels
def cluster_aware_layout(G, memberships=None, labels=None, scale=1.0, weight_factor=1.0, sigma=0.5, seed=42):
    H = nx.Graph()
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    for u, v in G.edges():
        if memberships is not None:
            dist = euclidean(memberships[node_idx[u]], memberships[node_idx[v]])
            sim = np.exp(-dist ** 2 / (2 * sigma ** 2))
            weight = weight_factor * sim
        elif labels is not None:
            weight = weight_factor if labels[u] == labels[v] else 0.1
        else:
            weight = 1.0
        H.add_edge(u, v, weight=weight)

    pos = nx.spring_layout(H, weight='weight', scale=scale, seed=seed)
    return pos

# draw graph nodes and edges colored by clusters, label nodes
def draw_graph_with_labels(G, labels, memberships=None, title=None, pos=None):
    if pos is None:
        if memberships is not None:
            pos = cluster_aware_layout(G, memberships=memberships, scale=1.0, weight_factor=1, seed=42)
        else:
            pos = cluster_aware_layout(G, labels=labels, scale=1.0, weight_factor=1, seed=42)

    cluster_ids = sorted(set(labels.values()))
    colors = cm.get_cmap('tab20', len(cluster_ids))(range(len(cluster_ids)))
    color_map = {cid: colors[i] for i, cid in enumerate(cluster_ids)}

    plt.figure(figsize=(10, 8))
    for cid in cluster_ids:
        nodes = [n for n, l in labels.items() if l == cid]
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes, node_color=[color_map[cid]],
            label=f'cluster {cid+1}', node_size=200, alpha=0.95
        )
    intra_edges = [(u, v) for u, v in G.edges() if labels[u] == labels[v]]
    inter_edges = [(u, v) for u, v in G.edges() if labels[u] != labels[v]]

    if intra_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=intra_edges,
            edge_color="dimgray", style="solid", alpha=0.7, width=2
        )
    if inter_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=inter_edges,
            edge_color="lightgray", style="dotted", alpha=0.8, width=2
        )
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")
    plt.title(title or "clustered graph")
    plt.axis('off')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
