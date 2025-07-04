import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def compute_edge_agreement_weights(G, lap_res, signless_res, alpha=0.5, beta=0.5):
    lap_labels = lap_res['labels']
    sig_labels = signless_res['labels']
    memberships = signless_res.get('memberships')
    edge_weights = {}

    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    if memberships is not None:
        if isinstance(memberships, dict):
            memberships_arr = np.array([memberships[n] for n in node_list])
        else:
            memberships_arr = memberships  # assumes memberships already matches node_list order
    else:
        memberships_arr = None

    for u, v in G.edges():
        w_lap = 1 if lap_labels[u] == lap_labels[v] else 0
        if memberships_arr is not None:
            i_u = node_to_idx[u]
            i_v = node_to_idx[v]
            w_sign = np.dot(memberships_arr[i_u], memberships_arr[i_v])
        else:
            w_sign = 1 if sig_labels[u] == sig_labels[v] else 0
        edge_weights[(u, v)] = -0.2 if (w_lap == 0 and w_sign == 0) else (alpha * w_lap + beta * w_sign)
    return edge_weights


def build_layout(G, lap_res, signless_res, alpha=0.5, beta=0.5, seed=42):
    """
    generate a node layout using spring layout with edge agreement weights.
    """
    ew = compute_edge_agreement_weights(G, lap_res, signless_res, alpha, beta)
    H = nx.Graph()
    for u, v in G.edges():
        H.add_edge(u, v, weight=ew[(u, v)])
    return nx.spring_layout(H, weight='weight', seed=seed), ew

def draw_dual_colored_graph(
    G, lap_res, sig_res, alpha=0.5, beta=0.5, shortcuts = None, title='lap/sig overlay', figsize=(8, 6)
):
    """
    plot a graph with nodes colored by signless clusters and node borders by laplacian clusters.
    edges style/color shows clustering agreement.
    """
    pos, ew = build_layout(G, lap_res, sig_res, alpha, beta)
    nodes = list(pos.keys())
    lap_labels = {n: lap_res['labels'][n] for n in nodes}
    sig_labels = {n: sig_res['labels'][n] for n in nodes}

    shortcuts = set() if shortcuts is None else {tuple(sorted(e)) for e in shortcuts}

    lap_clusters = sorted(set(lap_labels.values()))
    sig_clusters = sorted(set(sig_labels.values()))
    cmap_lap = cm.get_cmap('tab20', len(lap_clusters))
    cmap_sig = cm.get_cmap('Pastel1', len(sig_clusters))
    lap_colors = {c: cmap_lap(i) for i, c in enumerate(lap_clusters)}
    sig_colors = {c: cmap_sig(i) for i, c in enumerate(sig_clusters)}

    node_colors = [sig_colors[sig_labels[n]] for n in nodes]
    node_borders = [lap_colors[lap_labels[n]] for n in nodes]

    edge_colors = []
    edge_styles = []
    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        same_lap = lap_labels[u] == lap_labels[v]
        same_sig = sig_labels[u] == sig_labels[v]
        if same_lap and same_sig:
            color = 'dimgray'
            style = 'solid'
        elif same_lap or same_sig:
            color = 'lightgray'
            style = 'solid'
        else:
            color = 'lightgray'
            style = 'dotted'
        if ((u,v) in shortcuts or (v,u) in shortcuts) or G[u][v].get('contracted'):
            color = 'black'
            style = 'solid'
        edge_colors.append(color)
        edge_styles.append(style)

    fig, ax = plt.subplots(figsize=figsize)
    for (u, v), color, style in zip(G.edges(), edge_colors, edge_styles):
        if u in pos and v in pos:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, edge_color=color, style=style)

    nx.draw_networkx_nodes(
        G, pos, nodelist=nodes, ax=ax,
        node_color=node_colors,
        edgecolors=node_borders,
        linewidths=1.5, node_size=200
    )
    nx.draw_networkx_labels(G, pos, labels={n: n for n in nodes}, font_size=8, ax=ax)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor=lap_colors[c], markeredgewidth=1.5, label=f'lap {c}')
        for c in lap_clusters
    ]
    ax.legend(handles=handles, loc='upper right', title='lap clusters')
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    return ew

def compute_alpha_beta(sil_lap, sil_signless):
    """
    compute mixing weights (alpha, beta) from silhouette scores.
    fallback to 0.5/0.5 if both are zero.
    """
    sil_lap = 0.0 if sil_lap is None else sil_lap
    sil_signless = 0.0 if sil_signless is None else sil_signless
    total = sil_lap + sil_signless
    if total == 0:
        return 0.5, 0.5
    a = sil_lap / total
    b = sil_signless / total
    return a, b
