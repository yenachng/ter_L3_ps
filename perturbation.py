import networkx as nx
from collections import defaultdict, Counter
from itertools import combinations
import random
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import subspace_angles
from contraction import full_contraction_ham
from dual_clus import draw_dual_colored_graph

def weighted_imbalance_perturbation(G, edge_weights=None, lap_labels=None, num_nodes_per_cluster=3,
                                    keep=2, min_frac=0.7, weight_attr='weight', nodes_to_perturb=None, **kwargs):
    """
    Remove weaker edges from selected nodes, keeping at most `keep` strongest edges per node.
    For each target node, edges with weight less than `min_frac * max_edge_weight` (for that node) 
    are removed, as long as each endpoint's degree stays above `keep`.
    If `nodes_to_perturb` is not provided, it selects up to `num_nodes_per_cluster` nodes per cluster (using `lap_labels`) 
    that have exactly two relatively strong connections (≥ `min_frac` of their maximum edge weight).
    """
    Gp = G.copy()
    # determine which nodes to perturb if not explicitly given
    if nodes_to_perturb is None:
        nodes_to_perturb = []
        if lap_labels is not None:
            # group nodes by cluster label
            cluster_map = defaultdict(list)
            for node, label in lap_labels.items():
                cluster_map[label].append(node)
            # for each cluster, find candidate nodes with exactly two strong edges
            for cluster_nodes in cluster_map.values():
                cand_nodes = []
                for n in cluster_nodes:
                    nbrs = list(G.neighbors(n))
                    if len(nbrs) < 2:
                        continue
                    # determine edge weights for all neighbors of n
                    weights = []
                    for v in nbrs:
                        e = (min(n, v), max(n, v))
                        if edge_weights and e in edge_weights:
                            w = edge_weights[e]
                        else:
                            w = G[n][v].get(weight_attr, 1.0)
                        weights.append(w)
                    max_w = max(weights)
                    # identify neighbors with "strong" edges relative to max_w
                    strong_nbrs = [v for v, w in zip(nbrs, weights) if w >= min_frac * max_w]
                    if len(strong_nbrs) == 2:
                        cand_nodes.append(n)
                if not cand_nodes:
                    continue
                # randomly select a subset of candidates to perturb
                random.shuffle(cand_nodes)
                sel_num = min(num_nodes_per_cluster, max(0, len(cand_nodes) - 3))
                if sel_num > 0:
                    selected = random.sample(cand_nodes, sel_num)
                    nodes_to_perturb.extend(selected)
        else:
            # if no cluster info, consider all nodes with exactly two strong edges
            for n in G.nodes():
                nbrs = list(G.neighbors(n))
                if len(nbrs) < 2:
                    continue
                weights = []
                for v in nbrs:
                    e = (min(n, v), max(n, v))
                    if edge_weights and e in edge_weights:
                        w = edge_weights[e]
                    else:
                        w = G[n][v].get(weight_attr, 1.0)
                    weights.append(w)
                max_w = max(weights)
                strong_nbrs = [v for v, w in zip(nbrs, weights) if w >= min_frac * max_w]
                if len(strong_nbrs) == 2:
                    nodes_to_perturb.append(n)
    # perform perturbation: prune weaker edges from each selected node
    for n in nodes_to_perturb:
        neighbors = list(Gp.neighbors(n))
        if len(neighbors) <= keep:
            continue
        # compute weights of all edges from n
        edge_w_list = []
        for v in neighbors:
            e = (min(n, v), max(n, v))
            if edge_weights and e in edge_weights:
                w = edge_weights[e]
            else:
                w = Gp[n][v].get(weight_attr, 1.0)
            edge_w_list.append(w)
        # determine top-`keep` edges by weight
        if not edge_w_list:  # no neighbors
            continue
        sorted_idx = sorted(range(len(edge_w_list)), key=lambda i: edge_w_list[i], reverse=True)
        top_idx_set = set(sorted_idx[:keep])
        max_w = edge_w_list[sorted_idx[0]]
        # remove edges that are not in top keep and below weight threshold
        for i, v in enumerate(neighbors):
            if i not in top_idx_set and edge_w_list[i] < max_w * min_frac:
                if Gp.has_edge(n, v) and Gp.degree(n) > keep and Gp.degree(v) > keep:
                    Gp.remove_edge(n, v)
    return Gp

def perturbation_by_levels(G, contracted_edges, lap_labels, sig_labels, edge_weights=None, min_deg=2, print_stats=True, **kwargs):
    """
    Iteratively remove edges that are not part of strong connections or provided 'contracted_edges'.
    In each iteration, identify nodes with exactly two 'strong' incident edges (either in `contracted_edges` or connecting nodes within the same cluster in both labelings).
    Remove other edges from those nodes if it can be done without dropping node degree below `min_deg`.
    Repeat until no further changes occur.
    """
    Gp = G.copy()
    contracted_set = {tuple(sorted(e)) for e in contracted_edges}
    round_num = 0
    while True:
        changed = False
        # level 1: nodes with exactly 2 incident contracted edges
        contracted_count = Counter()
        for u, v in Gp.edges():
            e = tuple(sorted((u, v)))
            if e in contracted_set:
                contracted_count[u] += 1
                contracted_count[v] += 1
        level1_nodes = [n for n, count in contracted_count.items() if count == 2]
        # level 2: nodes with exactly 2 incident "strong" edges (either contracted or within-cluster edges), excluding level1 nodes
        strong_edges = []
        for u, v in Gp.edges():
            e = tuple(sorted((u, v)))
            same_cluster = (lap_labels.get(u) == lap_labels.get(v) and sig_labels.get(u) == sig_labels.get(v))
            if e in contracted_set or same_cluster:
                strong_edges.append(e)
        strong_count = Counter()
        for u, v in strong_edges:
            strong_count[u] += 1
            strong_count[v] += 1
        level2_nodes = [n for n, count in strong_count.items() if count == 2 and n not in level1_nodes]
        if print_stats:
            print(f"round {round_num}: level1_nodes={level1_nodes}, level2_nodes={level2_nodes}")
        # remove non-contracted edges from level1 nodes
        for n in level1_nodes:
            nbrs = list(Gp.neighbors(n))
            # identify the two contracted neighbors (if exactly 2)
            contracted_nbrs = [v for v in nbrs if tuple(sorted((n, v))) in contracted_set]
            if len(contracted_nbrs) == 2:
                for v in nbrs:
                    e = tuple(sorted((n, v)))
                    if v not in contracted_nbrs and e not in contracted_set:
                        if Gp.has_edge(n, v) and Gp.degree(n) > min_deg and Gp.degree(v) > min_deg:
                            Gp.remove_edge(n, v)
                            changed = True
        # remove non-strong edges from level2 nodes
        for n in level2_nodes:
            nbrs = list(Gp.neighbors(n))
            # identify two strongest neighbors (prefer contracted edges, otherwise any strong edges)
            strong_nbrs = [v for v in nbrs if tuple(sorted((n, v))) in contracted_set or 
                           (lap_labels.get(n) == lap_labels.get(v) and sig_labels.get(n) == sig_labels.get(v))]
            if len(strong_nbrs) >= 2:
                # consider all pairs of strong neighbors and choose the pair to keep that minimizes removal of contracted edges
                best_pair = None
                best_score = None
                for u, v in combinations(strong_nbrs, 2):
                    e1 = tuple(sorted((n, u))); e2 = tuple(sorted((n, v)))
                    # score 0 if any edge is contracted (prefer keeping contracted edges), otherwise 1
                    score = 0 if (e1 in contracted_set or e2 in contracted_set) else 1
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pair = {u, v}
                keep_set = best_pair if best_pair is not None else set()
                for v in nbrs:
                    e = tuple(sorted((n, v)))
                    if v not in keep_set and e not in contracted_set:
                        if Gp.has_edge(n, v) and Gp.degree(n) > min_deg and Gp.degree(v) > min_deg:
                            Gp.remove_edge(n, v)
                            changed = True
        round_num += 1
        if not changed:
            break
    return Gp

def random_edge_perturbation(G, fraction=0.1, min_deg=1, **kwargs):
    """
    Randomly remove a fraction of edges from the graph.
    Chooses approximately `fraction * 100%` of edges at random to delete, ensuring no node's degree falls below `min_deg`.
    """
    Gp = G.copy()
    edges = list(Gp.edges())
    random.shuffle(edges)
    num_to_remove = int(len(edges) * fraction)
    removed = 0
    for u, v in edges:
        if removed >= num_to_remove:
            break
        if Gp.has_edge(u, v) and Gp.degree(u) > min_deg and Gp.degree(v) > min_deg:
            Gp.remove_edge(u, v)
            removed += 1
    return Gp

def inter_cluster_perturbation(G, lap_labels, sig_labels, min_deg=1, **kwargs):
    """
    Remove edges that connect nodes in different clusters.
    Drops edges where the two endpoints do not share the same cluster label in both `lap_labels` and `sig_labels`.
    Ensures no node's degree falls below `min_deg`.
    """
    Gp = G.copy()
    for u, v in list(Gp.edges()):
        if lap_labels.get(u) != lap_labels.get(v) or sig_labels.get(u) != sig_labels.get(v):
            if Gp.degree(u) > min_deg and Gp.degree(v) > min_deg:
                Gp.remove_edge(u, v)
    return Gp

def intra_cluster_perturbation(G, lap_labels, sig_labels, min_deg=1, **kwargs):
    """
    Remove edges that connect nodes within the same cluster.
    Drops edges where both `lap_labels` and `sig_labels` assign the same cluster to both endpoints.
    Ensures no node's degree falls below `min_deg`.
    """
    Gp = G.copy()
    for u, v in list(Gp.edges()):
        if lap_labels.get(u) == lap_labels.get(v) and sig_labels.get(u) == sig_labels.get(v):
            if Gp.degree(u) > min_deg and Gp.degree(v) > min_deg:
                Gp.remove_edge(u, v)
    return Gp

def preserve_cycle_perturbation(G, contracted_edges, min_deg=2, **kwargs):
    """
    Preserve a set of important edges (e.g., a Hamiltonian cycle) and remove other edges.
    Only edges not in `contracted_edges` are considered for removal, and removals ensure node degrees stay >= `min_deg`.
    """
    Gp = G.copy()
    important_set = {tuple(sorted(e)) for e in contracted_edges}
    for u, v in list(Gp.edges()):
        if tuple(sorted((u, v))) not in important_set:
            if Gp.degree(u) > min_deg and Gp.degree(v) > min_deg:
                Gp.remove_edge(u, v)
    return Gp

def break_cycle_perturbation(G, contracted_edges, min_deg=1, **kwargs):
    """
    Remove a specified set of important edges (e.g., edges in a Hamiltonian cycle) from the graph.
    Only edges in `contracted_edges` are removed, and removals ensure node degrees stay >= `min_deg` (to avoid isolating nodes entirely).
    """
    Gp = G.copy()
    important_set = {tuple(sorted(e)) for e in contracted_edges}
    for u, v in list(Gp.edges()):
        if tuple(sorted((u, v))) in important_set:
            if Gp.degree(u) > min_deg and Gp.degree(v) > min_deg:
                Gp.remove_edge(u, v)
    return Gp

def align_embeddings(G1, G2, vecs1, vecs2):
    """
    Align eigenvector embeddings of two graphs by reordering to match common node indices.
    Returns the eigenvector matrices for the common nodes in the order of `G1` and `G2`.
    """
    nodes1 = np.array(list(G1.nodes()))
    nodes2 = np.array(list(G2.nodes()))
    common_nodes = sorted(set(nodes1) & set(nodes2))
    idx1 = [np.where(nodes1 == n)[0][0] for n in common_nodes]
    idx2 = [np.where(nodes2 == n)[0][0] for n in common_nodes]
    return vecs1[idx1], vecs2[idx2]

def get_laplacian_subspace(G, k=5, normalized=True):
    """
    Compute the smallest k eigenvalues and corresponding eigenvectors of the Laplacian of graph G.
    Returns eigenvalues (array) and eigenvectors (matrix) sorted in ascending order.
    If `normalized` is True, uses the normalized Laplacian; otherwise uses the combinatorial Laplacian.
    """
    L = nx.normalized_laplacian_matrix(G) if normalized else nx.laplacian_matrix(G)
    L = L.astype(np.float64)
    n = L.shape[0]
    # ensure k is less than n (Laplacian has at least one zero eigenvalue)
    k = min(k, n - 1)
    vals, vecs = eigsh(L, k=k, which='SM')
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]

def max_principal_angle(X1, X2):
    """
    Compute the maximum principal angle (in degrees) between two subspaces spanned by columns of X1 and X2.
    Uses the largest angle between the subspaces spanned by the columns of X1 and X2.
    """
    angles = subspace_angles(X1, X2)
    return np.degrees(np.max(angles)) if angles.size > 0 else 0.0

def spectrum_diff(evals1, evals2):
    """
    Compute the maximum absolute difference between two sets of eigenvalues.
    If the lengths differ, the shorter list is zero-padded.
    """
    n = max(len(evals1), len(evals2))
    e1 = np.pad(evals1, (0, n - len(evals1)), 'constant', constant_values=0)
    e2 = np.pad(evals2, (0, n - len(evals2)), 'constant', constant_values=0)
    return float(np.max(np.abs(e1 - e2)))

def analyze_perturbations(all_data, perturb_func, func_name="custom", num_nodes_per_cluster=3,
                          keep=2, n_trials=5, angle_threshold=10, perturb_kwargs=None):
    """
    Run perturbation analysis on a collection of graph data using a given perturbation function.

    Parameters:
    - all_data: list of data dicts, each containing at least 'graph', 'lap_results' (with 'labels'), 'sig_results' (with 'labels'),
                'all_chains' (contracted edges), and optionally 'edge_weights', 'alpha', 'beta'.
    - perturb_func: function to apply for perturbation (one of the perturbation functions defined above).
    - func_name: a label for the perturbation function (for display purposes).
    - num_nodes_per_cluster, keep: parameters used by some perturbation methods (e.g., weighted imbalance).
    - n_trials: number of random trials to run for each graph (for methods with randomness).
    - angle_threshold: threshold (in degrees) to count significant changes (not used explicitly, could be for filtering results).
    - perturb_kwargs: additional keyword arguments to pass to the perturbation function.

    Returns:
    - results: a list of dictionaries with keys 'graph_idx', 'trial', 'angle', 'spectrum_diff', 'func_type', 'nodes_perturbed'.
    """
    if perturb_kwargs is None:
        perturb_kwargs = {}
    results = []
    for idx, data in enumerate(all_data):
        G = data.get("graph") or data["laplacian"]["graph"]
        edge_weights = data.get("edge_weights", {})
        lap_labels = data.get("lap_results", data.get("laplacian", {})).get("labels")
        sig_labels = data.get("sig_results", data.get("signless", {})).get("labels")
        all_chains = data.get("all_chains", [])
        alpha = data.get("alpha", data.get("a_orig", 0.5))
        beta = data.get("beta", data.get("b_orig", 0.5))
        if lap_labels is None:
            continue
        # prepare cluster mapping for weighted method if needed
        cluster_map = defaultdict(list)
        for node, c in lap_labels.items():
            cluster_map[c].append(node)
        for trial in range(n_trials):
            # if perturbation function is weighted, ensure its specific parameters are passed
            if perturb_func == weighted_imbalance_perturbation or func_name == "weighted":
                perturb_kwargs.update({"lap_labels": lap_labels, "edge_weights": edge_weights, 
                                       "num_nodes_per_cluster": num_nodes_per_cluster, "keep": keep})
            if perturb_func == perturbation_by_levels or func_name == "shortcut":
                perturb_kwargs.update({"contracted_edges": all_chains, "lap_labels": lap_labels, 
                                       "sig_labels": sig_labels, "edge_weights": edge_weights, "min_deg": 2, "print_stats": False})
            if perturb_func == inter_cluster_perturbation or func_name == "inter_cluster":
                perturb_kwargs.update({"lap_labels": lap_labels, "sig_labels": sig_labels, "min_deg": 2})
            if perturb_func == intra_cluster_perturbation or func_name == "intra_cluster":
                perturb_kwargs.update({"lap_labels": lap_labels, "sig_labels": sig_labels, "min_deg": 2})
            if perturb_func == preserve_cycle_perturbation or func_name == "preserve_cycle":
                perturb_kwargs.update({"contracted_edges": all_chains, "min_deg": 2})
            if perturb_func == break_cycle_perturbation or func_name == "break_cycle":
                perturb_kwargs.update({"contracted_edges": all_chains, "min_deg": 1})
            if perturb_func == random_edge_perturbation or func_name == "random":
                perturb_kwargs.setdefault("fraction", 0.1)
                perturb_kwargs.setdefault("min_deg", 1)
            # apply the perturbation function to get a perturbed graph
            Gp = perturb_func(G, **perturb_kwargs)
            # contract the perturbed graph (if using contraction, otherwise skip)
            # note: full_contraction_ham should be defined in the 'contraction' module
            # Gp_contr, _ = full_contraction_ham(Gp)
            # for now, assume Gp_contr is same as Gp if contraction function is not available
            Gp_contr = Gp
            # compute spectral subspaces for original and perturbed graphs
            k = max(2, len(set(lap_labels.values())))
            vals_ref, vecs_ref = get_laplacian_subspace(G, k=k)
            vals_pert, vecs_pert = get_laplacian_subspace(Gp_contr, k=k)
            aligned_vecs_ref, aligned_vecs_pert = align_embeddings(G, Gp_contr, vecs_ref, vecs_pert)
            angle = max_principal_angle(aligned_vecs_ref, aligned_vecs_pert)
            delta_spec = spectrum_diff(vals_ref, vals_pert)
            # filter labels for nodes present in perturbed (contracted) graph
            filtered_lap_labels = {n: lap_labels[n] for n in Gp_contr.nodes() if n in lap_labels}
            filtered_sig_labels = {n: sig_labels[n] for n in Gp_contr.nodes() if n in sig_labels}
            lap_res_plot = dict(data.get("lap_results", data.get("laplacian", {})), labels=filtered_lap_labels)
            sig_res_plot = dict(data.get("sig_results", data.get("signless", {})), labels=filtered_sig_labels)
            # visualization (if needed)
            # draw_dual_colored_graph(Gp_contr, lap_res_plot, sig_res_plot,
            #                         alpha=alpha, beta=beta, shortcuts=all_chains,
            #                         title=f"perturbed {idx} ({func_name}), trial {trial} | θ={angle:.2f}°, Δλ={delta_spec:.4f}")
            # log results
            results.append({
                "graph_idx": idx,
                "trial": trial,
                "angle": angle,
                "spectrum_diff": delta_spec,
                "func_type": func_name,
                "nodes_perturbed": perturb_kwargs.get("nodes_to_perturb", [])
            })
    return results
