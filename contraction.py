import networkx as nx

# return normalized edge tuple
def normalize_key(u, v):
    return tuple(sorted((u, v)))

# contract all chains of degree-2 nodes into shortcuts
def contract_degree2_chains(G):
    Gc = G.copy()
    deg2 = [n for n in Gc.nodes() if Gc.degree(n) == 2]
    used = set()
    chains = []
    for n in deg2:
        if n in used or n not in Gc or Gc.degree(n) != 2:
            continue
        left_path = []
        curr, prev = n, None
        while True:
            nbrs = [x for x in Gc.neighbors(curr) if x != prev]
            if prev is not None and (Gc.degree(curr) != 2 or curr in used or len(nbrs) != 1):
                break
            if prev is not None:
                left_path.insert(0, curr)
            if not nbrs:
                break
            prev, curr = curr, nbrs[0]
        left_end = curr

        right_path = []
        curr, prev = n, None
        while True:
            nbrs = [x for x in Gc.neighbors(curr) if x != prev]
            if prev is not None and (Gc.degree(curr) != 2 or curr in used or len(nbrs) != 1):
                break
            if prev is not None:
                right_path.append(curr)
            if not nbrs:
                break
            prev, curr = curr, nbrs[-1]
        right_end = curr

        chain = [left_end] + left_path + [n] + right_path + [right_end]
        if left_end == right_end or Gc.degree(left_end) == 2 or Gc.degree(right_end) == 2:
            continue
        for node in chain:
            used.add(node)
        Gc.add_edge(left_end, right_end)
        for node in chain[1:-1]:
            if node in Gc:
                Gc.remove_node(node)
        chains.append(normalize_key(left_end, right_end))
    return Gc, chains

# remove extraneous edges from nodes with two contracted incident edges
def clean_ham_cond(G, contracted_edges):
    Gc = G.copy()
    contracted_set = set(normalize_key(*e) for e in contracted_edges)
    for node in list(Gc.nodes()):
        incident_contracted = [nbr for nbr in Gc.neighbors(node) if normalize_key(node, nbr) in contracted_set]
        if len(incident_contracted) == 2:
            for nbr in list(Gc.neighbors(node)):
                if normalize_key(node, nbr) not in contracted_set:
                    Gc.remove_edge(node, nbr)
    return Gc

# iteratively contract all degree-2 chains, cleaning at each step
def full_contraction_ham(G):
    Gc = G.copy()
    all_chains = []
    while True:
        Gc, chains = contract_degree2_chains(Gc)
        if not chains:
            break
        Gc = clean_ham_cond(Gc, chains)
        all_chains.extend(chains)
    return Gc, all_chains
