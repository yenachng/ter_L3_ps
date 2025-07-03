import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

def join_graphs(G1, H):
    G = nx.union(G1, H)
    for u in G1.nodes():
        for v in H.nodes():
            G.add_edge(u, v)
    return G

def disjoint_union(graphs):
    H = nx.Graph()
    for G in graphs:
        H = nx.union(H, G)
    return H

def invert_mapping(d):
    inv = {}
    for key, color in d.items():
        inv.setdefault(color, []).append(key)
    return inv

def N_k_n(k, n):
    if n < 2*k:
        raise ValueError("n must be at least 2*k")
    G1 = nx.complete_graph(k)
    G1 = nx.relabel_nodes(G1, lambda i: ('G1', i))
    num_H1 = n - 2*k
    H1 = nx.complete_graph(num_H1)
    H1 = nx.relabel_nodes(H1, lambda i: ('H1', i))
    H2 = nx.empty_graph(k)
    H2 = nx.relabel_nodes(H2, lambda i: ('H2', i))
    H = disjoint_union([H1, H2])
    G = join_graphs(G1, H)
    
    vertex_colors = {}
    for node in G.nodes():
        group, _ = node
        if group == 'G1':
            vertex_colors[node] = "blue"
        elif group == 'H1':
            vertex_colors[node] = "gray"
        elif group == 'H2':
            vertex_colors[node] = "black"
        else:
            vertex_colors[node] = "unknown"
    
    edge_colors = {}
    for u, v in G.edges():
        c1 = vertex_colors[u]
        c2 = vertex_colors[v]
        if c1 == c2:
            if c1 == "gray":
                edge_colors[(u, v)] = 'pink'
            else:
                edge_colors[(u, v)] = 'green'
        else:
            edge_colors[(u, v)] = 'gray'
    
    return G#, vertex_colors, edge_colors


def L_k_n(k, n):
    G1 = nx.empty_graph(1)
    G1 = nx.relabel_nodes(G1, lambda i: ('G1', i))

    H1 = nx.complete_graph(k)
    H1 = nx.relabel_nodes(H1, lambda i: ('H1', i))
    
    H2 = nx.complete_graph(n - k - 1)
    H2 = nx.relabel_nodes(H2, lambda i: ('H2', i))
    
    H = disjoint_union([H1, H2])
    
    G = join_graphs(G1, H)
    
    vertex_colors = {}
    for node in G.nodes():
        group, _ = node
        if group == 'G1':
            vertex_colors[node] = 'pink'
        elif group == 'H1':
            vertex_colors[node] = 'black'
        elif group == 'H2':
            vertex_colors[node] = 'gray'
        else:
            vertex_colors[node] = 'unknown'
    
    edge_colors = {}
    for u, v in G.edges():
        c1 = vertex_colors[u]
        c2 = vertex_colors[v]
        if c1 == c2:
            if c1 == 'black':
                edge_colors[(u, v)] = 'black'
            else:
                edge_colors[(u, v)] = 'gray'
        else:
            edge_colors[(u, v)] = 'red'
    
    return G #, vertex_colors, edge_colors

def L_k_n_bar(k, n):
    H1 = nx.complete_graph(k)
    H1 = nx.relabel_nodes(H1, lambda i: ('H1', i))
    
    H2 = nx.complete_graph(n - k - 1)
    H2 = nx.relabel_nodes(H2, lambda i: ('H2', i))
    
    G = disjoint_union([H1, H2])
    
    vertex_colors = {}
    for node in G.nodes():
        group, _ = node
        if group == 'H1':
            vertex_colors[node] = 'black'
        elif group == 'H2':
            vertex_colors[node] = 'gray'
        else:
            vertex_colors[node] = 'unknown'
    
    edge_colors = {}
    for u, v in G.edges():
        c1 = vertex_colors[u]
        c2 = vertex_colors[v]
        if c1 == c2:
            if c1 == 'black':
                edge_colors[(u, v)] = 'black'
            else:
                edge_colors[(u, v)] = 'gray'
        else:
            edge_colors[(u, v)] = 'red'
    
    return G #, vertex_colors, edge_colors

def N_k_n_bar(k, n):
    if n < 2*k + 1:
        raise ValueError("n must be at least 2*k+1")

    G1 = nx.complete_graph(k)
    G1 = nx.relabel_nodes(G1, lambda i: ('G1', i))

    num_H1 = n - 2*k - 1
    H1 = nx.complete_graph(num_H1)
    H1 = nx.relabel_nodes(H1, lambda i: ('H1', i))

    H2 = nx.empty_graph(k+1)
    H2 = nx.relabel_nodes(H2, lambda i: ('H2', i))

    H = disjoint_union([H1, H2])

    G = join_graphs(G1, H)
    
    vertex_colors = {}
    for node in G.nodes():
        group, _ = node
        if group == 'G1':
            vertex_colors[node] = "blue"
        elif group == 'H1':
            vertex_colors[node] = "gray"
        elif group == 'H2':
            vertex_colors[node] = "black"
        else:
            vertex_colors[node] = "unknown"
    
    edge_colors = {}
    for u, v in G.edges():
        c1 = vertex_colors[u]
        c2 = vertex_colors[v]
        if c1 == c2:
            if c1 == "gray":
                edge_colors[(u, v)] = 'pink'
            else:
                edge_colors[(u, v)] = 'green'
        else:
            edge_colors[(u, v)] = 'gray'
    
    return G #, vertex_colors, edge_colors

def add_one_edge_extremal(G):
    H = G.copy()
    nodes = list(H.nodes())
    non_edges = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:] if not H.has_edge(u, v)]
    if non_edges:
        u, v = random.choice(non_edges)
        H.add_edge(u, v)
    return H

def generate_hamiltonian_extremals(n=12, k_values=None):
    if k_values is None:
        k_values = list(range(1, n // 2))

    hamiltonian_graphs = []

    for k in k_values:
        try:
            G1 = add_one_edge_extremal(N_k_n(k, n))
            hamiltonian_graphs.append(G1)

            G3 = add_one_edge_extremal(L_k_n(k, n))
            hamiltonian_graphs.append(G3)
        except Exception:
            continue

    return hamiltonian_graphs

def generate_hamiltonian_extremals_cat(n=12, k_values=None):
    if k_values is None:
        k_values = list(range(1, n // 2))

    hamiltonian_graphs = []

    for k in k_values:
        try:
            G1 = add_one_edge_extremal(N_k_n(k, n))
            hamiltonian_graphs.append({
                'graph': G1,
                'type': 'N_k_n',
                'k': k,
                'n': n
            })

            G3 = add_one_edge_extremal(L_k_n(k, n))
            hamiltonian_graphs.append({
                'graph': G3,
                'type': 'L_k_n',
                'k': k,
                'n': n
            })
        except Exception:
            continue

    return hamiltonian_graphs


def generate_random_hamiltonian_graphs(n=12, num_graphs=20, extra_edge_prob=0.3):
    graphs = []
    for _ in range(num_graphs):
        nodes = list(range(n))
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from((i, (i+1) % n) for i in range(n))
        possible_edges = [(u, v) for u in nodes for v in nodes if u < v and not G.has_edge(u, v)]
        for u, v in possible_edges:
            if random.random() < extra_edge_prob:
                G.add_edge(u, v)
        graphs.append({
            'graph': G,
            'type': 'random',
            'k': None,
            'n': n
        })
    return graphs