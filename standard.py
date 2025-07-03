import networkx as nx
import numpy as np
from scipy.sparse import csgraph, diags
from scipy.sparse.linalg import eigsh
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# compute (normalized) laplacian matrix
def compute_laplacian(G, normalized=True):
    A = nx.to_scipy_sparse_array(G, weight='weight', dtype=float)
    return csgraph.laplacian(A, normed=normalized)

# get k smallest eigenpairs for laplacian
def compute_laplacian_eigenpairs(L, k, v0=None, tol=1e-6):
    vals, vecs = eigsh(L, k=k, v0=v0, tol=tol, which='SM')
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]

# row-normalize a matrix
def normalize_rows(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return X / norm

# normalize an edge as sorted tuple
def normalize_key(u, v):
    return tuple(sorted((u, v)))

# plot eigengap spectrum and highlights
def plot_eigengap(gaps, k=None, cutoff_t=None, threshold=None, signal_idxs=None, mode=None):
    plt.figure(figsize=(8, 4))
    plt.plot(gaps, label='raw gaps', alpha=0.4)
    if threshold is not None:
        plt.axhline(threshold, color='dimgray', linestyle='--', label=f'thresh={threshold:.3f}')
    if signal_idxs is not None and len(signal_idxs):
        plt.scatter(signal_idxs, np.array(gaps)[signal_idxs], color='red', marker='x', label='signal gaps')
    if cutoff_t is not None:
        plt.axvline(cutoff_t, color='gray', linestyle='--', label=f'tail start={cutoff_t}')
    if k is not None:
        plt.axvline(k-1, color='red', linestyle='dotted', label=f'k={k} ({mode})')
    plt.xlabel('gap index')
    plt.ylabel('eigengap')
    plt.title('eigengap analysis')
    plt.legend()
    plt.tight_layout()
    plt.show()

# find candidate k by looking for prominent eigengap peaks
def find_candidate_ks(gaps, evals, evecs, method='median', prominence=0.01, thresh_factor=1, min_window=20, sample_size=100):
    dim = len(gaps)
    candidates = []
    k_to_index = {}
    while dim >= min_window:
        g = gaps[:dim]
        threshold = np.median(g) + thresh_factor * np.std(g) if method == 'median' else np.mean(g) + thresh_factor * np.std(g)
        peaks, _ = find_peaks(g, height=threshold, prominence=prominence)
        if not len(peaks):
            break
        first_p = peaks[0]
        highest_p = peaks[np.argmax(g[peaks])]
        for p in {first_p, highest_p}:
            k = p + 1
            if k in k_to_index:
                idx = k_to_index[k]
                if dim < candidates[idx]['dim']:
                    candidates[idx]['dim'] = dim
                continue
            if k >= dim:
                continue
            eigengap = g[p]
            X = normalize_rows(evecs[:, 1:k+1])
            sample_size = max(sample_size, k)
            idx_sample = np.random.choice(X.shape[0], min(sample_size, X.shape[0]), replace=False)
            Xs = X[idx_sample]
            sil = silhouette_score(Xs, KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs).labels_) if k > 1 else -1
            score = (eigengap / k) * (1 + sil) if sil >= 0.2 else -np.inf
            candidates.append({
                'k': k,
                'eigengap': eigengap,
                'silhouette': sil,
                'score': score,
                'index': p,
                'threshold': threshold,
                'type': 'first' if p == first_p else 'highest',
                'dim': dim
            })
            k_to_index[k] = len(candidates) - 1
        dim = highest_p + 1
    return candidates

# perturb laplacian by random edge flips
def perturb_laplacian(L, noise_level, n):
    A = -L.copy()
    A.setdiag(0)
    A = A.tolil()
    deg = np.array(-A.sum(axis=1)).flatten()
    deg[deg < 0] = 0
    if deg.sum() == 0:
        return L
    prob = deg / deg.sum()
    num_edges = int(noise_level * n)
    for _ in range(num_edges):
        i, j = np.random.choice(n, 2, p=prob)
        if i != j:
            A[i, j] = A[j, i] = 1
    A = A.tocsr()
    D = diags(np.array(A.sum(axis=1)).flatten())
    return D - A

# check stability of candidate k's to laplacian noise
def check_candidate_stability(L, candidates, noise_level=0.01, repeats=5):
    n = L.shape[0]
    for c in candidates:
        k = c['k']
        eigengap = c['eigengap']
        cnt = 0
        for _ in range(repeats):
            Lp = perturb_laplacian(L, noise_level, n)
            vals, _ = eigsh(Lp, k=k+2, which='SM', tol=1e-6)
            vals.sort()
            gap = vals[k] - vals[k-1]
            if gap >= 0.5 * eigengap:
                cnt += 1
        c['stability'] = cnt / repeats
        c['score_stable'] = c['score'] * c['stability']
    return candidates

# pick best candidate by stability and score
def select_best_candidate(candidates):
    candidates = [c for c in candidates if c.get('stability',0) >= 0.5 and c['score_stable'] > -np.inf]
    if not candidates:
        return None
    candidates.sort(key=lambda c: c['score_stable'], reverse=True)
    return candidates[0]

# main spectral k selection and clustering pipeline
def spectral_k_pipeline(L, evals, evecs, gaps, method='median', prominence=0.01, thresh_factor=1, min_window=10, sample_size=100, noise_level=0.01, repeats=5):
    candidates = find_candidate_ks(
        gaps, evals, evecs, method=method, prominence=prominence,
        thresh_factor=thresh_factor, min_window=min_window, sample_size=sample_size
    )
    candidates = check_candidate_stability(L, candidates, noise_level=noise_level, repeats=repeats)
    best = select_best_candidate(candidates)
    return best, candidates

# laplacian clustering pipeline returning best k and label dict
def laplacian_clustering_pipeline(
    G,
    contracted_edges=None,
    method='median',
    prominence=0.01,
    thresh_factor=0.5,
    min_window=10,
    sample_size=100,
    noise_level=0.01,
    repeats=5,
    plot=False
):
    L = compute_laplacian(G, normalized=True)
    evals, evecs = np.linalg.eigh(L.toarray())
    gaps = np.diff(evals)
    best, candidates = spectral_k_pipeline(
        L, evals, evecs, gaps, prominence=prominence, thresh_factor=thresh_factor, min_window=min_window, 
        sample_size=sample_size, noise_level=noise_level, repeats=repeats
    )
    
    print(f"candidate ks: {candidates}")
    k = best['k'] if best and best.get('k', None) else None
    print(f"found k={k}")

    if k and k >= 1:
        X = evecs[:, 1:k+1]
        X = normalize(X)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        label_dict = dict(zip(G.nodes(), labels))
        if contracted_edges:
            parent = {n: n for n in G.nodes()}
            def find(u):
                while parent[u] != u:
                    parent[u] = parent[parent[u]]
                    u = parent[u]
                return u
            def union(u, v):
                pu, pv = find(u), find(v)
                if pu != pv:
                    parent[pu] = pv
            for u, v in contracted_edges:
                if u in parent and v in parent:
                    union(u, v)
            comp_to_label = {}
            for n in G.nodes():
                p = find(n)
                if p not in comp_to_label:
                    comp_to_label[p] = label_dict[n]
                label_dict[n] = comp_to_label[p]
        sil = silhouette_score(X, list(label_dict.values())) if k > 1 else None
    else:
        label_dict = None
        sil = None

    if plot:
        if k and k >= 1:
            plot_eigengap(gaps, k=k, cutoff_t=best.get('dim', None), threshold=best.get('threshold', None), signal_idxs=[best['index']], mode=best.get('type'))
        else:
            plot_eigengap(gaps)

    return {
        'graph': G,
        'laplacian': L,
        'evals': evals,
        'evecs': evecs,
        'gaps': gaps,
        'best': best,
        'candidates': candidates,
        'k': k,
        'labels': label_dict,
        'silhouette': sil
    }
