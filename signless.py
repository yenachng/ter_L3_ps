import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from vis import draw_graph_with_labels
from perturbation import align_embeddings

# compute unnormalized signless laplacian
def compute_signless_laplacian(G):
    A = nx.to_scipy_sparse_array(G, weight='weight', dtype=float, format='csr')
    D = diags(A.sum(axis=1).flatten())
    return D + A

# random perturbation of signless laplacian
def perturb_signless_laplacian(Q, noise_level, n):
    A = Q.copy()
    A.setdiag(0)
    A = -A
    A = A.tolil()
    deg = np.array(A.sum(axis=1)).flatten()
    deg[deg < 0] = 0
    if deg.sum() == 0:
        return Q
    prob = deg / deg.sum()
    num_edges = int(noise_level * n)
    for _ in range(num_edges):
        i, j = np.random.choice(n, 2, p=prob)
        if i != j:
            A[i, j] = A[j, i] = -1
    A = -A
    A = A.tocsr()
    D = diags(np.array(A.sum(axis=1)).flatten())
    return D + A

# eigengap smoothing (univariate spline)
def analyze_eigengap_spectrum(evals, smooth=0.01):
    gaps = -np.diff(evals)
    x = np.arange(len(gaps))
    spline = UnivariateSpline(x, gaps, s=smooth)
    smoothed = spline(x)
    return evals, gaps, smoothed

# find where smoothed gaps go below threshold and stay there
def find_tail_start(smoothed, global_thresh):
    for i in range(len(smoothed)):
        if np.all(smoothed[i:] < global_thresh):
            return i
    return len(smoothed)

# compute soft cluster memberships (fuzzy-c-means style, but with kmeans distances)
def fuzzy_cmeans_memberships(X, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    dists = km.fit_transform(X)
    memberships = np.exp(-dists)
    memberships /= memberships.sum(axis=1, keepdims=True)
    labels = np.argmax(memberships, axis=1)
    return labels, memberships

# silhouette score for soft/hard clusters
def compute_silhouette_general(X, labels, memberships=None):
    if X.shape[0] < 2 or len(np.unique(labels)) < 2:
        return -1
    try:
        return silhouette_score(X, labels)
    except Exception:
        return -1

# check eigengap candidate stability to laplacian noise
def check_candidate_stability_signless(Q, candidates, tail_start, noise_level=0.01, repeats=5):
    n = Q.shape[0]
    for c in candidates:
        k = c['k']
        cnt = 0
        for _ in range(repeats):
            Qp = perturb_signless_laplacian(Q, noise_level, n)
            vals, _ = eigsh(Qp, k=tail_start+2, which='LM', v0=np.ones(n), tol=1e-6)
            idx = np.argsort(vals)[::-1]
            vals = vals[idx]
            gaps = -np.diff(vals)
            if c['mode'] == 'dom':
                gap = gaps[k-1] if k <= len(gaps) else 0
                if gap >= 0.5 * c['eigengap']:
                    cnt += 1
            else:
                idxs = c.get('signal_idxs')
                if idxs is not None and len(idxs) > 0:
                    stable = True
                    for i in idxs:
                        i = int(i)
                        if i >= len(gaps) or gaps[i] < 0.5 * c['eigengap']:
                            stable = False
                            break
                    if stable:
                        cnt += 1
                else:
                    if np.median(gaps) >= 0.5 * c['eigengap']:
                        cnt += 1
        c['stability'] = cnt / repeats
        c['score_stable'] = c['score'] * c['stability']
    return candidates

# plot gaps and eigengap selection
def plot_gap_analysis(evals, gaps, smoothed, tail_start, global_thresh, k, signal_idxs=None, mode=None, thresh=None):
    plt.figure(figsize=(8, 4))
    plt.plot(gaps, label='raw gaps', alpha=0.4)
    plt.plot(smoothed, label='smoothed', linewidth=1.5)
    plt.axhline(global_thresh, color='dimgray', linestyle='--', label=f'global thresh = {global_thresh:.3f}')
    if thresh is not None:
        plt.axhline(thresh, color='blue', linestyle='-.', label=f'head thresh = {thresh:.3f}')
    if signal_idxs is not None and len(signal_idxs):
        plt.scatter(signal_idxs, np.array(gaps)[signal_idxs], color='red', marker='x', label='signal gaps')
    if tail_start is not None:
        plt.axvline(tail_start, color='gray', linestyle='--', label=f'tail start = {tail_start}')
    if k is not None:
        plt.axvline(k-1, color='red', linestyle='dotted', label=f'k = {k} ({mode})')
    plt.xlabel('gap index')
    plt.ylabel('eigengap')
    plt.title('eigengap analysis')
    plt.legend()
    plt.tight_layout()
    plt.show()

# find strong eigengap peaks (signal gaps)
def find_strong_peaks(head_gaps, prominence=0.01, alpha=0.1):
    if len(head_gaps) == 0:
        return np.array([], dtype=int), None
    peaks, props = find_peaks(head_gaps, prominence=prominence)
    min_thresh = np.median(head_gaps) + alpha * np.std(head_gaps)
    strong_peaks = [p for p in peaks if head_gaps[p] > min_thresh]
    if head_gaps[0] > min_thresh and 0 not in strong_peaks:
        strong_peaks = [0] + strong_peaks
    return np.unique(strong_peaks), min_thresh

# try dominant single eigengap candidate
def try_dominant_candidate(head_gaps, evecs, dom_idx, tail_start):
    k_dom = dom_idx + 1
    if k_dom > evecs.shape[1]:
        return None
    dom_gap = head_gaps[dom_idx]
    rest = np.delete(head_gaps, dom_idx)
    mean_gap = rest.mean() if rest.size else 1e-9
    std_gap = rest.std() if rest.size else 1e-9
    next_gap = np.partition(head_gaps, -2)[-2] if len(head_gaps) > 1 else 1e-9
    dominant_conditions = [
        lambda: dom_gap / (mean_gap + 1e-9) >= 1.2,
        lambda: dom_gap / (np.median(head_gaps) + 1e-9) >= 1.2,
        lambda: dom_gap / (next_gap + 1e-9) >= 1.3,
        lambda: (dom_gap - mean_gap) / (std_gap + 1e-9) >= 2,
        lambda: dom_gap > np.median(head_gaps) + np.std(head_gaps),
    ]
    for cond in dominant_conditions:
        if rest.size and std_gap > 0 and cond():
            x_dom = normalize(evecs[:, :k_dom])
            labels_dom = KMeans(n_clusters=k_dom, n_init=10, random_state=42).fit_predict(x_dom)
            sil_dom = silhouette_score(x_dom, labels_dom) if k_dom > 1 else -1
            score_dom = sil_dom + 1 / (1 + abs(k_dom - tail_start) + 1e-6)
            return {
                'k': k_dom,
                'mode': 'dom',
                'eigengap': dom_gap,
                'silhouette': sil_dom,
                'score': score_dom,
                'signal_idxs': [dom_idx],
                'embedding_dim': k_dom,
                'head_thresh': None,
                'alpha': None
            }
    return None

# candidate selection for multi-peak (perron cluster) mode
def perron_candidates(head_gaps, evecs, tail_start, prominence=None, alphas=None, min_k=2):
    if alphas is None:
        alphas = [0.05, 0.15, 0.3, 0.5]
    if prominence is None:
        prominence = 0.01
    candidates = []
    for alpha in alphas:
        peaks, min_thresh = find_strong_peaks(head_gaps, prominence=prominence, alpha=alpha)
        k = len(peaks)
        if k < min_k or k > evecs.shape[1]:
            continue
        x_perron = normalize(evecs[:, :k])
        labels_perron, memberships = fuzzy_cmeans_memberships(x_perron, k)
        sil_perron = compute_silhouette_general(x_perron, labels_perron, memberships)
        score_perron = sil_perron + 1 / (tail_start + 1e-6)
        candidates.append({
            'k': k,
            'mode': 'perron',
            'eigengap': np.median(head_gaps[peaks]) if len(peaks) else 0,
            'silhouette': sil_perron,
            'score': score_perron,
            'signal_idxs': list(peaks),
            'embedding_dim': k,
            'head_thresh': min_thresh,
            'alpha': alpha,
            'prominence': prominence,
            'memberships': memberships,
            'labels': labels_perron,
        })
    return candidates

# fallback: select all gaps above global threshold
def fallback_candidate(head_gaps, evecs, tail_start, global_thresh):
    signal_idxs = [i for i, gap in enumerate(head_gaps) if gap >= global_thresh]
    k = len(signal_idxs) if signal_idxs else 2
    k = min(max(2, k), evecs.shape[1])
    x_fallback = normalize(evecs[:, :k])
    labels_fb, memberships = fuzzy_cmeans_memberships(x_fallback, k)
    sil_fb = compute_silhouette_general(x_fallback, labels_fb, memberships)
    score_fb = sil_fb + 1 / (1 + abs(k - tail_start) + 1e-6)
    return {
        'k': k,
        'mode': 'fallback',
        'eigengap': head_gaps[signal_idxs[0]] if signal_idxs else (head_gaps[0] if len(head_gaps) else 0),
        'silhouette': sil_fb,
        'score': score_fb,
        'signal_idxs': signal_idxs if signal_idxs else [0],
        'embedding_dim': k,
        'head_thresh': global_thresh,
        'alpha': None,
        'memberships': memberships,
        'labels': labels_fb,
    }

# optimize k for signless laplacian clustering
def optimize_k_signless(G, k_frac=0.75, smooth=0.05, min_sil=0.2, prominence=0.05):
    Q = compute_signless_laplacian(G)
    n = G.number_of_nodes()
    k_eig = max(2, int(n * k_frac))
    evals, evecs = eigsh(Q, k=k_eig, which='LM', v0=np.ones(n), tol=1e-6)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    _, gaps, smoothed = analyze_eigengap_spectrum(evals, smooth=smooth)
    global_thresh = np.median(smoothed) + 1.5 * np.std(smoothed)
    tail_start = find_tail_start(smoothed, global_thresh)
    head_gaps = gaps[:tail_start+1]
    candidates = []
    dom_peaks, min_thresh = find_strong_peaks(head_gaps, prominence=prominence, alpha=0.15)
    if len(dom_peaks) == 1:
        dom_idx = dom_peaks[0]
        dom_candidate = try_dominant_candidate(head_gaps, evecs, dom_idx, tail_start)
        if dom_candidate is not None and dom_candidate['silhouette'] >= min_sil:
            candidates.append(dom_candidate)
    else:
        perron = perron_candidates(head_gaps, evecs, tail_start, prominence=None, alphas=None, min_k=2)
        candidates.extend([c for c in perron if c['silhouette'] >= min_sil])
    if not candidates:
        candidates.append(fallback_candidate(head_gaps, evecs, tail_start, global_thresh))
    best = max(candidates, key=lambda c: c['score'], default=None)
    return best, candidates, evals, evecs, gaps, smoothed, tail_start, global_thresh

# cluster signless embedding by k-means
def spectral_cluster_signless(evecs, k, normalize_rows=True):
    X = evecs[:, :k].copy()
    if normalize_rows:
        X = normalize(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    sil_score = silhouette_score(X, labels) if k > 1 else None
    inertia = km.inertia_
    return labels, sil_score, inertia

# cluster using perron embedding and fuzzy memberships
def cluster_perron_embedding(result):
    k = result['k']
    if k < 1:
        return None, None, None, None
    evecs = result['evecs'][:, :k+1]
    X = normalize(evecs)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    distances = km.fit_transform(X)
    memberships = np.exp(-distances)
    memberships /= memberships.sum(axis=1, keepdims=True)
    labels = np.argmax(memberships, axis=1)
    sil_score = silhouette_score(X, labels) if k > 1 else None
    inertia = km.inertia_
    return memberships, labels, sil_score, inertia

# main signless laplacian clustering pipeline
def signless_laplacian_clustering_pipeline(
    G,
    contracted_edges=None,
    k_frac=0.75,
    smooth=0.01,
    min_sil=0.1,
    min_stability=0.4,
    noise_level=0.01,
    repeats=5,
    prominence=0.05,
    plot=True
):
    best, candidates, evals, evecs, gaps, smoothed, tail_start, global_thresh = optimize_k_signless(
        G, k_frac=k_frac, smooth=smooth, min_sil=min_sil, prominence=prominence
    )
    k = best['k'] if best and best.get('k', None) else None
    print(f"k:{k}")
    if k and k >= 1:
        X = normalize(evecs[:, :k])
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        labels = km.labels_
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
        memberships = None
        if best.get('mode') == 'perron' and hasattr(km, 'transform'):
            memberships = km.transform(X)
            memberships = np.exp(-memberships)
            memberships /= memberships.sum(axis=1, keepdims=True)
        sil = silhouette_score(X, list(label_dict.values())) if k > 1 else None
    else:
        label_dict = None
        memberships = None
        sil = None

    if plot:
        if k and k >= 1:
            plot_gap_analysis(
                evals, gaps, smoothed, tail_start, global_thresh, k,
                signal_idxs=best.get('signal_idxs', []),
                mode=best.get('mode', None),
                thresh=best.get('head_thresh', None)
            )
            if label_dict:
                draw_graph_with_labels(G, labels=label_dict, title=f"signless laplacian clustering (k={k}, {best.get('mode')})")
        else:
            plot_gap_analysis(evals, gaps, smoothed, tail_start, global_thresh, None, signal_idxs=None, mode=None, thresh=None)

    return {
        'graph': G,
        'k': k,
        'labels': label_dict,
        'memberships': memberships,
        'silhouette': sil,
        'best': best,
        'candidates': candidates,
        'evals': evals,
        'evecs': evecs
    }
