"""Step 3 - the five clustering variants compared in the project."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.preprocessing import normalize


def jaccard_affinity(M, n_neighbors=10):
    """kNN graph weighted by Jaccard similarity of the two baskets (binary data)."""
    M = sparse.csr_matrix(M, copy=True)
    M.data[:] = 1
    inter = (M @ M.T).toarray()
    sizes = np.asarray(M.sum(axis=1)).ravel()
    J = inter / np.maximum(sizes[:, None] + sizes[None, :] - inter, 1)
    np.fill_diagonal(J, 0)
    k = min(n_neighbors, J.shape[0] - 1)
    idx = np.argpartition(-J, k, axis=1)[:, :k]
    A = np.zeros_like(J)
    rows = np.arange(J.shape[0])[:, None]
    A[rows, idx] = J[rows, idx]
    return np.maximum(A, A.T)


def run_all(X, n_clusters=6, n_neighbors=10, rbf_gamma=1.0, seed=42, M=None) -> dict:
    """M (binary customer x product matrix) enables the Jaccard variant."""
    spectral_knn = dict(affinity="nearest_neighbors", n_neighbors=n_neighbors,
                        random_state=seed)
    models = {
        "Spectral (kNN, k-means labels)": SpectralClustering(
            n_clusters=n_clusters, assign_labels="kmeans", **spectral_knn),
        "Spectral (kNN, discretize)": SpectralClustering(
            n_clusters=n_clusters, assign_labels="discretize", **spectral_knn),
        "Spectral (RBF kernel)": SpectralClustering(
            n_clusters=n_clusters, affinity="rbf", gamma=rbf_gamma, random_state=seed),
        "KMeans": KMeans(n_clusters=n_clusters, n_init=10, random_state=seed),
        "Agglomerative": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
    }
    labels = {name: m.fit_predict(X) for name, m in models.items()}

    # Hypothesis 2: does the similarity measure matter?
    # Unit-length rows make Euclidean kNN identical to a cosine kNN.
    labels["Spectral (cosine kNN)"] = SpectralClustering(
        n_clusters=n_clusters, assign_labels="kmeans", **spectral_knn
    ).fit_predict(normalize(X))
    if M is not None:
        labels["Spectral (Jaccard kNN)"] = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", assign_labels="kmeans",
            random_state=seed).fit_predict(jaccard_affinity(M, n_neighbors))
    return labels


def sweep_k(X, k_range=range(2, 11), seed=42):
    """Silhouette for K-Means across k - a quick way to justify the choice of k."""
    from sklearn.metrics import silhouette_score
    return {k: silhouette_score(X, KMeans(k, n_init=10, random_state=seed).fit_predict(X))
            for k in k_range}
