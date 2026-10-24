"""Step 4 - internal metrics + business profile of each cluster."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def evaluate(X, labels_by_method: dict) -> pd.DataFrame:
    rows = []
    for name, labels in labels_by_method.items():
        if len(set(labels)) < 2:
            continue
        rows.append({
            "Method": name,
            "Silhouette (higher)": silhouette_score(X, labels),
            "Silhouette cosine (higher)": silhouette_score(X, labels, metric="cosine"),
            "Davies-Bouldin (lower)": davies_bouldin_score(X, labels),
            "Calinski-Harabasz (higher)": calinski_harabasz_score(X, labels),
            "Cluster sizes": np.bincount(labels).tolist(),
        })
    return pd.DataFrame(rows)


def agreement(labels_by_method: dict) -> pd.DataFrame:
    """Adjusted Rand index between every pair of methods (1 = identical)."""
    from sklearn.metrics import adjusted_rand_score
    names = list(labels_by_method)
    return pd.DataFrame(
        [[adjusted_rand_score(labels_by_method[a], labels_by_method[b]) for b in names]
         for a in names], index=names, columns=names)


def top_products_per_cluster(M, products, labels, descriptions: dict | None = None, top_n=5):
    """Products bought by the largest share of each cluster, with lift vs. everyone."""
    M = M.tocsc() if hasattr(M, "tocsc") else M
    overall = np.asarray(M.mean(axis=0)).ravel()
    rows = []
    for c in sorted(set(labels)):
        share = np.asarray(M[labels == c].mean(axis=0)).ravel()
        lift = share / np.maximum(overall, 1e-9)
        # rank by share*lift: popular *and* distinctive
        for j in np.argsort(-(share * lift))[:top_n]:
            code = products[j]
            rows.append({"Cluster": c, "StockCode": code,
                         "Description": (descriptions or {}).get(code, ""),
                         "Share of cluster": round(float(share[j]), 3),
                         "Lift": round(float(lift[j]), 2)})
    return pd.DataFrame(rows)
