"""Product side of the project: group products and find 'bought together' items.

Customers are the columns here: two products are similar if the same customers buy them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from . import clustering, features


@dataclass
class ProductFeatureSet:
    matrix: object            # products x customers, binary sparse
    products: np.ndarray
    X_reduced: np.ndarray
    popularity: np.ndarray    # customers who bought each product
    explained_variance: float


def make_product_features(df, min_customers_per_product=50, min_products_per_customer=20,
                          max_products=1000, n_components=50, seed=42):
    M, _, prod = features.build_customer_product_matrix(
        df, min_products_per_customer, min_customers_per_product, "binary")
    pop = np.asarray(M.sum(axis=0)).ravel()
    if M.shape[1] > max_products:
        keep = np.sort(np.argsort(-pop)[:max_products])
        M, prod, pop = M[:, keep], prod[keep], pop[keep]
    P = M.T.tocsr()
    n_components = max(2, min(n_components, P.shape[1] - 1, P.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    X = svd.fit_transform(normalize(P))
    return ProductFeatureSet(P, prod, X, pop, float(svd.explained_variance_ratio_.sum()))


def cluster_products(pf: ProductFeatureSet, k=8, n_neighbors=10, seed=42) -> dict:
    return {
        "KMeans": KMeans(k, n_init=10, random_state=seed).fit_predict(pf.X_reduced),
        "Spectral (kNN)": SpectralClustering(
            k, affinity="nearest_neighbors", n_neighbors=n_neighbors,
            assign_labels="kmeans", random_state=seed).fit_predict(pf.X_reduced),
        "Spectral (Jaccard kNN)": SpectralClustering(
            k, affinity="precomputed", assign_labels="kmeans", random_state=seed
        ).fit_predict(clustering.jaccard_affinity(pf.matrix, n_neighbors)),
    }


def group_table(pf, labels, descriptions=None, top_n=8) -> pd.DataFrame:
    """Most popular products in each product group."""
    rows = []
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        for j in idx[np.argsort(-pf.popularity[idx])][:top_n]:
            code = pf.products[j]
            rows.append({"Group": c, "Group size": len(idx), "StockCode": code,
                         "Description": (descriptions or {}).get(code, ""),
                         "Customers": int(pf.popularity[j])})
    return pd.DataFrame(rows)


def similar_products(pf, code, descriptions=None, n=8) -> pd.DataFrame:
    """'Customers who bought X also bought...' by cosine similarity of customer baskets."""
    i = int(np.where(pf.products == code)[0][0])
    Xn = normalize(pf.X_reduced)
    sim = Xn @ Xn[i]
    order = [j for j in np.argsort(-sim) if j != i][:n]
    return pd.DataFrame({"StockCode": pf.products[order],
                         "Description": [(descriptions or {}).get(pf.products[j], "") for j in order],
                         "Similarity": np.round(sim[order], 3)})
