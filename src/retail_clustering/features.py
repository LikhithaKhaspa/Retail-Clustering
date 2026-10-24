"""Step 2 - customer x product matrix, L2 normalisation, SVD."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


WEIGHTINGS = ("binary", "invoices", "log-invoices")


@dataclass
class FeatureSet:
    matrix: sparse.csr_matrix   # weighted customer x product matrix that was clustered
    binary: sparse.csr_matrix   # 0/1 version, used for profiling and Jaccard similarity
    customers: np.ndarray       # row labels
    products: np.ndarray        # column labels (StockCode)
    X_reduced: np.ndarray       # L2-normalised then SVD-reduced features
    explained_variance: float
    weighting: str = "binary"


def build_customer_product_matrix(df, min_products_per_customer=20,
                                  min_customers_per_product=50, weighting="binary"):
    """Customer x product matrix, filtered to active customers and popular products.

    weighting:
      "binary"       1 if the customer ever bought the product (default)
      "invoices"     number of separate orders containing the product ("how often")
      "log-invoices" log(1 + invoices), which stops very frequent items dominating

    Returns (M, customers, products). Filtering always uses the binary view, so the
    thresholds mean the same thing whichever weighting is chosen.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    # sort=False keeps first-appearance order, so customers/products are indexed in
    # the same order for every weighting (and results stay reproducible).
    pairs = (df.groupby(["Customer ID", "StockCode"], sort=False)["Invoice"].nunique()
               .rename("n").reset_index())
    c_codes, customers = pd.factorize(pairs["Customer ID"])
    p_codes, products = pd.factorize(pairs["StockCode"])
    n = pairs["n"].to_numpy(dtype=np.float32)
    if weighting == "binary":
        vals = np.ones_like(n)
    elif weighting == "invoices":
        vals = n
    else:
        vals = np.log1p(n)
    M = sparse.csr_matrix((vals, (c_codes, p_codes)),
                          shape=(len(customers), len(products)))
    customers, products = np.asarray(customers), np.asarray(products)

    # Filter popular products first, then active customers (order matters:
    # dropping rare products can reduce a customer's basket size).
    B = (M > 0).astype(np.float32).tocsr()
    keep_p = np.asarray(B.sum(axis=0)).ravel() >= min_customers_per_product
    M, B, products = M[:, keep_p], B[:, keep_p], products[keep_p]
    keep_c = np.asarray(B.sum(axis=1)).ravel() >= min_products_per_customer
    M, customers = M[keep_c], customers[keep_c]
    return M.tocsr(), customers, products


def sample_matrix(M, customers, products, max_customers=2000, max_products=500, seed=42):
    rng = np.random.default_rng(seed)
    if M.shape[0] > max_customers:
        idx = np.sort(rng.choice(M.shape[0], max_customers, replace=False))
        M, customers = M[idx], customers[idx]
    if M.shape[1] > max_products:
        # keep the most popular products rather than a random subset
        popularity = np.asarray((M > 0).sum(axis=0)).ravel()
        top = np.sort(np.argsort(-popularity)[:max_products])
        M, products = M[:, top], products[top]
    return M.tocsr(), customers, products


def reduce_features(M, n_components=50, seed=42):
    X = normalize(M, norm="l2")
    n_components = max(2, min(n_components, M.shape[1] - 1, M.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    X_reduced = svd.fit_transform(X)
    return X_reduced, float(svd.explained_variance_ratio_.sum())


def make_features(df, min_products_per_customer=20, min_customers_per_product=50,
                  max_customers=2000, max_products=500, n_components=50, seed=42,
                  weighting="binary"):
    M, cust, prod = build_customer_product_matrix(
        df, min_products_per_customer, min_customers_per_product, weighting)
    M, cust, prod = sample_matrix(M, cust, prod, max_customers, max_products, seed)
    X_reduced, ev = reduce_features(M, n_components, seed)
    B = (M > 0).astype(np.float32).tocsr()
    return FeatureSet(M, B, cust, prod, X_reduced, ev, weighting)
