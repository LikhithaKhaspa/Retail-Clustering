"""Step 5 - PCA / t-SNE scatter grids and cluster-size bars."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def embed_2d(X, method="pca", seed=42):
    if method == "tsne":
        perp = min(30, max(5, (len(X) - 1) // 3))
        return TSNE(2, perplexity=perp, init="pca", random_state=seed).fit_transform(X)
    return PCA(2, random_state=seed).fit_transform(X)


def scatter_grid(coords, labels_by_method, title_suffix, ncols=3):
    n = len(labels_by_method)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.3 * nrows), squeeze=False)
    for ax, (name, labels) in zip(axes.ravel(), labels_by_method.items()):
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
        ax.set_title(f"{name}\n({title_suffix})", fontsize=10)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def size_bars(labels_by_method):
    df = pd.DataFrame({m: pd.Series(l).value_counts().sort_index()
                       for m, l in labels_by_method.items()}).fillna(0)
    ax = df.plot.bar(figsize=(9, 4.5), width=0.85)
    ax.set_xlabel("Cluster index"); ax.set_ylabel("Customers")
    ax.set_title("Cluster size distribution")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.get_figure().tight_layout()
    return ax.get_figure()


def scatter_interactive(coords, labels, title, x="Dim 1", y="Dim 2"):
    """Plotly scatter: hover shows the cluster; click a legend entry to hide a cluster."""
    import plotly.express as px
    df = pd.DataFrame({x: coords[:, 0], y: coords[:, 1],
                       "Cluster": pd.Series(labels).astype(str).values})
    order = sorted(df["Cluster"].unique(), key=int)
    fig = px.scatter(df, x=x, y=y, color="Cluster", title=title,
                     category_orders={"Cluster": order}, opacity=0.75,
                     color_discrete_sequence=px.colors.qualitative.T10)
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10),
                      legend_title_text="Cluster")
    return fig
