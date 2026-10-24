"""Interactive dashboard (single scrollable page, following the wireframe in the CA1 spec).

    streamlit run app.py
"""
import io, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from retail_clustering import (clustering, data, evaluation, features, plots,
                               products, profiling)

DEMO_FILE = os.path.join(HERE, "data", "synthetic_retail.csv")

st.set_page_config(page_title="Retail Customer Clustering Dashboard", layout="wide")
st.title("Retail Customer Clustering Dashboard")
st.caption("Spectral clustering vs K-Means vs agglomerative clustering on Online Retail II "
           "transactions. Scroll down: data, features, clustering, metrics, plots, segments, products.")


# ---------------------------------------------------------------- cached steps
@st.cache_data(show_spinner="Loading and cleaning the transactions...")
def load_and_clean(raw: bytes, name: str):
    buf = io.BytesIO(raw)
    buf.name = name
    raw_df = data.load_transactions(buf)
    clean, report = data.clean_transactions(raw_df)
    return clean, report, raw_df.shape


@st.cache_data(show_spinner="Building the customer x product matrix...")
def build_features(_df, key, **params):
    return features.make_features(_df, **params)


@st.cache_data(show_spinner="Clustering (the spectral methods can take a minute)...")
def run_methods(_fs, key, k, n_neighbors):
    return clustering.run_all(_fs.X_reduced, k, n_neighbors=n_neighbors, M=_fs.binary)


@st.cache_data(show_spinner="Computing t-SNE (about 20-60 seconds, only once per setting)...")
def tsne_embedding(_fs, key):
    return plots.embed_2d(_fs.X_reduced, "tsne")


@st.cache_data(show_spinner="Sweeping k...")
def k_sweep(_fs, key):
    return clustering.sweep_k(_fs.X_reduced)


@st.cache_data(show_spinner="Grouping products...")
def product_groups(_df, key, k):
    pf = products.make_product_features(_df)
    return pf, products.cluster_products(pf, k)


# ----------------------------------------------------------------- 1. upload
st.header("1. Data")
up = st.file_uploader("Upload the Online Retail II file (CSV or XLSX)", type=["csv", "xlsx"])
use_demo = False
if os.path.exists(DEMO_FILE):
    use_demo = st.checkbox("No file? Use the small synthetic demo dataset", value=False,
                           help="Fake transactions in the same format. Real results need the real file.")
if up is not None:
    raw_bytes, fname = up.getvalue(), up.name
elif use_demo:
    raw_bytes, fname = open(DEMO_FILE, "rb").read(), "synthetic_retail.csv"
    st.warning("Showing synthetic demo data, not Online Retail II.")
else:
    st.info("Upload a file to begin. The sidebar holds the pipeline settings.")
    st.stop()

# --------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Pipeline settings")
    st.caption("Filtering thresholds")
    min_cp = st.slider("Min products per customer", 5, 60, 20,
                       help="Customers who bought fewer distinct products are dropped.")
    min_pc = st.slider("Min customers per product", 5, 100, 50,
                       help="Products bought by fewer customers are dropped.")
    st.caption("Sample size (keeps spectral clustering fast)")
    max_c = st.select_slider("Max customers", [500, 1000, 2000, 3000, 5000], 2000)
    max_p = st.select_slider("Max products", [100, 250, 500, 1000], 500)
    st.caption("Representation")
    weighting = st.selectbox("Customer x product values", features.WEIGHTINGS, 0,
                             help="binary = bought or not; invoices = how many separate orders; "
                                  "log-invoices = log(1 + orders).")
    n_comp = st.slider("SVD components", 5, 100, 50)

df, report, raw_shape = load_and_clean(raw_bytes, fname)
dkey = (fname, len(raw_bytes))

# ------------------------------------------------- 1. preprocessing summary
c1, c2, c3 = st.columns(3)
c1.metric("Rows loaded", f"{raw_shape[0]:,}")
c2.metric("Columns", raw_shape[1])
c3.metric("Rows after cleaning", f"{len(df):,}", f"{len(df) / raw_shape[0]:.0%} of the raw rows kept", delta_color="off")
with st.expander("Row counts after each cleaning step"):
    st.dataframe(pd.Series(report, name="rows").to_frame(), width="stretch")
with st.expander("Preview of the cleaned transactions"):
    st.dataframe(df.head(100), width="stretch")

# ----------------------------------------------------- 2. feature engineering
st.header("2. Feature engineering")
params = dict(min_products_per_customer=min_cp, min_customers_per_product=min_pc,
              max_customers=max_c, max_products=max_p, n_components=n_comp,
              weighting=weighting)
fkey = (dkey, tuple(sorted(params.items())))
try:
    fs = build_features(df, fkey, **params)
except Exception as exc:                       # e.g. thresholds too strict
    st.error(f"Could not build the matrix with these settings: {exc}. "
             "Try lowering the filtering thresholds.")
    st.stop()
if fs.matrix.shape[0] < 30 or fs.matrix.shape[1] < 10:
    st.error(f"Only {fs.matrix.shape[0]} customers x {fs.matrix.shape[1]} products survive the "
             "filters. Lower the thresholds in the sidebar.")
    st.stop()
density = fs.binary.nnz / (fs.binary.shape[0] * fs.binary.shape[1])
f1, f2, f3 = st.columns(3)
f1.metric("Customer x product matrix", f"{fs.matrix.shape[0]:,} x {fs.matrix.shape[1]:,}")
f2.metric("Non-zero cells", f"{density:.1%}")
f3.metric(f"Variance kept by {fs.X_reduced.shape[1]} SVD components", f"{fs.explained_variance:.1%}")

# ------------------------------------------------------ 3. clustering controls
st.header("3. Clustering controls")
k1, k2 = st.columns(2)
k = k1.slider("Number of clusters (k)", 2, 10, 6)
n_nb = k2.slider("Spectral: n_neighbors", 5, 30, 10,
                 help="Size of the neighbourhood graph. Raise it if you see 'graph not fully connected'.")
labels = run_methods(fs, (fkey, k, n_nb), k, n_nb)

# ------------------------------------------------------------- 4. evaluation
st.header("4. Evaluation metrics")
metrics = evaluation.evaluate(fs.X_reduced, labels)
st.dataframe(metrics, width="stretch", hide_index=True)
st.caption("Silhouette and Calinski-Harabasz: higher is better. Davies-Bouldin: lower is better. "
           "Scores are computed in the reduced SVD space the methods were fitted on.")
with st.expander("How much do the methods agree? (Adjusted Rand Index, 1 = identical)"):
    st.dataframe(evaluation.agreement(labels).round(2), width="stretch")
if st.checkbox("Show silhouette vs k for K-Means (helps choose k)"):
    sweep = k_sweep(fs, fkey)
    best = max(sweep, key=sweep.get)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(list(sweep), list(sweep.values()), marker="o")
    ax.axvline(best, ls="--", c="grey")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette (K-Means)")
    ax.set_title(f"Highest silhouette at k = {best}  (range {min(sweep.values()):.3f} to {max(sweep.values()):.3f})")
    st.pyplot(fig)
    plt.close(fig)

# ----------------------------------------------------------- 5. visualisation
st.header("5. Cluster structure: PCA and t-SNE")
method = st.selectbox("Method to display", list(labels), key="viz_method")
pca_xy = plots.embed_2d(fs.X_reduced, "pca")
show_tsne = st.checkbox("Compute t-SNE", value=True)
left, right = st.columns(2)
with left:
    st.plotly_chart(plots.scatter_interactive(pca_xy, labels[method], f"{method}: PCA", "PC1", "PC2"),
                    width="stretch")
with right:
    if show_tsne:
        tsne_xy = tsne_embedding(fs, fkey)
        st.plotly_chart(plots.scatter_interactive(tsne_xy, labels[method], f"{method}: t-SNE",
                                                  "t-SNE 1", "t-SNE 2"), width="stretch")
    else:
        st.info("Tick 'Compute t-SNE' to show the t-SNE panel.")
with st.expander("All methods side by side (PCA)"):
    fig = plots.scatter_grid(pca_xy, labels, "PCA 2D")
    st.pyplot(fig)
    plt.close(fig)
st.subheader("Cluster size distribution")
fig = plots.size_bars(labels)
st.pyplot(fig)
plt.close(fig)
st.caption("Cluster numbers are labels within each method: cluster 3 of one method is not the same group as cluster 3 of another.")

# ---------------------------------------------------------- 6. segment profiles
st.header("6. Customer segments")
pm = st.selectbox("Method to profile", list(labels), key="profile_method")
st.write("**Who are they?** Spend, order frequency and product variety per cluster "
         "(medians are shown because a few very large buyers can inflate averages).")
summary = profiling.segment_summary(df, fs.customers, labels[pm])
st.dataframe(summary, width="stretch", hide_index=True)
st.write("**What do they buy?** `Share` is the fraction of the cluster that bought the product, "
         "`Lift` is how many times more often than the average customer.")
desc = df.drop_duplicates("StockCode").set_index("StockCode")["Description"].to_dict()
st.dataframe(evaluation.top_products_per_cluster(fs.binary, fs.products, labels[pm], desc),
             width="stretch", hide_index=True)
out = pd.DataFrame({"Customer ID": fs.customers, "Cluster": labels[pm]})
st.download_button("Download customer-to-cluster file (CSV)", out.to_csv(index=False),
                   "customer_clusters.csv", "text/csv")

# ------------------------------------------------------------ 7. product groups
st.header("7. Product groups and cross-selling")
st.write("Products are similar when the same customers buy them.")
pk = st.slider("Number of product groups", 3, 15, 8)
pf, plabels = product_groups(df, (dkey, pk), pk)
gm = st.selectbox("Grouping method", list(plabels), key="group_method")
sizes = pd.Series(plabels[gm]).value_counts().sort_index()
st.caption("Group sizes: " + ", ".join(f"{g}: {n}" for g, n in sizes.items()))
st.dataframe(products.group_table(pf, plabels[gm], desc), width="stretch", hide_index=True)
st.subheader("Customers who bought X also bought...")
options = {f"{c} - {desc.get(c, '')}": c for c in pf.products[::-1]}
pick = st.selectbox("Product", list(options))
st.dataframe(products.similar_products(pf, options[pick], desc), width="stretch", hide_index=True)
