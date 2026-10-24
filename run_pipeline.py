"""Command-line run of the whole project.

    python run_pipeline.py --data data/online_retail_II.csv --k 6 --products
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import matplotlib; matplotlib.use("Agg")
import numpy as np
import pandas as pd

from retail_clustering import (clustering, data, evaluation, features, plots,
                               products, profiling)

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--data", required=True)
p.add_argument("--k", type=int, default=6)
p.add_argument("--country", default=None, help="e.g. 'United Kingdom'")
p.add_argument("--max-customers", type=int, default=2000)
p.add_argument("--max-products", type=int, default=500)
p.add_argument("--min-products-per-customer", type=int, default=20)
p.add_argument("--min-customers-per-product", type=int, default=50)
p.add_argument("--svd-components", type=int, default=50)
p.add_argument("--weighting", choices=features.WEIGHTINGS, default="binary")
p.add_argument("--out", default="outputs")
p.add_argument("--no-tsne", action="store_true")
p.add_argument("--products", action="store_true", help="also group products + similar-product table")
p.add_argument("--product-k", type=int, default=8)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

raw = data.load_transactions(a.data)
df, report = data.clean_transactions(raw, a.country)
for k, v in report.items():
    print(f"{k:40s}{v:>10,}")

fs = features.make_features(
    df, min_products_per_customer=a.min_products_per_customer,
    min_customers_per_product=a.min_customers_per_product,
    max_customers=a.max_customers, max_products=a.max_products,
    n_components=a.svd_components, weighting=a.weighting)
print(f"\nMatrix {fs.matrix.shape} ({a.weighting})  SVD variance kept: {fs.explained_variance:.1%}")

labels = clustering.run_all(fs.X_reduced, a.k, M=fs.binary)
res = evaluation.evaluate(fs.X_reduced, labels)
print("\n", res.to_string(index=False))
res.to_csv(f"{a.out}/metrics.csv", index=False)

desc = df.drop_duplicates("StockCode").set_index("StockCode")["Description"].to_dict()
best = res.sort_values("Silhouette (higher)").iloc[-1]["Method"]
evaluation.top_products_per_cluster(fs.binary, fs.products, labels[best], desc) \
    .to_csv(f"{a.out}/cluster_profiles.csv", index=False)
summary = profiling.segment_summary(df, fs.customers, labels[best])
summary.to_csv(f"{a.out}/segment_summary.csv", index=False)
print(f"\nSegment summary ({best}):\n", summary.to_string(index=False))
pd.DataFrame({"Customer ID": fs.customers, **labels}).to_csv(f"{a.out}/customer_labels.csv", index=False)

plots.scatter_grid(plots.embed_2d(fs.X_reduced), labels, "PCA 2D").savefig(f"{a.out}/pca.png", dpi=130)
if not a.no_tsne:
    plots.scatter_grid(plots.embed_2d(fs.X_reduced, "tsne"), labels, "t-SNE 2D").savefig(f"{a.out}/tsne.png", dpi=130)
plots.size_bars(labels).savefig(f"{a.out}/sizes.png", dpi=130)

if a.products:
    pf = products.make_product_features(df)
    pl = products.cluster_products(pf, a.product_k)
    products.group_table(pf, pl["KMeans"], desc).to_csv(f"{a.out}/product_groups_kmeans.csv", index=False)
    products.group_table(pf, pl["Spectral (Jaccard kNN)"], desc).to_csv(f"{a.out}/product_groups_spectral.csv", index=False)
    print(f"\nProduct groups: {pf.matrix.shape[0]} products; sizes "
          + "; ".join(f"{k}: {sorted(np.bincount(v).tolist())}" for k, v in pl.items()))
print(f"\nBest silhouette: {best}. Results in {a.out}/")
