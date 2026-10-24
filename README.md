# Data Analysis using Spectral Clustering Algorithms

Groups the customers of the **Online Retail II** dataset by *which products they buy*, compares
Spectral, K-Means and Agglomerative clustering with three validation metrics, 2-D visualisations
and business profiles, and offers the whole workflow as a notebook, a command-line script and an
interactive Streamlit dashboard.

## Quick start
```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 1. dashboard (upload data/online_retail_II.csv, or tick the synthetic demo box)
streamlit run app.py

# 2. command line (writes tables and plots to outputs/)
python run_pipeline.py --data data/online_retail_II.csv --k 6 --products

# 3. notebook (documented, step by step)
pip install -r requirements-dev.txt notebook
jupyter notebook notebooks/customer_segmentation.ipynb      # then Kernel > Restart & Run All

# 4. tests
python -m pytest -q
```
Get the data from the UCI repository (Chen, 2015, CC BY 4.0) and save it as `data/online_retail_II.csv`.
No dataset? `python scripts/make_synthetic_data.py` writes a fake one in the same format
(`data/synthetic_retail.csv` is included). Synthetic results are only for testing.

## Pipeline
| Step | Module | What happens |
|---|---|---|
| 1 Clean | `data.py` | drop cancellations, qty/price <= 0, missing Customer ID, non-product stock codes; row count recorded after each step |
| 2 Features | `features.py` | sparse customer x product matrix (binary, order counts or log order counts), filtering thresholds, sampling, L2 normalisation, Truncated SVD |
| 3 Cluster | `clustering.py` | Spectral (kNN + k-means labels, kNN + discretize, RBF, cosine kNN, Jaccard kNN), K-Means, Ward agglomerative |
| 4 Evaluate | `evaluation.py` | Silhouette, Davies-Bouldin, Calinski-Harabasz, cosine silhouette, Adjusted Rand Index between methods, top products and lift per cluster |
| 5 Visualise | `plots.py` | PCA and t-SNE (static grids and interactive Plotly), cluster-size bars |
| 6 Profile | `profiling.py` | spend, orders, product variety and order value per segment |
| 7 Products | `products.py` | product groups and "customers who bought X also bought" lookup |

## Dashboard (one scrollable page, following the wireframe in the project specification)
1. **Data**: upload, rows loaded, columns, rows after cleaning, step-by-step counts
2. **Feature engineering**: matrix shape, density, variance kept by the SVD
3. **Clustering controls**: number of clusters (2-10), spectral neighbours
4. **Evaluation metrics**: table for all seven methods, agreement matrix, silhouette against k
5. **PCA and t-SNE** side by side for any method, plus the cluster-size distribution
6. **Customer segments**: spend / frequency / variety table and representative products, CSV download
7. **Product groups and cross-selling**

Sidebar sliders: minimum products per customer, minimum customers per product, sample size, SVD components and value weighting.

## Tests
`python -m pytest -q` runs 21 tests: each cleaning rule (including a regression test for the
description-filter bug below), matrix shape for known thresholds, weighting, fixed-seed reproducibility
of cluster assignments and metrics, profiling arithmetic, the command-line script, and the dashboard
running end to end (including a too-strict-filters error case).

## Acknowledgement and what is new
The overall design (clean, customer x product matrix, L2 + SVD, three algorithms, three metrics,
Streamlit dashboard) follows an earlier student project on this dataset. This repository
differs as follows:

* **Cleaning bug fixed.** The earlier filter searched descriptions for `POSTAGE|DISCOUNT|ADJUSTMENT|M|DOTCOM`;
  the bare `M` removes every product whose description contains a capital M (468,353 rows kept).
  Filtering on the stock-code pattern keeps 802,632 rows.
* Sparse matrix with a choice of binary, order-count or log-count values.
* Five extra comparisons: cosine and Jaccard spectral variants (tests the similarity-measure hypothesis),
  silhouette against k, agreement between methods.
* Segment profiling, product grouping and cross-selling lookup.
* A documented notebook, a single-page dashboard with interactive plots, a command-line script,
  automated tests and pinned dependencies.
* Honest conclusions: on this data spectral clustering did not clearly outperform K-Means.

## Ethics
Secondary analysis of a public, anonymised dataset (UCI Online Retail II, CC BY 4.0). No participants
and no personal data are collected: data category A0, participant category C0 (to be confirmed with the supervisor).

## Limitations
Internal metrics are computed in the reduced SVD space the models were fitted on, which favours compact
clusters; the SVD keeps about a third of the variance; spectral clustering runs on a 2,000-customer sample
because memory grows with the square of the number of customers.
