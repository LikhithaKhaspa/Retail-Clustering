# Build it yourself, step by step

Type each piece, run it, and look at the output before moving on.

## 0. The idea
Group customers who buy similar products. Each customer is a 0/1 vector ("bought product X?").
Shrink it with SVD, then cluster. Spectral clustering builds a neighbour graph between customers
and clusters using eigenvectors of the graph Laplacian, so it can follow curved or elongated
groups that K-Means (round blobs) cannot.

## 1. Setup
Make a venv, install `requirements.txt`, create `data/ src/ outputs/`. Load the data in a
notebook: `df.head()`, `df.info()`, `df.isna().sum()`, `df.Country.value_counts()`.

## 2. Clean (`data.py`)
Remove, printing the row count after each step: invoices starting with C; Quantity <= 0 or
Price <= 0; missing Customer ID; non-product codes (POST, D, M, DOT, BANK CHARGES). Product
codes look like 5 digits plus an optional letter suffix.

## 3. Customer x product matrix (`features.py`)
Build a scipy sparse matrix from (customer, product) pairs. Keep products bought by >= 50
customers and customers with >= 20 distinct products.

## 4. Normalise and reduce
`normalize(M, norm="l2")`, then `TruncatedSVD(50)`. Print the explained variance.

## 5. Cluster (`clustering.py`)
Start with K-Means. Then add SpectralClustering (nearest_neighbors), then RBF, then
Agglomerative. Use `random_state=42`.

## 6. Evaluate (`evaluation.py`)
Silhouette (higher better), Davies-Bouldin (lower), Calinski-Harabasz (higher). Then compute
which products each cluster buys far more than average (lift) and name the segments.

## 7. Visualise (`plots.py`)
PCA scatter per method, t-SNE the same (slow, keep to ~3000 points), grouped bar of sizes.

## 8. Dashboard (`app.py`)
`st.file_uploader`, sidebar sliders, `st.tabs`, `st.cache_data` on the expensive steps.

## 9. Make it yours
1. Choose k with a silhouette sweep.
2. Try cosine/Jaccard affinity (the slides hypothesise they beat Euclidean).
3. Add RFM features from InvoiceDate and Quantity*Price.
4. Cluster products instead (transpose the matrix) for cross-selling groups.
5. Check stability across seeds with Adjusted Rand Index.

## 10. Extensions already built in this repo
* `notebooks/customer_segmentation.ipynb` walks through the whole workflow with explanations.
* `profiling.py` describes each segment by spend, orders and product variety.
* `tests/` holds the automated tests (`python -m pytest -q`).
* `clustering.py` also has cosine-kNN and Jaccard-kNN spectral variants (tests hypothesis 2 from the slides).
* `products.py` clusters products (transpose the matrix) and offers a "customers who bought X also bought" lookup;
  run `python run_pipeline.py --data data/online_retail_II.csv --products`.

## Pitfalls
* Spectral clustering is O(n^2) memory, hence the ~2000-customer sample.
* "Graph is not fully connected" warning: raise n_neighbors.
* Silhouette in SVD space flatters K-Means.
* Windows: close any CSV open in Excel before rerunning, or saving to `outputs/` fails with PermissionError.
