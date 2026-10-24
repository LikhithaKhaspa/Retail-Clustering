"""Functional and reproducibility tests for the clustering pipeline.

Run from the project root:   python -m pytest -q
"""
import io, os, subprocess, sys

import numpy as np
import pandas as pd
import pytest

from retail_clustering import (clustering, data, evaluation, features, plots,
                               products, profiling)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def row(inv, code, qty=1, price=2.0, cust=1, desc="WIDGET"):
    return dict(Invoice=inv, StockCode=code, Description=desc, Quantity=qty, InvoiceDate="2010-01-01",
                Price=price, **{"Customer ID": cust}, Country="United Kingdom")


# ------------------------------------------------------------------ cleaning
def test_cleaning_removes_each_invalid_category():
    raw = pd.DataFrame([
        row("1001", "85123A"),                          # valid
        row("C1002", "85123A"),                         # cancelled invoice
        row("1003", "85123A", qty=-1),                  # negative quantity
        row("1004", "85123A", price=0.0),               # zero price
        row("1005", "85123A", cust=np.nan),             # missing customer
        row("1006", "POST", desc="POSTAGE"),            # fee
        row("1007", "M", desc="Manual"),                # manual adjustment
        row("1008", "BANK CHARGES", desc="Bank Charges"),
        row("1009", "22423", desc="REGENCY CAKESTAND 3 TIER"),   # real product containing capital M
    ])
    clean, report = data.clean_transactions(raw)
    assert set(clean["StockCode"]) == {"85123A", "22423"}
    assert len(clean) == 2
    assert report["raw rows"] == 9


def test_description_with_capital_m_is_kept():
    """Regression test for the original filter that removed any description containing 'M'."""
    raw = pd.DataFrame([row("1", "22423", desc="MEDIUM CERAMIC TOP STORAGE JAR")])
    clean, _ = data.clean_transactions(raw)
    assert len(clean) == 1


def test_cleaning_report_never_increases():
    clean, report = data.clean_transactions(data.load_transactions(
        io.StringIO(make_csv_text())))
    counts = list(report.values())
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == len(clean) > 0


def make_csv_text():
    from conftest import make_transactions
    return make_transactions(50, 30).to_csv(index=False)


def test_missing_columns_raise_clear_error():
    with pytest.raises(ValueError, match="Missing columns"):
        data.load_transactions(io.StringIO("a,b\n1,2\n"))


# ------------------------------------------------------------------- matrix
def small_df():
    # 6 customers, 5 products. Customer 6 buys one product; product E is bought by one customer.
    rows = []
    for cust in range(1, 6):
        for prod in ["11111", "22222", "33333", "44444"]:
            rows.append(row(f"I{cust}", prod, cust=cust))
    rows.append(row("I1", "55555", cust=1))                   # product E, only customer 1
    rows.append(row("I6", "11111", cust=6))                   # customer 6, only one product
    return pd.DataFrame(rows)


def test_matrix_shape_after_known_thresholds():
    M, cust, prod = features.build_customer_product_matrix(
        small_df(), min_products_per_customer=2, min_customers_per_product=2)
    assert M.shape == (5, 4)                       # customer 6 and product E are filtered out
    assert sorted(cust) == [1, 2, 3, 4, 5]
    assert "55555" not in prod
    assert set(np.unique(M.toarray())) == {1.0}


def test_weighting_counts_separate_invoices():
    rows = [row("A", "11111"), row("B", "11111"), row("C", "11111"), row("A", "22222")]
    df = pd.DataFrame(rows)
    for weighting, expected in [("binary", 1.0), ("invoices", 3.0), ("log-invoices", np.log1p(3))]:
        M, cust, prod = features.build_customer_product_matrix(df, 1, 1, weighting)
        j = list(prod).index("11111")
        assert M[0, j] == pytest.approx(expected)


def test_filters_are_independent_of_weighting(clean_df):
    shapes = {w: features.build_customer_product_matrix(clean_df, 10, 10, w)[0].shape
              for w in features.WEIGHTINGS}
    assert len(set(shapes.values())) == 1


def test_unknown_weighting_rejected(clean_df):
    with pytest.raises(ValueError):
        features.build_customer_product_matrix(clean_df, 1, 1, "bogus")


def test_reduced_features_are_dense_and_finite(clean_df):
    fs = features.make_features(clean_df, 5, 5, 200, 100, n_components=20)
    assert fs.X_reduced.shape == (fs.matrix.shape[0], 20)
    assert np.isfinite(fs.X_reduced).all()
    assert 0 < fs.explained_variance <= 1


# ----------------------------------------------------------- reproducibility
def test_same_seed_gives_identical_assignments_and_metrics(clean_df):
    def run():
        fs = features.make_features(clean_df, 5, 5, 200, 100, n_components=20)
        labels = clustering.run_all(fs.X_reduced, 4, M=fs.binary)
        return fs, labels, evaluation.evaluate(fs.X_reduced, labels)

    fs1, l1, m1 = run()
    fs2, l2, m2 = run()
    assert np.array_equal(fs1.customers, fs2.customers)
    for name in l1:
        assert np.array_equal(l1[name], l2[name]), name
    pd.testing.assert_frame_equal(m1.drop(columns="Cluster sizes"), m2.drop(columns="Cluster sizes"))


def test_all_methods_return_requested_number_of_clusters(clean_df):
    fs = features.make_features(clean_df, 5, 5, 200, 100, n_components=20)
    labels = clustering.run_all(fs.X_reduced, 4, M=fs.binary)
    assert len(labels) == 7                       # 5 original variants + cosine + Jaccard
    for name, lab in labels.items():
        assert len(lab) == fs.X_reduced.shape[0]
        assert len(set(lab)) == 4, name


def test_jaccard_does_not_modify_caller_matrix(clean_df):
    fs = features.make_features(clean_df, 5, 5, 200, 100, n_components=20, weighting="invoices")
    before = fs.matrix.copy()
    clustering.jaccard_affinity(fs.matrix)
    assert (before != fs.matrix).nnz == 0


def test_planted_segments_are_recovered(clean_df):
    """The synthetic data have 4 hidden segments, so K-Means should score well on them."""
    fs = features.make_features(clean_df, 5, 5, 300, 120, n_components=20)
    labels = clustering.run_all(fs.X_reduced, 4, M=fs.binary)
    m = evaluation.evaluate(fs.X_reduced, labels).set_index("Method")
    assert m.loc["KMeans", "Silhouette (higher)"] > 0.15


# ------------------------------------------------------- evaluation/profiling
def test_agreement_matrix_has_unit_diagonal():
    labels = {"a": np.array([0, 0, 1, 1]), "b": np.array([1, 1, 0, 0]), "c": np.array([0, 1, 0, 1])}
    ari = evaluation.agreement(labels)
    assert np.allclose(np.diag(ari), 1)
    assert ari.loc["a", "b"] == pytest.approx(1.0)           # same grouping, different names


def test_segment_summary_known_values():
    df = pd.DataFrame([row("A", "11111", qty=2, price=5.0, cust=1),     # spend 10
                       row("B", "22222", qty=1, price=20.0, cust=1),    # spend 20, 2 orders
                       row("C", "11111", qty=3, price=10.0, cust=2)])   # spend 30, 1 order
    s = profiling.segment_summary(df, [1, 2], [0, 1]).set_index("Cluster")
    assert s.loc[0, "mean_spend"] == 30 and s.loc[0, "mean_orders"] == 2
    assert s.loc[0, "mean_products"] == 2 and s.loc[0, "mean_order_value"] == 15
    assert s.loc[1, "mean_spend"] == 30 and s.loc[1, "mean_orders"] == 1


def test_top_products_lift_is_above_one_for_signature_items(clean_df):
    fs = features.make_features(clean_df, 5, 5, 300, 120, n_components=20)
    lab = clustering.run_all(fs.X_reduced, 4, M=fs.binary)["KMeans"]
    top = evaluation.top_products_per_cluster(fs.binary, fs.products, lab, top_n=3)
    assert (top["Lift"] > 1).all()
    assert top["Share of cluster"].between(0, 1).all()


# ------------------------------------------------------------------ products
def test_similar_products_exclude_the_query(clean_df):
    pf = products.make_product_features(clean_df, 5, 5, max_products=100, n_components=15)
    code = pf.products[0]
    sim = products.similar_products(pf, code, n=5)
    assert code not in set(sim["StockCode"]) and len(sim) == 5
    assert sim["Similarity"].is_monotonic_decreasing


def test_product_groups_cover_every_product(clean_df):
    pf = products.make_product_features(clean_df, 5, 5, max_products=100, n_components=15)
    groups = products.cluster_products(pf, 5)
    assert set(groups) == {"KMeans", "Spectral (kNN)", "Spectral (Jaccard kNN)"}
    for lab in groups.values():
        assert len(lab) == len(pf.products)


# ---------------------------------------------------------------- end to end
def test_command_line_pipeline_runs(demo_csv, tmp_path):
    out = tmp_path / "out"
    res = subprocess.run([sys.executable, os.path.join(ROOT, "run_pipeline.py"), "--data", demo_csv,
                          "--k", "4", "--max-customers", "250", "--max-products", "100",
                          "--min-products-per-customer", "5", "--min-customers-per-product", "5",
                          "--svd-components", "15", "--no-tsne", "--out", str(out)],
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-1500:]
    metrics = pd.read_csv(out / "metrics.csv")
    assert len(metrics) == 7
    for f in ["cluster_profiles.csv", "segment_summary.csv", "customer_labels.csv", "pca.png", "sizes.png"]:
        assert (out / f).exists(), f


def test_streamlit_app_runs_end_to_end(demo_csv, monkeypatch):
    st = pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    raw = open(demo_csv, "rb").read()

    class FakeUpload(io.BytesIO):
        name = "demo.csv"
        size = len(raw)

        def getvalue(self):
            return raw

    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: FakeUpload(raw))
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=600)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    headers = [h.value for h in at.header]
    assert any("Evaluation" in h for h in headers) and any("Product groups" in h for h in headers)
    assert len(at.dataframe) >= 4

    # moving a slider must also re-run cleanly
    at.slider[[s.label for s in at.slider].index("Number of clusters (k)")].set_value(4).run()
    assert not at.exception, [e.value for e in at.exception]


def test_app_reports_friendly_error_when_filters_too_strict(demo_csv, monkeypatch):
    st = pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest
    raw = open(demo_csv, "rb").read()

    class FakeUpload(io.BytesIO):
        name = "demo.csv"
        size = len(raw)

        def getvalue(self):
            return raw

    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: FakeUpload(raw))
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=600)
    at.run()
    at.slider[[s.label for s in at.slider].index("Min customers per product")].set_value(100).run()
    at.slider[[s.label for s in at.slider].index("Min products per customer")].set_value(60).run()
    assert not at.exception                       # no crash, a readable error instead
    assert len(at.error) >= 1
