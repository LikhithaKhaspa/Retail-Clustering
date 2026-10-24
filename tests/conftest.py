import os, sys
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def make_transactions(n_customers=300, n_products=120, n_segments=4, seed=0):
    """Small fake dataset in the Online Retail II schema with hidden customer segments."""
    rng = np.random.default_rng(seed)
    codes = [f"{10000 + i}" for i in range(n_products)]
    seg_of_product = rng.integers(0, n_segments, n_products)
    rows, inv = [], 1000
    for cust in range(n_customers):
        seg = rng.integers(0, n_segments)
        w = np.where(seg_of_product == seg, 6.0, 0.4)
        w = w / w.sum()
        for _ in range(rng.integers(3, 8)):
            inv += 1
            for j in rng.choice(n_products, rng.integers(5, 15), replace=False, p=w):
                rows.append((inv, codes[j], f"ITEM {j}", int(rng.integers(1, 6)),
                             "2010-01-01", round(float(rng.uniform(1, 9)), 2),
                             20000 + cust, "United Kingdom"))
    return pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity",
                                       "InvoiceDate", "Price", "Customer ID", "Country"])


@pytest.fixture(scope="session")
def clean_df():
    from retail_clustering import data
    df, _ = data.clean_transactions(make_transactions())
    return df


@pytest.fixture(scope="session")
def demo_csv(tmp_path_factory):
    """Raw (dirty) CSV including cancellations, fees and missing ids."""
    df = make_transactions()
    dirty = [df,
             df.sample(frac=0.03, random_state=1).assign(StockCode="POST", Description="POSTAGE"),
             df.sample(frac=0.02, random_state=2).assign(Invoice=lambda d: "C" + d["Invoice"].astype(str)),
             df.sample(frac=0.03, random_state=3).assign(**{"Customer ID": np.nan})]
    path = tmp_path_factory.mktemp("data") / "demo.csv"
    pd.concat(dirty).sample(frac=1, random_state=4).to_csv(path, index=False)
    return str(path)
