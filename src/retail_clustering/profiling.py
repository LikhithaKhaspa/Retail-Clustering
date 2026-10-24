"""Step 6 - describe each segment in business terms (spend, frequency, diversity)."""
from __future__ import annotations

import pandas as pd


def customer_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per customer: total spend, orders, distinct products, basket value."""
    g = df.assign(_value=df["Quantity"] * df["Price"]).groupby("Customer ID")
    t = pd.DataFrame({
        "total_spend": g["_value"].sum(),
        "n_orders": g["Invoice"].nunique(),
        "n_products": g["StockCode"].nunique(),
    })
    t["avg_order_value"] = t["total_spend"] / t["n_orders"]
    return t


def segment_summary(df: pd.DataFrame, customers, labels) -> pd.DataFrame:
    """Average behaviour of each cluster.

    Medians are shown next to means because a handful of wholesale customers
    can pull the mean spend of a cluster up by a large amount.
    """
    t = customer_table(df)
    lab = pd.Series(labels, index=pd.Index(customers, name="Customer ID"), name="Cluster")
    t = t.join(lab, how="inner")
    out = t.groupby("Cluster").agg(
        customers=("total_spend", "size"),
        mean_spend=("total_spend", "mean"),
        median_spend=("total_spend", "median"),
        mean_orders=("n_orders", "mean"),
        mean_products=("n_products", "mean"),
        mean_order_value=("avg_order_value", "mean"),
    )
    out["share_of_customers"] = out["customers"] / out["customers"].sum()
    return out.round(2).reset_index()
