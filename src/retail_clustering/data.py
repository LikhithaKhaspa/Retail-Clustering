"""Step 1 - load and clean the Online Retail II transactions."""
from __future__ import annotations

import pandas as pd

# Real products have a 5-digit stock code with an optional 1-2 letter suffix
# (e.g. 85123A). Everything else (POST, D, M, DOT, BANK CHARGES, AMAZONFEE...)
# is a fee, discount or manual adjustment and not a product.
PRODUCT_CODE = r"^\d{5}[A-Za-z]{0,2}$"

COLUMN_ALIASES = {"CustomerID": "Customer ID", "UnitPrice": "Price", "InvoiceNo": "Invoice"}
REQUIRED = ["Invoice", "StockCode", "Quantity", "Price", "Customer ID"]


def load_transactions(path_or_buffer) -> pd.DataFrame:
    """Read a CSV (or Excel) export of Online Retail II."""
    name = str(getattr(path_or_buffer, "name", path_or_buffer)).lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path_or_buffer)
    else:
        df = pd.read_csv(path_or_buffer, encoding="latin-1", low_memory=False)
    df = df.rename(columns=COLUMN_ALIASES)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")
    return df


def clean_transactions(df: pd.DataFrame, country: str | None = None):
    """Return (clean_df, report) where report counts rows dropped at each step."""
    report = {"raw rows": len(df)}
    if country and "Country" in df.columns:
        df = df[df["Country"] == country]
        report[f"after country == {country}"] = len(df)

    df = df[~df["Invoice"].astype(str).str.startswith("C")]  # cancellations
    report["after removing cancellations"] = len(df)

    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    report["after removing qty/price <= 0"] = len(df)

    df = df.dropna(subset=["Customer ID"])
    report["after dropping missing Customer ID"] = len(df)

    df = df[df["StockCode"].astype(str).str.match(PRODUCT_CODE)]
    report["after keeping real product codes"] = len(df)

    df = df.copy()
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["StockCode"] = df["StockCode"].astype(str).str.upper()
    return df, report
