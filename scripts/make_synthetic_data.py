"""Generate a small fake dataset in the Online Retail II schema.

Use it to test the pipeline without downloading the real 90 MB file.
Customers belong to hidden segments that prefer different product ranges,
and ~5% of rows are cancellations / fees so the cleaning step has work to do.
"""
import argparse
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--out", default="data/synthetic_retail.csv")
p.add_argument("--customers", type=int, default=1500)
p.add_argument("--products", type=int, default=400)
p.add_argument("--segments", type=int, default=6)
a = p.parse_args()

rng = np.random.default_rng(0)
codes = [f"{10000 + i}" for i in range(a.products)]
names = [f"ITEM {i} {rng.choice(['MUG','BAG','CANDLE','CARD','LAMP','TIN'])}" for i in range(a.products)]
seg_of_prod = rng.integers(0, a.segments, a.products)
rows, inv = [], 500000
for cust in range(a.customers):
    seg = rng.integers(0, a.segments)
    w = np.where(seg_of_prod == seg, 6.0, 0.4) * rng.gamma(2, 1, a.products)
    w /= w.sum()
    for _ in range(rng.integers(2, 12)):                       # invoices
        inv += 1
        for j in rng.choice(a.products, rng.integers(3, 15), replace=False, p=w):
            rows.append((inv, codes[j], names[j], int(rng.integers(1, 12)),
                         "2010-06-01 10:00", round(float(rng.uniform(0.5, 9)), 2),
                         12000 + cust, "United Kingdom"))
df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity",
                                 "InvoiceDate", "Price", "Customer ID", "Country"])
junk = df.sample(frac=0.03, random_state=1).copy()
junk["StockCode"] = rng.choice(["POST", "D", "M", "DOT", "BANK CHARGES"], len(junk))
junk["Description"] = "POSTAGE"
canc = df.sample(frac=0.02, random_state=2).copy()
canc["Invoice"] = "C" + canc["Invoice"].astype(str); canc["Quantity"] *= -1
nocust = df.sample(frac=0.03, random_state=3).copy(); nocust["Customer ID"] = np.nan
out = pd.concat([df, junk, canc, nocust]).sample(frac=1, random_state=4)
out.to_csv(a.out, index=False)
print(f"wrote {a.out}: {out.shape}")
