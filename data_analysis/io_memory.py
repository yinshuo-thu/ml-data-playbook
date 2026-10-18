"""CSV 分块统计、可空整数、内存检查；Parquet 作为可选输出。"""

import argparse
from pathlib import Path
import pandas as pd
from playbook.data import orders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", action="store_true")
    args = parser.parse_args()
    out = Path("outputs/io")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "orders.csv"
    orders().to_csv(path, index=False)
    total = 0.0
    count = 0
    for chunk in pd.read_csv(path, usecols=["amount"], chunksize=3):
        values = pd.to_numeric(chunk["amount"], errors="coerce")
        total += values.sum()
        count += values.count()
    print("non-missing mean", total / count if count else None)
    x = pd.read_csv(path, dtype={"order_id": "Int64"})
    x["city"] = x["city"].astype("category")
    print(x.memory_usage(deep=True))
    if args.parquet:
        x.to_parquet(out / "orders.parquet", index=False)
        print(pd.read_parquet(out / "orders.parquet").head())


if __name__ == "__main__":
    main()
