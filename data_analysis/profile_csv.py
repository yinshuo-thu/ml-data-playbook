"""Inspect any CSV: field types, missingness, cardinality and numeric ranges."""

import argparse
from pathlib import Path

import pandas as pd


def profile(frame):
    """Return one audit row per input column; no automatic row/column deletion."""
    result = pd.DataFrame(
        {
            "dtype": frame.dtypes.astype(str),
            "missing_count": frame.isna().sum(),
            "missing_rate": frame.isna().mean(),
            "nunique": frame.nunique(),
        }
    )
    numeric = frame.select_dtypes("number")
    result["numeric_min"] = numeric.min()
    result["numeric_median"] = numeric.median()
    result["numeric_max"] = numeric.max()
    return result.sort_values("missing_rate", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/data_audit.csv"))
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args()
    frame = pd.read_csv(args.input, encoding=args.encoding)
    result = profile(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index_label="column")
    print(
        f"rows={len(frame)}, columns={frame.shape[1]}, duplicate_rows={frame.duplicated().sum()}"
    )
    print(result.head(20))
    print(args.output)


if __name__ == "__main__":
    main()
