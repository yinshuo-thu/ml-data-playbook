"""Train/predict a single-table CSV baseline with explicit split semantics."""

import argparse
from pathlib import Path
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    log_loss,
)
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from .modeling import build_pipeline


def read_table(path):
    frame = pd.read_csv(path)
    if frame.empty or not frame.columns.is_unique:
        raise ValueError("Input must have rows and unique column names")
    return frame


def train(args):
    df = read_table(args.input)
    if args.target not in df or df[args.target].isna().any():
        raise ValueError("Target is absent or has missing values")
    for col in [args.id_col, args.time_col, args.group_col]:
        if col and col not in df:
            raise ValueError(f"Missing metadata column: {col}")
    if args.target in {args.id_col, args.time_col, args.group_col}:
        raise ValueError("Target cannot also be an ID/time/group column")
    if args.id_col and (df[args.id_col].isna().any() or not df[args.id_col].is_unique):
        raise ValueError("ID must be non-null and unique")
    if args.group_col and args.time_col:
        raise ValueError(
            "Combined group/time splitting requires a custom scheme; see docs"
        )
    if args.time_col:
        df[args.time_col] = pd.to_datetime(df[args.time_col], errors="raise", utc=True)
        if df[args.time_col].isna().any():
            raise ValueError("Time column has missing timestamps")
        df = df.sort_values(args.time_col, kind="stable").reset_index(drop=True)
    drop = [args.target] + [
        c for c in [args.id_col, args.time_col, args.group_col] if c
    ]
    X = df.drop(columns=list(dict.fromkeys(drop)))
    if X.shape[1] == 0:
        raise ValueError("No predictors left after dropping metadata")
    y = df[args.target]
    if args.task == "regression":
        y = pd.to_numeric(y, errors="raise")
        if not np.isfinite(y).all():
            raise ValueError("Regression target must be finite")
    elif y.nunique() < 2:
        raise ValueError("Classification needs at least two classes")
    ix = np.arange(len(df))
    if args.time_col:
        # Split complete timestamps: never place the same timestamp on both sides.
        times = df[args.time_col].drop_duplicates().sort_values()
        if len(times) < 2:
            raise ValueError("Time split requires at least two distinct timestamps")
        cut = max(1, min(len(times) - 1, int(0.8 * len(times))))
        threshold = times.iloc[cut]
        a = ix[df[args.time_col].lt(threshold)]
        b = ix[df[args.time_col].ge(threshold)]
    elif args.group_col:
        if df[args.group_col].isna().any():
            raise ValueError("Group column has missing groups")
        a, b = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
                X, y, groups=df[args.group_col]
            )
        )
    else:
        a, b = train_test_split(
            ix,
            test_size=0.2,
            random_state=42,
            stratify=y if args.task == "classification" else None,
        )
    estimator = build_pipeline(args.task, args.model).fit(X.iloc[a], y.iloc[a])
    pred = estimator.predict(X.iloc[b])
    if args.task == "regression":
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y.iloc[b], pred))),
            "mae": float(mean_absolute_error(y.iloc[b], pred)),
        }
    else:
        classes = estimator.classes_
        if not set(y.iloc[b]).issubset(set(classes)):
            raise ValueError("Validation contains classes absent from training")
        metrics = {
            "accuracy": float(accuracy_score(y.iloc[b], pred)),
            "log_loss": float(
                log_loss(y.iloc[b], estimator.predict_proba(X.iloc[b]), labels=classes)
            ),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    # Refit a fresh model on all labels only after the holdout assessment.
    final = clone(estimator).fit(X, y)
    bundle = {
        "pipeline": final,
        "columns": X.columns.tolist(),
        "task": args.task,
        "id_col": args.id_col,
        "target": args.target,
        "dtypes": X.dtypes.astype(str).to_dict(),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    joblib.dump(bundle, args.output / "model.joblib")
    detail = pd.DataFrame(
        {"source_row": b, "actual": y.iloc[b].to_numpy(), "prediction": pred}
    )
    detail.to_csv(args.output / "validation_predictions.csv", index=False)
    report = {
        "task": args.task,
        "model": args.model,
        "train_rows": len(a),
        "valid_rows": len(b),
        "split": "time" if args.time_col else "group" if args.group_col else "random",
        "metrics": metrics,
        "versions": bundle["versions"],
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def predict(args):
    # Load only artifacts you trust: joblib is a pickle-based executable format.
    bundle = joblib.load(args.model)
    df = read_table(args.input)
    missing = set(bundle["columns"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing predictors: {sorted(missing)}")
    # Reject changed storage types instead of silently treating category IDs as measurements.
    for col, dtype in bundle["dtypes"].items():
        if pd.api.types.is_numeric_dtype(np.dtype(dtype)):
            df[col] = pd.to_numeric(df[col], errors="raise")
        else:
            df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    X = df[bundle["columns"]]
    output = pd.DataFrame({"prediction": bundle["pipeline"].predict(X)})
    id_col = bundle["id_col"]
    if id_col:
        if id_col not in df or df[id_col].isna().any() or not df[id_col].is_unique:
            raise ValueError("Prediction input must contain unique non-null IDs")
        output.insert(0, id_col, df[id_col].to_numpy())
    if bundle["task"] == "classification":
        proba = bundle["pipeline"].predict_proba(X)
        for i, cls in enumerate(bundle["pipeline"].classes_):
            output[f"probability_{cls}"] = proba[:, i]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output)} rows to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "train", help="Train on CSV; evaluate holdout then refit all rows"
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--target", required=True)
    p.add_argument(
        "--task", choices=["regression", "classification"], default="regression"
    )
    p.add_argument("--model", choices=["linear", "forest"], default="linear")
    p.add_argument("--id-col")
    p.add_argument("--time-col")
    p.add_argument("--group-col")
    p.add_argument("--output", type=Path, default=Path("outputs/custom"))
    p.set_defaults(func=train)
    p = sub.add_parser("predict", help="Predict with a trusted local model artifact")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    p.set_defaults(func=predict)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
