"""House Prices: run with Python 3, pandas, numpy and scikit-learn>=1.2.

Download competition CSV files into data/house_prices, or pass --data.
Scores are local validation results, not Kaggle leaderboard results.
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def make_features(raw):
    """Fixed, row-local rules: no statistics or target values are learned here."""
    x = raw.copy()
    for col in ["MSSubClass", "MoSold"]:
        x[col] = x[col].astype(str)
    # The dictionary defines NA in these fields as absence of the facility.
    absent = [
        "Alley",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
    ]
    x[absent] = x[absent].fillna("None")
    x["LotFrontageMissing"] = x["LotFrontage"].isna().astype(int)
    # min_count preserves uncertainty instead of silently treating NA as zero.
    x["TotalSF"] = x[["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]].sum(axis=1, min_count=3)
    x["TotalBath"] = (
        x["FullBath"]
        + 0.5 * x["HalfBath"]
        + x["BsmtFullBath"]
        + 0.5 * x["BsmtHalfBath"]
    )
    x["HouseAge"] = x["YrSold"] - x["YearBuilt"]
    x["SinceRemodel"] = x["YrSold"] - x["YearRemodAdd"]
    # Mark inconsistent ages as unknown rather than inventing a valid age.
    for col in ["HouseAge", "SinceRemodel"]:
        x[col] = x[col].mask(x[col] < 0)
    return x


def build_pipeline(model):
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    prep = ColumnTransformer(
        [
            ("num", numeric, make_column_selector(dtype_include=np.number)),
            ("cat", categorical, make_column_selector(dtype_exclude=np.number)),
        ],
        sparse_threshold=1.0,
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(make_features, validate=False)),
            ("prep", prep),
            ("model", model),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/house_prices"))
    args = parser.parse_args()
    out = Path("outputs/house_prices")
    out.mkdir(parents=True, exist_ok=True)
    if not (args.data / "train.csv").exists() or not (args.data / "test.csv").exists():
        raise FileNotFoundError(
            "Download train.csv/test.csv first; see cases/house_prices/README.md"
        )
    train = pd.read_csv(args.data / "train.csv")
    test = pd.read_csv(args.data / "test.csv")
    assert train["Id"].is_unique and test["Id"].is_unique
    assert set(train["Id"]).isdisjoint(test["Id"])
    assert train["SalePrice"].notna().all() and train["SalePrice"].gt(0).all()
    assert set(train.columns) - {"SalePrice"} == set(test.columns)
    audit = pd.DataFrame(
        {
            "dtype": train.dtypes.astype(str),
            "missing_rate": train.isna().mean(),
            "nunique": train.nunique(),
        }
    ).sort_values("missing_rate", ascending=False)
    audit.to_csv(out / "data_audit.csv")
    X = train.drop(columns=["Id", "SalePrice"])
    y = np.log(train["SalePrice"])  # Exact natural-log metric space.
    X_dev, X_hold, y_dev, y_hold = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {"dummy": build_pipeline(DummyRegressor(strategy="mean"))}
    for alpha in [1.0, 10.0, 100.0]:
        candidates[f"ridge_{alpha:g}"] = build_pipeline(
            Ridge(alpha=alpha, solver="lsqr")
        )
    candidates["random_forest"] = build_pipeline(
        RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )
    )
    rows = []
    for name, pipe in candidates.items():
        scores = -cross_val_score(
            pipe,
            X_dev,
            y_dev,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
            error_score="raise",
        )
        assert np.isfinite(scores).all()
        rows.append(
            {
                "model": name,
                "cv_mean": scores.mean(),
                "cv_std": scores.std(),
                "fold_scores": scores.tolist(),
            }
        )
        print(name, "CV", scores.mean().round(6), flush=True)
    table = pd.DataFrame(rows).sort_values("cv_mean")
    table.to_csv(out / "cv_scores.csv", index=False)
    winner = table.iloc[0]["model"]
    selected = clone(candidates[winner]).fit(X_dev, y_dev)
    hold_pred = selected.predict(X_hold)
    assert np.isfinite(hold_pred).all()
    hold_rmse = np.sqrt(mean_squared_error(y_hold, hold_pred))
    errors = X_hold[["Neighborhood", "OverallQual", "GrLivArea"]].copy()
    errors["Id"] = train.loc[X_hold.index, "Id"]
    errors["actual"] = np.exp(y_hold)
    errors["predicted"] = np.exp(hold_pred)
    errors["log_error"] = hold_pred - y_hold
    errors["abs_log_error"] = errors["log_error"].abs()
    errors.sort_values("abs_log_error", ascending=False).to_csv(
        out / "holdout_errors.csv", index=False
    )
    errors.groupby("Neighborhood").agg(
        n=("abs_log_error", "size"),
        mean_abs_log_error=("abs_log_error", "mean"),
    ).to_csv(out / "holdout_by_neighborhood.csv")
    # Refit chosen settings on every labelled row for the final submission file.
    final = clone(candidates[winner]).fit(X, y)
    predictions = np.exp(final.predict(test.drop(columns="Id")))
    submission = pd.DataFrame({"Id": test["Id"], "SalePrice": predictions})
    assert len(submission) == len(test)
    assert submission["Id"].equals(test["Id"])
    assert np.isfinite(predictions).all() and (predictions > 0).all()
    submission.to_csv(out / "submission.csv", index=False)
    summary = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "development_rows": len(X_dev),
        "holdout_rows": len(X_hold),
        "selected_by_development_cv": winner,
        "holdout_log_rmse": float(hold_rmse),
        "note": "Local validation only. No Kaggle submission made.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
