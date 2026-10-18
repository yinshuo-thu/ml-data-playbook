"""Reusable fold-safe preprocessing. Keep fit inside each training fold."""

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def build_pipeline(task="regression", model="linear"):
    if task not in {"regression", "classification"} or model not in {
        "linear",
        "forest",
    }:
        raise ValueError("Unsupported task/model")
    numeric = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant", fill_value="Unknown", keep_empty_features=True
                ),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    prep = ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
        ],
        sparse_threshold=1.0,
    )
    if task == "regression":
        estimator = (
            Ridge(alpha=10, solver="lsqr")
            if model == "linear"
            else RandomForestRegressor(
                n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=42
            )
        )
    else:
        estimator = (
            LogisticRegression(max_iter=1000)
            if model == "linear"
            else RandomForestClassifier(
                n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=42
            )
        )
    return Pipeline([("prep", prep), ("model", estimator)])
