"""Deterministic synthetic fixtures: never downloads private or external data."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification


def orders():
    return pd.DataFrame(
        {
            "order_id": [1, 2, 2, 3, 4, 5, 6],
            "customer": [" Alice ", "Bob", "Bob", "alice", "陈", "Dana", "Erin"],
            "city": [
                "Shanghai",
                "Beijing",
                "Beijing",
                "Shanghai",
                None,
                "Shenzhen",
                "Beijing",
            ],
            "amount": ["100", "200", "200", "bad", "50", "5000", None],
            "date": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-02",
                "2025-01-03",
                "invalid",
                "2025-01-05",
                "2025-01-06",
            ],
        }
    )


def tabular(task="regression", n=500):
    rng = np.random.default_rng(42)
    if task == "regression":
        a, y = make_regression(n_samples=n, n_features=6, noise=15, random_state=42)
    else:
        a, y = make_classification(
            n_samples=n,
            n_features=6,
            n_informative=4,
            weights=[0.8, 0.2],
            random_state=42,
        )
    x = pd.DataFrame(a, columns=[f"x{i}" for i in range(6)])
    x["segment"] = rng.choice(["A", "B", "C"], n)
    x.loc[rng.choice(n, n // 10, replace=False), "x0"] = np.nan
    return x, pd.Series(y, name="target")
