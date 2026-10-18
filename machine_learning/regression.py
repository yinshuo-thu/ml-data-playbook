"""回归：混合类型 Pipeline、留出验证、RMSE 和 MAE。"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from playbook.data import tabular
from playbook.modeling import build_pipeline


def main():
    x, y = tabular()
    a, b = train_test_split(np.arange(len(x)), test_size=0.2, random_state=42)
    model = build_pipeline().fit(x.iloc[a], y.iloc[a])
    pred = model.predict(x.iloc[b])
    print(
        {
            "rmse": float(np.sqrt(mean_squared_error(y.iloc[b], pred))),
            "mae": float(mean_absolute_error(y.iloc[b], pred)),
        }
    )


if __name__ == "__main__":
    main()
