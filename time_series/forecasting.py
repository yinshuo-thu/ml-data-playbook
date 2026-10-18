"""逐时点一步预测：仅使用前一天已观测的目标构造 lag/rolling。"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def features(frame):
    x = frame.sort_values("date").copy()
    x["lag1"] = x["y"].shift(1)
    x["lag7"] = x["y"].shift(7)
    x["rolling7"] = x["y"].shift(1).rolling(7).mean()
    x["weekday"] = x["date"].dt.dayofweek
    return x.dropna()


def main():
    rng = np.random.default_rng(42)
    t = np.arange(500)
    x = features(
        pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=500),
                "y": np.sin(t * 2 * np.pi / 7) + 0.003 * t + rng.normal(0, 0.2, len(t)),
            }
        )
    )
    cols = ["lag1", "lag7", "rolling7", "weekday"]
    train = x.iloc[:390]
    valid = x.iloc[390:]
    model = HistGradientBoostingRegressor(max_iter=100, random_state=42).fit(
        train[cols], train.y
    )
    pred = model.predict(valid[cols])
    print("model RMSE", np.sqrt(mean_squared_error(valid.y, pred)))
    print("seasonal naive RMSE", np.sqrt(mean_squared_error(valid.y, valid.lag7)))
    out = Path("outputs/time_series")
    out.mkdir(parents=True, exist_ok=True)
    valid.assign(prediction=pred).to_csv(out / "predictions.csv", index=False)
    # 评估假设：每天预测前都拿到了昨天真实 y。一次性多步预测不能这么使用验证期 y。


if __name__ == "__main__":
    main()
