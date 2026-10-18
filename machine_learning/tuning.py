"""开发集 GridSearchCV 调参，最后仅评估一次预留集。"""

from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from playbook.data import tabular
from playbook.modeling import build_pipeline


def main():
    x, y = tabular()
    xt, xv, yt, yv = train_test_split(x, y, test_size=0.2, random_state=42)
    search = GridSearchCV(
        build_pipeline(),
        {"model__alpha": [0.1, 1, 10, 100]},
        cv=KFold(5, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        error_score="raise",
    )
    search.fit(xt, yt)
    print("best", search.best_params_, "development CV RMSE", -search.best_score_)
    print("holdout RMSE", np.sqrt(mean_squared_error(yv, search.predict(xv))))


if __name__ == "__main__":
    main()
