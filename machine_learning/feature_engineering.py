"""训练折内按组中位数填补：自定义 sklearn transformer，不使用目标统计。"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from playbook.data import tabular
from playbook.modeling import build_pipeline


class GroupMedianImputer(TransformerMixin, BaseEstimator):
    """Fill a numeric field by group; unknown groups use the training median.

    Input/output are DataFrames. Use for ordinary input features, not target
    encoding. All-empty numeric training fields require a separate policy.
    """

    def __init__(self, group="segment", value="x0"):
        self.group = group
        self.value = value

    def fit(self, X, y=None):
        if X[self.value].notna().sum() == 0:
            raise ValueError("Numeric training field is entirely missing")
        self.global_median_ = X[self.value].median()
        self.group_medians_ = X.groupby(self.group)[self.value].median()
        return self

    def transform(self, X):
        check_is_fitted(self, ["global_median_", "group_medians_"])
        result = X.copy()
        result[f"{self.value}_was_missing"] = result[self.value].isna().astype(int)
        values = result[self.group].map(self.group_medians_).fillna(self.global_median_)
        result[self.value] = result[self.value].fillna(values)
        return result


def main():
    X, y = tabular()
    xt, xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
    estimator = Pipeline(
        [("group_fill", GroupMedianImputer()), ("regression", build_pipeline())]
    ).fit(xt, yt)
    print("validation RMSE", np.sqrt(np.mean((estimator.predict(xv) - yv) ** 2)))
    print("training group medians", estimator.named_steps["group_fill"].group_medians_)


if __name__ == "__main__":
    main()
