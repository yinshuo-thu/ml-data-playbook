"""用开发验证集置换重要性检查原始字段；相关变量会分摊重要性。"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from playbook.data import tabular
from playbook.modeling import build_pipeline


def main():
    x, y = tabular()
    xt, xv, yt, yv = train_test_split(x, y, test_size=0.2, random_state=42)
    model = build_pipeline(model="forest").fit(xt, yt)
    result = permutation_importance(
        model,
        xv,
        yv,
        n_repeats=5,
        random_state=42,
        scoring="neg_root_mean_squared_error",
    )
    print(
        pd.DataFrame(
            {
                "feature": x.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        ).sort_values("importance", ascending=False)
    )
    # 这是用于开发的验证集；据此选择特征后不再把它称作独立测试。


if __name__ == "__main__":
    main()
