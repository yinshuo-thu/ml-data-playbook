"""独立样本 A/B 均值差：Welch t 检验与 bootstrap 置信区间。"""

import numpy as np
from scipy.stats import ttest_ind


def main():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 300)
    b = rng.normal(0.2, 1.2, 320)
    differences = np.array(
        [
            rng.choice(b, len(b), replace=True).mean()
            - rng.choice(a, len(a), replace=True).mean()
            for _ in range(2000)
        ]
    )
    print("B-A", b.mean() - a.mean())
    print("95% percentile bootstrap CI", np.quantile(differences, [0.025, 0.975]))
    print("Welch test", ttest_ind(b, a, equal_var=False))
    # 重复观测/时序样本不能直接独立逐行重采样；考虑 cluster/block bootstrap。
    # p 值不是原假设为真的概率；多次检验需控制多重比较。


if __name__ == "__main__":
    main()
