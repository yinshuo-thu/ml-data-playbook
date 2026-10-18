"""数组形状、广播、轴、布尔索引、向量化；无文件输入。"""

import numpy as np


def main():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(5, 3))  # 5 个样本、3 个特征
    mean = x.mean(axis=0)  # 沿样本轴聚合 -> 每个特征的均值，shape=(3,)
    scaled = (x - mean) / x.std(axis=0)  # 广播（broadcasting）
    print("shape", x.shape, "column mean", mean)
    print("positive rows", x[x[:, 0] > 0])
    print("matrix product", x.T @ x)
    print("finite", np.isfinite(scaled).all())
    print("clipping", np.clip(x, -1, 1))


if __name__ == "__main__":
    main()
