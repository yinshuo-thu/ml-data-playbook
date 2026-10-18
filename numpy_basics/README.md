# NumPy 基础

运行 `python -m numpy_basics.arrays`。
输入为固定随机种子的 5×3 数组；输出 shape、列均值、布尔筛选和矩阵乘法。

- `axis=0` 聚合行，得到每列统计；`axis=1` 聚合列，得到每行统计。
- `(n,d) - (d,)` 利用广播；不兼容形状会报错。
- `a * b` 逐元素乘，`a @ b` 矩阵乘；不要混淆。
- `np.isfinite` 同时排查 NaN 和正负无穷。
- `reshape` 改变形状，不会自动理解样本/时间/通道的业务轴。

[示例](arrays.py) · [问题索引](../docs/TASK_INDEX.md)
