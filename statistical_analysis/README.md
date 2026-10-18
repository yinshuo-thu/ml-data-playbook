# 统计分析

运行 `python -m statistical_analysis.bootstrap_ab`。
输入为两组独立合成观测；输出 B-A 的均值差、percentile bootstrap 95% 置信区间、Welch t 检验。

适合独立样本均值比较。重复用户、成组随机实验和时序观测需按独立采样单元处理，
例如 cluster bootstrap 或 block bootstrap。若是同一对象的前后对比，应考虑配对方法。
置信区间与 p 值不能替代效应大小、实验设计和多重比较控制。

[统计代码](bootstrap_ab.py) · [问题索引](../docs/TASK_INDEX.md)
