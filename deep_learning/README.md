# 深度学习 · PyTorch

安装 `python -m pip install -e '.[dl]'`，命令在仓库根目录执行。
所有示例默认 CPU，使用离线数据，不下载预训练权重。

| 文件 | 内容 | 命令与输出 |
|---|---|---|
| [tensors.py](tensors.py) | tensor、梯度、设备、NumPy 转换 | `python -m deep_learning.tensors` |
| [mlp.py](mlp.py) | digits 多分类，训练/验证/测试，早停，checkpoint 重载 | `python -m deep_learning.mlp --epochs 20`；`outputs/mlp/` |
| [cnn.py](cnn.py) | 8×8 灰度图、卷积、池化、形状 | `python -m deep_learning.cnn --epochs 5` |
| [sequence.py](sequence.py) | GRU 时间窗口和一步预测 | `python -m deep_learning.sequence --epochs 8` |

## MLP 完整流程

1. 原始数据先分开发集与测试集；开发集再分训练与验证。
2. StandardScaler 只在训练部分 fit。
3. DataLoader 打乱训练样本，固定随机种子。
4. `model.train()` 开启训练行为；zero_grad → forward → loss → backward → step。
5. `model.eval()` 进入评估行为，配合 inference_mode 关闭梯度记录。
6. 按验证损失保存最佳参数，连续 5 次未改善就早停。
7. 恢复最佳参数，仅评估一次测试集。
8. 保存 state_dict 与缩放参数，重新加载并比较预测是否一致。

`python -m deep_learning.mlp --epochs 20 --device mps` 可在支持的 Apple 设备尝试加速。
`--device cuda` 需要兼容的 GPU 与安装包；并不保证跨设备逐位一致。

## 迁移到自己的数据

- 修改 Dataset / TensorDataset 的输入，保留划分、缩放拟合范围、评估和保存逻辑。
- 多分类：最后一层输出类别数；标签转为 `torch.long`。
- 回归：输出维度通常为 1，使用 MSELoss；检查 squeeze 后预测与标签形状一致。
- 大数据：自定义 Dataset 懒读取，避免把全部数据一次性放进 GPU。
- `num_workers=0` 是跨平台起点；多进程数据加载应保留 main 保护，并按机器测试吞吐。
- CNN 和 GRU 示例用于演示结构；若要据评分调参，请套用 MLP 的三段划分。

本库没有把简单 MLP 示例包装为通用深度学习平台；先从能验证正确性的最小流程扩展。
