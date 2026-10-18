# 常见问题排查

| 症状 | 先检查 | 处理 |
|---|---|---|
| `No module named playbook` | 是否在仓库根目录、是否用对环境 | `python -m pip install -e .`；按文档用 `python -m ...` |
| `No module named torch` | 是否安装可选依赖 | `python -m pip install -e '.[dl]'` |
| Parquet/Excel 缺引擎 | 是否安装 IO 依赖 | `python -m pip install -e '.[io]'` |
| Python 版本不支持 | `python --version` | 新建 Python 3.10+ 环境，推荐 3.11/3.12 |
| 字符串不能转 float | 哪列混入文本 | `pd.to_numeric(errors="coerce")` 后审计新增缺失 |
| 新类别导致错误 | 训练/预测类别是否一致 | `handle_unknown="ignore"`；确认未知类别含义 |
| merge 行数增加 | 右表连接键是否唯一 | `validate="many_to_one"`，先检查重复键 |
| 赋值后没生效 | 是否链式赋值 | `df.loc[mask, col] = value` |
| 预测全为空 | 是否按索引错误对齐 | 核对 index，再决定是否 `.to_numpy()` 按位置赋值 |
| CV 极好、上线很差 | 时间、用户、目标、预处理泄漏 | 先查划分和可用时点 |
| 分类 accuracy 高但没用 | 正例比例 | 同时报 AP、召回、混淆矩阵与概率指标 |
| AUC 报只有一个类别 | 当前验证折是否包含两类 | 分层/调整划分；时序场景不可随意打乱补救 |
| `Found unknown categories` | 手工编码或未设置忽略新类别 | 用完整保存的 Pipeline |
| 内存不足 | 独热编码是否变为稠密矩阵 | 保持稀疏；高基数列慎重编码 |
| CUDA out of memory | batch、图是否被保留 | 减 batch；推理使用 inference_mode；记录标量用 `.item()` |
| 梯度没变化 | 是否调用 backward 和 step | 核对 zero_grad → forward → loss → backward → step |
| loss 出 NaN | 学习率、输入有限值、标签范围 | 排查源数据、减学习率、检查 loss 输入 |
| 同种子仍有差异 | GPU 算法、线程、库版本 | 记录环境；必要时使用确定性算法并接受性能代价 |

## PyTorch 形状

- MLP：输入 `(batch, features)`。
- CNN：输入 `(batch, channels, height, width)`。
- 本库 GRU：`batch_first=True`，输入 `(batch, time, features)`。
- 多分类 CrossEntropyLoss：输入为 logits `(batch, classes)`，标签为 long `(batch,)`。
- CrossEntropyLoss 前不再手动 softmax。
- 预测阶段使用 `model.eval()` 和 `torch.inference_mode()`，两者用途不同。

## 安装和模型保存

```bash
python -m pip check
python -c "import sys; print(sys.executable)"
python -c "import pandas, sklearn; print(pandas.__version__, sklearn.__version__)"
```

模型与预处理必须一起保存。加载前确认来源可信并尽量使用相同版本。
GPU checkpoint 可用 `map_location="cpu"` 加载；本库 MLP 保存 CPU state_dict 和标准化参数。
