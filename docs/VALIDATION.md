# 验证记录

验证日期：2026-10-18。平台：macOS arm64；独立虚拟环境 Python 3.13.12。

| 检查 | 本地结果 |
|---|---|
| 核心离线示例 | 20 / 20 通过 |
| 深度学习离线示例 | 4 / 4 通过，CPU |
| 数据/模型/CLI 边界检查 | 11 / 11 通过 |
| 文档相对链接 | 已检查，包含在边界检查中 |
| House Prices Python 流程 | 已使用 1460 行训练、1459 行测试的真实数据副本运行 |
| House Prices 分步 Notebook | 全部代码单元已执行，Notebook 执行使用原有 Python 3.9 / sklearn 1.6 环境 |
| Python 语法和静态检查 | 编译检查与 Ruff F/E9 检查通过 |

## 真实案例结果

在上述独立环境中，开发集五折 CV 选择 `random_forest`，
预留 292 行的对数 RMSE 为 **0.146799**。
`submission.csv` 含 1459 行、Id 顺序与比赛测试数据一致、预测价格均为正且有限。
这是本地结果，没有上传 Kaggle。随机森林跨依赖版本可能出现细小差异；不要把版本差异当作策略改进。

MLP 的独立测试准确率为 **0.9611**，运行 20 个 epoch，
按开发验证损失选择 checkpoint，重载后从保存的缩放参数重新计算预测并检查一致性。
CNN/GRU 是结构示例，不声称经过充分调参。

## 复现命令

```bash
python -m scripts.smoke --suite core
python -m scripts.smoke --suite deep
python -m unittest discover -s tests -v
python -m cases.house_prices.train --data data/house_prices
```

前三条无需下载比赛数据；deep 需安装 `.[dl]`。真实比赛输入不随仓库分发。
Kaggle 下载命令和 GPU/MPS 路径未在本次验证中执行；只验证 CPU。
JupyterLab 的启动安装步骤是使用指引，不代表已在新虚拟环境中验证 UI。

## 直接依赖版本

| 依赖 | 版本 |
|---|---|
| numpy | 2.5.4 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |
| joblib | 1.6.0 |
| torch | 2.7.1 |
| pyarrow | 26.0.0 |
| openpyxl | 3.1.5 |

[机器可读结果](validation_results.json) · [验证版本清单](../requirements-tested.txt)

GitHub Actions 会在 push/PR 时执行检查。远端运行状态应以实际 Actions 页面为准；
本地通过不能替代未发生的远端检查。
