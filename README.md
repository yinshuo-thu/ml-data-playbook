# ML & Data Playbook · 数据分析与建模速查库

面向日常查阅的中文代码手册：**从“我想做什么”找到代码、命令、输入输出和注意事项。**
包含 Pandas、NumPy、SQL、统计分析、scikit-learn、时间序列和 PyTorch；附真实 Kaggle 房价比赛完整案例。

## 从这里开始

- **遇到具体问题** → [按问题查代码](docs/TASK_INDEX.md)
- **只想复制命令** → [常用命令](docs/COMMANDS.md)
- **第一次配置** → [安装与环境](docs/SETUP.md)
- **从 CSV 训练并预测自己的数据** → [通用建模工具](docs/CSV_WORKFLOW.md)
- **从头学习一个真实比赛** → [House Prices](cases/house_prices/README.md)
- **程序报错或分数异常** → [排错手册](docs/TROUBLESHOOTING.md)

## 三分钟跑通

在仓库根目录执行，Python 3.10+。以下使用已安装的 uv 创建 Python 3.13 环境；无需 uv 的安装方式见[环境说明](docs/SETUP.md)。基础示例无需网络数据。

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
python -m data_analysis.cleaning
python -m machine_learning.regression
python -m scripts.make_demo_data
python -m playbook.cli train --input data/demo/regression_train.csv --target target --id-col id --output outputs/demo
python -m playbook.cli predict --input data/demo/regression_test.csv --model outputs/demo/model.joblib --output outputs/demo/predictions.csv
```

Windows PowerShell 用 `.venv\Scripts\Activate.ps1` 激活环境。输出统一保存在被 Git 忽略的 `outputs/`。

## 功能目录

| 目录 | 用途 | 入口 |
|---|---|---|
| `numpy_basics/` | 数组、广播、轴、向量化 | [数组示例](numpy_basics/arrays.py) |
| `data_analysis/` | 清洗、筛选、聚合、连接、日期、分块读写 | [目录说明](data_analysis/README.md) |
| `visualization/` | 分布、散点、缺失率、相关性 | [绘图示例](visualization/eda_report.py) |
| `statistical_analysis/` | A/B、置信区间、bootstrap | [统计示例](statistical_analysis/bootstrap_ab.py) |
| `sql/` | SQL 聚合、窗口函数、参数化查询 | [SQL 示例](sql/sqlite_analysis.py) |
| `machine_learning/` | 回归、分类、调参、重要性、文本、聚类 | [目录说明](machine_learning/README.md) |
| `time_series/` | lag、rolling、一步预测与时序验证 | [目录说明](time_series/README.md) |
| `deep_learning/` | Tensor、MLP、CNN、GRU、checkpoint | [目录说明](deep_learning/README.md) |
| `cases/house_prices/` | 真实 Kaggle 比赛端到端流程 | [完整教程](cases/house_prices/README.md) |
| `playbook/` | 可复用预处理、通用 CSV 命令行工具 | [使用说明](docs/CSV_WORKFLOW.md) |
| `docs/` | 索引、速查、验证、指标、排错 | [问题索引](docs/TASK_INDEX.md) |
| `scripts/`、`tests/` | 演示数据、全量运行检查、边界测试 | [验证说明](docs/VALIDATION.md) |

## 如何选择

- 业务表格任务：先用 Pandas 检查数据，建立正确划分，再用线性模型或树模型做基准。
- 分类：明确正类和错误成本，输出概率，按任务选择 AUC / AP / log loss / F1。
- 时间数据：先确定预测时点能拿到什么信息；不能默认随机划分。
- 图像/序列：先读 [PyTorch 使用说明](deep_learning/README.md)，基础环境无需安装 torch。
- 数据较大：优先 `usecols`、合理 `dtype`、Parquet、分块聚合，见 [Pandas 速查](docs/PANDAS_COOKBOOK.md)。

## 运行与复现

```bash
python -m scripts.smoke --suite core
python -m pip install -e '.[dl]'
python -m scripts.smoke --suite deep
python -m unittest discover -s tests -v
```

[验证记录](docs/VALIDATION.md) 明确区分已运行的示例、真实数据案例和需要额外依赖的功能。
示例中的合成数据仅用于演示 API；真实比赛分数仅是本地验证分数，没有承诺排行榜表现。

[学习路线](docs/LEARNING_PATH.md) · [验证与泄漏](docs/VALIDATION_AND_LEAKAGE.md) · [指标与模型选择](docs/METRICS.md) · [官方参考](docs/REFERENCES.md)

[GitHub 同步与维护](docs/GIT_SYNC.md) · [新增示例约定](CONTRIBUTING.md)
