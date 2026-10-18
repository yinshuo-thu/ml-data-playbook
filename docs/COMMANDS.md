# 常用命令

命令均在仓库根目录执行，先激活 `.venv`。每个模块都可以单独运行。

| 我想做什么 | 命令 | 结果 |
|---|---|---|
| 数组形状与广播 | `python -m numpy_basics.arrays` | 终端输出 |
| 检查自己的 CSV | `python -m data_analysis.profile_csv --input data/demo/regression_train.csv` | 先生成演示数据；输出 `outputs/data_audit.csv` |
| 清洗脏数据 | `python -m data_analysis.cleaning` | `outputs/cleaning/*.csv` |
| 分组、透视与长宽转换 | `python -m data_analysis.groupby_reshape` | 终端输出 |
| 多表连接并检查行数 | `python -m data_analysis.joins` | 终端输出与未匹配行 |
| CSV 分块与内存 | `python -m data_analysis.io_memory` | `outputs/io/orders.csv` |
| Parquet 读写 | `python -m data_analysis.io_memory --parquet` | 需安装 `.[io]` |
| 日期、时区、字符串 | `python -m data_analysis.dates_strings` | 终端输出 |
| 自动生成 EDA 图 | `python -m visualization.eda_report` | `outputs/eda/report.png` |
| A/B 和 bootstrap | `python -m statistical_analysis.bootstrap_ab` | 均值差、置信区间、p 值 |
| SQL 查询 | `python -m sql.sqlite_analysis` | 终端输出 |
| 回归模型 | `python -m machine_learning.regression` | RMSE / MAE |
| 二分类概率 | `python -m machine_learning.classification` | AUC / AP / log loss |
| 自定义按组填补 | `python -m machine_learning.feature_engineering` | 训练折内分组中位数 |
| 超参数搜索 | `python -m machine_learning.tuning` | 开发 CV 和预留分数 |
| 置换重要性 | `python -m machine_learning.feature_importance` | 按原始字段排序 |
| 文本分类 | `python -m machine_learning.text_classification` | 新文本类别概率 |
| PCA 和聚类 | `python -m machine_learning.clustering_pca` | 轮廓系数 |
| 选择验证划分 | `python -m machine_learning.validation` | 四种划分对照 |
| 时序一步预测 | `python -m time_series.forecasting` | `outputs/time_series/predictions.csv` |
| Tensor 与自动求导 | `python -m deep_learning.tensors` | 需安装 `.[dl]` |
| 完整 MLP 训练 | `python -m deep_learning.mlp --epochs 20` | `outputs/mlp/metrics.json`、checkpoint |
| CNN 图像分类 | `python -m deep_learning.cnn --epochs 5` | 留出准确率 |
| GRU 序列预测 | `python -m deep_learning.sequence --epochs 8` | RMSE 与基准对比 |
| 核心示例全跑 | `python -m scripts.smoke --suite core` | `outputs/smoke_core.json` |
| 深度学习全跑 | `python -m scripts.smoke --suite deep` | `outputs/smoke_deep.json` |
| 边界测试 | `python -m unittest discover -s tests -v` | 测试结果 |

## 自己的 CSV

```bash
python -m scripts.make_demo_data
python -m playbook.cli train --input data/demo/regression_train.csv --target target --id-col id --output outputs/regression
python -m playbook.cli predict --input data/demo/regression_test.csv --model outputs/regression/model.joblib --output outputs/regression/predictions.csv
python -m playbook.cli train --input data/demo/classification_train.csv --target target --id-col id --task classification --model forest --output outputs/classification
python -m playbook.cli predict --input data/demo/classification_test.csv --model outputs/classification/model.joblib --output outputs/classification/predictions.csv
```

时间或分组划分：训练命令追加 `--time-col timestamp` 或 `--group-col customer_id`，替换成真实列名。
预测命令不会自动生成所有比赛的提交格式；官方列名、概率列、ID 和顺序需根据比赛规则检查。

## 搜索仓库

GitHub 页面按 `.` 可进入网页版编辑器，再全局搜索；也可用 GitHub 仓库搜索。
本地：

```bash
rg 'merge|many_to_one' data_analysis docs
rg 'rolling|shift|泄漏' time_series docs
rg 'CrossEntropyLoss|logits' deep_learning
python -m playbook.cli --help
python -m playbook.cli train --help
```
