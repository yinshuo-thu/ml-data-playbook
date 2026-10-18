# 按问题查代码

每一行都是可直接点击的入口。命令汇总见 [COMMANDS](COMMANDS.md)。

| 问题 / 关键词 | 看哪里 | 查什么 |
|---|---|---|
| 快速检查自己的 CSV | [CSV 体检](../data_analysis/profile_csv.py) | 缺失率、唯一值、数值范围 |
| CSV 读入后类型不对 / 数字混字符串 | [清洗](../data_analysis/cleaning.py) | `pd.to_numeric(errors="coerce")` |
| 缺失值怎么处理 | [清洗](../data_analysis/cleaning.py)、[Pipeline](../playbook/modeling.py) | 业务含义、训练折内填补 |
| 重复订单怎么查 | [清洗](../data_analysis/cleaning.py) | `duplicated`、隔离异常记录 |
| 筛选行列、批量赋值、索引对不齐 | [Pandas 速查](PANDAS_COOKBOOK.md) | `loc`、`iloc`、索引对齐 |
| 每个客户/城市的总量或均值 | [分组](../data_analysis/groupby_reshape.py) | `groupby().agg()` |
| 给每行加一个所属组的统计量 | [分组](../data_analysis/groupby_reshape.py) | `transform` |
| 长表变宽表 / 透视表 / melt | [分组](../data_analysis/groupby_reshape.py) | `pivot_table`、`melt` |
| merge 后行数翻倍 / 未匹配记录 | [连接](../data_analysis/joins.py) | `validate`、`indicator` |
| CSV 很大 / 内存不够 | [IO](../data_analysis/io_memory.py)、[速查](PANDAS_COOKBOOK.md) | `usecols`、`dtype`、`chunksize` |
| Excel / Parquet 怎么读 | [速查](PANDAS_COOKBOOK.md) | IO 可选依赖与格式 |
| 日期解析 / 时区 / 字符串提取 | [日期文本](../data_analysis/dates_strings.py) | `.dt`、`.str`、UTC |
| 广播报错 / axis / reshape | [NumPy](../numpy_basics/arrays.py) | shape、axis、向量化 |
| 直方图 / 相关性 / 缺失率图 | [绘图](../visualization/eda_report.py) | 保存图片、关闭 figure |
| A/B 差异显著吗 / 置信区间 | [统计](../statistical_analysis/bootstrap_ab.py) | Welch 检验、bootstrap 假设 |
| SQL 分组 / 窗口排序 | [SQL](../sql/sqlite_analysis.py) | SUM、ROW_NUMBER、绑定参数 |
| 从一张 CSV 训练并保存预测模型 | [CSV 工具](CSV_WORKFLOW.md) | train / predict |
| 数值和类别特征混合 | [预处理](../playbook/modeling.py) | ColumnTransformer |
| 验证集出现新类别 | [预处理](../playbook/modeling.py) | `handle_unknown="ignore"` |
| 线性回归 / Ridge / 随机森林 | [回归](../machine_learning/regression.py) | Pipeline、RMSE、MAE |
| 二分类 / 概率 / 正类 / 阈值 | [分类](../machine_learning/classification.py)、[指标](METRICS.md) | AUC、AP、混淆矩阵 |
| 按组填补 / 自定义 Transformer | [特征工程](../machine_learning/feature_engineering.py) | 训练组中位数、新组回退全局中位数 |
| 调参 / GridSearchCV | [搜索](../machine_learning/tuning.py) | 嵌套参数 `model__alpha` |
| 如何防止数据泄漏 | [验证手册](VALIDATION_AND_LEAKAGE.md) | fit 范围、时间可用性 |
| 同一客户多行如何划分 | [划分](../machine_learning/validation.py) | GroupKFold |
| 模型重要性 / 为什么预测错 | [重要性](../machine_learning/feature_importance.py)、[房价教程](../cases/house_prices/README.md) | permutation、残差分组 |
| 文本转特征 / TF-IDF | [文本](../machine_learning/text_classification.py) | 训练内学习词表 |
| 聚类 / PCA | [无监督](../machine_learning/clustering_pca.py) | 标准化、轮廓系数 |
| 昨日值 / 滚动均值 / 预测未来 | [时序](../time_series/forecasting.py)、[时序说明](../time_series/README.md) | shift 在 rolling 之前 |
| 从头做一个真实 Kaggle 比赛 | [House Prices](../cases/house_prices/README.md) | 下载、字典、CV、提交 |
| PyTorch 自动求导 / device | [Tensor](../deep_learning/tensors.py) | backward、detach |
| DataLoader / MLP / 早停 / 保存模型 | [MLP](../deep_learning/mlp.py) | train/eval、checkpoint |
| CNN / NCHW / 交叉熵 | [CNN](../deep_learning/cnn.py) | logits 与 long 标签 |
| GRU / 序列窗口 | [GRU](../deep_learning/sequence.py) | batch-time-feature |
| 模型训练后怎么加载预测 | [CSV 工具](CSV_WORKFLOW.md)、[MLP](../deep_learning/mlp.py) | joblib、state_dict |
| 报错、分数虚高或复现不了 | [排错](TROUBLESHOOTING.md) | 环境、类型、划分、随机性 |
