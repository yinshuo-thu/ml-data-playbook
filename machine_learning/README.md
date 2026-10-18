# 机器学习

| 文件 | 用途 | 运行 |
|---|---|---|
| [regression.py](regression.py) | 混合特征回归基准 | `python -m machine_learning.regression` |
| [classification.py](classification.py) | 二分类概率与指标 | `python -m machine_learning.classification` |
| [validation.py](validation.py) | IID、分层、分组、时间划分 | `python -m machine_learning.validation` |
| [feature_engineering.py](feature_engineering.py) | 训练折内按组中位数填补 | `python -m machine_learning.feature_engineering` |
| [tuning.py](tuning.py) | 开发集网格搜索，预留集评估 | `python -m machine_learning.tuning` |
| [feature_importance.py](feature_importance.py) | 置换重要性 | `python -m machine_learning.feature_importance` |
| [text_classification.py](text_classification.py) | TF-IDF + 逻辑回归 | `python -m machine_learning.text_classification` |
| [clustering_pca.py](clustering_pca.py) | PCA 与 KMeans | `python -m machine_learning.clustering_pca` |

基础数据在运行时生成，不需要外部下载。文本示例是 API 演示，没有泛化评分。
完整混合特征预处理位于 [playbook/modeling.py](../playbook/modeling.py)。
自己的 CSV 请用 [通用入口](../docs/CSV_WORKFLOW.md)。

调参路径使用两层下划线：`model__alpha` 指 Pipeline 中 model 步骤的 alpha。
拟合、验证、选择与最终重训是不同阶段；不要把训练集拟合分数当作泛化分数。
