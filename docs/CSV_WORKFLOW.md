# 通用 CSV 建模工具

适合：单张 CSV、每行一个样本、目标列完整、输入特征为数值或类别。
不自动理解日期、长文本、图像、因果关系、样本权重或标签重叠；需要这些能力时从示例扩展。

## 演示输入

```bash
python -m scripts.make_demo_data
```

`data/demo/regression_train.csv`：500 行，`id`、6 个数值列、`segment` 类别列、`target`。
`regression_test.csv`：100 行，无 target。classification 文件对应二分类。

训练前可先体检：

```bash
python -m data_analysis.profile_csv --input data/demo/regression_train.csv
```

输出到 `outputs/data_audit.csv`。此工具整表读取，超大数据先参考分块 IO。

## 训练、评估与保存

```bash
python -m playbook.cli train --input data/demo/regression_train.csv --target target --id-col id --output outputs/my_model
```

- 默认 80% 训练、20% 验证；分类采用分层划分。
- 数值中位数填补、缺失标记与标准化；类别 Unknown 填补和独热编码。
- 先在训练部分拟合并计算验证指标，再克隆模型、用全部标签重训。
- `outputs/my_model/model.joblib` 包含完整 Pipeline、字段顺序、类型和版本。
- `metrics.json` 记录划分和分数；`validation_predictions.csv` 保留验证明细。
- `source_row` 是读取后的行位置；时间划分时表示按时间排序后的行位置。

分类与树模型：

```bash
python -m playbook.cli train --input data/demo/classification_train.csv --target target --id-col id --task classification --model forest --output outputs/my_classifier
```

`--model linear`：回归为 Ridge，分类为 LogisticRegression。
`--model forest`：随机森林。它是快速基准工具，不执行自动调参；调参见 [tuning.py](../machine_learning/tuning.py)。

## 分组与时间

```bash
python -m playbook.cli train --input data/train.csv --target y --group-col customer_id --output outputs/grouped
python -m playbook.cli train --input data/train.csv --target y --time-col timestamp --output outputs/temporal
```

分组划分让相同 group 不跨边界。时间划分以完整时间戳为单位，过去约 80% 不同时间戳训练，之后验证。
ID/time/group 列从预测特征中移除；时间与分组不能同时指定，组合方案需要自己设计。
时间划分**不会自动设置 purge/gap**。如果标签到未来才结算，或训练标签区间与验证期重叠，请按 [验证手册](VALIDATION_AND_LEAKAGE.md) 扩展。

## 预测

```bash
python -m playbook.cli predict --input data/demo/regression_test.csv --model outputs/my_model/model.joblib --output outputs/my_model/predictions.csv
```

缺少训练特征会直接报错；列顺序自动按训练顺序恢复。额外字段被忽略。
类别编码列应在训练前按业务类型处理好；CSV 无法保存完整类型语义，尤其是数字形式的类别代码和前导零 ID。
分类结果包含预测类别及 `probability_<class>` 列，不假设正类永远在第二列。
默认输出字段 `prediction`，Kaggle 提交时需改为官方指定列名。

只加载你信任的 `joblib` 文件；它使用 pickle 格式，加载可执行代码。
保存的版本记录用于追溯，不能保证不同 scikit-learn 大版本间可加载。
