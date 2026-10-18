# 指标与模型选择

| 任务/指标 | 解释 | 容易误用的地方 |
|---|---|---|
| RMSE | 均方误差开根号，与 y 同单位，大误差权重更大 | 不能跨单位直接比较 |
| MAE | 平均绝对误差，与 y 同单位 | 对极端误差惩罚比 RMSE 小 |
| 对数 RMSE | 在 log(y) 空间计算 RMSE | 要求正值；log 与 log1p 不相同 |
| R² | 相对常数均值基准的解释程度 | 可以为负，不等于准确率 |
| ROC AUC | 正样本相对负样本的概率排序 | 输入概率/分数，不是硬分类 |
| AP | average precision，概括 precision-recall 表现 | 基准与正例比例有关，不能脱离 prevalence |
| Log loss | 评价概率预测 | 过度自信且预测错误会被重罚 |
| F1 | precision 与 recall 的调和平均 | 依赖阈值和正类定义 |
| Accuracy | 预测类别正确比例 | 极端不平衡时可能误导 |

## 最小公式

```python
rmse = np.sqrt(np.mean((prediction - actual) ** 2))
mae = np.mean(np.abs(prediction - actual))
```

`predict_proba` 的列顺序由 `model.classes_` 决定，先找正类对应列。
多分类 AUC 必须明确 one-vs-rest/one-vs-one 和平均方式；不要直接套二分类代码。
阈值、概率校准、类别权重应在开发阶段评估，避免在最终测试标签上优化。

## 选模型的顺序

1. 建立常数/多数类/季节性朴素基准。
2. 线性模型快速验证数据管道。
3. 随机森林或梯度提升学习非线性。
4. 数据规模和结构适合时尝试深度学习。
5. 同一划分、同一指标比较，记录方差与运行成本。

本库通用工具提供 linear / forest；不把小数据上的单次胜出当作普遍结论。
