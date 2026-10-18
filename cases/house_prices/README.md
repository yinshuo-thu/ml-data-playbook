# 真实 Kaggle 案例：House Prices

目标：用 79 个房屋解释变量预测 `SalePrice`。本地数据副本含 1,460 条训练、1,459 条测试记录。
本案例串起 Pandas 检查、语义清洗、特征工程、Pipeline、五折 CV、独立预留评估、误差分析和提交文件。
[官方比赛](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) · [完整代码](train.py) · [分步 Notebook](walkthrough.ipynb)

## 1. 获取真实数据

登录 Kaggle，从比赛 Data 页获取 `train.csv`、`test.csv`、`data_description.txt`、`sample_submission.csv`，
放入仓库根目录下的 `data/house_prices/`。若比赛要求接受规则，先在 Kaggle 自行完成。
不需要下载网上整理的“清洗后”数据，避免无法追溯的标签或处理泄漏。

也可安装并配置官方 Kaggle CLI 后下载：

```bash
python -m pip install -e '.[kaggle]'
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/house_prices
python -m zipfile -e data/house_prices/house-prices-advanced-regression-techniques.zip data/house_prices
```

Kaggle 凭据放在官方工具要求的位置，不提交进仓库。

## 2. 运行

```bash
python -m cases.house_prices.train --data data/house_prices
```

结果位于 `outputs/house_prices/`：

| 文件 | 用途 |
|---|---|
| `data_audit.csv` | 每列类型、缺失率、取值数量 |
| `cv_scores.csv` | 候选模型五折平均、标准差、每折分数 |
| `summary.json` | 划分大小、选中模型、预留分数 |
| `holdout_errors.csv` | 真实价格、预测价格、误差最大的样本 |
| `holdout_by_neighborhood.csv` | 按社区统计误差与样本数 |
| `submission.csv` | 官方目标格式 `Id,SalePrice`，不自动上传 |

## 3. 先读数据字典

```python
train = pd.read_csv("data/house_prices/train.csv")
test = pd.read_csv("data/house_prices/test.csv")
audit = pd.DataFrame(
    {
        "dtype": train.dtypes.astype(str),
        "missing_rate": train.isna().mean(),
        "nunique": train.nunique(),
    }
).sort_values("missing_rate", ascending=False)
```

`PoolQC` 缺失很多，但字典中 NA 表示没有泳池。`LotFrontage` 缺失则通常代表长度未知。
两者不能一律填零。CSV 解析器也会将一些字符串当作缺失，因此必须结合数据字典审计。

## 4. 固定规则清洗与特征

[make_features](train.py) 中每次复制输入，按行构造特征，不使用目标或全表统计量：

- `MSSubClass` 是类型代码，转类别；月份也按类别处理。
- 字典明确“无设施”的类别缺失填 `"None"`；其余类别缺失交给 Unknown 填补。
- `LotFrontageMissing` 标识临街长度是否缺失。
- `TotalSF` = 地下室 + 一楼 + 二楼面积，缺项时保留未知。
- `TotalBath` = 完整卫生间 + 0.5 × 半卫生间（含地下室）。
- `HouseAge` = 成交年份 - 建造年份；负值先标为未知。
- `SinceRemodel` = 成交年份 - 翻修年份。

不根据整个训练集目标手工删异常样本；先让验证和残差暴露问题。

## 5. 指标与划分

比赛指标是预测与真实价格的自然对数 RMSE：

```python
y = np.log(train["SalePrice"])
X = train.drop(columns=["Id", "SalePrice"])
X_dev, X_hold, y_dev, y_hold = train_test_split(X, y, test_size=0.2, random_state=42)
```

这会预留 292 行，开发部分 1,168 行。开发集再做固定五折 CV。
这是本例的随机验证方案；实际部署到未来成交房屋时，应额外考虑时间验证。
`Id` 用于提交，不直接当作房价解释变量。

## 6. Pipeline 的作用

数值列：训练折中位数填补 + 缺失指示 + StandardScaler。
类别列：Unknown 填补 + OneHotEncoder（新类别忽略）。
所有预处理与模型一起交给交叉验证，每折重新拟合。
不先把训练和比赛测试数据拼起来学习统计量，也不在划分之前填补全表中位数。

比较：常数 Dummy、Ridge 的三个 alpha、随机森林。
根据开发 CV 平均误差选择模型，预留集只用于选好方案后的评估。

## 7. 如何读结果

先看常数基准是否明显被击败，再比较候选的每折分数和标准差。
平均差距很小时，不要把单次胜出视为稳健结论。

```python
errors.sort_values("abs_log_error", ascending=False).head(10)
errors.groupby("Neighborhood").agg(
    n=("abs_log_error", "size"),
    mean_abs_log_error=("abs_log_error", "mean"),
)
```

检查高价房、面积极端样本与样本量很少的社区。不要只看平均误差而忽略组内数量。
根据预留误差继续调参后，原预留集已不再是独立证据。

## 8. 最终预测

选中参数后，在所有有标签数据上重新训练：

```python
final.fit(X, y)
price = np.exp(final.predict(test.drop(columns="Id")))
submission = pd.DataFrame({"Id": test["Id"], "SalePrice": price})
submission.to_csv("outputs/house_prices/submission.csv", index=False)
```

程序检查行数、ID 顺序、价格为正且有限。上传前对照官方 sample_submission。
本仓库不自动提交，不声称任何排行榜名次。

## 可迁移到其他比赛

先换数据字典、目标、指标和划分，再换特征工程。
一般混合类型预处理和 Pipeline 思路可直接参考；“没有车库”等房屋语义不能迁移到其他业务。
数据获取、类别不平衡、时序泄漏、多表连接需按目标任务单独判断。
