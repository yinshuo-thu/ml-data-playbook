# Pandas 常用操作速查

以下片段是迁移到自己数据时的写法模板。`df` 表示你的 DataFrame，列名须替换。
可以先用 `from playbook.data import orders; df = orders()` 获得示例订单。

## 读写与概览

```python
import pandas as pd

df = pd.read_csv("data/orders.csv", usecols=["order_id", "city", "amount"],
                 dtype={"order_id": "Int64", "city": "string"})
print(df.shape)
print(df.head())
df.info()
print(df.describe(include="all"))
print(df.isna().mean().sort_values(ascending=False))
df.to_csv("outputs/orders.csv", index=False)
```

Excel：`pd.read_excel("data/book.xlsx", sheet_name="Sheet1")`；安装 `.[io]`。
Parquet：`df.to_parquet("outputs/orders.parquet", index=False)`，适合保留类型并压缩。
CSV 默认会把 `NA` 等字符串识别成缺失；若业务中它是合法代码，考虑 `keep_default_na=False` 并明确 `na_values`。

## 筛选与赋值

```python
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
subset = df.loc[df["amount"].gt(100) & df["city"].isin(["Beijing", "Shanghai"]),
                ["order_id", "amount"]].copy()
df.loc[df["amount"].lt(0), "amount"] = pd.NA
first_three = df.iloc[:3]
```

不要用 `df[df.amount < 0]["amount"] = ...` 链式赋值。
Pandas 按索引对齐 Series；若明确按位置赋值，先确认顺序再 `.to_numpy()`。
布尔条件使用 `&`、`|`，每个条件加括号；不能直接用 Python 的 `and` / `or`。

## 缺失、重复与异常

```python
missing = df.isna().sum()
dup = df.loc[df.duplicated("order_id", keep=False)]
df["amount_missing"] = df["amount"].isna().astype("int8")
df["city"] = df["city"].fillna("Unknown")
```

填补中位数、缩尾分位点、标准化统计量均应在训练折内学习。
不要无条件 `fillna(0)`、删除全部缺失行或按 3σ 删除异常。先看单位、字典、业务规则和样本来源。

## 分组、连接、长宽表

```python
summary = df.groupby("city", dropna=False).agg(
    n=("order_id", "size"), revenue=("amount", "sum"), median=("amount", "median"))
df["city_mean"] = df.groupby("city")["amount"].transform("mean")
```

`size` 包括缺失行，`count` 只数非缺失值。`agg` 返回组级结果，`transform` 返回与原表同长度结果。
`city_mean` 若使用预测目标，必须折内计算；普通探索聚合不能直接照搬成预测特征。

```python
# customers 必须先从真实客户表读入
joined = df.merge(customers, on="customer_id", how="left",
                  validate="many_to_one", indicator=True)
wide = df.pivot_table(index="city", columns="month", values="amount", aggfunc="sum")
long = wide.reset_index().melt(id_vars="city", var_name="month", value_name="amount")
```

连接前检查键是否唯一、是否为空。Pandas 的空连接键可能互相匹配，与通常 SQL NULL 语义不同。
`pivot` 要求索引与列组合唯一；`pivot_table` 可以指定聚合规则。

## 时间、窗口与性能

```python
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values(["customer_id", "date"])
df["previous_amount"] = df.groupby("customer_id")["amount"].shift(1)
df["past_7_mean"] = df.groupby("customer_id")["amount"].transform(
    lambda s: s.shift(1).rolling(7, min_periods=3).mean())
```

这里是 7 条记录，不是 7 天。时间窗口需设置时间索引并使用 `rolling("7D")`，还要明确边界是否包含当前时点。
组内 rolling 要避免把另一个客户的记录带入窗口。

- 优先向量化 `.str`、`.dt`、算术运算和 `groupby`，再考虑逐行 `apply`。
- 分块统计均值应累计总和与有效数量，不能直接平均各块的均值。
- `category` 对低基数重复字符串可能节省内存；高基数唯一 ID 不一定适合。
- 先 `df.memory_usage(deep=True)` 检查；转 float32 前评估精度损失。
