# SQL 与 Pandas

运行 `python -m sql.sqlite_analysis`。
使用 Python 自带 sqlite3，无需数据库服务器。内置订单写入内存数据库后展示：

- `GROUP BY` 分城市求和。
- `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)` 组内排序。
- 通过 `params` 绑定查询参数，避免字符串拼接。
- `pd.read_sql_query` 将查询结果读成 DataFrame。

真实数据库应使用只读账号或明确事务权限；不同 SQL 方言的日期函数有差别。
[示例](sqlite_analysis.py) · [Pandas 连接](../data_analysis/joins.py)
