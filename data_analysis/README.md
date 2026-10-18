# 数据分析与清洗

| 文件 | 场景 | 输入 | 输出 |
|---|---|---|---|
| [profile_csv.py](profile_csv.py) | 体检任意 CSV | `--input` 指定 CSV | 类型、缺失、唯一值、数值范围 |
| [cleaning.py](cleaning.py) | 金额/日期非法、重复订单、字符串空格 | 内置脏订单 | valid / invalid / duplicates CSV |
| [groupby_reshape.py](groupby_reshape.py) | 城市汇总、组内占比、长宽表 | 清洗后的订单 | 终端表格 |
| [joins.py](joins.py) | 订单连接客户表 | 两张内置表 | 连接结果和未匹配行 |
| [io_memory.py](io_memory.py) | 分块聚合、内存、Parquet | 生成的 CSV | IO 文件与统计 |
| [dates_strings.py](dates_strings.py) | 字符串提取、UTC 转换、分箱排名 | 内置表格 | 终端表格 |

运行示例：`python -m data_analysis.cleaning`，在仓库根目录执行。
[可复制语法速查](../docs/PANDAS_COOKBOOK.md) · [问题索引](../docs/TASK_INDEX.md)

遇到缺失先解释业务含义；遇到重复先核对主键和版本；连接先约束基数；不要用静默丢行掩盖问题。
