"""同一张表用 SQL 分组、窗口函数与参数化查询。"""

import sqlite3
import pandas as pd


def main():
    orders = pd.DataFrame(
        {"id": [1, 2, 3, 4], "city": ["A", "A", "B", "B"], "amount": [5, 8, 3, 20]}
    )
    with sqlite3.connect(":memory:") as db:
        orders.to_sql("orders", db, index=False, if_exists="replace")
        print(
            pd.read_sql_query(
                "SELECT city, SUM(amount) AS revenue FROM orders GROUP BY city", db
            )
        )
        print(
            pd.read_sql_query(
                """SELECT *, ROW_NUMBER() OVER
            (PARTITION BY city ORDER BY amount DESC, id) AS rank_in_city
            FROM orders WHERE amount >= ?""",
                db,
                params=[5],
            )
        )
    # 用户输入用 params 绑定，不拼接进 SQL 字符串。


if __name__ == "__main__":
    main()
