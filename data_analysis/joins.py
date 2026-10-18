"""带基数检查的连接，防止 join 后行数悄悄翻倍。"""

import pandas as pd


def main():
    orders = pd.DataFrame(
        {"order_id": [1, 2, 3], "customer_id": [10, 10, 99], "amount": [5, 8, 9]}
    )
    customers = pd.DataFrame({"customer_id": [10, 20], "city": ["Shanghai", "Beijing"]})
    result = orders.merge(
        customers, on="customer_id", how="left", validate="many_to_one", indicator=True
    )
    assert len(result) == len(orders)
    unmatched = result.loc[result["_merge"] == "left_only"]
    print(result)
    print("unmatched", unmatched)
    more = pd.DataFrame({"order_id": [4], "customer_id": [20], "amount": [2]})
    print(pd.concat([orders, more], ignore_index=True))


if __name__ == "__main__":
    main()
