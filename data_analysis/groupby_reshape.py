"""聚合 agg、同长度变换 transform、透视与长宽表转换。"""

from playbook.data import orders
from data_analysis.cleaning import clean_orders


def main():
    x, _, _ = clean_orders(orders())
    summary = x.groupby("city", dropna=False).agg(
        revenue=("amount", "sum"),
        orders=("order_id", "size"),
        customers=("customer", "nunique"),
    )
    x["city_revenue"] = x.groupby("city")["amount"].transform("sum")
    x["share"] = x["amount"] / x["city_revenue"]
    wide = x.pivot_table(
        index="city", columns="customer", values="amount", aggfunc="sum", fill_value=0
    )
    long = wide.reset_index().melt(
        id_vars="city", var_name="customer", value_name="amount"
    )
    print(summary)
    print(x)
    print(wide)
    print(long)


if __name__ == "__main__":
    main()
