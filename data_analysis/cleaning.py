"""清洗脏订单：类型、日期、字符串、重复与无效记录审计。"""

from pathlib import Path
import pandas as pd
from playbook.data import orders


def clean_orders(raw):
    x = raw.copy()
    x["customer"] = x["customer"].astype("string").str.strip().str.casefold()
    x["amount"] = pd.to_numeric(x["amount"], errors="coerce")
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    duplicates = x.loc[x.duplicated("order_id", keep=False)].copy()
    # 本例重复订单完全相同，才保留首条。生产中应按更新时间等确定版本。
    x = x.drop_duplicates("order_id", keep="first")
    x["city"] = x["city"].fillna("Unknown")
    invalid = x.loc[x["amount"].isna() | x["date"].isna()].copy()
    valid = x.loc[x["amount"].notna() & x["date"].notna()].copy()
    return valid, invalid, duplicates


def main():
    valid, invalid, duplicates = clean_orders(orders())
    out = Path("outputs/cleaning")
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("valid", valid),
        ("invalid", invalid),
        ("duplicates", duplicates),
    ]:
        frame.to_csv(out / f"{name}.csv", index=False)
        print(name, frame.to_dict("records"))


if __name__ == "__main__":
    main()
