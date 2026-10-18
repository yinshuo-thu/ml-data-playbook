"""日期解析、时区、文本提取、分箱与排名。"""

import pandas as pd


def main():
    x = pd.DataFrame(
        {
            "code": [" sku-001 ", "SKU-020", None],
            "time": ["2025-01-01T00:00:00Z", "2025-01-02T06:00:00Z", "bad"],
            "value": [10, 30, 20],
        }
    )
    x["sku"] = (
        x["code"]
        .astype("string")
        .str.strip()
        .str.upper()
        .str.extract(r"(SKU-\d+)", expand=False)
    )
    x["utc"] = pd.to_datetime(x["time"], utc=True, errors="coerce")
    x["local"] = x["utc"].dt.tz_convert("Asia/Shanghai")
    x["hour"] = x["local"].dt.hour
    x["bucket"] = pd.cut(
        x["value"], bins=[0, 15, 25, 100], labels=["low", "mid", "high"]
    )
    x["rank"] = x["value"].rank(method="dense", ascending=False)
    print(x)
    # 无时区的本地时间应先 tz_localize，再 tz_convert；不要直接当 UTC。


if __name__ == "__main__":
    main()
