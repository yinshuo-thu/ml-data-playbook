"""普通 KFold、分层、分组和按时间划分的最小可运行对照。"""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit


def main():
    x = np.arange(60).reshape(-1, 1)
    y = np.tile([0, 1], 30)
    groups = np.repeat(np.arange(20), 3)
    splits = {
        "iid": KFold(3, shuffle=True, random_state=42).split(x),
        "stratified": StratifiedKFold(3, shuffle=True, random_state=42).split(x, y),
        "group": GroupKFold(3).split(x, y, groups),
        "time": TimeSeriesSplit(3, gap=2).split(x),
    }
    for name, iterator in splits.items():
        a, b = next(iterator)
        if name == "group":
            assert set(groups[a]).isdisjoint(groups[b])
        if name == "time":
            assert a.max() < b.min() - 2
        print(name, "train", len(a), "valid", len(b))
    # 时序样本有重叠标签区间时，gap 应按标签可用时间设计，不能盲用 2。


if __name__ == "__main__":
    main()
