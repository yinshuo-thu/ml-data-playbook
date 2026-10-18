"""Create offline demo input for the generic CSV CLI. Run from repository root."""

from pathlib import Path
import numpy as np
from playbook.data import tabular


def main():
    out = Path("data/demo")
    out.mkdir(parents=True, exist_ok=True)
    for task in ["regression", "classification"]:
        x, y = tabular(task, n=600)
        x.insert(0, "id", np.arange(len(x)))
        labelled = x.iloc[:500].copy()
        labelled["target"] = y.iloc[:500]
        labelled.to_csv(out / f"{task}_train.csv", index=False)
        x.iloc[500:].to_csv(out / f"{task}_test.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
