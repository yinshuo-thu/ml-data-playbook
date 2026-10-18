"""无网络生成分布图、散点图、相关性热图和缺失率图。"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from playbook.data import tabular


def main():
    x, y = tabular()
    out = Path("outputs/eda")
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].hist(y, bins=30)
    axes[0, 0].set_title("Target distribution")
    axes[0, 1].scatter(x["x1"], y, s=8, alpha=0.4)
    axes[0, 1].set_title("x1 vs target")
    corr = x.select_dtypes("number").corr()
    im = axes[1, 0].imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(im, ax=axes[1, 0])
    axes[1, 0].set_title("Feature correlation")
    x.isna().mean().plot.bar(ax=axes[1, 1])
    axes[1, 1].set_title("Missing rate")
    fig.tight_layout()
    fig.savefig(out / "report.png", dpi=150)
    plt.close(fig)
    print(out / "report.png")


if __name__ == "__main__":
    main()
