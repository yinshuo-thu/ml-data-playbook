"""二分类：分层划分、概率、AUC、AP、log loss 和混淆矩阵。"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
)
from playbook.data import tabular
from playbook.modeling import build_pipeline


def main():
    x, y = tabular("classification")
    xt, xv, yt, yv = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    model = build_pipeline("classification").fit(xt, yt)
    p = model.predict_proba(xv)[:, list(model.classes_).index(1)]
    print(
        {
            "auc": roc_auc_score(yv, p),
            "average_precision": average_precision_score(yv, p),
            "log_loss": log_loss(yv, p),
        }
    )
    print(confusion_matrix(yv, p >= 0.5))
    # 阈值需要按业务成本在开发验证中选择；不能用测试标签选阈值。


if __name__ == "__main__":
    main()
