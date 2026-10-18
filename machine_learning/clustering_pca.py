"""无监督学习：标准化、PCA 降维、KMeans 与轮廓系数。"""

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def main():
    x, _ = make_blobs(n_samples=300, n_features=5, centers=3, random_state=42)
    z = StandardScaler().fit_transform(x)
    labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(z)
    reduced = PCA(n_components=2, random_state=42).fit_transform(z)
    print("projection shape", reduced.shape, "silhouette", silhouette_score(z, labels))
    # 轮廓系数衡量当前数据中的聚类分离，不等于有监督预测精度。


if __name__ == "__main__":
    main()
