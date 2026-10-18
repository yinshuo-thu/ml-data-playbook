"""文本分类：TF-IDF 在训练集内拟合，线性分类器预测。玩具语料演示 API。"""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def main():
    texts = [
        "great product",
        "excellent service",
        "love this item",
        "very happy",
        "good quality",
        "bad product",
        "terrible service",
        "hate this item",
        "very broken",
        "poor quality",
    ]
    labels = [1] * 5 + [0] * 5
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("model", LogisticRegression()),
        ]
    ).fit(texts, labels)
    print(pipe.predict_proba(["excellent product", "broken item"]))
    # 此处无泛化评分；真实任务需独立划分。中文可配字符 n-gram 或分词器。


if __name__ == "__main__":
    main()
