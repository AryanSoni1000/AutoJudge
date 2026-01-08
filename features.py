from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np

KEYWORDS = [
    "dp", "dynamic programming", "graph", "tree",
    "recursion", "dfs", "bfs", "segment tree"
]

def build_features(text_series):
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X_text = tfidf.fit_transform(text_series)

    # Numeric features
    text_length = np.array([len(t) for t in text_series]).reshape(-1, 1)

    keyword_count = np.array([
        sum(t.lower().count(k) for k in KEYWORDS)
        for t in text_series
    ]).reshape(-1, 1)

    X = hstack([X_text, text_length, keyword_count])

    return X, tfidf
