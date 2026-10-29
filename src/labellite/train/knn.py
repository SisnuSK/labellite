"""k-nearest-neighbour classification on L2-normalized (cosine) features."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def knn_fit(train_x: np.ndarray, train_y: np.ndarray, k: int = 20) -> KNeighborsClassifier:
    clf = KNeighborsClassifier(n_neighbors=min(k, len(train_y)), metric="cosine", algorithm="brute")
    return clf.fit(l2_normalize(train_x), train_y)


def knn_accuracy(train_x, train_y, test_x, test_y, k: int = 20) -> float:
    clf = knn_fit(train_x, train_y, k)
    return float((clf.predict(l2_normalize(test_x)) == np.asarray(test_y)).mean())
