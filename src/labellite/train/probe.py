"""Linear probe: logistic regression on L2-normalized frozen features.

The inverse regularization strength ``C`` is chosen on the val split (highest accuracy, ties
broken by lower val log-loss). Every ``C`` is fitted from scratch: warm-starting along the path
let L-BFGS stop immediately on large training sets, making neighbouring ``C`` values identical.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from labellite.train.knn import l2_normalize

DEFAULT_C_GRID = tuple(10.0**k for k in range(-2, 5))  # 0.01 ... 10,000


@dataclass
class ProbeFit:
    model: LogisticRegression
    C: float
    val_accuracy: float
    val_log_loss: float
    search: list[dict]
    converged: bool


def logits(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    """(N, K) class logits; binary models are expanded to [0, z] so softmax == predict_proba."""
    z = model.decision_function(l2_normalize(x))
    if z.ndim == 1:
        z = np.stack([np.zeros_like(z), z], axis=1)
    return z


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    c_grid=DEFAULT_C_GRID,
    max_iter: int = 2000,
    seed: int = 0,
) -> ProbeFit:
    xtr, xval = l2_normalize(train_x), l2_normalize(val_x)
    classes = np.arange(int(max(train_y.max(), val_y.max())) + 1)
    if len(np.unique(train_y)) != len(classes):
        raise ValueError("every class must appear in the training subset")
    best, search = None, []
    for C in sorted(c_grid):
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(xtr, train_y)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        probs = model.predict_proba(xval)
        acc = float((model.classes_[probs.argmax(1)] == val_y).mean())
        loss = float(log_loss(val_y, probs, labels=classes))
        search.append({"C": C, "val_accuracy": acc, "val_log_loss": loss, "converged": converged})
        if best is None or (acc, -loss) > (best[1]["val_accuracy"], -best[1]["val_log_loss"]):
            best = (model, search[-1])
    model, chosen = best
    return ProbeFit(
        model,
        chosen["C"],
        chosen["val_accuracy"],
        chosen["val_log_loss"],
        search,
        chosen["converged"],
    )
