"""Split conformal prediction (LAC, APS) and the triage policy (RQ3).

Scores (higher = less conforming), for class probabilities ``p``:

* **LAC** (least ambiguous set-valued classifier; Sadinle et al., 2019): ``1 - p_y``.
* **APS** (adaptive prediction sets; Romano et al., 2020), randomized: the total probability of
  the classes ranked above ``y`` plus ``u * p_y`` with ``u ~ U(0, 1)`` drawn per item.

With ``n`` calibration scores, the threshold ``q`` is their ``ceil((n + 1)(1 - alpha)) / n``
empirical quantile, and the prediction set of a test item is every class whose score is
``<= q``. If calibration and test data are exchangeable, ``P(y in set) >= 1 - alpha``.
Randomized APS can return an empty set; triage routes those to the expert.

Triage: an image whose set holds exactly one class is auto-labeled with it; every other image
goes to the expert. Assuming the expert is always right, the only errors are wrong singletons, and
a wrong singleton is a miscovered item, so the system's error rate is at most ``alpha``.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.model_selection import train_test_split


def split_calibration(
    y: np.ndarray, ts_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified, disjoint row indices into the calibration split: one part fits the
    temperature, the other calibrates conformal prediction (whose guarantee needs a score
    function fixed independently of its calibration data)."""
    idx = np.arange(len(y))
    ts, conf = train_test_split(idx, train_size=ts_fraction, stratify=y, random_state=seed)
    return np.sort(ts), np.sort(conf)


def lac_scores(probs: np.ndarray) -> np.ndarray:
    """(N, K) score of every class."""
    return 1.0 - probs


def aps_scores(probs: np.ndarray, u: np.ndarray) -> np.ndarray:
    """(N, K) randomized APS score of every class; ``u`` is one U(0, 1) draw per item."""
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_p = np.take_along_axis(probs, order, axis=1)
    above = np.cumsum(sorted_p, axis=1) - sorted_p  # mass of the classes ranked higher
    scores = np.empty_like(probs)
    np.put_along_axis(scores, order, above + u[:, None] * sorted_p, axis=1)
    return scores


def class_scores(method: str, probs: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
    if method == "lac":
        return lac_scores(probs)
    if method == "aps":
        return aps_scores(probs, u)
    raise ValueError(f"unknown conformal method {method!r}")


def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """``ceil((n + 1)(1 - alpha)) / n`` empirical quantile; ``inf`` if n is too small."""
    n = len(cal_scores)
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        return float("inf")
    return float(np.sort(cal_scores)[rank - 1])


def calibrate(
    method: str, probs: np.ndarray, y: np.ndarray, alpha: float, u: np.ndarray | None = None
) -> float:
    scores = class_scores(method, probs, u)
    return conformal_quantile(scores[np.arange(len(y)), y], alpha)


def predict_sets(
    method: str, probs: np.ndarray, q: float, u: np.ndarray | None = None
) -> np.ndarray:
    """(N, K) boolean membership of each class in each item's prediction set."""
    return class_scores(method, probs, u) <= q


def evaluate_sets(sets: np.ndarray, y: np.ndarray) -> dict:
    """Coverage, set size and triage outcomes on items with true labels ``y``."""
    n, k = sets.shape
    covered = sets[np.arange(n), y]
    size = sets.sum(1)
    single = size == 1
    per_class = np.array([covered[y == c].mean() if (y == c).any() else np.nan for c in range(k)])
    n_single = int(single.sum())
    return {
        "coverage": float(covered.mean()),
        "mean_set_size": float(size.mean()),
        "empty_fraction": float((size == 0).mean()),
        "auto_fraction": float(single.mean()),  # auto-labeled = expert effort saved
        "routed_fraction": float(1 - single.mean()),
        "auto_error": float(1 - covered[single].mean()) if n_single else float("nan"),
        "system_error": float((single & ~covered).mean()),  # wrong auto-labels / all images
        "min_class_coverage": float(np.nanmin(per_class)),
        "class_coverage": per_class.tolist(),
    }


def coverage_distribution(n_cal: int, alpha: float, n_test: int) -> tuple[float, float]:
    """Mean and standard deviation of test-set coverage under exchangeability.

    Coverage conditional on the calibration set is Beta(n + 1 - l, l) with
    ``l = floor((n + 1) alpha)`` (Vovk, 2012; Angelopoulos & Bates, 2021); the empirical
    coverage of ``n_test`` items adds binomial noise on top.
    """
    l_ = math.floor((n_cal + 1) * alpha)
    if l_ == 0:
        return 1.0, 0.0
    a, b = n_cal + 1 - l_, l_
    mean = a / (a + b)
    var_beta = a * b / ((a + b) ** 2 * (a + b + 1))
    # E[c(1 - c)] / n_test for c ~ Beta(a, b)
    var_test = (mean * (1 - mean) - var_beta) / n_test
    return mean, math.sqrt(var_beta + var_test)
