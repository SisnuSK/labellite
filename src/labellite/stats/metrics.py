"""Classification metrics, bootstrap confidence intervals, and paired model comparisons."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def accuracy(y: np.ndarray, probs: np.ndarray) -> float:
    return float((probs.argmax(1) == y).mean())


def macro_f1(y: np.ndarray, probs: np.ndarray) -> float:
    return float(
        f1_score(
            y, probs.argmax(1), average="macro", labels=np.arange(probs.shape[1]), zero_division=0
        )
    )


def balanced_accuracy(y: np.ndarray, probs: np.ndarray) -> float:
    return float(balanced_accuracy_score(y, probs.argmax(1)))


def auroc(y: np.ndarray, probs: np.ndarray) -> float:
    """Binary AUROC of the positive-class score. NaN if only one class is present."""
    if probs.shape[1] != 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, probs[:, 1]))


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "balanced_accuracy": balanced_accuracy,
    "auroc": auroc,
}


def metric_names(num_classes: int) -> list[str]:
    """AUROC is reported for binary tasks only."""
    return [m for m in METRICS if m != "auroc" or num_classes == 2]


def evaluate(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    return {m: METRICS[m](y, probs) for m in metric_names(probs.shape[1])}


class WeightedMetric:
    """Fast metric on a bootstrap resample given as per-item multiplicity weights.

    A resample of n items is ``w = bincount(idx, minlength=n)``; every metric here is a function
    of weighted counts (a confusion matrix, or weighted rank sums for AUROC), so each resample
    costs a few ``bincount`` calls instead of a full sklearn metric. With ``w = 1`` the values
    equal the sklearn-based ``METRICS``.
    """

    def __init__(self, y: np.ndarray, probs: np.ndarray, metric: str):
        self.metric, self.k = metric, probs.shape[1]
        self.y = np.asarray(y)
        self.cell = self.y * self.k + probs.argmax(1)
        if metric == "auroc":
            self.uniq_inv = np.unique(probs[:, 1], return_inverse=True)[1]
            self.n_groups = int(self.uniq_inv.max()) + 1
            self.pos = self.y == 1

    def __call__(self, w: np.ndarray) -> float:
        if self.metric == "auroc":
            return self._auroc(w)
        k = self.k
        cm = np.bincount(self.cell, weights=w, minlength=k * k).reshape(k, k)
        tp, support, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
        if self.metric == "accuracy":
            return float(tp.sum() / cm.sum())
        if self.metric == "balanced_accuracy":
            present = support > 0
            return float((tp[present] / support[present]).mean())
        if self.metric == "macro_f1":
            denom = support + predicted
            return float(np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-300), 0.0).mean())
        raise KeyError(self.metric)

    def _auroc(self, w: np.ndarray) -> float:
        g = self.n_groups
        pos_w = np.bincount(self.uniq_inv, weights=w * self.pos, minlength=g)
        neg_w = np.bincount(self.uniq_inv, weights=w * ~self.pos, minlength=g)
        total_pos, total_neg = pos_w.sum(), neg_w.sum()
        if total_pos == 0 or total_neg == 0:
            return float("nan")
        neg_below = np.cumsum(neg_w) - neg_w  # negatives with a strictly lower score
        return float((pos_w * (neg_below + 0.5 * neg_w)).sum() / (total_pos * total_neg))


def _resample_weights(n: int, n_resamples: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        yield np.bincount(rng.integers(0, n, n), minlength=n).astype(float)


def bootstrap_ci(
    y: np.ndarray,
    probs: np.ndarray,
    metric: str,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap (1 - alpha) CI, resampling test items."""
    fn = WeightedMetric(y, probs, metric)
    reps = np.array([fn(w) for w in _resample_weights(len(y), n_resamples, seed)])
    lo, hi = np.nanpercentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return METRICS[metric](y, probs), float(lo), float(hi)


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """Exact (binomial) McNemar test on paired per-item correctness."""
    correct_a, correct_b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b = int((correct_a & ~correct_b).sum())  # only A right
    c = int((~correct_a & correct_b).sum())  # only B right
    p = 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)
    return {"only_a_correct": b, "only_b_correct": c, "p_value": p}


def paired_bootstrap(
    y: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    metric: str,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Difference metric(A) - metric(B) on the same resampled items, with a two-sided p-value."""
    fa, fb = WeightedMetric(y, probs_a, metric), WeightedMetric(y, probs_b, metric)
    diffs = np.array([fa(w) - fb(w) for w in _resample_weights(len(y), n_resamples, seed)])
    observed = METRICS[metric](y, probs_a) - METRICS[metric](y, probs_b)
    # Two-sided p: how often the bootstrap difference falls on the other side of zero, with the
    # (k + 1) / (n + 1) correction so a finite bootstrap never reports p = 0.
    tail = min((diffs <= 0).sum(), (diffs >= 0).sum())
    p = min(1.0, 2 * (tail + 1) / (n_resamples + 1))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(observed), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(p)}


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted
