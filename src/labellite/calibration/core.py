"""Temperature scaling and calibration metrics (RQ2).

* ECE: top-label expected calibration error with ``n_bins`` equal-width confidence bins on
  (0, 1], weighted by bin size.
* Brier score: mean over items of the squared error between the probability vector and the
  one-hot label, summed over classes (range 0 to 2).
* NLL: mean negative log-probability of the true class.

Temperature scaling (Guo et al., 2017) divides the logits by one scalar ``T`` fitted by
minimizing NLL. It never changes the predicted class, so accuracy is unaffected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from labellite.train.probe import softmax

_EPS = 1e-12


def nll(y: np.ndarray, probs: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], _EPS, None)).mean())


def brier(y: np.ndarray, probs: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1
    return float(((probs - onehot) ** 2).sum(1).mean())


def reliability_table(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    """Per confidence bin: item count, mean confidence and accuracy (empty bins included)."""
    conf, correct = probs.max(1), probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    # Bins are (lo, hi]; confidence is always > 0, so bin 0 is (0, 1/n].
    b = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    count = np.bincount(b, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        mean_conf = np.bincount(b, weights=conf, minlength=n_bins) / count
        acc = np.bincount(b, weights=correct, minlength=n_bins) / count
    return pd.DataFrame(
        {
            "lo": edges[:-1],
            "hi": edges[1:],
            "count": count,
            "confidence": mean_conf,
            "accuracy": acc,
        }
    )


def ece(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    t = reliability_table(y, probs, n_bins)
    t = t[t["count"] > 0]
    return float((t["count"] * (t["accuracy"] - t["confidence"]).abs()).sum() / len(y))


def calibration_metrics(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    return {"ece": ece(y, probs, n_bins), "brier": brier(y, probs), "nll": nll(y, probs)}


def fit_temperature(logits: np.ndarray, y: np.ndarray, bounds=(0.05, 20.0)) -> float:
    """The ``T`` in ``bounds`` minimizing NLL of ``softmax(logits / T)`` (searched in log T)."""
    logits = np.asarray(logits, dtype=np.float64)

    def loss(log_t: float) -> float:
        return nll(y, softmax(logits / np.exp(log_t)))

    res = minimize_scalar(
        loss,
        bounds=(np.log(bounds[0]), np.log(bounds[1])),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return float(np.exp(res.x))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(np.asarray(logits, dtype=np.float64) / temperature)
