"""Simulated active learning on cached features (RQ5).

The true labels play the expert. Each run starts from the nested 1% label-budget subset of the
given seed (the same images as the Phase 3 1% runs), then for ``rounds`` rounds: fit a linear
probe (C chosen on val, exactly as in Phase 3), evaluate it on test, and query
``query_fraction`` of the train pool with one strategy.

Strategies (each picks the ``n`` highest-scoring unlabeled pool images):

* ``random``: uniform.
* ``least_confidence``: ``1 - max_k p_k``.
* ``entropy``: ``-sum_k p_k log p_k``. For two classes it ranks images exactly like
  least-confidence (both are monotone in ``max_k p_k``).
* ``conformal_set_size``: size of the LAC prediction set at level ``conformal_alpha``, with the
  threshold calibrated on the cal split for the current probe; ties broken at random.
* ``kcenter``: k-center greedy (Sener & Savarese, 2018) on L2-normalized features: repeatedly
  take the pool image farthest from every labeled or already-picked image.
"""

from __future__ import annotations

import numpy as np
import torch

from labellite.conformal.core import calibrate, predict_sets
from labellite.train.knn import l2_normalize
from labellite.train.probe import fit_linear_probe

STRATEGIES = ("random", "least_confidence", "entropy", "conformal_set_size", "kcenter")


def kcenter_greedy(
    feats: np.ndarray, labeled: np.ndarray, candidates: np.ndarray, n: int, chunk: int = 8192
) -> np.ndarray:
    """``n`` rows of ``candidates`` chosen greedily to minimize the max distance from any
    candidate to the labeled set. ``feats`` must be L2-normalized (distance = 2 - 2 cos)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cand = torch.from_numpy(feats[candidates]).to(device, torch.float32)
    lab = torch.from_numpy(feats[labeled]).to(device, torch.float32)
    min_d = torch.empty(len(cand), device=device)
    for i in range(0, len(cand), chunk):  # nearest labeled image for every candidate
        min_d[i : i + chunk] = (2 - 2 * cand[i : i + chunk] @ lab.T).min(1).values
    picked = []
    for _ in range(n):
        j = int(torch.argmax(min_d))
        picked.append(j)
        min_d = torch.minimum(min_d, 2 - 2 * cand @ cand[j])
        min_d[j] = -1.0  # never pick twice
    return candidates[np.array(picked)]


def query(
    strategy: str,
    n: int,
    pool: np.ndarray,
    probs: np.ndarray | None,
    rng: np.random.Generator,
    *,
    feats: np.ndarray | None = None,
    labeled: np.ndarray | None = None,
    set_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """Rows (into the train split) to label next. ``probs`` / ``set_sizes`` are for ``pool``."""
    if strategy == "random":
        return rng.choice(pool, n, replace=False)
    if strategy == "kcenter":
        return kcenter_greedy(feats, labeled, pool, n)
    if strategy == "least_confidence":
        score = 1 - probs.max(1)
    elif strategy == "entropy":
        score = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(1)
    elif strategy == "conformal_set_size":
        score = set_sizes.astype(float)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    # Highest score first; a random permutation first makes ties fall in random order.
    perm = rng.permutation(len(pool))
    order = perm[np.argsort(-score[perm], kind="stable")]
    return pool[order[:n]]


def run_active_learning(
    feats: np.ndarray,
    y_train: np.ndarray,
    initial: np.ndarray,
    val: tuple[np.ndarray, np.ndarray],
    cal: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    *,
    strategy: str,
    query_size: int,
    rounds: int,
    seed: int,
    alpha: float,
    c_grid,
    evaluate,
) -> list[dict]:
    """One simulated run. ``feats`` / ``y_train`` cover the train pool, ``initial`` indexes it;
    ``val``, ``cal`` and ``test`` are ``(features, labels)``. Returns one dict per round."""
    rng = np.random.default_rng(seed)
    normed = l2_normalize(feats) if strategy == "kcenter" else None
    labeled = np.sort(np.asarray(initial))
    history = []
    for rnd in range(rounds + 1):
        fit = fit_linear_probe(feats[labeled], y_train[labeled], *val, c_grid=c_grid, seed=seed)
        # The probe was fitted on L2-normalized features, so every input is normalized the same.
        test_probs = fit.model.predict_proba(l2_normalize(test[0]))
        history.append(
            {"round": rnd, "n_labeled": len(labeled), "C": fit.C, **evaluate(test[1], test_probs)}
        )
        if rnd == rounds:
            break
        pool = np.setdiff1d(np.arange(len(y_train)), labeled, assume_unique=True)
        probs = sizes = None
        if strategy in ("least_confidence", "entropy", "conformal_set_size"):
            probs = fit.model.predict_proba(l2_normalize(feats[pool]))
        if strategy == "conformal_set_size":
            cal_probs = fit.model.predict_proba(l2_normalize(cal[0]))
            q = calibrate("lac", cal_probs, cal[1], alpha)
            sizes = predict_sets("lac", probs, q).sum(1)
        new = query(
            strategy, query_size, pool, probs, rng, feats=normed, labeled=labeled, set_sizes=sizes
        )
        labeled = np.sort(np.concatenate([labeled, new]))
    return history
