import numpy as np
import pandas as pd
import pytest

from labellite.active.loop import kcenter_greedy, query, run_active_learning
from labellite.active.pipeline import label_savings
from labellite.stats.metrics import evaluate
from labellite.train.knn import l2_normalize


def blobs(n, k=3, d=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, (k, d))
    y = rng.integers(0, k, n)
    return (centers[y] + rng.normal(0, 1.5, (n, d))).astype(np.float32), y


def test_uncertainty_queries_pick_the_least_confident():
    pool = np.array([10, 11, 12, 13])
    probs = np.array([[0.9, 0.1], [0.5, 0.5], [0.6, 0.4], [0.99, 0.01]])
    rng = np.random.default_rng(0)
    for strategy in ("least_confidence", "entropy"):
        np.testing.assert_array_equal(query(strategy, 2, pool, probs, rng), [11, 12])
    sizes = np.array([1, 2, 1, 2])
    picked = query("conformal_set_size", 2, pool, probs, rng, set_sizes=sizes)
    assert set(picked) == {11, 13}
    r = query("random", 3, pool, None, rng)
    assert len(set(r)) == 3 and set(r) <= set(pool)


def test_kcenter_greedy_picks_far_points_first():
    # Labeled point on +x; candidates: near +x, -x (farthest), +y, then a near-duplicate of -x.
    feats = l2_normalize(
        np.array([[1, 0], [0.99, 0.1], [-1, 0], [0, 1], [-0.99, 0.05]], dtype=np.float32)
    )
    picked = kcenter_greedy(feats, np.array([0]), np.array([1, 2, 3, 4]), 2)
    np.testing.assert_array_equal(picked, [2, 3])  # -x, then +y (not -x's twin)


@pytest.mark.parametrize("strategy", ["random", "entropy", "conformal_set_size", "kcenter"])
def test_loop_grows_labels_without_repeats(strategy):
    x, y = blobs(1200)
    tr, va, ca, te = slice(0, 600), slice(600, 800), slice(800, 1000), slice(1000, None)
    initial = np.concatenate([np.flatnonzero(y[tr] == c)[:3] for c in range(3)])
    hist = run_active_learning(
        x[tr], y[tr], initial, (x[va], y[va]), (x[ca], y[ca]), (x[te], y[te]),
        strategy=strategy, query_size=20, rounds=3, seed=0, alpha=0.1,
        c_grid=[0.1, 1.0, 10.0], evaluate=evaluate,
    )  # fmt: skip
    assert [h["n_labeled"] for h in hist] == [9, 29, 49, 69]
    assert all(0 <= h["accuracy"] <= 1 for h in hist)
    assert hist[-1]["accuracy"] > 0.5


def test_label_savings_interpolates():
    rows = []
    for strategy, curve in (("random", [0.5, 0.6, 0.6]), ("entropy", [0.5, 0.7, 0.8])):
        for n, v in zip((100, 200, 300), curve, strict=True):
            rows.append({"dataset": "d", "model": "m", "strategy": strategy, "n_labeled": n,
                         "accuracy": v})  # fmt: skip
    s = label_savings(pd.DataFrame(rows), "accuracy").set_index("strategy")
    # Random plateaus: it first reaches its final 0.6 at 200 labels, so its savings are 0.
    assert s.loc["random", "labels_to_match_random"] == 200
    assert s.loc["random", "label_savings"] == 0
    assert s.loc["entropy", "labels_to_match_random"] == pytest.approx(150)
    assert s.loc["entropy", "label_savings"] == pytest.approx(0.25)
