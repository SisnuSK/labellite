import numpy as np
import pytest

from labellite.stats.metrics import (
    accuracy,
    auroc,
    balanced_accuracy,
    bootstrap_ci,
    evaluate,
    holm,
    macro_f1,
    mcnemar,
    paired_bootstrap,
)


def onehot_probs(pred, k):
    p = np.full((len(pred), k), 0.1 / (k - 1))
    p[np.arange(len(pred)), pred] = 0.9
    return p


def test_hand_checked_metrics():
    # y:    0 0 0 0 1 1 ; pred: 0 0 0 1 1 0
    y = np.array([0, 0, 0, 0, 1, 1])
    probs = onehot_probs(np.array([0, 0, 0, 1, 1, 0]), 2)
    assert accuracy(y, probs) == pytest.approx(4 / 6)
    # recall class0 = 3/4, class1 = 1/2
    assert balanced_accuracy(y, probs) == pytest.approx((3 / 4 + 1 / 2) / 2)
    # F1 class0: P=3/4 R=3/4 -> 0.75 ; class1: P=1/2 R=1/2 -> 0.5
    assert macro_f1(y, probs) == pytest.approx(0.625)


def test_auroc_hand_checked():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    probs = np.stack([1 - scores, scores], 1)
    assert auroc(y, probs) == pytest.approx(0.75)  # 3 of 4 pos/neg pairs ordered correctly
    assert "auroc" not in evaluate(np.array([0, 1, 2]), onehot_probs(np.array([0, 1, 2]), 3))


def test_bootstrap_ci_contains_estimate_and_narrows_with_n():
    rng = np.random.default_rng(0)

    def make(n):
        y = rng.integers(0, 2, n)
        pred = np.where(rng.random(n) < 0.8, y, 1 - y)
        return y, onehot_probs(pred, 2)

    y, p = make(200)
    est, lo, hi = bootstrap_ci(y, p, "accuracy", n_resamples=500)
    assert lo <= est <= hi
    y2, p2 = make(5000)
    _, lo2, hi2 = bootstrap_ci(y2, p2, "accuracy", n_resamples=500)
    assert (hi2 - lo2) < (hi - lo)


def test_mcnemar_counts_and_extremes():
    a = np.array([1, 1, 1, 0, 1, 1], bool)
    b = np.array([1, 0, 0, 0, 1, 1], bool)
    r = mcnemar(a, b)
    assert r["only_a_correct"] == 2 and r["only_b_correct"] == 0
    assert r["p_value"] == pytest.approx(0.5)  # exact binomial, 2 of 2 discordant
    assert mcnemar(a, a)["p_value"] == 1.0
    big_a = np.ones(200, bool)
    big_b = np.r_[np.ones(100, bool), np.zeros(100, bool)]
    assert mcnemar(big_a, big_b)["p_value"] < 1e-20


def test_paired_bootstrap_detects_real_difference():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    good = onehot_probs(np.where(rng.random(2000) < 0.9, y, 1 - y), 2)
    bad = onehot_probs(np.where(rng.random(2000) < 0.7, y, 1 - y), 2)
    r = paired_bootstrap(y, good, bad, "accuracy", n_resamples=300)
    assert r["diff"] > 0.1 and r["ci_low"] > 0 and r["p_value"] < 0.01
    assert r["p_value"] == 2 / 301  # floor of the (k + 1) / (n + 1) estimate, never 0


def test_holm_matches_hand_computation():
    # p sorted: 0.01*3=0.03, 0.02*2=0.04, 0.04*1=0.04 (monotone)
    np.testing.assert_allclose(holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])
    np.testing.assert_allclose(holm([0.5, 0.6]), [1.0, 1.0])


@pytest.mark.parametrize("k", [2, 3])
def test_weighted_metrics_match_sklearn_on_resamples(k):
    from labellite.stats.metrics import METRICS, WeightedMetric

    rng = np.random.default_rng(3)
    n = 300
    y = rng.integers(0, k, n)
    logits = rng.standard_normal((n, k)) + 1.5 * np.eye(k)[y]
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    probs[:20, -1] = probs[20:40, -1]  # create tied scores
    probs = probs / probs.sum(1, keepdims=True)
    for metric in [m for m in METRICS if m != "auroc" or k == 2]:
        fn = WeightedMetric(y, probs, metric)
        assert fn(np.ones(n)) == pytest.approx(METRICS[metric](y, probs))
        for _ in range(5):
            idx = rng.integers(0, n, n)
            w = np.bincount(idx, minlength=n).astype(float)
            assert fn(w) == pytest.approx(METRICS[metric](y[idx], probs[idx]), abs=1e-12)
