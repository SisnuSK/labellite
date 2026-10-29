import numpy as np
import pytest

from labellite.conformal.core import (
    aps_scores,
    calibrate,
    conformal_quantile,
    coverage_distribution,
    evaluate_sets,
    predict_sets,
    split_calibration,
)
from labellite.train.probe import softmax


def synthetic(n, k=5, seed=0):
    """Labels drawn from softmax(z), with the model reporting softmax(1.5 z) (miscalibrated, so
    coverage relies on conformal calibration, not on the probabilities being right)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, (n, k))
    y = np.array([rng.choice(k, p=row) for row in softmax(z)])
    return softmax(1.5 * z), y, rng.random(n)


@pytest.mark.parametrize("method", ["lac", "aps"])
@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_coverage_meets_target_on_exchangeable_data(method, alpha):
    probs, y, u = synthetic(20000)
    cal, test = slice(0, 4000), slice(4000, None)
    q = calibrate(method, probs[cal], y[cal], alpha, u[cal])
    sets = predict_sets(method, probs[test], q, u[test])
    r = evaluate_sets(sets, y[test])
    mean, sd = coverage_distribution(4000, alpha, 16000)
    assert r["coverage"] >= 1 - alpha - 3 * sd
    assert abs(r["coverage"] - mean) <= 3 * sd


def test_coverage_holds_on_average_over_many_calibration_draws():
    probs, y, u = synthetic(30000, seed=1)
    covs = []
    for rep in range(40):
        idx = np.random.default_rng(rep).permutation(len(y))
        cal, test = idx[:500], idx[500:]
        q = calibrate("lac", probs[cal], y[cal], 0.1)
        covs.append(evaluate_sets(predict_sets("lac", probs[test], q), y[test])["coverage"])
    mean, _ = coverage_distribution(500, 0.1, len(y) - 500)
    assert np.mean(covs) == pytest.approx(mean, abs=0.006)


def test_quantile_rank_and_small_n():
    scores = np.arange(1, 10) / 10  # n = 9
    # ceil(10 * 0.9) = 9 -> the 9th smallest score.
    assert conformal_quantile(scores, 0.1) == pytest.approx(0.9)
    # ceil(10 * 0.8) = 8 -> the 8th smallest.
    assert conformal_quantile(scores, 0.2) == pytest.approx(0.8)
    # Too few points for alpha = 0.05: the set must contain every class.
    assert conformal_quantile(scores, 0.05) == float("inf")


def test_aps_scores_hand_computed():
    probs = np.array([[0.5, 0.3, 0.2]])
    s = aps_scores(probs, np.array([0.5]))
    # class 0: 0 + 0.5 * 0.5; class 1: 0.5 + 0.5 * 0.3; class 2: 0.8 + 0.5 * 0.2
    np.testing.assert_allclose(s, [[0.25, 0.65, 0.9]])
    # APS sets are a prefix of the classes ranked by probability.
    sets = predict_sets("aps", probs, 0.7, np.array([0.5]))
    np.testing.assert_array_equal(sets, [[True, True, False]])


def test_triage_metrics():
    sets = np.array([[True, False], [False, True], [True, True], [False, False], [False, True]])
    y = np.array([0, 0, 1, 1, 1])
    r = evaluate_sets(sets, y)
    assert r["coverage"] == pytest.approx(3 / 5)
    assert r["auto_fraction"] == pytest.approx(3 / 5)
    assert r["routed_fraction"] == pytest.approx(2 / 5)
    assert r["auto_error"] == pytest.approx(1 / 3)  # item 1: singleton {1}, true 0
    assert r["system_error"] == pytest.approx(1 / 5)
    assert r["empty_fraction"] == pytest.approx(1 / 5)
    assert r["class_coverage"] == [0.5, pytest.approx(2 / 3)]
    assert r["min_class_coverage"] == 0.5


def test_split_calibration_is_disjoint_stratified_and_seeded():
    y = np.repeat([0, 1, 2], [100, 60, 40])
    a, b = split_calibration(y, 0.5, seed=0)
    assert len(np.intersect1d(a, b)) == 0 and len(a) + len(b) == len(y)
    assert np.bincount(y[a]).tolist() == [50, 30, 20]
    a2, _ = split_calibration(y, 0.5, seed=0)
    np.testing.assert_array_equal(a, a2)
