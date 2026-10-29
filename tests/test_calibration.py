import numpy as np
import pytest

from labellite.calibration.core import (
    apply_temperature,
    brier,
    ece,
    fit_temperature,
    nll,
    reliability_table,
)
from labellite.train.probe import softmax


def test_ece_hand_computed():
    # Bin (0.8, 0.867]: confidences 0.85, 0.85 -> 1 of 2 correct: |0.5 - 0.85| = 0.35, weight 2/4.
    # Bin (0.533, 0.6]: confidences 0.6, 0.6 -> 2 of 2 correct: |1 - 0.6| = 0.4, weight 2/4.
    probs = np.array([[0.85, 0.15], [0.85, 0.15], [0.4, 0.6], [0.4, 0.6]])
    y = np.array([0, 1, 1, 1])
    assert ece(y, probs, n_bins=15) == pytest.approx(0.5 * 0.35 + 0.5 * 0.4)
    t = reliability_table(y, probs, 15)
    assert t["count"].sum() == 4 and (t["count"] > 0).sum() == 2


def test_ece_zero_when_perfectly_calibrated():
    # Confidence 0.75 everywhere, 3 of every 4 correct.
    probs = np.tile([0.75, 0.25], (400, 1))
    y = np.tile([0, 0, 0, 1], 100)
    assert ece(y, probs) == pytest.approx(0.0, abs=1e-12)


def test_ece_bin_edges_are_right_closed():
    # Confidence exactly 0.6 = 9/15 belongs to bin (8/15, 9/15].
    t = reliability_table(np.array([0]), np.array([[0.6, 0.4]]), 15)
    assert t.loc[t["count"] == 1, "hi"].item() == pytest.approx(0.6)


def test_brier_and_nll_hand_computed():
    probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    y = np.array([0, 0])
    assert brier(y, probs) == pytest.approx(((0.2**2 + 0.2**2) + (0.7**2 + 0.7**2)) / 2)
    assert nll(y, probs) == pytest.approx(-(np.log(0.8) + np.log(0.3)) / 2)


def test_temperature_recovers_known_scale():
    # Labels drawn from softmax(z); the model reports 3 * z (overconfident), so T should be ~3.
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, (20000, 4))
    p = softmax(z)
    y = np.array([rng.choice(4, p=row) for row in p])
    t = fit_temperature(3 * z, y)
    assert t == pytest.approx(3.0, rel=0.05)
    assert ece(y, apply_temperature(3 * z, t)) < ece(y, softmax(3 * z))
    # Scaling never changes the predicted class.
    assert (apply_temperature(3 * z, t).argmax(1) == z.argmax(1)).all()
