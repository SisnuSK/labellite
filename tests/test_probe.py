import json

import numpy as np
import pandas as pd
import pytest

from labellite.train.probe import fit_linear_probe, logits, softmax
from labellite.train.runs import DataView, RunId, is_done, load_preds, save_run
from labellite.train.zeroshot import build_prompts, zero_shot_logits


def blobs(n_per_class, k, dim=16, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, dim)) * 3
    y = np.repeat(np.arange(k), n_per_class)
    return centers[y] + noise * rng.standard_normal((len(y), dim)), y


@pytest.mark.parametrize("k", [2, 4])
def test_linear_probe_learns_and_logits_match_proba(k):
    xtr, ytr = blobs(30, k, seed=0)
    xva, yva = blobs(30, k, seed=0, noise=0.31)
    fit = fit_linear_probe(xtr, ytr, xva, yva, c_grid=(0.01, 1.0, 100.0))
    assert fit.val_accuracy > 0.95 and fit.C in (0.01, 1.0, 100.0) and len(fit.search) == 3
    z = logits(fit.model, xva)
    assert z.shape == (len(yva), k)
    from labellite.train.knn import l2_normalize

    np.testing.assert_allclose(softmax(z), fit.model.predict_proba(l2_normalize(xva)), atol=1e-6)


def test_c_search_picks_best_val_accuracy():
    xtr, ytr = blobs(5, 3, seed=1, noise=2.0)
    xva, yva = blobs(50, 3, seed=1, noise=2.0)
    fit = fit_linear_probe(xtr, ytr, xva, yva, c_grid=(1e-4, 1.0, 1e4))
    assert fit.val_accuracy == max(s["val_accuracy"] for s in fit.search)


def test_probe_requires_every_class_in_training():
    xtr, ytr = blobs(10, 2)
    xva, yva = blobs(10, 3)
    with pytest.raises(ValueError, match="every class"):
        fit_linear_probe(xtr, ytr, xva, yva, c_grid=(1.0,))


def test_zero_shot_logits():
    text = np.eye(3)[:2]  # 2 classes, unit vectors
    img = np.array([[2.0, 0, 0], [0, 0.5, 0], [1.0, 1.0, 0]])
    z = zero_shot_logits(img, text, logit_scale=100.0)
    np.testing.assert_allclose(z[:2], [[100, 0], [0, 100]])
    np.testing.assert_allclose(z[2], [100 / np.sqrt(2)] * 2)
    assert build_prompts(["a {}.", "b {}."], ["x", "y"]) == [["a x.", "b x."], ["a y.", "b y."]]


def test_save_run_and_is_done(tmp_path):
    x, y = blobs(10, 2, dim=4)
    ids = [f"img{i}" for i in range(len(y))]
    split = np.array(["train", "val", "cal", "test"] * 5)
    splits = pd.DataFrame({"id": ids, "label": y, "split": split, "eval_subset": split == "test"})
    view = DataView(splits, pd.DataFrame(), x, pd.Series(np.arange(len(ids)), index=ids))
    run = RunId("toy", "tiny", "linear", 1.0, 0)
    outputs = {
        s: {"probs": np.full((5, 2), 0.5), "logits": np.zeros((5, 2))}
        for s in ("val", "cal", "test")
    }
    assert not is_done(run, tmp_path, view.split_hash)
    m = save_run(
        run,
        tmp_path,
        view,
        outputs,
        train_ids=view.ids("train"),
        config={},
        details={},
        seconds=0.0,
    )
    assert is_done(run, tmp_path, view.split_hash) and not is_done(run, tmp_path, "other-hash")
    assert m["n_train"] == 5 and "accuracy" in m["metrics"]["test"]
    assert load_preds(run, tmp_path)["test_probs"].shape == (5, 2)
    saved = json.loads((run.dir(tmp_path) / "run.json").read_text())
    assert saved["split_hash"] == view.split_hash
