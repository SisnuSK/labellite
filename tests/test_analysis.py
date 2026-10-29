import numpy as np
import pandas as pd
import pytest

from labellite.analysis.results import analyze
from labellite.config import DatasetConfig
from labellite.data.splits import save_splits
from labellite.stats.metrics import accuracy
from labellite.train.runs import DataView, RunId, save_run


@pytest.fixture
def fake_results(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    split = np.array(["train", "val", "cal", "test"] * (n // 4))
    splits = pd.DataFrame(
        {
            "id": [f"i{k}" for k in range(n)],
            "label": y,
            "split": split,
            "eval_subset": split == "test",
        }
    )
    budgets = pd.DataFrame(columns=["id", "budget", "seed"])
    cfg = DatasetConfig(name="pcam", root=".", split_fractions={"val": 0.5, "cal": 0.5})
    save_splits(splits, budgets, cfg, tmp_path / "splits" / "pcam")
    view = DataView(splits, budgets, None, None)

    def preds(quality):
        out = {}
        for s in ("val", "cal", "test"):
            ys = view.labels(view.ids(s))
            p1 = np.clip(
                np.where(ys == 1, quality, 1 - quality) + 0.1 * rng.standard_normal(len(ys)), 0, 1
            )
            out[s] = {"probs": np.stack([1 - p1, p1], 1)}
        return out

    results = tmp_path / "results"
    for model, q in (("resnet50", 0.7), ("phikon", 0.9)):
        for budget, seeds in ((0.1, (0, 1, 2)), (1.0, (0,))):
            for seed in seeds:
                save_run(
                    RunId("pcam", model, "linear", budget, seed),
                    results,
                    view,
                    preds(q),
                    train_ids=None,
                    config={},
                    details={},
                    seconds=0,
                )
    save_run(
        RunId("pcam", "clip_vitb16", "zeroshot", 0.0, 0),
        results,
        view,
        preds(0.6),
        train_ids=None,
        config={},
        details={},
        seconds=0,
    )
    return results, tmp_path / "splits", view


def test_analyze_writes_tables_figures_and_tests(fake_results):
    results, splits_dir, view = fake_results
    analyze(results, splits_dir, n_resamples=50, log=lambda *_: None)
    table = pd.read_csv(results / "tables" / "main_results.csv")
    row = table.query("model == 'phikon' and budget == 1.0 and metric == 'accuracy'").iloc[0]
    probs = np.load(RunId("pcam", "phikon", "linear", 1.0, 0).dir(results) / "preds.npz")[
        "test_probs"
    ]
    assert row["mean"] == pytest.approx(accuracy(view.labels(view.ids("test")), probs))
    assert row["ci_low"] <= row["mean"] <= row["ci_high"] and row["n_seeds"] == 1
    seeded = table.query("model == 'phikon' and budget == 0.1 and metric == 'auroc'").iloc[0]
    assert seeded["n_seeds"] == 3 and seeded["std"] >= 0
    assert (results / "figures" / "label_efficiency_pcam.png").exists()
    md = (results / "tables" / "main_results_pcam.md").read_text(encoding="utf-8")
    assert "Phikon" in md and "zero-shot" in md and "AUROC" in md
    tests = pd.read_csv(results / "tables" / "paired_tests.csv")
    assert set(tests["test"]) == {"mcnemar_accuracy", "paired_bootstrap_auroc"}
    assert (tests["p_holm"] >= tests["p_value"] - 1e-12).all()
