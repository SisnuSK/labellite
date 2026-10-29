import numpy as np
import pandas as pd
import yaml

from labellite.analysis.uncertainty import conformal_pipeline
from labellite.config import DatasetConfig, MainConfig
from labellite.data.splits import save_splits
from labellite.train.probe import softmax
from labellite.train.runs import DataView, RunId, save_run


def test_conformal_pipeline_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    n = 4000
    split = np.array(["train", "val", "cal", "test"] * (n // 4))
    k = 10  # EuroSAT's class count
    z_true = rng.normal(0, 2, (n, k))
    y = np.array([rng.choice(k, p=p) for p in softmax(z_true)])
    splits = pd.DataFrame(
        {"id": [f"i{j}" for j in range(n)], "label": y, "split": split, "eval_subset": False}
    )
    budgets = pd.DataFrame(columns=["id", "budget", "seed"])
    dcfg = DatasetConfig(
        name="eurosat",
        root=".",
        splits_dir=tmp_path / "splits",
        split_fractions={"train": 0.25, "val": 0.25, "cal": 0.25, "test": 0.25},
    )
    save_splits(splits, budgets, dcfg, dcfg.out_dir)
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "eurosat.yaml").write_text(
        yaml.safe_dump(dcfg.model_dump(mode="json")), encoding="utf-8"
    )
    view = DataView(splits, budgets, None, None)
    results = tmp_path / "results"

    def outputs(scale):
        out = {}
        for s in ("val", "cal", "test"):
            z = scale * z_true[split == s]
            out[s] = {"logits": z, "probs": softmax(z)}
        return out

    for method, budget, seeds, scale in (
        ("linear", 0.1, (0, 1), 2.0),
        ("finetune", 0.1, (0,), 0.7),
    ):
        for seed in seeds:
            run = RunId("eurosat", "resnet50", method, budget, seed)
            save_run(
                run, results, view, outputs(scale),
                train_ids=None, config={}, details={}, seconds=0,
            )  # fmt: skip
    # k-NN runs have no logits and are skipped.
    save_run(
        RunId("eurosat", "resnet50", "knn", 0.1, 0), results, view,
        {s: {"probs": softmax(z_true[split == s])} for s in ("val", "cal", "test")},
        train_ids=None, config={}, details={}, seconds=0,
    )  # fmt: skip

    cfg = MainConfig(
        datasets=["eurosat"],
        models=["resnet50"],
        results_dir=results,
        datasets_config_dir=tmp_path / "ds",
    )
    conformal_pipeline(cfg, log=lambda *_: None)

    cal = pd.read_csv(results / "tables" / "calibration_runs.csv")
    assert len(cal) == 3 and set(cal["method"]) == {"linear", "finetune"}
    # The reported logits are 2x / 0.7x the true ones: temperature scaling should undo that.
    lin = cal[cal["method"] == "linear"]
    assert np.allclose(lin["temperature"], 2.0, rtol=0.15)
    assert (cal["ece_after"] < cal["ece_before"]).all()
    conf = pd.read_csv(results / "tables" / "conformal_runs.csv")
    assert len(conf) == 3 * 2 * 2  # runs x methods x alphas
    assert conf["within_3sd"].all()
    control = pd.read_csv(results / "tables" / "conformal_control.csv")
    assert control["within_3sd"].all()
    classes = pd.read_csv(results / "tables" / "class_coverage.csv")
    assert len(classes) == len(conf) * k
    for name in ("calibration_eurosat.md", "conformal_eurosat.md"):
        assert (results / "tables" / name).read_text(encoding="utf-8").count("ResNet-50") >= 2
    for stem in ("reliability_eurosat", "class_coverage_eurosat"):
        assert (results / "figures" / f"{stem}.png").exists()
