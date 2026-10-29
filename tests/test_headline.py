import numpy as np
import pandas as pd
import pytest

from labellite.analysis.headline import best_triage, label_efficiency, shift_coverage


def test_label_efficiency_divides_by_finetuned_full_budget():
    rows = [
        ("resnet50", "finetune", 1.0, 0.9),
        ("phikon", "linear", 0.1, 0.81),
        ("phikon", "linear", 1.0, 0.95),
        ("phikon", "knn", 0.1, 0.5),
    ]
    main = pd.DataFrame(
        [
            {"dataset": "d", "metric": "accuracy", "model": m, "method": me, "budget": b, "mean": v}
            for m, me, b, v in rows
        ]
    )
    eff = label_efficiency(main)
    assert len(eff) == 1 and eff.loc[0, "model"] == "phikon"
    assert eff.loc[0, "ratio"] == pytest.approx(0.9)


def test_best_triage_picks_largest_auto_fraction():
    rows = []
    for model, auto in (("dinov2_vitb14", 0.8), ("phikon", 0.9)):
        for seed in range(3):
            rows.append(
                {
                    "dataset": "d",
                    "model": model,
                    "method": "linear",
                    "budget": 0.1,
                    "seed": seed,
                    "conformal": "lac",
                    "alpha": 0.05,
                    "coverage": 0.95,
                    "auto_fraction": auto,
                    "auto_error": 0.04,
                    "system_error": 0.03,
                }
            )
    best = best_triage(pd.DataFrame(rows), 0.05)
    assert best.loc[0, "model"] == "phikon" and best.loc[0, "n_seeds"] == 3


def test_shift_coverage_compares_clean_and_most_severe():
    shift = pd.DataFrame(
        [
            {
                "model": "phikon",
                "method": "linear",
                "budget": 0.1,
                "seed": 0,
                "severity": s,
                "coverage_lac_0.1": c,
                "auroc": a,
            }
            for s, c, a in ((0, 0.9, 0.98), (1, 0.89, 0.97), (3, 0.85, 0.95))
        ]
    )
    r = shift_coverage(shift).iloc[0]
    assert (r.coverage_clean, r.coverage_shifted, r.severity) == (0.9, 0.85, 3)


def test_confusions_counts_error_pairs():
    from labellite.analysis.headline import confusions

    y = np.array([0, 0, 0, 1, 1, 2])
    pred = np.array([1, 1, 0, 0, 1, 2])
    c = confusions(y, pred, ["a", "b", "c"])
    assert c[["true", "pred", "count"]].values.tolist() == [["a", "b", 2], ["b", "a", 1]]
    assert c["share_of_errors"].iloc[0] == pytest.approx(2 / 3) and c["n_errors"].iloc[0] == 3
