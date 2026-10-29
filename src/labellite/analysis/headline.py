"""``labellite headline``: the summary numbers quoted in the README, report and resume bullets,
recomputed from the committed result tables so none of them is typed by hand.

Reads ``results/tables/{main_results,conformal_runs,shift_runs_pcam,active_learning_savings}.csv``
and the saved test predictions of the error-gallery reference runs (seed 0, 10% labels), and
writes ``results/tables/headline.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from labellite.analysis.results import (
    DATASET_LABELS,
    METHOD_LABELS,
    METRIC_LABELS,
    MODEL_LABELS,
    _test_labels,
)
from labellite.analysis.uncertainty import _agg
from labellite.data.datasets import ADAPTERS
from labellite.explain.gallery import REFERENCE as GALLERY_REFERENCE
from labellite.train.runs import RunId, load_preds

FROZEN = ("resnet50", "dinov2_vitb14", "clip_vitb16", "phikon")
REFERENCE = ("resnet50", "finetune", 1.0)  # fully supervised baseline: fine-tuned at 100%


def _name(model: str, method: str) -> str:
    return f"{MODEL_LABELS[model]} {METHOD_LABELS[method]}"


def label_efficiency(main: pd.DataFrame, budget: float = 0.1) -> pd.DataFrame:
    """Frozen linear probes at ``budget`` as a fraction of the fine-tuned ResNet-50 at 100%."""
    rows = []
    for (dataset, metric), d in main.groupby(["dataset", "metric"]):
        ref = d[(d.model == REFERENCE[0]) & (d.method == REFERENCE[1])]
        ref = ref[ref.budget.round(4) == REFERENCE[2]]
        if ref.empty:
            continue
        ref_value = float(ref["mean"].iloc[0])
        probes = d[(d.method == "linear") & d.model.isin(FROZEN) & (d.budget.round(4) == budget)]
        for r in probes.itertuples():
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "model": r.model,
                    "budget": budget,
                    "value": r.mean,
                    "reference": ref_value,
                    "ratio": r.mean / ref_value,
                }
            )
    return pd.DataFrame(rows)


def best_triage(conf: pd.DataFrame, alpha: float, budget: float = 0.1) -> pd.DataFrame:
    """Per dataset, the frozen linear probe + score with the largest mean auto-labeled share."""
    d = conf[(conf.method == "linear") & (conf.alpha == alpha) & (conf.budget.round(4) == budget)]
    cols = ["coverage", "auto_fraction", "auto_error", "system_error"]
    g = _agg(d, ["dataset", "model", "conformal"], cols)
    return g.loc[g.groupby("dataset")["auto_fraction_mean"].idxmax()].reset_index(drop=True)


def shift_coverage(
    shift: pd.DataFrame, score: str = "lac", alpha: float = 0.1, budget: float = 0.1
) -> pd.DataFrame:
    """Coverage and AUROC at the clean (s = 0) and most severe shift, per model / method."""
    d = shift[shift.budget.round(4) == budget]
    cov = f"coverage_{score}_{alpha:g}"
    g = _agg(d, ["model", "method", "severity"], [cov, "auroc"])
    lo, hi = g.severity.min(), g.severity.max()
    clean = g[g.severity == lo].set_index(["model", "method"])
    worst = g[g.severity == hi].set_index(["model", "method"])
    out = pd.DataFrame(
        {
            "coverage_clean": clean[f"{cov}_mean"],
            "coverage_shifted": worst[f"{cov}_mean"],
            "auroc_clean": clean["auroc_mean"],
            "auroc_shifted": worst["auroc_mean"],
        }
    )
    out["severity"] = hi
    return out.reset_index()


def confusions(y: np.ndarray, pred: np.ndarray, classes, top: int = 5) -> pd.DataFrame:
    """The ``top`` most frequent (true, predicted) error pairs, as counts and shares of errors."""
    wrong = y != pred
    pairs = pd.DataFrame(
        {"true": np.asarray(classes)[y[wrong]], "pred": np.asarray(classes)[pred[wrong]]}
    )
    out = pairs.value_counts().rename("count").reset_index().head(top)
    out["share_of_errors"] = out["count"] / max(int(wrong.sum()), 1)
    return out.assign(n_errors=int(wrong.sum()), n_test=len(y))


def error_breakdown(results_dir: Path, splits_dir: Path, budget: float = 0.1) -> list[str]:
    out = []
    for dataset, (model, method) in GALLERY_REFERENCE.items():
        run = RunId(dataset, model, method, budget, 0)
        if not (run.dir(results_dir) / "preds.npz").exists():
            continue
        y = _test_labels(splits_dir, dataset)
        pred = load_preds(run, results_dir)["test_probs"].argmax(1)
        c = confusions(y, pred, ADAPTERS[dataset].class_names)
        out += [
            "",
            f"**Most frequent errors, {DATASET_LABELS[dataset]}** ({_name(model, method)}, "
            f"{100 * budget:g}% labels, seed 0: {c.n_errors.iloc[0]:,} errors in "
            f"{c.n_test.iloc[0]:,} test images).",
            "",
            "| True | Predicted | Count | Share of errors |",
            "|---|---|---|---|",
        ]
        for r in c.itertuples():
            out.append(f"| {r.true} | {r.pred} | {r.count:,} | {100 * r.share_of_errors:.1f}% |")
    return out


def headline_markdown(results_dir: Path, splits_dir: Path = Path("data/splits")) -> str:
    tables = Path(results_dir) / "tables"
    main = pd.read_csv(tables / "main_results.csv")
    conf = pd.read_csv(tables / "conformal_runs.csv")
    shift = pd.read_csv(tables / "shift_runs_pcam.csv")
    savings = pd.read_csv(tables / "active_learning_savings.csv")

    out = [
        "### Headline numbers",
        "",
        "Generated by `labellite headline` from the result tables; means over 3 seeds.",
        "",
    ]

    out += [
        "**Label efficiency.** Frozen linear probe at 10% labels / fine-tuned ResNet-50 at "
        "100% labels.",
        "",
        "| Dataset | Metric | Probe | Probe @10% | Fine-tuned @100% | Ratio |",
        "|---|---|---|---|---|---|",
    ]
    eff = label_efficiency(main)
    eff = eff[eff.metric.isin(["accuracy", "auroc"])]
    for r in eff.sort_values(
        ["dataset", "metric", "ratio"], ascending=[True, True, False]
    ).itertuples():
        out.append(
            f"| {DATASET_LABELS[r.dataset]} | {METRIC_LABELS[r.metric]} | "
            f"{MODEL_LABELS[r.model]} | {r.value:.3f} | {r.reference:.3f} | "
            f"{100 * r.ratio:.1f}% |"
        )

    for alpha in (0.05, 0.1):
        out += [
            "",
            f"**Triage at {100 * (1 - alpha):.0f}% target coverage, 10% labels** "
            "(frozen linear probe with the largest auto-labeled share).",
            "",
            "| Dataset | Probe | Score | Coverage | Auto-labeled | Auto-label error | "
            "System error |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in best_triage(conf, alpha).itertuples():
            out.append(
                f"| {DATASET_LABELS[r.dataset]} | {MODEL_LABELS[r.model]} | "
                f"{r.conformal.upper()} | {100 * r.coverage_mean:.1f}% | "
                f"{100 * r.auto_fraction_mean:.1f}% | {100 * r.auto_error_mean:.1f}% | "
                f"{100 * r.system_error_mean:.1f}% |"
            )

    sc = shift_coverage(shift)
    out += [
        "",
        f"**Stain shift (PCam, 10% labels, severity 0 → {int(sc.severity.max())}).** "
        "LAC coverage at a 90% target, thresholds from clean images.",
        "",
        "| Model | Coverage clean | Coverage shifted | AUROC clean | AUROC shifted |",
        "|---|---|---|---|---|",
    ]
    for r in sc.sort_values("coverage_shifted").itertuples():
        out.append(
            f"| {_name(r.model, r.method)} | {100 * r.coverage_clean:.1f}% | "
            f"{100 * r.coverage_shifted:.1f}% | {r.auroc_clean:.3f} | "
            f"{r.auroc_shifted:.3f} |"
        )

    out += [
        "",
        "**Active learning.** Labels a strategy needs to reach random sampling's final "
        "score, and the saving relative to random.",
        "",
        "| Dataset | Model | Strategy | Labels | Savings |",
        "|---|---|---|---|---|",
    ]
    s = savings[savings.strategy != "random"].sort_values(
        ["dataset", "model", "label_savings"], ascending=[True, True, False]
    )
    for r in s.itertuples():
        labels = (
            "not reached"
            if pd.isna(r.labels_to_match_random)
            else f"{r.labels_to_match_random:,.0f}"
        )
        sav = "–" if pd.isna(r.label_savings) else f"{100 * r.label_savings:+.0f}%"
        out.append(
            f"| {DATASET_LABELS[r.dataset]} | {MODEL_LABELS[r.model]} | "
            f"{r.strategy.replace('_', ' ')} | {labels} | {sav} |"
        )
    out += error_breakdown(results_dir, splits_dir)
    return "\n".join(out) + "\n"


def headline(results_dir: Path, splits_dir: Path = Path("data/splits"), log=print) -> str:
    text = headline_markdown(results_dir, splits_dir)
    path = Path(results_dir) / "tables" / "headline.md"
    path.write_text(text, encoding="utf-8")
    log(f"wrote {path}")
    return text
