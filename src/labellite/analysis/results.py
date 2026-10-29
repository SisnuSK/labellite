"""``labellite analyze`` (Phase 3 part): main results table, label-efficiency curves, and paired
model comparisons, all computed on the test split from saved run predictions.

Outputs (committed):

* ``results/tables/main_results.csv``: long format, one row per (dataset, model, method,
  budget, metric) with mean/std over seeds and, for single runs, a 95% bootstrap CI
* ``results/tables/main_results_<dataset>.md``: the same, as a readable table
* ``results/tables/paired_tests.csv`` / ``.md``: McNemar (accuracy) and paired bootstrap
  (AUROC, binary only) between methods at each (dataset, budget), seed 0, Holm-corrected within
  each (dataset, budget, test) family
* ``results/figures/label_efficiency_<dataset>.{png,pdf}``
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from labellite.data.splits import load_splits
from labellite.stats.metrics import (
    METRICS,
    bootstrap_ci,
    holm,
    mcnemar,
    metric_names,
    paired_bootstrap,
)
from labellite.train.runs import RunId, load_preds

MODEL_LABELS = {
    "resnet50": "ResNet-50",
    "dinov2_vitb14": "DINOv2 ViT-B/14",
    "clip_vitb16": "CLIP ViT-B/16",
    "phikon": "Phikon",
}
METHOD_LABELS = {
    "linear": "linear probe",
    "knn": "k-NN",
    "zeroshot": "zero-shot",
    "finetune": "fine-tuned",
}
# Reference categorical palette (light mode), fixed order: colour follows the model.
MODEL_COLORS = {
    "resnet50": "#2a78d6",
    "dinov2_vitb14": "#eb6834",
    "clip_vitb16": "#1baf7a",
    "phikon": "#eda100",
}
METHOD_STYLE = {
    "linear": dict(linestyle="-", marker="o"),
    "knn": dict(linestyle="--", marker="s"),
    "finetune": dict(linestyle=":", marker="D"),
    "zeroshot": dict(linestyle="none", marker="*"),
}
DATASET_LABELS = {"pcam": "PatchCamelyon", "eurosat": "EuroSAT"}
PRIMARY_METRIC = {"pcam": "auroc", "eurosat": "accuracy"}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "balanced_accuracy": "Balanced acc.",
    "auroc": "AUROC",
}


def collect_runs(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((Path(results_dir) / "runs").glob("*/*/*/*/run.json")):
        r = json.loads(path.read_text(encoding="utf-8"))["run"]
        rows.append(RunId(**r).__dict__)
    return pd.DataFrame(rows, columns=["dataset", "model", "method", "budget", "seed"])


def _test_labels(splits_dir: Path, dataset: str) -> np.ndarray:
    splits, _ = load_splits(Path(splits_dir) / dataset)
    return splits.loc[splits["split"] == "test", "label"].to_numpy()


def main_table(
    runs: pd.DataFrame, results_dir: Path, splits_dir: Path, n_resamples: int = 1000
) -> pd.DataFrame:
    rows = []
    for (dataset, model, method, budget), g in runs.groupby(
        ["dataset", "model", "method", "budget"]
    ):
        y = _test_labels(splits_dir, dataset)
        probs = {
            s: load_preds(RunId(dataset, model, method, budget, s), results_dir)["test_probs"]
            for s in g["seed"]
        }
        for metric in metric_names(next(iter(probs.values())).shape[1]):
            vals = np.array([METRICS[metric](y, p) for p in probs.values()])
            row = dict(
                dataset=dataset, model=model, method=method, budget=budget, metric=metric,
                n_seeds=len(vals), mean=vals.mean(),
                std=vals.std(ddof=1) if len(vals) > 1 else np.nan,
                ci_low=np.nan, ci_high=np.nan,
            )  # fmt: skip
            if len(vals) == 1:
                _, row["ci_low"], row["ci_high"] = bootstrap_ci(
                    y, next(iter(probs.values())), metric, n_resamples=n_resamples
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _cell(r: pd.Series) -> str:
    if r["n_seeds"] > 1:
        return f"{r['mean']:.3f} ± {r['std']:.3f}"
    if np.isnan(r["ci_low"]):
        return f"{r['mean']:.3f}"
    return f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"


def _budget_label(b: float) -> str:
    return "0 (zero-shot)" if b == 0 else f"{b * 100:g}%"


def table_markdown(table: pd.DataFrame, dataset: str) -> str:
    t = table[table["dataset"] == dataset].copy()
    t["cell"] = t.apply(_cell, axis=1)
    metrics = [m for m in METRIC_LABELS if m in set(t["metric"])]
    budgets = sorted(t["budget"].unique())
    order = {m: i for i, m in enumerate(MODEL_LABELS)}
    morder = {m: i for i, m in enumerate(METHOD_LABELS)}
    combos = sorted(
        t[["model", "method"]].drop_duplicates().itertuples(index=False),
        key=lambda r: (order.get(r.model, 99), morder.get(r.method, 99)),
    )
    lines = [
        f"### {DATASET_LABELS.get(dataset, dataset)} (test split)",
        "",
        "Mean ± std over 3 seeds for the 1% and 10% budgets; single runs (100%, zero-shot) show "
        "the estimate with a 95% bootstrap CI (1,000 resamples of test items).",
        "",
    ]
    for metric in metrics:
        lines += [f"**{METRIC_LABELS[metric]}**", ""]
        lines.append("| Model | Method | " + " | ".join(_budget_label(b) for b in budgets) + " |")
        lines.append("|---|---|" + "---|" * len(budgets))
        cells = t[t["metric"] == metric].set_index(["model", "method", "budget"])["cell"]
        for c in combos:
            vals = [cells.get((c.model, c.method, b), "") for b in budgets]
            lines.append(
                f"| {MODEL_LABELS.get(c.model, c.model)} | {METHOD_LABELS.get(c.method, c.method)}"
                f" | " + " | ".join(vals) + " |"
            )
        lines.append("")
    return "\n".join(lines)


def paired_tests(
    runs: pd.DataFrame, results_dir: Path, splits_dir: Path, n_resamples: int = 1000
) -> pd.DataFrame:
    """All pairs of (model, method) at each (dataset, budget), using seed-0 runs."""
    rows = []
    trained = runs[(runs["seed"] == 0) & (runs["budget"] > 0)]
    for (dataset, budget), g in trained.groupby(["dataset", "budget"]):
        y = _test_labels(splits_dir, dataset)
        probs = {
            (m, me): load_preds(RunId(dataset, m, me, budget, 0), results_dir)["test_probs"]
            for m, me in g[["model", "method"]].itertuples(index=False)
        }
        family = []
        for a, b in itertools.combinations(sorted(probs), 2):
            pa, pb = probs[a], probs[b]
            r = mcnemar(pa.argmax(1) == y, pb.argmax(1) == y)
            family.append(
                dict(test="mcnemar_accuracy", diff=METRICS["accuracy"](y, pa)
                     - METRICS["accuracy"](y, pb), a=a, b=b, **r)
            )  # fmt: skip
            if pa.shape[1] == 2:
                r = paired_bootstrap(y, pa, pb, "auroc", n_resamples=n_resamples)
                family.append(dict(test="paired_bootstrap_auroc", a=a, b=b, **r))
        fam = pd.DataFrame(family)
        for idx in fam.groupby("test").groups.values():
            fam.loc[idx, "p_holm"] = holm(fam.loc[idx, "p_value"])
        fam.insert(0, "budget", budget)
        fam.insert(0, "dataset", dataset)
        rows.append(fam)
    out = pd.concat(rows, ignore_index=True)
    out["a"] = out["a"].map(lambda t: f"{t[0]}/{t[1]}")
    out["b"] = out["b"].map(lambda t: f"{t[0]}/{t[1]}")
    return out


def _p_label(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def paired_markdown(tests: pd.DataFrame) -> str:
    lines = [
        "### Paired comparisons on the test split (seed-0 runs)",
        "",
        "McNemar's exact test on per-image correctness, and a paired bootstrap of the AUROC "
        "difference (PCam). `diff` = A − B. p-values are Holm-corrected within each "
        "(dataset, budget, test) family.",
        "",
        "| Dataset | Budget | Test | A | B | diff | p (Holm) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in tests.itertuples():
        lines.append(
            f"| {r.dataset} | {_budget_label(r.budget)} | {r.test} | {r.a} | {r.b} | "
            f"{r.diff:+.4f} | {_p_label(r.p_holm)} |"
        )
    return "\n".join(lines) + "\n"


def label_efficiency_figure(table: pd.DataFrame, dataset: str, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric = PRIMARY_METRIC[dataset]
    t = table[(table["dataset"] == dataset) & (table["metric"] == metric)]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    budgets = sorted(b for b in t["budget"].unique() if b > 0)
    for model in MODEL_LABELS:
        for method in ("linear", "knn", "finetune"):
            s = t[(t["model"] == model) & (t["method"] == method)].sort_values("budget")
            if s.empty:
                continue
            err = s["std"].fillna(0).to_numpy()
            ax.errorbar(
                s["budget"] * 100, s["mean"], yerr=err, color=MODEL_COLORS[model], linewidth=2,
                markersize=6, capsize=3, label=f"{MODEL_LABELS[model]} · {METHOD_LABELS[method]}",
                **METHOD_STYLE[method],
            )  # fmt: skip
    zs = t[t["method"] == "zeroshot"]
    for r in zs.itertuples():
        ax.axhline(r.mean, color=MODEL_COLORS[r.model], linewidth=1, linestyle="-.", alpha=0.8)
        ax.annotate(
            f"{MODEL_LABELS[r.model]} zero-shot (0 labels): {r.mean:.3f}",
            (budgets[0] * 100, r.mean), xytext=(2, 4), textcoords="offset points",
            fontsize=8, color="#52514e",
        )  # fmt: skip
    ax.set_xscale("log")
    ax.set_xticks([b * 100 for b in budgets], [f"{b * 100:g}%" for b in budgets])
    ax.minorticks_off()
    ax.set_xlabel("Labelled training images (% of train split)")
    ax.set_ylabel(f"Test {METRIC_LABELS[metric]}")
    ax.set_title(f"{DATASET_LABELS[dataset]}: label efficiency", loc="left", fontsize=11)
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b8b7b0")
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.text(
        0.01, 0.005, "Error bars: ±1 std over 3 seeds (1%, 10%). Single runs at 100%.",
        fontsize=7, color="#52514e",
    )  # fmt: skip
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"label_efficiency_{dataset}.{ext}" for ext in ("png", "pdf")]
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return paths


def analyze(results_dir: Path, splits_dir: Path, n_resamples: int = 1000, log=print) -> None:
    runs = collect_runs(results_dir)
    if runs.empty:
        raise FileNotFoundError(f"no runs under {results_dir}/runs; run `labellite probe` first")
    tables, figures = Path(results_dir) / "tables", Path(results_dir) / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    table = main_table(runs, results_dir, splits_dir, n_resamples)
    table.to_csv(tables / "main_results.csv", index=False, float_format="%.6f")
    for dataset in sorted(table["dataset"].unique()):
        (tables / f"main_results_{dataset}.md").write_text(
            table_markdown(table, dataset) + "\n", encoding="utf-8"
        )
        for p in label_efficiency_figure(table, dataset, figures):
            log(f"wrote {p}")
    tests = paired_tests(runs, results_dir, splits_dir, n_resamples)
    tests.to_csv(tables / "paired_tests.csv", index=False, float_format="%.6g")
    (tables / "paired_tests.md").write_text(paired_markdown(tests), encoding="utf-8")
    log(f"wrote {tables} ({len(runs)} runs)")
