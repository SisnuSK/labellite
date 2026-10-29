"""``labellite conformal`` (Phase 4): temperature scaling, calibration metrics, split conformal
prediction (LAC / APS) and triage, from saved run predictions.

Runs with logits are evaluated: linear probes, ResNet-50 fine-tuning and CLIP zero-shot. k-NN is
left out: its outputs are vote fractions over k = 20 neighbours (multiples of 0.05, often exactly
0 or 1), which have no logits to temperature-scale.

Per run: the calibration split is divided (stratified, fixed seed) into a part that fits the
temperature and a part that calibrates the conformal threshold; everything is evaluated on test.

Outputs (committed):

* ``results/tables/calibration_runs.csv``, ``conformal_runs.csv``, ``class_coverage.csv``: one
  row per run (and conformal method / alpha / class)
* ``results/tables/calibration_<dataset>.md``, ``conformal_<dataset>.md``: mean ± std over seeds
* ``results/figures/reliability_<dataset>.{png,pdf}``,
  ``results/figures/class_coverage_<dataset>.{png,pdf}``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from labellite.analysis.results import (
    DATASET_LABELS,
    METHOD_LABELS,
    METHOD_STYLE,
    MODEL_COLORS,
    MODEL_LABELS,
    _budget_label,
    collect_runs,
)
from labellite.calibration.core import (
    apply_temperature,
    calibration_metrics,
    fit_temperature,
    reliability_table,
)
from labellite.config import DatasetConfig, MainConfig, load_config
from labellite.conformal.core import (
    calibrate,
    coverage_distribution,
    evaluate_sets,
    predict_sets,
    split_calibration,
)
from labellite.data.datasets import ADAPTERS
from labellite.data.splits import load_splits
from labellite.stats.metrics import auroc
from labellite.train.probe import softmax
from labellite.train.runs import RunId, load_preds

LOGIT_METHODS = ("linear", "finetune", "zeroshot")
RUN_COLS = ["dataset", "model", "method", "budget", "seed"]
GRID, AXIS, MUTED = "#e6e5e0", "#b8b7b0", "#52514e"


@dataclass
class Context:
    """Labels and fixed random draws shared by every run of one dataset."""

    dataset: str
    class_names: tuple[str, ...]
    y_cal: np.ndarray
    y_test: np.ndarray
    ts_idx: np.ndarray  # rows of cal that fit the temperature
    conf_idx: np.ndarray  # rows of cal that calibrate conformal prediction
    u_cal: np.ndarray  # APS randomization, one draw per item
    u_test: np.ndarray


def make_context(cfg: MainConfig, dataset: str) -> Context:
    dcfg = load_config(cfg.datasets_config_dir / f"{dataset}.yaml", DatasetConfig)
    splits, _ = load_splits(dcfg.out_dir)
    y_cal = splits.loc[splits["split"] == "cal", "label"].to_numpy()
    y_test = splits.loc[splits["split"] == "test", "label"].to_numpy()
    ts_idx, conf_idx = split_calibration(y_cal, cfg.cal_ts_fraction, cfg.cal_split_seed)
    rng = np.random.default_rng(cfg.cal_split_seed)
    return Context(
        dataset,
        ADAPTERS[dataset].class_names,
        y_cal,
        y_test,
        ts_idx,
        conf_idx,
        rng.random(len(y_cal)),
        rng.random(len(y_test)),
    )


def evaluate_run(
    z_cal: np.ndarray, z_test: np.ndarray, ctx: Context, cfg: MainConfig
) -> tuple[dict, list[dict]]:
    """Temperature-scale with the cal TS part, calibrate conformal on the cal conformal part, and
    evaluate on ``z_test`` (the clean test logits, or shifted ones in the RQ4 experiment)."""
    y, ts, conf = ctx.y_cal, ctx.ts_idx, ctx.conf_idx
    temperature = fit_temperature(z_cal[ts], y[ts])
    before, after = softmax(z_test.astype(np.float64)), apply_temperature(z_test, temperature)
    cal = {"temperature": temperature, "accuracy": float((after.argmax(1) == ctx.y_test).mean())}
    if after.shape[1] == 2:
        cal["auroc"] = auroc(ctx.y_test, after)
    for name, probs in (("before", before), ("after", after)):
        for k, v in calibration_metrics(ctx.y_test, probs, cfg.ece_bins).items():
            cal[f"{k}_{name}"] = v
    p_conf = apply_temperature(z_cal[conf], temperature)
    rows = []
    for method in cfg.conformal_methods:
        for alpha in cfg.conformal_alphas:
            q = calibrate(method, p_conf, y[conf], alpha, ctx.u_cal[conf])
            sets = predict_sets(method, after, q, ctx.u_test)
            mean, sd = coverage_distribution(len(conf), alpha, len(ctx.y_test))
            r = evaluate_sets(sets, ctx.y_test)
            rows.append(
                {
                    "conformal": method,
                    "alpha": alpha,
                    "threshold": q,
                    **r,
                    "expected_coverage": mean,
                    "expected_coverage_sd": sd,
                    "within_3sd": abs(r["coverage"] - mean) <= 3 * sd,
                }
            )
    return cal, rows


def exchangeable_control(z_cal: np.ndarray, z_test: np.ndarray, ctx: Context, cfg: MainConfig):
    """Conformal calibrated on a random stratified half of the *test* split and evaluated on the
    other half, so calibration and evaluation images are exchangeable by construction. Coverage
    here should match theory; a gap in the main analysis then comes from the data, not the code.
    The temperature is still fitted on the cal split (independent of test)."""
    temperature = fit_temperature(z_cal[ctx.ts_idx], ctx.y_cal[ctx.ts_idx])
    probs = apply_temperature(z_test, temperature)
    a, b = split_calibration(ctx.y_test, 0.5, cfg.cal_split_seed)
    rows = []
    for method in cfg.conformal_methods:
        for alpha in cfg.conformal_alphas:
            q = calibrate(method, probs[a], ctx.y_test[a], alpha, ctx.u_test[a])
            sets = predict_sets(method, probs[b], q, ctx.u_test[b])
            cov = float(sets[np.arange(len(b)), ctx.y_test[b]].mean())
            mean, sd = coverage_distribution(len(a), alpha, len(b))
            rows.append(
                {
                    "conformal": method,
                    "alpha": alpha,
                    "coverage": cov,
                    "expected_coverage": mean,
                    "expected_coverage_sd": sd,
                    "within_3sd": abs(cov - mean) <= 3 * sd,
                }
            )
    return rows


def logit_runs(cfg: MainConfig, dataset: str) -> pd.DataFrame:
    runs = collect_runs(cfg.results_dir)
    keep = (
        (runs["dataset"] == dataset)
        & runs["method"].isin(LOGIT_METHODS)
        & runs["model"].isin(cfg.models)
    )
    return runs[keep].reset_index(drop=True)


def _check_logits(preds: dict, run: RunId) -> None:
    """Stored probabilities must be the softmax of the stored logits (temperature 1)."""
    for split in ("cal", "test"):
        if not np.allclose(softmax(preds[f"{split}_logits"]), preds[f"{split}_probs"], atol=1e-4):
            raise ValueError(f"{run}: {split} probs are not softmax(logits)")


def run_uncertainty(cfg: MainConfig, log=print) -> dict[str, pd.DataFrame]:
    cal_rows, conf_rows, class_rows, control_rows = [], [], [], []
    for dataset in cfg.datasets:
        ctx = make_context(cfg, dataset)
        runs = logit_runs(cfg, dataset)
        log(
            f"{dataset}: {len(runs)} runs; cal split {len(ctx.y_cal):,} = "
            f"{len(ctx.ts_idx):,} temperature + {len(ctx.conf_idx):,} conformal"
        )
        for r in runs.itertuples(index=False):
            run = RunId(*r)
            preds = load_preds(run, cfg.results_dir)
            _check_logits(preds, run)
            cal, rows = evaluate_run(preds["cal_logits"], preds["test_logits"], ctx, cfg)
            key = dict(zip(RUN_COLS, r, strict=True))
            cal_rows.append({**key, **cal})
            for row in exchangeable_control(preds["cal_logits"], preds["test_logits"], ctx, cfg):
                control_rows.append({**key, **row})
            for row in rows:
                per_class = row.pop("class_coverage")
                conf_rows.append({**key, **row})
                for c, cov in enumerate(per_class):
                    class_rows.append(
                        {
                            **key,
                            "conformal": row["conformal"],
                            "alpha": row["alpha"],
                            "class": ctx.class_names[c],
                            "coverage": cov,
                            "n_test": int((ctx.y_test == c).sum()),
                        }
                    )
    return {
        "calibration": pd.DataFrame(cal_rows),
        "conformal": pd.DataFrame(conf_rows),
        "class_coverage": pd.DataFrame(class_rows),
        "control": pd.DataFrame(control_rows),
    }


# --------------------------------------------------------------------------- tables


def _order(df: pd.DataFrame) -> pd.DataFrame:
    mo = {m: i for i, m in enumerate(MODEL_LABELS)}
    me = {m: i for i, m in enumerate(METHOD_LABELS)}
    return df.assign(_mo=df["model"].map(mo), _me=df["method"].map(me)).sort_values(
        ["_mo", "_me", "budget"]
    )


def _agg(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    g = df.groupby(by, sort=False)[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
    return out.join(g.size().rename("n_seeds")).reset_index()


def _fmt(r, col: str, scale: float = 1.0, digits: int = 3) -> str:
    m, s = r[f"{col}_mean"] * scale, r[f"{col}_std"] * scale
    if np.isnan(m):
        return "–"
    return f"{m:.{digits}f}" if r["n_seeds"] < 2 else f"{m:.{digits}f} ± {s:.{digits}f}"


def _row_label(r) -> str:
    return (
        f"| {MODEL_LABELS[r['model']]} | {METHOD_LABELS[r['method']]} | "
        f"{_budget_label(r['budget'])} |"
    )


def calibration_markdown(cal: pd.DataFrame, dataset: str, cfg: MainConfig) -> str:
    cols = [f"{k}_{w}" for k in ("ece", "brier", "nll") for w in ("before", "after")]
    a = _order(_agg(cal[cal["dataset"] == dataset], RUN_COLS[:4], ["temperature", *cols]))
    lines = [
        f"### {DATASET_LABELS[dataset]}: calibration on the test split",
        "",
        f"ECE ({cfg.ece_bins} equal-width bins, top-label), Brier score and NLL before → after "
        "temperature scaling. `T` is fitted on half of the calibration split. Mean ± std over 3 "
        "seeds at 1% and 10%; single runs at 100% and zero-shot. Lower is better for all three.",
        "",
        "| Model | Method | Labels | T | ECE | Brier | NLL |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in a.iterrows():
        cells = [_fmt(r, "temperature", digits=2)] + [
            f"{_fmt(r, f'{k}_before')} → {_fmt(r, f'{k}_after')}" for k in ("ece", "brier", "nll")
        ]
        lines.append(_row_label(r) + " " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


CONF_COLS = [
    "coverage",
    "mean_set_size",
    "auto_fraction",
    "auto_error",
    "system_error",
    "min_class_coverage",
]


def conformal_markdown(
    conf: pd.DataFrame, control: pd.DataFrame, dataset: str, cfg: MainConfig, ctx: Context
) -> str:
    d = conf[conf["dataset"] == dataset]
    within, total = int(d["within_3sd"].sum()), len(d)
    c = control[control["dataset"] == dataset]
    gap = 100 * (d["coverage"] - d["expected_coverage"])
    lines = [
        f"### {DATASET_LABELS[dataset]}: split conformal prediction and triage on the test split",
        "",
        f"Calibrated on {len(ctx.conf_idx):,} calibration images (temperature-scaled "
        f"probabilities), evaluated on {len(ctx.y_test):,} test images. Auto-labeled = "
        "prediction set of exactly one class (the share of images the expert does not review); "
        "auto-label error = error rate among auto-labeled images; system error = wrong "
        "auto-labels as a share of all images (≤ α by the coverage guarantee if the expert is "
        "always right). Percentages; mean ± std over 3 seeds at 1% and 10%.",
        "",
        f"Coverage check: {within} of {total} (run, method, α) results lie within 3 sd of the "
        "coverage expected under exchangeability (coverage minus expected: "
        f"{gap.min():+.1f} to {gap.max():+.1f} points). Exchangeable control (calibrate on a "
        f"random half of test, evaluate on the other half): {int(c['within_3sd'].sum())} of "
        f"{len(c)} within 3 sd.",
    ]
    for alpha in cfg.conformal_alphas:
        mean, sd = coverage_distribution(len(ctx.conf_idx), alpha, len(ctx.y_test))
        a = _order(
            _agg(d[d["alpha"] == alpha], [*RUN_COLS[:4], "conformal"], CONF_COLS)
        ).sort_values(["_mo", "_me", "budget", "conformal"])
        lines += [
            "",
            f"**Target coverage {100 * (1 - alpha):g}%** (α = {alpha:g}; expected test coverage "
            f"{100 * mean:.2f}% ± {100 * sd:.2f}%)",
            "",
            "| Model | Method | Labels | Score | Coverage | Set size | Auto-labeled | "
            "Auto-label error | System error | Min class coverage |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in a.iterrows():
            cells = [
                r["conformal"].upper(),
                _fmt(r, "coverage", 100, 1),
                _fmt(r, "mean_set_size", 1, 2),
                _fmt(r, "auto_fraction", 100, 1),
                _fmt(r, "auto_error", 100, 1),
                _fmt(r, "system_error", 100, 1),
                _fmt(r, "min_class_coverage", 100, 1),
            ]
            lines.append(_row_label(r) + " " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- figures


def style_axes(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def save_figure(fig, out_dir: Path, stem: str, exts=("png", "pdf"), dpi=None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.{ext}" for ext in exts]
    for p in paths:
        fig.savefig(p, bbox_inches="tight", dpi=dpi or "figure")
    return paths


def figure_runs(cfg: MainConfig, dataset: str) -> list[RunId]:
    """Seed-0 runs at the figure budget (plus zero-shot), in model order."""
    runs = logit_runs(cfg, dataset)
    sel = runs[(runs["seed"] == 0) & runs["budget"].isin([0.0, cfg.figure_budget])]
    return [RunId(*r) for r in _order(sel)[RUN_COLS].itertuples(index=False)]


def reliability_figure(cfg: MainConfig, dataset: str, ctx: Context, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = figure_runs(cfg, dataset)
    ncol = 3
    nrow = -(-len(runs) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), dpi=150, squeeze=False)
    lo = 1 / len(ctx.class_names)
    for ax, run in zip(axes.flat, runs, strict=False):
        preds = load_preds(run, cfg.results_dir)
        z_cal, z_test = preds["cal_logits"], preds["test_logits"]
        t = fit_temperature(z_cal[ctx.ts_idx], ctx.y_cal[ctx.ts_idx])
        ax.plot([lo, 1], [lo, 1], color=AXIS, linewidth=1, linestyle="--")
        color, ymin = MODEL_COLORS[run.model], lo
        for name, probs, kw in (
            ("before", softmax(z_test), dict(color=MUTED, marker="o", mfc="white", alpha=0.9)),
            (f"after (T = {t:.2f})", apply_temperature(z_test, t), dict(color=color, marker="o")),
        ):
            tab = reliability_table(ctx.y_test, probs, cfg.ece_bins)
            tab = tab[tab["count"] >= 20]  # hide bins too small to estimate accuracy
            ymin = min(ymin, float(tab["accuracy"].min()))
            e = calibration_metrics(ctx.y_test, probs, cfg.ece_bins)["ece"]
            ax.plot(
                tab["confidence"], tab["accuracy"], linewidth=2, markersize=5,
                label=f"{name}: ECE {e:.3f}", **kw,
            )  # fmt: skip
        ax.set_xlim(lo - 0.02, 1.01)
        ax.set_ylim(max(0.0, ymin - 0.03), 1.01)
        budget = "0 labels" if run.budget == 0 else f"{_budget_label(run.budget)} labels"
        ax.set_title(
            f"{MODEL_LABELS[run.model]} · {METHOD_LABELS[run.method]}\n{budget}",
            fontsize=9, loc="left",
        )  # fmt: skip
        style_axes(ax)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    for ax in axes.flat[len(runs) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Mean confidence (top class)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy", fontsize=8)
    fig.suptitle(
        f"{DATASET_LABELS[dataset]}: reliability before and after temperature scaling (test)",
        x=0.01, ha="left", fontsize=11,
    )  # fmt: skip
    fig.text(
        0.01, -0.01,
        f"Seed-0 runs. Dashed line: perfect calibration. Bins with < 20 test images not drawn; "
        f"ECE uses all {cfg.ece_bins} bins.",
        fontsize=7, color=MUTED,
    )  # fmt: skip
    fig.tight_layout()
    paths = save_figure(fig, out_dir, f"reliability_{dataset}")
    plt.close(fig)
    return paths


def class_coverage_figure(
    classes: pd.DataFrame, cfg: MainConfig, dataset: str, ctx: Context, out_dir: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alpha = max(cfg.conformal_alphas)
    keys = {(r.model, r.method, r.budget, r.seed) for r in figure_runs(cfg, dataset)}
    d = classes[(classes["dataset"] == dataset) & (classes["alpha"] == alpha)]
    d = d[[k in keys for k in zip(d["model"], d["method"], d["budget"], d["seed"], strict=True)]]
    d = d[d["method"] != "zeroshot"]
    methods = list(cfg.conformal_methods)
    n_cls = len(ctx.class_names)
    width = max(4.0, 0.55 * n_cls + 1.5)
    fig, axes = plt.subplots(
        1, len(methods), figsize=(width * len(methods), 3.6), dpi=150, sharey=True, squeeze=False
    )
    series = [(m, me) for m in MODEL_LABELS for me in ("linear", "finetune")]
    series = [s for s in series if ((d["model"] == s[0]) & (d["method"] == s[1])).any()]
    offsets = np.linspace(-0.3, 0.3, len(series))
    for ax, method in zip(axes[0], methods, strict=True):
        ax.axhline(1 - alpha, color=MUTED, linewidth=1, linestyle="--")
        for off, (model, me) in zip(offsets, series, strict=True):
            s = d[(d["conformal"] == method) & (d["model"] == model) & (d["method"] == me)]
            s = s.set_index("class").loc[list(ctx.class_names)]
            ax.plot(
                np.arange(n_cls) + off, s["coverage"], linestyle="none", markersize=6,
                marker=METHOD_STYLE[me]["marker"], color=MODEL_COLORS[model],
                markeredgecolor="white", markeredgewidth=0.8,
                label=f"{MODEL_LABELS[model]} · {METHOD_LABELS[me]}",
            )  # fmt: skip
        ax.set_xticks(np.arange(n_cls), ctx.class_names, rotation=35 if n_cls > 2 else 0,
                      ha="right" if n_cls > 2 else "center")  # fmt: skip
        ax.set_title(f"{method.upper()}", fontsize=10, loc="left")
        style_axes(ax)
    axes[0, 0].set_ylabel("Test coverage within class", fontsize=8)
    axes[0, -1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(
        f"{DATASET_LABELS[dataset]}: class-conditional coverage at "
        f"{100 * (1 - alpha):g}% target ({_budget_label(cfg.figure_budget)} labels, seed 0)",
        x=0.01, ha="left", fontsize=11,
    )  # fmt: skip
    fig.text(0.01, -0.02, "Dashed line: target coverage (guaranteed marginally, not per class).",
             fontsize=7, color=MUTED)  # fmt: skip
    fig.tight_layout()
    paths = save_figure(fig, out_dir, f"class_coverage_{dataset}")
    plt.close(fig)
    return paths


def conformal_pipeline(cfg: MainConfig, log=print) -> None:
    out = run_uncertainty(cfg, log)
    tables, figures = Path(cfg.results_dir) / "tables", Path(cfg.results_dir) / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    out["calibration"].to_csv(tables / "calibration_runs.csv", index=False, float_format="%.6g")
    out["conformal"].to_csv(tables / "conformal_runs.csv", index=False, float_format="%.6g")
    out["class_coverage"].to_csv(tables / "class_coverage.csv", index=False, float_format="%.6g")
    out["control"].to_csv(tables / "conformal_control.csv", index=False, float_format="%.6g")
    for dataset in cfg.datasets:
        ctx = make_context(cfg, dataset)
        (tables / f"calibration_{dataset}.md").write_text(
            calibration_markdown(out["calibration"], dataset, cfg), encoding="utf-8"
        )
        (tables / f"conformal_{dataset}.md").write_text(
            conformal_markdown(out["conformal"], out["control"], dataset, cfg, ctx),
            encoding="utf-8",
        )
        for p in reliability_figure(cfg, dataset, ctx, figures):
            log(f"wrote {p}")
        for p in class_coverage_figure(out["class_coverage"], cfg, dataset, ctx, figures):
            log(f"wrote {p}")
        d = out["conformal"][out["conformal"]["dataset"] == dataset]
        c = out["control"][out["control"]["dataset"] == dataset]
        log(
            f"{dataset}: coverage within 3 sd of expected in {int(d['within_3sd'].sum())}/{len(d)}"
            f" (exchangeable control: {int(c['within_3sd'].sum())}/{len(c)})"
        )
    log(f"wrote {tables}")
