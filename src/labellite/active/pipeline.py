"""``labellite activelearn``: run the simulated loop for every (dataset, model, strategy, seed),
then summarize.

Per run: ``results/active/<dataset>/<model>/<strategy>_s<seed>.json`` (history + manifest);
existing runs are skipped. Outputs (committed): ``results/tables/active_learning.csv`` (one row
per run and round), ``results/tables/active_learning_<dataset>.md`` and
``results/figures/active_learning_<dataset>.{png,pdf}``.

Label savings: the target is random sampling's final mean metric. For every strategy, the labels
at which its mean curve first reaches the target (linear interpolation between rounds) are
compared with the labels random itself first needs (``1 - strategy / random``, so random is 0).
When random's curve plateaus, it can reach its own final value before the last round.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from labellite.active.loop import run_active_learning
from labellite.analysis.results import DATASET_LABELS, MODEL_COLORS, MODEL_LABELS, PRIMARY_METRIC
from labellite.analysis.uncertainty import MUTED, save_figure, style_axes
from labellite.config import ALConfig, MainConfig, load_config
from labellite.stats.metrics import evaluate
from labellite.train.pipeline import load_view
from labellite.utils.manifest import build_manifest

STRATEGY_LABELS = {
    "random": "random",
    "least_confidence": "least confidence",
    "entropy": "entropy",
    "conformal_set_size": "conformal set size",
    "kcenter": "k-center greedy",
}
# Colour follows the strategy here (one panel per model); fixed order from the reference palette.
STRATEGY_COLORS = {
    "random": "#52514e",
    "least_confidence": "#2a78d6",
    "entropy": "#eb6834",
    "conformal_set_size": "#1baf7a",
    "kcenter": "#8a5cd1",
}
STRATEGY_MARKERS = {
    "random": "o",
    "least_confidence": "s",
    "entropy": "^",
    "conformal_set_size": "D",
    "kcenter": "v",
}
METRIC_LABELS = {"accuracy": "Accuracy", "auroc": "AUROC", "macro_f1": "Macro-F1"}


def _run_path(cfg: ALConfig, dataset: str, model: str, strategy: str, seed: int) -> Path:
    return Path(cfg.results_dir) / dataset / model / f"{strategy}_s{seed}.json"


def _loop(cfg, main, feats, y_train, initial, val, cal, test, strategy, seed) -> list[dict]:
    return run_active_learning(
        feats,
        y_train,
        initial,
        val,
        cal,
        test,
        strategy=strategy,
        query_size=round(cfg.query_fraction * len(y_train)),
        rounds=cfg.rounds,
        seed=seed,
        alpha=cfg.conformal_alpha,
        c_grid=main.c_grid,
        evaluate=evaluate,
    )


def _one(cfg, main, dataset, model, strategy, seed, data) -> tuple[str, float]:
    import time

    from threadpoolctl import threadpool_limits

    feats, y_train, rows_of, val, cal, test, view = data
    initial = rows_of[view.train_ids(cfg.initial_fraction, seed)].to_numpy()
    t0 = time.perf_counter()
    # Same BLAS thread count in every run (worker process or not): L-BFGS results depend on it
    # at the last digits, which can flip the C chosen on val when candidates nearly tie.
    with threadpool_limits(limits=cfg.blas_threads):
        history = _loop(cfg, main, feats, y_train, initial, val, cal, test, strategy, seed)
    seconds = time.perf_counter() - t0
    manifest = build_manifest(
        "active_learning",
        cfg.model_dump(mode="json"),
        run={"dataset": dataset, "model": model, "strategy": strategy, "seed": seed},
        split_hash=view.split_hash,
        history=history,
        seconds=round(seconds, 1),
    )
    path = _run_path(cfg, dataset, model, strategy, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return f"{dataset}/{model}/{strategy} s={seed}", seconds


def run_all(cfg: ALConfig, force: bool = False, log=print) -> None:
    main = load_config(cfg.main_config, MainConfig)
    for dataset in cfg.datasets:
        for model in cfg.models:
            jobs = [
                (s, seed)
                for s in cfg.strategies
                for seed in cfg.seeds
                if force or not _run_path(cfg, dataset, model, s, seed).exists()
            ]
            if not jobs:
                continue
            view = load_view(main, dataset, model)
            train_ids = view.ids("train").reset_index(drop=True)
            rows_of = pd.Series(np.arange(len(train_ids)), index=train_ids)

            def split(name, view=view):
                ids = view.ids(name)
                return view.x(ids), view.labels(ids)

            data = (
                view.x(train_ids),
                view.labels(train_ids),
                rows_of,
                split("val"),
                split("cal"),
                split("test"),
                view,
            )
            log(f"{dataset}/{model}: {len(jobs)} runs on {len(train_ids):,} pool images")
            # k-center runs on the GPU: one at a time. Several at once overflowed the 8 GB
            # card, and Windows then pages GPU memory to system RAM, slowing them ~30x.
            cpu = [j for j in jobs if j[0] != "kcenter"]
            gpu = [j for j in jobs if j[0] == "kcenter"]
            if cpu:
                done = Parallel(n_jobs=min(cfg.n_jobs, len(cpu)), backend="loky")(
                    delayed(_one)(cfg, main, dataset, model, s, seed, data) for s, seed in cpu
                )
                for name, sec in done:
                    log(f"  {name}: {sec / 60:.1f} min")
            for s, seed in gpu:
                name, sec = _one(cfg, main, dataset, model, s, seed, data)
                log(f"  {name}: {sec / 60:.1f} min")


def collect(cfg: ALConfig) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(cfg.results_dir).glob("*/*/*_s*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        for h in m["history"]:
            rows.append({**m["run"], **h})
    return pd.DataFrame(rows)


def _first_reach(n: np.ndarray, v: np.ndarray, target: float) -> float:
    """Labels at which the curve ``v(n)`` first reaches ``target`` (interpolated), or NaN."""
    hit = np.flatnonzero(v >= target)
    if len(hit) == 0:
        return np.nan
    i = hit[0]
    if i == 0:
        return float(n[0])
    return float(n[i - 1] + (target - v[i - 1]) / (v[i] - v[i - 1]) * (n[i] - n[i - 1]))


def label_savings(curves: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per (dataset, model, strategy): labels needed to reach random's final mean ``metric``."""
    out = []
    mean = curves.groupby(["dataset", "model", "strategy", "n_labeled"])[metric].mean()
    for (dataset, model), g in mean.groupby(level=[0, 1]):
        rand = g.loc[dataset, model, "random"]
        target = rand.iloc[-1]
        n_random = _first_reach(rand.index.to_numpy(float), rand.to_numpy(), target)
        for strategy, s in g.groupby(level=2):
            s = s.droplevel([0, 1, 2])
            needed = _first_reach(s.index.to_numpy(float), s.to_numpy(), target)
            out.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "strategy": strategy,
                    "metric": metric,
                    "random_final": target,
                    "random_labels_to_reach": n_random,
                    "labels_to_match_random": needed,
                    "label_savings": 1 - needed / n_random,
                }
            )
    return pd.DataFrame(out)


def summary_markdown(curves: pd.DataFrame, savings: pd.DataFrame, dataset: str) -> str:
    metric = PRIMARY_METRIC[dataset]
    d = curves[curves["dataset"] == dataset]
    last = d[d["round"] == d["round"].max()]
    first = d[d["round"] == 0]
    n0, n1 = int(first["n_labeled"].iloc[0]), int(last["n_labeled"].iloc[0])
    lines = [
        f"### {DATASET_LABELS[dataset]}: simulated active learning (linear probe)",
        "",
        f"Start: the 1% label-budget subset ({n0:,} images); 10 rounds of +1% of the train pool "
        f"(final {n1:,} labels). Test {METRIC_LABELS[metric]} after the last round and the mean "
        "area under the curve over rounds (mean ± std over 3 seeds). Labels to match random = "
        "labels at which the strategy's mean curve first reaches random's final mean "
        "(interpolated); savings = 1 − that / the labels random itself first needs.",
        "",
        f"| Model | Strategy | Final {METRIC_LABELS[metric]} | Mean over rounds | "
        "Labels to match random | Savings |",
        "|---|---|---|---|---|---|",
    ]
    auc = d.groupby(["model", "strategy", "seed"])[metric].mean().groupby(["model", "strategy"])
    fin = last.groupby(["model", "strategy"])[metric]
    sv = savings[savings["dataset"] == dataset].set_index(["model", "strategy"])
    for model in MODEL_LABELS:
        for strategy in STRATEGY_LABELS:
            if (model, strategy) not in fin.groups:
                continue
            f, a = fin.get_group((model, strategy)), auc.get_group((model, strategy))
            r = sv.loc[(model, strategy)]
            need = r["labels_to_match_random"]
            need_s = "not reached" if np.isnan(need) else f"{need:,.0f}"
            save_s = "–" if np.isnan(need) else f"{100 * r['label_savings']:+.0f}%"
            lines.append(
                f"| {MODEL_LABELS[model]} | {STRATEGY_LABELS[strategy]} | "
                f"{f.mean():.3f} ± {f.std():.3f} | {a.mean():.3f} ± {a.std():.3f} | "
                f"{need_s} | {save_s} |"
            )
    return "\n".join(lines) + "\n"


def al_figure(curves: pd.DataFrame, dataset: str, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric = PRIMARY_METRIC[dataset]
    d = curves[curves["dataset"] == dataset]
    models = [m for m in MODEL_LABELS if m in set(d["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6), dpi=150)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models, strict=True):
        g = d[d["model"] == model].groupby(["strategy", "n_labeled"])[metric]
        stats = g.mean().to_frame("mean").join(g.std().rename("std")).reset_index()
        for strategy in STRATEGY_LABELS:
            s = stats[stats["strategy"] == strategy]
            if s.empty:
                continue
            ax.plot(
                s["n_labeled"], s["mean"], color=STRATEGY_COLORS[strategy], linewidth=2,
                marker=STRATEGY_MARKERS[strategy], markersize=4.5,
                linestyle="--" if strategy == "random" else "-",
                label=STRATEGY_LABELS[strategy],
            )  # fmt: skip
            ax.fill_between(
                s["n_labeled"], s["mean"] - s["std"], s["mean"] + s["std"],
                color=STRATEGY_COLORS[strategy], alpha=0.12, linewidth=0,
            )  # fmt: skip
        ax.set_title(MODEL_LABELS[model], fontsize=10, loc="left", color=MODEL_COLORS[model])
        ax.set_xlabel("Labelled training images", fontsize=8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        style_axes(ax)
    axes[0].set_ylabel(f"Test {METRIC_LABELS[metric]}", fontsize=8)
    axes[-1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(
        f"{DATASET_LABELS[dataset]}: active learning with a linear probe",
        x=0.01, ha="left", fontsize=11,
    )  # fmt: skip
    fig.text(0.01, -0.02, "Lines: mean over 3 seeds; bands: ±1 std. Each round adds 1% of the "
             "train pool.", fontsize=7, color=MUTED)  # fmt: skip
    fig.tight_layout()
    paths = save_figure(fig, out_dir, f"active_learning_{dataset}")
    plt.close(fig)
    return paths


def activelearn(cfg: ALConfig, force: bool = False, log=print) -> None:
    main = load_config(cfg.main_config, MainConfig)
    run_all(cfg, force, log)
    curves = collect(cfg)
    tables, figures = Path(main.results_dir) / "tables", Path(main.results_dir) / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    curves.to_csv(tables / "active_learning.csv", index=False, float_format="%.6g")
    savings = pd.concat(
        [
            label_savings(curves[curves["dataset"] == ds], PRIMARY_METRIC[ds])
            for ds in sorted(curves["dataset"].unique())
        ]
    )
    savings.to_csv(tables / "active_learning_savings.csv", index=False, float_format="%.6g")
    for dataset in sorted(curves["dataset"].unique()):
        (tables / f"active_learning_{dataset}.md").write_text(
            summary_markdown(curves, savings, dataset), encoding="utf-8"
        )
        for p in al_figure(curves, dataset, figures):
            log(f"wrote {p}")
