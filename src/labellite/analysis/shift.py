"""``labellite shift`` (RQ4): accuracy, calibration and conformal coverage of PCam models when the
test images are stain-shifted (see ``labellite.data.shift``).

Nothing is refitted on shifted data: every model, temperature and conformal threshold comes from
clean training / calibration images, as a deployed model would. Steps:

1. Extract frozen features of the shifted test split, per backbone and severity (feature cache;
   the cache key includes the shift parameters).
2. Predict with the saved linear probes (``model.joblib``) and the saved fine-tuned ResNet-50
   weights. Logits go to ``results/shift/runs/<dataset>/<model>/<method>/b<b>_s<s>/s<sev>.npz``
   (git-ignored). Severity 0 is the clean test split, whose logits are already saved.
3. Evaluate with ``labellite.analysis.uncertainty.evaluate_run``.

CLIP zero-shot is not included (it is near chance on clean PCam). k-NN has no logits.

Outputs (committed): ``results/tables/shift_runs_<dataset>.csv``,
``results/tables/shift_<dataset>.md``, ``results/figures/shift_<dataset>.{png,pdf}`` and
``results/figures/stain_shift_examples.{png,pdf}``.
"""

from __future__ import annotations

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
)
from labellite.analysis.uncertainty import (
    MUTED,
    RUN_COLS,
    _agg,
    _fmt,
    _order,
    _row_label,
    evaluate_run,
    logit_runs,
    make_context,
    save_figure,
    style_axes,
)
from labellite.config import DatasetConfig, FinetuneConfig, MainConfig, ShiftConfig, load_config
from labellite.data.datasets import get_adapter
from labellite.data.shift import StainShifted, stain_shift
from labellite.data.splits import load_splits, split_hash
from labellite.features.cache import FeatureCache
from labellite.train.runs import RunId, load_preds

SHIFT_METHODS = ("linear", "finetune")


def _shift_path(results_dir: Path, run: RunId, severity: float) -> Path:
    base = run.dir(Path(results_dir) / "shift")
    return base / f"s{severity:g}.npz"


def _inputs(scfg: ShiftConfig):
    cfg = load_config(scfg.main_config, MainConfig)
    dcfg = load_config(cfg.datasets_config_dir / f"{scfg.dataset}.yaml", DatasetConfig)
    splits, _ = load_splits(dcfg.out_dir)
    test_ids = splits.loc[splits["split"] == "test", "id"].reset_index(drop=True)
    return cfg, get_adapter(dcfg.name, dcfg.root), splits, test_ids


def extract_shifted(scfg: ShiftConfig, log=print) -> None:
    from labellite.features.extract import extract_to_cache
    from labellite.models.backbones import load_backbone

    cfg, base, splits, test_ids = _inputs(scfg)
    for model in scfg.models:
        backbone = None
        for sev in scfg.severities:
            adapter = StainShifted(base, sev, scfg.hematoxylin_step, scfg.eosin_step)
            cache = FeatureCache.for_run(cfg.cache_dir, model, adapter.name)
            if backbone is None:
                backbone = load_backbone(model)
            outcome = extract_to_cache(
                backbone,
                adapter,
                test_ids,
                split_hash(splits),
                cache,
                batch_size=scfg.batch_size,
                num_workers=scfg.num_workers,
                device=scfg.device,
                amp=scfg.amp,
                config=scfg.model_dump(mode="json"),
                log=log,
            )
            log(f"{model}/{adapter.name}: {outcome}")
        del backbone


def predict_linear(scfg: ShiftConfig, log=print) -> None:
    import joblib

    from labellite.train.probe import logits

    cfg, base, splits, test_ids = _inputs(scfg)
    runs = logit_runs(cfg, scfg.dataset)
    runs = runs[(runs["method"] == "linear") & runs["model"].isin(scfg.models)]
    for sev in scfg.severities:
        adapter = StainShifted(base, sev, scfg.hematoxylin_step, scfg.eosin_step)
        for model, group in runs.groupby("model", sort=False):
            feats, ids, _ = FeatureCache.for_run(cfg.cache_dir, model, adapter.name).load_for_split(
                split_hash(splits)
            )
            if not ids.reset_index(drop=True).equals(test_ids):
                raise ValueError(f"{model}/{adapter.name}: cached ids are not the test split")
            for r in group.itertuples(index=False):
                run = RunId(*r)
                out = _shift_path(cfg.results_dir, run, sev)
                if out.exists():
                    continue
                probe = joblib.load(run.dir(cfg.results_dir) / "model.joblib")
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, test_logits=logits(probe, feats).astype(np.float32))
            log(f"{model} linear: severity {sev:g} predicted ({len(group)} runs)")


def predict_finetune(scfg: ShiftConfig, log=print, chunk: int = 4096) -> None:
    import torch

    from labellite.train.finetune import Preprocessor, build_model, predict_logits

    cfg, base, _, test_ids = _inputs(scfg)
    fcfg = load_config(scfg.finetune_config, FinetuneConfig)
    runs = logit_runs(cfg, scfg.dataset)
    runs = [RunId(*r) for r in runs[runs["method"] == "finetune"].itertuples(index=False)]
    todo = [
        (run, sev)
        for run in runs
        for sev in scfg.severities
        if not _shift_path(cfg.results_dir, run, sev).exists()
    ]
    if not todo:
        return
    clean = base.load_array(test_ids)
    amp = fcfg.mixed_precision and fcfg.device.startswith("cuda")
    for sev in scfg.severities:
        pending = [run for run, s in todo if s == sev]
        if not pending:
            continue
        images = np.empty_like(clean)
        for i in range(0, len(clean), chunk):  # float64 stain maths: keep memory bounded
            images[i : i + chunk] = stain_shift(
                clean[i : i + chunk], sev, scfg.hematoxylin_step, scfg.eosin_step
            )
        images_t = torch.from_numpy(images)
        for run in pending:
            model = build_model(base.num_classes)
            ckpt = fcfg.checkpoint_dir / scfg.dataset / f"b{run.budget:g}_s{run.seed}_best.pt"
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            model = model.to(fcfg.device).to(memory_format=torch.channels_last)
            prep = Preprocessor(fcfg, model.pretrained_cfg["mean"], model.pretrained_cfg["std"])
            z = predict_logits(model, images_t, prep, 2 * fcfg.batch_size, amp)
            out = _shift_path(cfg.results_dir, run, sev)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, test_logits=z.astype(np.float32))
            log(f"resnet50 finetune b={run.budget:g} s={run.seed}: severity {sev:g} predicted")
            del model


def evaluate_shift(scfg: ShiftConfig, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg, *_ = _inputs(scfg)
    ctx = make_context(cfg, scfg.dataset)
    runs = logit_runs(cfg, scfg.dataset)
    runs = runs[runs["method"].isin(SHIFT_METHODS) & runs["model"].isin(scfg.models)]
    cal_rows, conf_rows = [], []
    for r in runs.itertuples(index=False):
        run = RunId(*r)
        preds = load_preds(run, cfg.results_dir)
        key = dict(zip(RUN_COLS, r, strict=True))
        for sev in [0.0, *scfg.severities]:
            if sev == 0:
                z_test = preds["test_logits"]
            else:
                with np.load(_shift_path(cfg.results_dir, run, sev)) as f:
                    z_test = f["test_logits"]
            cal, rows = evaluate_run(preds["cal_logits"], z_test, ctx, cfg)
            cal_rows.append({**key, "severity": sev, **cal})
            for row in rows:
                row.pop("class_coverage")
                conf_rows.append({**key, "severity": sev, **row})
    log(f"evaluated {len(runs)} runs x {1 + len(scfg.severities)} severities")
    return pd.DataFrame(cal_rows), pd.DataFrame(conf_rows)


def shift_markdown(cal: pd.DataFrame, conf: pd.DataFrame, scfg: ShiftConfig, cfg) -> str:
    sevs = sorted(cal["severity"].unique())
    alpha = max(cfg.conformal_alphas)
    lac = conf[(conf["conformal"] == "lac") & (conf["alpha"] == alpha)]
    aps = conf[(conf["conformal"] == "aps") & (conf["alpha"] == alpha)]
    tables = [
        ("Accuracy", cal, "accuracy", 1, 3),
        ("AUROC", cal, "auroc", 1, 3),
        ("ECE after temperature scaling (T fitted on clean images)", cal, "ece_after", 1, 3),
        (f"LAC coverage (%) at {100 * (1 - alpha):g}% target", lac, "coverage", 100, 1),
        (f"APS coverage (%) at {100 * (1 - alpha):g}% target", aps, "coverage", 100, 1),
        ("Auto-labeled (%), LAC", lac, "auto_fraction", 100, 1),
        ("System error (%): wrong auto-labels / all images, LAC", lac, "system_error", 100, 1),
    ]
    lines = [
        f"### {DATASET_LABELS[scfg.dataset]}: synthetic stain shift of the test split",
        "",
        f"At severity s, hematoxylin is scaled by (1 − {scfg.hematoxylin_step:g}·s) and eosin by "
        f"(1 + {scfg.eosin_step:g}·s) in Ruifrok–Johnston stain space; severity 0 is the clean "
        "test split. Models, temperatures and conformal thresholds all come from clean images. "
        "Mean ± std over 3 seeds at 1% and 10% labels.",
    ]
    for title, df, col, scale, digits in tables:
        a = _agg(df, [*RUN_COLS[:4], "severity"], [col])
        wide = {}
        for _, r in a.iterrows():
            k = (r["model"], r["method"], r["budget"])
            wide.setdefault(k, {})[r["severity"]] = _fmt(r, col, scale, digits)
        keys = _order(pd.DataFrame(list(wide), columns=["model", "method", "budget"]))
        lines += [
            "",
            f"**{title}**",
            "",
            "| Model | Method | Labels | " + " | ".join(f"s = {s:g}" for s in sevs) + " |",
            "|---|---|---|" + "---|" * len(sevs),
        ]
        for _, k in keys.iterrows():
            cells = wide[(k["model"], k["method"], k["budget"])]
            lines.append(_row_label(k) + " " + " | ".join(cells[s] for s in sevs) + " |")
    return "\n".join(lines) + "\n"


def shift_figure(cal, conf, scfg: ShiftConfig, cfg, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alpha = max(cfg.conformal_alphas)
    lac = conf[(conf["conformal"] == "lac") & (conf["alpha"] == alpha)]
    panels = [
        ("AUROC", cal, "auroc"),
        ("ECE (after temperature scaling)", cal, "ece_after"),
        (f"LAC coverage ({100 * (1 - alpha):g}% target)", lac, "coverage"),
        ("System error (LAC): wrong auto-labels", lac, "system_error"),
    ]
    budget = cfg.figure_budget
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.4), dpi=150)
    for ax, (title, df, col) in zip(axes, panels, strict=True):
        a = _agg(df[df["budget"] == budget], [*RUN_COLS[:4], "severity"], [col])
        for model in MODEL_LABELS:
            for method in SHIFT_METHODS:
                s = a[(a["model"] == model) & (a["method"] == method)].sort_values("severity")
                if s.empty:
                    continue
                ax.errorbar(
                    s["severity"], s[f"{col}_mean"], yerr=s[f"{col}_std"].fillna(0),
                    color=MODEL_COLORS[model], linewidth=2, markersize=5, capsize=3,
                    label=f"{MODEL_LABELS[model]} · {METHOD_LABELS[method]}",
                    **METHOD_STYLE[method],
                )  # fmt: skip
        if col == "coverage":
            ax.axhline(1 - alpha, color=MUTED, linewidth=1, linestyle="--")
        if col == "system_error":
            ax.axhline(alpha, color=MUTED, linewidth=1, linestyle="--")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xticks([0, *scfg.severities])
        ax.set_xlabel("Stain-shift severity (0 = clean)", fontsize=8)
        style_axes(ax)
    axes[-1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(
        f"{DATASET_LABELS[scfg.dataset]}: synthetic stain shift "
        f"({_budget_label(budget)} labels)",
        x=0.01, ha="left", fontsize=11,
    )  # fmt: skip
    fig.text(
        0.01, -0.03,
        "Error bars: ±1 std over 3 seeds. Dashed lines: target coverage 1 − α and the α bound on "
        "system error, which hold only without shift.",
        fontsize=7, color=MUTED,
    )  # fmt: skip
    fig.tight_layout()
    paths = save_figure(fig, out_dir, f"shift_{scfg.dataset}")
    plt.close(fig)
    return paths


def examples_figure(scfg: ShiftConfig, out_dir: Path, n: int = 5) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, base, splits, test_ids = _inputs(scfg)
    labels = splits.set_index("id").loc[test_ids, "label"].to_numpy()
    rng = np.random.default_rng(0)
    pick = np.concatenate([rng.choice(np.flatnonzero(labels == c), n // 2 + c, replace=False)
                           for c in (0, 1)])[:n]  # fmt: skip
    clean = base.load_array(test_ids.iloc[pick])
    sevs = [0, *scfg.severities]
    fig, axes = plt.subplots(len(sevs), n, figsize=(1.5 * n, 1.55 * len(sevs)), dpi=150)
    for i, sev in enumerate(sevs):
        shifted = stain_shift(clean, sev, scfg.hematoxylin_step, scfg.eosin_step)
        for j in range(n):
            ax = axes[i, j]
            ax.imshow(shifted[j])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(base.class_names[labels[pick[j]]], fontsize=8, color=MUTED)
            if j == 0:
                ax.set_ylabel(f"s = {sev:g}", fontsize=8)
    fig.suptitle("PCam test images under the synthetic stain shift", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "stain_shift_examples")
    plt.close(fig)
    return paths


def run_shift(scfg: ShiftConfig, log=print) -> None:
    cfg = load_config(scfg.main_config, MainConfig)
    extract_shifted(scfg, log)
    predict_linear(scfg, log)
    predict_finetune(scfg, log)
    cal, conf = evaluate_shift(scfg, log)
    tables, figures = Path(cfg.results_dir) / "tables", Path(cfg.results_dir) / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    wide = conf.pivot_table(
        index=[*RUN_COLS, "severity"],
        columns=["conformal", "alpha"],
        values=["coverage", "mean_set_size", "auto_fraction", "system_error"],
    )
    wide.columns = [f"{a}_{b}_{c:g}" for a, b, c in wide.columns]
    cal.merge(wide.reset_index(), on=[*RUN_COLS, "severity"]).to_csv(
        tables / f"shift_runs_{scfg.dataset}.csv", index=False, float_format="%.6g"
    )
    (tables / f"shift_{scfg.dataset}.md").write_text(
        shift_markdown(cal, conf, scfg, cfg), encoding="utf-8"
    )
    for p in [*shift_figure(cal, conf, scfg, cfg, figures), *examples_figure(scfg, figures)]:
        log(f"wrote {p}")
