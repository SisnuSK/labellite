"""``labellite explain``: Grad-CAM galleries, the explanation figure, and a review sample.

For each dataset, the explained runs are the seed-0 runs at ``figure_budget`` (10% labels): the
linear probe of every backbone and the fine-tuned ResNet-50. Outputs:

* ``results/error_gallery/<dataset>/<model>_<method>/{confident_errors,uncertain}.png`` (PNG
  only): the
  16 wrong test images with the highest confidence, and the 16 with the smallest margin between
  the two most probable classes, each shown as image + Grad-CAM overlay
* ``results/figures/explanations_<dataset>.{png,pdf}``: two confident-correct, two
  confident-wrong and two uncertain test images (picked with the reference model) with every
  model's map
* ``results/error_gallery/<dataset>/review/``: a review sample of 100 random test errors of the
  reference model (``review_sample.csv`` plus one tile per image in ``tiles/``, git-ignored),
  with empty ``failure_type`` / ``notes`` columns for a manual review; ``labellite review``
  tallies the filled-in sheet

The reference model is the most accurate linear probe at 10% labels: Phikon on PCam and DINOv2
on EuroSAT (Phase 3 tables). Maps are a qualitative aid, not evidence of model reasoning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from labellite.analysis.results import DATASET_LABELS, METHOD_LABELS, MODEL_LABELS
from labellite.analysis.uncertainty import MUTED, save_figure
from labellite.config import DatasetConfig, FinetuneConfig, MainConfig, load_config
from labellite.data.datasets import get_adapter
from labellite.data.splits import load_splits
from labellite.train.probe import softmax
from labellite.train.runs import RunId, load_preds

REFERENCE = {"pcam": ("phikon", "linear"), "eurosat": ("dinov2_vitb14", "linear")}
N_GALLERY, N_REVIEW = 16, 100
REVIEW_COLUMNS = ["failure_type", "notes"]
# Suggested failure types for the review sheet (free text is also fine).
FAILURE_TYPES = (
    "label_noise",
    "ambiguous",
    "artifact_or_blur",
    "stain_or_color",
    "small_or_edge_region",
    "look_alike_class",
    "model_error",
)


class Explainer:
    """One explained run: model on the GPU + Grad-CAM + its preprocessing."""

    def __init__(self, run: RunId, cfg: MainConfig, num_classes: int, device: str):
        import joblib
        import torch
        from PIL import Image

        from labellite.explain.gradcam import Autocast, GradCAM, ProbeHead, target_layer

        self.device = device
        if run.method == "finetune":
            from labellite.train.finetune import Preprocessor, build_model

            fcfg = load_config(f"configs/finetune/resnet50_{run.dataset}.yaml", FinetuneConfig)
            fcfg = fcfg.model_copy(update={"device": device})
            model = build_model(num_classes)
            ckpt = fcfg.checkpoint_dir / run.dataset / f"b{run.budget:g}_s{run.seed}_best.pt"
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            model.eval().requires_grad_(False)
            prep = Preprocessor(fcfg, model.pretrained_cfg["mean"], model.pretrained_cfg["std"])
            self.prepare = lambda batch: prep(torch.from_numpy(batch)).contiguous()
            layer, prefix = model.layer4, 0
            model = Autocast(model, enabled=device.startswith("cuda"))
        else:
            from labellite.models.backbones import load_backbone

            backbone = load_backbone(run.model)
            probe = joblib.load(run.dir(cfg.results_dir) / "model.joblib")
            model = ProbeHead(backbone.model, probe.coef_, probe.intercept_)
            # Through PIL, exactly as during feature extraction.
            self.prepare = lambda batch: torch.stack(
                [backbone.transform(Image.fromarray(img)) for img in batch]
            ).to(device)
            layer, prefix = target_layer(run.model, backbone.model)
        self.model = model.to(device)
        self.cam = GradCAM(self.model, layer, prefix)

    def __call__(self, images: np.ndarray, batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
        """Maps at the native image size and softmax probabilities, for uint8 (N, H, W, 3)."""
        maps, probs = [], []
        for i in range(0, len(images), batch_size):
            m, z = self.cam(self.prepare(images[i : i + batch_size]), out_size=images.shape[1])
            maps.append(m)
            probs.append(softmax(z))
        return np.concatenate(maps), np.concatenate(probs)


def explained_runs(cfg: MainConfig, dataset: str) -> list[RunId]:
    runs = [RunId(dataset, m, "linear", cfg.figure_budget, 0) for m in cfg.models]
    return [*runs, RunId(dataset, "resnet50", "finetune", cfg.figure_budget, 0)]


def _label(run: RunId) -> str:
    return f"{MODEL_LABELS[run.model]} · {METHOD_LABELS[run.method]}"


def _overlay(ax, img, cam):
    ax.imshow(img)
    ax.imshow(cam, cmap="inferno", alpha=0.45, vmin=0, vmax=1)
    _bare(ax)


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def pick(probs: np.ndarray, y: np.ndarray, kind: str, n: int) -> np.ndarray:
    """Test rows: ``confident_errors``, ``confident_correct`` or ``uncertain`` (smallest top-2
    margin), most extreme first."""
    pred, conf = probs.argmax(1), probs.max(1)
    if kind == "uncertain":
        top2 = np.sort(probs, axis=1)[:, -2:]
        return np.argsort(top2[:, 1] - top2[:, 0], kind="stable")[:n]
    rows = np.flatnonzero(pred != y if kind == "confident_errors" else pred == y)
    return rows[np.argsort(-conf[rows], kind="stable")[:n]]


def gallery_figure(images, maps, probs, y, classes, title, path_stem: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, cols = len(images), 4
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, 2 * cols, figsize=(2 * cols * 1.25, rows * 1.55), dpi=150)
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        a, b = axes[i // cols, 2 * (i % cols)], axes[i // cols, 2 * (i % cols) + 1]
        if i >= n:
            a.set_visible(False)
            b.set_visible(False)
            continue
        p = probs[i]
        a.imshow(images[i])
        _bare(a)
        _overlay(b, images[i], maps[i])
        a.set_title(f"true {classes[y[i]]}", fontsize=6, color=MUTED, loc="left")
        b.set_title(f"pred {classes[p.argmax()]} {p.max():.2f}", fontsize=6, loc="left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=9)
    fig.text(0.01, 0.0, "Grad-CAM overlays are a qualitative aid, not proof of what the model "
             "uses.", fontsize=6, color=MUTED)  # fmt: skip
    fig.tight_layout()
    # Galleries are browsed, not printed: PNG only, at a lower resolution, to keep the repo small.
    paths = save_figure(fig, path_stem.parent, path_stem.name, exts=("png",), dpi=100)
    plt.close(fig)
    return paths


def explanation_figure(cases, results, classes, dataset, out_dir) -> list[Path]:
    """``cases``: list of (kind, test row); ``results``: run -> (maps, probs) for those rows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = list(results)
    images = results[runs[0]][2]
    ncol = 1 + len(runs)
    fig, axes = plt.subplots(len(cases), ncol, figsize=(1.55 * ncol, 1.7 * len(cases)), dpi=150)
    kind_label = {
        "confident_correct": "confident, correct",
        "confident_errors": "confident, wrong",
        "uncertain": "uncertain",
    }
    for r, (kind, y) in enumerate(cases):
        ax = axes[r, 0]
        ax.imshow(images[r])
        _bare(ax)
        ax.set_ylabel(f"{kind_label[kind]}\ntrue: {classes[y]}", fontsize=6.5)
        for c, run in enumerate(runs, start=1):
            maps, probs, _ = results[run]
            ax = axes[r, c]
            _overlay(ax, images[r], maps[r])
            p = probs[r]
            ok = p.argmax() == y
            head = f"{MODEL_LABELS[run.model]}\n{METHOD_LABELS[run.method]}\n" if r == 0 else ""
            ax.set_title(
                f"{head}{classes[p.argmax()]} {p.max():.2f}", fontsize=6,
                color="#1a1a19" if ok else "#c0392b", loc="left",
            )  # fmt: skip
    axes[0, 0].set_title("image\n\n", fontsize=6, loc="left")
    fig.suptitle(
        f"{DATASET_LABELS[dataset]}: Grad-CAM for each model's predicted class "
        f"({_budget(runs[0])} labels, seed 0)",
        x=0.01, ha="left", fontsize=9, y=1.0,
    )  # fmt: skip
    fig.text(
        0.01, -0.01,
        "Rows picked with the reference model; titles: each model's prediction and probability "
        "(red = wrong). Qualitative aid only: heatmaps are not proof of what a model uses.",
        fontsize=6, color=MUTED,
    )  # fmt: skip
    fig.tight_layout()
    paths = save_figure(fig, out_dir, f"explanations_{dataset}")
    plt.close(fig)
    return paths


def _budget(run: RunId) -> str:
    return f"{run.budget * 100:g}%"


def review_sample(explainer, run, adapter, test_ids, y, probs, classes, out_dir, log):
    """100 random test errors of ``run`` as tiles + a CSV sheet for manual review."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    errors = np.flatnonzero(probs.argmax(1) != y)
    rows = np.sort(np.random.default_rng(0).choice(errors, min(N_REVIEW, len(errors)), False))
    images = adapter.load_array(test_ids.iloc[rows])
    maps, p = explainer(images)
    tiles = out_dir / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    records = []
    for k, (r, m) in enumerate(zip(rows, maps, strict=True)):
        fig, (a, b) = plt.subplots(1, 2, figsize=(4, 2.2), dpi=120)
        a.imshow(images[k])
        _bare(a)
        _overlay(b, images[k], m)
        fig.suptitle(
            f"#{k:03d}  true {classes[y[r]]}  pred {classes[p[k].argmax()]} {p[k].max():.2f}",
            fontsize=7,
        )
        name = f"{k:03d}.png"
        fig.savefig(tiles / name, bbox_inches="tight")
        plt.close(fig)
        records.append(
            {
                "tile": f"tiles/{name}",
                "id": test_ids[r],
                "true": classes[y[r]],
                "pred": classes[p[k].argmax()],
                "confidence": round(float(p[k].max()), 4),
                **dict.fromkeys(REVIEW_COLUMNS, ""),
            }
        )
    sheet = out_dir / "review_sample.csv"
    if sheet.exists():
        log(f"kept existing {sheet} (it may hold review notes)")
    else:
        pd.DataFrame(records).to_csv(sheet, index=False)
        log(f"wrote {sheet} ({len(records)} errors of {_label(run)})")
    (out_dir / "README.md").write_text(
        f"Review sample: {len(records)} random test errors of {_label(run)} "
        f"({_budget(run)} labels, seed 0).\n\nOpen each `tiles/NNN.png`, then fill `failure_type` "
        f"(suggested: {', '.join(FAILURE_TYPES)}) and `notes` in `review_sample.csv`. "
        "`labellite review --sheet <this folder>/review_sample.csv` tallies the result.\n",
        encoding="utf-8",
    )


def explain(cfg: MainConfig, device: str = "cuda", log=print) -> None:
    import torch

    gallery_root = Path(cfg.results_dir) / "error_gallery"
    figures = Path(cfg.results_dir) / "figures"
    for dataset in cfg.datasets:
        dcfg = load_config(cfg.datasets_config_dir / f"{dataset}.yaml", DatasetConfig)
        splits, _ = load_splits(dcfg.out_dir)
        test = splits[splits["split"] == "test"].reset_index(drop=True)
        y, test_ids = test["label"].to_numpy(), test["id"]
        adapter = get_adapter(dataset, dcfg.root)
        classes = adapter.class_names
        ref = RunId(dataset, *REFERENCE[dataset], cfg.figure_budget, 0)
        ref_probs = load_preds(ref, cfg.results_dir)["test_probs"]
        cases = [
            (kind, row)
            for kind in ("confident_correct", "confident_errors", "uncertain")
            for row in pick(ref_probs, y, kind, 2)
        ]
        case_rows = np.array([r for _, r in cases])
        case_images = adapter.load_array(test_ids.iloc[case_rows])
        results = {}
        for run in explained_runs(cfg, dataset):
            explainer = Explainer(run, cfg, adapter.num_classes, device)
            saved = load_preds(run, cfg.results_dir)["test_probs"]
            out = gallery_root / dataset / f"{run.model}_{run.method}"
            for kind in ("confident_errors", "uncertain"):
                rows = pick(saved, y, kind, N_GALLERY)
                images = adapter.load_array(test_ids.iloc[rows])
                maps, probs = explainer(images)
                # The explained model must reproduce the saved predictions.
                agree = (probs.argmax(1) == saved[rows].argmax(1)).mean()
                gap = np.abs(probs - saved[rows]).max()
                log(
                    f"{dataset}/{_label(run)} {kind}: class agreement {agree:.3f}, "
                    f"max abs dp {gap:.4f}"
                )
                title = f"{DATASET_LABELS[dataset]} · {_label(run)} ({_budget(run)} labels): " + (
                    "confident errors" if kind == "confident_errors" else "most uncertain"
                )
                gallery_figure(images, maps, probs, y[rows], classes, title, out / kind)
            maps, probs = explainer(case_images)
            results[run] = (maps, probs, case_images)
            if run == ref:
                review_sample(explainer, run, adapter, test_ids, y, ref_probs, classes,
                              gallery_root / dataset / "review", log)  # fmt: skip
            del explainer
            torch.cuda.empty_cache()
        for p in explanation_figure(
            [(k, y[r]) for k, r in cases], results, classes, dataset, figures
        ):
            log(f"wrote {p}")


def summarize_review(sheet: Path) -> str:
    df = pd.read_csv(sheet, keep_default_na=False)
    done = df[df["failure_type"].str.strip() != ""]
    lines = [
        f"Reviewed {len(done)} of {len(df)} images in {sheet}.",
        "",
        "| Failure type | Count | Share of reviewed |",
        "|---|---|---|",
    ]
    for t, c in done["failure_type"].str.strip().value_counts().items():
        lines.append(f"| {t} | {c} | {100 * c / len(done):.0f}% |")
    return "\n".join(lines) + "\n"
