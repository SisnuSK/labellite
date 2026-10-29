"""``labellite`` command-line interface.

Each command validates its YAML config before doing anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from labellite import __version__
from labellite.config import (
    ALConfig,
    DatasetConfig,
    ExtractConfig,
    FinetuneConfig,
    MainConfig,
    ShiftConfig,
    load_config,
)

app = typer.Typer(help="LabelLite: label-efficient, uncertainty-aware image analysis.")
data_app = typer.Typer(help="Download datasets, build fixed splits, print dataset stats.")
app.add_typer(data_app, name="data")
features_app = typer.Typer(help="Benchmark extraction and sanity-check cached features.")
app.add_typer(features_app, name="features")

ConfigOpt = Annotated[
    Path, typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML config file.")
]


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[bool, typer.Option("--version", help="Print version and exit.")] = False,
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@data_app.command("download")
def data_download(config: ConfigOpt) -> None:
    """Download and unpack a dataset into its ``root`` (checksums verified where published)."""
    from labellite.data.download import download_dataset

    cfg = load_config(config, DatasetConfig)
    download_dataset(cfg.name, cfg.root)
    typer.echo(f"{cfg.name}: downloaded to {cfg.root}")


@data_app.command("splits")
def data_splits(config: ConfigOpt) -> None:
    """Build the fixed splits and label-budget subsets and write them to ``data/splits/``."""
    from labellite.data.datasets import get_adapter
    from labellite.data.splits import build_splits, label_budgets, save_splits

    cfg = load_config(config, DatasetConfig)
    splits = build_splits(get_adapter(cfg.name, cfg.root).index(), cfg)
    budgets = label_budgets(splits[splits["split"] == "train"], cfg.label_budgets, cfg.budget_seeds)
    manifest = save_splits(splits, budgets, cfg, cfg.out_dir)
    typer.echo(f"{cfg.name}: wrote {cfg.out_dir} (split_hash {manifest['split_hash'][:12]})")


@data_app.command("stats")
def data_stats(config: ConfigOpt) -> None:
    """Print sizes, image shapes, and class distributions per split."""
    from labellite.data.datasets import get_adapter
    from labellite.data.splits import load_splits
    from labellite.data.stats import summarize

    cfg = load_config(config, DatasetConfig)
    splits, budgets = load_splits(cfg.out_dir)
    typer.echo(summarize(get_adapter(cfg.name, cfg.root), splits, budgets))


def _extract_inputs(cfg: ExtractConfig):
    """The dataset adapter, ordered image IDs, and split hash an extraction runs over."""
    from labellite.data.datasets import get_adapter
    from labellite.data.splits import load_splits, split_hash

    dcfg = load_config(cfg.dataset_config, DatasetConfig)
    splits, _ = load_splits(dcfg.out_dir)
    return get_adapter(dcfg.name, dcfg.root), splits, split_hash(splits)


@app.command()
def extract(config: ConfigOpt) -> None:
    """Extract and cache frozen backbone features for one (model, dataset)."""
    from labellite.features.cache import FeatureCache
    from labellite.features.extract import extract_to_cache
    from labellite.models.backbones import load_backbone
    from labellite.utils.seed import seed_everything

    cfg = load_config(config, ExtractConfig)
    seed_everything(cfg.seed)
    adapter, splits, shash = _extract_inputs(cfg)
    outcome = extract_to_cache(
        load_backbone(cfg.model),
        adapter,
        splits["id"],
        shash,
        FeatureCache.for_run(cfg.cache_dir, cfg.model, cfg.dataset),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=cfg.device,
        amp=cfg.amp,
        config=cfg.model_dump(mode="json"),
        log=typer.echo,
    )
    typer.echo(f"{cfg.model}/{cfg.dataset}: {outcome}")


@features_app.command("benchmark")
def features_benchmark(
    config: ConfigOpt,
    n_batches: Annotated[int, typer.Option(help="Timed batches.")] = 30,
) -> None:
    """Measure extraction throughput and estimate full-run time and disk use (writes nothing)."""
    from labellite.features.cache import FEATURE_DTYPE
    from labellite.features.extract import benchmark
    from labellite.models.backbones import load_backbone

    cfg = load_config(config, ExtractConfig)
    adapter, splits, _ = _extract_inputs(cfg)
    backbone = load_backbone(cfg.model)
    rate = benchmark(
        backbone,
        adapter,
        splits["id"],
        n_batches=n_batches,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=cfg.device,
        amp=cfg.amp,
    )
    n = len(splits)
    gb = n * backbone.dim * np.dtype(FEATURE_DTYPE).itemsize / 1e9
    typer.echo(
        f"{cfg.model}/{cfg.dataset}: {rate:,.0f} img/s | {n:,} images -> ~{n / rate / 60:,.1f} min"
        f" | features {n:,} x {backbone.dim} {np.dtype(FEATURE_DTYPE).name} = {gb:.2f} GB"
    )


@features_app.command("check")
def features_check(
    config: ConfigOpt,
    k: Annotated[int, typer.Option(help="Neighbours.")] = 20,
) -> None:
    """Sanity check: cosine k-NN trained on the 10% label budget (seed 0), scored on val."""
    from labellite.data.splits import load_splits, split_hash
    from labellite.features.cache import FeatureCache
    from labellite.train.knn import knn_accuracy

    cfg = load_config(config, ExtractConfig)
    dcfg = load_config(cfg.dataset_config, DatasetConfig)
    splits, budgets = load_splits(dcfg.out_dir)
    cache = FeatureCache.for_run(cfg.cache_dir, cfg.model, cfg.dataset)
    feats, ids, _ = cache.load_for_split(split_hash(splits))
    row = pd.Series(np.arange(len(ids)), index=ids)
    train_ids = budgets[(budgets["budget"] == 0.1) & (budgets["seed"] == 0)]["id"]
    labels = splits.set_index("id")["label"]
    val_ids = splits.loc[splits["split"] == "val", "id"]
    acc = knn_accuracy(
        feats[row[train_ids].to_numpy()],
        labels[train_ids].to_numpy(),
        feats[row[val_ids].to_numpy()],
        labels[val_ids].to_numpy(),
        k=k,
    )
    chance = labels[val_ids].value_counts(normalize=True).max()
    typer.echo(
        f"{cfg.model}/{cfg.dataset}: {k}-NN val accuracy {acc:.4f} with 10% labels "
        f"({len(train_ids):,} train, {len(val_ids):,} val); majority-class baseline {chance:.4f}"
    )


@app.command()
def probe(
    config: ConfigOpt,
    force: Annotated[bool, typer.Option(help="Redo runs that already exist.")] = False,
) -> None:
    """Train linear probes / k-NN on cached features and run CLIP zero-shot."""
    from labellite.train.pipeline import run_probes

    run_probes(load_config(config, MainConfig), force=force, log=typer.echo)


@app.command()
def finetune(
    config: ConfigOpt,
    force: Annotated[bool, typer.Option(help="Redo runs that already exist.")] = False,
    benchmark: Annotated[
        bool, typer.Option(help="Only measure training throughput and estimate total time.")
    ] = False,
) -> None:
    """Fully fine-tune ResNet-50 at every label budget and seed."""
    from labellite.train.pipeline import _finetune_inputs, benchmark_finetune, run_finetune

    cfg = load_config(config, FinetuneConfig)
    if not benchmark:
        run_finetune(cfg, force=force, log=typer.echo)
        return
    rate = benchmark_finetune(cfg)
    _, view = _finetune_inputs(cfg)
    n_train, n_val = len(view.ids("train")), len(view.ids("val"))
    total_imgs, total_evals = 0, 0
    for b in cfg.label_budgets:
        n_seeds = 1 if b >= 1.0 else len(cfg.seeds)
        n = n_train if b >= 1.0 else len(view.train_ids(b, cfg.seeds[0]))
        total_imgs += n_seeds * n * cfg.epochs[b]
        total_evals += n_seeds * (cfg.evals_per_run + 1)
    train_min = total_imgs / rate / 60
    # val/cal/test inference runs at roughly 3x training throughput (no backward pass)
    eval_min = (total_evals * n_val + 3 * n_val * 3 * len(cfg.label_budgets)) / (3 * rate) / 60
    typer.echo(
        f"{cfg.dataset}: {rate:,.0f} train img/s at {cfg.input_size}px | "
        f"{total_imgs:,} training images -> ~{train_min:,.0f} min + ~{eval_min:,.0f} min eval "
        "(upper bound: early stopping can end runs sooner)"
    )


@app.command()
def conformal(config: ConfigOpt) -> None:
    """Temperature scaling, split conformal prediction (LAC/APS) and triage."""
    from labellite.analysis.uncertainty import conformal_pipeline

    conformal_pipeline(load_config(config, MainConfig), log=typer.echo)


@app.command()
def shift(config: ConfigOpt) -> None:
    """RQ4: stain-shift the PCam test split, re-predict with the clean models, and evaluate."""
    from labellite.analysis.shift import run_shift

    run_shift(load_config(config, ShiftConfig), log=typer.echo)


@app.command()
def activelearn(
    config: ConfigOpt,
    force: Annotated[bool, typer.Option(help="Redo runs that already exist.")] = False,
) -> None:
    """Simulated active-learning loop on cached features."""
    from labellite.active.pipeline import activelearn as run_al

    run_al(load_config(config, ALConfig), force=force, log=typer.echo)


@app.command()
def explain(
    config: ConfigOpt,
    device: Annotated[str, typer.Option(help="Torch device for Grad-CAM.")] = "cuda",
) -> None:
    """Grad-CAM error galleries, the explanation figure, and a manual-review sample."""
    from labellite.explain.gallery import explain as run_explain

    run_explain(load_config(config, MainConfig), device=device, log=typer.echo)


@app.command()
def review(
    sheet: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="review_sample.csv")],
) -> None:
    """Tally the failure types filled into a review sheet written by ``explain``."""
    from labellite.explain.gallery import summarize_review

    typer.echo(summarize_review(sheet))


@app.command()
def analyze(
    results_dir: Annotated[Path, typer.Option(help="Results directory.")] = Path("results"),
    splits_dir: Annotated[Path, typer.Option(help="Split files.")] = Path("data/splits"),
    n_resamples: Annotated[int, typer.Option(help="Bootstrap resamples.")] = 1000,
) -> None:
    """Build all tables and figures from saved results."""
    from labellite.analysis.results import analyze as run_analysis

    run_analysis(results_dir, splits_dir, n_resamples, log=typer.echo)


@app.command()
def headline(
    results_dir: Annotated[Path, typer.Option(help="Results directory.")] = Path("results"),
    splits_dir: Annotated[Path, typer.Option(help="Split files.")] = Path("data/splits"),
) -> None:
    """Recompute the summary numbers quoted in the README and report from the result tables."""
    from labellite.analysis.headline import headline as run_headline

    run_headline(results_dir, splits_dir, log=typer.echo)


if __name__ == "__main__":
    app()
