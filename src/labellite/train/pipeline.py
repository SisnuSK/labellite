"""``labellite probe``: linear probes, k-NN, and CLIP zero-shot on cached features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from labellite.config import DatasetConfig, FinetuneConfig, MainConfig, load_config
from labellite.data.splits import load_splits, split_hash
from labellite.features.cache import FeatureCache
from labellite.train.knn import knn_fit, l2_normalize
from labellite.train.probe import fit_linear_probe, logits, softmax
from labellite.train.runs import (
    EVAL_SPLITS,
    DataView,
    RunId,
    Timer,
    budget_seeds,
    is_done,
    save_run,
)
from labellite.train.zeroshot import build_prompts, text_embeddings, zero_shot_logits
from labellite.utils.seed import seed_everything


def load_view(cfg: MainConfig, dataset: str, model: str) -> DataView:
    dcfg = load_config(cfg.datasets_config_dir / f"{dataset}.yaml", DatasetConfig)
    splits, budgets = load_splits(dcfg.out_dir)
    feats, ids, _ = FeatureCache.for_run(cfg.cache_dir, model, dataset).load_for_split(
        split_hash(splits)
    )
    return DataView(splits, budgets, feats, pd.Series(np.arange(len(ids)), index=ids))


def _linear(view: DataView, train_ids, seed: int, cfg: MainConfig):
    fit = fit_linear_probe(
        view.x(train_ids),
        view.labels(train_ids),
        view.x(view.ids("val")),
        view.labels(view.ids("val")),
        c_grid=cfg.c_grid,
        seed=seed,
    )
    outputs = {}
    for split in EVAL_SPLITS:
        z = logits(fit.model, view.x(view.ids(split)))
        outputs[split] = {"logits": z, "probs": softmax(z)}
    details = {"C": fit.C, "converged": fit.converged, "c_search": fit.search}
    return outputs, details, fit.model


def _knn(view: DataView, train_ids, cfg: MainConfig):
    clf = knn_fit(view.x(train_ids), view.labels(train_ids), k=cfg.knn_k)
    outputs = {
        split: {"probs": clf.predict_proba(l2_normalize(view.x(view.ids(split))))}
        for split in EVAL_SPLITS
    }
    return outputs, {"k": clf.n_neighbors, "metric": "cosine"}, clf


def run_probes(cfg: MainConfig, force: bool = False, log=print) -> None:
    for dataset in cfg.datasets:
        for model in cfg.models:
            view = load_view(cfg, dataset, model)
            for budget in cfg.label_budgets:
                for seed in budget_seeds(budget, cfg.seeds):
                    train_ids = view.train_ids(budget, seed)
                    for method in ("linear", "knn"):
                        run = RunId(dataset, model, method, budget, seed)
                        if not force and is_done(run, cfg.results_dir, view.split_hash):
                            continue
                        seed_everything(seed)
                        with Timer() as t:
                            if method == "linear":
                                outputs, details, fitted = _linear(view, train_ids, seed, cfg)
                            else:
                                outputs, details, fitted = _knn(view, train_ids, cfg)
                        m = save_run(
                            run,
                            cfg.results_dir,
                            view,
                            outputs,
                            train_ids=train_ids,
                            config=cfg.model_dump(mode="json"),
                            details=details,
                            model=fitted,
                            seconds=t.seconds,
                        )
                        log(
                            f"{dataset}/{model}/{method} b={budget:g} s={seed}: "
                            f"val acc {m['metrics']['val']['accuracy']:.4f} ({t.seconds:.0f}s)"
                        )
            if model == "clip_vitb16":
                run_zero_shot(cfg, dataset, view, force, log)


def run_zero_shot(cfg: MainConfig, dataset: str, view: DataView, force: bool, log) -> None:
    run = RunId(dataset, "clip_vitb16", "zeroshot", 0.0, 0)
    if not force and is_done(run, cfg.results_dir, view.split_hash):
        return
    from labellite.models.backbones import load_clip

    prompts_cfg = yaml.safe_load(cfg.zeroshot_prompts.read_text(encoding="utf-8"))[dataset]
    prompts = build_prompts(prompts_cfg["templates"], prompts_cfg["classnames"])
    with Timer() as t:
        clip, tokenizer = load_clip()
        clip.eval()
        text = text_embeddings(prompts, clip, tokenizer)
        scale = clip.logit_scale.exp().item()
        outputs = {}
        for split in EVAL_SPLITS:
            z = zero_shot_logits(view.x(view.ids(split)), text, scale)
            outputs[split] = {"logits": z, "probs": softmax(z)}
    m = save_run(
        run,
        cfg.results_dir,
        view,
        outputs,
        train_ids=None,
        config=cfg.model_dump(mode="json"),
        details={"prompts": prompts, "logit_scale": scale},
        seconds=t.seconds,
    )
    log(f"{dataset}/clip_vitb16/zeroshot: val acc {m['metrics']['val']['accuracy']:.4f}")


def _finetune_inputs(cfg: FinetuneConfig):
    from labellite.data.datasets import get_adapter

    dcfg = load_config(cfg.dataset_config, DatasetConfig)
    splits, budgets = load_splits(dcfg.out_dir)
    view = DataView(splits, budgets, None, None)
    return get_adapter(dcfg.name, dcfg.root), view


def _load_images(adapter, view: DataView, log) -> dict:
    """uint8 image tensors for the train split and each evaluation split, loaded once."""
    import torch

    images = {}
    for split in ("train", *EVAL_SPLITS):
        with Timer() as t:
            ids = view.ids(split)
            images[split] = (ids.reset_index(drop=True), torch.from_numpy(adapter.load_array(ids)))
        log(f"  loaded {split}: {tuple(images[split][1].shape)} in {t.seconds:.0f}s")
    return images


def run_finetune(cfg: FinetuneConfig, force: bool = False, log=print) -> None:
    import torch

    from labellite.train.finetune import finetune, predict_logits
    from labellite.train.probe import softmax

    adapter, view = _finetune_inputs(cfg)
    pending = []
    for budget in cfg.label_budgets:
        for seed in budget_seeds(budget, cfg.seeds):
            run = RunId(cfg.dataset, "resnet50", "finetune", budget, seed)
            if force or not is_done(run, cfg.results_dir, view.split_hash):
                pending.append((budget, seed))
    if not pending:
        return
    log(f"{cfg.dataset}: loading images into memory")
    images = _load_images(adapter, view, log)
    train_ids, train_all = images["train"]
    train_row = pd.Series(np.arange(len(train_ids)), index=train_ids)
    val_x, val_y = images["val"][1], view.labels(view.ids("val"))
    amp = cfg.mixed_precision and cfg.device.startswith("cuda")
    for budget, seed in pending:
        run = RunId(cfg.dataset, "resnet50", "finetune", budget, seed)
        seed_everything(seed, deterministic=cfg.deterministic)
        ids = view.train_ids(budget, seed)
        rows = torch.as_tensor(train_row[ids].to_numpy())
        train_x = train_all if budget >= 1.0 else train_all.index_select(0, rows)
        log(f"{cfg.dataset}/resnet50/finetune b={budget:g} s={seed}: {len(ids):,} images")
        ckpt = cfg.checkpoint_dir / cfg.dataset / f"b{budget:g}_s{seed}.pt"
        with Timer() as t:
            model, prep, history = finetune(
                cfg,
                train_x,
                view.labels(ids),
                val_x,
                val_y,
                adapter.num_classes,
                seed=seed,
                budget=budget,
                ckpt_path=ckpt,
                log=log,
            )
            outputs = {}
            for split in EVAL_SPLITS:
                z = predict_logits(model, images[split][1], prep, 2 * cfg.batch_size, amp)
                outputs[split] = {"logits": z, "probs": softmax(z)}
        torch.save(model.state_dict(), ckpt.with_name(ckpt.stem + "_best.pt"))
        m = save_run(
            run,
            cfg.results_dir,
            view,
            outputs,
            train_ids=ids,
            config=cfg.model_dump(mode="json"),
            details={"timm": "resnet50.tv2_in1k", **history},
            seconds=t.seconds,
        )
        log(f"  -> val acc {m['metrics']['val']['accuracy']:.4f} ({t.seconds / 60:.1f} min)")
        del model


def benchmark_finetune(cfg: FinetuneConfig, steps: int = 80, warmup: int = 15) -> float:
    """Steady-state training images/second (GPU augmentation + optimizer step) on in-memory
    images, timed after ``warmup`` steps, with no evaluation or checkpointing."""
    import time

    import torch
    import torch.nn.functional as F

    from labellite.train.finetune import Preprocessor, build_model

    adapter, view = _finetune_inputs(cfg)
    seed_everything(0, deterministic=cfg.deterministic)
    ids = view.ids("train").iloc[: cfg.batch_size * 8]
    x = torch.from_numpy(adapter.load_array(ids))
    y = torch.as_tensor(view.labels(ids))
    model = build_model(adapter.num_classes).to(cfg.device).to(memory_format=torch.channels_last)
    prep = Preprocessor(cfg, model.pretrained_cfg["mean"], model.pretrained_cfg["std"])
    gen = torch.Generator(device=cfg.device).manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model.train()
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        idx = torch.randint(0, len(ids), (cfg.batch_size,))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.mixed_precision):
            loss = F.cross_entropy(model(prep(x[idx], gen)), y[idx].to(cfg.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    return steps * cfg.batch_size / (time.perf_counter() - t0)
