"""Full fine-tuning of an ImageNet-pretrained ResNet-50 (the supervised baseline).

Images are held in memory as uint8 tensors (PCam train: ~7.3 GB) and batches are augmented,
resized, and normalized on the GPU. This avoids DataLoader worker processes, which were slow to
start and crashed (WinError 6) on Windows, and removes the CPU decoding bottleneck.

Recipe: AdamW, linear warm-up then cosine decay (per step), bf16 autocast. Augmentation, applied
per image at native resolution before resizing: random horizontal / vertical flips and 90-degree
rotations (both domains are orientation-free), and for pathology optional mild per-image
brightness / contrast / saturation jitter (factors in [0.9, 1.1]). The model is evaluated on val
``evals_per_run`` times; the best-val-accuracy weights are kept, and training stops early after
``early_stopping_patience`` evaluations without improvement.

Resumable: a checkpoint (model, optimizer, scheduler, RNG state, best weights, history) is
written after every evaluation. A resumed run continues from the last evaluation with a fresh
shuffle of the current epoch.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from labellite.config import FinetuneConfig

TIMM_ID = "resnet50.tv2_in1k"


def build_model(num_classes: int) -> nn.Module:
    import timm

    return timm.create_model(TIMM_ID, pretrained=True, num_classes=num_classes)


def lr_lambda(total_steps: int, warmup_steps: int):
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return f


def augment(x: torch.Tensor, gen: torch.Generator, color_jitter: bool) -> torch.Tensor:
    """Per-image random flips, 90-degree rotations, and optional colour jitter.

    ``x``: (B, 3, H, W) float in [0, 1], square images. ``gen`` lives on x's device.
    """
    b = x.shape[0]

    def coin():
        return torch.rand(b, generator=gen, device=x.device) < 0.5

    x = torch.where(coin()[:, None, None, None], x.flip(-1), x)
    x = torch.where(coin()[:, None, None, None], x.flip(-2), x)
    k = torch.randint(0, 4, (b,), generator=gen, device=x.device)
    rotated = x.clone()
    for r in (1, 2, 3):
        m = k == r
        if m.any():
            rotated[m] = torch.rot90(x[m], r, dims=(-2, -1))
    x = rotated
    if color_jitter:

        def factor():
            return 0.9 + 0.2 * torch.rand(b, 1, 1, 1, generator=gen, device=x.device)

        x = x * factor()  # brightness
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * factor() + mean  # contrast
        gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
        x = (x - gray) * factor() + gray  # saturation
        x = x.clamp(0, 1)
    return x


class Preprocessor:
    """uint8 (B, H, W, 3) CPU batch -> normalized (B, 3, S, S) channels-last device tensor."""

    def __init__(self, cfg: FinetuneConfig, mean, std):
        self.size, self.device, self.jitter = cfg.input_size, cfg.device, cfg.color_jitter
        self.mean = torch.tensor(mean, device=cfg.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=cfg.device).view(1, 3, 1, 1)

    def __call__(self, batch: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
        x = batch.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        if gen is not None:
            x = augment(x, gen, self.jitter)
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return x.contiguous(memory_format=torch.channels_last)


@torch.inference_mode()
def predict_logits(
    model: nn.Module, images: torch.Tensor, prep: Preprocessor, batch_size: int, amp: bool
) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out.append(model(prep(images[i : i + batch_size])).float().cpu())
    return torch.cat(out).numpy()


def finetune(
    cfg: FinetuneConfig,
    train_x: torch.Tensor,
    train_y: np.ndarray,
    val_x: torch.Tensor,
    val_y: np.ndarray,
    num_classes: int,
    *,
    seed: int,
    budget: float,
    ckpt_path: Path,
    log=print,
    max_steps: int | None = None,
) -> tuple[nn.Module, Preprocessor, dict]:
    """Fine-tune on in-memory uint8 images; return the best-val model, its preprocessor, and a
    history dict. ``max_steps`` caps training (benchmark and tests)."""
    device = cfg.device
    amp = cfg.mixed_precision and device.startswith("cuda")
    model = build_model(num_classes).to(device).to(memory_format=torch.channels_last)
    prep = Preprocessor(cfg, model.pretrained_cfg["mean"], model.pretrained_cfg["std"])
    train_y_t = torch.tensor(np.asarray(train_y), dtype=torch.long)  # copy: input may be read-only

    n, bs = len(train_y), cfg.batch_size
    steps_per_epoch = n // bs if n > bs else 1  # drop the last partial batch
    epochs = cfg.epochs[budget]
    total = epochs * steps_per_epoch if max_steps is None else max_steps
    eval_every = max(1, total // cfg.evals_per_run)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(total, max(1, int(cfg.warmup_fraction * total)))
    )
    shuffle_gen = torch.Generator().manual_seed(seed)
    aug_gen = torch.Generator(device=device).manual_seed(seed)

    state = {"step": 0, "best_acc": -1.0, "best_step": 0, "bad_evals": 0, "history": []}
    best_weights = copy.deepcopy(model.state_dict())
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        shuffle_gen.set_state(ck["shuffle_gen"])
        aug_gen.set_state(ck["aug_gen"].to("cpu"))
        state, best_weights = ck["state"], ck["best_weights"]
        log(f"  resumed from step {state['step']}")

    def save_ckpt():
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "shuffle_gen": shuffle_gen.get_state(),
                "aug_gen": aug_gen.get_state(),
                "state": state,
                "best_weights": best_weights,
            },
            tmp,
        )
        tmp.replace(ckpt_path)

    t0 = time.perf_counter()
    done = state.get("finished", False)
    while not done:
        order = torch.randperm(n, generator=shuffle_gen)
        model.train()
        for b in range(steps_per_epoch):
            idx = order[b * bs : (b + 1) * bs]
            images = prep(train_x[idx], aug_gen)
            labels = train_y_t[idx].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = F.cross_entropy(model(images), labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            state["step"] += 1
            if state["step"] % eval_every == 0 or state["step"] >= total:
                val_logits = predict_logits(model, val_x, prep, 2 * bs, amp)
                val_acc = float((val_logits.argmax(1) == val_y).mean())
                state["history"].append(
                    {"step": state["step"], "train_loss": loss.item(), "val_accuracy": val_acc}
                )
                if val_acc > state["best_acc"]:
                    state.update(best_acc=val_acc, best_step=state["step"], bad_evals=0)
                    best_weights = copy.deepcopy(model.state_dict())
                else:
                    state["bad_evals"] += 1
                stop = state["bad_evals"] >= cfg.early_stopping_patience
                done = stop or state["step"] >= total
                state["finished"] = done
                state["stopped_early"] = stop and state["step"] < total
                log(
                    f"  step {state['step']}/{total} val acc {val_acc:.4f}"
                    f" (best {state['best_acc']:.4f})"
                )
                save_ckpt()
                model.train()
                if done:
                    break
    model.load_state_dict(best_weights)
    history = {
        **{k: v for k, v in state.items() if k != "history"},
        "history": state["history"],
        "total_steps": total,
        "epochs_planned": epochs,
        "steps_per_epoch": steps_per_epoch,
        "eval_every_steps": eval_every,
        "seconds_this_run": round(time.perf_counter() - t0, 1),
    }
    return model, prep, history
