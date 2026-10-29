"""Batch feature extraction into the feature cache (resumable)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from labellite.data.datasets import DatasetAdapter, ImageDataset
from labellite.features.cache import FEATURE_DTYPE, FeatureCache, key_hash, make_key
from labellite.models.backbones import Backbone
from labellite.utils.manifest import build_manifest


def _loader(adapter, ids, transform, batch_size, num_workers, start=0) -> DataLoader:
    ids = list(ids)[start:]
    ds = ImageDataset(adapter, ids, np.zeros(len(ids)), transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


@torch.inference_mode()
def _forward(backbone: Backbone, images: torch.Tensor, device: str, amp: bool) -> np.ndarray:
    images = images.to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.startswith("cuda")):
        feats = backbone.model(images)
    return feats.float().cpu().numpy()


def extract_to_cache(
    backbone: Backbone,
    adapter: DatasetAdapter,
    ids: pd.Series,
    split_hash: str,
    cache: FeatureCache,
    *,
    batch_size: int = 128,
    num_workers: int = 4,
    device: str = "cuda",
    amp: bool = True,
    commit_every: int = 20,
    config: dict | None = None,
    log=print,
) -> str:
    """Extract features for ``ids`` into ``cache``. Returns ``"hit"``, ``"built"`` or ``"resumed"``.

    A valid cache is a hit and nothing runs. A stale cache is cleared and rebuilt. A partial run
    with the same key resumes after the last committed batch.
    """
    backbone.model.to(device)
    key = make_key(
        backbone.name,
        backbone.weights_hash(),
        backbone.transform_spec,
        adapter.name,
        split_hash,
        backbone.dim,
    )
    status = cache.status(key)
    if status == "valid":
        log(f"cache hit: {cache.root}")
        return "hit"
    if status == "stale":
        log(f"stale cache at {cache.root} (differs in {cache.stale_reasons(key)}); rebuilding")
        cache.clear()

    cache.root.mkdir(parents=True, exist_ok=True)
    n, khash = len(ids), key_hash(key)
    done = 0
    if cache.progress_path.exists() and cache.partial_path.exists():
        progress = json.loads(cache.progress_path.read_text())
        if progress.get("key_hash") == khash and progress.get("n") == n:
            done = progress["done"]
    outcome = "resumed" if done else "built"
    if done:
        log(f"resuming {cache.root} at row {done:,}/{n:,}")
        out = np.load(cache.partial_path, mmap_mode="r+")
    else:
        out = np.lib.format.open_memmap(
            cache.partial_path, mode="w+", dtype=FEATURE_DTYPE, shape=(n, backbone.dim)
        )

    def commit(arr: np.memmap, rows: int) -> None:
        arr.flush()
        cache.progress_path.write_text(json.dumps({"key_hash": khash, "n": n, "done": rows}))

    t0 = time.perf_counter()
    row = done
    loader = _loader(adapter, ids, backbone.transform, batch_size, num_workers, start=done)
    for b, (images, _, _) in enumerate(loader, 1):
        feats = _forward(backbone, images, device, amp)
        if not np.isfinite(feats).all():
            raise FloatingPointError(f"non-finite features in rows {row}..{row + len(feats)}")
        out[row : row + len(feats)] = feats.astype(FEATURE_DTYPE)
        row += len(feats)
        if b % commit_every == 0:
            commit(out, row)
            rate = (row - done) / (time.perf_counter() - t0)
            log(f"  {row:,}/{n:,} images ({rate:,.0f} img/s)")
    elapsed = time.perf_counter() - t0
    commit(out, row)
    del out  # close the memmap so the file can be renamed (required on Windows)

    pd.DataFrame({"id": list(ids)}).to_csv(cache.ids_path, index=False)
    Path(cache.partial_path).replace(cache.features_path)
    meta = build_manifest(
        "extract",
        config or {},
        key=key,
        key_hash=khash,
        source=backbone.source,
        n_images=n,
        extraction={
            "seconds": round(elapsed, 1),
            "images_this_run": n - done,
            "resumed_from_row": done,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "device": device,
            "amp_fp16": amp and device.startswith("cuda"),
        },
    )
    cache.meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    cache.progress_path.unlink(missing_ok=True)
    log(f"wrote {cache.root}: {n:,} x {backbone.dim} in {elapsed:,.0f}s")
    return outcome


def benchmark(
    backbone: Backbone,
    adapter: DatasetAdapter,
    ids: pd.Series,
    *,
    n_batches: int = 30,
    warmup: int = 5,
    batch_size: int = 128,
    num_workers: int = 4,
    device: str = "cuda",
    amp: bool = True,
) -> float:
    """End-to-end images/second (data loading + forward), measured after ``warmup`` batches."""
    backbone.model.to(device)
    # A contiguous block from a random offset, like the sequential reads of a real extraction.
    size = min(len(ids), (n_batches + warmup) * batch_size)
    start = int(np.random.default_rng(0).integers(0, len(ids) - size + 1))
    block = ids.iloc[start : start + size]
    loader = _loader(adapter, block, backbone.transform, batch_size, num_workers)
    seen, t0 = 0, None
    for b, (images, _, _) in enumerate(loader):
        if b == warmup:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        _forward(backbone, images, device, amp)
        if b >= warmup:
            seen += len(images)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return seen / (time.perf_counter() - t0)
