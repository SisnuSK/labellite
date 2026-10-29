"""Running and storing probe / kNN / zero-shot / fine-tune results.

Each run lives in ``<results_dir>/runs/<dataset>/<model>/<method>/b<budget>_s<seed>/``:

* ``preds.npz``: for each of ``val``, ``cal``, ``test``: ``<split>_logits`` (N, K) float32 and
  ``<split>_probs`` (N, K) float32, rows in the order of that split in ``splits.csv.gz``.
  (kNN has no logits; its vote fractions are stored as probs only.)
* ``model.joblib``: the fitted probe (linear / kNN only)
* ``run.json``: manifest: config, split hash, train-subset hash, chosen hyper-parameters, val
  metrics, timings, git commit, hardware

``preds.npz`` and ``model.joblib`` are git-ignored; ``run.json`` is committed.
Budget 0 denotes CLIP zero-shot (no labels); budget 1.0 uses the full train split (seed 0 only).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from labellite.data.splits import split_hash, subset_hash
from labellite.stats.metrics import evaluate
from labellite.utils.manifest import build_manifest

EVAL_SPLITS = ("val", "cal", "test")


@dataclass(frozen=True)
class RunId:
    dataset: str
    model: str
    method: str
    budget: float
    seed: int

    def dir(self, results_dir: Path) -> Path:
        return (
            Path(results_dir)
            / "runs"
            / self.dataset
            / self.model
            / self.method
            / f"b{self.budget:g}_s{self.seed}"
        )


@dataclass
class DataView:
    """Cached features and labels of one dataset, indexed by split."""

    splits: pd.DataFrame
    budgets: pd.DataFrame
    feats: np.ndarray
    rows: pd.Series  # image id -> row in feats

    @property
    def split_hash(self) -> str:
        return split_hash(self.splits)

    def ids(self, split: str) -> pd.Series:
        return self.splits.loc[self.splits["split"] == split, "id"]

    def labels(self, ids: pd.Series) -> np.ndarray:
        return self.splits.set_index("id").loc[ids, "label"].to_numpy()

    def x(self, ids: pd.Series) -> np.ndarray:
        return np.asarray(self.feats[self.rows[ids].to_numpy()], dtype=np.float32)

    def train_ids(self, budget: float, seed: int) -> pd.Series:
        if budget >= 1.0:
            return self.ids("train")
        b = self.budgets
        ids = b.loc[(b["budget"] == budget) & (b["seed"] == seed), "id"]
        if ids.empty:
            raise KeyError(f"no label-budget subset for budget={budget}, seed={seed}")
        return ids


def budget_seeds(budget: float, seeds: list[int]) -> list[int]:
    return [0] if budget >= 1.0 or budget == 0 else list(seeds)


def is_done(run: RunId, results_dir: Path, split_hash_: str) -> bool:
    path = run.dir(results_dir) / "run.json"
    if not (path.exists() and (run.dir(results_dir) / "preds.npz").exists()):
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("split_hash") == split_hash_


def save_run(
    run: RunId,
    results_dir: Path,
    view: DataView,
    outputs: dict[str, dict[str, np.ndarray]],
    *,
    train_ids: pd.Series | None,
    config: dict,
    details: dict,
    model=None,
    seconds: float,
) -> dict:
    """Write preds.npz, model.joblib and run.json.

    ``outputs[split]`` is ``{"probs": ..., "logits": ...}`` (logits optional).
    """
    out_dir = run.dir(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"{split}_{kind}": np.asarray(arr, dtype=np.float32)
        for split, d in outputs.items()
        for kind, arr in d.items()
    }
    np.savez_compressed(out_dir / "preds.npz", **arrays)
    if model is not None:
        joblib.dump(model, out_dir / "model.joblib")
    metrics = {
        split: evaluate(view.labels(view.ids(split)), d["probs"]) for split, d in outputs.items()
    }
    manifest = build_manifest(
        "run",
        config,
        run=run.__dict__,
        split_hash=view.split_hash,
        train_subset_hash=None if train_ids is None else subset_hash(train_ids),
        n_train=0 if train_ids is None else len(train_ids),
        details=details,
        metrics=metrics,
        seconds=round(seconds, 2),
    )
    (out_dir / "run.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def load_preds(run: RunId, results_dir: Path) -> dict[str, np.ndarray]:
    with np.load(run.dir(results_dir) / "preds.npz") as z:
        return dict(z)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
