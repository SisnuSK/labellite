"""On-disk feature cache with integrity checks.

Layout, one directory per (model, dataset): ``<cache_dir>/<model>/<dataset>/``

* ``features.npy``: (N, D) float16, rows in the order of ``ids.csv``
* ``ids.csv``: one image ID per row
* ``meta.json``: the cache key, its hash, and extraction details (hardware, timings, ...)

The **cache key** is everything that determines the features: model name, a hash of the model
weights, the preprocessing, the dataset, the split-table hash, and the extractor version. A
cache whose key differs from the expected one is *stale*: loading it raises, and ``extract``
rebuilds it, so stale features are never reused silently.

While extraction runs, rows go into ``features.partial.npy`` and ``progress.json`` records how
many rows are done (and for which key), so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EXTRACTOR_VERSION = 1
FEATURE_DTYPE = np.float16


class StaleCacheError(RuntimeError):
    pass


def make_key(
    model: str, weights_hash: str, transform_spec: dict, dataset: str, split_hash: str, dim: int
) -> dict:
    return {
        "model": model,
        "weights_hash": weights_hash,
        "transform": transform_spec,
        "dataset": dataset,
        "split_hash": split_hash,
        "dim": dim,
        "dtype": np.dtype(FEATURE_DTYPE).name,
        "extractor_version": EXTRACTOR_VERSION,
    }


def key_hash(key: dict) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


@dataclass
class FeatureCache:
    root: Path

    @classmethod
    def for_run(cls, cache_dir: Path, model: str, dataset: str) -> FeatureCache:
        return cls(Path(cache_dir) / model / dataset)

    features_path = property(lambda self: self.root / "features.npy")
    partial_path = property(lambda self: self.root / "features.partial.npy")
    ids_path = property(lambda self: self.root / "ids.csv")
    meta_path = property(lambda self: self.root / "meta.json")
    progress_path = property(lambda self: self.root / "progress.json")

    def meta(self) -> dict | None:
        if not self.meta_path.exists():
            return None
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def status(self, key: dict) -> str:
        """``"valid"``, ``"stale"`` (exists, different key or files missing), or ``"missing"``."""
        meta = self.meta()
        if meta is None:
            return "missing"
        complete = self.features_path.exists() and self.ids_path.exists()
        return "valid" if complete and meta.get("key_hash") == key_hash(key) else "stale"

    def stale_reasons(self, key: dict) -> list[str]:
        meta = self.meta() or {}
        old = meta.get("key", {})
        return [k for k in key if old.get(k) != key[k]] or ["files missing or incomplete"]

    def load(self, key: dict) -> tuple[np.ndarray, pd.Series]:
        """Features and their IDs. Raises ``StaleCacheError`` unless the cache matches ``key``."""
        status = self.status(key)
        if status != "valid":
            detail = "" if status == "missing" else f" (differs in: {self.stale_reasons(key)})"
            raise StaleCacheError(f"feature cache {self.root} is {status}{detail}; re-run extract")
        feats = np.load(self.features_path, mmap_mode="r")
        ids = pd.read_csv(self.ids_path)["id"]
        if len(ids) != len(feats):
            raise StaleCacheError(f"{self.root}: {len(ids)} ids but {len(feats)} feature rows")
        return feats, ids

    def load_for_split(self, split_hash: str) -> tuple[np.ndarray, pd.Series, dict]:
        """Load on CPU without the backbone: check the stored key is self-consistent and was built
        for the current split table. (Weights and preprocessing are checked by ``extract``.)
        """
        meta = self.meta()
        if meta is None or key_hash(meta["key"]) != meta.get("key_hash"):
            raise StaleCacheError(f"{self.root}: missing or inconsistent meta.json; re-run extract")
        if meta["key"]["split_hash"] != split_hash:
            raise StaleCacheError(
                f"{self.root} was built for a different split table; re-run extract"
            )
        feats, ids = self.load(meta["key"])
        return feats, ids, meta

    def clear(self) -> None:
        for p in (self.features_path, self.partial_path, self.ids_path, self.meta_path):
            p.unlink(missing_ok=True)
        self.progress_path.unlink(missing_ok=True)
