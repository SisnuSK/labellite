"""Fixed train / val / cal / test splits, label-budget subsets, and split hashing.

Split files are written to ``data/splits/<dataset>/`` and committed:

* ``splits.csv.gz``: ``id, label, split, eval_subset``
* ``budgets.csv.gz``: ``id, budget, seed``, the training IDs in each label-budget subset
  (fractions below 1.0 only; the 100% budget is the whole train split)
* ``manifest.json``: config, counts, and hashes

Budgets are nested per seed: for a given seed, the 1% subset is contained in the 10% subset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from labellite.config import DatasetConfig

EVAL_SPLITS = ("test",)
_GZ = {"method": "gzip", "mtime": 0}  # mtime=0 keeps the bytes reproducible


def allocate(n: int, fractions: dict[str, float]) -> dict[str, int]:
    """Split ``n`` items into integer counts proportional to ``fractions`` (largest remainder)."""
    raw = {k: n * f for k, f in fractions.items()}
    counts = {k: int(np.floor(v)) for k, v in raw.items()}
    leftover = n - sum(counts.values())
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[:leftover]:
        counts[k] += 1
    return counts


def stratified_assign(
    strata: pd.Series, fractions: dict[str, float], rng: np.random.Generator
) -> pd.Series:
    """Assign every row a split name, preserving the stratum mix in each split."""
    out = pd.Series(index=strata.index, dtype=object)
    for _, idx in sorted(strata.groupby(strata).groups.items()):
        idx = rng.permutation(np.asarray(idx))
        start = 0
        for name, c in allocate(len(idx), fractions).items():
            out.loc[idx[start : start + c]] = name
            start += c
    return out


def _strata(df: pd.DataFrame) -> pd.Series:
    return df["label"].astype(str)


def build_splits(index: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """Turn an adapter index into the project's fixed splits.

    * EuroSAT (no official split): stratified split of all images by ``split_fractions``.
    * PCam: official train/test kept; official valid divided into val/cal.
    """
    rng = np.random.default_rng(cfg.split_seed)
    df = index.copy()
    divide = {"eurosat": "all", "pcam": "valid"}[cfg.name]
    df["split"] = df["source_split"].replace({"valid": "val"})
    mask = df["source_split"] == divide
    df.loc[mask, "split"] = stratified_assign(_strata(df[mask]), cfg.split_fractions, rng)

    df["eval_subset"] = df["split"].isin(EVAL_SPLITS)
    if cfg.eval_subset_size is not None:
        for name in EVAL_SPLITS:
            m = df["split"] == name
            n = int(m.sum())
            if n > cfg.eval_subset_size:
                frac = cfg.eval_subset_size / n
                keep = stratified_assign(_strata(df[m]), {"in": frac, "out": 1 - frac}, rng)
                df.loc[m, "eval_subset"] = keep == "in"
    return df[["id", "label", "split", "eval_subset"]].reset_index(drop=True)


def label_budgets(train: pd.DataFrame, fractions: list[float], seeds: list[int]) -> pd.DataFrame:
    """Nested, class-stratified label-budget subsets of the train split (fractions < 1 only).

    Each class keeps ``round(fraction * n_class)`` images, at least one.
    """
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = {
            c: rng.permutation(g["id"].to_numpy())
            for c, g in sorted(train.groupby("label"), key=lambda kv: kv[0])
        }
        for frac in sorted(f for f in fractions if f < 1.0):
            for ids in order.values():
                k = max(1, int(round(frac * len(ids))))
                rows.extend((i, frac, seed) for i in ids[:k])
    return pd.DataFrame(rows, columns=["id", "budget", "seed"])


def split_hash(df: pd.DataFrame, columns: tuple[str, ...] = ("id", "label", "split")) -> str:
    """Order-independent SHA-256 over the given columns."""
    lines = sorted("\t".join(map(str, r)) for r in df[list(columns)].itertuples(index=False))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def subset_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def save_splits(
    splits: pd.DataFrame, budgets: pd.DataFrame, cfg: DatasetConfig, out_dir: Path
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out_dir / "splits.csv.gz", index=False, compression=_GZ)
    budgets.to_csv(out_dir / "budgets.csv.gz", index=False, compression=_GZ)
    manifest = {
        "dataset": cfg.name,
        "config": cfg.model_dump(mode="json", exclude={"root"}),
        "split_hash": split_hash(splits),
        "split_counts": splits["split"].value_counts().sort_index().to_dict(),
        "per_split_hash": {
            s: split_hash(g) for s, g in sorted(splits.groupby("split"), key=lambda kv: kv[0])
        },
        "budget_hash": {
            f"{b}_seed{s}": subset_hash(g["id"])
            for (b, s), g in sorted(budgets.groupby(["budget", "seed"]))
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_splits(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(out_dir / "splits.csv.gz"), pd.read_csv(out_dir / "budgets.csv.gz")
