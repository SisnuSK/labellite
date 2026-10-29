"""Dataset summary: sizes, image shapes, and class distribution per split."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from labellite.data.datasets import DatasetAdapter
from labellite.data.splits import EVAL_SPLITS


def image_shapes(adapter: DatasetAdapter, ids, n: int = 20, seed: int = 0) -> Counter:
    """(height, width, channels) of ``n`` randomly chosen images."""
    ids = list(ids)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ids), size=min(n, len(ids)), replace=False)
    return Counter(np.asarray(adapter.load_image(ids[i])).shape for i in pick)


def summarize(adapter: DatasetAdapter, splits: pd.DataFrame, budgets: pd.DataFrame) -> str:
    names = adapter.class_names
    lines = [f"== {adapter.name}: {len(splits):,} images, {len(names)} classes =="]
    table = pd.crosstab(splits["split"], splits["label"].map(dict(enumerate(names))))
    table["total"] = table.sum(axis=1)
    table.loc["all"] = table.sum()
    lines += ["", "Class counts per split:", table.to_string()]
    if splits["eval_subset"].ne(splits["split"].isin(EVAL_SPLITS)).any():
        sub = splits[splits["eval_subset"]].groupby("split").size()
        lines += ["", "Fixed evaluation subsets:", sub.to_string()]
    lines += ["", "Image shapes (H, W, C) of 20 random images per split:"]
    for split, g in sorted(splits.groupby("split"), key=lambda kv: kv[0]):
        shapes = ", ".join(f"{s} x{c}" for s, c in image_shapes(adapter, g["id"]).items())
        lines.append(f"  {split:<9} {shapes}")
    if len(budgets):
        labeled = budgets.merge(splits[["id", "label"]], on="id")
        counts = pd.crosstab(
            [labeled["budget"], labeled["seed"]], labeled["label"].map(dict(enumerate(names)))
        )
        counts["total"] = counts.sum(axis=1)
        lines += ["", "Label-budget subsets (train images per class):", counts.to_string()]
    return "\n".join(lines)
