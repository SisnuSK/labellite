"""Tiny on-disk fakes of each dataset's layout, for loader and split tests."""

import h5py
import numpy as np
import pytest
from PIL import Image

from labellite.data.datasets import EuroSAT


@pytest.fixture
def fake_eurosat(tmp_path):
    rng = np.random.default_rng(0)
    for cls in EuroSAT.class_names:
        d = tmp_path / "EuroSAT_RGB" / cls
        d.mkdir(parents=True)
        for i in range(20):
            arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{cls}_{i + 1}.jpg")
    return tmp_path


@pytest.fixture
def fake_pcam(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 400), ("valid", 100), ("test", 100)):
        y = np.repeat([0, 1], n // 2).astype(np.uint8).reshape(-1, 1, 1, 1)
        x = rng.integers(0, 255, (n, 96, 96, 3), dtype=np.uint8)
        with h5py.File(tmp_path / f"camelyonpatch_level_2_split_{split}_y.h5", "w") as f:
            f["y"] = y
        with h5py.File(tmp_path / f"camelyonpatch_level_2_split_{split}_x.h5", "w") as f:
            f["x"] = x
    return tmp_path
