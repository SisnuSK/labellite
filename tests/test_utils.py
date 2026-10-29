import json
import random

import numpy as np

from labellite.utils.manifest import build_manifest, write_manifest
from labellite.utils.seed import seed_everything


def test_seed_everything_is_reproducible():
    seed_everything(123)
    a = (random.random(), np.random.rand(3).tolist())
    seed_everything(123)
    b = (random.random(), np.random.rand(3).tolist())
    assert a == b


def test_manifest_roundtrip(tmp_path):
    m = build_manifest("extract", {"model": "dinov2_vitb14"}, split_hash="abc")
    assert {"git", "hardware", "packages", "created_utc"} <= m.keys()
    assert m["split_hash"] == "abc"
    path = write_manifest(m, tmp_path / "run" / "manifest.json")
    assert json.loads(path.read_text())["config"] == {"model": "dinov2_vitb14"}
