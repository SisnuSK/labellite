import numpy as np
import pytest
import torch
from torch import nn

from labellite.data.datasets import EuroSAT
from labellite.features.cache import FeatureCache, StaleCacheError, make_key
from labellite.features.extract import extract_to_cache
from labellite.models.backbones import Backbone, make_transform, weights_hash
from labellite.train.knn import knn_accuracy


def tiny_backbone(seed: int = 0) -> Backbone:
    torch.manual_seed(seed)
    # BatchNorm adds a 0-d buffer (num_batches_tracked) that weights_hash must handle.
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, stride=2),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    transform, spec = make_transform((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), "bilinear", size=32)
    return Backbone("tiny", model.eval(), transform, spec, 8)


def run(backbone, adapter, ids, cache, **kw):
    return extract_to_cache(
        backbone,
        adapter,
        ids,
        "split-hash-A",
        cache,
        batch_size=16,
        num_workers=0,
        device="cpu",
        amp=False,
        log=lambda *_: None,
        **kw,
    )


@pytest.fixture
def setup(fake_eurosat, tmp_path):
    adapter = EuroSAT(fake_eurosat)
    ids = adapter.index()["id"].head(50)
    cache = FeatureCache(tmp_path / "cache" / "tiny" / "eurosat")
    return adapter, ids, cache


def key_for(backbone, split_hash="split-hash-A"):
    return make_key(
        "tiny", backbone.weights_hash(), backbone.transform_spec, "eurosat", split_hash, 8
    )


def test_build_then_hit_and_matches_direct_forward(setup):
    adapter, ids, cache = setup
    bb = tiny_backbone()
    assert run(bb, adapter, ids, cache) == "built"
    assert run(bb, adapter, ids, cache) == "hit"
    feats, cached_ids = cache.load(key_for(bb))
    assert feats.shape == (50, 8) and feats.dtype == np.float16
    assert list(cached_ids) == list(ids)
    with torch.no_grad():
        direct = bb.model(torch.stack([bb.transform(adapter.load_image(i)) for i in ids[:3]]))
    np.testing.assert_allclose(feats[:3], direct.numpy(), rtol=1e-2, atol=1e-3)
    assert cache.meta()["n_images"] == 50


def test_changed_weights_make_cache_stale(setup):
    adapter, ids, cache = setup
    old, new = tiny_backbone(0), tiny_backbone(1)
    assert weights_hash(old.model) != weights_hash(new.model)
    run(old, adapter, ids, cache)
    with pytest.raises(StaleCacheError, match="weights_hash"):
        cache.load(key_for(new))
    assert run(new, adapter, ids, cache) == "built"  # rebuilt, not reused
    cache.load(key_for(new))


def test_changed_split_or_transform_makes_cache_stale(setup):
    adapter, ids, cache = setup
    bb = tiny_backbone()
    run(bb, adapter, ids, cache)
    with pytest.raises(StaleCacheError):
        cache.load(key_for(bb, split_hash="split-hash-B"))
    with pytest.raises(StaleCacheError, match="different split table"):
        cache.load_for_split("split-hash-B")
    cache.load_for_split("split-hash-A")
    bb.transform, bb.transform_spec = make_transform((0.4,) * 3, (0.5,) * 3, "bilinear", size=32)
    assert cache.status(key_for(bb)) == "stale"


def test_tampered_meta_is_rejected(setup):
    adapter, ids, cache = setup
    run(tiny_backbone(), adapter, ids, cache)
    cache.meta_path.write_text(cache.meta_path.read_text().replace('"dim": 8', '"dim": 9'))
    with pytest.raises(StaleCacheError, match="inconsistent"):
        cache.load_for_split("split-hash-A")


class FlakyAdapter:
    """Wraps an adapter and fails once when a given image is first read."""

    def __init__(self, inner, fail_on):
        self.inner, self.fail_on, self.failed = inner, fail_on, False
        self.name = inner.name

    def load_image(self, image_id):
        if image_id == self.fail_on and not self.failed:
            self.failed = True
            raise OSError("simulated crash")
        return self.inner.load_image(image_id)


def test_interrupted_extraction_resumes_with_identical_features(setup, tmp_path):
    adapter, ids, cache = setup
    bb = tiny_backbone()
    clean = FeatureCache(tmp_path / "clean")
    run(bb, adapter, ids, clean)
    flaky = FlakyAdapter(adapter, fail_on=ids.iloc[40])
    with pytest.raises(OSError, match="simulated"):
        run(bb, flaky, ids, cache, commit_every=1)
    assert cache.status(key_for(bb)) == "missing"  # never marked complete
    assert run(bb, flaky, ids, cache, commit_every=1) == "resumed"
    assert cache.meta()["extraction"]["resumed_from_row"] == 32  # 2 committed batches of 16
    np.testing.assert_array_equal(cache.load(key_for(bb))[0], clean.load(key_for(bb))[0])


def test_knn_separates_clusters():
    rng = np.random.default_rng(0)
    centers = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    y = np.repeat([0, 1, 2], 50)
    x = centers[y] + 0.05 * rng.standard_normal((150, 3))
    assert knn_accuracy(x[::2], y[::2], x[1::2], y[1::2], k=5) == 1.0
