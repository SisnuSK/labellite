import numpy as np
import pandas as pd
import pytest

from labellite.config import DatasetConfig
from labellite.data.datasets import EuroSAT, ImageDataset, PCam
from labellite.data.download import md5sum
from labellite.data.splits import (
    allocate,
    build_splits,
    label_budgets,
    load_splits,
    save_splits,
    split_hash,
)


def cfg(name, fractions, **kw):
    return DatasetConfig(name=name, root=".", split_fractions=fractions, **kw)


EUROSAT_FRACS = {"train": 0.6, "val": 0.1, "cal": 0.1, "test": 0.2}


def test_allocate_sums_and_rounds():
    assert allocate(10, {"a": 0.5, "b": 0.5}) == {"a": 5, "b": 5}
    out = allocate(7, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
    assert sum(out.values()) == 7
    assert sorted(out.values()) == [2, 2, 3]


def test_eurosat_index_and_image(fake_eurosat):
    ds = EuroSAT(fake_eurosat)
    idx = ds.index()
    assert len(idx) == 200
    assert idx["label"].value_counts().eq(20).all()
    assert np.asarray(ds.load_image(idx["id"][0])).shape == (64, 64, 3)


def test_eurosat_splits_are_stratified_disjoint_and_complete(fake_eurosat):
    idx = EuroSAT(fake_eurosat).index()
    s = build_splits(idx, cfg("eurosat", EUROSAT_FRACS))
    assert s["id"].is_unique and set(s["id"]) == set(idx["id"])
    per_class = pd.crosstab(s["label"], s["split"])
    # 20 images per class -> exactly 12 / 2 / 2 / 4
    assert (per_class["train"] == 12).all() and (per_class["test"] == 4).all()
    assert (per_class["val"] == 2).all() and (per_class["cal"] == 2).all()


def test_splits_are_deterministic_and_seed_dependent(fake_eurosat):
    idx = EuroSAT(fake_eurosat).index()
    a = build_splits(idx, cfg("eurosat", EUROSAT_FRACS, split_seed=0))
    b = build_splits(idx, cfg("eurosat", EUROSAT_FRACS, split_seed=0))
    c = build_splits(idx, cfg("eurosat", EUROSAT_FRACS, split_seed=1))
    assert split_hash(a) == split_hash(b)
    assert split_hash(a) != split_hash(c)


def test_split_hash_is_order_independent_and_label_sensitive(fake_eurosat):
    s = build_splits(EuroSAT(fake_eurosat).index(), cfg("eurosat", EUROSAT_FRACS))
    assert split_hash(s) == split_hash(s.sample(frac=1, random_state=0))
    flipped = s.copy()
    flipped.loc[0, "label"] = (flipped.loc[0, "label"] + 1) % 10
    assert split_hash(s) != split_hash(flipped)


def test_pcam_keeps_official_train_test_and_divides_valid(fake_pcam):
    ds = PCam(fake_pcam)
    idx = ds.index()
    s = build_splits(idx, cfg("pcam", {"val": 0.5, "cal": 0.5}))
    counts = s["split"].value_counts().to_dict()
    assert counts == {"train": 400, "test": 100, "val": 50, "cal": 50}
    assert s.loc[s["split"] == "test", "id"].str.startswith("test/").all()
    assert s.loc[s["split"].isin(["val", "cal"]), "id"].str.startswith("valid/").all()
    assert (pd.crosstab(s["split"], s["label"])[0] == pd.crosstab(s["split"], s["label"])[1]).all()
    assert np.asarray(ds.load_image("valid/3")).shape == (96, 96, 3)


def test_eval_subset_is_stratified_and_fixed_size(fake_pcam):
    s = build_splits(
        PCam(fake_pcam).index(), cfg("pcam", {"val": 0.5, "cal": 0.5}, eval_subset_size=20)
    )
    sub = s[s["eval_subset"]]
    assert set(sub["split"]) == {"test"} and len(sub) == 20
    assert sub["label"].value_counts().eq(10).all()


def test_label_budgets_nested_stratified_and_seeded(fake_eurosat):
    s = build_splits(EuroSAT(fake_eurosat).index(), cfg("eurosat", EUROSAT_FRACS))
    train = s[s["split"] == "train"]
    b = label_budgets(train, [0.1, 0.5, 1.0], seeds=[0, 1])
    assert set(b["budget"]) == {0.1, 0.5}  # 1.0 means the full train split
    assert set(b["id"]) <= set(train["id"])
    for seed in (0, 1):
        small = set(b[(b["budget"] == 0.1) & (b["seed"] == seed)]["id"])
        big = set(b[(b["budget"] == 0.5) & (b["seed"] == seed)]["id"])
        assert small < big
    labels = b.merge(train[["id", "label"]], on="id")
    per_class = labels[(labels["budget"] == 0.5) & (labels["seed"] == 0)]["label"].value_counts()
    assert per_class.eq(6).all() and len(per_class) == 10  # round(0.5 * 12)
    s0 = set(b[(b["budget"] == 0.1) & (b["seed"] == 0)]["id"])
    s1 = set(b[(b["budget"] == 0.1) & (b["seed"] == 1)]["id"])
    assert s0 != s1


def test_budget_keeps_at_least_one_per_class(fake_eurosat):
    s = build_splits(EuroSAT(fake_eurosat).index(), cfg("eurosat", EUROSAT_FRACS))
    b = label_budgets(s[s["split"] == "train"], [0.01], seeds=[0])
    assert len(b) == 10


def test_save_load_roundtrip(fake_eurosat, tmp_path):
    c = cfg("eurosat", EUROSAT_FRACS)
    s = build_splits(EuroSAT(fake_eurosat).index(), c)
    b = label_budgets(s[s["split"] == "train"], [0.1], [0])
    out = tmp_path / "splits"
    m = save_splits(s, b, c, out)
    s2, b2 = load_splits(out)
    assert split_hash(s2) == m["split_hash"]
    assert len(b2) == len(b)
    before = md5sum(out / "splits.csv.gz")
    save_splits(s, b, c, out)
    assert md5sum(out / "splits.csv.gz") == before  # byte-reproducible


def test_image_dataset(fake_eurosat):
    ds = EuroSAT(fake_eurosat)
    idx = ds.index().head(5)
    torch_ds = ImageDataset(ds, idx["id"], idx["label"], transform=lambda im: np.asarray(im))
    img, label, i = torch_ds[2]
    assert img.shape == (64, 64, 3) and label == idx["label"][2] and i == 2


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        cfg("eurosat", {"train": 0.6, "test": 0.3})
