import numpy as np
import timm
import torch

from labellite.config import FinetuneConfig
from labellite.data.datasets import EuroSAT, PCam
from labellite.train import finetune as ft


def tiny_model(num_classes):
    return timm.create_model("resnet18", pretrained=False, num_classes=num_classes)


def test_finetune_runs_resumes_and_keeps_best(fake_eurosat, tmp_path, monkeypatch):
    monkeypatch.setattr(ft, "build_model", tiny_model)
    cfg = FinetuneConfig(
        dataset="eurosat",
        epochs={1.0: 2},
        batch_size=16,
        input_size=32,
        evals_per_run=2,
        device="cpu",
        mixed_precision=False,
        color_jitter=True,
    )
    adapter = EuroSAT(fake_eurosat)
    idx = adapter.index()
    tr, va = idx.iloc[::2], idx.iloc[1::2].head(40)
    tx, vx = (torch.from_numpy(adapter.load_array(d["id"])) for d in (tr, va))
    assert tx.shape == (100, 64, 64, 3) and tx.dtype == torch.uint8
    args = (cfg, tx, tr["label"].to_numpy(), vx, va["label"].to_numpy(), 10)
    kw = dict(seed=0, budget=1.0, ckpt_path=tmp_path / "run.pt", log=lambda *_: None)
    model, prep, hist = ft.finetune(*args, **kw)
    assert hist["total_steps"] == 2 * (100 // 16) and hist["finished"]
    assert len(hist["history"]) >= 2 and (tmp_path / "run.pt").exists()
    assert hist["best_acc"] == max(h["val_accuracy"] for h in hist["history"])
    # A finished checkpoint is picked up and not retrained.
    _, _, again = ft.finetune(*args, **kw)
    assert again["step"] == hist["step"] and again["history"] == hist["history"]
    logits = ft.predict_logits(model, vx, prep, 16, amp=False)
    assert logits.shape == (40, 10) and np.isfinite(logits).all()


def test_augment_preserves_shape_range_and_is_seeded():
    x = torch.rand(8, 3, 12, 12)
    a = ft.augment(x, torch.Generator().manual_seed(0), color_jitter=True)
    b = ft.augment(x, torch.Generator().manual_seed(0), color_jitter=True)
    assert a.shape == x.shape and torch.equal(a, b) and a.min() >= 0 and a.max() <= 1
    plain = ft.augment(x, torch.Generator().manual_seed(1), color_jitter=False)
    # Without jitter each image is a flip/rotation of the original: same multiset of pixels.
    for i in range(8):
        assert torch.allclose(plain[i].flatten().sort().values, x[i].flatten().sort().values)


def test_pcam_load_array_matches_load_image(fake_pcam):
    ds = PCam(fake_pcam)
    ids = ["train/5", "valid/2", "train/1", "train/5"]
    arr = ds.load_array(ids)
    for i, image_id in enumerate(ids):
        np.testing.assert_array_equal(arr[i], np.asarray(ds.load_image(image_id)))


def test_lr_schedule_warms_up_then_decays():
    f = ft.lr_lambda(total_steps=100, warmup_steps=10)
    assert f(0) == 0.1 and f(9) == 1.0
    assert f(10) == 1.0 and abs(f(55) - 0.5) < 1e-9 and f(100) == 0.0
