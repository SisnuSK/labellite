from pathlib import Path

import pytest
from pydantic import ValidationError

from labellite.config import (
    ALConfig,
    DatasetConfig,
    ExtractConfig,
    FinetuneConfig,
    MainConfig,
    ShiftConfig,
    load_config,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    ("relpath", "schema"),
    [
        ("main.yaml", MainConfig),
        ("al.yaml", ALConfig),
        ("extract/dinov2_pcam.yaml", ExtractConfig),
        ("finetune/resnet50_pcam.yaml", FinetuneConfig),
        ("datasets/eurosat.yaml", DatasetConfig),
        ("datasets/pcam.yaml", DatasetConfig),
        ("shift.yaml", ShiftConfig),
    ],
)
def test_shipped_configs_validate(relpath, schema):
    cfg = load_config(CONFIGS / relpath, schema)
    assert isinstance(cfg, schema)


def test_unknown_key_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("model: dinov2_vitb14\ndataset: pcam\nbatch_sizee: 64\n")
    with pytest.raises(ValidationError, match="batch_sizee"):
        load_config(p, ExtractConfig)


def test_unknown_dataset_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("model: dinov2_vitb14\ndataset: imagenet\n")
    with pytest.raises(ValidationError):
        load_config(p, ExtractConfig)


def test_shift_rejects_hematoxylin_below_zero():
    with pytest.raises(ValidationError, match="hematoxylin_step"):
        ShiftConfig(models=["phikon"], severities=[1, 2, 4], hematoxylin_step=0.25)
