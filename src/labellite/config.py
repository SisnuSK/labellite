"""YAML + pydantic config schemas.

Every CLI command takes one YAML file and validates it against a schema here. Schemas use
``extra="forbid"`` so a typo in a config key fails loudly instead of being ignored. Later
phases extend these schemas as stages are implemented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DatasetName = Literal["pcam", "eurosat"]
BackboneName = Literal["resnet50", "dinov2_vitb14", "dinov2_vits14", "clip_vitb16", "phikon"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(StrictModel):
    """How a dataset's fixed splits and label-budget subsets are built (see data/splits.py)."""

    name: DatasetName
    root: Path
    splits_dir: Path = Path("data/splits")
    split_seed: int = 0
    # Fractions used to divide the part of the data with no official split: all of EuroSAT, or
    # PCam's official valid set.
    split_fractions: dict[str, float]
    label_budgets: list[float] = [0.01, 0.1, 1.0]
    budget_seeds: list[int] = [0, 1, 2]
    # If set, each evaluation split larger than this gets a fixed stratified subset.
    eval_subset_size: int | None = Field(None, gt=0)

    @field_validator("split_fractions")
    @classmethod
    def _fractions_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        if abs(sum(v.values()) - 1.0) > 1e-9 or any(f <= 0 for f in v.values()):
            raise ValueError(f"split_fractions must be positive and sum to 1, got {v}")
        return v

    @property
    def out_dir(self) -> Path:
        return self.splits_dir / self.name


class ExtractConfig(StrictModel):
    model: BackboneName
    dataset: DatasetName
    datasets_config_dir: Path = Path("configs/datasets")
    batch_size: int = Field(128, gt=0)
    num_workers: int = Field(4, ge=0)
    device: str = "cuda"
    amp: bool = True  # fp16 autocast on CUDA
    cache_dir: Path = Path("data/cache/features")
    seed: int = 0

    @property
    def dataset_config(self) -> Path:
        return self.datasets_config_dir / f"{self.dataset}.yaml"


class FinetuneConfig(StrictModel):
    model: Literal["resnet50"] = "resnet50"
    dataset: DatasetName
    datasets_config_dir: Path = Path("configs/datasets")
    label_budgets: list[float] = [0.01, 0.1, 1.0]
    seeds: list[int] = [0, 1, 2]
    # Small budgets need more passes over their few images; keys are label budgets.
    epochs: dict[float, int] = {0.01: 60, 0.1: 30, 1.0: 10}
    evals_per_run: int = Field(
        15, gt=0
    )  # val evaluations (for early stopping) spread over training
    early_stopping_patience: int = Field(4, gt=0)  # in evaluations
    batch_size: int = Field(128, gt=0)
    lr: float = Field(1e-4, gt=0)
    weight_decay: float = Field(0.05, ge=0)
    warmup_fraction: float = Field(0.05, ge=0, lt=1)
    input_size: int = Field(224, gt=0)
    color_jitter: bool = False  # mild stain/colour jitter, for pathology
    mixed_precision: bool = True  # bf16 autocast
    deterministic: bool = False  # deterministic cuDNN kernels are much slower for training
    device: str = "cuda"
    checkpoint_dir: Path = Path("checkpoints/finetune")
    results_dir: Path = Path("results")

    @property
    def dataset_config(self) -> Path:
        return self.datasets_config_dir / f"{self.dataset}.yaml"


class MainConfig(StrictModel):
    datasets: list[DatasetName]
    models: list[BackboneName]
    label_budgets: list[float] = [0.01, 0.1, 1.0]
    seeds: list[int] = [0, 1, 2]
    knn_k: int = Field(20, gt=0)
    c_grid: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    bootstrap_resamples: int = Field(1000, gt=0)
    conformal_alphas: list[float] = [0.10, 0.05]
    conformal_methods: list[Literal["lac", "aps"]] = ["lac", "aps"]
    ece_bins: int = Field(15, gt=0)
    # The cal split is divided (stratified) into a temperature-fitting part and a conformal
    # calibration part, so the conformal score function does not depend on its own calibration
    # data.
    cal_ts_fraction: float = Field(0.5, gt=0, lt=1)
    cal_split_seed: int = 0
    figure_budget: float = 0.1  # label budget shown in the reliability / coverage figures
    cache_dir: Path = Path("data/cache/features")
    results_dir: Path = Path("results")
    datasets_config_dir: Path = Path("configs/datasets")
    zeroshot_prompts: Path = Path("configs/zeroshot.yaml")


class ShiftConfig(StrictModel):
    """Synthetic stain shift of the PCam test split (RQ4); see data/shift.py."""

    dataset: Literal["pcam"] = "pcam"
    severities: list[float] = [1, 2, 3]
    hematoxylin_step: float = Field(0.15, gt=0)
    eosin_step: float = Field(0.25, gt=0)
    models: list[BackboneName]
    batch_size: int = Field(256, gt=0)
    num_workers: int = Field(4, ge=0)
    device: str = "cuda"
    amp: bool = True
    main_config: Path = Path("configs/main.yaml")
    finetune_config: Path = Path("configs/finetune/resnet50_pcam.yaml")

    @model_validator(mode="after")
    def _hematoxylin_stays_positive(self) -> ShiftConfig:
        if any(1 - self.hematoxylin_step * s <= 0 for s in self.severities):
            raise ValueError("hematoxylin_step * severity must stay below 1")
        return self


class ALConfig(StrictModel):
    datasets: list[DatasetName]
    models: list[BackboneName]
    strategies: list[
        Literal["random", "least_confidence", "entropy", "conformal_set_size", "kcenter"]
    ]
    initial_fraction: float = Field(0.01, gt=0, le=1)
    query_fraction: float = Field(0.01, gt=0, le=1)
    rounds: int = Field(10, gt=0)
    seeds: list[int] = [0, 1, 2]
    conformal_alpha: float = Field(0.10, gt=0, lt=1)
    n_jobs: int = Field(6, gt=0)  # parallel (strategy, seed) runs
    # BLAS threads per run, identical for every run so results do not depend on how runs were
    # scheduled (4 = 24 logical CPUs / 6 jobs, joblib's own default for the worker processes).
    blas_threads: int = Field(4, gt=0)
    # Feature cache, C grid, dataset configs and the tables/figures directory come from here.
    main_config: Path = Path("configs/main.yaml")
    results_dir: Path = Path("results/active")


def load_config[T: BaseModel](path: str | Path, schema: type[T]) -> T:
    """Load a YAML file and validate it against ``schema``."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return schema.model_validate(raw)
