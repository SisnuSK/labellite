"""Frozen backbones used as feature extractors.

Every backbone is wrapped the same way: ``model(images) -> (B, D)`` features, plus the exact
preprocessing used and a hash of its weights. Images of every dataset (64 px EuroSAT, 96 px PCam)
are resized to the model's 224 px input without cropping, so no part of a patch is lost, and
normalized with that model's own mean/std.

| name          | weights                                   | feature                       |
|---------------|-------------------------------------------|-------------------------------|
| resnet50      | timm ``resnet50.tv2_in1k`` (ImageNet-1k)  | global-average-pooled, 2048-d |
| dinov2_vitb14 | timm ``vit_base_patch14_dinov2.lvd142m``  | CLS token, 768-d              |
| dinov2_vits14 | timm ``vit_small_patch14_dinov2.lvd142m`` | CLS token, 384-d              |
| clip_vitb16   | open_clip ``ViT-B-16`` / ``openai``       | projected image embedding, 512-d |
| phikon        | HF ``owkin/phikon`` (pinned revision)     | CLS token, 768-d              |
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn
from torchvision.transforms import v2 as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224
PHIKON_REPO = "owkin/phikon"
PHIKON_REVISION = "057cc0295895c2df3dd7681a89680da6015cbefe"


@dataclass
class Backbone:
    name: str
    model: nn.Module  # images (B, 3, H, W) -> features (B, D)
    transform: Callable
    transform_spec: dict
    dim: int
    source: dict = field(default_factory=dict)  # where the weights came from

    def weights_hash(self) -> str:
        return weights_hash(self.model)


def weights_hash(model: nn.Module) -> str:
    """SHA-256 over every parameter and buffer (name, dtype, shape, bytes), in name order."""
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        t = t.detach().cpu().contiguous()
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())  # reshape: 0-d tensors
    return h.hexdigest()


def make_transform(mean, std, interpolation: str, size: int = INPUT_SIZE) -> tuple[Callable, dict]:
    interp = {
        "bilinear": T.InterpolationMode.BILINEAR,
        "bicubic": T.InterpolationMode.BICUBIC,
    }[interpolation]
    transform = T.Compose(
        [
            T.ToImage(),
            T.Resize((size, size), interpolation=interp, antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=list(mean), std=list(std)),
        ]
    )
    spec = {
        "resize": [size, size],
        "interpolation": interpolation,
        "crop": None,
        "mean": [round(m, 8) for m in mean],
        "std": [round(s, 8) for s in std],
    }
    return transform, spec


class _ClipImageEncoder(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip = clip_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_image(x)


class _HFClsToken(nn.Module):
    def __init__(self, vit: nn.Module):
        super().__init__()
        self.vit = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=x).last_hidden_state[:, 0]


def _timm(name: str, timm_id: str, **kwargs) -> Backbone:
    import timm

    model = timm.create_model(timm_id, pretrained=True, num_classes=0, **kwargs)
    cfg = model.pretrained_cfg
    transform, spec = make_transform(cfg["mean"], cfg["std"], cfg["interpolation"])
    return Backbone(name, model, transform, spec, model.num_features, {"timm": timm_id})


def load_clip(pretrained: str = "openai"):
    """The full open_clip model (image + text towers) and its tokenizer, for zero-shot use."""
    import open_clip

    # OpenAI's CLIP weights were trained with QuickGELU; the plain "ViT-B-16" config uses GELU.
    model = open_clip.create_model(
        "ViT-B-16", pretrained=pretrained, force_quick_gelu=pretrained == "openai"
    )
    return model, open_clip.get_tokenizer("ViT-B-16")


def _clip() -> Backbone:
    import open_clip

    model, _ = load_clip()
    transform, spec = make_transform(
        open_clip.OPENAI_DATASET_MEAN, open_clip.OPENAI_DATASET_STD, "bicubic"
    )
    dim = model.visual.output_dim
    return Backbone(
        "clip_vitb16",
        _ClipImageEncoder(model),
        transform,
        spec,
        dim,
        {"open_clip": "ViT-B-16/openai"},
    )


def _phikon() -> Backbone:
    from transformers import ViTModel

    vit = ViTModel.from_pretrained(PHIKON_REPO, revision=PHIKON_REVISION, add_pooling_layer=False)
    # Mean/std and 224 px input are from the repo's preprocessor_config.json; HF's ViT image
    # processor resizes bilinearly by default.
    transform, spec = make_transform(IMAGENET_MEAN, IMAGENET_STD, "bilinear")
    source = {"hf": PHIKON_REPO, "revision": PHIKON_REVISION}
    return Backbone("phikon", _HFClsToken(vit), transform, spec, vit.config.hidden_size, source)


LOADERS: dict[str, Callable[[], Backbone]] = {
    "resnet50": lambda: _timm("resnet50", "resnet50.tv2_in1k"),
    "dinov2_vitb14": lambda: _timm(
        "dinov2_vitb14", "vit_base_patch14_dinov2.lvd142m", img_size=INPUT_SIZE
    ),
    "dinov2_vits14": lambda: _timm(
        "dinov2_vits14", "vit_small_patch14_dinov2.lvd142m", img_size=INPUT_SIZE
    ),
    "clip_vitb16": _clip,
    "phikon": _phikon,
}


def load_backbone(name: str) -> Backbone:
    backbone = LOADERS[name]()
    backbone.model.eval().requires_grad_(False)
    return backbone
