"""Grad-CAM (Selvaraju et al., 2017) for the CNN and the ViTs, as a qualitative aid only.

For a target layer with activations ``A`` and the gradient ``G`` of the explained class's
log-probability (``log softmax``; for two classes, its log-odds) with respect to ``A``, the
map is ``relu(sum_c mean_spatial(G_c) * A_c)``, upsampled to the image and scaled to [0, 1]
per image.

* ResNet-50: the output of ``layer4`` (7 x 7 at 224 px).
* ViTs (DINOv2, Phikon, CLIP): the patch tokens entering the last transformer block (the output
  of its first LayerNorm), reshaped to the patch grid. The tokens *leaving* the last block cannot
  be used: the pooled CLS feature does not depend on them, so their gradient is zero.

The linear probes become a differentiable head on the frozen backbone
(``logits = W normalize(f) + b``), so every map explains the same prediction that was scored.
A heatmap shows where the gradient-weighted activations are large; it is not proof of what a
model "looks at".
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Autocast(nn.Module):
    """Runs a model under bf16 autocast and returns float32 logits. The fine-tuned ResNet-50 was
    trained and evaluated in bf16; in fp32 its probabilities differ by up to 0.75 on some
    EuroSAT images, so its maps must be computed in the precision its predictions came from."""

    def __init__(self, model: nn.Module, enabled: bool = True):
        super().__init__()
        self.model, self.enabled = model, enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=self.enabled):
            return self.model(x).float()


class ProbeHead(nn.Module):
    """Frozen backbone + the fitted logistic-regression probe (on L2-normalized features)."""

    def __init__(self, backbone: nn.Module, coef: np.ndarray, intercept: np.ndarray):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("w", torch.as_tensor(coef, dtype=torch.float32))
        self.register_buffer("b", torch.as_tensor(intercept, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = F.normalize(self.backbone(x).float(), dim=1)
        z = f @ self.w.T + self.b
        if z.shape[1] == 1:  # binary sklearn model: logits [0, z], as in train.probe.logits
            z = torch.cat([torch.zeros_like(z), z], dim=1)
        return z


def target_layer(model_name: str, backbone: nn.Module) -> tuple[nn.Module, int]:
    """(layer whose output is explained, number of prefix tokens to drop; 0 for a CNN)."""
    if model_name == "resnet50":
        return backbone.layer4, 0
    if model_name.startswith("dinov2"):
        return backbone.blocks[-1].norm1, backbone.num_prefix_tokens
    if model_name == "phikon":
        vit = backbone.vit  # transformers >= 5 keeps the blocks in ``layers``
        blocks = vit.layers if hasattr(vit, "layers") else vit.encoder.layer
        return blocks[-1].layernorm_before, 1
    if model_name == "clip_vitb16":
        return backbone.clip.visual.transformer.resblocks[-1].ln_1, 1
    raise ValueError(f"no Grad-CAM target layer for {model_name!r}")


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module, prefix_tokens: int = 0):
        self.model, self.prefix = model.eval(), prefix_tokens
        self._acts = None
        layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self._acts = output

    def _to_maps(self, acts: torch.Tensor, batch: int) -> torch.Tensor:
        """Activations as (B, C, h, w)."""
        if acts.ndim == 4:
            return acts
        if acts.shape[0] != batch:  # sequence-first transformer layout (N, B, C)
            acts = acts.transpose(0, 1)
        tokens = acts[:, self.prefix :]
        side = int(round(tokens.shape[1] ** 0.5))
        if side * side != tokens.shape[1]:
            raise ValueError(f"{tokens.shape[1]} patch tokens do not form a square grid")
        return tokens.transpose(1, 2).reshape(batch, -1, side, side)

    def __call__(
        self, x: torch.Tensor, classes: torch.Tensor | None = None, out_size: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Maps (B, S, S) in [0, 1] and the logits (B, K). ``classes`` defaults to the
        predicted class of each image."""
        x = x.detach().requires_grad_(True)  # frozen weights: grads must flow from the input
        logits = self.model(x)
        if classes is None:
            classes = logits.argmax(1)
        acts = self._acts
        # Explain the log-probability, not the raw logit: binary probes output [0, z], whose
        # class-0 logit is constant (zero gradient), and logits are only defined up to a shift.
        score = torch.log_softmax(logits.float(), 1).gather(1, classes.view(-1, 1)).sum()
        (grads,) = torch.autograd.grad(score, acts)
        a, g = self._to_maps(acts.float(), len(x)), self._to_maps(grads.float(), len(x))
        cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
        size = out_size or x.shape[-1]
        cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)[:, 0]
        peak = cam.flatten(1).max(1).values.clamp_min(1e-12).view(-1, 1, 1)
        return (cam / peak).detach().cpu().numpy(), logits.detach().cpu().numpy()
