"""CLIP zero-shot classification from cached CLIP image embeddings and text prompts."""

from __future__ import annotations

import numpy as np
import torch

from labellite.train.knn import l2_normalize


@torch.inference_mode()
def text_embeddings(class_prompts: list[list[str]], model, tokenizer) -> np.ndarray:
    """(K, D) unit-norm class embeddings: each class's prompt embeddings are normalized,
    averaged, and re-normalized (prompt ensembling)."""
    out = []
    for prompts in class_prompts:
        e = model.encode_text(tokenizer(prompts)).float()
        e = e / e.norm(dim=-1, keepdim=True)
        out.append(e.mean(0))
    emb = torch.stack(out)
    return (emb / emb.norm(dim=-1, keepdim=True)).numpy()


def zero_shot_logits(image_emb: np.ndarray, text_emb: np.ndarray, logit_scale: float) -> np.ndarray:
    """CLIP logits: ``logit_scale * cos(image, class)``."""
    return logit_scale * l2_normalize(image_emb) @ text_emb.T


def build_prompts(templates: list[str], classnames: list[str]) -> list[list[str]]:
    return [[t.format(c) for t in templates] for c in classnames]
