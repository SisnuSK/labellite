"""Synthetic stain shift for PCam (RQ4; replaces the dropped Camelyon17-WILDS cross-hospital test).

H&E images are converted to stain space with the Ruifrok & Johnston colour-deconvolution
matrices (``skimage.color.hed_from_rgb`` / ``rgb_from_hed``), and at severity ``s`` the
hematoxylin concentration is scaled by ``1 - h_step * s`` and eosin by ``1 + e_step * s``: a
systematic shift towards weaker nuclear and stronger eosin staining, applied identically to every
test image. It is a synthetic perturbation, not a model of any real site.

The transform works on optical density without clipping the stain concentrations
(``skimage.color.rgb2hed`` clips them, which changes images by up to ~20/255 even with no shift),
so severity 0 returns the input exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import hed_from_rgb, rgb_from_hed

from labellite.data.datasets import DatasetAdapter


def stain_shift(img: np.ndarray, severity: float, h_step: float, e_step: float) -> np.ndarray:
    """Shift uint8 RGB images (``(..., 3)``) in H&E stain space; returns uint8."""
    if severity == 0:
        return img
    od = -np.log((img.astype(np.float64) + 1) / 256)
    stains = od @ hed_from_rgb
    stains[..., 0] *= 1 - h_step * severity
    stains[..., 1] *= 1 + e_step * severity
    out = np.exp(-(stains @ rgb_from_hed)) * 256 - 1
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


class StainShifted(DatasetAdapter):
    """Wraps an adapter so every image comes back stain-shifted. The adapter ``name`` encodes the
    shift parameters, so features cached for one setting are never reused for another."""

    def __init__(self, base: DatasetAdapter, severity: float, h_step: float, e_step: float):
        super().__init__(base.root)
        self.base, self.severity, self.h_step, self.e_step = base, severity, h_step, e_step
        self.name = f"{base.name}_stain_s{severity:g}_h{h_step:g}_e{e_step:g}"
        self.class_names = base.class_names

    def index(self) -> pd.DataFrame:
        return self.base.index()

    def _shift(self, x: np.ndarray) -> np.ndarray:
        return stain_shift(x, self.severity, self.h_step, self.e_step)

    def load_image(self, image_id: str) -> Image.Image:
        return Image.fromarray(self._shift(np.asarray(self.base.load_image(image_id))))

    def load_array(self, ids) -> np.ndarray:
        return self._shift(self.base.load_array(ids))
