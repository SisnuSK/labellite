"""Dataset adapters.

Each adapter exposes the same two things:

* ``index()``: a DataFrame with one row per image and columns ``id`` (stable string ID),
  ``label`` (int), and ``source_split`` (the publisher's split, or ``"all"`` if there is none).
* ``load_image(id)``: the image as an RGB ``PIL.Image``.

``ImageDataset`` wraps an adapter plus a list of IDs as a torch ``Dataset``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class DatasetAdapter(ABC):
    name: ClassVar[str]
    class_names: ClassVar[tuple[str, ...]]

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @abstractmethod
    def index(self) -> pd.DataFrame: ...

    @abstractmethod
    def load_image(self, image_id: str) -> Image.Image: ...

    def load_array(self, ids) -> np.ndarray:
        """(N, H, W, 3) uint8 array of the given images, in order (for in-memory training)."""
        return np.stack([np.asarray(self.load_image(i)) for i in ids])

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


class EuroSAT(DatasetAdapter):
    """EuroSAT RGB: ``<root>/EuroSAT_RGB/<ClassName>/<ClassName>_<n>.jpg``. No official split."""

    name = "eurosat"
    class_names = (
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake",
    )

    @property
    def image_dir(self) -> Path:
        return self.root / "EuroSAT_RGB"

    def index(self) -> pd.DataFrame:
        rows = []
        for label, cls in enumerate(self.class_names):
            for p in sorted((self.image_dir / cls).glob("*.jpg")):
                rows.append((f"{cls}/{p.name}", label))
        if not rows:
            raise FileNotFoundError(f"no EuroSAT images under {self.image_dir}")
        df = pd.DataFrame(rows, columns=["id", "label"])
        df["source_split"] = "all"
        return df

    def load_image(self, image_id: str) -> Image.Image:
        with Image.open(self.image_dir / image_id) as im:
            return im.convert("RGB")


class PCam(DatasetAdapter):
    """PatchCamelyon: decompressed HDF5 files ``camelyonpatch_level_2_split_<split>_{x,y}.h5``.

    IDs are ``<split>/<row>``. ``x`` holds (N, 96, 96, 3) uint8 images, ``y`` (N, 1, 1, 1) labels.
    """

    name = "pcam"
    class_names = ("normal", "tumor")
    splits = ("train", "valid", "test")

    def __init__(self, root: str | Path):
        super().__init__(root)
        self._handles: dict[str, object] = {}

    def _path(self, split: str, kind: str) -> Path:
        return self.root / f"camelyonpatch_level_2_split_{split}_{kind}.h5"

    def index(self) -> pd.DataFrame:
        import h5py

        frames = []
        for split in self.splits:
            with h5py.File(self._path(split, "y"), "r") as f:
                y = np.asarray(f["y"]).reshape(-1).astype(int)
            frames.append(
                pd.DataFrame(
                    {
                        "id": [f"{split}/{i}" for i in range(len(y))],
                        "label": y,
                        "source_split": split,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def load_image(self, image_id: str) -> Image.Image:
        import h5py

        split, row = image_id.split("/")
        # One open handle per split per process (DataLoader workers each get their own copy).
        if split not in self._handles:
            self._handles[split] = h5py.File(self._path(split, "x"), "r")
        return Image.fromarray(self._handles[split]["x"][int(row)]).convert("RGB")

    def load_array(self, ids) -> np.ndarray:
        import h5py

        parts = pd.Series(list(ids)).str.split("/", expand=True)
        out = None
        for split, group in parts.groupby(0):
            rows = group[1].astype(int).to_numpy()
            with h5py.File(self._path(split, "x"), "r") as f:
                x = f["x"]
                if len(rows) > len(x) // 4:  # mostly-full read: one sequential pass
                    block = x[:][rows]
                else:  # h5py fancy indexing needs increasing, unique indices
                    uniq, inverse = np.unique(rows, return_inverse=True)
                    block = x[uniq][inverse]
            if out is None:
                out = np.empty((len(parts), *block.shape[1:]), dtype=block.dtype)
            out[group.index.to_numpy()] = block
        return out

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state


ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "eurosat": EuroSAT,
    "pcam": PCam,
}


def get_adapter(name: str, root: str | Path) -> DatasetAdapter:
    return ADAPTERS[name](root)


class ImageDataset(Dataset):
    """Torch dataset over a subset of an adapter's images. Yields ``(image, label, index)``."""

    def __init__(self, adapter: DatasetAdapter, ids, labels, transform=None):
        self.adapter = adapter
        self.ids = list(ids)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        img = self.adapter.load_image(self.ids[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i]), i
