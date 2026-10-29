"""Download sources for each dataset, as verified against the publishers (see data/DATASETS.md).

MD5s are the ones published by the Zenodo records.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RemoteFile:
    url: str
    filename: str
    size: int | None
    md5: str | None


_ZENODO_PCAM = "https://zenodo.org/records/2546921/files/{}?download=1"
_PCAM_FILES = {
    "camelyonpatch_level_2_split_train_x.h5.gz": (
        6_421_353_462,
        "1571f514728f59376b705fc836ff4b63",
    ),
    "camelyonpatch_level_2_split_train_y.h5.gz": (21_378, "35c2d7259d906cfc8143347bb8e05be7"),
    "camelyonpatch_level_2_split_valid_x.h5.gz": (805_965_320, "d5b63470df7cfa627aeec8b9dc0c066e"),
    "camelyonpatch_level_2_split_valid_y.h5.gz": (3_038, "2b85f58b927af9964a4c15b8f7e8f179"),
    "camelyonpatch_level_2_split_test_x.h5.gz": (800_875_929, "d8c2d60d490dbd479f8199bdfa0cf6ec"),
    "camelyonpatch_level_2_split_test_y.h5.gz": (3_040, "60a7035772fbdb7f34eb86d4420cf66a"),
    "camelyonpatch_level_2_split_train_meta.csv": (15_047_830, "5a3dd671e465cfd74b5b822125e65b0a"),
    "camelyonpatch_level_2_split_valid_meta.csv": (1_850_810, "67589e00a4a37ec317f2d1932c7502ca"),
    "camelyonpatch_level_2_split_test_meta.csv": (1_608_584, "3455fd69135b66734e1008f3af684566"),
}

SOURCES: dict[str, list[RemoteFile]] = {
    "pcam": [
        RemoteFile(_ZENODO_PCAM.format(name), name, size, md5)
        for name, (size, md5) in _PCAM_FILES.items()
    ],
    "eurosat": [
        RemoteFile(
            "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1",
            "EuroSAT_RGB.zip",
            94_658_721,
            "f46e308c4d50d4bf32fedad2d3d62f3b",
        )
    ],
}
