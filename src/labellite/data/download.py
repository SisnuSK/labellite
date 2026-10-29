"""Resumable, checksum-verified downloads and safe extraction."""

from __future__ import annotations

import gzip
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from labellite.data.sources import SOURCES, RemoteFile

CHUNK = 1 << 20


def md5sum(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fetch(remote: RemoteFile, dest_dir: Path) -> Path:
    """Download ``remote`` into ``dest_dir``, resuming a partial ``.part`` file if present.

    TLS certificates are always verified. Raises if the size or MD5 does not match.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / remote.filename
    if dest.exists() and _verify(remote, dest):
        return dest
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(remote.url)
    if offset:
        req.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        if offset and resp.status != 206:  # server ignored Range: start over
            offset = 0
        with open(part, "ab" if offset else "wb") as f:
            shutil.copyfileobj(resp, f, CHUNK)
    part.replace(dest)
    if not _verify(remote, dest):
        raise RuntimeError(f"{dest.name}: size/MD5 mismatch after download; delete it and retry.")
    return dest


def _verify(remote: RemoteFile, path: Path) -> bool:
    if remote.size is not None and path.stat().st_size != remote.size:
        return False
    return remote.md5 is None or md5sum(path) == remote.md5


def gunzip(src: Path, dest: Path) -> Path:
    if dest.exists():
        return dest
    tmp = dest.with_name(dest.name + ".part")
    with gzip.open(src, "rb") as fin, open(tmp, "wb") as fout:
        shutil.copyfileobj(fin, fout, CHUNK)
    tmp.replace(dest)
    return dest


def safe_extract(archive: Path, dest: Path) -> None:
    """Extract a zip archive, refusing absolute paths and ``..`` components."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                raise RuntimeError(f"unsafe path in archive: {name}")
        z.extractall(dest)


def download_dataset(name: str, root: Path) -> None:
    """Download and unpack ``name`` into ``root`` in the layout its adapter expects."""
    for remote in SOURCES[name]:
        path = fetch(remote, root)
        if path.suffix == ".gz" and path.name.endswith(".h5.gz"):
            gunzip(path, path.with_suffix(""))
        elif path.suffix == ".zip":
            safe_extract(path, root)
