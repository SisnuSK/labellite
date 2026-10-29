"""Run manifests: what ran, on what code, with which config and hardware."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = ("torch", "torchvision", "timm", "open_clip_torch", "scikit-learn", "numpy")


def git_info(cwd: str | Path | None = None) -> dict[str, Any]:
    """Current commit and whether the working tree has uncommitted changes (None if unknown)."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain") if commit else None
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "gpu": None,
    }
    try:
        import torch
    except ImportError:
        return info
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    return info


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_manifest(command: str, config: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Assemble a manifest. Callers add stage-specific fields (split hashes, timings, ...)."""
    return {
        "command": command,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_info(),
        "config": config,
        "hardware": hardware_info(),
        "packages": package_versions(),
        **extra,
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
