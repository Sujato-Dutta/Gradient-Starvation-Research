from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import dump_config


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def create_run_directory(config: dict[str, Any]) -> Path:
    experiment = config.get("experiment", {})
    root = Path(experiment.get("output_root", "results"))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = root / f"{experiment.get('name', 'experiment')}-{stamp}"
    run_dir = base
    index = 1
    while run_dir.exists():
        run_dir = Path(f"{base}-{index}")
        index += 1
    run_dir.mkdir(parents=True)
    dump_config(config, run_dir / "config.resolved.yaml")
    write_environment(run_dir / "environment.json")
    return run_dir


def executable_source_fingerprint() -> tuple[str, int]:
    """Hash the Python source that determines synthetic experiment execution.

    Git commit plus ``git_dirty`` cannot identify an uncommitted run. This digest
    covers the root runner and every Python file under ``src`` so a result can be
    matched to an exact reviewed source tree. The resolved YAML and dependency
    versions are recorded separately.
    """
    root = Path(__file__).resolve().parents[2]
    files = [root / "run_experiment.py", *sorted((root / "src").rglob("*.py"))]
    digest = hashlib.sha256()
    count = 0
    for source in files:
        if not source.is_file():
            continue
        relative = source.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
        count += 1
    return digest.hexdigest(), count


def write_environment(path: Path) -> None:
    source_sha256, source_file_count = executable_source_fingerprint()
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "command": sys.argv,
        "source_sha256": source_sha256,
        "source_file_count": source_file_count,
        "source_scope": "run_experiment.py and src/**/*.py",
    }
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        metadata["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        metadata["git_commit"] = None
        metadata["git_dirty"] = None
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)

