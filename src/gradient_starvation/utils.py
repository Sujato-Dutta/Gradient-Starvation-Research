from __future__ import annotations

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


def write_environment(path: Path) -> None:
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "command": sys.argv,
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

