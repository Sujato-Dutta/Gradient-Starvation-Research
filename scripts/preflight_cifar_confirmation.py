"""Validate the frozen CIFAR confirmation cell on DGX without training B or W."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_cifar_matched import preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=33001)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        provenance = preflight(
            config,
            config_path=args.config,
            environment_path=args.environment_manifest,
            data_root=args.data_root,
            seed=args.seed,
            device=torch.device("cuda"),
        )
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10(root=str(args.data_root), train=True, download=False)
        if len(dataset) != 50_000:
            raise ValueError(f"Expected 50,000 CIFAR-10 training images, found {len(dataset)}.")
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"CIFAR confirmation preflight failed: {exc}\n")
    print(json.dumps({
        "status": "preflight_passed_no_training_performed",
        "seed_namespace_check": args.seed,
        "training_images": len(dataset),
        **provenance,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
