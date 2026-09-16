"""Phase-1 CIFAR W-only or cue-only pilot. Never trains the B condition.

Run only inside a SLURM *compute-node* allocation. This script does not freeze
the preregistration and its outputs must not be counted as confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gradient_starvation.cifar_protocol import (  # noqa: E402
    CNN4GN, CIFARRibbonDataset, balanced_brier_skill, cifar_splits, stable_seed,
)


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _weights_digest(model: torch.nn.Module) -> str:
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def _split_digest(splits: dict[str, np.ndarray]) -> str:
    result = hashlib.sha256()
    for name in sorted(splits):
        result.update(name.encode("utf-8"))
        result.update(np.asarray(splits[name], dtype="<i8").tobytes())
    return result.hexdigest()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    logits_parts = []
    label_parts = []
    for inputs, labels in loader:
        logits_parts.append(model(inputs.to(device)).cpu())
        label_parts.append(labels.cpu())
    logits = torch.cat(logits_parts)
    labels = torch.cat(label_parts)
    return {
        "accuracy": float((logits.argmax(1) == labels).float().mean()),
        "brier_skill": float(balanced_brier_skill(logits, labels)),
        "cross_entropy": float(F.cross_entropy(logits, labels)),
    }


def train_pilot(*, base: object, config: dict, condition: str, seed: int,
                amplitude: float, learning_rate: float, weight_decay: float,
                steps: int, eval_every: int, device: torch.device) -> tuple[list[dict], torch.nn.Module]:
    if condition not in {"weak_only", "strong_only"}:
        raise ValueError("Phase-1 runner refuses B training and non-pilot conditions.")
    if seed not in config["seed_blocks"]["weak_only_calibration"]:
        raise ValueError("Pilot seed must be in the weak-only calibration namespace.")
    if not (steps > 0 and eval_every > 0 and learning_rate > 0 and 0 <= weight_decay < 1):
        raise ValueError("Invalid pilot training hyperparameters.")
    torch.manual_seed(stable_seed("cifar-init", seed) % (2**63))
    np.random.seed(stable_seed("cifar-numpy", seed) % (2**32))
    random.seed(stable_seed("cifar-python", seed))
    splits = cifar_splits(
        np.asarray(base.targets), seed=seed,
        train_per_class=config["dataset"]["train_per_class"],
        validation_per_class=config["dataset"]["validation_per_class"],
        diagnostic_per_class=config["dataset"]["diagnostic_per_class"],
        probe_fit_per_class=config["dataset"]["diagnostic_probe_fit_per_class"],
    )
    common = dict(
        block_seed=seed, condition=condition,
        reliability=config["dataset"]["cue_reliability"], amplitude=amplitude,
    )
    train = CIFARRibbonDataset(base, splits["train"], training=True, **common)
    validation = CIFARRibbonDataset(base, splits["validation"], training=False, **common)
    validation_loader = DataLoader(validation, batch_size=512, shuffle=False, num_workers=0)
    model = CNN4GN(
        channels=tuple(config["model"]["channels"]),
        groups=config["model"]["group_norm_groups"],
        classes=config["model"]["classification_classes"],
    ).to(device)
    initial_digest = _weights_digest(model)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate,
        momentum=config["training"]["momentum"], weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    start = time.monotonic()
    trace = [{"step": 0, "epoch": 0, "wall_seconds": 0.0,
              "learning_rate": learning_rate, "train_loss": None,
              **evaluate(model, validation_loader, device)}]
    update = 0
    epoch = 0
    while update < steps:
        train.set_epoch(epoch)
        generator = torch.Generator().manual_seed(stable_seed("cifar-order", seed, epoch) % (2**63))
        loader = DataLoader(
            train, batch_size=config["training"]["batch_size"], shuffle=True,
            generator=generator, num_workers=0,
        )
        for inputs, labels in loader:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs.to(device))
            loss = F.cross_entropy(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            update += 1
            if update % eval_every == 0 or update == steps:
                trace.append({
                    "step": update, "epoch": epoch, "wall_seconds": time.monotonic() - start,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train_loss": float(loss.detach().cpu()),
                    **evaluate(model, validation_loader, device),
                })
            if update == steps:
                break
        epoch += 1
    trace[0]["initial_weights_sha256"] = initial_digest
    return trace, model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cifar_matched_preregistration.draft.json")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=("weak_only", "strong_only"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--amplitude", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--download", action="store_true", help="Allow torchvision to download CIFAR-10 on the compute node")
    args = parser.parse_args()
    if socket.gethostname().startswith("dgx-login"):
        parser.error("Training on the DGX login node is forbidden; obtain a SLURM compute allocation.")
    if args.output.exists():
        parser.error("Output path already exists; pilot outcomes must not be overwritten.")
    if args.condition == "strong_only" and not 0 < args.amplitude <= 1:
        parser.error("Strong-only feasibility needs an amplitude in (0,1].")
    if args.condition == "weak_only" and args.amplitude != 0:
        parser.error("W-only calibration must use amplitude zero; the ribbon remains neutral.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use a GPU compute allocation or --device cpu for a tiny smoke test.")
    config = json.loads(args.config.read_text())
    if config["status"] != "draft_do_not_run_confirmation":
        parser.error("Calibration runner expects a draft protocol, never a confirmation file.")
    try:
        import torchvision
        from torchvision.datasets import CIFAR10
    except Exception as exc:
        parser.error(f"torchvision import failed; install a build matching PyTorch: {exc}")
    try:
        base = CIFAR10(root=str(args.data_root), train=True, download=args.download)
    except Exception as exc:
        parser.error(f"CIFAR-10 is unavailable at {args.data_root}; stage verified data or use --download on a compute node: {exc}")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    trace, model = train_pilot(
        base=base, config=config, condition=args.condition, seed=args.seed,
        amplitude=args.amplitude, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, steps=args.steps,
        eval_every=args.eval_every, device=device,
    )
    archive = args.data_root / "cifar-10-python.tar.gz"
    split_hash = _split_digest(cifar_splits(
        np.asarray(base.targets), seed=args.seed,
        train_per_class=config["dataset"]["train_per_class"],
        validation_per_class=config["dataset"]["validation_per_class"],
        diagnostic_per_class=config["dataset"]["diagnostic_per_class"],
        probe_fit_per_class=config["dataset"]["diagnostic_probe_fit_per_class"],
    ))
    metadata = {
        "status": "exploratory_W_only_or_cue_only_pilot_not_confirmation",
        "condition": args.condition, "seed": args.seed, "amplitude": args.amplitude,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "steps": args.steps, "eval_every": args.eval_every,
        "device": str(device), "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
        "config_sha256": _digest(args.config),
        "protocol_code_sha256": _digest(ROOT / "src/gradient_starvation/cifar_protocol.py"),
        "runner_code_sha256": _digest(Path(__file__)),
        "split_indices_sha256": split_hash,
        "dataset_archive_sha256": _digest(archive) if archive.is_file() else None,
        "initial_weights_sha256": trace[0]["initial_weights_sha256"],
        "final_weights_sha256": _weights_digest(model),
        "first_validation": {key: trace[0][key] for key in ("accuracy", "brier_skill", "cross_entropy")},
        "final_validation": {key: trace[-1][key] for key in ("accuracy", "brier_skill", "cross_entropy")},
        "wall_seconds": trace[-1]["wall_seconds"],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    torch.save(model.state_dict(), args.output / "final_model.pt")
    print(json.dumps({"output": str(args.output), **metadata["final_validation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
