"""Matched CIFAR training and checkpoint-grid diagnostics.

This module contains reusable computation, not permission to run confirmation.
The CLI must enforce a frozen preregistration before reading real CIFAR data.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .cifar_protocol import (
    CNN4GN, CIFARRibbonDataset, balanced_brier_skill, cifar_splits,
    cue_metadata, hadamard_codes, ribbon_input, stable_seed,
)


def split_for_seed(base: object, config: dict, seed: int) -> dict[str, np.ndarray]:
    data = config["dataset"]
    return cifar_splits(
        np.asarray(base.targets), seed=seed,
        train_per_class=data["train_per_class"],
        validation_per_class=data["validation_per_class"],
        diagnostic_per_class=data["diagnostic_per_class"],
        probe_fit_per_class=data["diagnostic_probe_fit_per_class"],
    )


def split_sha256(splits: dict[str, np.ndarray]) -> str:
    """Hash source-image IDs in the same canonical order as calibration."""
    result = hashlib.sha256()
    for name in sorted(splits):
        result.update(name.encode("utf-8"))
        result.update(np.asarray(splits[name], dtype="<i8").tobytes())
    return result.hexdigest()


class PairedCIFARDataset(Dataset):
    """Read and augment each image once; W is an exact B-ribbon ablation."""

    def __init__(self, base: object, indices: np.ndarray, *, block_seed: int,
                 training: bool, reliability: float, amplitude: float) -> None:
        self.both = CIFARRibbonDataset(
            base, indices, block_seed=block_seed, condition="both", training=training,
            reliability=reliability, amplitude=amplitude,
        )

    def set_epoch(self, epoch: int) -> None:
        self.both.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.both)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        both, label = self.both[index]
        weak = both.clone()
        weak[:, 32:, :] = 0
        return both, weak, label, int(self.both.indices[index])


class CIFAREvaluationDataset(Dataset):
    """Fixed, unaugmented source images under a specified evaluation cue."""

    def __init__(self, base: object, indices: np.ndarray, *, block_seed: int,
                 amplitude: float, mode: str, wrong_index: int | None = None) -> None:
        if mode not in {"neutral", "consistent", "random", "conflict"}:
            raise ValueError(f"Unknown evaluation mode {mode!r}.")
        if (mode == "conflict") != (wrong_index is not None):
            raise ValueError("wrong_index is required only for conflict mode.")
        if wrong_index is not None and not 0 <= wrong_index < 9:
            raise ValueError("wrong_index must be in 0..8.")
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.block_seed = block_seed
        self.amplitude = amplitude
        self.mode = mode
        self.wrong_index = wrong_index
        self.codes = hadamard_codes()
        self.assignment = np.random.default_rng(
            stable_seed("cifar-code-assignment", block_seed)
        ).permutation(10)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_id = int(self.indices[index])
        image, label = self.base[image_id]
        pixels = np.asarray(image, dtype=np.uint8)
        if pixels.shape != (32, 32, 3):
            raise ValueError("CIFAR source image must be HWC RGB 32x32.")
        core = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255.0
        metadata = cue_metadata(
            block_seed=self.block_seed, image_id=image_id, label=int(label),
            reliability=1.0, assignment=self.assignment,
        )
        if self.mode == "neutral":
            code = None
        elif self.mode == "consistent":
            code = self.codes[int(self.assignment[int(label)])]
        elif self.mode == "random":
            # Uniform over all ten codes, independent of the true class.
            generator = np.random.default_rng(
                stable_seed("cifar-eval-random", self.block_seed, image_id)
            )
            code = self.codes[int(generator.integers(0, 10))]
        else:
            correct = int(self.assignment[int(label)])
            wrong = [value for value in range(10) if value != correct]
            code = self.codes[wrong[self.wrong_index]]
        return ribbon_input(core, code=code, row=metadata.row,
                            column=metadata.column, amplitude=self.amplitude), int(label)


@torch.no_grad()
def evaluate_classifier(model: torch.nn.Module, dataset: Dataset, device: torch.device,
                        *, batch_size: int = 256) -> dict[str, float]:
    if len(dataset) == 0:
        raise ValueError("Evaluation dataset must be nonempty.")
    model.eval()
    logits, labels = [], []
    for inputs, targets in DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=0):
        logits.append(model(inputs.to(device)).cpu())
        labels.append(targets.cpu())
    scores = torch.cat(logits)
    truth = torch.cat(labels)
    result = {
        "accuracy": float((scores.argmax(1) == truth).float().mean()),
        "brier_skill": float(balanced_brier_skill(scores, truth)),
        "cross_entropy": float(F.cross_entropy(scores, truth)),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite classifier diagnostic.")
    return result


def evaluate_behavior(model: torch.nn.Module, base: object, indices: np.ndarray,
                      *, block_seed: int, amplitude: float,
                      device: torch.device, batch_size: int = 256) -> dict[str, dict[str, float]]:
    common = dict(base=base, indices=indices, block_seed=block_seed,
                  amplitude=amplitude)
    result = {
        mode: evaluate_classifier(model, CIFAREvaluationDataset(mode=mode, **common),
                                  device, batch_size=batch_size)
        for mode in ("neutral", "random", "consistent")
    }
    conflict = [evaluate_classifier(
        model, CIFAREvaluationDataset(mode="conflict", wrong_index=i, **common),
        device, batch_size=batch_size,
    ) for i in range(9)]
    # Every source image contributes all nine wrong codes, so an equal average
    # across these nine complete datasets is the within-image average.
    result["conflict"] = {
        key: float(np.mean([part[key] for part in conflict]))
        for key in ("accuracy", "brier_skill", "cross_entropy")
    }
    return result


def weights_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def summarize_response(trace: list[dict], *, beta: float, scale: float,
                       accuracy_gate: float, scale_gate: float,
                       gap_tolerance: float, material_fraction: float) -> dict:
    """Operational, checkpoint-grid B-minus-W outcomes (not continuous flow)."""
    steps = np.asarray([row["step"] for row in trace], dtype=np.float64)
    both = np.asarray([row["both"]["brier_skill"] for row in trace], dtype=np.float64)
    weak = np.asarray([row["weak_only"]["brier_skill"] for row in trace], dtype=np.float64)
    accuracy = np.asarray([row["weak_only"]["accuracy"] for row in trace], dtype=np.float64)
    if not (len(steps) >= 2 and steps[0] == 0 and np.all(np.diff(steps) > 0)
            and np.isfinite(steps).all() and np.isfinite(both).all()
            and np.isfinite(weak).all() and np.isfinite(accuracy).all()):
        raise ValueError("Response trace must have increasing finite checkpoints from zero.")
    if not (0 < beta <= 1 and scale > 0 and 0 <= accuracy_gate <= 1
            and scale_gate > 0 and gap_tolerance >= 0 and material_fraction > 0):
        raise ValueError("Invalid response thresholds.")
    gap = both - weak
    hits = np.flatnonzero(weak >= beta)
    first_hit = int(hits[0]) if len(hits) and hits[0] > 0 and weak[0] < beta else None
    gate = bool(first_hit is not None and accuracy[-1] >= accuracy_gate
                and weak[-1] >= scale_gate)
    horizon = float(steps[-1])
    deficit = (weak - both) / scale
    interval_widths = np.diff(steps)
    signed_auc = float(np.sum((deficit[:-1] + deficit[1:]) * interval_widths / 2) / horizon)
    positive_area = 0.0
    for a, b, dt in zip(deficit[:-1], deficit[1:], interval_widths):
        if a >= 0 and b >= 0:
            positive_area += float(dt * (a + b) / 2)
        elif a > 0 > b:
            fraction = float(a / (a - b))
            positive_area += float(dt * a * fraction / 2)
        elif a < 0 < b:
            fraction = float(-a / (b - a))
            positive_area += float(dt * b * (1 - fraction) / 2)
    positive_auc = positive_area / horizon
    threshold = -material_fraction * scale
    # Linear interpolation is an operational rule for the logged checkpoints.
    # It does not assert that the unobserved training path is piecewise linear.
    max_contiguous = 0.0
    current = 0.0
    for i in range(len(gap) - 1):
        a, b = float(gap[i]), float(gap[i + 1])
        dt = float(steps[i + 1] - steps[i])
        if a < threshold and b < threshold:
            current += dt
        elif a >= threshold and b >= threshold:
            max_contiguous = max(max_contiguous, current)
            current = 0.0
        else:
            fraction = (threshold - a) / (b - a)
            if a < threshold:
                current += dt * fraction
                max_contiguous = max(max_contiguous, current)
                current = 0.0
            else:
                current = dt * (1 - fraction)
    max_contiguous = max(max_contiguous, current)
    return {
        "weak_only_gate_passed": gate,
        "weak_only_first_hit_right_censored": first_hit is None,
        "weak_only_first_hit_update": int(steps[first_hit]) if first_hit is not None else None,
        "gap_at_weak_only_first_hit": float(gap[first_hit]) if first_hit is not None else None,
        "primary_causal_certificate": bool(gate and gap[first_hit] < -gap_tolerance),
        "secondary_any_time_certificate": bool(gate and np.any(gap[1:] < -gap_tolerance)),
        "signed_normalized_deficit_auc": signed_auc,
        "positive_normalized_deficit_auc": positive_auc,
        "minimum_gap": float(np.min(gap[1:])),
        "maximum_material_deficit_duration_updates": max_contiguous,
        "maximum_material_deficit_duration_fraction": max_contiguous / horizon,
        "checkpoint_gap": [float(value) for value in gap],
    }


def train_paired(base: object, config: dict, *, seed: int, device: torch.device,
                 checkpoint: Callable[[int, torch.nn.Module, torch.nn.Module, dict], None] | None = None,
                 ) -> tuple[list[dict], torch.nn.Module, torch.nn.Module, dict]:
    """Lockstep SGD with one sampled B/W input pair per source image."""
    train_spec = config["training"]
    steps = train_spec["horizon_updates"]
    checkpoints = train_spec["checkpoint_updates"]
    if not (isinstance(steps, int) and steps > 0 and checkpoints
            and checkpoints == sorted(set(checkpoints)) and checkpoints[-1] == steps):
        raise ValueError("Invalid paired training horizon/checkpoint grid.")
    if seed not in config["seed_blocks"]["confirmation"]:
        raise ValueError("Paired training seed is outside the confirmation namespace.")
    splits = split_for_seed(base, config, seed)
    pair = PairedCIFARDataset(
        base, splits["train"], block_seed=seed, training=True,
        reliability=config["dataset"]["cue_reliability"],
        amplitude=config["dataset"]["cue_amplitude"],
    )
    diagnostic = CIFAREvaluationDataset(
        base, splits["diagnostic_evaluation"], block_seed=seed,
        amplitude=config["dataset"]["cue_amplitude"], mode="neutral",
    )
    torch.manual_seed(stable_seed("cifar-init", seed) % (2**63))
    np.random.seed(stable_seed("cifar-numpy", seed) % (2**32))
    random.seed(stable_seed("cifar-python", seed))
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    both = CNN4GN(
        channels=tuple(config["model"]["channels"]),
        groups=config["model"]["group_norm_groups"],
        classes=config["model"]["classification_classes"],
    ).to(device)
    weak = copy.deepcopy(both)
    initial = weights_sha256(both)
    assert initial == weights_sha256(weak)
    optimizers = [torch.optim.SGD(
        model.parameters(), lr=train_spec["learning_rate"],
        momentum=train_spec["momentum"], weight_decay=train_spec["weight_decay"],
    ) for model in (both, weak)]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
                  for opt in optimizers]
    start = time.monotonic()
    trace = [{"step": 0, "epoch": 0, "wall_seconds": 0.0,
              "both": evaluate_classifier(both, diagnostic, device),
              "weak_only": evaluate_classifier(weak, diagnostic, device)}]
    if trace[0]["both"] != trace[0]["weak_only"]:
        raise AssertionError("Shared initialization failed the common-neutral-response null check.")
    if checkpoint is not None:
        checkpoint(0, both, weak, trace[-1])
    order_hash = hashlib.sha256()
    update = 0
    epoch = 0
    checkpoint_set = set(checkpoints)
    while update < steps:
        pair.set_epoch(epoch)
        generator = torch.Generator().manual_seed(
            stable_seed("cifar-order", seed, epoch) % (2**63)
        )
        # Match the standalone calibration runner's RandomSampler path,
        # including DataLoader's generator consumption for its base seed.
        loader = DataLoader(pair, batch_size=train_spec["batch_size"],
                            shuffle=True, generator=generator, num_workers=0)
        for x_b, x_w, labels, image_ids in loader:
            if not torch.equal(x_b[:, :, :32], x_w[:, :, :32]) or torch.count_nonzero(x_w[:, :, 32:]):
                raise AssertionError("Paired core or exact ribbon ablation failed.")
            order_hash.update(np.asarray(image_ids, dtype="<i8").tobytes())
            losses = []
            for model, optimizer, scheduler, inputs in zip(
                (both, weak), optimizers, schedulers, (x_b, x_w)
            ):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(inputs.to(device)), labels.to(device))
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach().cpu()))
            update += 1
            if update in checkpoint_set:
                trace.append({
                    "step": update, "epoch": epoch,
                    "wall_seconds": time.monotonic() - start,
                    "train_loss_both": losses[0], "train_loss_weak_only": losses[1],
                    "both": evaluate_classifier(both, diagnostic, device),
                    "weak_only": evaluate_classifier(weak, diagnostic, device),
                })
                if checkpoint is not None:
                    checkpoint(update, both, weak, trace[-1])
            if update == steps:
                break
        epoch += 1
    return trace, both, weak, {
        "initial_weights_sha256": initial,
        "final_weights_sha256_both": weights_sha256(both),
        "final_weights_sha256_weak_only": weights_sha256(weak),
        "training_image_order_sha256": order_hash.hexdigest(),
        "split_indices_sha256": split_sha256(splits),
        "split_counts": {name: len(ids) for name, ids in splits.items()},
    }
