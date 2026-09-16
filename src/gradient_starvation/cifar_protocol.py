"""Deterministic CIFAR-10 core/ribbon construction for the *draft* image study.

No train/test outcome is selected here. The original 32x32 core is preserved;
the B and W tensors differ only in the appended ribbon. All random choices use
stable SHA-256-derived seeds, never Python's process-dependent ``hash``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


def stable_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def cifar_splits(labels: np.ndarray, *, seed: int, train_per_class: int = 4000,
                 validation_per_class: int = 500, diagnostic_per_class: int = 500,
                 probe_fit_per_class: int = 250) -> dict[str, np.ndarray]:
    """Partition original image IDs once per class; no image crosses splits."""
    y = np.asarray(labels)
    if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or set(np.unique(y)) != set(range(10)):
        raise ValueError("Expected one-dimensional integer CIFAR-10 labels 0..9.")
    if min(train_per_class, validation_per_class, diagnostic_per_class) <= 0:
        raise ValueError("Every split count must be positive.")
    if not 0 < probe_fit_per_class < diagnostic_per_class:
        raise ValueError("Probe-fit count must lie strictly inside the diagnostic count.")
    counts = train_per_class + validation_per_class + diagnostic_per_class
    split = {name: [] for name in ("train", "validation", "probe_fit", "diagnostic_evaluation")}
    for label in range(10):
        ids = np.flatnonzero(y == label)
        if len(ids) != counts:
            raise ValueError(f"Class {label} has {len(ids)} images; expected {counts}.")
        rng = np.random.default_rng(stable_seed("cifar-split", seed, label))
        ids = rng.permutation(ids)
        a, b, c = train_per_class, validation_per_class, probe_fit_per_class
        split["train"].extend(ids[:a])
        split["validation"].extend(ids[a:a + b])
        split["probe_fit"].extend(ids[a + b:a + b + c])
        split["diagnostic_evaluation"].extend(ids[a + b + c:])
    return {name: np.asarray(indices, dtype=np.int64) for name, indices in split.items()}


def hadamard_codes() -> torch.Tensor:
    matrix = np.array([[1]], dtype=np.int8)
    for _ in range(4):
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    # The first row is constant; the next ten rows are balanced.
    codes = matrix[1:11].reshape(10, 4, 4).copy()
    return torch.from_numpy(codes)


@dataclass(frozen=True)
class CueMetadata:
    code_index: int
    row: int
    column: int
    reliable: bool


def cue_metadata(*, block_seed: int, image_id: int, label: int,
                 reliability: float, assignment: np.ndarray) -> CueMetadata:
    if not 0 <= label < 10 or not 0 <= reliability <= 1:
        raise ValueError("Invalid class label or cue reliability.")
    if sorted(np.asarray(assignment).tolist()) != list(range(10)):
        raise ValueError("Assignment must permute ten distinct code indices.")
    rng = np.random.default_rng(stable_seed("cifar-cue", block_seed, image_id))
    reliable = bool(rng.random() < reliability)
    code = int(assignment[label])
    if not reliable:
        wrong = [int(value) for value in assignment if value != code]
        code = wrong[int(rng.integers(0, 9))]
    return CueMetadata(
        code_index=code,
        row=int(rng.integers(0, 5)),
        column=int(rng.integers(0, 29)),
        reliable=reliable,
    )


def augment_core(core: torch.Tensor, *, seed: int, training: bool) -> torch.Tensor:
    if core.shape != (3, 32, 32) or not torch.isfinite(core).all():
        raise ValueError("Core must be a finite RGB 32x32 tensor.")
    if not training:
        return core
    rng = np.random.default_rng(seed)
    padded = F.pad(core.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
    top = int(rng.integers(0, 9))
    left = int(rng.integers(0, 9))
    cropped = padded[:, top:top + 32, left:left + 32]
    return torch.flip(cropped, dims=(-1,)) if rng.random() < 0.5 else cropped


def ribbon_input(core: torch.Tensor, *, code: torch.Tensor | None,
                 row: int, column: int, amplitude: float) -> torch.Tensor:
    """Return 3x40x32 input; neutral ribbon is exactly zero after normalization."""
    if core.shape != (3, 32, 32) or not torch.isfinite(core).all():
        raise ValueError("Core must be a finite RGB 32x32 tensor.")
    if not 0 <= row <= 4 or not 0 <= column <= 28 or not 0 <= amplitude <= 1:
        raise ValueError("Invalid ribbon position or amplitude.")
    if code is not None and code.shape != (4, 4):
        raise ValueError("Cue code must be a 4x4 pattern.")
    output = torch.zeros((3, 40, 32), dtype=core.dtype)
    output[:, :32, :] = 2 * core - 1
    if code is not None:
        output[:, 32 + row:36 + row, column:column + 4] = amplitude * code.to(core.dtype)
    return output


def paired_ribbon_inputs(core: torch.Tensor, *, code: torch.Tensor,
                         row: int, column: int, amplitude: float) -> tuple[torch.Tensor, torch.Tensor]:
    both = ribbon_input(core, code=code, row=row, column=column, amplitude=amplitude)
    weak = ribbon_input(core, code=None, row=row, column=column, amplitude=amplitude)
    return both, weak


class CIFARRibbonDataset(Dataset):
    """A condition-specific view of a torchvision CIFAR10 dataset.

    ``set_epoch`` is for single-process loaders; pilot runs use num_workers=0.
    The B/W matching contract is tested via ``paired_ribbon_inputs`` and stable
    per-image/per-epoch seeds, but this pilot class does not run B training.
    """

    def __init__(self, base: object, indices: np.ndarray, *, block_seed: int,
                 condition: str, training: bool, reliability: float, amplitude: float) -> None:
        if condition not in {"both", "weak_only", "strong_only"}:
            raise ValueError("Unknown cue condition.")
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.block_seed = block_seed
        self.condition = condition
        self.training = training
        self.reliability = reliability
        self.amplitude = amplitude
        self.epoch = 0
        self.codes = hadamard_codes()
        rng = np.random.default_rng(stable_seed("cifar-code-assignment", block_seed))
        self.assignment = rng.permutation(10)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Epoch must be non-negative.")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_id = int(self.indices[index])
        image, label = self.base[image_id]
        # torchvision CIFAR10 supplies PIL images; test fixtures can supply arrays.
        pixels = np.asarray(image, dtype=np.uint8)
        if pixels.shape != (32, 32, 3):
            raise ValueError("CIFAR source image must be HWC RGB 32x32.")
        core = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255.0
        core = augment_core(
            core, seed=stable_seed("cifar-augment", self.block_seed, image_id, self.epoch),
            training=self.training,
        )
        metadata = cue_metadata(
            block_seed=self.block_seed, image_id=image_id, label=int(label),
            reliability=self.reliability, assignment=self.assignment,
        )
        if self.condition == "strong_only":
            core = torch.full_like(core, 0.5)
        code = None if self.condition == "weak_only" else self.codes[metadata.code_index]
        return ribbon_input(core, code=code, row=metadata.row,
                            column=metadata.column, amplitude=self.amplitude), int(label)


class CNN4GN(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int] = (32, 64, 128, 128),
                 groups: int = 8, classes: int = 10) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            if out_channels % groups:
                raise ValueError("Each channel count must be divisible by GroupNorm groups.")
            layers.extend((nn.Conv2d(in_channels, out_channels, 3, padding=1),
                           nn.GroupNorm(groups, out_channels), nn.ReLU(), nn.MaxPool2d(2)))
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(in_channels, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(inputs))


def balanced_brier_skill(logits: torch.Tensor, labels: torch.Tensor, classes: int = 10) -> torch.Tensor:
    """Class-balanced skill: uniform predictions=0, perfect predictions=1."""
    if logits.ndim != 2 or logits.shape[1] != classes or labels.shape != (len(logits),):
        raise ValueError("Expected [batch, classes] logits and matching labels.")
    if torch.any((labels < 0) | (labels >= classes)):
        raise ValueError("Labels out of range.")
    probabilities = logits.softmax(dim=1)
    targets = F.one_hot(labels.long(), num_classes=classes).to(probabilities.dtype)
    squared = (probabilities - targets).square().sum(dim=1)
    counts = torch.bincount(labels.long(), minlength=classes)
    if torch.any(counts == 0):
        raise ValueError("Every class must occur in a class-balanced response batch.")
    balanced_mse = torch.stack([squared[labels == label].mean() for label in range(classes)]).mean()
    return 1 - balanced_mse / (1 - 1 / classes)
