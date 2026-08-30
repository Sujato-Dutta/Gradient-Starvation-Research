"""Outcome-free infrastructure for exact paired semi-real image interventions.

Real image pixels are the weak/core content. A generated label-aligned patch lives
in a separate input channel, so the weak-only intervention removes that channel
bitwise without editing the real image. Dataset loading is lazy and never downloads
unless the caller explicitly requests it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


SUPPORTED_DATASETS = ("mnist", "fashion_mnist")


@dataclass(frozen=True)
class SemiRealTaskSpec:
    dataset_name: str
    negative_class: int
    positive_class: int
    cue_strength: float = 4.0
    image_size: int = 14
    cue_patch_size: int = 2

    def __post_init__(self) -> None:
        if self.dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(f"dataset_name must be one of {SUPPORTED_DATASETS}.")
        if self.negative_class == self.positive_class:
            raise ValueError("The two real-image classes must be distinct.")
        if not torch.isfinite(torch.tensor(self.cue_strength)) or self.cue_strength <= 0:
            raise ValueError("cue_strength must be finite and positive.")
        if self.image_size < 2:
            raise ValueError("image_size must be at least two.")
        if not 1 <= self.cue_patch_size <= self.image_size:
            raise ValueError("cue_patch_size must lie in [1, image_size].")


@dataclass(frozen=True)
class SemiRealBatch:
    x: torch.Tensor
    y: torch.Tensor
    signed_labels: torch.Tensor
    sample_ids: torch.Tensor
    condition: str
    spec: SemiRealTaskSpec

    def __post_init__(self) -> None:
        if self.condition not in {"both", "weak_only"}:
            raise ValueError("condition must be 'both' or 'weak_only'.")
        n = len(self.y)
        expected = (n, 2, self.spec.image_size, self.spec.image_size)
        if tuple(self.x.shape) != expected:
            raise ValueError(f"x must have shape {expected}; received {tuple(self.x.shape)}.")
        for name, value in {
            "y": self.y,
            "signed_labels": self.signed_labels,
            "sample_ids": self.sample_ids,
        }.items():
            if tuple(value.shape) != (n,):
                raise ValueError(f"{name} must have shape ({n},).")
        if self.y.dtype != torch.long or self.sample_ids.dtype != torch.long:
            raise TypeError("y and sample_ids must use torch.long.")
        if self.signed_labels.dtype != self.x.dtype:
            raise TypeError("signed_labels must share x dtype.")
        if not torch.isfinite(self.x).all() or not torch.isfinite(self.signed_labels).all():
            raise ValueError("x and signed_labels must contain only finite values.")
        if not torch.equal(torch.unique(self.y), torch.tensor([0, 1], device=self.y.device)):
            raise ValueError("A semi-real batch must contain both binary labels 0 and 1.")
        expected_signed = self.y.mul(2).sub(1).to(self.signed_labels.dtype)
        if not torch.equal(self.signed_labels, expected_signed):
            raise ValueError("signed_labels must equal 2*y-1 exactly.")
        if len(torch.unique(self.sample_ids)) != n:
            raise ValueError("sample_ids must be unique within a semi-real batch.")

    def to(self, device: torch.device | str) -> "SemiRealBatch":
        return SemiRealBatch(
            x=self.x.to(device),
            y=self.y.to(device),
            signed_labels=self.signed_labels.to(device),
            sample_ids=self.sample_ids.to(device),
            condition=self.condition,
            spec=self.spec,
        )


@dataclass(frozen=True)
class SemiRealProbe:
    """Fixed cue-free real-image probe defining the common response functionals."""

    core_images: torch.Tensor
    y: torch.Tensor
    signed_labels: torch.Tensor
    sample_ids: torch.Tensor
    spec: SemiRealTaskSpec

    def __post_init__(self) -> None:
        n = len(self.y)
        expected = (n, 1, self.spec.image_size, self.spec.image_size)
        if tuple(self.core_images.shape) != expected:
            raise ValueError(
                f"core_images must have shape {expected}; received "
                f"{tuple(self.core_images.shape)}."
            )
        if not torch.isfinite(self.core_images).all():
            raise ValueError("Probe images must contain only finite values.")
        if tuple(self.y.shape) != (n,) or tuple(self.signed_labels.shape) != (n,) or tuple(self.sample_ids.shape) != (n,):
            raise ValueError("Probe labels and sample_ids must be one-dimensional.")
        if self.y.dtype != torch.long or self.sample_ids.dtype != torch.long:
            raise TypeError("Probe y and sample_ids must use torch.long.")
        if self.signed_labels.dtype != self.core_images.dtype:
            raise TypeError("Probe signed_labels must share core_images dtype.")
        if not torch.equal(torch.unique(self.y), torch.tensor([0, 1], device=self.y.device)):
            raise ValueError("A semi-real probe must contain both binary labels 0 and 1.")
        expected_signed = self.y.mul(2).sub(1).to(self.signed_labels.dtype)
        if not torch.equal(self.signed_labels, expected_signed):
            raise ValueError("Probe signed_labels must equal 2*y-1 exactly.")
        if not torch.isfinite(self.signed_labels).all():
            raise ValueError("Probe signed labels must be finite.")
        if len(torch.unique(self.sample_ids)) != n:
            raise ValueError("Probe sample_ids must be unique.")

    def to(self, device: torch.device | str) -> "SemiRealProbe":
        return SemiRealProbe(
            core_images=self.core_images.to(device),
            y=self.y.to(device),
            signed_labels=self.signed_labels.to(device),
            sample_ids=self.sample_ids.to(device),
            spec=self.spec,
        )


@dataclass(frozen=True)
class SemiRealStatistics:
    core_response: torch.Tensor
    cue_response: torch.Tensor
    core_drift: torch.Tensor
    cue_drift: torch.Tensor
    loss: torch.Tensor
    accuracy: torch.Tensor
    sigmoid_weights: torch.Tensor


class SemiRealCNN(nn.Module):
    """Small two-channel CNN used by the frozen semi-real design."""

    def __init__(self, channels: tuple[int, int] = (16, 32)):
        super().__init__()
        first, second = channels
        if first < 1 or second < 1:
            raise ValueError("CNN channel widths must be positive.")
        self.features = nn.Sequential(
            nn.Conv2d(2, first, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(first, second, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.readout = nn.Linear(second, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.features(x).flatten(1)).squeeze(1)


def prepare_core_images(images: torch.Tensor, image_size: int) -> torch.Tensor:
    """Map raw uint8 grayscale images deterministically to `[-1,1]` and resize."""
    if images.dtype != torch.uint8:
        raise TypeError("Raw semi-real images must be uint8 so scaling is unambiguous.")
    if images.ndim == 3:
        images = images[:, None]
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("images must have shape (n,h,w) or (n,1,h,w).")
    if image_size < 2:
        raise ValueError("image_size must be at least two.")
    core = images.to(torch.float32).div(127.5).sub(1.0)
    if tuple(core.shape[-2:]) != (image_size, image_size):
        core = F.interpolate(core, size=(image_size, image_size), mode="area")
    return core.contiguous()


def _validate_core_inputs(
    core_images: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: torch.Tensor,
    spec: SemiRealTaskSpec,
) -> None:
    n = len(labels)
    if tuple(core_images.shape) != (n, 1, spec.image_size, spec.image_size):
        raise ValueError("core_images do not match labels/spec image size.")
    if tuple(sample_ids.shape) != (n,):
        raise ValueError("sample_ids must align with labels.")
    if labels.dtype != torch.long:
        raise TypeError("Binary labels must use torch.long.")
    if not torch.equal(torch.unique(labels), torch.tensor([0, 1], device=labels.device)):
        raise ValueError("labels must contain both binary classes 0 and 1.")
    if not torch.isfinite(core_images).all():
        raise ValueError("core_images must contain only finite values.")


def make_paired_semi_real_task(
    core_images: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: torch.Tensor,
    spec: SemiRealTaskSpec,
) -> tuple[SemiRealBatch, SemiRealBatch]:
    """Create a bitwise-paired generated-cue intervention on real image content."""
    _validate_core_inputs(core_images, labels, sample_ids, spec)
    signed = labels.mul(2).sub(1).to(core_images.dtype)
    both_x = core_images.new_zeros(len(labels), 2, spec.image_size, spec.image_size)
    both_x[:, :1] = core_images
    patch = spec.cue_patch_size
    both_x[:, 1, :patch, :patch] = signed[:, None, None] * spec.cue_strength
    weak_x = both_x.clone()
    weak_x[:, 1].zero_()
    shared = {
        "y": labels.clone(),
        "signed_labels": signed,
        "sample_ids": sample_ids.clone(),
        "spec": spec,
    }
    return (
        SemiRealBatch(x=both_x, condition="both", **shared),
        SemiRealBatch(x=weak_x, condition="weak_only", **shared),
    )


def make_semi_real_probe(
    core_images: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: torch.Tensor,
    spec: SemiRealTaskSpec,
) -> SemiRealProbe:
    _validate_core_inputs(core_images, labels, sample_ids, spec)
    return SemiRealProbe(
        core_images=core_images,
        y=labels.clone(),
        signed_labels=labels.mul(2).sub(1).to(core_images.dtype),
        sample_ids=sample_ids.clone(),
        spec=spec,
    )


def _probe_inputs(probe: SemiRealProbe) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(probe.y)
    shape = (n, 2, probe.spec.image_size, probe.spec.image_size)
    cue_free = probe.core_images.new_zeros(shape)
    cue_free[:, :1] = probe.core_images
    aligned = cue_free.clone()
    conflicted = cue_free.clone()
    patch = probe.spec.cue_patch_size
    cue = probe.signed_labels[:, None, None] * probe.spec.cue_strength
    aligned[:, 1, :patch, :patch] = cue
    conflicted[:, 1, :patch, :patch] = -cue
    return cue_free, aligned, conflicted


def semi_real_evaluation_accuracies(
    model: nn.Module, probe: SemiRealProbe
) -> dict[str, float]:
    """Evaluate one trained arm on cue-free, aligned, and conflicted inputs."""
    cue_free, aligned, conflicted = _probe_inputs(probe)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return {
                name: float(((model(inputs) >= 0).long() == probe.y).float().mean().item())
                for name, inputs in (
                    ("cue_free", cue_free),
                    ("aligned", aligned),
                    ("conflicted", conflicted),
                )
            }
    finally:
        model.train(was_training)


def semi_real_probe_responses(
    model: nn.Module, probe: SemiRealProbe
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return identified core and generated-cue responses common to both arms.

    `M_w` is the mean signed logit on fixed real images with the generated channel
    zero. `M_s` is the mean signed aligned-versus-conflicted cue contrast on those
    same images. Neither functional depends on the training condition.
    """
    cue_free, aligned, conflicted = _probe_inputs(probe)
    core = (probe.signed_labels * model(cue_free)).mean()
    cue = (probe.signed_labels * (model(aligned) - model(conflicted))).mean() / 2.0
    return core, cue


def _gradients(
    value: torch.Tensor,
    parameters: list[nn.Parameter],
    *,
    retain_graph: bool,
) -> list[torch.Tensor]:
    gradients = torch.autograd.grad(
        value, parameters, retain_graph=retain_graph, allow_unused=True
    )
    return [
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients)
    ]


def semi_real_statistics(
    model: nn.Module,
    batch: SemiRealBatch,
    probe: SemiRealProbe,
) -> SemiRealStatistics:
    """Compute exact full-batch CE response drifts without populating `.grad`."""
    if batch.spec != probe.spec:
        raise ValueError("Training batch and response probe must share one task spec.")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    logits = model(batch.x)
    loss = F.binary_cross_entropy_with_logits(logits, batch.y.to(logits.dtype))
    core_response, cue_response = semi_real_probe_responses(model, probe)
    core_gradients = _gradients(core_response, parameters, retain_graph=True)
    cue_gradients = _gradients(cue_response, parameters, retain_graph=True)
    loss_gradients = _gradients(loss, parameters, retain_graph=False)

    def drift(response_gradients: Iterable[torch.Tensor]) -> torch.Tensor:
        return -sum(
            (response_gradient * loss_gradient).sum()
            for response_gradient, loss_gradient in zip(
                response_gradients, loss_gradients
            )
        )

    signed_margin = batch.signed_labels * logits
    return SemiRealStatistics(
        core_response=core_response.detach(),
        cue_response=cue_response.detach(),
        core_drift=drift(core_gradients).detach(),
        cue_drift=drift(cue_gradients).detach(),
        loss=loss.detach(),
        accuracy=((logits >= 0).long() == batch.y).float().mean().detach(),
        sigmoid_weights=torch.sigmoid(-signed_margin).detach(),
    )


def balanced_sample_ids(
    labels: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    per_class: int,
    seed: int,
) -> torch.Tensor:
    """Select a reproducible balanced subset, returning stable source IDs."""
    if tuple(labels.shape) != tuple(candidate_ids.shape) or labels.ndim != 1:
        raise ValueError("labels and candidate_ids must be aligned vectors.")
    if per_class < 1:
        raise ValueError("per_class must be positive.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    selected: list[torch.Tensor] = []
    for label in (0, 1):
        ids = candidate_ids[labels == label]
        if len(ids) < per_class:
            raise ValueError(f"Class {label} has fewer than {per_class} candidates.")
        order = torch.randperm(len(ids), generator=generator)[:per_class]
        selected.append(ids[order])
    return torch.cat(selected).sort().values


def load_torchvision_binary_split(
    dataset_name: str,
    root: str | Path,
    *,
    train: bool,
    negative_class: int,
    positive_class: int,
    image_size: int,
    selected_sample_ids: torch.Tensor | None = None,
    download: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load MNIST/FashionMNIST raw bytes with stable original sample IDs.

    No stochastic transform is used. `download` defaults to false so merely
    validating a configuration cannot make a network request.
    """
    try:
        from torchvision import datasets
    except ImportError as error:  # pragma: no cover - optional environment
        raise RuntimeError(
            "Semi-real loading requires a reviewed, exactly pinned torchvision environment."
        ) from error
    dataset_classes = {
        "mnist": datasets.MNIST,
        "fashion_mnist": datasets.FashionMNIST,
    }
    if dataset_name not in dataset_classes:
        raise ValueError(f"dataset_name must be one of {SUPPORTED_DATASETS}.")
    dataset = dataset_classes[dataset_name](
        root=str(root), train=bool(train), download=bool(download)
    )
    targets = torch.as_tensor(dataset.targets, dtype=torch.long)
    source_ids = torch.arange(len(targets), dtype=torch.long)
    eligible = (targets == int(negative_class)) | (targets == int(positive_class))
    eligible_ids = source_ids[eligible]
    if selected_sample_ids is None:
        selected = eligible_ids
    else:
        selected = torch.as_tensor(selected_sample_ids, dtype=torch.long)
        if len(torch.unique(selected)) != len(selected):
            raise ValueError("selected_sample_ids must be unique.")
        if not bool(torch.isin(selected, eligible_ids).all()):
            raise ValueError("selected_sample_ids include an ineligible class/index.")
    raw_source = torch.as_tensor(dataset.data)
    if raw_source.dtype != torch.uint8:
        raise TypeError("Official torchvision pixels must remain uint8.")
    raw = raw_source[selected]
    selected_targets = targets[selected]
    labels = (selected_targets == int(positive_class)).long()
    return prepare_core_images(raw, image_size), labels, selected
