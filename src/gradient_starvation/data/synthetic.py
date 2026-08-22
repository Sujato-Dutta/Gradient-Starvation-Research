"""Paired two-cue tasks for causal gradient-starvation experiments.

The latent coordinates are label-aligned: a positive coordinate supports the
correct class and a negative coordinate opposes it.  Both conditions share the
same labels, coordinates, and background noise.  The weak-only condition is
therefore an exact ablation of the strong input channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SyntheticTaskSpec:
    """Parameters defining a paired two-channel temporal classification task."""

    sequence_length: int = 20
    n_samples: int = 4096
    rho: float = 1.0
    lag_separation: int = 0
    regime: str = "positive"
    cue_noise: float = 0.1
    background_noise: float = 0.0

    def __post_init__(self) -> None:
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be at least 1.")
        if self.n_samples < 1:
            raise ValueError("n_samples must be at least 1.")
        if self.rho <= 0:
            raise ValueError("rho must be positive.")
        if not 0 <= self.lag_separation < self.sequence_length:
            raise ValueError("lag_separation must be in [0, sequence_length).")
        if self.regime not in {"positive", "negative"}:
            raise ValueError("regime must be 'positive' or 'negative'.")
        if self.cue_noise < 0 or self.background_noise < 0:
            raise ValueError("cue_noise and background_noise must be non-negative.")

    @property
    def strong_time(self) -> int:
        """The dominant cue is presented at the final recurrent time step."""
        return self.sequence_length - 1

    @property
    def weak_time(self) -> int:
        """The weak cue precedes the dominant cue by ``lag_separation`` steps."""
        return self.strong_time - self.lag_separation


@dataclass(frozen=True)
class SyntheticBatch:
    """A generated task condition and its label-aligned latent coordinates."""

    x: torch.Tensor
    y: torch.Tensor
    signed_labels: torch.Tensor
    z_s: torch.Tensor
    z_w: torch.Tensor
    condition: str
    spec: SyntheticTaskSpec

    def __post_init__(self) -> None:
        if self.condition not in {"both", "weak_only"}:
            raise ValueError("condition must be 'both' or 'weak_only'.")
        expected = (self.spec.n_samples, self.spec.sequence_length, 2)
        if tuple(self.x.shape) != expected:
            raise ValueError(f"x must have shape {expected}; received {tuple(self.x.shape)}.")
        for name, value in {
            "y": self.y,
            "signed_labels": self.signed_labels,
            "z_s": self.z_s,
            "z_w": self.z_w,
        }.items():
            if tuple(value.shape) != (self.spec.n_samples,):
                raise ValueError(f"{name} must have shape ({self.spec.n_samples},).")

    def to(self, device: torch.device | str) -> "SyntheticBatch":
        """Return the batch on ``device`` while retaining its immutable spec."""
        return SyntheticBatch(
            x=self.x.to(device),
            y=self.y.to(device),
            signed_labels=self.signed_labels.to(device),
            z_s=self.z_s.to(device),
            z_w=self.z_w.to(device),
            condition=self.condition,
            spec=self.spec,
        )


def _coordinates(spec: SyntheticTaskSpec, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate label-aligned cue coordinates for the requested correlation regime."""
    n_samples = spec.n_samples
    dtype = torch.get_default_dtype()
    if spec.regime == "positive":
        z_s = torch.full((n_samples,), float(spec.rho), dtype=dtype)
        z_w = torch.ones(n_samples, dtype=dtype)
    else:
        # The anti-correlated latent factor yields E[z_s z_w] < 0 while each
        # feature remains individually positively label-aligned in expectation.
        factor = torch.empty(n_samples, dtype=dtype).bernoulli_(0.5, generator=generator)
        factor = factor.mul_(2.0).sub_(1.0).mul_(1.5)
        z_s = float(spec.rho) * (1.0 + factor)
        z_w = 1.0 - factor
    if spec.cue_noise:
        z_s = z_s + float(spec.rho * spec.cue_noise) * torch.randn(
            n_samples, generator=generator, dtype=dtype
        )
        z_w = z_w + float(spec.cue_noise) * torch.randn(
            n_samples, generator=generator, dtype=dtype
        )
    return z_s, z_w


def make_paired_task(spec: SyntheticTaskSpec, seed: int) -> tuple[SyntheticBatch, SyntheticBatch]:
    """Return matched both-feature and weak-only batches for one random seed."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    y = torch.randint(0, 2, (spec.n_samples,), generator=generator, dtype=torch.long)
    signed_labels = y.mul(2).sub(1).to(torch.get_default_dtype())
    z_s, z_w = _coordinates(spec, generator)

    x = torch.zeros(spec.n_samples, spec.sequence_length, 2, dtype=torch.get_default_dtype())
    if spec.background_noise:
        x = x + float(spec.background_noise) * torch.randn(
            x.shape, generator=generator, dtype=x.dtype
        )
    x[:, spec.strong_time, 0] += signed_labels * z_s
    x[:, spec.weak_time, 1] += signed_labels * z_w
    weak_x = x.clone()
    weak_x[:, :, 0] = 0

    shared = {
        "y": y,
        "signed_labels": signed_labels,
        "z_s": z_s,
        "z_w": z_w,
        "spec": spec,
    }
    return (
        SyntheticBatch(x=x, condition="both", **shared),
        SyntheticBatch(x=weak_x, condition="weak_only", **shared),
    )
