from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn

from ..data.synthetic import SyntheticBatch, SyntheticTaskSpec


class RecurrentBinaryClassifier(nn.Module):
    input_size: int = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


class DenseLinearRNN(RecurrentBinaryClassifier):
    """Linear RNN with trainable dense recurrence, input, and readout."""

    def __init__(self, width: int, bulk_gain: float = 0.8, input_size: int = 2):
        super().__init__()
        self.width = width
        self.input_size = input_size
        self.recurrent = nn.Parameter(torch.randn(width, width) * (bulk_gain / math.sqrt(width)))
        # Both input and readout vectors have O(1) Euclidean norm.  The older
        # input-size scaling made the input norm grow as sqrt(width), causing
        # the projected geometry and optimization speed to diverge with N.
        self.input = nn.Parameter(torch.randn(width, input_size) / math.sqrt(width))
        self.readout = nn.Parameter(torch.randn(width) / math.sqrt(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.new_zeros(x.shape[0], self.width)
        for time in range(x.shape[1]):
            h = h @ self.recurrent.T + x[:, time] @ self.input.T
        return h @ self.readout

    def mode_responses(self, spec: SyntheticTaskSpec) -> torch.Tensor:
        return _linear_probe_responses(self, spec, self.recurrent, self.input, self.readout)


class LowRankLinearRNN(RecurrentBinaryClassifier):
    """Frozen random bulk plus trainable low-rank recurrent task structure."""

    def __init__(
        self, width: int, bulk_gain: float = 0.8, rank: int = 2, input_size: int = 2
    ):
        super().__init__()
        self.width = width
        self.rank = rank
        self.input_size = input_size
        self.register_buffer("bulk", torch.randn(width, width) * (bulk_gain / math.sqrt(width)))
        self.left = nn.Parameter(torch.randn(rank, width))
        self.right = nn.Parameter(torch.randn(rank, width))
        self.input = nn.Parameter(torch.randn(width, input_size) / math.sqrt(width))
        self.readout = nn.Parameter(torch.randn(width) / math.sqrt(width))

    @property
    def recurrent(self) -> torch.Tensor:
        return self.bulk + self.left.T @ self.right / self.width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent = self.recurrent
        h = x.new_zeros(x.shape[0], self.width)
        for time in range(x.shape[1]):
            h = h @ recurrent.T + x[:, time] @ self.input.T
        return h @ self.readout

    def mode_responses(self, spec: SyntheticTaskSpec) -> torch.Tensor:
        return _linear_probe_responses(self, spec, self.recurrent, self.input, self.readout)


class NonlinearRNN(RecurrentBinaryClassifier):
    def __init__(self, kind: str, width: int, input_size: int = 2):
        super().__init__()
        self.kind = kind
        self.width = width
        self.input_size = input_size
        if kind == "tanh":
            self.encoder: nn.Module = nn.RNN(
                input_size, width, nonlinearity="tanh", batch_first=True, bias=False
            )
        elif kind == "gru":
            self.encoder = nn.GRU(input_size, width, batch_first=True, bias=False)
        else:
            raise ValueError(f"Unsupported nonlinear recurrent model: {kind}")
        self.readout = nn.Linear(width, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(x)
        return self.readout(hidden[-1]).squeeze(-1)


def _linear_probe_responses(
    model: nn.Module,
    spec: SyntheticTaskSpec,
    recurrent: torch.Tensor,
    input_matrix: torch.Tensor,
    readout: torch.Tensor,
) -> torch.Tensor:
    del model
    probes = input_matrix.new_zeros(2, spec.sequence_length, input_matrix.shape[1])
    # Feature strength belongs to the latent coordinate z_s.  The mode response
    # is the network response to a *unit* cue; including rho here as well would
    # apply the strong-feature scale twice in ``synthetic_logits``.
    probes[0, spec.strong_time, 0] = 1.0
    probes[1, spec.weak_time, 1] = 1.0
    h = probes.new_zeros(2, recurrent.shape[0])
    for time in range(spec.sequence_length):
        h = h @ recurrent.T + probes[:, time] @ input_matrix.T
    return h @ readout


def differentiable_mode_responses(
    model: RecurrentBinaryClassifier, batch: SyntheticBatch, ridge: float = 1e-6
) -> torch.Tensor:
    """Return common strong/weak response functionals for both causal conditions.

    Linear models use their exact channel-localized impulse responses. Nonlinear
    models use an odd symmetric unit-probe contrast, ``[f(+probe)-f(-probe)]/2``.
    Crucially, the functional depends on the model and task specification but not on
    whether the current training batch is ``both`` or ``weak_only``. Shared
    parameters therefore imply an exactly shared initial weak response, as required
    by the causal crossover theorem.

    ``ridge`` is retained for API compatibility with older callers; ridge-fitted,
    condition-dependent coefficients are deliberately no longer used.
    """
    del ridge
    if hasattr(model, "mode_responses"):
        return model.mode_responses(batch.spec)  # type: ignore[attr-defined]

    probes = batch.x.new_zeros(4, batch.spec.sequence_length, model.input_size)
    probes[0, batch.spec.strong_time, 0] = 1.0
    probes[1, batch.spec.strong_time, 0] = -1.0
    probes[2, batch.spec.weak_time, 1] = 1.0
    probes[3, batch.spec.weak_time, 1] = -1.0
    outputs = model(probes)
    return torch.stack(
        (0.5 * (outputs[0] - outputs[1]), 0.5 * (outputs[2] - outputs[3]))
    )


def synthetic_logits(
    model: RecurrentBinaryClassifier, batch: SyntheticBatch
) -> torch.Tensor:
    if hasattr(model, "mode_responses") and batch.spec.background_noise == 0:
        mode = differentiable_mode_responses(model, batch)
        effective_z_s = batch.z_s if batch.condition == "both" else torch.zeros_like(batch.z_s)
        margin = mode[0] * effective_z_s + mode[1] * batch.z_w
        return batch.signed_labels * margin
    return model(batch.x)


def build_model(config: Mapping[str, object], *, kind: str | None = None) -> RecurrentBinaryClassifier:
    model_kind = kind or str(config.get("kind", "low_rank_linear"))
    width = int(config.get("width", 512))
    if model_kind == "low_rank_linear":
        return LowRankLinearRNN(
            width=width,
            bulk_gain=float(config.get("bulk_gain", 0.8)),
            rank=int(config.get("rank", 2)),
        )
    if model_kind == "dense_linear":
        return DenseLinearRNN(width=width, bulk_gain=float(config.get("bulk_gain", 0.8)))
    if model_kind in {"tanh", "gru"}:
        return NonlinearRNN(kind=model_kind, width=width)
    raise ValueError(f"Unknown model kind: {model_kind}")
