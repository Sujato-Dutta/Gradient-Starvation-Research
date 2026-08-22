from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .data.synthetic import SyntheticBatch, SyntheticTaskSpec
from .models.recurrent import (
    DenseLinearRNN,
    RecurrentBinaryClassifier,
    differentiable_mode_responses,
    synthetic_logits,
)


@dataclass
class ProjectedStatistics:
    mode: torch.Tensor
    field: torch.Tensor
    sensitivity: torch.Tensor
    geometry: torch.Tensor
    predicted_drift: torch.Tensor
    direct_drift: torch.Tensor | None
    margins: torch.Tensor
    sigmoid_weights: torch.Tensor
    loss: torch.Tensor


@dataclass
class CausalDriftDecomposition:
    """Exact fixed-time decomposition of the matched weak-drift deficit."""

    both_drift: torch.Tensor
    matched_weak_only_drift: torch.Tensor
    causal_deficit: torch.Tensor
    ce_gating: torch.Tensor
    geometry_shift: torch.Tensor
    cross_transport: torch.Tensor
    reconstruction_error: torch.Tensor


@dataclass
class CounterfactualDriftCorrection:
    """Minimum-norm correction that protects weak drift and strong progress."""

    gradients: list[torch.Tensor]
    weak_drift_before: torch.Tensor
    weak_drift_after: torch.Tensor
    strong_drift_before: torch.Tensor
    strong_drift_after: torch.Tensor
    target_weak_drift: torch.Tensor
    deficit: torch.Tensor
    alpha: torch.Tensor
    protected_norm_sq: torch.Tensor
    correction_norm: torch.Tensor
    feasible: bool


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _scalar_gradients(
    value: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    create_graph: bool,
    retain_graph: bool = True,
) -> list[torch.Tensor]:
    parameter_list = list(parameters)
    gradients = torch.autograd.grad(
        value,
        parameter_list,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return [
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameter_list, gradients)
    ]


def gradient_gram(
    mode: torch.Tensor, model: torch.nn.Module, *, create_graph: bool = False
) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
    parameters = _trainable_parameters(model)
    gradients = [
        _scalar_gradients(mode[index], parameters, create_graph=create_graph)
        for index in range(2)
    ]
    geometry = mode.new_zeros(2, 2)
    rows: list[torch.Tensor] = []
    for a in range(2):
        row = []
        for b in range(2):
            row.append(sum((ga * gb).sum() for ga, gb in zip(gradients[a], gradients[b])))
        rows.append(torch.stack(row))
    geometry = torch.stack(rows)
    return geometry, gradients


def projected_statistics(
    model: RecurrentBinaryClassifier,
    batch: SyntheticBatch,
    *,
    compute_direct_drift: bool = False,
    create_graph: bool = False,
) -> ProjectedStatistics:
    """Compute the exact two-mode CE field and empirical parameter geometry."""
    logits = synthetic_logits(model, batch)
    signed_labels = batch.signed_labels
    margins = signed_labels * logits
    loss = F.binary_cross_entropy_with_logits(logits, batch.y.float())
    weights = torch.sigmoid(-margins)
    curvature = torch.sigmoid(margins) * weights
    coordinates = torch.stack((batch.z_s, batch.z_w), dim=1)
    field = (coordinates * weights[:, None]).mean(dim=0)
    sensitivity = torch.einsum("ni,nj,n->ij", coordinates, coordinates, curvature) / len(coordinates)
    mode = differentiable_mode_responses(model, batch)
    geometry, mode_gradients = gradient_gram(mode, model, create_graph=create_graph)
    predicted = geometry @ field

    direct = None
    if compute_direct_drift:
        parameters = _trainable_parameters(model)
        loss_gradients = _scalar_gradients(
            loss, parameters, create_graph=False, retain_graph=True
        )
        direct = torch.stack(
            [
                -sum((gm * gl).sum() for gm, gl in zip(mode_gradient, loss_gradients))
                for mode_gradient in mode_gradients
            ]
        )

    return ProjectedStatistics(
        mode=mode,
        field=field,
        sensitivity=sensitivity,
        geometry=geometry,
        predicted_drift=predicted,
        direct_drift=direct,
        margins=margins,
        sigmoid_weights=weights,
        loss=loss,
    )


def fixed_geometry_susceptibility(stats: ProjectedStatistics) -> torch.Tensor:
    """Return chi_(w<-s) = G_ws A_ss + G_ww A_sw."""
    return stats.geometry[1, 0] * stats.sensitivity[0, 0] + stats.geometry[1, 1] * stats.sensitivity[0, 1]


def matched_weak_drift_decomposition(
    both: ProjectedStatistics,
    weak_only: ProjectedStatistics,
    z_w: torch.Tensor,
) -> CausalDriftDecomposition:
    """Decompose the weak-only minus both-feature drift at matched weak state.

    The weak-only geometry is evaluated at its current parameter state, while
    its CE field is evaluated at the both-feature weak response.  The resulting
    identity separates direct CE margin gating, the change in weak geometry,
    and transport through the strong/weak cross-geometry term::

        F_w^W - F_w^B
          = G_ww^B [g_w(0,m_w) - g_w(m_s,m_w)]
          + (G_ww^W - G_ww^B) g_w(0,m_w)
          - G_ws^B g_s(m_s,m_w).
    """
    matched_margin = both.mode[1] * z_w
    matched_weights = torch.sigmoid(-matched_margin)
    matched_g_w = (z_w * matched_weights).mean()

    both_drift = both.predicted_drift[1]
    matched_weak_drift = weak_only.geometry[1, 1] * matched_g_w
    ce_gating = both.geometry[1, 1] * (matched_g_w - both.field[1])
    geometry_shift = (weak_only.geometry[1, 1] - both.geometry[1, 1]) * matched_g_w
    cross_transport = -both.geometry[1, 0] * both.field[0]
    causal_deficit = matched_weak_drift - both_drift
    reconstructed = ce_gating + geometry_shift + cross_transport
    return CausalDriftDecomposition(
        both_drift=both_drift,
        matched_weak_only_drift=matched_weak_drift,
        causal_deficit=causal_deficit,
        ce_gating=ce_gating,
        geometry_shift=geometry_shift,
        cross_transport=cross_transport,
        reconstruction_error=reconstructed - causal_deficit,
    )


def counterfactual_drift_correction(
    model: RecurrentBinaryClassifier,
    batch: SyntheticBatch,
    target_weak_drift: torch.Tensor,
    *,
    feasibility_epsilon: float = 1e-12,
    max_alpha: float | None = None,
) -> CounterfactualDriftCorrection:
    """Return the minimum-norm CE-gradient correction protecting two modes.

    The correction is restricted to the component of ``grad(m_w)`` orthogonal
    to ``grad(m_s)``.  Consequently it leaves the instantaneous strong-mode
    drift unchanged and, when feasible and uncapped, raises weak drift to the
    weak-only target.  This is the constructive mitigation associated with the
    causal drift-comparison theorem.
    """
    parameters = _trainable_parameters(model)
    logits = synthetic_logits(model, batch)
    loss = F.binary_cross_entropy_with_logits(logits, batch.y.float())
    mode = differentiable_mode_responses(model, batch)
    loss_gradients = _scalar_gradients(loss, parameters, create_graph=False)
    mode_gradients = [
        _scalar_gradients(mode[index], parameters, create_graph=False)
        for index in range(2)
    ]

    def inner(left: list[torch.Tensor], right: list[torch.Tensor]) -> torch.Tensor:
        return sum((a * b).sum() for a, b in zip(left, right))

    strong_gradient, weak_gradient = mode_gradients
    strong_norm_sq = inner(strong_gradient, strong_gradient)
    weak_strong_inner = inner(weak_gradient, strong_gradient)
    projection_scale = torch.where(
        strong_norm_sq > feasibility_epsilon,
        weak_strong_inner / strong_norm_sq.clamp_min(feasibility_epsilon),
        strong_norm_sq.new_zeros(()),
    )
    protected_direction = [
        weak - projection_scale * strong
        for weak, strong in zip(weak_gradient, strong_gradient)
    ]
    protected_norm_sq = inner(protected_direction, protected_direction)

    erm_velocity = [-gradient for gradient in loss_gradients]
    weak_before = inner(weak_gradient, erm_velocity)
    strong_before = inner(strong_gradient, erm_velocity)
    target = target_weak_drift.detach().to(weak_before)
    deficit = torch.relu(target - weak_before)
    feasible = bool(protected_norm_sq.detach() > feasibility_epsilon or deficit.detach() <= 0)
    alpha = torch.where(
        protected_norm_sq > feasibility_epsilon,
        deficit / protected_norm_sq.clamp_min(feasibility_epsilon),
        protected_norm_sq.new_zeros(()),
    )
    if max_alpha is not None:
        alpha = alpha.clamp(max=float(max_alpha))

    corrected_gradients = [
        loss_gradient - alpha * direction
        for loss_gradient, direction in zip(loss_gradients, protected_direction)
    ]
    corrected_velocity = [-gradient for gradient in corrected_gradients]
    weak_after = inner(weak_gradient, corrected_velocity)
    strong_after = inner(strong_gradient, corrected_velocity)
    correction_norm = alpha.abs() * protected_norm_sq.sqrt()
    return CounterfactualDriftCorrection(
        gradients=corrected_gradients,
        weak_drift_before=weak_before,
        weak_drift_after=weak_after,
        strong_drift_before=strong_before,
        strong_drift_after=strong_after,
        target_weak_drift=target,
        deficit=deficit,
        alpha=alpha,
        protected_norm_sq=protected_norm_sq,
        correction_norm=correction_norm,
        feasible=feasible,
    )


def exact_dense_linear_geometry(model: DenseLinearRNN, spec: SyntheticTaskSpec) -> torch.Tensor:
    """Evaluate the finite-width analytic Gram matrix for channel-localized modes."""
    W, B, c = model.recurrent, model.input, model.readout
    lags = [spec.sequence_length - 1 - spec.strong_time, spec.sequence_length - 1 - spec.weak_time]
    channels = [0, 1]
    # Mode responses use unit channel-localized probes.  Feature strength is
    # carried by the task coordinates, not by the response geometry.
    amplitudes = [1.0, 1.0]
    p_values: list[torch.Tensor] = []
    q_values: list[torch.Tensor] = []
    s_values: list[torch.Tensor] = []
    for lag, channel, amplitude in zip(lags, channels, amplitudes):
        powers = [torch.eye(model.width, device=W.device, dtype=W.dtype)]
        for _ in range(lag):
            powers.append(powers[-1] @ W)
        b = B[:, channel]
        p_values.append(amplitude * (powers[lag] @ b))
        q_values.append(amplitude * (powers[lag].T @ c))
        recurrent_gradient = torch.zeros_like(W)
        for k in range(lag):
            left = powers[k].T @ c
            right = powers[lag - 1 - k] @ b
            recurrent_gradient = recurrent_gradient + torch.outer(left, right)
        s_values.append(amplitude * recurrent_gradient)

    rows = []
    for a in range(2):
        row = []
        for b in range(2):
            value = (p_values[a] * p_values[b]).sum() + (s_values[a] * s_values[b]).sum()
            if channels[a] == channels[b]:
                value = value + (q_values[a] * q_values[b]).sum()
            row.append(value)
        rows.append(torch.stack(row))
    return torch.stack(rows)


def integrate_projected_flow(
    initial_mode: np.ndarray,
    coordinates: np.ndarray,
    times: np.ndarray,
    geometry_trajectory: np.ndarray,
) -> np.ndarray:
    """Integrate joint CE mode dynamics using an externally closed G(t)."""
    if len(times) != len(geometry_trajectory):
        raise ValueError("times and geometry_trajectory must have equal length.")
    modes = np.zeros((len(times), 2), dtype=np.float64)
    modes[0] = initial_mode

    def drift(mode: np.ndarray, geometry: np.ndarray) -> np.ndarray:
        margin = coordinates @ mode
        weights = 1.0 / (1.0 + np.exp(np.clip(margin, -60.0, 60.0)))
        field = np.mean(coordinates * weights[:, None], axis=0)
        return geometry @ field

    for index in range(1, len(times)):
        dt = times[index] - times[index - 1]
        g0 = geometry_trajectory[index - 1]
        g1 = geometry_trajectory[index]
        gm = 0.5 * (g0 + g1)
        m = modes[index - 1]
        k1 = drift(m, g0)
        k2 = drift(m + 0.5 * dt * k1, gm)
        k3 = drift(m + 0.5 * dt * k2, gm)
        k4 = drift(m + dt * k3, g1)
        modes[index] = m + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return modes
