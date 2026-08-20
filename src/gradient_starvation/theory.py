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


def exact_dense_linear_geometry(model: DenseLinearRNN, spec: SyntheticTaskSpec) -> torch.Tensor:
    """Evaluate the finite-width analytic Gram matrix for channel-localized modes."""
    W, B, c = model.recurrent, model.input, model.readout
    lags = [spec.sequence_length - 1 - spec.strong_time, spec.sequence_length - 1 - spec.weak_time]
    channels = [0, 1]
    amplitudes = [spec.rho, 1.0]
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
