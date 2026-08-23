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
class CrossoverDecomposition:
    """Transfer-versus-suppression reparameterization of the drift deficit.

    This is an exact sign relabelling of :class:`CausalDriftDecomposition`, not a
    second derivation.  It expresses the same identity in the form used to state
    the transfer-to-starvation crossover::

        d_w = t_geom - s_ce

    where ``d_w > 0`` means the strong feature *helps* weak learning at this
    optimization time (transfer) and ``d_w < 0`` means it causally suppresses it
    (starvation).  The crossover time ``tau*`` is a zero of ``d_w``.
    """

    d_w: torch.Tensor
    t_geom: torch.Tensor
    s_ce: torch.Tensor
    geometry_self_term: torch.Tensor
    cross_transport_term: torch.Tensor
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
    """Compute the exact two-mode CE field and empirical parameter geometry.

    **Cross-entropy only.** The field ``g_a = E[z_a sigma(-margin)]``, the sensitivity
    ``A``, the margin statistics and GSI-5 are all defined by the logistic loss.  Under
    an MSE objective the parameters follow a different flow and every one of these
    diagnostics would describe a loss that is not being minimized.  Callers training
    with ``objective: mse`` must not log these; ``training.train_single`` refuses that
    combination rather than emitting mislabelled columns.
    """
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

    Matching convention
    -------------------
    This function uses the **m_w-matched-field** convention: the weak-only CE
    field is recomputed at the both-feature weak response ``both.mode[1]``, while
    the weak-only *geometry* is taken at whatever parameter state
    ``weak_only`` was measured in.  The alternative -- comparing the two
    conditions at equal optimization time ``tau`` and using the weak-only field
    at its own ``m_w`` -- is a different quantity and yields different numbers.
    The project standardizes on the convention implemented here because it is
    the one this identity is exact for; see
    ``tests/test_theory.py::test_matching_conventions_are_distinct``.
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


@dataclass
class EqualTimeDriftDifference:
    """Difference of the two conditions' own weak drifts at equal optimization time.

    This is exactly ``d/dtau [ m_w(both) - m_w(weak-only) ]``: each condition's drift
    is evaluated with *its own* field at *its own* weak response.  It is therefore the
    quantity whose sign change can be compared against the crossing of the response
    gap, and the only one for which "drift leads response" is meaningful.

    It is **not** the matched-state deficit of :func:`crossover_decomposition`, which
    recomputes the weak-only field at the both-feature ``m_w`` and so does not
    differentiate the equal-time gap.  The two cross at different times -- on the
    saved tanh run, means of ``1.741`` (matched) versus ``1.611`` (equal-time) -- so
    they must never be substituted for one another.

    No exact three-term geometry/CE split accompanies this quantity: the two
    conditions are evaluated at different weak responses, so their CE fields are not
    related by a single gating factor.  ``geometry_difference`` and
    ``field_difference`` below are a *descriptive* split, labelled as such, not the
    exact identity that :class:`CausalDriftDecomposition` provides.
    """

    d_w_equal_time: torch.Tensor
    both_drift: torch.Tensor
    weak_only_drift: torch.Tensor
    geometry_difference: torch.Tensor
    field_difference: torch.Tensor


def equal_time_drift_difference(
    both: ProjectedStatistics, weak_only: ProjectedStatistics
) -> EqualTimeDriftDifference:
    """Return the equal-time weak-drift difference and a descriptive split.

    Each condition's projected weak drift is ``F_w = sum_b G_wb g_b`` evaluated in its
    own state.  Their difference is the derivative of the equal-time response gap.

    The descriptive split holds the other factor at the both-feature value in turn::

        geometry_difference = (G_ww^B - G_ww^W) g_w^W        + G_ws^B g_s^B
        field_difference    =  G_ww^W (g_w^B - g_w^W)

    These sum to the difference exactly, but the attribution between them is a
    choice of ordering, not a unique decomposition.  Use
    :func:`crossover_decomposition` when an exact identity is required.
    """
    both_drift = both.predicted_drift[1]
    weak_only_drift = weak_only.predicted_drift[1]
    geometry_difference = (
        (both.geometry[1, 1] - weak_only.geometry[1, 1]) * weak_only.field[1]
        + both.geometry[1, 0] * both.field[0]
    )
    field_difference = weak_only.geometry[1, 1] * (both.field[1] - weak_only.field[1])
    return EqualTimeDriftDifference(
        d_w_equal_time=both_drift - weak_only_drift,
        both_drift=both_drift,
        weak_only_drift=weak_only_drift,
        geometry_difference=geometry_difference,
        field_difference=field_difference,
    )


def crossover_decomposition(
    both: ProjectedStatistics,
    weak_only: ProjectedStatistics,
    z_w: torch.Tensor,
) -> CrossoverDecomposition:
    """Re-express the matched drift deficit as transfer minus CE suppression.

    Given the exact identity proved by :func:`matched_weak_drift_decomposition`,

        F_w^W - F_w^B = ce_gating + geometry_shift + cross_transport,

    define the *causal weak drift difference* with the opposite sign so that a
    positive value means the strong feature helps::

        d_w    = F_w^B - F_w^W = -(F_w^W - F_w^B)
        s_ce   =  ce_gating
        t_geom = -geometry_shift - cross_transport

    which gives ``d_w = t_geom - s_ce`` identically.  ``t_geom`` collects the two
    geometry-mediated channels -- the change in the weak self-geometry and
    transport through the strong/weak cross term -- while ``s_ce`` isolates the
    cross-entropy margin gate produced by the strong feature raising the margin.

    This is a relabelling, not an independent derivation: it delegates entirely
    to :func:`matched_weak_drift_decomposition` so the two can never disagree.
    It inherits that function's m_w-matched-field convention.

    A sign change of ``d_w`` over optimization time is the transfer-to-starvation
    crossover.  The predicted crossover time solves ``t_geom = s_ce``.  Producing
    *sufficient conditions* for that sign change is open theory work; this
    function only measures the two competing terms.
    """
    parts = matched_weak_drift_decomposition(both, weak_only, z_w)
    s_ce = parts.ce_gating
    geometry_self_term = -parts.geometry_shift
    cross_transport_term = -parts.cross_transport
    t_geom = geometry_self_term + cross_transport_term
    d_w = -parts.causal_deficit
    return CrossoverDecomposition(
        d_w=d_w,
        t_geom=t_geom,
        s_ce=s_ce,
        geometry_self_term=geometry_self_term,
        cross_transport_term=cross_transport_term,
        reconstruction_error=(t_geom - s_ce) - d_w,
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


@dataclass
class FiniteStepDeviation:
    """One-step strong-response deviation between a corrected and ERM update."""

    learning_rates: list[float]
    deviations: list[float]
    log_log_slope: float
    instantaneous_strong_drift_change: float
    correction_norm: float


def finite_step_deviation_sweep(
    model: RecurrentBinaryClassifier,
    batch: SyntheticBatch,
    target_weak_drift: torch.Tensor,
    learning_rates: Iterable[float],
    *,
    max_alpha: float | None = None,
) -> FiniteStepDeviation:
    """Measure how the one-step strong response diverges between CDC and ERM.

    Result 1 of ``research_scope/cdc_theorem.md`` is exact and proved: the
    *instantaneous first-order* strong drift is unchanged.  Result 3 is a target:
    after an actual step of size ``eta`` the strong responses should differ by
    ``O(eta^2)``, because the first-order terms cancel and the leading survivor is
    the curvature term.

    This routine measures that scaling.  It takes one gradient computation at the
    current parameter state, forms both candidate velocities, then evaluates
    ``m_s`` after stepping along each with several ``eta`` and fits the log-log
    slope.  A slope near 2 is consistent with the target.

    The measurement is empirical.  It is not the bound Result 3 calls for, which
    needs an explicit Lipschitz neighbourhood and a bound on the corrected update
    norm.
    """
    parameters = _trainable_parameters(model)
    correction = counterfactual_drift_correction(
        model, batch, target_weak_drift, max_alpha=max_alpha
    )
    logits = synthetic_logits(model, batch)
    loss = F.binary_cross_entropy_with_logits(logits, batch.y.float())
    erm_gradients = _scalar_gradients(loss, parameters, create_graph=False)

    erm_velocity = [-gradient for gradient in erm_gradients]
    corrected_velocity = [-gradient for gradient in correction.gradients]
    baseline = [parameter.detach().clone() for parameter in parameters]

    def strong_response_after(velocity: list[torch.Tensor], step: float) -> float:
        with torch.no_grad():
            for parameter, start, direction in zip(parameters, baseline, velocity):
                parameter.copy_(start + step * direction)
            value = float(differentiable_mode_responses(model, batch)[0].detach())
            for parameter, start in zip(parameters, baseline):
                parameter.copy_(start)
        return value

    rates: list[float] = []
    deviations: list[float] = []
    for rate in learning_rates:
        corrected = strong_response_after(corrected_velocity, float(rate))
        erm = strong_response_after(erm_velocity, float(rate))
        rates.append(float(rate))
        deviations.append(abs(corrected - erm))

    usable = [
        (rate, deviation)
        for rate, deviation in zip(rates, deviations)
        if deviation > 0 and np.isfinite(deviation)
    ]
    slope = float("nan")
    if len(usable) >= 2:
        slope = float(
            np.polyfit(
                np.log([rate for rate, _ in usable]),
                np.log([deviation for _, deviation in usable]),
                deg=1,
            )[0]
        )
    return FiniteStepDeviation(
        learning_rates=rates,
        deviations=deviations,
        log_log_slope=slope,
        instantaneous_strong_drift_change=float(
            (correction.strong_drift_after - correction.strong_drift_before).detach()
        ),
        correction_norm=float(correction.correction_norm.detach()),
    )


#: Correction families sharing one weak-drift target, differing only in what they
#: protect.  This is the ablation set that tests whether CDC's distinctness comes
#: from constraining a theory-defined feature response rather than from generic
#: gradient projection.
CORRECTION_CONSTRAINTS = (
    "strong_response",       # CDC: project orthogonal to grad(m_s)
    "loss_gradient",         # ablation iii: project orthogonal to grad(L_train)
    "unconstrained",         # ablation ii: same rescue direction, no projection
    "bloop",                 # loss-gradient projection with an EMA-smoothed rescue
    "pcgrad",                # symmetric conflict removal between the two directions
)


def drift_correction(
    model: RecurrentBinaryClassifier,
    batch: SyntheticBatch,
    target_weak_drift: torch.Tensor,
    *,
    constraint: str = "strong_response",
    feasibility_epsilon: float = 1e-12,
    max_alpha: float | None = None,
    rescue_state: list[torch.Tensor] | None = None,
    ema_decay: float = 0.9,
) -> CounterfactualDriftCorrection:
    """Apply a weak-drift rescue under one of several protection rules.

    All families share the same causal target -- the instantaneous weak drift of a
    matched weak-only shadow model -- so the comparison isolates *what is
    protected*, not *what is aimed at*.

    ``strong_response`` (CDC)
        Rescue along the component of ``grad(m_w)`` orthogonal to ``grad(m_s)``.
        Preserves the instantaneous strong-response drift exactly.
    ``loss_gradient``
        The same rescue projected orthogonal to ``grad(L_train)`` instead.  This is
        the object generic gradient surgery protects, and is the ablation that tests
        whether constraining the feature response specifically matters.
    ``unconstrained``
        Rescue along ``grad(m_w)`` with no projection at all.
    ``bloop``
        Loss-gradient projection applied to an exponentially smoothed rescue
        direction, following the EMA idea of Bloop.  Requires ``rescue_state``,
        which the caller carries across steps.
    ``pcgrad``
        Symmetric conflict removal: if the ERM velocity and the rescue direction
        conflict, each has the other's conflicting component removed.

    Only ``strong_response`` carries the proved Result 1 guarantee.  The others are
    expected to perturb the strong drift, and the returned
    ``strong_drift_after - strong_drift_before`` records by how much.
    """
    if constraint not in CORRECTION_CONSTRAINTS:
        raise ValueError(
            f"Unknown constraint {constraint!r}; expected one of {CORRECTION_CONSTRAINTS}."
        )
    parameters = _trainable_parameters(model)
    logits = synthetic_logits(model, batch)
    loss = F.binary_cross_entropy_with_logits(logits, batch.y.float())
    mode = differentiable_mode_responses(model, batch)
    loss_gradients = _scalar_gradients(loss, parameters, create_graph=False)
    strong_gradient = _scalar_gradients(mode[0], parameters, create_graph=False)
    weak_gradient = _scalar_gradients(mode[1], parameters, create_graph=False)

    def inner(left: list[torch.Tensor], right: list[torch.Tensor]) -> torch.Tensor:
        return sum((a * b).sum() for a, b in zip(left, right))

    def project_out(
        vector: list[torch.Tensor], direction: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        norm_sq = inner(direction, direction)
        scale = torch.where(
            norm_sq > feasibility_epsilon,
            inner(vector, direction) / norm_sq.clamp_min(feasibility_epsilon),
            norm_sq.new_zeros(()),
        )
        return [v - scale * d for v, d in zip(vector, direction)]

    erm_velocity = [-gradient for gradient in loss_gradients]
    rescue = weak_gradient
    if constraint == "bloop":
        if rescue_state is None:
            rescue_state = [torch.zeros_like(g) for g in weak_gradient]
        for index, gradient in enumerate(weak_gradient):
            rescue_state[index].mul_(ema_decay).add_(gradient, alpha=1.0 - ema_decay)
        rescue = [state.clone() for state in rescue_state]

    if constraint == "strong_response":
        protected = project_out(rescue, strong_gradient)
    elif constraint in {"loss_gradient", "bloop"}:
        protected = project_out(rescue, loss_gradients)
    elif constraint == "unconstrained":
        protected = [direction.clone() for direction in rescue]
    else:  # pcgrad
        conflict = inner(rescue, erm_velocity)
        protected = (
            project_out(rescue, erm_velocity)
            if bool(conflict.detach() < 0)
            else [direction.clone() for direction in rescue]
        )

    protected_norm_sq = inner(protected, protected)
    weak_before = inner(weak_gradient, erm_velocity)
    strong_before = inner(strong_gradient, erm_velocity)
    target = target_weak_drift.detach().to(weak_before)
    deficit = torch.relu(target - weak_before)
    # The achievable weak-drift gain per unit alpha is grad(m_w) . protected, which
    # equals ||protected||^2 only in the CDC case; solving with the correct
    # denominator is what lets every family actually reach the shared target.
    gain = inner(weak_gradient, protected)
    feasible = bool(gain.detach() > feasibility_epsilon or deficit.detach() <= 0)
    alpha = torch.where(
        gain > feasibility_epsilon,
        deficit / gain.clamp_min(feasibility_epsilon),
        gain.new_zeros(()),
    )
    if max_alpha is not None:
        alpha = alpha.clamp(max=float(max_alpha))

    corrected_gradients = [
        gradient - alpha * direction
        for gradient, direction in zip(loss_gradients, protected)
    ]
    corrected_velocity = [-gradient for gradient in corrected_gradients]
    return CounterfactualDriftCorrection(
        gradients=corrected_gradients,
        weak_drift_before=weak_before,
        weak_drift_after=inner(weak_gradient, corrected_velocity),
        strong_drift_before=strong_before,
        strong_drift_after=inner(strong_gradient, corrected_velocity),
        target_weak_drift=target,
        deficit=deficit,
        alpha=alpha,
        protected_norm_sq=protected_norm_sq,
        correction_norm=alpha.abs() * protected_norm_sq.sqrt(),
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
