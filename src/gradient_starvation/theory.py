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
    # ``predicted_drift`` is the closed ``Gg`` component.  It is the exact response
    # drift only when the signed logits have an exact, parameter-independent
    # two-mode realization.  For nonlinear probe responses, direct autograd is
    # the universal cross-kernel drift and this residual quantifies what ``Gg`` omits.
    mode_residual_rms: torch.Tensor
    projected_drift_residual: torch.Tensor | None


@dataclass(frozen=True)
class DiscreteCrossoverCertificate:
    """Exact finite-step analogue of the transfer/suppression tail-area theorem."""

    single_transfer_to_suppression: bool
    drift_crossover_step: int | None
    peak_step: int | None
    response_equality_step: int | None
    positive_area: float
    negative_tail_area: float
    tail_area_margin: float
    response_equality_reached: bool
    strict_outcome_starvation: bool
    weak_only_learnable: bool
    causal_starvation_certified: bool


@dataclass
class RankOneDriftRatio:
    """Ordering-invariant noiseless rank-one weak-drift factorization."""

    both_gate: torch.Tensor
    weak_only_gate: torch.Tensor
    both_geometry_factor: torch.Tensor
    weak_only_geometry_factor: torch.Tensor
    both_factorized_drift: torch.Tensor
    weak_only_factorized_drift: torch.Tensor
    projected_both_drift: torch.Tensor
    projected_weak_only_drift: torch.Tensor
    both_factorization_error: torch.Tensor
    weak_only_factorization_error: torch.Tensor
    log_drift_ratio: torch.Tensor
    drift_difference: torch.Tensor
    positive_drifts: bool


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
    optimization time (transfer) and ``d_w < 0`` means it causally suppresses the
    weak learning *rate*.  A zero is a transfer-to-suppression crossover.  Calling
    the outcome starvation additionally requires a weak-response crossing and the
    weak-only learnability gate.
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
    """Compute CE mode diagnostics and finite-width response drifts.

    ``direct_drift`` is the universal response/logit cross-kernel derivative from
    Theorem A.1 when requested. ``predicted_drift = Gg`` is exact only when signed
    logits are exactly realized by the two parameter-independent coordinates, as in
    the linear synthetic model with zero background noise. For nonlinear unit-probe
    tanh/GRU responses it is a projected component; ``projected_drift_residual`` and
    ``mode_residual_rms`` expose rather than hide that distinction.

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
    effective_z_s = (
        batch.z_s if batch.condition == "both" else torch.zeros_like(batch.z_s)
    )
    coordinates = torch.stack((effective_z_s, batch.z_w), dim=1)
    field = (coordinates * weights[:, None]).mean(dim=0)
    sensitivity = torch.einsum("ni,nj,n->ij", coordinates, coordinates, curvature) / len(coordinates)
    mode = differentiable_mode_responses(model, batch)
    projected_margins = coordinates @ mode
    mode_residual_rms = (margins - projected_margins).square().mean().sqrt()
    geometry, mode_gradients = gradient_gram(mode, model, create_graph=create_graph)
    predicted = geometry @ field

    direct = None
    projected_drift_residual = None
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
        projected_drift_residual = direct - predicted

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
        mode_residual_rms=mode_residual_rms,
        projected_drift_residual=projected_drift_residual,
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
    """Difference of the two conditions' weak drifts at equal optimization time.

    ``d_w_equal_time`` is the difference of the two projected ``Gg`` drifts. It is
    exactly ``d/dtau[m_w(both)-m_w(weak-only)]`` only under exact two-mode logit
    realization. When both inputs carry ``direct_drift``,
    ``exact_d_w_equal_time`` is the universal direct-autograd derivative and
    ``projection_residual`` records the difference. Nonlinear crossover claims must
    use the exact field.

    It is **not** the matched-state deficit of :func:`crossover_decomposition`, which
    recomputes the weak-only field at the both-feature ``m_w`` and so does not
    differentiate the equal-time gap.  Historical projected runs crossed at
    different times under the two conventions, which is why neither may be
    substituted for the universal direct-autograd response-gap derivative.

    Attribution between geometry and field is **not unique**, and this class does not
    pretend otherwise.  Splitting a product difference ``G_B g_B - G_W g_W`` requires
    choosing which factor is held at which condition, and the two admissible choices
    are both exact:

    ``ordering_a``  ``(G_B - G_W) g_W  +  G_B (g_B - g_W)``
    ``ordering_b``  ``(G_B - G_W) g_B  +  G_W (g_B - g_W)``

    Both reconstruct the difference identically, and on the tanh run they disagree
    about which channel dominates late.  Any claim of the form "late suppression is
    geometry-driven" is therefore a statement about the chosen ordering, not about the
    system, unless it holds under both.  :attr:`dominance_is_ordering_invariant`
    records whether it does.
    """

    d_w_equal_time: torch.Tensor
    both_drift: torch.Tensor
    weak_only_drift: torch.Tensor
    cross_transport: torch.Tensor
    geometry_difference_a: torch.Tensor
    field_difference_a: torch.Tensor
    geometry_difference_b: torch.Tensor
    field_difference_b: torch.Tensor
    reconstruction_error_a: torch.Tensor
    reconstruction_error_b: torch.Tensor
    # The additive product decompositions above reconstruct the projected ``Gg``
    # difference.  When direct autograd drifts are present, these fields expose the
    # exact derivative of the response gap and the omitted projection residual.
    exact_d_w_equal_time: torch.Tensor | None = None
    projection_residual: torch.Tensor | None = None

    @property
    def exact_drift_available(self) -> bool:
        return self.exact_d_w_equal_time is not None

    @property
    def dominance_is_ordering_invariant(self) -> bool:
        """True when both orderings agree on which channel is larger in magnitude."""
        a = bool(
            self.geometry_difference_a.abs() > self.field_difference_a.abs()
        )
        b = bool(
            self.geometry_difference_b.abs() > self.field_difference_b.abs()
        )
        return a == b


def equal_time_drift_difference(
    both: ProjectedStatistics, weak_only: ProjectedStatistics
) -> EqualTimeDriftDifference:
    """Return the equal-time weak-drift difference under both exact orderings.

    Each condition's projected weak drift is ``F_w = sum_b G_wb g_b`` evaluated in its
    own state, so their difference is the derivative of the equal-time response gap::

        d_w = [G_ws^B g_s^B + G_ww^B g_w^B] - [G_ww^W g_w^W]

    The strong channel contributes only through the both-feature condition because
    the weak-only intervention sets its effective coordinate to zero. The common
    strong probe response may be nonzero in the weak-only model, but its CE field
    component is exactly zero, so it contributes no weak-only loss drift.

    The remaining product difference ``G_ww^B g_w^B - G_ww^W g_w^W`` admits two exact
    splits, and **both are computed** because they can disagree about which channel
    dominates::

        ordering A: (G_ww^B - G_ww^W) g_w^W + G_ww^B (g_w^B - g_w^W)
        ordering B: (G_ww^B - G_ww^W) g_w^B + G_ww^W (g_w^B - g_w^W)

    Each reconstructs ``d_w`` exactly; ``reconstruction_error_a`` and
    ``reconstruction_error_b`` are returned so that is checkable rather than assumed.
    An earlier version mixed the two -- pairing ordering A's geometry term with
    ordering B's field term -- which reconstructed nothing and inflated the apparent
    geometry contribution by up to ``0.75`` on the tanh run.
    """
    both_drift = both.predicted_drift[1]
    weak_only_drift = weak_only.predicted_drift[1]
    difference = both_drift - weak_only_drift

    geometry_gap = both.geometry[1, 1] - weak_only.geometry[1, 1]
    field_gap = both.field[1] - weak_only.field[1]
    cross_transport = both.geometry[1, 0] * both.field[0]

    geometry_a = geometry_gap * weak_only.field[1] + cross_transport
    field_a = both.geometry[1, 1] * field_gap
    geometry_b = geometry_gap * both.field[1] + cross_transport
    field_b = weak_only.geometry[1, 1] * field_gap

    exact_difference = None
    projection_residual = None
    if both.direct_drift is not None and weak_only.direct_drift is not None:
        exact_difference = both.direct_drift[1] - weak_only.direct_drift[1]
        projection_residual = exact_difference - difference

    return EqualTimeDriftDifference(
        d_w_equal_time=difference,
        both_drift=both_drift,
        weak_only_drift=weak_only_drift,
        cross_transport=cross_transport,
        geometry_difference_a=geometry_a,
        field_difference_a=field_a,
        geometry_difference_b=geometry_b,
        field_difference_b=field_b,
        reconstruction_error_a=(geometry_a + field_a) - difference,
        reconstruction_error_b=(geometry_b + field_b) - difference,
        exact_d_w_equal_time=exact_difference,
        projection_residual=projection_residual,
    )


def rank_one_drift_ratio(
    both: ProjectedStatistics,
    weak_only: ProjectedStatistics,
    rho: float,
) -> RankOneDriftRatio:
    """Evaluate Theorem C's noiseless rank-one weak-drift factorization.

    The caller is responsible for the theorem hypotheses: deterministic positive
    cues ``z^B=(rho,1)``, ``z^W=(0,1)``, and exact two-mode realization.  The
    returned factorization errors make violations numerically visible.  A finite
    log ratio is emitted only when both factorized weak drifts are strictly positive.
    """
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive.")

    rho_tensor = both.mode.new_tensor(float(rho))
    both_gate = torch.sigmoid(-(rho_tensor * both.mode[0] + both.mode[1]))
    weak_gate = torch.sigmoid(-weak_only.mode[1])
    both_factor = rho_tensor * both.geometry[1, 0] + both.geometry[1, 1]
    weak_factor = weak_only.geometry[1, 1]
    both_drift = both_gate * both_factor
    weak_drift = weak_gate * weak_factor
    positive = bool((both_drift.detach() > 0) and (weak_drift.detach() > 0))
    log_ratio = (
        torch.log(both_drift / weak_drift)
        if positive
        else both_drift.new_tensor(float("nan"))
    )
    return RankOneDriftRatio(
        both_gate=both_gate,
        weak_only_gate=weak_gate,
        both_geometry_factor=both_factor,
        weak_only_geometry_factor=weak_factor,
        both_factorized_drift=both_drift,
        weak_only_factorized_drift=weak_drift,
        projected_both_drift=both.predicted_drift[1],
        projected_weak_only_drift=weak_only.predicted_drift[1],
        both_factorization_error=both_drift - both.predicted_drift[1],
        weak_only_factorization_error=weak_drift - weak_only.predicted_drift[1],
        log_drift_ratio=log_ratio,
        drift_difference=both_drift - weak_drift,
        positive_drifts=positive,
    )


def discrete_crossover_certificate(
    response_gap: Iterable[float],
    *,
    weak_only_learnable: bool = False,
    tolerance: float = 0.0,
) -> DiscreteCrossoverCertificate:
    """Certify the exact finite-step tail-area criterion from realized responses.

    For values ``Delta_k``, the increments ``Delta_(k+1)-Delta_k`` replace the
    continuous drift integral exactly.  The strict certificate requires one block
    of positive increments followed by one block of negative increments.  It does
    not infer behaviour between logged points and does not label causal starvation
    unless ``weak_only_learnable`` is supplied by the preregistered target gate.
    """
    values = np.asarray(list(response_gap), dtype=np.float64)
    if values.ndim != 1 or len(values) < 3:
        raise ValueError("response_gap must contain at least three scalar values.")
    if not np.isfinite(values).all():
        raise ValueError("response_gap must contain only finite values.")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative.")

    increments = np.diff(values)
    negative = np.flatnonzero(increments < -tolerance)
    if len(negative) == 0:
        return DiscreteCrossoverCertificate(
            single_transfer_to_suppression=False,
            drift_crossover_step=None,
            peak_step=None,
            response_equality_step=None,
            positive_area=float("nan"),
            negative_tail_area=float("nan"),
            tail_area_margin=float("nan"),
            response_equality_reached=False,
            strict_outcome_starvation=False,
            weak_only_learnable=bool(weak_only_learnable),
            causal_starvation_certified=False,
        )

    crossover = int(negative[0])
    strict_pattern = bool(
        crossover > 0
        and np.all(increments[:crossover] > tolerance)
        and np.all(increments[crossover:] < -tolerance)
    )
    if not strict_pattern:
        return DiscreteCrossoverCertificate(
            single_transfer_to_suppression=False,
            drift_crossover_step=crossover,
            peak_step=crossover,
            response_equality_step=None,
            positive_area=float("nan"),
            negative_tail_area=float("nan"),
            tail_area_margin=float("nan"),
            response_equality_reached=False,
            strict_outcome_starvation=False,
            weak_only_learnable=bool(weak_only_learnable),
            causal_starvation_certified=False,
        )

    initial = float(values[0])
    peak = float(values[crossover])
    positive_area = peak - initial
    negative_tail = peak - float(values[-1])
    margin = negative_tail - positive_area
    post_peak = values[crossover + 1 :]
    reached_positions = np.flatnonzero(post_peak <= initial + tolerance)
    equality_step = (
        crossover + 1 + int(reached_positions[0])
        if len(reached_positions)
        else None
    )
    equality_reached = bool(margin >= -tolerance)
    strict_starvation = bool(np.any(post_peak < initial - tolerance))
    return DiscreteCrossoverCertificate(
        single_transfer_to_suppression=True,
        drift_crossover_step=crossover,
        peak_step=crossover,
        response_equality_step=equality_step,
        positive_area=positive_area,
        negative_tail_area=negative_tail,
        tail_area_margin=margin,
        response_equality_reached=equality_reached,
        strict_outcome_starvation=strict_starvation,
        weak_only_learnable=bool(weak_only_learnable),
        causal_starvation_certified=bool(strict_starvation and weak_only_learnable),
    )


def transverse_hitting_time_error_bound(
    uniform_trajectory_error: float,
    crossing_slope_lower_bound: float,
    *,
    grid_spacing: float = 0.0,
) -> float:
    """Return Corollary D's deterministic first-hitting-time error bound."""
    if not np.isfinite(uniform_trajectory_error) or uniform_trajectory_error < 0:
        raise ValueError("uniform_trajectory_error must be finite and non-negative.")
    if not np.isfinite(crossing_slope_lower_bound) or crossing_slope_lower_bound <= 0:
        raise ValueError("crossing_slope_lower_bound must be finite and positive.")
    if not np.isfinite(grid_spacing) or grid_spacing < 0:
        raise ValueError("grid_spacing must be finite and non-negative.")
    return float(
        uniform_trajectory_error / crossing_slope_lower_bound + grid_spacing
    )


def zero_disorder_initialization_failure_bound(width: int, epsilon: float) -> float:
    """Return Theorem E.2's Chebyshev/union initialization bound.

    For the six initial scalars, the sum of coordinate variances is ``9/N``.
    The returned value is therefore ``min(1, 9/(N epsilon^2))``.
    """
    if width < 1:
        raise ValueError("width must be positive.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")
    return float(min(1.0, 9.0 / (float(width) * epsilon**2)))


def zero_disorder_trajectory_failure_bound(
    width: int,
    delta: float,
    lipschitz_constant: float,
    horizon: float,
) -> float:
    """Propagate Theorem E.2's initialization bound through Gronwall.

    The caller must establish that ``lipschitz_constant`` is valid on a common
    compact tube containing both trajectories over ``[0,horizon]``.
    """
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive.")
    if not np.isfinite(lipschitz_constant) or lipschitz_constant < 0:
        raise ValueError("lipschitz_constant must be finite and non-negative.")
    if not np.isfinite(horizon) or horizon < 0:
        raise ValueError("horizon must be finite and non-negative.")
    initial_epsilon = delta * np.exp(-lipschitz_constant * horizon)
    return zero_disorder_initialization_failure_bound(width, initial_epsilon)


def cdc_finite_step_deviation_bound(
    lipschitz_constant: float,
    learning_rate: float,
    erm_velocity_norm: float,
    corrected_velocity_norm: float,
) -> float:
    """Return CDC Result 3's local one-step strong-response bound.

    This evaluates ``L eta^2 (||v||^2+||v'||^2)/2``.  The caller must establish
    local ``L``-smoothness of the protected response on both step segments.
    """
    values = {
        "lipschitz_constant": lipschitz_constant,
        "learning_rate": learning_rate,
        "erm_velocity_norm": erm_velocity_norm,
        "corrected_velocity_norm": corrected_velocity_norm,
    }
    if any(not np.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError(f"bound inputs must be finite and non-negative: {values}")
    return float(
        0.5
        * lipschitz_constant
        * learning_rate**2
        * (erm_velocity_norm**2 + corrected_velocity_norm**2)
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
    # Project for every mathematically nonzero strong gradient.  The feasibility
    # tolerance applies to the *rescue* norm, not to this orthogonality identity;
    # switching the projection off for a small-but-nonzero ``a`` would invalidate
    # exact first-order preservation.
    strong_is_nonzero = strong_norm_sq > 0
    safe_strong_norm_sq = torch.where(
        strong_is_nonzero, strong_norm_sq, torch.ones_like(strong_norm_sq)
    )
    projection_scale = torch.where(
        strong_is_nonzero,
        weak_strong_inner / safe_strong_norm_sq,
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
        is_nonzero = norm_sq > 0
        safe_norm_sq = torch.where(is_nonzero, norm_sq, torch.ones_like(norm_sq))
        scale = torch.where(
            is_nonzero,
            inner(vector, direction) / safe_norm_sq,
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
