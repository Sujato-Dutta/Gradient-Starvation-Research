"""Pure outcome-analysis contracts for prospective breadth studies.

These functions do not train models or read result directories. They make the
multi-beta first-hit and weak-rescue/strong-preservation decision rules testable
before any confirmatory outcomes exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .metrics import causal_metrics, classify_causal_regime


SHADOW_METHOD_PUBLICATION_LABELS = {
    "counterfactual_drift": "CDC",
    "bloop": "Bloop-style shadow-target rescue",
    "pcgrad": "PCGrad-style shadow-target rescue",
}


@dataclass(frozen=True)
class FirstHitProfilePoint:
    beta: float
    both_hitting_time: float
    weak_hitting_time: float
    gated_delta_tw: float
    right_censored: bool
    weak_only_learnable: bool
    target_met_at_initialization: bool
    regime: str
    weak_auc_gap: float
    causal_starvation_certified: bool | None


@dataclass(frozen=True)
class TradeoffDecision:
    weak_rescue_equivalent: bool
    strong_preservation_superior: bool
    joint_tradeoff_superior: bool


def first_hitting_profile(
    times: Sequence[float],
    both_values: Sequence[float],
    weak_values: Sequence[float],
    beta_values: Iterable[float],
    *,
    delay_tolerance: float,
    outcome_suppression_certified: bool | None = None,
) -> list[FirstHitProfilePoint]:
    """Evaluate a prespecified beta grid with prospective first-hit semantics.

    The whole logged trajectory is consumed once and every beta uses the existing
    linearly interpolated first hit. A terminal response above beta is not a
    substitute. `outcome_suppression_certified`, when provided, must come from an
    independent response-gap/tail-area check and is combined only with nondegenerate
    weak-only learnability.
    """
    times_array = np.asarray(times, dtype=np.float64)
    both_array = np.asarray(both_values, dtype=np.float64)
    weak_array = np.asarray(weak_values, dtype=np.float64)
    if times_array.ndim != 1 or len(times_array) < 2:
        raise ValueError("times must be a one-dimensional grid with at least two points.")
    if times_array.shape != both_array.shape or times_array.shape != weak_array.shape:
        raise ValueError("times, both_values, and weak_values must have the same shape.")
    if not (
        np.isfinite(times_array).all()
        and np.isfinite(both_array).all()
        and np.isfinite(weak_array).all()
    ):
        raise ValueError("Profile trajectories must contain only finite values.")
    if not np.all(np.diff(times_array) > 0):
        raise ValueError("times must be strictly increasing.")
    if not np.isfinite(delay_tolerance) or delay_tolerance < 0:
        raise ValueError("delay_tolerance must be finite and non-negative.")

    betas = np.asarray(list(beta_values), dtype=np.float64)
    if betas.ndim != 1 or len(betas) == 0 or not np.isfinite(betas).all():
        raise ValueError("beta_values must be a nonempty finite one-dimensional grid.")
    if not np.all(np.diff(betas) > 0):
        raise ValueError("beta_values must be strictly increasing and unique.")

    points: list[FirstHitProfilePoint] = []
    for beta in betas:
        causal = causal_metrics(times_array, both_array, weak_array, float(beta))
        verdict = classify_causal_regime(
            weak_hitting_time=causal.weak_hitting_time,
            both_hitting_time=causal.both_hitting_time,
            tau_max=float(times_array[-1]),
            delta=float(delay_tolerance),
            initial_tau=float(times_array[0]),
        )
        nondegenerate_learnability = bool(
            verdict.weak_only_learnable and not verdict.target_met_at_initialization
        )
        certificate = (
            None
            if outcome_suppression_certified is None
            else bool(outcome_suppression_certified and nondegenerate_learnability)
        )
        points.append(
            FirstHitProfilePoint(
                beta=float(beta),
                both_hitting_time=float(causal.both_hitting_time),
                weak_hitting_time=float(causal.weak_hitting_time),
                gated_delta_tw=float(verdict.delta_tw),
                right_censored=bool(verdict.right_censored),
                weak_only_learnable=nondegenerate_learnability,
                target_met_at_initialization=bool(verdict.target_met_at_initialization),
                regime=str(verdict.regime),
                weak_auc_gap=float(causal.weak_auc_gap),
                causal_starvation_certified=certificate,
            )
        )
    return points


def pareto_frontier(
    weak_rescue: Mapping[str, float],
    strong_deviation: Mapping[str, float],
    *,
    tolerance: float = 0.0,
) -> dict[str, bool]:
    """Return the empirical Pareto set (maximize rescue, minimize deviation)."""
    if set(weak_rescue) != set(strong_deviation) or not weak_rescue:
        raise ValueError("Tradeoff mappings must have the same nonempty method set.")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative.")
    values = {
        method: (float(weak_rescue[method]), float(strong_deviation[method]))
        for method in weak_rescue
    }
    if any(not np.isfinite(value) for pair in values.values() for value in pair):
        raise ValueError("Tradeoff values must be finite.")

    frontier: dict[str, bool] = {}
    for method, (rescue, deviation) in values.items():
        dominated = False
        for other, (other_rescue, other_deviation) in values.items():
            if other == method:
                continue
            no_worse = (
                other_rescue >= rescue - tolerance
                and other_deviation <= deviation + tolerance
            )
            strictly_better = (
                other_rescue > rescue + tolerance
                or other_deviation < deviation - tolerance
            )
            if no_worse and strictly_better:
                dominated = True
                break
        frontier[method] = not dominated
    return frontier


def tradeoff_decision(
    weak_rescue_difference_ci: tuple[float, float],
    *,
    weak_equivalence_margin: float,
    strong_deviation_advantage_ci: tuple[float, float],
) -> TradeoffDecision:
    """Apply the frozen intersection-union rule for a CDC-vs-style comparison.

    `weak_rescue_difference_ci` is CDC minus comparator, with larger rescue better.
    It must lie wholly inside `[-margin,+margin]`. The strong quantity is comparator
    absolute deviation minus CDC absolute deviation, so a wholly positive interval
    means CDC preserves the strong trajectory better. CDC wins the joint tradeoff
    only if both conditions hold.
    """
    weak_low, weak_high = map(float, weak_rescue_difference_ci)
    strong_low, strong_high = map(float, strong_deviation_advantage_ci)
    values = (weak_low, weak_high, strong_low, strong_high, weak_equivalence_margin)
    if any(not np.isfinite(value) for value in values):
        raise ValueError("Tradeoff confidence limits and margin must be finite.")
    if weak_low > weak_high or strong_low > strong_high:
        raise ValueError("Confidence interval lower endpoints must not exceed upper endpoints.")
    if weak_equivalence_margin <= 0:
        raise ValueError("weak_equivalence_margin must be positive.")
    weak_equivalent = bool(
        weak_low >= -weak_equivalence_margin
        and weak_high <= weak_equivalence_margin
    )
    strong_superior = bool(strong_low > 0.0)
    return TradeoffDecision(
        weak_rescue_equivalent=weak_equivalent,
        strong_preservation_superior=strong_superior,
        joint_tradeoff_superior=bool(weak_equivalent and strong_superior),
    )
