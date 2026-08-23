from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


def gsi5(sigmoid_weights: np.ndarray) -> float:
    weights = np.asarray(sigmoid_weights, dtype=np.float64)
    if weights.size == 0:
        return float("nan")
    denominator = np.square(weights).sum()
    if denominator <= np.finfo(float).tiny:
        return 1.0
    effective = weights.sum() ** 2 / denominator
    return float(np.clip(1.0 - effective / len(weights), 0.0, 1.0))


def first_hitting_time(times: np.ndarray, values: np.ndarray, target: float) -> float:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    indices = np.flatnonzero(values >= target)
    if not len(indices):
        return float("inf")
    index = int(indices[0])
    if index == 0 or values[index] == values[index - 1]:
        return float(times[index])
    fraction = (target - values[index - 1]) / (values[index] - values[index - 1])
    return float(times[index - 1] + fraction * (times[index] - times[index - 1]))


def weak_trajectory_auc_gap(
    times: np.ndarray, both_values: np.ndarray, weak_values: np.ndarray
) -> float:
    return float(np.trapezoid(np.asarray(weak_values) - np.asarray(both_values), np.asarray(times)))


@dataclass(frozen=True)
class CausalMetrics:
    both_hitting_time: float
    weak_hitting_time: float
    delta_tw: float
    right_censored: bool
    weak_auc_gap: float


def causal_metrics(
    times: np.ndarray,
    both_values: np.ndarray,
    weak_values: np.ndarray,
    beta: float,
) -> CausalMetrics:
    both_time = first_hitting_time(times, both_values, beta)
    weak_time = first_hitting_time(times, weak_values, beta)
    censored = not (math.isfinite(both_time) and math.isfinite(weak_time))
    delta = both_time - weak_time if not censored else float("nan")
    return CausalMetrics(
        both_hitting_time=both_time,
        weak_hitting_time=weak_time,
        delta_tw=delta,
        right_censored=censored,
        weak_auc_gap=weak_trajectory_auc_gap(times, both_values, weak_values),
    )


CAUSAL_REGIMES = ("unlearnable", "degenerate", "transfer", "neutral", "starvation")


@dataclass(frozen=True)
class CausalRegimeVerdict:
    """Learnability-gated classification of one paired run."""

    regime: str
    weak_only_learnable: bool
    target_met_at_initialization: bool
    weak_hitting_time: float
    both_hitting_time: float
    delta_tw: float
    right_censored: bool


def classify_causal_regime(
    *,
    weak_hitting_time: float,
    both_hitting_time: float,
    tau_max: float,
    delta: float,
    initial_tau: float = 0.0,
) -> CausalRegimeVerdict:
    """Classify a paired run only after establishing counterfactual learnability.

    A positive delay means nothing unless the weak feature demonstrably learns in
    the weak-only counterfactual.  The gate is applied first, and ``delta_tw`` is
    left undefined outside the learnable region.

    Regimes
    -------
    ``unlearnable``
        The weak-only condition never reaches the target within ``tau_max``.
        There is no counterfactual learning to suppress, so the point is
        indeterminate rather than starved.
    ``degenerate``
        The target was already satisfied at ``initial_tau``, before any
        optimization step.  This is a measurement artifact -- initialization noise
        exceeding the threshold -- not a phase.  It is bucketed separately so it
        cannot be silently read as "learned instantly" or "neutral".
    ``starvation``
        ``delta_tw > delta``.  This includes the right-censored case where the
        both-feature condition never reaches the target while weak-only does,
        which is the strongest form of starvation and carries ``delta_tw = inf``.
    ``neutral``
        ``abs(delta_tw) <= delta``.
    ``transfer``
        ``delta_tw < -delta``.  The strong feature accelerates weak learning.

    Note that ``delta_tw`` at a single threshold is a far less sensitive detector
    than the trajectory AUC gap: at small ``beta`` both conditions can cross the
    threshold almost immediately and diverge only afterwards.  Report both.
    """
    if delta < 0:
        raise ValueError("delta must be non-negative.")
    weak_learnable = math.isfinite(weak_hitting_time) and weak_hitting_time <= tau_max
    if not weak_learnable:
        return CausalRegimeVerdict(
            regime="unlearnable",
            weak_only_learnable=False,
            target_met_at_initialization=False,
            weak_hitting_time=weak_hitting_time,
            both_hitting_time=both_hitting_time,
            delta_tw=float("nan"),
            right_censored=True,
        )
    if weak_hitting_time <= initial_tau:
        return CausalRegimeVerdict(
            regime="degenerate",
            weak_only_learnable=True,
            target_met_at_initialization=True,
            weak_hitting_time=weak_hitting_time,
            both_hitting_time=both_hitting_time,
            delta_tw=float("nan"),
            right_censored=False,
        )
    censored = not math.isfinite(both_hitting_time)
    delta_tw = float("inf") if censored else both_hitting_time - weak_hitting_time
    if delta_tw > delta:
        regime = "starvation"
    elif delta_tw < -delta:
        regime = "transfer"
    else:
        regime = "neutral"
    return CausalRegimeVerdict(
        regime=regime,
        weak_only_learnable=True,
        target_met_at_initialization=False,
        weak_hitting_time=weak_hitting_time,
        both_hitting_time=both_hitting_time,
        delta_tw=delta_tw,
        right_censored=censored,
    )


def classify_causal_regime_from_trajectories(
    times: np.ndarray,
    both_values: np.ndarray,
    weak_values: np.ndarray,
    *,
    beta: float,
    tau_max: float | None = None,
    delta: float,
) -> CausalRegimeVerdict:
    """Convenience wrapper computing the hitting times before classifying."""
    times = np.asarray(times, dtype=float)
    if len(times) == 0:
        raise ValueError("times must not be empty.")
    return classify_causal_regime(
        weak_hitting_time=first_hitting_time(times, weak_values, beta),
        both_hitting_time=first_hitting_time(times, both_values, beta),
        tau_max=float(times[-1]) if tau_max is None else float(tau_max),
        delta=delta,
        initial_tau=float(times[0]),
    )


def sign_crossing_time(
    times: np.ndarray, values: np.ndarray, *, descending: bool = True
) -> float:
    """Return the first sign crossing of ``values``, linearly interpolated.

    With ``descending=True`` (the default) this locates the first ``+ -> -``
    transition, which for the causal weak-drift difference ``d_w`` is the
    transfer-to-starvation crossover time ``tau*``.  Set ``descending=False`` to
    locate the first ``- -> +`` transition instead, which is what a
    starvation-to-transfer architecture would show.

    Return values are deliberately distinguishable:

    ``nan``
        Not decidable -- fewer than two finite samples were supplied.
    ``inf``
        Decidable and the answer is "never": the series has two or more finite
        samples and does not cross in the requested direction.
    finite float
        The interpolated crossing time.

    Exact zeros are treated as neither positive nor negative and are skipped, so
    a ``+, 0, -`` series crosses once, at the zero, while ``+, 0, +`` does not
    cross at all.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.shape != values.shape:
        raise ValueError(
            f"times and values must have the same shape; received {times.shape} "
            f"and {values.shape}."
        )
    keep = np.isfinite(times) & np.isfinite(values)
    times, values = times[keep], values[keep]
    if len(times) < 2:
        return float("nan")
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]

    signs = np.sign(values)
    before, after = (1.0, -1.0) if descending else (-1.0, 1.0)
    previous: int | None = None
    for index in range(len(values)):
        if signs[index] == 0:
            continue
        if previous is not None and signs[previous] == before and signs[index] == after:
            t0, t1 = times[previous], times[index]
            v0, v1 = values[previous], values[index]
            span = v0 - v1
            if span == 0:  # pragma: no cover - unreachable for opposite signs
                return float(t0)
            return float(t0 + (t1 - t0) * v0 / span)
        previous = index
    return float("inf")


def n_sign_changes(values: np.ndarray) -> int:
    """Count sign changes in ``values``, ignoring exact zeros and non-finite entries.

    A trajectory theorem that permits the drift-deficit sign to change must also
    admit more than one change.  Reporting the count alongside the first crossing
    keeps a multi-crossing series from being summarized as a single clean
    transition.
    """
    values = np.asarray(values, dtype=float)
    signs = np.sign(values[np.isfinite(values)])
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return 0
    return int(np.count_nonzero(np.diff(nonzero) != 0))


def mean_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(finite.mean())
    if len(finite) == 1:
        return mean, float("nan"), float("nan")
    sem = stats.sem(finite)
    half_width = float(stats.t.ppf((1 + confidence) / 2, len(finite) - 1) * sem)
    return mean, mean - half_width, mean + half_width

