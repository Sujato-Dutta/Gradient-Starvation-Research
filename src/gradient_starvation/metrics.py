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

