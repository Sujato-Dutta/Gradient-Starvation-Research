"""Independent, piecewise-linear audit of paired weak-response trajectories.

These are *observed-grid* diagnostics. Interpolation does not certify what an
unobserved continuous trajectory did between checkpoints. In particular, an
AUC deficit, a threshold delay, and a gap at the W first hit are distinct.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PairedResponseAudit:
    weak_hit: float
    both_hit: float
    weak_gate: bool
    degenerate_target: bool
    initial_gap: float
    initial_match: bool
    gap_at_weak_hit: float
    at_hit_certificate: bool | None
    any_time_certificate: bool | None
    any_time_negative_gap: bool
    signed_weak_auc_deficit: float
    positive_part_deficit_area: float
    minimum_gap: float

    def as_dict(self) -> dict[str, float | bool | None]:
        return asdict(self)


def _first_hit(times: np.ndarray, values: np.ndarray, beta: float) -> float:
    """First hit on the recorded polygonal trajectory, including exact knots."""
    if values[0] >= beta:
        return float(times[0])
    for index in range(1, len(times)):
        if values[index] >= beta:
            before, after = float(values[index - 1]), float(values[index])
            fraction = (beta - before) / (after - before)
            return float(times[index - 1] + fraction * (times[index] - times[index - 1]))
    return math.inf


def _positive_linear_area(left: float, right: float, width: float) -> float:
    """Exact area above zero of a line segment with endpoint values."""
    if left >= 0 and right >= 0:
        return width * (left + right) / 2
    if left <= 0 and right <= 0:
        return 0.0
    fraction = -left / (right - left)
    if left > 0:
        return width * fraction * left / 2
    return width * (1 - fraction) * right / 2


def audit_paired_response(
    times: np.ndarray,
    both_values: np.ndarray,
    weak_values: np.ndarray,
    *,
    beta: float,
    gap_tolerance: float = 1e-6,
    initial_tolerance: float = 1e-5,
) -> PairedResponseAudit:
    """Compute separate at-hit, any-time, delay, and area evidence.

    ``None`` means a certificate cannot be judged because W fails its
    nondegenerate gate or the two response functions differ at initialization.
    A failed W gate is never a negative finding about starvation.
    """
    t = np.asarray(times, dtype=np.float64)
    b = np.asarray(both_values, dtype=np.float64)
    w = np.asarray(weak_values, dtype=np.float64)
    if t.ndim != 1 or len(t) < 2 or b.shape != t.shape or w.shape != t.shape:
        raise ValueError("Paired times and responses must be equal-length 1D arrays of length >=2.")
    if not (np.isfinite(t).all() and np.isfinite(b).all() and np.isfinite(w).all()):
        raise ValueError("Paired times and responses must be finite.")
    if not np.all(np.diff(t) > 0):
        raise ValueError("Paired times must be strictly increasing.")
    if not (np.isfinite(beta) and np.isfinite(gap_tolerance) and np.isfinite(initial_tolerance)):
        raise ValueError("Beta and tolerances must be finite.")
    if gap_tolerance < 0 or initial_tolerance < 0:
        raise ValueError("Tolerances must be non-negative.")

    gap = b - w
    initial_gap = float(gap[0])
    initial_match = abs(initial_gap) <= initial_tolerance
    weak_hit = _first_hit(t, w, beta)
    both_hit = _first_hit(t, b, beta)
    degenerate = bool(beta <= w[0])
    gate = bool(math.isfinite(weak_hit) and weak_hit > t[0] and not degenerate)
    gap_at_hit = float(np.interp(weak_hit, t, gap)) if gate else math.nan
    any_negative = bool(np.any(gap[1:] < -gap_tolerance))
    assessable = gate and initial_match
    widths = np.diff(t)
    deficit = -gap
    signed_area = float(np.sum(widths * (deficit[:-1] + deficit[1:]) / 2))
    positive_area = float(sum(
        _positive_linear_area(float(deficit[i]), float(deficit[i + 1]), float(widths[i]))
        for i in range(len(widths))
    ))
    return PairedResponseAudit(
        weak_hit=weak_hit,
        both_hit=both_hit,
        weak_gate=gate,
        degenerate_target=degenerate,
        initial_gap=initial_gap,
        initial_match=initial_match,
        gap_at_weak_hit=gap_at_hit,
        at_hit_certificate=bool(gap_at_hit < -gap_tolerance) if assessable else None,
        any_time_certificate=any_negative if assessable else None,
        any_time_negative_gap=any_negative,
        signed_weak_auc_deficit=signed_area,
        positive_part_deficit_area=positive_area,
        minimum_gap=float(np.min(gap[1:])),
    )
