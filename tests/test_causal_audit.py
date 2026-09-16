import math

import numpy as np
import pytest

from gradient_starvation.causal_audit import audit_paired_response


def audit(both, weak, beta=0.5):
    return audit_paired_response(
        np.arange(len(both), dtype=float), np.asarray(both, dtype=float),
        np.asarray(weak, dtype=float), beta=beta, gap_tolerance=0,
    )


def test_positive_auc_and_gate_do_not_imply_at_hit_certificate():
    result = audit([0, 0.6, 0.2], [0, 0.5, 0.8])
    assert result.weak_gate
    assert result.gap_at_weak_hit == pytest.approx(0.1)
    assert result.signed_weak_auc_deficit > 0
    assert result.any_time_certificate is True
    assert result.at_hit_certificate is False


def test_suppression_throughout_is_not_excluded_by_lack_of_crossing():
    result = audit([0, 0.1, 0.2], [0, 0.6, 0.9])
    assert result.at_hit_certificate is True
    assert result.any_time_certificate is True
    assert result.positive_part_deficit_area == pytest.approx(result.signed_weak_auc_deficit)


def test_failed_gate_is_indeterminate_even_if_gap_is_negative():
    result = audit([0, 0.1, 0.2], [0, 0.2, 0.3])
    assert not result.weak_gate
    assert result.any_time_negative_gap
    assert result.at_hit_certificate is None
    assert result.any_time_certificate is None
    assert math.isnan(result.gap_at_weak_hit)


def test_degenerate_target_and_mismatched_initial_response_abstain():
    degenerate = audit([0.5, 0.6], [0.5, 0.7])
    assert degenerate.degenerate_target
    assert degenerate.at_hit_certificate is None
    mismatch = audit([0.01, 0.1, 0.2], [0, 0.6, 0.8])
    assert mismatch.weak_gate and not mismatch.initial_match
    assert mismatch.at_hit_certificate is None


def test_piecewise_positive_area_splits_a_zero_crossing():
    result = audit([0, 1, -1], [0, 0, 0], beta=2)
    assert result.signed_weak_auc_deficit == pytest.approx(-0.5)
    assert result.positive_part_deficit_area == pytest.approx(0.25)


@pytest.mark.parametrize("times,both,weak,beta", [
    ([0], [0], [0], 1),
    ([0, 0], [0, 1], [0, 1], 1),
    ([0, 1], [0, float("nan")], [0, 1], 1),
    ([0, 1], [0, 1], [0], 1),
    ([0, 1], [0, 1], [0, 1], float("nan")),
])
def test_invalid_inputs_fail_closed(times, both, weak, beta):
    with pytest.raises(ValueError):
        audit_paired_response(times, both, weak, beta=beta)
