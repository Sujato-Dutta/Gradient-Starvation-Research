import math

import numpy as np
import pytest

from gradient_starvation.metrics import (
    causal_metrics,
    first_hitting_time,
    gsi5,
    n_sign_changes,
    sign_crossing_time,
)


def test_hitting_time_interpolates_and_causal_delay_is_positive():
    times = np.array([0.0, 1.0, 2.0])
    both = np.array([0.0, 0.2, 0.6])
    weak = np.array([0.0, 0.5, 0.8])
    result = causal_metrics(times, both, weak, beta=0.4)
    assert result.delta_tw > 0
    assert not result.right_censored
    assert first_hitting_time(times, both, 2.0) == math.inf


def test_gsi5_is_zero_for_uniform_weights_and_positive_for_concentrated_weights():
    assert abs(gsi5(np.ones(10))) < 1e-12
    assert gsi5(np.array([1.0, 0.0, 0.0, 0.0])) > 0.7


def test_sign_crossing_time_interpolates_the_first_descending_crossing():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([2.0, 1.0, -1.0, -2.0])
    # Crossing lies midway between tau=1 (value 1) and tau=2 (value -1).
    assert sign_crossing_time(times, values) == pytest.approx(1.5)


def test_sign_crossing_time_returns_infinity_when_it_never_crosses():
    times = np.array([0.0, 1.0, 2.0])
    monotone_positive = np.array([1.0, 2.0, 3.0])
    assert sign_crossing_time(times, monotone_positive) == math.inf
    # A - to + crossing is not a + to - crossing.
    assert sign_crossing_time(times, np.array([-1.0, 0.5, 2.0])) == math.inf


def test_sign_crossing_time_returns_nan_when_undecidable():
    """'No data' must be distinguishable from 'never crosses'."""
    assert math.isnan(sign_crossing_time(np.array([]), np.array([])))
    assert math.isnan(sign_crossing_time(np.array([0.0]), np.array([1.0])))
    # Non-finite entries are dropped, which can leave too few samples.
    assert math.isnan(
        sign_crossing_time(np.array([0.0, 1.0]), np.array([1.0, np.nan]))
    )


def test_sign_crossing_time_handles_exact_zeros_and_direction():
    times = np.array([0.0, 1.0, 2.0])
    # +, 0, - crosses at the zero itself.
    assert sign_crossing_time(times, np.array([2.0, 0.0, -2.0])) == pytest.approx(1.0)
    # +, 0, + only touches zero and does not cross.
    assert sign_crossing_time(times, np.array([2.0, 0.0, 2.0])) == math.inf
    # The ascending direction finds the mirrored transition.
    assert sign_crossing_time(
        times, np.array([-1.0, 1.0, 2.0]), descending=False
    ) == pytest.approx(0.5)


def test_sign_crossing_time_sorts_by_time_and_validates_shapes():
    shuffled_times = np.array([2.0, 0.0, 1.0])
    shuffled_values = np.array([-1.0, 2.0, 1.0])
    assert sign_crossing_time(shuffled_times, shuffled_values) == pytest.approx(1.5)
    with pytest.raises(ValueError, match="same shape"):
        sign_crossing_time(np.array([0.0, 1.0]), np.array([1.0]))


def test_sign_crossing_time_reports_only_the_first_of_several_crossings():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([1.0, -1.0, 1.0, -1.0])
    assert sign_crossing_time(times, values) == pytest.approx(0.5)
    assert n_sign_changes(values) == 3


def test_n_sign_changes_ignores_zeros_and_non_finite_entries():
    assert n_sign_changes(np.array([1.0, 2.0, 3.0])) == 0
    assert n_sign_changes(np.array([1.0, 0.0, 1.0])) == 0
    assert n_sign_changes(np.array([1.0, 0.0, -1.0])) == 1
    assert n_sign_changes(np.array([1.0, np.nan, -1.0])) == 1
    assert n_sign_changes(np.array([])) == 0
    assert n_sign_changes(np.array([5.0])) == 0
