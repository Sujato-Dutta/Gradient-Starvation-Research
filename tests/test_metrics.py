import math

import numpy as np

from gradient_starvation.metrics import causal_metrics, first_hitting_time, gsi5


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

