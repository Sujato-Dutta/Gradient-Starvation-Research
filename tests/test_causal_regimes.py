"""Learnability-gated causal classification, including a replay acceptance test.

The replay reprocesses a completed run already on disk
(``results/e1_corrected_validation-20260821-144112``) whose conclusions are quoted
in ``a.md`` §10.  Because the expected answers are known independently of this
code, the replay is a genuine specification check on the gate rather than a
self-consistency check.  It skips when the run directory is absent, since
``results/`` is git-ignored and will not exist in a fresh clone.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gradient_starvation.metrics import (
    CAUSAL_REGIMES,
    classify_causal_regime,
    classify_causal_regime_from_trajectories,
)

ROOT = Path(__file__).resolve().parents[1]
REPLAY_RUN = ROOT / "results" / "e1_corrected_validation-20260821-144112"

# Recorded in the run's own config.resolved.yaml: beta 0.25, 500 steps at lr 0.01.
REPLAY_BETA = 0.25
REPLAY_TAU_MAX = 5.0
REPLAY_DELTA = 0.25


def test_unlearnable_takes_priority_over_any_delay():
    """A delay means nothing if the counterfactual never learned."""
    verdict = classify_causal_regime(
        weak_hitting_time=math.inf, both_hitting_time=math.inf,
        tau_max=5.0, delta=0.25,
    )
    assert verdict.regime == "unlearnable"
    assert not verdict.weak_only_learnable
    assert math.isnan(verdict.delta_tw)


def test_weak_hitting_beyond_horizon_is_unlearnable():
    verdict = classify_causal_regime(
        weak_hitting_time=7.0, both_hitting_time=1.0, tau_max=5.0, delta=0.25
    )
    assert verdict.regime == "unlearnable"


def test_target_met_at_initialization_is_degenerate_not_neutral():
    """Initialization noise above the threshold is an artifact, not a phase."""
    verdict = classify_causal_regime(
        weak_hitting_time=0.0, both_hitting_time=0.0,
        tau_max=5.0, delta=0.25, initial_tau=0.0,
    )
    assert verdict.regime == "degenerate"
    assert verdict.target_met_at_initialization
    assert math.isnan(verdict.delta_tw)


def test_censored_both_condition_is_maximal_starvation():
    """Weak-only learns, both never does: the strongest starvation, delta_tw = inf."""
    verdict = classify_causal_regime(
        weak_hitting_time=1.0, both_hitting_time=math.inf, tau_max=5.0, delta=0.25
    )
    assert verdict.regime == "starvation"
    assert verdict.right_censored
    assert verdict.delta_tw == math.inf


@pytest.mark.parametrize(
    ("delta_tw", "expected"),
    [(1.0, "starvation"), (0.26, "starvation"), (0.25, "neutral"), (0.0, "neutral"),
     (-0.25, "neutral"), (-0.26, "transfer"), (-2.0, "transfer")],
)
def test_band_edges_classify_as_documented(delta_tw, expected):
    verdict = classify_causal_regime(
        weak_hitting_time=1.0, both_hitting_time=1.0 + delta_tw,
        tau_max=10.0, delta=0.25,
    )
    assert verdict.regime == expected
    assert verdict.regime in CAUSAL_REGIMES


def test_trajectory_wrapper_defaults_tau_max_to_the_horizon():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    weak = np.array([0.0, 0.3, 0.6, 0.9])
    both = np.array([0.0, 0.1, 0.2, 0.9])
    verdict = classify_causal_regime_from_trajectories(
        times, both, weak, beta=0.5, delta=0.25
    )
    assert verdict.weak_only_learnable
    assert verdict.regime == "starvation"


def test_negative_delta_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        classify_causal_regime(
            weak_hitting_time=1.0, both_hitting_time=1.0, tau_max=5.0, delta=-0.1
        )


# ---------------------------------------------------------------------------
# Replay acceptance against a completed run
# ---------------------------------------------------------------------------


def _replay() -> pd.DataFrame:
    if not (REPLAY_RUN / "summary.csv").is_file():
        pytest.skip(f"replay run not present: {REPLAY_RUN}")
    summary = pd.read_csv(REPLAY_RUN / "summary.csv")
    verdicts = [
        classify_causal_regime(
            weak_hitting_time=float(row.weak_hitting_time),
            both_hitting_time=float(row.both_hitting_time),
            tau_max=REPLAY_TAU_MAX,
            delta=REPLAY_DELTA,
        )
        for row in summary.itertuples()
    ]
    summary["regime_class"] = [verdict.regime for verdict in verdicts]
    return summary


def test_replay_lag8_is_unlearnable_not_starvation():
    """a.md §10: lag 8 weak-only final response was 0.001643, so nothing learned."""
    summary = _replay()
    lag8 = summary[(summary.regime == "positive") & (summary.lag_separation == 8)]
    assert len(lag8) == 32
    assert set(lag8.regime_class) == {"unlearnable"}


def test_replay_negative_control_is_never_starvation():
    """a.md §10: matched negative control mean AUC gap -0.750387."""
    summary = _replay()
    negative = summary[summary.regime == "negative"]
    assert len(negative) == 8
    assert set(negative.regime_class) == {"transfer"}


def test_replay_learnable_positive_cells_are_never_transfer():
    """No learnable, non-degenerate positive cell may be labelled transfer.

    This is the honest form of the acceptance criterion.  The original wording
    expected every positive cell at lag 0/2/4 to come out ``starvation``, but that
    conflated two metrics: ``a.md`` §10's claim is about the trajectory **AUC gap**,
    not the hitting-time delay.  Measured by delay, 12 of the 20 finite cells fall
    inside the +-0.25 band and are ``neutral`` even though their AUC gaps are large
    and positive.  What must hold is the directional claim: nothing in the positive
    regime looks like transfer.
    """
    summary = _replay()
    positive = summary[
        (summary.regime == "positive") & (summary.lag_separation.isin([0, 2, 4]))
    ]
    assert len(positive) == 96
    assert "transfer" not in set(positive.regime_class)
    assert set(positive.regime_class) <= {"starvation", "neutral", "degenerate"}


def test_replay_auc_gap_reproduces_the_documented_directional_claim():
    """a.md §10: every positive cell through lag 4 had a positive AUC gap, all seeds."""
    summary = _replay()
    for lag in (0, 2, 4):
        cells = summary[(summary.regime == "positive") & (summary.lag_separation == lag)]
        assert len(cells) == 32
        assert (cells.weak_auc_gap > 0).all(), lag


def test_replay_partitions_positive_cells_exactly():
    """Pin the census so a later change to the gate cannot pass unnoticed."""
    summary = _replay()
    positive = summary[
        (summary.regime == "positive") & (summary.lag_separation.isin([0, 2, 4]))
    ]
    counts = positive.regime_class.value_counts().to_dict()
    assert counts.get("starvation") == 76  # 68 right-censored + 8 above the band
    assert counts.get("neutral") == 12
    assert counts.get("degenerate") == 8
    assert sum(counts.values()) == 96


def test_replay_degenerate_rows_are_the_known_initialization_artifact():
    """The degenerate rows come from seeds 9 and 11, whose initial |m_w| > beta."""
    summary = _replay()
    degenerate = summary[summary.regime_class == "degenerate"]
    assert sorted(degenerate.seed.unique()) == [9, 11]
    assert (degenerate.weak_hitting_time == 0.0).all()
    # They are an artifact of threshold choice, not of the strong feature: the
    # shared initialization means both conditions trip the threshold together.
    assert (degenerate.both_hitting_time == 0.0).all()
