from __future__ import annotations

import math

import pytest

from gradient_starvation.study_metrics import (
    SHADOW_METHOD_PUBLICATION_LABELS,
    first_hitting_profile,
    pareto_frontier,
    tradeoff_decision,
)


def test_first_hit_profile_uses_first_crossing_not_terminal_thresholding():
    points = first_hitting_profile(
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 0.4, 0.2, 0.8],
        [0.0, 0.6, 0.1, 1.2],
        [0.25, 0.5, 1.0],
        delay_tolerance=0.1,
        outcome_suppression_certified=True,
    )
    assert [point.beta for point in points] == [0.25, 0.5, 1.0]
    # beta=.5 is first reached on the early upswing, despite both paths falling later.
    assert points[1].weak_hitting_time == pytest.approx(5.0 / 6.0)
    assert points[1].both_hitting_time == pytest.approx(2.5)
    assert points[1].regime == "starvation"
    assert points[1].causal_starvation_certified
    # BOTH never reaches beta=1, while WEAK does: right-censored maximal delay.
    assert math.isinf(points[2].both_hitting_time)
    assert points[2].weak_hitting_time == pytest.approx(2.8181818181818183)
    assert points[2].right_censored


def test_first_hit_profile_keeps_initial_targets_degenerate():
    point = first_hitting_profile(
        [0.0, 1.0],
        [0.2, 0.3],
        [0.2, 0.4],
        [0.1],
        delay_tolerance=0.01,
        outcome_suppression_certified=True,
    )[0]
    assert point.target_met_at_initialization
    assert not point.weak_only_learnable
    assert point.regime == "degenerate"
    assert not point.causal_starvation_certified


@pytest.mark.parametrize("betas", [[0.5, 0.5], [1.0, 0.5], []])
def test_first_hit_profile_rejects_nonprospective_beta_grids(betas):
    with pytest.raises(ValueError, match="beta_values"):
        first_hitting_profile(
            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], betas, delay_tolerance=0.1
        )


def test_pareto_frontier_handles_dominance_and_ties():
    frontier = pareto_frontier(
        {"cdc": 1.0, "bloop": 1.0, "pcgrad": 0.8, "tie": 1.0},
        {"cdc": 0.1, "bloop": 0.3, "pcgrad": 0.05, "tie": 0.1},
    )
    assert frontier == {
        "cdc": True,
        "bloop": False,
        "pcgrad": True,
        "tie": True,
    }


def test_tradeoff_superiority_requires_equivalent_rescue_and_better_preservation():
    accepted = tradeoff_decision(
        (-0.01, 0.02),
        weak_equivalence_margin=0.05,
        strong_deviation_advantage_ci=(0.1, 0.3),
    )
    assert accepted.weak_rescue_equivalent
    assert accepted.strong_preservation_superior
    assert accepted.joint_tradeoff_superior

    rejected = tradeoff_decision(
        (-0.01, 0.02),
        weak_equivalence_margin=0.05,
        strong_deviation_advantage_ci=(-0.01, 0.3),
    )
    assert rejected.weak_rescue_equivalent
    assert not rejected.strong_preservation_superior
    assert not rejected.joint_tradeoff_superior


def test_publication_labels_do_not_claim_canonical_baseline_fidelity():
    assert SHADOW_METHOD_PUBLICATION_LABELS == {
        "counterfactual_drift": "CDC",
        "bloop": "Bloop-style shadow-target rescue",
        "pcgrad": "PCGrad-style shadow-target rescue",
    }
