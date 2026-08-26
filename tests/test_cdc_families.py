"""The CDC correction families, and the empirical Result 3 scaling.

The families share one causal target -- the matched weak-only shadow drift -- and
differ only in what they protect.  That is what makes the comparison an ablation of
the *constraint* rather than of the objective.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gradient_starvation.data.synthetic import SyntheticTaskSpec, make_paired_task
from gradient_starvation.losses import SHADOW_REQUIRING_METHODS, training_objective
from gradient_starvation.models.recurrent import build_model
from gradient_starvation.theory import (
    CORRECTION_CONSTRAINTS,
    counterfactual_drift_correction,
    drift_correction,
    finite_step_deviation_sweep,
    projected_statistics,
)
from gradient_starvation.training import train_paired

DENSE = {"kind": "dense_linear", "width": 24, "bulk_gain": 0.5}


def _state_with_deficit(seed: int = 5, *, deficit: float = 0.5):
    """A parameter state plus a weak-drift target strictly above the ERM drift."""
    spec = SyntheticTaskSpec(
        sequence_length=8, n_samples=256, rho=4, lag_separation=2, cue_noise=0.1
    )
    both, _ = make_paired_task(spec, seed=seed)
    torch.manual_seed(1)
    model = build_model(DENSE)
    baseline = projected_statistics(model, both, compute_direct_drift=True)
    assert baseline.direct_drift is not None
    return model, both, baseline.direct_drift[1] + deficit


@pytest.mark.parametrize("constraint", CORRECTION_CONSTRAINTS)
def test_every_family_reaches_the_shared_causal_target(constraint):
    """The ablation isolates the constraint, so the target must be common."""
    model, both, target = _state_with_deficit()
    correction = drift_correction(model, both, target, constraint=constraint)
    assert correction.feasible
    assert float(correction.weak_drift_after) >= float(correction.target_weak_drift) - 1e-5
    assert float(correction.correction_norm) > 0


def test_only_the_strong_response_constraint_preserves_the_strong_drift():
    """This is the whole distinctness claim, and it is measurable.

    Result 1 is exact for CDC. Projecting against grad(L_train) instead -- what
    generic gradient surgery protects -- leaves a large strong-drift perturbation.
    """
    model, both, target = _state_with_deficit()
    changes = {}
    for constraint in CORRECTION_CONSTRAINTS:
        correction = drift_correction(model, both, target, constraint=constraint)
        changes[constraint] = abs(
            float(correction.strong_drift_after - correction.strong_drift_before)
        )

    assert changes["strong_response"] < 1e-6
    for constraint in ("loss_gradient", "bloop", "unconstrained", "pcgrad"):
        assert changes[constraint] > 100 * max(changes["strong_response"], 1e-12), (
            f"{constraint} perturbed the strong drift by {changes[constraint]:.3e}, "
            "which is not clearly worse than CDC; the ablation would be uninformative"
        )
    # The loss-gradient family, which is the generic-surgery object, should be the
    # most damaging to the protected feature response.
    assert changes["loss_gradient"] > changes["unconstrained"]


def test_drift_correction_matches_the_original_cdc_implementation():
    """`counterfactual_drift_correction` must stay the CDC special case."""
    model, both, target = _state_with_deficit()
    legacy = counterfactual_drift_correction(model, both, target)
    generalized = drift_correction(model, both, target, constraint="strong_response")
    for field in (
        "weak_drift_before",
        "weak_drift_after",
        "strong_drift_before",
        "strong_drift_after",
        "alpha",
        "uncapped_alpha",
        "deficit",
        "target_residual",
    ):
        torch.testing.assert_close(
            getattr(legacy, field), getattr(generalized, field), rtol=1e-6, atol=1e-9
        )
    assert legacy.cap_binding is generalized.cap_binding is False


def test_cdc_euclidean_solution_is_not_invariant_to_diagonal_rescaling():
    """A non-isometric coordinate change alters the pushed-forward correction.

    This 3D linear-algebra example isolates the metric issue: each coordinate
    system computes its own feasible minimum-Euclidean-norm CDC correction, but a
    diagonal rescaling does not push one solution to the other.
    """
    strong = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
    weak = torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64)
    rescaling = torch.diag(torch.tensor([2.0, 0.5, 3.0], dtype=torch.float64))
    deficit = torch.tensor(0.7, dtype=torch.float64)

    def euclidean_cdc(a, b):
        protected = b - (torch.dot(b, a) / torch.dot(a, a)) * a
        gain = torch.dot(b, protected)
        assert gain > 0  # The solution is feasible in this coordinate system.
        return deficit * protected / gain

    correction = euclidean_cdc(strong, weak)
    strong_rescaled = rescaling.T @ strong
    weak_rescaled = rescaling.T @ weak
    correction_rescaled = euclidean_cdc(strong_rescaled, weak_rescaled)
    pushed_forward = rescaling @ correction_rescaled

    zero = deficit.new_zeros(())
    for candidate in (correction, pushed_forward):
        torch.testing.assert_close(
            torch.dot(strong, candidate), zero, atol=1e-14, rtol=0
        )
        torch.testing.assert_close(
            torch.dot(weak, candidate), deficit, atol=1e-14, rtol=0
        )
    assert not torch.allclose(correction, pushed_forward, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(
    "correction_function", [counterfactual_drift_correction, drift_correction]
)
def test_correction_reports_a_binding_alpha_cap(correction_function):
    model, both, target = _state_with_deficit()
    correction = correction_function(model, both, target, max_alpha=0.0)
    assert correction.feasible
    assert correction.cap_binding
    assert float(correction.uncapped_alpha) > 0.0
    assert float(correction.alpha) == 0.0
    torch.testing.assert_close(
        correction.target_residual, correction.deficit, rtol=1e-6, atol=1e-9
    )


@pytest.mark.parametrize(
    "correction_function", [counterfactual_drift_correction, drift_correction]
)
@pytest.mark.parametrize("feasibility_epsilon", [-1.0, float("nan"), float("inf")])
def test_correction_rejects_invalid_feasibility_tolerance(
    correction_function, feasibility_epsilon
):
    model, both, target = _state_with_deficit()
    with pytest.raises(ValueError, match="feasibility_epsilon"):
        correction_function(
            model, both, target, feasibility_epsilon=feasibility_epsilon
        )


@pytest.mark.parametrize(
    "correction_function", [counterfactual_drift_correction, drift_correction]
)
@pytest.mark.parametrize("max_alpha", [-1.0, float("nan"), float("inf")])
def test_correction_rejects_invalid_alpha_cap(correction_function, max_alpha):
    model, both, target = _state_with_deficit()
    with pytest.raises(ValueError, match="max_alpha"):
        correction_function(model, both, target, max_alpha=max_alpha)


def test_unknown_constraint_is_rejected():
    model, both, target = _state_with_deficit()
    with pytest.raises(ValueError, match="Unknown constraint"):
        drift_correction(model, both, target, constraint="magic")


def test_no_correction_is_applied_without_a_deficit():
    """CDC must be inactive while the strong feature is already helping.

    Connects to the crossover result: before tau*, the both-feature weak drift
    already exceeds its counterfactual, so there is nothing to rescue.
    """
    model, both, target = _state_with_deficit(deficit=-0.5)
    correction = drift_correction(model, both, target, constraint="strong_response")
    assert float(correction.deficit) == 0.0
    assert float(correction.alpha) == 0.0
    assert float(correction.correction_norm) == 0.0


@pytest.mark.parametrize("method", sorted(SHADOW_REQUIRING_METHODS))
def test_shadow_methods_are_rejected_by_the_single_model_objective(method):
    spec = SyntheticTaskSpec(sequence_length=5, n_samples=16, rho=2, lag_separation=1)
    both, _ = make_paired_task(spec, seed=0)
    model = build_model({"kind": "dense_linear", "width": 6, "bulk_gain": 0.3})
    with pytest.raises(ValueError, match="requires paired both/weak-only training"):
        training_objective(model, both, {"method": method})


@pytest.mark.parametrize("method", sorted(SHADOW_REQUIRING_METHODS))
def test_every_shadow_method_trains_and_logs_diagnostics(method):
    spec = SyntheticTaskSpec(sequence_length=6, n_samples=32, rho=3, lag_separation=2)
    both, weak = make_paired_task(spec, seed=7)
    history, states = train_paired(
        {"kind": "dense_linear", "width": 8, "bulk_gain": 0.3},
        both, weak,
        {"steps": 3, "learning_rate": 0.01, "log_every": 1, "full_batch": True,
         "weight_decay": 0.0},
        {"method": method},
        seed=8,
    )
    both_rows = [row for row in history if row["condition"] == "both"]
    assert len(both_rows) == 4
    assert all("cdc_alpha" in row for row in both_rows)
    assert all("cdc_uncapped_alpha" in row for row in both_rows)
    assert all("cdc_cap_binding" in row for row in both_rows)
    assert all("cdc_target_residual" in row for row in both_rows)
    assert all(not row["cdc_cap_binding"] for row in both_rows)
    assert all(row["cdc_target_met"] for row in both_rows)
    assert set(states) == {"both", "weak_only"}
    if method == "counterfactual_drift":
        assert max(abs(row["cdc_strong_drift_change"]) for row in both_rows) < 1e-5


def test_bloop_carries_state_across_steps():
    """The EMA is what distinguishes Bloop from a plain loss-gradient projection."""
    spec = SyntheticTaskSpec(sequence_length=6, n_samples=64, rho=4, lag_separation=2)
    both, weak = make_paired_task(spec, seed=9)
    shared = dict(
        steps=6, learning_rate=0.02, log_every=1, full_batch=True, weight_decay=0.0
    )
    bloop, _ = train_paired(
        DENSE, both, weak, shared, {"method": "bloop"}, seed=10
    )
    plain, _ = train_paired(
        DENSE, both, weak, shared, {"method": "loss_gradient_projection"}, seed=10
    )
    bloop_alpha = [row["cdc_alpha"] for row in bloop if row["condition"] == "both"]
    plain_alpha = [row["cdc_alpha"] for row in plain if row["condition"] == "both"]
    assert bloop_alpha != plain_alpha


# ---------------------------------------------------------------------------
# Result 3: empirical O(eta^2) scaling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_config",
    [DENSE, {"kind": "low_rank_linear", "width": 24, "bulk_gain": 0.5, "rank": 2}],
    ids=["dense_linear", "low_rank_linear"],
)
def test_finite_step_deviation_is_second_order_in_the_learning_rate(model_config):
    """The sweep observes the proved conditional Result 3 scaling.

    This test supplies a setting where the local behavior is quadratic, but the
    sweep itself does not verify the theorem's neighbourhood hypotheses or constants.

    Must run in float64. In float32 the deviation reaches the representation floor
    at small eta and the fitted slope drops to ~1.78, which would look like a
    first-order effect that is really quantization.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        spec = SyntheticTaskSpec(
            sequence_length=10, n_samples=256, rho=4, lag_separation=2, cue_noise=0.1
        )
        both, weak = make_paired_task(spec, seed=3)
        torch.manual_seed(0)
        model = build_model(model_config)
        shadow = build_model(model_config)
        shadow.load_state_dict(model.state_dict())
        main = torch.optim.SGD(model.parameters(), lr=0.01)
        other = torch.optim.SGD(shadow.parameters(), lr=0.01)
        for _ in range(40):
            for network, batch, optimizer in ((model, both, main), (shadow, weak, other)):
                optimizer.zero_grad(set_to_none=True)
                loss, _ = training_objective(network, batch, {"method": "erm"})
                loss.backward()
                optimizer.step()
        shadow_stats = projected_statistics(shadow, weak, compute_direct_drift=True)
        result = finite_step_deviation_sweep(
            model, both, shadow_stats.direct_drift[1],
            [0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125],
        )
        assert result.correction_norm > 0, "no correction applied; nothing to measure"
        assert abs(result.instantaneous_strong_drift_change) < 1e-12
        assert 1.9 <= result.log_log_slope <= 2.1, result.log_log_slope
        ratios = np.array(result.deviations) / np.array(result.learning_rates) ** 2
        assert ratios.max() / ratios.min() < 1.05
    finally:
        torch.set_default_dtype(previous)


def test_finite_step_sweep_restores_the_parameters_it_probes():
    model, both, target = _state_with_deficit()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    finite_step_deviation_sweep(model, both, target, [0.01, 0.005])
    for parameter, original in zip(model.parameters(), before):
        assert torch.equal(parameter.detach(), original)
