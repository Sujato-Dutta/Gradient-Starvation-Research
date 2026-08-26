import pytest
import torch

from gradient_starvation.data.synthetic import SyntheticTaskSpec, make_paired_task
from gradient_starvation.losses import training_objective
from gradient_starvation.models.recurrent import DenseLinearRNN, build_model
from gradient_starvation.theory import (
    cdc_finite_step_deviation_bound,
    counterfactual_drift_correction,
    crossover_decomposition,
    discrete_crossover_certificate,
    exact_dense_linear_geometry,
    gradient_gram,
    initial_frozen_empirical_kernel,
    integrate_frozen_logistic_sgd,
    frozen_kernel_discrete_error_bound,
    matched_weak_drift_decomposition,
    paired_initial_response_jet,
    projected_statistics,
    rank_one_drift_ratio,
    transverse_hitting_time_error_bound,
    zero_disorder_initialization_failure_bound,
    zero_disorder_trajectory_failure_bound,
)


def test_finite_width_geometry_matches_autograd():
    torch.manual_seed(3)
    spec = SyntheticTaskSpec(sequence_length=6, n_samples=32, rho=2, lag_separation=3)
    model = DenseLinearRNN(width=5, bulk_gain=0.4)
    mode = model.mode_responses(spec)
    autograd_geometry, _ = gradient_gram(mode, model)
    analytic_geometry = exact_dense_linear_geometry(model, spec)
    torch.testing.assert_close(autograd_geometry, analytic_geometry, rtol=2e-5, atol=2e-6)


def test_projected_ce_identity_matches_direct_gradient_flow():
    torch.manual_seed(5)
    spec = SyntheticTaskSpec(
        sequence_length=5, n_samples=128, rho=2, lag_separation=2,
        cue_noise=0.05, background_noise=0,
    )
    both, _ = make_paired_task(spec, seed=9)
    model = DenseLinearRNN(width=6, bulk_gain=0.3)
    stats = projected_statistics(model, both, compute_direct_drift=True)
    assert stats.direct_drift is not None
    torch.testing.assert_close(stats.predicted_drift, stats.direct_drift, rtol=2e-5, atol=2e-6)


def test_causal_drift_decomposition_reconstructs_exact_deficit():
    torch.manual_seed(13)
    spec = SyntheticTaskSpec(sequence_length=6, n_samples=64, rho=3, lag_separation=2)
    both, weak = make_paired_task(spec, seed=7)
    model_both = build_model({"kind": "low_rank_linear", "width": 8, "bulk_gain": 0.3})
    model_weak = build_model({"kind": "low_rank_linear", "width": 8, "bulk_gain": 0.3})
    model_weak.load_state_dict(model_both.state_dict())
    both_stats = projected_statistics(model_both, both)
    weak_stats = projected_statistics(model_weak, weak)
    result = matched_weak_drift_decomposition(both_stats, weak_stats, both.z_w)

    torch.testing.assert_close(
        result.ce_gating + result.geometry_shift + result.cross_transport,
        result.causal_deficit,
        rtol=2e-5,
        atol=2e-6,
    )
    assert result.reconstruction_error.abs() < 2e-6


def test_counterfactual_correction_matches_target_and_preserves_strong_drift():
    torch.manual_seed(17)
    spec = SyntheticTaskSpec(sequence_length=6, n_samples=64, rho=3, lag_separation=3)
    both, _ = make_paired_task(spec, seed=9)
    model = build_model({"kind": "low_rank_linear", "width": 8, "bulk_gain": 0.3})
    baseline = projected_statistics(model, both, compute_direct_drift=True)
    assert baseline.direct_drift is not None
    target = baseline.direct_drift[1] + 0.25
    correction = counterfactual_drift_correction(model, both, target)

    assert correction.feasible
    torch.testing.assert_close(correction.weak_drift_after, target, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        correction.strong_drift_after,
        correction.strong_drift_before,
        rtol=2e-5,
        atol=2e-6,
    )
    assert correction.correction_norm > 0


def _paired_statistics(seed: int, *, rho: float = 3.0, lag: int = 2):
    """Build a matched both/weak-only statistics pair from one shared init."""
    spec = SyntheticTaskSpec(sequence_length=6, n_samples=64, rho=rho, lag_separation=lag)
    both, weak = make_paired_task(spec, seed=seed)
    model_both = build_model({"kind": "low_rank_linear", "width": 8, "bulk_gain": 0.3})
    model_weak = build_model({"kind": "low_rank_linear", "width": 8, "bulk_gain": 0.3})
    model_weak.load_state_dict(model_both.state_dict())
    return (
        projected_statistics(model_both, both),
        projected_statistics(model_weak, weak),
        both,
    )


def test_crossover_decomposition_is_an_exact_reparameterization():
    """``d_w = t_geom - s_ce`` must hold identically, not approximately."""
    torch.manual_seed(23)
    both_stats, weak_stats, both = _paired_statistics(seed=7)

    crossover = crossover_decomposition(both_stats, weak_stats, both.z_w)
    torch.testing.assert_close(
        crossover.t_geom - crossover.s_ce,
        crossover.d_w,
        rtol=2e-5,
        atol=2e-6,
    )
    assert crossover.reconstruction_error.abs() < 2e-6


def test_crossover_decomposition_agrees_with_the_three_term_identity():
    """The reparameterization must not introduce a second source of truth."""
    torch.manual_seed(29)
    both_stats, weak_stats, both = _paired_statistics(seed=11)

    parts = matched_weak_drift_decomposition(both_stats, weak_stats, both.z_w)
    crossover = crossover_decomposition(both_stats, weak_stats, both.z_w)

    # d_w is the negated deficit; the two geometry channels are the negated
    # geometry shift and negated cross transport; s_ce is the CE gate verbatim.
    torch.testing.assert_close(crossover.d_w, -parts.causal_deficit, rtol=0, atol=0)
    torch.testing.assert_close(crossover.s_ce, parts.ce_gating, rtol=0, atol=0)
    torch.testing.assert_close(
        crossover.geometry_self_term, -parts.geometry_shift, rtol=0, atol=0
    )
    torch.testing.assert_close(
        crossover.cross_transport_term, -parts.cross_transport, rtol=0, atol=0
    )


def test_matching_conventions_are_distinct():
    """Document that m_w-matched-field and tau-matched-field differ.

    The project standardizes on the m_w-matched-field convention used by
    ``matched_weak_drift_decomposition``.  This test exists so that nobody later
    swaps in the tau-matched alternative believing the two are interchangeable:
    the weak-only CE field evaluated at the both-feature ``m_w`` is a different
    number from the same field evaluated at the weak-only ``m_w``, whenever the
    two conditions have separated.
    """
    torch.manual_seed(31)
    spec = SyntheticTaskSpec(sequence_length=6, n_samples=64, rho=4, lag_separation=2)
    both, weak = make_paired_task(spec, seed=13)

    # Drive the two conditions apart so their weak responses genuinely differ.
    model_both = build_model({"kind": "dense_linear", "width": 10, "bulk_gain": 0.3})
    model_weak = build_model({"kind": "dense_linear", "width": 10, "bulk_gain": 0.3})
    model_weak.load_state_dict(model_both.state_dict())
    for model, batch in ((model_both, both), (model_weak, weak)):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        for _ in range(15):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = training_objective(model, batch, {"method": "erm"})
            loss.backward()
            optimizer.step()

    both_stats = projected_statistics(model_both, both)
    weak_stats = projected_statistics(model_weak, weak)
    assert not torch.isclose(both_stats.mode[1], weak_stats.mode[1], rtol=1e-3)

    # Convention 1 (adopted): weak-only CE field recomputed at the both m_w.
    adopted = crossover_decomposition(both_stats, weak_stats, both.z_w)

    # Convention 2 (rejected): weak-only drift taken at its own m_w, i.e. the
    # raw tau-matched difference with no field re-evaluation.
    tau_matched_d_w = both_stats.predicted_drift[1] - weak_stats.predicted_drift[1]

    assert not torch.isclose(adopted.d_w, tau_matched_d_w, rtol=1e-3, atol=1e-8), (
        "The two matching conventions coincided; this test can no longer detect "
        "a silent convention swap."
    )


def test_linear_exact_realization_has_zero_mode_and_drift_residuals():
    """The closed Gg corollary is exact on its stated linear/no-background scope."""
    torch.manual_seed(37)
    spec = SyntheticTaskSpec(
        sequence_length=6,
        n_samples=128,
        rho=3,
        lag_separation=2,
        cue_noise=0.1,
        background_noise=0.0,
    )
    both, _ = make_paired_task(spec, seed=19)
    model = DenseLinearRNN(width=9, bulk_gain=0.3)
    stats = projected_statistics(model, both, compute_direct_drift=True)

    assert stats.mode_residual_rms < 2e-6
    assert stats.projected_drift_residual is not None
    torch.testing.assert_close(
        stats.projected_drift_residual,
        torch.zeros_like(stats.projected_drift_residual),
        rtol=2e-5,
        atol=2e-6,
    )


def test_nonlinear_projected_modes_expose_the_omitted_drift_residual():
    """Nonlinear probe responses must never be silently labelled exact Gg dynamics."""
    torch.manual_seed(41)
    spec = SyntheticTaskSpec(
        sequence_length=7,
        n_samples=96,
        rho=3,
        lag_separation=3,
        cue_noise=0.15,
        background_noise=0.0,
    )
    both, _ = make_paired_task(spec, seed=23)
    model = build_model({"kind": "tanh", "width": 10})
    stats = projected_statistics(model, both, compute_direct_drift=True)

    assert stats.mode_residual_rms > 1e-5
    assert stats.projected_drift_residual is not None
    assert torch.isfinite(stats.projected_drift_residual).all()
    assert stats.projected_drift_residual.norm() > 1e-7


def test_rank_one_log_ratio_factorizes_noiseless_linear_weak_drifts():
    torch.manual_seed(43)
    spec = SyntheticTaskSpec(
        sequence_length=5,
        n_samples=64,
        rho=4,
        lag_separation=0,
        cue_noise=0.0,
        background_noise=0.0,
    )
    both, weak = make_paired_task(spec, seed=29)
    # At the zero-disorder/zero-lag anchor, the weak geometry factors concentrate
    # near positive constants, satisfying Theorem C's explicit positivity premise.
    model_both = DenseLinearRNN(width=64, bulk_gain=0.0)
    model_weak = DenseLinearRNN(width=64, bulk_gain=0.0)
    model_weak.load_state_dict(model_both.state_dict())
    both_stats = projected_statistics(model_both, both, compute_direct_drift=True)
    weak_stats = projected_statistics(model_weak, weak, compute_direct_drift=True)

    certificate = rank_one_drift_ratio(both_stats, weak_stats, rho=spec.rho)
    assert certificate.positive_drifts
    torch.testing.assert_close(
        certificate.both_factorization_error,
        torch.zeros_like(certificate.both_factorization_error),
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        certificate.weak_only_factorization_error,
        torch.zeros_like(certificate.weak_only_factorization_error),
        rtol=2e-5,
        atol=2e-6,
    )
    assert torch.isfinite(certificate.log_drift_ratio)
    assert torch.sign(certificate.log_drift_ratio) == torch.sign(
        certificate.drift_difference
    )


def test_discrete_tail_area_certificate_distinguishes_suppression_from_starvation():
    starved = discrete_crossover_certificate(
        [0.0, 1.0, 2.0, 1.0, 0.0, -1.0], weak_only_learnable=True
    )
    assert starved.single_transfer_to_suppression
    assert starved.drift_crossover_step == 2
    assert starved.peak_step == 2
    assert starved.response_equality_step == 4
    assert starved.positive_area == 2.0
    assert starved.negative_tail_area == 3.0
    assert starved.tail_area_margin == 1.0
    assert starved.strict_outcome_starvation
    assert starved.causal_starvation_certified

    suppressed_only = discrete_crossover_certificate([0.0, 1.0, 2.0, 1.5, 1.0])
    assert suppressed_only.single_transfer_to_suppression
    assert not suppressed_only.response_equality_reached
    assert not suppressed_only.strict_outcome_starvation
    assert suppressed_only.tail_area_margin == -1.0


def test_discrete_certificate_requires_one_strict_sign_change():
    oscillatory = discrete_crossover_certificate([0.0, 1.0, 0.5, 0.75, 0.25])
    assert not oscillatory.single_transfer_to_suppression
    assert not oscillatory.causal_starvation_certified


def test_transverse_hitting_time_bound_includes_grid_resolution():
    assert transverse_hitting_time_error_bound(
        0.06,
        0.3,
        prehit_separation=0.1,
        transversality_radius=0.25,
        grid_spacing=0.05,
    ) == 0.25


@pytest.mark.parametrize("prehit_separation", [0.0, -1.0, float("nan"), float("inf")])
def test_transverse_hitting_time_bound_requires_valid_prehit_separation(
    prehit_separation,
):
    with pytest.raises(ValueError, match="prehit_separation"):
        transverse_hitting_time_error_bound(
            0.01,
            1.0,
            prehit_separation=prehit_separation,
            transversality_radius=1.0,
        )


@pytest.mark.parametrize("transversality_radius", [0.0, -1.0, float("nan"), float("inf")])
def test_transverse_hitting_time_bound_requires_valid_transversality_radius(
    transversality_radius,
):
    with pytest.raises(ValueError, match="transversality_radius"):
        transverse_hitting_time_error_bound(
            0.01,
            1.0,
            prehit_separation=0.1,
            transversality_radius=transversality_radius,
        )


def test_transverse_hitting_time_bound_enforces_local_hypotheses():
    with pytest.raises(ValueError, match="strictly smaller"):
        transverse_hitting_time_error_bound(
            0.1,
            1.0,
            prehit_separation=0.1,
            transversality_radius=1.0,
        )
    with pytest.raises(ValueError, match="exceeds transversality_radius"):
        transverse_hitting_time_error_bound(
            0.06,
            0.3,
            prehit_separation=0.1,
            transversality_radius=0.24,
            grid_spacing=0.05,
        )


def test_zero_disorder_probability_bounds_have_the_proved_scaling():
    coarse = zero_disorder_initialization_failure_bound(100, 0.5)
    wide = zero_disorder_initialization_failure_bound(400, 0.5)
    assert coarse == 0.36
    assert wide == 0.09
    propagated = zero_disorder_trajectory_failure_bound(
        width=400,
        delta=1.0,
        lipschitz_constant=0.5,
        horizon=1.0,
        tube_radius=2.0,
    )
    assert propagated >= zero_disorder_initialization_failure_bound(400, 1.0)


def test_zero_disorder_trajectory_bound_saturates_at_the_tube_radius():
    radius_limited = zero_disorder_trajectory_failure_bound(
        width=1000,
        delta=2.0,
        lipschitz_constant=0.2,
        horizon=3.0,
        tube_radius=0.5,
    )
    delta_limited = zero_disorder_trajectory_failure_bound(
        width=1000,
        delta=0.5,
        lipschitz_constant=0.2,
        horizon=3.0,
        tube_radius=2.0,
    )
    assert radius_limited == delta_limited


@pytest.mark.parametrize("tube_radius", [0.0, -1.0, float("nan"), float("inf")])
def test_zero_disorder_trajectory_bound_requires_a_valid_tube_radius(tube_radius):
    with pytest.raises(ValueError, match="tube_radius"):
        zero_disorder_trajectory_failure_bound(
            width=400,
            delta=1.0,
            lipschitz_constant=0.5,
            horizon=1.0,
            tube_radius=tube_radius,
        )


def test_cdc_finite_step_bound_controls_a_quadratic_protected_response():
    # m_s(x)=L||x||^2/2 has exactly L-Lipschitz gradient.  The correction changes
    # only the coordinate orthogonal to grad m_s(theta), so first-order terms cancel.
    lipschitz = 2.0
    eta = 0.1
    theta = torch.tensor([1.0, 0.0], dtype=torch.float64)
    erm_velocity = torch.tensor([0.3, 0.4], dtype=torch.float64)
    corrected_velocity = torch.tensor([0.3, 0.6], dtype=torch.float64)

    def response(value: torch.Tensor) -> torch.Tensor:
        return 0.5 * lipschitz * value.square().sum()

    observed = abs(
        float(response(theta + eta * corrected_velocity))
        - float(response(theta + eta * erm_velocity))
    )
    bound = cdc_finite_step_deviation_bound(
        lipschitz,
        eta,
        float(erm_velocity.norm()),
        float(corrected_velocity.norm()),
    )
    assert observed <= bound


def test_nonlinear_probe_response_is_common_across_causal_conditions():
    """Shared parameters must imply Delta(0)=0 for the theorem response functional."""
    spec = SyntheticTaskSpec(
        sequence_length=7, n_samples=64, rho=4, lag_separation=2, cue_noise=0.1
    )
    both, weak = make_paired_task(spec, seed=31)
    for kind in ("tanh", "gru"):
        torch.manual_seed(47)
        model_both = build_model({"kind": kind, "width": 9})
        model_weak = build_model({"kind": kind, "width": 9})
        model_weak.load_state_dict(model_both.state_dict())
        both_mode = projected_statistics(model_both, both).mode
        weak_mode = projected_statistics(model_weak, weak).mode
        torch.testing.assert_close(both_mode, weak_mode, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["tanh", "gru"])
def test_full_empirical_ntk_reconstructs_exact_initial_response_drift(kind):
    spec = SyntheticTaskSpec(
        sequence_length=5, n_samples=12, rho=2, lag_separation=2, cue_noise=0.1
    )
    both, weak = make_paired_task(spec, seed=71)
    torch.manual_seed(72)
    both_model = build_model({"kind": kind, "width": 6}, kind=kind)
    weak_model = build_model({"kind": kind, "width": 6}, kind=kind)
    weak_model.load_state_dict(both_model.state_dict())
    before = {
        name: value.detach().clone() for name, value in both_model.state_dict().items()
    }

    both_kernel = initial_frozen_empirical_kernel(both_model, both)
    weak_kernel = initial_frozen_empirical_kernel(weak_model, weak)
    jet = paired_initial_response_jet(both_model, weak_model, both, weak)

    torch.testing.assert_close(
        both_kernel.signed_ntk,
        both_kernel.signed_logit_jacobian @ both_kernel.signed_logit_jacobian.T,
    )
    torch.testing.assert_close(
        both_kernel.response_cross_kernel,
        both_kernel.signed_logit_jacobian @ both_kernel.response_jacobian,
    )
    torch.testing.assert_close(
        both_kernel.signed_ntk, both_kernel.signed_ntk.T, rtol=0, atol=0
    )
    both_drift = (
        both_kernel.response_cross_kernel @ torch.sigmoid(-both_kernel.signed_logits)
        / spec.n_samples
    )
    weak_drift = (
        weak_kernel.response_cross_kernel @ torch.sigmoid(-weak_kernel.signed_logits)
        / spec.n_samples
    )
    torch.testing.assert_close(both_drift - weak_drift, jet.d_0, rtol=2e-5, atol=2e-6)
    assert all(parameter.grad is None for parameter in both_model.parameters())
    for name, value in before.items():
        assert torch.equal(value, both_model.state_dict()[name]), name


def test_frozen_logistic_sgd_uses_experiment_time_and_initial_drift():
    spec = SyntheticTaskSpec(sequence_length=5, n_samples=12, rho=2, lag_separation=2)
    both, _ = make_paired_task(spec, seed=73)
    torch.manual_seed(74)
    model = build_model({"kind": "tanh", "width": 6})
    kernel = initial_frozen_empirical_kernel(model, both)
    trajectory = integrate_frozen_logistic_sgd(
        kernel, steps=5, learning_rate=0.02, log_every=2
    )

    assert trajectory.steps.tolist() == [0, 2, 4, 5]
    torch.testing.assert_close(
        trajectory.tau, torch.tensor([0.0, 0.04, 0.08, 0.10], dtype=torch.float64)
    )
    expected = (
        kernel.response_cross_kernel @ torch.sigmoid(-kernel.signed_logits)
        / spec.n_samples
    )
    torch.testing.assert_close(
        trajectory.response_drift[0], expected.to(torch.float64), rtol=2e-6, atol=1e-9
    )


def test_frozen_kernel_discrete_bound_vanishes_when_secant_kernels_do_not_move():
    exact = frozen_kernel_discrete_error_bound(
        step=100,
        learning_rate=0.01,
        n_samples=32,
        initial_kernel_norm=4.0,
        initial_cross_kernel_norm=2.0,
        secant_kernel_drift_bound=0.0,
        secant_cross_kernel_drift_bound=0.0,
    )
    perturbed = frozen_kernel_discrete_error_bound(
        step=100,
        learning_rate=0.01,
        n_samples=32,
        initial_kernel_norm=4.0,
        initial_cross_kernel_norm=2.0,
        secant_kernel_drift_bound=0.1,
        secant_cross_kernel_drift_bound=0.1,
    )
    assert exact.logit_error == 0.0
    assert exact.response_error == 0.0
    assert perturbed.logit_error > 0.0
    assert perturbed.response_error > 0.0
