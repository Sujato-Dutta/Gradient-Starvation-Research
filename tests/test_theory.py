import torch

from gradient_starvation.data.synthetic import SyntheticTaskSpec, make_paired_task
from gradient_starvation.losses import training_objective
from gradient_starvation.models.recurrent import DenseLinearRNN, build_model
from gradient_starvation.theory import (
    counterfactual_drift_correction,
    crossover_decomposition,
    exact_dense_linear_geometry,
    gradient_gram,
    matched_weak_drift_decomposition,
    projected_statistics,
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
