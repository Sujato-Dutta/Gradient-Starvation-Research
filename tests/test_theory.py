import torch

from gradient_starvation.data.synthetic import SyntheticTaskSpec, make_paired_task
from gradient_starvation.models.recurrent import DenseLinearRNN, build_model
from gradient_starvation.theory import (
    counterfactual_drift_correction,
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
