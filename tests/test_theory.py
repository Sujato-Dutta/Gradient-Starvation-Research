import torch

from gradient_starvation.data.synthetic import SyntheticTaskSpec, make_paired_task
from gradient_starvation.models.recurrent import DenseLinearRNN
from gradient_starvation.theory import (
    exact_dense_linear_geometry,
    gradient_gram,
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

