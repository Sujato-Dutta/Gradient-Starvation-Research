import math

import pytest
import torch

from gradient_starvation.data.synthetic import SyntheticTaskSpec, make_paired_task
from gradient_starvation.losses import training_objective
from gradient_starvation.models.recurrent import build_model, synthetic_logits


@pytest.mark.parametrize("kind", ["low_rank_linear", "dense_linear", "tanh", "gru"])
@pytest.mark.parametrize(
    "mitigation",
    [
        {"method": "erm"},
        {"method": "spectral_decoupling", "spectral_coefficient": 0.01},
        {"method": "interaction", "coefficient": 0.01, "mode": "proxy"},
    ],
)
def test_every_supported_model_and_proxy_mitigation_backpropagates(kind, mitigation):
    spec = SyntheticTaskSpec(sequence_length=5, n_samples=12, rho=2, lag_separation=2)
    both, _ = make_paired_task(spec, seed=4)
    model = build_model({"kind": kind, "width": 6, "bulk_gain": 0.3, "rank": 2})
    loss, diagnostics = training_objective(model, both, mitigation)
    loss.backward()

    assert loss.isfinite()
    assert diagnostics["ce_loss"] > 0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_second_order_interaction_penalty_backpropagates_for_linear_model():
    spec = SyntheticTaskSpec(sequence_length=5, n_samples=12, rho=2, lag_separation=2)
    both, _ = make_paired_task(spec, seed=8)
    model = build_model({"kind": "low_rank_linear", "width": 6, "bulk_gain": 0.3, "rank": 2})
    loss, diagnostics = training_objective(
        model, both, {"method": "interaction", "coefficient": 0.01, "mode": "second_order"}
    )
    loss.backward()

    assert loss.isfinite()
    assert math.isfinite(diagnostics["penalty_susceptibility"])


@pytest.mark.parametrize("kind", ["low_rank_linear", "dense_linear"])
@pytest.mark.parametrize("condition", ["both", "weak_only"])
def test_linear_synthetic_logits_equal_actual_network_output(kind, condition):
    """Regression test: feature strength must appear exactly once."""
    torch.manual_seed(11)
    spec = SyntheticTaskSpec(
        sequence_length=6,
        n_samples=32,
        rho=4,
        lag_separation=3,
        cue_noise=0.1,
        background_noise=0,
    )
    both, weak = make_paired_task(spec, seed=12)
    batch = both if condition == "both" else weak
    model = build_model({"kind": kind, "width": 7, "bulk_gain": 0.3, "rank": 2})
    torch.testing.assert_close(synthetic_logits(model, batch), model(batch.x), rtol=2e-5, atol=2e-6)


def test_counterfactual_drift_requires_paired_training():
    spec = SyntheticTaskSpec(sequence_length=5, n_samples=12, rho=2, lag_separation=2)
    both, _ = make_paired_task(spec, seed=4)
    model = build_model({"kind": "low_rank_linear", "width": 6, "rank": 2})
    with pytest.raises(ValueError, match="paired both/weak-only"):
        training_objective(model, both, {"method": "counterfactual_drift"})


@pytest.mark.parametrize("kind", ["low_rank_linear", "dense_linear"])
def test_linear_input_and_readout_use_width_stable_scaling(kind):
    torch.manual_seed(19)
    width = 128
    model = build_model({"kind": kind, "width": width, "bulk_gain": 0.3, "rank": 2})

    input_norms = model.input.detach().square().sum(dim=0)
    readout_norm = model.readout.detach().square().sum()

    torch.testing.assert_close(input_norms, torch.ones_like(input_norms), rtol=0.35, atol=0.35)
    torch.testing.assert_close(readout_norm, torch.ones_like(readout_norm), rtol=0.35, atol=0.35)
