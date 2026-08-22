import torch
import pytest

from gradient_starvation.data.synthetic import SyntheticTaskSpec, make_paired_task


def test_weak_only_is_an_exact_strong_channel_ablation():
    spec = SyntheticTaskSpec(sequence_length=8, n_samples=128, rho=4, lag_separation=3)
    both, weak = make_paired_task(spec, seed=7)
    assert torch.equal(both.y, weak.y)
    assert torch.equal(both.z_w, weak.z_w)
    assert torch.count_nonzero(weak.x[:, :, 0]) == 0
    assert torch.equal(both.x[:, :, 1], weak.x[:, :, 1])


def test_negative_control_has_nonpositive_cross_moment():
    spec = SyntheticTaskSpec(n_samples=100_000, regime="negative", cue_noise=0)
    both, _ = make_paired_task(spec, seed=11)
    assert float((both.z_s * both.z_w).mean()) < 0
    assert float(both.z_s.mean()) > 0
    assert float(both.z_w.mean()) > 0


def test_generation_is_deterministic_and_places_each_cue_at_its_configured_time():
    spec = SyntheticTaskSpec(
        sequence_length=7,
        n_samples=16,
        rho=3,
        lag_separation=4,
        cue_noise=0,
        background_noise=0,
    )
    both, weak = make_paired_task(spec, seed=5)
    repeated, _ = make_paired_task(spec, seed=5)

    assert torch.equal(both.x, repeated.x)
    assert torch.equal(both.x[:, spec.strong_time, 0], both.signed_labels * both.z_s)
    assert torch.equal(both.x[:, spec.weak_time, 1], both.signed_labels * both.z_w)
    assert torch.count_nonzero(both.x[:, :spec.strong_time, 0]) == 0
    assert torch.count_nonzero(both.x[:, :spec.weak_time, 1]) == 0
    moved = both.to("cpu")
    assert moved.spec is spec
    assert moved.condition == "both"
    assert torch.equal(moved.x, both.x)
    assert torch.equal(weak.y, both.y)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sequence_length": 0},
        {"n_samples": 0},
        {"rho": 0},
        {"sequence_length": 4, "lag_separation": 4},
        {"regime": "unknown"},
        {"cue_noise": -0.1},
    ],
)
def test_task_spec_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        SyntheticTaskSpec(**kwargs)
