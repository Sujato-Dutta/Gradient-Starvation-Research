import torch

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

