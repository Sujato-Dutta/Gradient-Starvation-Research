"""Outcome-free tests for the semi-real exact intervention and response contract."""

from __future__ import annotations

import copy

import pytest
import torch

from gradient_starvation.semi_real import (
    SemiRealCNN,
    SemiRealTaskSpec,
    balanced_sample_ids,
    make_paired_semi_real_task,
    make_semi_real_probe,
    prepare_core_images,
    semi_real_probe_responses,
    semi_real_statistics,
)


def _toy_data(spec: SemiRealTaskSpec, *, offset: int = 0):
    raw = torch.arange(8 * 8 * 8, dtype=torch.int64).remainder(256).to(torch.uint8)
    raw = raw.reshape(8, 8, 8)
    core = prepare_core_images(raw, spec.image_size)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    sample_ids = torch.arange(offset, offset + 8, dtype=torch.long)
    return core, labels, sample_ids


def test_semi_real_pair_differs_only_in_the_generated_cue_channel():
    spec = SemiRealTaskSpec("mnist", 3, 8, cue_strength=4.0, image_size=4)
    core, labels, sample_ids = _toy_data(spec)
    both, weak = make_paired_semi_real_task(core, labels, sample_ids, spec)

    assert torch.equal(both.y, weak.y)
    assert torch.equal(both.signed_labels, weak.signed_labels)
    assert torch.equal(both.sample_ids, weak.sample_ids)
    assert torch.equal(both.x[:, 0], weak.x[:, 0])
    assert torch.count_nonzero(weak.x[:, 1]) == 0
    patch = spec.cue_patch_size
    expected = both.signed_labels[:, None, None] * spec.cue_strength
    torch.testing.assert_close(both.x[:, 1, :patch, :patch], expected.expand(-1, patch, patch))
    assert torch.count_nonzero(both.x[:, 1, patch:, :]) == 0
    assert torch.count_nonzero(both.x[:, 1, :, patch:]) == 0


def test_common_core_probe_is_identical_at_shared_initialization():
    spec = SemiRealTaskSpec("fashion_mnist", 0, 6, image_size=8)
    core, labels, sample_ids = _toy_data(spec)
    both, weak = make_paired_semi_real_task(core, labels, sample_ids, spec)
    probe_core, probe_labels, probe_ids = _toy_data(spec, offset=100)
    probe = make_semi_real_probe(probe_core, probe_labels, probe_ids, spec)

    torch.manual_seed(11)
    both_model = SemiRealCNN(channels=(4, 5))
    weak_model = SemiRealCNN(channels=(4, 5))
    weak_model.load_state_dict(copy.deepcopy(both_model.state_dict()))
    both_stats = semi_real_statistics(both_model, both, probe)
    weak_stats = semi_real_statistics(weak_model, weak, probe)

    torch.testing.assert_close(both_stats.core_response, weak_stats.core_response, rtol=0, atol=0)
    torch.testing.assert_close(both_stats.cue_response, weak_stats.cue_response, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in both_model.parameters())
    assert all(parameter.grad is None for parameter in weak_model.parameters())
    assert torch.isfinite(both_stats.core_drift)
    assert torch.isfinite(weak_stats.core_drift)


def test_exact_core_drift_matches_a_fixed_velocity_finite_difference():
    spec = SemiRealTaskSpec("mnist", 3, 8, cue_strength=2.0, image_size=4)
    core, labels, sample_ids = _toy_data(spec)
    core = core.to(torch.float64)
    both, _ = make_paired_semi_real_task(core, labels, sample_ids, spec)
    probe = make_semi_real_probe(core.flip(0), labels.flip(0), sample_ids + 100, spec)

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(2 * spec.image_size * spec.image_size, 1, bias=True),
        torch.nn.Flatten(0),
    ).to(torch.float64)
    statistics = semi_real_statistics(model, both, probe)
    parameters = list(model.parameters())
    logits = model(both.x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, both.y.to(logits.dtype)
    )
    velocity = [-gradient.detach() for gradient in torch.autograd.grad(loss, parameters)]
    initial = [parameter.detach().clone() for parameter in parameters]

    def response_at(sign: float) -> torch.Tensor:
        with torch.no_grad():
            for parameter, value, direction in zip(parameters, initial, velocity):
                parameter.copy_(value + sign * 1e-6 * direction)
        return semi_real_probe_responses(model, probe)[0].detach()

    finite_difference = (response_at(1.0) - response_at(-1.0)) / (2e-6)
    torch.testing.assert_close(statistics.core_drift, finite_difference, rtol=2e-5, atol=2e-8)


def test_balanced_selection_is_reproducible_and_validated():
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    ids = torch.tensor([10, 11, 12, 20, 21, 22])
    first = balanced_sample_ids(labels, ids, per_class=2, seed=17)
    second = balanced_sample_ids(labels, ids, per_class=2, seed=17)
    assert torch.equal(first, second)
    assert len(first) == 4
    with pytest.raises(ValueError, match="fewer than"):
        balanced_sample_ids(labels, ids, per_class=4, seed=17)


def test_raw_image_scaling_requires_unambiguous_uint8_input():
    with pytest.raises(TypeError, match="uint8"):
        prepare_core_images(torch.zeros(4, 8, 8), image_size=4)
