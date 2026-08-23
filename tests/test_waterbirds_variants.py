"""Waterbirds CDC variants, tested without any dataset.

No Waterbirds experiment has been run. These tests exercise the estimators and the
constrained correction against synthetic tensors, and pin the oracle / group-agnostic
distinction so it cannot be lost in post-processing. See
``research_scope/waterbirds_setup.md`` for what the author must do to run the real
thing.
"""

from __future__ import annotations

import math

import pytest
import torch

from gradient_starvation.waterbirds import (
    GROUP_AGNOSTIC_METHODS,
    ORACLE_METHODS,
    WATERBIRDS_METHODS,
    _margin,
    _penultimate_features,
    constrained_weak_rescue,
    information_setting,
    modal_estimator_agreement,
    modal_feature_coordinates,
    oracle_feature_coordinates,
)


def test_oracle_and_group_agnostic_sets_are_disjoint_and_labelled():
    assert ORACLE_METHODS.isdisjoint(GROUP_AGNOSTIC_METHODS)
    assert ORACLE_METHODS | GROUP_AGNOSTIC_METHODS == WATERBIRDS_METHODS
    assert information_setting("counterfactual_drift_oracle") == "oracle_uses_group_labels"
    assert information_setting("counterfactual_drift_modal") == "group_agnostic"
    assert information_setting("erm") == "group_agnostic"
    # `interaction` also consumes the background annotation, so it is an oracle too.
    assert information_setting("interaction") == "oracle_uses_group_labels"
    with pytest.raises(ValueError, match="Unknown Waterbirds method"):
        information_setting("counterfactual_drift")


def test_oracle_coordinates_encode_label_background_agreement():
    labels = torch.tensor([1, 1, 0, 0])
    background = torch.tensor([1, 0, 1, 0])
    coordinates = oracle_feature_coordinates(labels, background)
    # y_signed * b_signed: (+1)(+1), (+1)(-1), (-1)(+1), (-1)(-1)
    assert coordinates[:, 0].tolist() == [1.0, -1.0, -1.0, 1.0]
    assert coordinates[:, 1].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_modal_coordinates_need_no_group_labels_and_track_a_planted_direction():
    """The estimator must recover a dominant direction when one is planted."""
    torch.manual_seed(0)
    n, d = 256, 16
    background = torch.randint(0, 2, (n,))
    signed_background = background.mul(2).sub(1).float()
    direction = torch.zeros(d)
    direction[3] = 1.0
    # Dominant planted variance along `direction`, correlated with the background.
    features = 0.05 * torch.randn(n, d) + 4.0 * signed_background[:, None] * direction
    labels = torch.randint(0, 2, (n,))

    coordinates = modal_feature_coordinates(features, labels)
    assert coordinates.shape == (n, 2)
    assert torch.allclose(coordinates[:, 1], torch.ones(n))
    # The estimator's own assumption must be satisfied here by construction.
    assert modal_estimator_agreement(features, background) > 0.95


def test_modal_estimator_agreement_is_low_when_the_assumption_fails():
    """A dominant direction unrelated to the background must be detected as such."""
    torch.manual_seed(1)
    n, d = 256, 16
    background = torch.randint(0, 2, (n,))
    nuisance = torch.randn(n)
    direction = torch.zeros(d)
    direction[7] = 1.0
    features = 0.05 * torch.randn(n, d) + 4.0 * nuisance[:, None] * direction
    agreement = modal_estimator_agreement(features, background)
    assert agreement < 0.3, agreement


def test_modal_estimator_agreement_is_nan_for_a_degenerate_direction():
    features = torch.ones(32, 8)
    background = torch.randint(0, 2, (32,))
    assert math.isnan(modal_estimator_agreement(features, background))


def test_constrained_rescue_preserves_the_strong_drift_and_writes_gradients():
    """Result 1 must hold on the Waterbirds head exactly as it does on synthetic data."""
    torch.manual_seed(2)
    n, d = 128, 12
    head = torch.nn.Linear(d, 2)
    features = torch.randn(n, d)
    labels = torch.randint(0, 2, (n,))
    background = torch.randint(0, 2, (n,))
    logits = head(features)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    margin = _margin(logits, labels)
    coordinates = oracle_feature_coordinates(labels, background)

    record = constrained_weak_rescue(
        head, margin, coordinates, target_weak_drift=5.0, loss=loss
    )
    assert record["cdc_feasible"]
    assert record["cdc_deficit"] > 0
    assert record["cdc_alpha"] > 0
    # The protected quantity is preserved to float32 rounding.
    assert abs(record["cdc_strong_drift_change"]) < 1e-4
    # And the weak drift actually reaches the requested target.
    assert record["cdc_weak_drift_after"] >= 5.0 - 1e-3
    # Gradients must be written for the optimizer to consume.
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_constrained_rescue_is_inactive_without_a_deficit():
    torch.manual_seed(3)
    head = torch.nn.Linear(8, 2)
    features = torch.randn(64, 8)
    labels = torch.randint(0, 2, (64,))
    background = torch.randint(0, 2, (64,))
    logits = head(features)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    record = constrained_weak_rescue(
        head, _margin(logits, labels), oracle_feature_coordinates(labels, background),
        target_weak_drift=-100.0, loss=loss,
    )
    assert record["cdc_deficit"] == 0.0
    assert record["cdc_alpha"] == 0.0


def test_penultimate_features_restore_the_head():
    """Swapping in Identity must not leave the model mutated."""
    model = torch.nn.Sequential()
    model.fc = torch.nn.Linear(6, 2)
    original = model.fc

    class Wrapper(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.fc = head

        def forward(self, x):
            return self.fc(x)

    wrapper = Wrapper(original)
    features = _penultimate_features(wrapper, torch.randn(10, 6))
    assert features.shape == (10, 6)          # Identity head passes features through
    assert wrapper.fc is original
    assert not features.requires_grad


def test_run_waterbirds_rejects_unknown_methods_before_touching_data(monkeypatch):
    """Method validation must not require the dataset to be present."""
    from gradient_starvation import waterbirds

    monkeypatch.setattr(
        waterbirds, "_setup",
        lambda config: (None, None, None, torch.nn.Linear(2, 2), 0),
    )
    monkeypatch.setattr(waterbirds, "create_run_directory", lambda config: __import__(
        "pathlib").Path("."))
    monkeypatch.setattr(waterbirds, "resolve_device", lambda requested: torch.device("cpu"))
    with pytest.raises(ValueError, match="Unknown Waterbirds methods"):
        waterbirds.run_waterbirds(
            {
                "experiment": {"device": "cpu"},
                "data": {},
                "training": {"seeds": [0], "epochs": 0},
                "mitigation": {"methods": ["counterfactual_drift"]},
            }
        )
