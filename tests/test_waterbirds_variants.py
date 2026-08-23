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
    capture_penultimate_features,
    leading_feature_direction,
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


def test_feature_capture_does_not_add_a_second_batchnorm_update():
    """Regression test: a second forward pass would advance BatchNorm twice.

    ``no_grad`` suppresses gradients but not running-statistic updates, so an extra
    forward in train mode would give the modal arm a different normalization
    trajectory from every baseline. The hook reads the head's input from the forward
    pass that already happened.
    """
    torch.manual_seed(11)

    class BatchNormNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.BatchNorm1d(6)
            self.fc = torch.nn.Linear(6, 2)

        def forward(self, x):
            return self.fc(self.norm(x))

    net = BatchNormNet()
    net.train()
    images = torch.randn(24, 6)

    baseline = BatchNormNet()
    baseline.load_state_dict(net.state_dict())
    baseline.train()
    baseline(images)                       # exactly one update, as a baseline arm does

    with capture_penultimate_features(net) as captured:
        net(images)
    assert len(captured) == 1
    assert captured[0].shape == (24, 6)
    assert not captured[0].requires_grad

    # The modal arm must have advanced BatchNorm exactly as much as the baseline.
    torch.testing.assert_close(
        net.norm.running_mean, baseline.norm.running_mean, rtol=0, atol=0
    )
    torch.testing.assert_close(
        net.norm.num_batches_tracked, baseline.norm.num_batches_tracked, rtol=0, atol=0
    )
    # And the hook must be removed afterwards.
    before = net.norm.running_mean.clone()
    net(images)
    assert not torch.equal(before, net.norm.running_mean)


def test_leading_direction_is_deterministic_and_shared():
    """Correction and diagnostic must describe the same component."""
    torch.manual_seed(12)
    features = torch.randn(64, 10)
    first = leading_feature_direction(features)
    second = leading_feature_direction(features)
    torch.testing.assert_close(first, second, rtol=0, atol=0)

    labels = torch.randint(0, 2, (64,))
    background = torch.randint(0, 2, (64,))
    # Passing the shared direction must reproduce the default exactly.
    torch.testing.assert_close(
        modal_feature_coordinates(features, labels),
        modal_feature_coordinates(features, labels, direction=first),
        rtol=0, atol=0,
    )
    assert modal_estimator_agreement(
        features, background
    ) == pytest.approx(modal_estimator_agreement(features, background, direction=first))


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


class _TwoBlockNet(torch.nn.Module):
    """Minimal backbone-plus-head stand-in for the ResNet layout."""

    def __init__(self, width: int = 6, hidden: int = 5):
        super().__init__()
        self.body = torch.nn.Linear(width, hidden)
        self.fc = torch.nn.Linear(hidden, 2)

    def forward(self, x):
        return self.fc(self.body(x))


def test_backbone_receives_its_erm_gradient_when_the_model_is_passed():
    """Regression test: CDC arms must not silently train only the head.

    Without the model argument the backbone keeps ``grad = None`` and the optimizer
    skips it, so the CDC arms would freeze the backbone while every baseline
    fine-tunes the whole network. That is an unfair comparison, not a variant method.
    """
    torch.manual_seed(5)
    net = _TwoBlockNet()
    features = torch.randn(32, 6)
    labels = torch.randint(0, 2, (32,))
    background = torch.randint(0, 2, (32,))
    logits = net(features)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    coordinates = oracle_feature_coordinates(labels, background)

    record = constrained_weak_rescue(
        net.fc, _margin(logits, labels), coordinates,
        target_weak_drift=5.0, loss=loss, model=net,
    )
    assert record["backbone_updated"] is True
    assert net.body.weight.grad is not None
    assert net.fc.weight.grad is not None
    # The head guarantee must survive the backbone gradient being written.
    assert abs(record["cdc_strong_drift_change"]) < 1e-4

    # An optimizer step must move both blocks.
    body_before = net.body.weight.detach().clone()
    head_before = net.fc.weight.detach().clone()
    torch.optim.SGD(net.parameters(), lr=1e-2).step()
    assert not torch.equal(body_before, net.body.weight.detach())
    assert not torch.equal(head_before, net.fc.weight.detach())


def test_omitting_the_model_is_recorded_rather_than_hidden():
    """Head-only mode remains available, but the record says so."""
    torch.manual_seed(6)
    net = _TwoBlockNet()
    features = torch.randn(16, 6)
    labels = torch.randint(0, 2, (16,))
    background = torch.randint(0, 2, (16,))
    logits = net(features)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    record = constrained_weak_rescue(
        net.fc, _margin(logits, labels), oracle_feature_coordinates(labels, background),
        target_weak_drift=5.0, loss=loss,
    )
    assert record["backbone_updated"] is False
    assert net.body.weight.grad is None


def test_cdc_arms_require_sgd_without_weight_decay_for_every_arm(monkeypatch):
    """AdamW would mean the logged guarantee does not describe the update taken.

    And the constraint has to bind the whole comparison: running only the CDC arms on
    SGD would swap a broken guarantee for an optimizer confound.
    """
    from pathlib import Path

    from gradient_starvation import waterbirds

    monkeypatch.setattr(
        waterbirds, "_setup",
        lambda config: (None, None, None, _TwoBlockNet(), 0),
    )
    monkeypatch.setattr(waterbirds, "create_run_directory", lambda config: Path("."))
    monkeypatch.setattr(waterbirds, "resolve_device", lambda requested: torch.device("cpu"))

    def run(training, methods):
        return waterbirds.run_waterbirds(
            {
                "experiment": {"device": "cpu"}, "data": {},
                "training": {"seeds": [0], "epochs": 0, **training},
                "mitigation": {"methods": methods},
            }
        )

    # The check must fire before any data is touched, which the stub loaders prove:
    # a ValueError here means validation ran first.
    with pytest.raises(ValueError, match="training.optimizer: sgd"):
        run({"optimizer": "adamw"}, ["erm", "counterfactual_drift_oracle"])
    with pytest.raises(ValueError, match="weight_decay: 0"):
        run(
            {"optimizer": "sgd", "weight_decay": 1e-4},
            ["erm", "counterfactual_drift_modal"],
        )

    # Without a CDC arm the requirement does not apply. Asserted on the validator
    # directly, because completing a run needs real loaders.
    assert waterbirds._require_raw_velocity_settings(
        ["counterfactual_drift_oracle"], {"optimizer": "sgd", "weight_decay": 0.0}
    ) is None
    for methods in (["erm"], ["erm", "spectral_decoupling"], ["interaction"]):
        assert not set(methods) & waterbirds.CDC_METHODS, methods


def test_shipped_waterbirds_config_satisfies_the_raw_velocity_requirement():
    """The config ships CDC arms, so it must also ship SGD with zero decay."""
    from pathlib import Path

    from gradient_starvation.config import load_config
    from gradient_starvation.waterbirds import CDC_METHODS

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "waterbirds.yaml")
    methods = set(config["mitigation"]["methods"])
    training = config["training"]
    if methods & CDC_METHODS:
        assert str(training.get("optimizer", "adamw")).lower() == "sgd"
        assert float(training.get("weight_decay", 0.0)) == 0.0
