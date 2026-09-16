import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradient_starvation.cifar_protocol import (
    CNN4GN, CIFARRibbonDataset, augment_core, balanced_brier_skill, cifar_splits,
    cue_metadata, hadamard_codes, paired_ribbon_inputs, stable_seed,
)
from scripts.run_cifar_calibration import _split_digest, train_pilot


SPEC = json.loads((Path(__file__).resolve().parents[1] / "configs/cifar_matched_preregistration.draft.json").read_text())


class FakeCIFAR:
    def __init__(self):
        self.targets = [label for label in range(10) for _ in range(10)]
        self.data = np.stack([
            np.full((32, 32, 3), (label * 20 + index) % 256, dtype=np.uint8)
            for label in range(10) for index in range(10)
        ])

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def test_splits_partition_images_without_leakage_and_are_deterministic():
    labels = np.repeat(np.arange(10), 10)
    kwargs = dict(seed=13001, train_per_class=5, validation_per_class=3,
                  diagnostic_per_class=2, probe_fit_per_class=1)
    first = cifar_splits(labels, **kwargs)
    second = cifar_splits(labels, **kwargs)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert sorted(np.concatenate(list(first.values())).tolist()) == list(range(100))
    assert {key: len(value) for key, value in first.items()} == {
        "train": 50, "validation": 30, "probe_fit": 10, "diagnostic_evaluation": 10,
    }
    assert _split_digest(first) == _split_digest(dict(reversed(list(first.items()))))
    assert _split_digest(first) != _split_digest(cifar_splits(labels, **{**kwargs, "seed": 13002}))


def test_codes_are_balanced_distinct_and_metadata_is_stable():
    codes = hadamard_codes()
    assert codes.shape == (10, 4, 4)
    assert torch.all(codes.sum(dim=(1, 2)) == 0)
    assert len(torch.unique(codes.reshape(10, -1), dim=0)) == 10
    assignment = np.random.default_rng(1).permutation(10)
    a = cue_metadata(block_seed=13001, image_id=42, label=5, reliability=0.9, assignment=assignment)
    b = cue_metadata(block_seed=13001, image_id=42, label=5, reliability=0.9, assignment=assignment)
    assert a == b
    assert 0 <= a.row <= 4 and 0 <= a.column <= 28
    assert stable_seed("x", 4) == stable_seed("x", 4)
    assert stable_seed("x", 4) != stable_seed("x", 5)


def test_paired_inputs_preserve_core_and_zero_only_the_strong_channel():
    core = torch.rand(3, 32, 32)
    augmented = augment_core(core, seed=12, training=True)
    assert torch.equal(augmented, augment_core(core, seed=12, training=True))
    both, weak = paired_ribbon_inputs(augmented, code=hadamard_codes()[0],
                                      row=3, column=7, amplitude=0.5)
    assert both.shape == weak.shape == (3, 40, 32)
    assert torch.equal(both[:, :32], weak[:, :32])
    assert torch.count_nonzero(weak[:, 32:]) == 0
    assert torch.count_nonzero(both[:, 32:]) == 3 * 16
    assert torch.equal(weak[:, :32], 2 * augmented - 1)


def test_dataset_views_are_matched_and_strong_only_blanks_core():
    base = FakeCIFAR()
    indices = np.array([0, 10, 20])
    common = dict(base=base, indices=indices, block_seed=13001,
                  training=True, reliability=0.9, amplitude=0.5)
    both = CIFARRibbonDataset(condition="both", **common)
    weak = CIFARRibbonDataset(condition="weak_only", **common)
    strong = CIFARRibbonDataset(condition="strong_only", **common)
    both.set_epoch(2)
    weak.set_epoch(2)
    x_b, y_b = both[1]
    x_w, y_w = weak[1]
    x_s, y_s = strong[1]
    assert y_b == y_w == y_s
    assert torch.equal(x_b[:, :32], x_w[:, :32])
    assert torch.equal(x_b[:, 32:], x_s[:, 32:])
    assert torch.count_nonzero(x_w[:, 32:]) == 0
    assert torch.count_nonzero(x_s[:, :32]) == 0


def test_cnn_shape_and_brier_skill_reference_values():
    model = CNN4GN()
    assert model(torch.zeros(2, 3, 40, 32)).shape == (2, 10)
    labels = torch.arange(10)
    assert balanced_brier_skill(torch.zeros(10, 10), labels).item() == pytest.approx(0, abs=1e-6)
    logits = torch.full((10, 10), -40.0)
    logits[torch.arange(10), labels] = 40
    assert balanced_brier_skill(logits, labels).item() == pytest.approx(1, abs=1e-6)


def test_w_only_runner_tiny_cpu_smoke_and_b_arm_refused():
    config = copy.deepcopy(SPEC)
    config["dataset"].update(train_per_class=5, validation_per_class=3,
                             diagnostic_per_class=2, diagnostic_probe_fit_per_class=1)
    with pytest.raises(ValueError, match="refuses B"):
        train_pilot(base=FakeCIFAR(), config=config, condition="both", seed=13001,
                    amplitude=0, learning_rate=0.01, weight_decay=0,
                    steps=1, eval_every=1, device=torch.device("cpu"))
    trace, model = train_pilot(base=FakeCIFAR(), config=config, condition="weak_only",
                               seed=13001, amplitude=0, learning_rate=0.01,
                               weight_decay=0, steps=1, eval_every=1,
                               device=torch.device("cpu"))
    assert [row["step"] for row in trace] == [0, 1]
    assert 0 <= trace[-1]["accuracy"] <= 1
    assert np.isfinite(trace[-1]["brier_skill"])
    assert model(torch.zeros(1, 3, 40, 32)).shape == (1, 10)
