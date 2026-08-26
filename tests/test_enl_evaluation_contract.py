from __future__ import annotations

from pathlib import Path

import pytest

import gradient_starvation.experiments as experiment_module
from gradient_starvation.experiments import run_enl_evaluate


def _minimal_evaluation_config(tmp_path: Path) -> dict:
    return {
        "experiment": {
            "name": "must_not_be_created",
            "output_root": str(tmp_path),
        },
        "training": {
            "paired_mode": "lockstep",
            "exact_response_drift": True,
        },
        "evaluation": {},
    }


def _forbid_run_directory(*args, **kwargs):
    pytest.fail("run_enl_evaluate created an output directory before contract verification")


def test_enl_evaluate_rejects_sequential_training_before_output(tmp_path, monkeypatch):
    config = _minimal_evaluation_config(tmp_path)
    config["training"]["paired_mode"] = "sequential"
    monkeypatch.setattr(experiment_module, "create_run_directory", _forbid_run_directory)

    with pytest.raises(
        ValueError,
        match=r"E-NL evaluation requires training\.paired_mode='lockstep'\.",
    ):
        run_enl_evaluate(config)

    assert not list(tmp_path.iterdir())


def test_enl_evaluate_requires_exact_response_drift_before_output(tmp_path, monkeypatch):
    config = _minimal_evaluation_config(tmp_path)
    config["training"]["exact_response_drift"] = False
    monkeypatch.setattr(experiment_module, "create_run_directory", _forbid_run_directory)

    with pytest.raises(
        ValueError,
        match=r"E-NL evaluation requires training\.exact_response_drift=true\.",
    ):
        run_enl_evaluate(config)

    assert not list(tmp_path.iterdir())
