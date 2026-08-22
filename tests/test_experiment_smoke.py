from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gradient_starvation.config import load_config
from gradient_starvation.experiments import run_e1, run_e2, run_e3
from gradient_starvation.plotting import _e1_plot_tables


ROOT = Path(__file__).resolve().parents[1]


def _smoke_config(name: str, tmp_path: Path) -> dict:
    config = load_config(ROOT / "configs" / name)
    config["experiment"]["output_root"] = str(tmp_path)
    config["experiment"]["device"] = "cpu"
    return config


def _assert_common_outputs(run_dir: Path) -> None:
    assert (run_dir / "config.resolved.yaml").is_file()
    assert (run_dir / "environment.json").is_file()
    assert (run_dir / "trajectories.csv").is_file()
    assert (run_dir / "summary.csv").is_file()
    assert (run_dir / "aggregate.csv").is_file()


def _assert_figure_pair(run_dir: Path, stem: str) -> None:
    assert (run_dir / f"{stem}.png").is_file()
    assert (run_dir / f"{stem}.pdf").is_file()


def test_e1_smoke_run_writes_documented_artifacts(tmp_path):
    config = _smoke_config("smoke.yaml", tmp_path)
    config["task"]["beta"] = -1_000_000.0
    run_dir = run_e1(config)

    _assert_common_outputs(run_dir)
    _assert_figure_pair(run_dir, "e1_phase_diagram")
    summary = pd.read_csv(run_dir / "summary.csv")
    trajectories = pd.read_csv(run_dir / "trajectories.csv")
    assert {"rho", "lag_separation", "regime", "delta_tw", "right_censored"} <= set(summary)
    assert {"both", "weak_only"} == set(trajectories["condition"])
    assert not summary["right_censored"].any()


def test_e1_plot_tables_keep_censored_grid_cells_visible():
    summary = pd.DataFrame(
        {
            "regime": ["positive"] * 4,
            "rho": [1.0, 2.0, 1.0, 2.0],
            "lag_separation": [0, 0, 4, 4],
            "weak_auc_gap": [1.0, 2.0, 3.0, 4.0],
            "delta_tw": [0.1, 0.2, np.nan, np.nan],
            "right_censored": [False, False, True, True],
        }
    )

    auc, delay, censored = _e1_plot_tables(summary)

    assert list(auc.index) == [4, 0]
    assert list(auc.columns) == [1.0, 2.0]
    assert auc.loc[4, 2.0] == 4.0
    assert np.isnan(delay.loc[4, 2.0])
    assert censored.loc[4, 2.0] == 1.0


def test_e2_smoke_run_writes_geometry_and_agreement_outputs(tmp_path):
    run_dir = run_e2(_smoke_config("e2_smoke.yaml", tmp_path))

    _assert_common_outputs(run_dir)
    _assert_figure_pair(run_dir, "e2_field_agreement")
    assert (run_dir / "convergence.csv").is_file()
    assert (run_dir / "e2_acceptance.json").is_file()
    geometry = json.loads((run_dir / "finite_n_geometry_check.json").read_text())
    summary = pd.read_csv(run_dir / "summary.csv")
    assert float(geometry["relative_error"]) < 1e-4
    assert {"network", "particle_closure"} <= set(pd.read_csv(run_dir / "trajectories.csv")["source"])
    assert np.isfinite(summary["trajectory_rmse"]).all()
    assert np.isfinite(summary["trajectory_nrmse"]).all()
    assert np.isfinite(summary["geometry_nrmse"]).all()
    assert np.isfinite(summary["margin_nrmse"]).all()
    assert np.isfinite(summary["gsi_rmse"]).all()
    assert np.isfinite(summary["max_identity_absolute_error"]).all()
    assert np.isfinite(summary["max_identity_relative_error"]).all()


def test_e2_rejects_calibration_seed_leakage(tmp_path):
    config = _smoke_config("e2_smoke.yaml", tmp_path)
    config["closure"]["seeds"] = list(config["training"]["seeds"])

    with pytest.raises(ValueError, match="must be disjoint"):
        run_e2(config)


def test_e3_smoke_run_covers_model_and_mitigation_matrix(tmp_path):
    run_dir = run_e3(_smoke_config("e3_smoke.yaml", tmp_path))

    _assert_common_outputs(run_dir)
    _assert_figure_pair(run_dir, "e3_mitigation")
    summary = pd.read_csv(run_dir / "summary.csv")
    assert set(summary["model_kind"]) == {"low_rank_linear", "tanh", "gru"}
    assert set(summary["method"]) == {
        "erm",
        "spectral_decoupling",
        "interaction",
        "counterfactual_drift",
    }
    assert len(summary) == 12
    trajectories = pd.read_csv(run_dir / "trajectories.csv")
    cdc = trajectories[trajectories["method"] == "counterfactual_drift"]
    assert cdc[cdc["condition"] == "both"]["cdc_target_met"].all()


def test_waterbirds_dependency_preflight_is_actionable(monkeypatch):
    import builtins
    from gradient_starvation import waterbirds

    original_import = builtins.__import__

    def block_optional_dependencies(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("torchvision") or name.startswith("wilds"):
            raise ImportError("optional dependency intentionally unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block_optional_dependencies)
    with pytest.raises(RuntimeError, match="requirements-waterbirds.txt"):
        waterbirds._setup({"data": {"root": "data", "batch_size": 1, "num_workers": 0}})
