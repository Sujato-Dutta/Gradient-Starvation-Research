from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gradient_starvation.config import load_config
from gradient_starvation.experiments import (
    _enl_summary,
    run_e1,
    run_e2,
    run_e2r,
    run_e3,
    run_enl,
)
from gradient_starvation.metrics import CAUSAL_REGIMES
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
    environment = json.loads((run_dir / "environment.json").read_text())
    assert len(environment["source_sha256"]) == 64
    assert environment["source_file_count"] > 0
    assert environment["source_scope"] == "run_experiment.py and src/**/*.py"


def _assert_figure_pair(run_dir: Path, stem: str) -> None:
    assert (run_dir / f"{stem}.png").is_file()
    assert (run_dir / f"{stem}.pdf").is_file()


def test_e1_smoke_run_writes_documented_artifacts(tmp_path):
    config = _smoke_config("smoke.yaml", tmp_path)
    config["task"]["beta"] = -1_000_000.0
    run_dir = run_e1(config)

    _assert_common_outputs(run_dir)
    _assert_figure_pair(run_dir, "e1_phase_diagram")
    _assert_figure_pair(run_dir, "e1_causal_regions")
    assert (run_dir / "regions.csv").is_file()
    summary = pd.read_csv(run_dir / "summary.csv")
    trajectories = pd.read_csv(run_dir / "trajectories.csv")
    assert {"rho", "lag_separation", "regime", "delta_tw", "right_censored"} <= set(summary)
    assert {"both", "weak_only"} == set(trajectories["condition"])
    assert not summary["right_censored"].any()

    # The gated classification is emitted alongside the pre-gate label, so the
    # run directories already on disk stay interpretable.
    assert {
        "regime_class", "weak_only_learnable", "target_met_at_initialization",
        "gated_delta_tw", "legacy_phase", "phase",
    } <= set(summary)
    assert set(summary["regime_class"]) <= set(CAUSAL_REGIMES)
    assert (summary["legacy_phase"] == summary["phase"]).all()


def test_e1_learnability_gate_config_keys_are_honoured(tmp_path):
    """`task.tau_max` and `task.delta_tw_tolerance` must reach the classifier."""
    config = _smoke_config("smoke.yaml", tmp_path)
    # An unreachable target makes every point unlearnable, which proves the gate runs.
    # `tau_max` cannot be used for this any more: it is validated to exceed the initial
    # time, precisely so a nonsensical horizon cannot masquerade as a finding.
    config["task"]["beta"] = 1_000_000.0
    config["task"]["tau_max"] = 0.5
    run_dir = run_e1(config)
    summary = pd.read_csv(run_dir / "summary.csv")
    assert set(summary["regime_class"]) == {"unlearnable"}
    assert not summary["weak_only_learnable"].any()


def test_e1_rejects_a_degenerate_horizon(tmp_path):
    """`tau_max` at or below the initial time would make every point unlearnable."""
    config = _smoke_config("smoke.yaml", tmp_path)
    config["task"]["beta"] = -1_000_000.0
    config["task"]["tau_max"] = 0.0
    with pytest.raises(ValueError, match="must exceed initial_tau"):
        run_e1(config)


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
    sources = set(pd.read_csv(run_dir / "trajectories.csv")["source"])
    # The reference is a mean over trained finite networks, so it must not be
    # labelled as a solved theory.
    assert {"network", "closure_reference"} <= sources
    assert "particle_closure" not in sources
    assert not any("dmft" in source.lower() for source in sources)
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


def test_waterbirds_config_methods_are_all_recognised():
    """Every method named in the shipped config must be a known Waterbirds method."""
    from gradient_starvation.waterbirds import WATERBIRDS_METHODS, information_setting

    config = load_config(ROOT / "configs" / "waterbirds.yaml")
    methods = config.get("mitigation", {}).get("methods", [])
    assert methods, "the Waterbirds config should name at least one method"
    for method in methods:
        assert method in WATERBIRDS_METHODS, method
        # Every method must resolve to an information setting, so no arm can be
        # tabulated without disclosing the supervision it consumed.
        assert information_setting(method) in {
            "oracle_uses_group_labels", "group_agnostic"
        }


def test_enl_smoke_run_writes_crossover_artifacts(tmp_path):
    run_dir = run_enl(_smoke_config("enl_smoke.yaml", tmp_path))

    _assert_common_outputs(run_dir)
    _assert_figure_pair(run_dir, "enl_crossover")
    assert (run_dir / "crossover.csv").is_file()

    trajectories = pd.read_csv(run_dir / "trajectories.csv")
    summary = pd.read_csv(run_dir / "summary.csv")
    crossover = pd.read_csv(run_dir / "crossover.csv")

    # Mechanistic columns must be present on the both-feature rows and absent from
    # the weak-only rows, since the decomposition is a property of the pair.
    both = trajectories[trajectories["condition"] == "both"]
    weak = trajectories[trajectories["condition"] == "weak_only"]
    assert {
        "d_w_matched", "d_w_equal_time", "d_w_equal_time_projected",
        "d_w_equal_time_exact", "equal_time_projection_residual", "t_geom", "s_ce",
        "t_geom_self", "t_geom_cross", "equal_time_cross_transport",
        "equal_time_geometry_a", "equal_time_field_a",
        "equal_time_geometry_b", "equal_time_field_b",
    } <= set(both)
    # Both additive orderings reconstruct the projected Gg difference.  The direct
    # autograd column is separately required and is the primary E-NL derivative.
    for ordering in ("a", "b"):
        assert (both[f"equal_time_reconstruction_error_{ordering}"].abs() < 2e-6).all()
        reconstructed = (
            both[f"equal_time_geometry_{ordering}"] + both[f"equal_time_field_{ordering}"]
        )
        np.testing.assert_allclose(
            reconstructed.to_numpy(), both["d_w_equal_time"].to_numpy(),
            rtol=2e-5, atol=2e-6,
        )
    assert summary["max_equal_time_reconstruction_error"].max() < 2e-6
    assert both[
        [
            "d_w_matched", "d_w_equal_time", "d_w_equal_time_projected",
            "d_w_equal_time_exact", "equal_time_projection_residual", "t_geom", "s_ce",
        ]
    ].notna().all().all()
    assert weak["d_w_matched"].isna().all()
    assert weak["d_w_equal_time"].isna().all()
    assert weak["d_w_equal_time_exact"].isna().all()
    # Exact, projected, and matched conventions must all be reported. The exact
    # direct-autograd convention drives tau_star_drift.
    assert {
        "tau_star_drift", "tau_star_drift_projected", "tau_star_projected_gap",
        "tau_star_matched_state", "tau_star_convention_gap",
        "max_equal_time_projection_residual", "max_mode_residual_rms",
    } <= set(summary)

    # The reparameterization must reconstruct at every logged step.
    assert (both["decomposition_reconstruction_error"].abs() < 2e-6).all()
    assert (summary["max_decomposition_reconstruction_error"] < 2e-6).all()

    assert set(summary["model_kind"]) == {"tanh", "gru"}
    assert {
        "tau_star_drift", "tau_star_response", "abs_tau_star_error",
        "drift_crossed", "response_crossed", "n_sign_changes_d_w", "phase",
    } <= set(summary)

    # Crossing counts must be reported so that "never crossed" stays visible
    # instead of being averaged away.
    for label in ("drift", "response"):
        columns = {
            f"n_{label}_crossed", f"n_{label}_never_crossed", f"n_{label}_undecidable",
        }
        assert columns <= set(crossover)
        counted = crossover[list(columns)].sum(axis=1)
        assert (counted == crossover["n_seeds"]).all()

    # The machine-readable starvation phase must obey the same learnability gate
    # as the causal certificate.
    starved = summary["phase"] == "transfer_then_starvation"
    assert summary.loc[starved, "tail_causal_starvation_certified"].all()


def test_enl_phase_does_not_call_an_unlearnable_outcome_crossing_starvation():
    both = pd.DataFrame(
        {
            "condition": ["both"] * 3,
            "tau": [0.0, 1.0, 2.0],
            "m_s": [0.0, 0.1, 0.2],
            "m_w": [0.0, 1.0, -1.0],
            "d_w_equal_time_exact": [1.0, -1.0, -1.0],
            "d_w_equal_time_projected": [1.0, -1.0, -1.0],
            "d_w_matched": [1.0, -1.0, -1.0],
            "t_geom": [1.0, -1.0, -1.0],
            "s_ce": [0.0, 0.0, 0.0],
            "decomposition_reconstruction_error": [0.0] * 3,
            "equal_time_geometry_a": [0.0] * 3,
            "equal_time_field_a": [0.0] * 3,
            "equal_time_geometry_b": [0.0] * 3,
            "equal_time_field_b": [0.0] * 3,
            "equal_time_projection_residual": [0.0] * 3,
            "mode_residual_rms": [0.0] * 3,
            "equal_time_reconstruction_error_a": [0.0] * 3,
            "equal_time_reconstruction_error_b": [0.0] * 3,
            "equal_time_dominance_ordering_invariant": [True] * 3,
            "accuracy": [1.0] * 3,
            "gsi5": [0.0] * 3,
        }
    )
    weak = pd.DataFrame(
        {
            "condition": ["weak_only"] * 3,
            "tau": [0.0, 1.0, 2.0],
            "m_w": [0.0, 0.5, 1.0],
            "mode_residual_rms": [0.0] * 3,
        }
    )

    result = _enl_summary(pd.concat([both, weak], ignore_index=True), beta=2.0, phase_delay=0.25)

    assert result["drift_crossed"]
    assert result["response_crossed"]
    assert result["tail_strict_outcome_starvation"]
    assert not result["tail_causal_starvation_certified"]
    assert result["phase"] == "transfer_then_outcome_crossing_unlearnable"


def test_enl_refuses_the_sequential_trainer(tmp_path):
    """The sequential path cannot form the paired decomposition, so it must refuse."""
    config = _smoke_config("enl_smoke.yaml", tmp_path)
    config["training"]["paired_mode"] = "sequential"
    with pytest.raises(ValueError, match="requires training.paired_mode: lockstep"):
        run_enl(config)


def test_enl_tau_star_identity_check_is_documented_as_degenerate(tmp_path):
    """``abs_tau_star_error`` is a self-consistency check, not a theory test.

    ``d_w = t_geom - s_ce`` is an exact identity, so a crossover time derived from
    ``t_geom = s_ce`` agrees with one derived from ``d_w = 0`` by construction.
    This test pins that expectation so nobody later reports the agreement as
    evidence for Corollary A.
    """
    run_dir = run_enl(_smoke_config("enl_smoke.yaml", tmp_path))
    summary = pd.read_csv(run_dir / "summary.csv")
    gaps = summary["abs_tau_star_error"].to_numpy(dtype=float)
    assert np.all((gaps == 0.0) | np.isnan(gaps) | (gaps < 1e-9))


def test_e2r_acceptance_record_cannot_pass_and_names_its_blocks(tmp_path):
    """The correct output for this phase is an acceptance record that refuses."""
    config = _smoke_config("e2r_solver_checks.yaml", tmp_path)
    config["model"]["widths"] = [16, 32]
    config["task"]["learning_rates"] = [0.04, 0.02]
    config["task"]["n_samples"] = 256
    config["task"]["tau_max"] = 0.4
    config["task"]["frozen_geometries"] = [[[1.0, 0.0], [0.0, 1.0]]]
    run_dir = run_e2r(config)

    assert (run_dir / "config.resolved.yaml").is_file()
    assert (run_dir / "environment.json").is_file()
    assert (run_dir / "checks.csv").is_file()
    assert (run_dir / "blocked_checks.txt").is_file()
    _assert_figure_pair(run_dir, "e2r_solver_checks")

    payload = json.loads((run_dir / "e2r_acceptance.json").read_text())
    assert payload["passed"] is False
    assert "passed_reason" in payload
    for blocked in (
        "check_b_weak_only_reduction", "check_d_mse_solver",
        "check_e_internal_convergence", "check_f_finite_width_against_frozen_prediction",
    ):
        assert payload[blocked] == "blocked", blocked
        assert payload["blocked_on"][blocked]["obligations"]
        assert "e2_theorem.md" in payload["blocked_on"][blocked]["source"]

    # The implemented checks must carry real numbers, not the blocked sentinel.
    assert payload["check_a_zero_disorder"]["implemented"] is True
    assert payload["check_a_zero_disorder"]["recurrent_block_inert_everywhere"] is True
    assert np.isfinite(payload["check_a_zero_disorder"]["max_exact_seeded_relative_error"])
    assert payload["check_c_frozen_geometry"]["geometry_held_constant"] is True
    assert payload["check_c_frozen_geometry"]["max_relative_error"] < 1e-2


def test_e2r_outputs_never_use_the_phrase_dmft_validation(tmp_path):
    """Guard the vocabulary, not just the numbers."""
    config = _smoke_config("e2r_solver_checks.yaml", tmp_path)
    config["model"]["widths"] = [16]
    config["task"]["learning_rates"] = [0.04, 0.02]
    config["task"]["n_samples"] = 128
    config["task"]["tau_max"] = 0.2
    config["task"]["frozen_geometries"] = [[[1.0, 0.0], [0.0, 1.0]]]
    run_dir = run_e2r(config)

    for artifact in ("e2r_acceptance.json", "blocked_checks.txt", "checks.csv"):
        text = (run_dir / artifact).read_text().lower()
        assert "dmft validation" not in text, artifact
        assert "validated dmft" not in text, artifact
