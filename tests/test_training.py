import pytest
import torch

from gradient_starvation.data.synthetic import SyntheticTaskSpec, make_paired_task
from gradient_starvation.models.recurrent import build_model
from gradient_starvation.training import (
    _snapshot_state,
    train_paired,
    train_paired_lockstep,
)


def test_small_paired_run_logs_complete_diagnostics():
    spec = SyntheticTaskSpec(sequence_length=6, n_samples=32, rho=2, lag_separation=2)
    both, weak = make_paired_task(spec, seed=0)
    history, states = train_paired(
        {"kind": "low_rank_linear", "width": 8, "bulk_gain": 0.4, "rank": 2},
        both,
        weak,
        {"steps": 2, "learning_rate": 0.01, "log_every": 1, "full_batch": True},
        {"method": "erm"},
        seed=0,
        identity_steps=1,
    )
    assert len(history) == 6
    assert {row["condition"] for row in history} == {"both", "weak_only"}
    assert "G_sw" in history[0]
    assert set(states) == {"both", "weak_only"}


def test_zero_step_paired_run_reuses_the_exact_initial_model_state():
    spec = SyntheticTaskSpec(sequence_length=5, n_samples=16, rho=2, lag_separation=2)
    both, weak = make_paired_task(spec, seed=2)
    _, states = train_paired(
        {"kind": "low_rank_linear", "width": 6, "bulk_gain": 0.3, "rank": 2},
        both,
        weak,
        {"steps": 0, "learning_rate": 0.01, "log_every": 1, "full_batch": True},
        {"method": "erm"},
        seed=3,
    )
    assert states["both"].keys() == states["weak_only"].keys()
    for name in states["both"]:
        assert torch.equal(states["both"][name], states["weak_only"][name]), name


def test_counterfactual_drift_training_logs_and_meets_instantaneous_target():
    spec = SyntheticTaskSpec(sequence_length=6, n_samples=32, rho=3, lag_separation=2)
    both, weak = make_paired_task(spec, seed=5)
    history, _ = train_paired(
        {"kind": "low_rank_linear", "width": 8, "bulk_gain": 0.3, "rank": 2},
        both,
        weak,
        {
            "steps": 2,
            "learning_rate": 0.01,
            "log_every": 1,
            "full_batch": True,
            "weight_decay": 0.0,
        },
        {"method": "counterfactual_drift"},
        seed=6,
    )
    both_rows = [row for row in history if row["condition"] == "both"]
    assert len(both_rows) == 3
    assert all(row["cdc_feasible"] for row in both_rows)
    assert all(row["cdc_target_met"] for row in both_rows)
    assert max(abs(row["cdc_strong_drift_change"]) for row in both_rows) < 1e-5


def test_counterfactual_drift_training_on_dense_linear_preserves_strong_drift():
    """Close the ``dense_linear`` + CDC coverage gap.

    CDC coverage otherwise centres on ``low_rank_linear`` and ``tanh``, yet
    ``dense_linear`` is the family the finite-width geometry theorem is stated
    for.  The guarantee under test is the instantaneous one: weak drift reaches
    the matched weak-only target while strong drift is unchanged.
    """
    spec = SyntheticTaskSpec(sequence_length=6, n_samples=32, rho=3, lag_separation=2)
    both, weak = make_paired_task(spec, seed=11)
    history, states = train_paired(
        {"kind": "dense_linear", "width": 8, "bulk_gain": 0.3},
        both,
        weak,
        {
            "steps": 2,
            "learning_rate": 0.01,
            "log_every": 1,
            "full_batch": True,
            "weight_decay": 0.0,
        },
        {"method": "counterfactual_drift"},
        seed=12,
    )
    both_rows = [row for row in history if row["condition"] == "both"]
    assert len(both_rows) == 3
    assert all(row["cdc_feasible"] for row in both_rows)
    assert all(row["cdc_target_met"] for row in both_rows)
    assert max(abs(row["cdc_strong_drift_change"]) for row in both_rows) < 1e-5
    # The dense family trains W, B and c, so the shadow model must be a real
    # second model rather than an alias of the both-feature parameters.
    assert set(states) == {"both", "weak_only"}
    assert not torch.equal(states["both"]["recurrent"], states["weak_only"]["recurrent"])


def test_counterfactual_drift_rejects_guarantee_breaking_optimizers():
    """Weight decay and clipping would invalidate the post-correction guarantee."""
    spec = SyntheticTaskSpec(sequence_length=5, n_samples=16, rho=2, lag_separation=1)
    both, weak = make_paired_task(spec, seed=13)
    model = {"kind": "dense_linear", "width": 6, "bulk_gain": 0.3}
    base = {"steps": 1, "learning_rate": 0.01, "log_every": 1, "full_batch": True}

    with pytest.raises(ValueError, match="weight_decay"):
        train_paired(
            model, both, weak, {**base, "weight_decay": 0.1},
            {"method": "counterfactual_drift"}, seed=14,
        )
    with pytest.raises(ValueError, match="gradient_clip"):
        train_paired(
            model, both, weak, {**base, "gradient_clip": 1.0},
            {"method": "counterfactual_drift"}, seed=14,
        )


def _trajectory_by_condition(history):
    """Group logged rows by condition, ordered by step."""
    grouped: dict[str, list[dict]] = {}
    for row in history:
        grouped.setdefault(row["condition"], []).append(row)
    return {key: sorted(rows, key=lambda row: row["step"]) for key, rows in grouped.items()}


@pytest.mark.parametrize(
    "model_config",
    [
        {"kind": "dense_linear", "width": 10, "bulk_gain": 0.4},
        {"kind": "low_rank_linear", "width": 10, "bulk_gain": 0.4, "rank": 2},
        {"kind": "tanh", "width": 10},
    ],
    ids=["dense_linear", "low_rank_linear", "tanh"],
)
def test_lockstep_reproduces_sequential_erm_trajectories(model_config):
    """Lockstep must be a pure restructuring of sequential ERM, not a new method.

    This is a stop condition for the crossover instrumentation work: if the two
    paths disagree, the mechanistic columns logged by the lockstep trainer are not
    measuring the same experiment the published E1/E3 results came from.  The
    tolerance is deliberately tight.  Do not loosen it to force a pass.
    """
    spec = SyntheticTaskSpec(sequence_length=6, n_samples=48, rho=3, lag_separation=2)
    both, weak = make_paired_task(spec, seed=21)
    training = {
        "steps": 12,
        "learning_rate": 0.05,
        "log_every": 3,
        "full_batch": True,
        "weight_decay": 0.0,
    }

    sequential, sequential_states = train_paired(
        model_config, both, weak, training, {"method": "erm"}, seed=31, identity_steps=2
    )
    lockstep, lockstep_states = train_paired(
        model_config,
        both,
        weak,
        {**training, "paired_mode": "lockstep"},
        {"method": "erm"},
        seed=31,
        identity_steps=2,
    )

    sequential_rows = _trajectory_by_condition(sequential)
    lockstep_rows = _trajectory_by_condition(lockstep)
    assert set(sequential_rows) == set(lockstep_rows) == {"both", "weak_only"}

    compared = ["m_s", "m_w", "loss", "accuracy", "drift_s", "drift_w",
                "G_ss", "G_sw", "G_ww", "g_s", "g_w", "margin_mean", "gsi5"]
    for condition in ("both", "weak_only"):
        left, right = sequential_rows[condition], lockstep_rows[condition]
        assert len(left) == len(right), condition
        for a, b in zip(left, right):
            assert a["step"] == b["step"]
            for column in compared:
                assert a[column] == pytest.approx(b[column], rel=1e-9, abs=1e-11), (
                    f"{condition}/{column} diverged at step {a['step']}: "
                    f"sequential={a[column]!r} lockstep={b[column]!r}"
                )

    # Final parameters must match too, not just the logged scalars.
    for condition in ("both", "weak_only"):
        assert sequential_states[condition].keys() == lockstep_states[condition].keys()
        for name in sequential_states[condition]:
            torch.testing.assert_close(
                sequential_states[condition][name],
                lockstep_states[condition][name],
                rtol=0,
                atol=0,
                msg=f"{condition}/{name} parameters diverged",
            )


def test_lockstep_shares_the_initialization_between_conditions():
    """Both models must start from bitwise-identical parameters."""
    spec = SyntheticTaskSpec(sequence_length=5, n_samples=16, rho=2, lag_separation=1)
    both, weak = make_paired_task(spec, seed=23)
    _, states = train_paired(
        {"kind": "dense_linear", "width": 6, "bulk_gain": 0.3},
        both,
        weak,
        {"steps": 0, "learning_rate": 0.01, "log_every": 1, "full_batch": True,
         "paired_mode": "lockstep"},
        {"method": "erm"},
        seed=24,
    )
    for name in states["both"]:
        assert torch.equal(states["both"][name], states["weak_only"][name]), name


def test_lockstep_logs_exact_crossover_columns():
    """The crossover reparameterization must reconstruct at every logged step."""
    spec = SyntheticTaskSpec(sequence_length=6, n_samples=48, rho=4, lag_separation=2)
    both, weak = make_paired_task(spec, seed=25)
    history, _ = train_paired(
        {"kind": "dense_linear", "width": 10, "bulk_gain": 0.4},
        both,
        weak,
        {"steps": 6, "learning_rate": 0.05, "log_every": 1, "full_batch": True,
         "paired_mode": "lockstep"},
        {"method": "erm"},
        seed=26,
    )
    both_rows = [row for row in history if row["condition"] == "both"]
    weak_rows = [row for row in history if row["condition"] == "weak_only"]
    assert len(both_rows) == len(weak_rows) == 7

    for row in both_rows:
        assert row["t_geom"] - row["s_ce"] == pytest.approx(row["d_w"], rel=2e-5, abs=2e-6)
        # float32 arithmetic: the two channels are summed in float32 inside the
        # decomposition, then widened to Python floats independently, so the
        # tolerance must be float32-scale rather than float64-scale.
        assert row["t_geom_self"] + row["t_geom_cross"] == pytest.approx(
            row["t_geom"], rel=1e-6, abs=1e-9
        )
        assert abs(row["decomposition_reconstruction_error"]) < 2e-6

    # The weak-only rows carry no crossover columns; the decomposition is a
    # property of the pair and is attached to the both-feature row only.
    assert all("d_w" not in row for row in weak_rows)


def test_lockstep_rejects_counterfactual_drift():
    """CDC has its own lockstep path; asking for both would be ambiguous."""
    spec = SyntheticTaskSpec(sequence_length=5, n_samples=16, rho=2, lag_separation=1)
    both, weak = make_paired_task(spec, seed=27)
    with pytest.raises(ValueError, match="already trains in lockstep"):
        train_paired_lockstep(
            {"kind": "dense_linear", "width": 6, "bulk_gain": 0.3},
            both,
            weak,
            {"steps": 1, "learning_rate": 0.01, "log_every": 1, "full_batch": True},
            {"method": "counterfactual_drift"},
            seed=28,
        )


def test_sequential_paired_states_are_not_aliased_between_conditions():
    """Regression test for a state-snapshot aliasing bug.

    ``tensor.detach().cpu()`` shares storage with the live parameter on CPU.  The
    sequential paired trainer trains the both-feature condition, resets the model
    with ``load_state_dict``, then trains weak-only in the same module.  Without a
    ``clone()`` in the snapshot, the returned both-feature state was silently
    overwritten and ended up bitwise equal to the weak-only state.

    No published result was affected -- every ``experiments.py`` call site discards
    the returned states -- but the snapshot was wrong and would have corrupted any
    future consumer.
    """
    spec = SyntheticTaskSpec(sequence_length=6, n_samples=48, rho=3, lag_separation=2)
    both, weak = make_paired_task(spec, seed=21)
    _, states = train_paired(
        {"kind": "dense_linear", "width": 10, "bulk_gain": 0.4},
        both,
        weak,
        {"steps": 8, "learning_rate": 0.05, "log_every": 4, "full_batch": True},
        {"method": "erm"},
        seed=31,
    )
    # After a genuine training run the two conditions must have diverged.
    assert not torch.equal(states["both"]["readout"], states["weak_only"]["readout"]), (
        "both-feature and weak-only snapshots are identical after training, which "
        "means the both-feature snapshot was aliased and overwritten"
    )


def test_snapshot_state_is_independent_of_later_mutation():
    """A snapshot must survive in-place mutation of the source model."""
    model = build_model({"kind": "dense_linear", "width": 5, "bulk_gain": 0.3})
    snapshot = _snapshot_state(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    for name, value in snapshot.items():
        assert not torch.equal(value, model.state_dict()[name]), name
