import pytest
import torch

from gradient_starvation.data.synthetic import SyntheticTaskSpec, make_paired_task
from gradient_starvation.models.recurrent import build_model
from gradient_starvation.training import _snapshot_state, train_paired


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
