import torch

from gradient_starvation.data.synthetic import SyntheticTaskSpec, make_paired_task
from gradient_starvation.training import train_paired


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

