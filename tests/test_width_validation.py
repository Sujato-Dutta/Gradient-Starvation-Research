from __future__ import annotations

from pathlib import Path

import pandas as pd

from gradient_starvation.width_validation import summarize_width_runs


def _write_run(run_dir: Path, width: int) -> None:
    run_dir.mkdir()
    summary_rows = []
    trajectory_rows = []
    for seed in [0, 1]:
        for method, gap in [("erm", 3.0 + seed), ("counterfactual_drift", -1.0 - seed)]:
            summary_rows.append(
                {
                    "model_kind": "tanh",
                    "method": method,
                    "rho": 4.0,
                    "lag_separation": 2,
                    "regime": "positive",
                    "seed": seed,
                    "weak_auc_gap": gap,
                    "final_both_m_s": 1.0,
                    "final_both_m_w": 2.0,
                    "final_accuracy": 1.0,
                }
            )
            for step in [0, 1]:
                trajectory_rows.append(
                    {
                        "method": method,
                        "condition": "both",
                        "seed": seed,
                        "step": step,
                        "cdc_feasible": method == "counterfactual_drift",
                        "cdc_target_met": method == "counterfactual_drift",
                        "cdc_strong_drift_change": 0.0,
                    }
                )
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False)
    pd.DataFrame(trajectory_rows).to_csv(run_dir / "trajectories.csv", index=False)


def test_summarize_width_runs_writes_validated_outputs(tmp_path):
    width_runs = {}
    for width in [32, 64]:
        run_dir = tmp_path / f"width_{width}"
        _write_run(run_dir, width)
        width_runs[width] = run_dir

    output = summarize_width_runs(width_runs, tmp_path / "combined")

    assert len(pd.read_csv(output / "summary.csv")) == 8
    paired = pd.read_csv(output / "paired_effects.csv")
    assert paired["all_seeds_improved"].all()
    assert (paired["cdc_minus_erm_mean"] < 0).all()
    assert (output / "diagnostics.csv").is_file()
    assert (output / "e3_tanh_cdc_width_robustness.png").is_file()
    assert (output / "e3_tanh_cdc_width_robustness.pdf").is_file()
