from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import mean_confidence_interval
from .utils import write_environment


REQUIRED_METHODS = {"erm", "counterfactual_drift"}


def _method_label(method: str) -> str:
    return "CDC" if method == "counterfactual_drift" else method.upper()


def summarize_width_runs(width_runs: Mapping[int, Path], output_dir: Path) -> Path:
    """Validate and combine matched-seed ERM/CDC runs across network widths."""
    if not width_runs:
        raise ValueError("At least one width run is required.")
    output_dir.mkdir(parents=True, exist_ok=False)
    frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, float | int]] = []
    sources: dict[str, str] = {}
    for width, run_dir in sorted(width_runs.items()):
        run_dir = Path(run_dir)
        summary = pd.read_csv(run_dir / "summary.csv")
        trajectories = pd.read_csv(run_dir / "trajectories.csv")
        if not REQUIRED_METHODS <= set(summary["method"]):
            raise ValueError(f"Width {width} does not contain both ERM and CDC.")
        selected = summary[summary["method"].isin(REQUIRED_METHODS)].copy()
        seed_sets = selected.groupby("method")["seed"].apply(set)
        if seed_sets["erm"] != seed_sets["counterfactual_drift"]:
            raise ValueError(f"Width {width} does not use matched ERM/CDC seeds.")
        selected["width"] = int(width)
        frames.append(selected)
        cdc = trajectories[
            (trajectories["method"] == "counterfactual_drift")
            & (trajectories["condition"] == "both")
        ]
        diagnostics.append(
            {
                "width": int(width),
                "n_logged_checks": int(len(cdc)),
                "feasible_fraction": float(cdc["cdc_feasible"].astype(bool).mean()),
                "target_met_fraction": float(cdc["cdc_target_met"].astype(bool).mean()),
                "max_abs_strong_drift_change": float(cdc["cdc_strong_drift_change"].abs().max()),
            }
        )
        sources[str(width)] = str(run_dir)

    combined = pd.concat(frames, ignore_index=True)
    keys = ["width", "model_kind", "method", "rho", "lag_separation", "regime", "seed"]
    if combined.duplicated(keys).any():
        raise ValueError("Duplicate width-validation run keys were found.")
    combined.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir / "diagnostics.csv", index=False)

    aggregate_rows: list[dict[str, float | int | str]] = []
    paired_rows: list[dict[str, float | int | bool]] = []
    for width, width_group in combined.groupby("width"):
        for method, group in width_group.groupby("method"):
            mean, low, high = mean_confidence_interval(group["weak_auc_gap"].to_numpy())
            aggregate_rows.append(
                {
                    "width": int(width),
                    "method": method,
                    "n_seeds": int(group["seed"].nunique()),
                    "weak_auc_gap_mean": mean,
                    "weak_auc_gap_ci95_low": low,
                    "weak_auc_gap_ci95_high": high,
                    "final_both_m_s_mean": float(group["final_both_m_s"].mean()),
                    "final_both_m_w_mean": float(group["final_both_m_w"].mean()),
                    "final_accuracy_mean": float(group["final_accuracy"].mean()),
                }
            )
        paired = width_group.pivot(index="seed", columns="method", values="weak_auc_gap")
        change = (paired["counterfactual_drift"] - paired["erm"]).to_numpy()
        mean, low, high = mean_confidence_interval(change)
        paired_rows.append(
            {
                "width": int(width),
                "n_seeds": int(len(change)),
                "cdc_minus_erm_mean": mean,
                "cdc_minus_erm_ci95_low": low,
                "cdc_minus_erm_ci95_high": high,
                "all_seeds_improved": bool(np.all(change < 0)),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    paired_effects = pd.DataFrame(paired_rows)
    aggregate.to_csv(output_dir / "aggregate.csv", index=False)
    paired_effects.to_csv(output_dir / "paired_effects.csv", index=False)
    (output_dir / "source_runs.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    write_environment(output_dir / "environment.json")
    _plot_width_robustness(aggregate, paired_effects, output_dir)
    return output_dir


def _plot_width_robustness(
    aggregate: pd.DataFrame, paired_effects: pd.DataFrame, output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
    colors = {"erm": "#4a4a4a", "counterfactual_drift": "#2f6fa3"}
    markers = {"erm": "o", "counterfactual_drift": "s"}
    offsets = {"erm": -2.0, "counterfactual_drift": 2.0}
    for method in ["erm", "counterfactual_drift"]:
        group = aggregate[aggregate["method"] == method].sort_values("width")
        x = group["width"].to_numpy(dtype=float) + offsets[method]
        mean = group["weak_auc_gap_mean"].to_numpy()
        error = np.vstack(
            (
                mean - group["weak_auc_gap_ci95_low"].to_numpy(),
                group["weak_auc_gap_ci95_high"].to_numpy() - mean,
            )
        )
        axes[0].errorbar(
            x,
            mean,
            yerr=error,
            fmt=markers[method],
            color=colors[method],
            capsize=4,
            linewidth=1.8,
            markersize=7,
            label=_method_label(method),
        )
    axes[0].axhline(0, color="#777777", linewidth=1)
    axes[0].set_title("Weak-trajectory gap by width")
    axes[0].set_ylabel("Weak-only minus both-feature AUC")
    axes[0].legend(frameon=False)

    paired = paired_effects.sort_values("width")
    paired_mean = paired["cdc_minus_erm_mean"].to_numpy()
    paired_error = np.vstack(
        (
            paired_mean - paired["cdc_minus_erm_ci95_low"].to_numpy(),
            paired["cdc_minus_erm_ci95_high"].to_numpy() - paired_mean,
        )
    )
    axes[1].errorbar(
        paired["width"],
        paired_mean,
        yerr=paired_error,
        fmt="s",
        color="#2f6fa3",
        capsize=4,
        linewidth=1.8,
        markersize=7,
    )
    axes[1].axhline(0, color="#777777", linewidth=1)
    axes[1].set_title("Paired CDC effect")
    axes[1].set_ylabel("CDC minus ERM AUC gap")
    for ax in axes:
        ax.set_xlabel("Network width")
        ax.set_xticks(sorted(aggregate["width"].unique()))
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Tanh CDC width robustness", fontsize=15)
    fig.savefig(output_dir / "e3_tanh_cdc_width_robustness.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "e3_tanh_cdc_width_robustness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
