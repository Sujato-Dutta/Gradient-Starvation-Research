from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save_all(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_e1(summary: pd.DataFrame, run_dir: Path, boundary=None) -> None:
    positive = summary[(summary["regime"] == "positive") & (~summary["right_censored"])]
    if positive.empty:
        return
    table = positive.pivot_table(
        index="lag_separation", columns="rho", values="delta_tw", aggfunc="mean"
    ).sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    if boundary is not None and not boundary.empty:
        boundary = boundary[
            boundary.rho.isin(table.columns) & boundary.lag_separation.isin(table.index)
        ]
        rho_positions = [list(table.columns).index(value) + 0.5 for value in boundary.rho]
        lag_rows = list(table.index)
        lag_positions = [lag_rows.index(value) + 0.5 for value in boundary.lag_separation]
        ax.plot(
            rho_positions, lag_positions, color='cyan', linewidth=2.0,
            label='Theory boundary', zorder=10,
        )
        ax.legend(loc='best')
    sns.heatmap(table, annot=True, fmt=".2f", cmap="magma", ax=ax, cbar_kws={"label": "Causal delay"})
    ax.set_title("Weak-feature hitting-time delay")
    ax.set_xlabel("Feature-strength ratio")
    ax.set_ylabel("Temporal separation")
    _save_all(fig, run_dir / "e1_phase_diagram")


def plot_e2(trajectories: pd.DataFrame, summary: pd.DataFrame, run_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    first_point = trajectories[["rho", "lag_separation"]].drop_duplicates().iloc[0]
    selected = trajectories[
        (trajectories.condition == trajectories.condition.iloc[0])
        &
        (trajectories["rho"] == first_point.rho)
        & (trajectories["lag_separation"] == first_point.lag_separation)
    ]
    for (source, width), group in selected.groupby(["source", "width"], dropna=False):
        averaged = group.groupby("tau", as_index=False)["m_w"].mean()
        label = source if source == "particle_closure" else f"network N={int(width)}"
        axes[0].plot(averaged.tau, averaged.m_w, label=label)
    axes[0].set(xlabel="Optimization time", ylabel="Weak response", title="Joint trajectories")
    axes[0].legend(fontsize=8)
    width_summary = summary.groupby("width", as_index=False)["trajectory_rmse"].mean()
    axes[1].plot(width_summary.width, width_summary.trajectory_rmse, "o-")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set(xlabel="Width", ylabel="Trajectory RMSE", title="Width convergence")
    fig.tight_layout()
    _save_all(fig, run_dir / "e2_field_agreement")


def plot_e3(summary: pd.DataFrame, run_dir: Path) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    sns.barplot(data=summary, x="method", y="weak_auc_gap", hue="model_kind", ax=axes[0], errorbar=("ci", 95))
    sns.barplot(data=summary, x="method", y="final_gsi5", hue="model_kind", ax=axes[1], errorbar=("ci", 95))
    sns.barplot(data=summary, x="method", y="final_accuracy", hue="model_kind", ax=axes[2], errorbar=("ci", 95))
    axes[0].set_title("Causal weak-trajectory gap")
    axes[1].set_title("Gradient concentration")
    axes[2].set_title("Average accuracy")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        legend = axis.get_legend()
        if legend is not None and axis is not axes[2]:
            legend.remove()
    fig.tight_layout()
    _save_all(fig, run_dir / "e3_mitigation")
