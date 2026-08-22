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


def _e1_plot_tables(
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build complete E1 grids without turning censoring into zero delay."""
    positive = summary[summary["regime"] == "positive"]
    if positive.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    lags = sorted(positive["lag_separation"].unique(), reverse=True)
    strengths = sorted(positive["rho"].unique())
    auc = positive.pivot_table(
        index="lag_separation", columns="rho", values="weak_auc_gap", aggfunc="mean"
    ).reindex(index=lags, columns=strengths)
    uncensored = positive[~positive["right_censored"]]
    delay = uncensored.pivot_table(
        index="lag_separation", columns="rho", values="delta_tw", aggfunc="mean"
    ).reindex(index=lags, columns=strengths)
    censored = positive.pivot_table(
        index="lag_separation", columns="rho", values="right_censored", aggfunc="mean"
    ).reindex(index=lags, columns=strengths)
    return auc, delay, censored


def plot_e1(summary: pd.DataFrame, run_dir: Path, boundary=None) -> None:
    auc, delay, censored = _e1_plot_tables(summary)
    if auc.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    auc_labels = auc.map(lambda value: f"{value:.2f}")
    for row in auc.index:
        for column in auc.columns:
            if censored.loc[row, column] > 0:
                auc_labels.loc[row, column] += "†"
    sns.heatmap(
        auc,
        annot=auc_labels,
        fmt="",
        cmap="vlag",
        center=0.0,
        ax=axes[0],
        cbar_kws={"label": "Weak-only minus both-feature AUC"},
    )
    axes[0].set_title("Causal weak-trajectory gap")

    delay_labels = delay.map(lambda value: "—" if pd.isna(value) else f"{value:.2f}")
    axes[1].set_facecolor("#d9d9d9")
    sns.heatmap(
        delay,
        annot=delay_labels,
        fmt="",
        cmap="magma",
        ax=axes[1],
        cbar_kws={"label": "Hitting-time delay"},
    )
    axes[1].set_title("Delay among uncensored pairs")
    if boundary is not None and not boundary.empty:
        boundary = boundary[
            boundary.rho.isin(auc.columns) & boundary.lag_separation.isin(auc.index)
        ]
        rho_positions = [list(auc.columns).index(value) + 0.5 for value in boundary.rho]
        lag_rows = list(auc.index)
        lag_positions = [lag_rows.index(value) + 0.5 for value in boundary.lag_separation]
        for ax in axes:
            ax.plot(
                rho_positions, lag_positions, color="cyan", linewidth=2.0,
                label="Theory boundary", zorder=10,
            )
            ax.legend(loc="best")
    for ax in axes:
        ax.set_xlabel("Feature-strength ratio")
        ax.set_ylabel("Temporal separation")
    fig.text(
        0.5,
        -0.02,
        "† at least one paired run is right-censored; — no pair reached the target in both conditions",
        ha="center",
        fontsize=9,
    )
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


def plot_enl(trajectories: pd.DataFrame, summary: pd.DataFrame, run_dir: Path) -> None:
    """Plot the competing crossover terms per architecture.

    Left: the two competing contributions ``T_geom`` and ``S_CE`` against
    optimization time.  Right: their difference ``d_w`` with the zero line and the
    measured drift-level crossing.  A series that never crosses is annotated as
    such rather than being left to look like a crossing just outside the axis.
    """
    if trajectories.empty or summary.empty:
        return
    both = trajectories[trajectories["condition"] == "both"]
    if both.empty or "d_w" not in both:
        return
    kinds = sorted(both["model_kind"].unique())
    fig, axes = plt.subplots(
        len(kinds), 2, figsize=(11.5, 3.6 * len(kinds)), squeeze=False, constrained_layout=True
    )
    for row, kind in enumerate(kinds):
        subset = both[both["model_kind"] == kind]
        averaged = subset.groupby("tau", as_index=False)[["t_geom", "s_ce", "d_w"]].mean()
        left, right = axes[row][0], axes[row][1]
        left.plot(averaged.tau, averaged.t_geom, label=r"$T_{geom}$ (transfer)")
        left.plot(averaged.tau, averaged.s_ce, label=r"$S_{CE}$ (suppression)")
        left.set(xlabel="Optimization time", ylabel="Drift contribution",
                 title=f"{kind}: competing terms")
        left.legend(fontsize=8)

        right.axhline(0.0, color="0.5", linewidth=1.0, linestyle="--")
        right.plot(averaged.tau, averaged.d_w, color="C3", label=r"$d_w = T_{geom} - S_{CE}$")
        crossings = summary[summary["model_kind"] == kind]["tau_star_drift"].to_numpy(dtype=float)
        finite = crossings[np.isfinite(crossings)]
        if len(finite):
            right.axvline(
                float(finite.mean()), color="C0", linewidth=1.5,
                label=rf"mean $\tau^*$ = {finite.mean():.3g}",
            )
        else:
            right.text(
                0.5, 0.08, "no sign change observed", transform=right.transAxes,
                ha="center", fontsize=9, color="0.3",
            )
        right.set(xlabel="Optimization time", ylabel="Causal weak-drift difference",
                  title=f"{kind}: transfer above zero, starvation below")
        right.legend(fontsize=8)
    _save_all(fig, run_dir / "enl_crossover")
