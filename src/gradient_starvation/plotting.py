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


def _final_causal_grid_tables(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate the audited final certificate and stricter at-hit subtype."""
    required = {
        "source", "regime", "rho", "lag_separation", "weak_gate",
        "any_time_certificate", "at_hit_certificate",
    }
    missing = required - set(records)
    if missing:
        raise ValueError(f"Final causal grid is missing columns: {sorted(missing)}")
    positive = records[(records.source == "dense_e1") & (records.regime == "positive")]
    if positive.empty:
        raise ValueError("Final causal grid has no positive dense-E1 records.")
    lags = sorted(positive.lag_separation.unique(), reverse=True)
    strengths = sorted(positive.rho.unique())
    codes = pd.DataFrame(index=lags, columns=strengths, dtype=float)
    labels = pd.DataFrame(index=lags, columns=strengths, dtype=object)
    for lag in lags:
        for rho in strengths:
            cell = positive[
                (positive.lag_separation == lag) & (positive.rho == rho)
            ]
            if cell.empty:
                raise ValueError(f"Missing audited cell lag={lag}, rho={rho}.")
            count = len(cell)
            gate = int(cell.weak_gate.eq(True).sum())  # noqa: E712
            causal = int(cell.any_time_certificate.eq(True).sum())  # noqa: E712
            at_hit = int(cell.at_hit_certificate.eq(True).sum())  # noqa: E712
            if gate == 0:
                code = 0  # indeterminate
            elif causal == 0:
                code = 1  # gate passed, no causal certificate
            elif causal < count:
                code = 2  # mixed across seeds
            else:
                code = 3  # all seeds certified
            codes.loc[lag, rho] = code
            labels.loc[lag, rho] = f"C {causal}/{count}\nAH {at_hit}/{count}"
    return codes, labels


def plot_final_causal_grid(records: pd.DataFrame, output_base: Path) -> None:
    """Plot the final causal definition rather than historical delay categories."""
    codes, labels = _final_causal_grid_tables(records)
    fig, ax = plt.subplots(figsize=(8.5, 5.3), constrained_layout=True)
    cmap = matplotlib.colors.ListedColormap(
        ["#d9d9d9", "#d6604d", "#92c5de", "#2166ac"]
    )
    sns.heatmap(
        codes, annot=labels, fmt="", cmap=cmap, vmin=-0.5, vmax=3.5,
        linewidths=0.8, linecolor="white", ax=ax,
        annot_kws={"fontsize": 9, "fontweight": "bold"},
        cbar_kws={"label": "Final causal status", "ticks": [0, 1, 2, 3]},
    )
    ax.collections[0].colorbar.set_ticklabels(
        ["indeterminate", "not certified", "mixed", "certified"]
    )
    ax.set(
        xlabel="Feature-strength ratio rho",
        ylabel="Temporal separation",
        title="Learnability-gated causal starvation and at-hit robustness",
    )
    fig.text(
        0.5, -0.015,
        "C = causal certificate (negative gap at some positive time); "
        "AH = stricter negative gap at the weak-only first hit.",
        ha="center", fontsize=8.5,
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    _save_all(fig, output_base)


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
        # `closure_reference` is the current label; `particle_closure` appears only in
        # run directories predating the rename and is still recognized so those
        # figures do not mislabel the reference as a trained network.
        label = (
            source
            if source in {"closure_reference", "particle_closure"}
            else f"network N={int(width)}"
        )
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

    Left: the matched-state projected terms ``T_geom`` and ``S_CE``. Right: the
    exact direct-autograd equal-time derivative used by the theorem, alongside the
    projected and matched-state diagnostics. A series that never crosses is
    annotated rather than extrapolated beyond the observed horizon.
    """
    if trajectories.empty or summary.empty:
        return
    both = trajectories[trajectories["condition"] == "both"]
    if both.empty or "d_w_equal_time_exact" not in both:
        return
    kinds = sorted(both["model_kind"].unique())
    fig, axes = plt.subplots(
        len(kinds), 2, figsize=(11.5, 3.6 * len(kinds)), squeeze=False, constrained_layout=True
    )
    columns = [
        "t_geom", "s_ce", "d_w_matched", "d_w_equal_time_projected",
        "d_w_equal_time_exact",
    ]
    for row, kind in enumerate(kinds):
        subset = both[both["model_kind"] == kind]
        averaged = subset.groupby("tau", as_index=False)[columns].mean()
        left, right = axes[row][0], axes[row][1]
        left.plot(averaged.tau, averaged.t_geom, label=r"$T_{geom}$ (transfer)")
        left.plot(averaged.tau, averaged.s_ce, label=r"$S_{CE}$ (suppression)")
        left.set(xlabel="Optimization time", ylabel="Projected drift contribution",
                 title=f"{kind}: matched-state projected identity")
        left.legend(fontsize=8)

        right.axhline(0.0, color="0.5", linewidth=1.0, linestyle="--")
        # Direct autograd is the theorem derivative for nonlinear probe responses.
        # Projected and matched diagnostics remain visible under distinct styles.
        right.plot(
            averaged.tau, averaged.d_w_equal_time_exact, color="C3",
            label=r"$d_w$ exact cross-kernel $=\frac{d}{d\tau}[m_w^B-m_w^W]$",
        )
        right.plot(
            averaged.tau, averaged.d_w_equal_time_projected, color="C1", linestyle="--",
            label=r"$d_w$ projected $Gg$",
        )
        right.plot(
            averaged.tau, averaged.d_w_matched, color="C7", linestyle=":",
            label=r"$d_w$ matched state $=T_{geom}-S_{CE}$",
        )
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
                  title=f"{kind}: transfer above zero, rate suppression below")
        right.legend(fontsize=8)
    _save_all(fig, run_dir / "enl_crossover")


# Ordered so the colour ramp reads unlearnable/degenerate (artifact buckets)
# through transfer -> neutral -> starvation (the physical axis).
_REGION_ORDER = ("unlearnable", "degenerate", "transfer", "neutral", "starvation")
_REGION_COLOURS = ("#bdbdbd", "#7b6888", "#2c7fb8", "#f0e442", "#d7301f")


def plot_e1_regions(summary: pd.DataFrame, run_dir: Path, boundary=None) -> None:
    """Categorical learnability-gated region map.

    Each cell shows the modal regime across seeds plus the seed agreement, so a
    split cell cannot be mistaken for a unanimous one.  Cells are never blank-filled
    as a numeric zero; an absent cell stays visibly empty.
    """
    if summary.empty or "regime_class" not in summary:
        return
    positive = summary[summary["regime"] == "positive"]
    if positive.empty:
        return
    lags = sorted(positive["lag_separation"].unique(), reverse=True)
    strengths = sorted(positive["rho"].unique())
    index = {value: position for position, value in enumerate(_REGION_ORDER)}

    codes = pd.DataFrame(index=lags, columns=strengths, dtype=float)
    labels = pd.DataFrame("", index=lags, columns=strengths, dtype=object)
    for lag in lags:
        for rho in strengths:
            cell = positive[
                (positive["lag_separation"] == lag) & (positive["rho"] == rho)
            ]
            if cell.empty:
                codes.loc[lag, rho] = np.nan
                labels.loc[lag, rho] = ""
                continue
            counts = cell["regime_class"].value_counts()
            modal = counts.index[0]
            codes.loc[lag, rho] = index.get(modal, np.nan)
            labels.loc[lag, rho] = (
                f"{modal}\n{int(counts.iloc[0])}/{int(counts.sum())}"
            )

    fig, ax = plt.subplots(figsize=(1.7 * len(strengths) + 3.4, 1.3 * len(lags) + 2.4),
                           constrained_layout=True)
    ax.set_facecolor("#f7f7f7")
    cmap = matplotlib.colors.ListedColormap(_REGION_COLOURS)
    sns.heatmap(
        codes.astype(float), annot=labels, fmt="", cmap=cmap,
        vmin=-0.5, vmax=len(_REGION_ORDER) - 0.5, ax=ax, linewidths=0.6,
        linecolor="white", annot_kws={"fontsize": 8},
        cbar_kws={"label": "Causal regime", "ticks": range(len(_REGION_ORDER))},
    )
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticklabels(_REGION_ORDER)
    if boundary is not None and not boundary.empty:
        overlay = boundary[
            boundary.rho.isin(strengths) & boundary.lag_separation.isin(lags)
        ]
        if not overlay.empty:
            ax.plot(
                [strengths.index(value) + 0.5 for value in overlay.rho],
                [lags.index(value) + 0.5 for value in overlay.lag_separation],
                color="black", linewidth=2.0, label="Theory boundary", zorder=10,
            )
            ax.legend(loc="best")
    ax.set(xlabel="Feature-strength ratio", ylabel="Temporal separation",
           title="Causal regime after the weak-only learnability gate")
    fig.text(
        0.5, -0.03,
        "Cell label is the modal regime and seed agreement. 'unlearnable' means the "
        "weak-only counterfactual never reached the target, so the point is "
        "indeterminate rather than starved; 'degenerate' means the target was already "
        "met at initialization.",
        ha="center", fontsize=8, wrap=True,
    )
    _save_all(fig, run_dir / "e1_causal_regions")


def plot_e2r(checks: pd.DataFrame, run_dir: Path) -> None:
    """Report the two implementable solver checks, and label the blocked ones.

    The right-hand panel is intentionally a text panel rather than an empty axis:
    a blank plot reads as missing data, whereas the blocked checks are a deliberate
    and documented state.
    """
    if checks.empty:
        return
    check_a = checks[checks["check"] == "check_a_zero_disorder"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)

    if not check_a.empty:
        for rate, group in check_a.groupby("learning_rate"):
            ordered = group.sort_values("width")
            axes[0].plot(
                ordered.width, ordered.wide_limit_relative_error, "o-",
                label=rf"wide-limit init, $\eta$={rate:g}",
            )
        for rate, group in check_a.groupby("learning_rate"):
            ordered = group.sort_values("width")
            axes[0].plot(
                ordered.width, ordered.exact_seeded_relative_error, "s--",
                label=rf"exact init, $\eta$={rate:g}",
            )
        axes[0].set_xscale("log", base=2)
        axes[0].set_yscale("log")
        axes[0].set(
            xlabel="Width", ylabel="Max relative mode error",
            title="Check A: zero disorder\nsolid = initial-condition concentration, "
                  "dashed = O($\\eta$) discretization",
        )
        axes[0].legend(fontsize=7)

    axes[1].axis("off")
    axes[1].text(
        0.02, 0.98,
        "Blocked checks\n"
        "\n"
        "B  weak-only reduction      obligations 1, 2\n"
        "D  MSE solver branch        obligations 1, 2\n"
        "E  internal convergence     obligations 2, 3\n"
        "F  finite-width vs frozen   obligations 1-4\n"
        "\n"
        'Source: research_scope/e2_theorem.md\n'
        '        § "Proof obligations"\n'
        "\n"
        "e2r_acceptance.json reports passed = false.\n"
        "No run in this phase may be described as\n"
        "DMFT validation: there is no independent\n"
        "solver to freeze a prediction from.",
        transform=axes[1].transAxes, va="top", ha="left",
        family="monospace", fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "#f0f0f0", "edgecolor": "#999999"},
    )
    _save_all(fig, run_dir / "e2r_solver_checks")
