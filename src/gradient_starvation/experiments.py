from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from .data.synthetic import SyntheticTaskSpec, make_paired_task
from .dmft import DMFTSpec, cue_quadrature, solve_frozen_geometry, solve_zero_disorder
from .losses import training_objective
from .metrics import (
    causal_metrics,
    classify_causal_regime,
    mean_confidence_interval,
    n_sign_changes,
    sign_crossing_time,
)
from .plotting import plot_e1, plot_e1_regions, plot_e2, plot_e2r, plot_e3, plot_enl
from .models.recurrent import DenseLinearRNN
from .theory import (
    discrete_crossover_certificate,
    exact_dense_linear_geometry,
    gradient_gram,
    integrate_projected_flow,
)
from .training import train_paired
from .utils import create_run_directory, resolve_device, seed_everything


def _task_spec(task: Mapping[str, Any], rho: float, lag: int, regime: str) -> SyntheticTaskSpec:
    return SyntheticTaskSpec(
        sequence_length=int(task.get("sequence_length", 20)),
        n_samples=int(task.get("n_samples", 4096)),
        rho=float(rho),
        lag_separation=int(lag),
        regime=regime,
        cue_noise=float(task.get("cue_noise", 0.1)),
        background_noise=float(task.get("background_noise", 0.0)),
    )


def _points_for_e1(task: Mapping[str, Any]) -> list[tuple[float, int, str]]:
    positive = [
        (float(rho), int(lag), "positive")
        for rho in task.get("rho_values", [1, 2, 4, 8])
        for lag in task.get("lag_separations", [0, 2, 4, 8])
    ]
    if "negative" not in task.get("regimes", ["positive"]):
        return positive
    explicit = task.get("negative_control_points")
    if explicit:
        negative = [
            (float(point["rho"]), int(point["lag_separation"]), "negative")
            for point in explicit
        ]
    else:
        negative = [
            (float(rho), int(lag), "negative")
            for rho in task.get("rho_values", [1, 2, 4, 8])
            for lag in task.get("lag_separations", [0, 2, 4, 8])
        ]
    return positive + negative


def _annotate(history: Iterable[dict[str, Any]], **metadata: Any) -> list[dict[str, Any]]:
    return [{**metadata, **row} for row in history]


def _write_aggregate(
    summary: pd.DataFrame,
    group_columns: list[str],
    metrics: list[str],
    path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for keys, group in summary.groupby(group_columns, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, key_values))
        row['n_seeds'] = int(group.seed.nunique()) if 'seed' in group else len(group)
        for metric in metrics:
            mean, low, high = mean_confidence_interval(group[metric].to_numpy())
            row[f'{metric}_mean'] = mean
            row[f'{metric}_ci95_low'] = low
            row[f'{metric}_ci95_high'] = high
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _paired_summary(
    frame: pd.DataFrame,
    beta: float,
    phase_delay: float,
    *,
    tau_max: float | None = None,
    delta: float | None = None,
) -> dict[str, Any]:
    both = frame[frame.condition == "both"].sort_values("tau")
    weak = frame[frame.condition == "weak_only"].sort_values("tau")
    if not np.allclose(both.tau.to_numpy(), weak.tau.to_numpy()):
        raise RuntimeError("Paired trajectories do not share the same optimization-time grid.")
    times = both.tau.to_numpy()
    causal = causal_metrics(times, both.m_w.to_numpy(), weak.m_w.to_numpy(), beta)

    # `legacy_phase` preserves the label emitted before the learnability gate
    # existed, so the run directories already on disk stay interpretable.
    if math.isfinite(causal.delta_tw):
        legacy_phase = "delayed_weak" if causal.delta_tw > phase_delay else "jointly_learned"
    elif math.isfinite(causal.weak_hitting_time) and not math.isfinite(causal.both_hitting_time):
        legacy_phase = "strongly_starved"
    else:
        legacy_phase = "indeterminate"

    verdict = classify_causal_regime(
        weak_hitting_time=causal.weak_hitting_time,
        both_hitting_time=causal.both_hitting_time,
        tau_max=float(times[-1]) if tau_max is None else float(tau_max),
        delta=phase_delay if delta is None else float(delta),
        initial_tau=float(times[0]),
    )
    return {
        **causal.__dict__,
        "regime_class": verdict.regime,
        "weak_only_learnable": verdict.weak_only_learnable,
        "target_met_at_initialization": verdict.target_met_at_initialization,
        "gated_delta_tw": verdict.delta_tw,
        "legacy_phase": legacy_phase,
        "phase": legacy_phase,
        "final_both_m_s": float(both.iloc[-1].m_s),
        "final_both_m_w": float(both.iloc[-1].m_w),
        "final_weak_m_s": float(weak.iloc[-1].m_s),
        "final_weak_m_w": float(weak.iloc[-1].m_w),
        "final_accuracy": float(both.iloc[-1].accuracy),
        "final_gsi5": float(both.iloc[-1].gsi5),
        "mean_A_sw": float(both.A_sw.mean()),
        "mean_chi": float(both.chi_w_from_s.mean()),
    }


def run_e1(config: dict[str, Any]) -> Path:
    run_dir = create_run_directory(config)
    device = resolve_device(str(config.get("experiment", {}).get("device", "auto")))
    task, training, model_config = config["task"], config["training"], config["model"]
    mitigation = config.get("mitigation", {"method": "erm"})
    trajectory_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for rho, lag, regime in _points_for_e1(task):
        for seed in training.get("seeds", [0, 1, 2, 3, 4]):
            spec = _task_spec(task, rho, lag, regime)
            both, weak = (batch.to(device) for batch in make_paired_task(spec, int(seed)))
            history, _ = train_paired(model_config, both, weak, training, mitigation, seed=int(seed))
            metadata = {"rho": rho, "lag_separation": lag, "regime": regime, "seed": int(seed)}
            annotated = _annotate(history, **metadata)
            trajectory_rows.extend(annotated)
            summary_rows.append(
                {
                    **metadata,
                    **_paired_summary(
                        pd.DataFrame(annotated),
                        beta=float(task.get("beta", 0.5)),
                        phase_delay=float(task.get("phase_delay", 1.0)),
                        tau_max=task.get("tau_max"),
                        delta=task.get("delta_tw_tolerance"),
                    ),
                }
            )
            pd.DataFrame(trajectory_rows).to_csv(run_dir / "trajectories.csv", index=False)
            pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    _write_aggregate(
        summary, ['rho', 'lag_separation', 'regime'],
        ['delta_tw', 'weak_auc_gap', 'final_accuracy', 'final_gsi5', 'mean_A_sw'],
        run_dir / 'aggregate.csv',
    )
    summary.groupby(
        ["rho", "lag_separation", "regime", "regime_class"], dropna=False
    ).size().rename("n_seeds").reset_index().to_csv(run_dir / "regions.csv", index=False)
    # `theory_boundary_file` stays optional: the analytic boundary needs the DMFT
    # solver, which is blocked.  Both figures accept the overlay when it exists.
    boundary_path = task.get('theory_boundary_file')
    boundary = pd.read_csv(boundary_path) if boundary_path else None
    plot_e1(summary, run_dir, boundary)
    plot_e1_regions(summary, run_dir, boundary)
    return run_dir


def _enl_summary(frame: pd.DataFrame, beta: float, phase_delay: float) -> dict[str, Any]:
    """Summarize one paired lockstep run at the level of the crossover mechanism.

    Two crossing times are reported and they are *not* the same quantity.

    ``tau_star_drift``
        First ``+ -> -`` sign change of the **exact direct-autograd equal-time**
        weak-drift difference, ``d/dtau[m_w(both)-m_w(weak-only)]``. This is the
        drift-level crossover. ``tau_star_drift_projected`` separately reports the
        sign change of the two-mode ``Gg`` projection; for tanh/GRU it is a
        diagnostic, not the theorem derivative.

    ``tau_star_response``
        First ``+ -> -`` sign change of ``m_w(both) - m_w(weak-only)`` itself.

        ``tau_star_drift`` is measured on the derivative of exactly this gap, which
        makes the lead time a like-for-like comparison -- but it does **not** make the
        ordering a theorem.  A ``+ -> -`` derivative crossing establishes only a local
        *maximum* of the gap.  For the gap itself to cross zero it must additionally
        integrate down through zero, i.e. the accumulated negative drift after ``tau*``
        must exceed the gap's value at ``tau*``.  A run can therefore cross in drift
        and never cross in response, in which case the strong feature suppressed the
        weak mode's *rate* without the weak response ever falling behind.  Under the
        final common-probe/direct-autograd run, this complete pattern occurs in 3/8
        tanh seeds; that is conditional empirical evidence, not a theorem.

    ``tau_star_matched_state``
        Sign change of the matched-state deficit from
        :func:`theory.crossover_decomposition`, which recomputes the weak-only field
        at the both-feature ``m_w``.  Reported because it is the quantity the exact
        three-term geometry/CE identity applies to -- but it is **not** the derivative
        of the equal-time gap and crosses earlier.  On the tanh regime the means
        differ by about ``0.13`` in ``tau``.  Do not use it for drift-versus-response
        comparisons.

    On ``tau_star_decomposition_check`` and ``abs_tau_star_error``: the note
    proposes predicting the crossover from ``T_geom = S_CE``.  Under the exact
    decomposition that is *degenerate*, because ``d_w = t_geom - s_ce`` is an
    identity, so ``t_geom = s_ce`` holds exactly when ``d_w = 0``.  The two
    crossing times therefore agree by construction.  They are still emitted, but
    strictly as a numerical self-consistency check on the logged columns -- not as
    evidence for Corollary A.  A genuinely independent predicted crossover time
    requires the DMFT solution, which is blocked; see
    ``research_scope/e2_theorem.md`` obligations 1 and 2.
    """
    both = frame[frame.condition == "both"].sort_values("tau")
    weak = frame[frame.condition == "weak_only"].sort_values("tau")
    if not np.allclose(both.tau.to_numpy(), weak.tau.to_numpy()):
        raise RuntimeError("Paired trajectories do not share the same optimization-time grid.")
    if "d_w_equal_time_exact" not in both:
        raise RuntimeError(
            "E-NL requires exact direct-autograd crossover columns; set "
            "training.paired_mode: lockstep and training.exact_response_drift: true."
        )

    tau = both.tau.to_numpy()
    d_w_equal_time = both.d_w_equal_time_exact.to_numpy(dtype=float)
    if not np.isfinite(d_w_equal_time).all():
        raise RuntimeError(
            "E-NL exact drift contains non-finite values; publication crossover "
            "timing cannot fall back to the projected Gg diagnostic."
        )
    d_w_projected = both.d_w_equal_time_projected.to_numpy(dtype=float)
    d_w_matched = both.d_w_matched.to_numpy()
    decomposition = both.t_geom.to_numpy() - both.s_ce.to_numpy()
    response_gap = both.m_w.to_numpy() - weak.m_w.to_numpy()

    tau_star_drift = sign_crossing_time(tau, d_w_equal_time)
    tau_star_projected = sign_crossing_time(tau, d_w_projected)
    tau_star_matched = sign_crossing_time(tau, d_w_matched)
    tau_star_check = sign_crossing_time(tau, decomposition)
    tau_star_response = sign_crossing_time(tau, response_gap)
    # `decomposition` reconstructs `d_w_matched` identically, so this gap is a
    # numerical self-consistency check on the matched-state family only.
    identity_gap = (
        abs(tau_star_matched - tau_star_check)
        if math.isfinite(tau_star_matched) and math.isfinite(tau_star_check)
        else (0.0 if tau_star_matched == tau_star_check else float("nan"))
    )
    lead = (
        tau_star_response - tau_star_drift
        if math.isfinite(tau_star_drift) and math.isfinite(tau_star_response)
        else float("nan")
    )
    convention_gap = (
        tau_star_drift - tau_star_matched
        if math.isfinite(tau_star_drift) and math.isfinite(tau_star_matched)
        else float("nan")
    )

    causal = causal_metrics(tau, both.m_w.to_numpy(), weak.m_w.to_numpy(), beta)
    weak_only_learnable = bool(
        math.isfinite(causal.weak_hitting_time)
        and causal.weak_hitting_time > float(tau[0])
    )
    finite_step_certificate = discrete_crossover_certificate(
        response_gap,
        weak_only_learnable=weak_only_learnable,
        tolerance=1e-10,
    )
    return {
        **causal.__dict__,
        "tail_single_transfer_to_suppression": (
            finite_step_certificate.single_transfer_to_suppression
        ),
        "tail_positive_area": finite_step_certificate.positive_area,
        "tail_negative_area": finite_step_certificate.negative_tail_area,
        "tail_area_margin": finite_step_certificate.tail_area_margin,
        "tail_response_equality_reached": (
            finite_step_certificate.response_equality_reached
        ),
        "tail_strict_outcome_starvation": (
            finite_step_certificate.strict_outcome_starvation
        ),
        "tail_causal_starvation_certified": (
            finite_step_certificate.causal_starvation_certified
        ),
        "tau_star_drift": tau_star_drift,
        "tau_star_drift_projected": tau_star_projected,
        "tau_star_projected_gap": (
            tau_star_drift - tau_star_projected
            if math.isfinite(tau_star_drift) and math.isfinite(tau_star_projected)
            else float("nan")
        ),
        "tau_star_matched_state": tau_star_matched,
        "tau_star_convention_gap": convention_gap,
        "tau_star_decomposition_check": tau_star_check,
        "abs_tau_star_error": identity_gap,
        "tau_star_response": tau_star_response,
        "drift_leads_response_by": lead,
        "drift_crossed": bool(math.isfinite(tau_star_drift)),
        "response_crossed": bool(math.isfinite(tau_star_response)),
        "n_sign_changes_d_w": n_sign_changes(d_w_equal_time),
        "n_sign_changes_d_w_matched": n_sign_changes(d_w_matched),
        "max_decomposition_reconstruction_error": float(
            both.decomposition_reconstruction_error.abs().max()
        ),
        "initial_d_w": float(d_w_equal_time[0]),
        "final_d_w": float(d_w_equal_time[-1]),
        "initial_d_w_projected": float(d_w_projected[0]),
        "final_d_w_projected": float(d_w_projected[-1]),
        "initial_d_w_matched": float(d_w_matched[0]),
        "final_d_w_matched": float(d_w_matched[-1]),
        "mean_t_geom": float(both.t_geom.mean()),
        "mean_s_ce": float(both.s_ce.mean()),
        "final_equal_time_geometry_a": float(both.iloc[-1].equal_time_geometry_a),
        "final_equal_time_field_a": float(both.iloc[-1].equal_time_field_a),
        "final_equal_time_geometry_b": float(both.iloc[-1].equal_time_geometry_b),
        "final_equal_time_field_b": float(both.iloc[-1].equal_time_field_b),
        "max_equal_time_projection_residual": float(
            both.equal_time_projection_residual.abs().max()
        ),
        "mean_equal_time_projection_residual": float(
            both.equal_time_projection_residual.mean()
        ),
        "max_mode_residual_rms": float(
            pd.concat([both.mode_residual_rms, weak.mode_residual_rms]).max()
        ),
        "max_equal_time_reconstruction_error": float(
            np.maximum(
                both.equal_time_reconstruction_error_a.abs(),
                both.equal_time_reconstruction_error_b.abs(),
            ).max()
        ),
        # True only if BOTH exact orderings agree on which channel dominates at the
        # end. When false, no ordering-independent dominance claim can be made.
        "final_dominance_ordering_invariant": bool(
            both.iloc[-1].equal_time_dominance_ordering_invariant
        ),
        "dominance_ordering_invariant_fraction": float(
            both.equal_time_dominance_ordering_invariant.mean()
        ),
        "final_both_m_s": float(both.iloc[-1].m_s),
        "final_both_m_w": float(both.iloc[-1].m_w),
        "final_weak_m_w": float(weak.iloc[-1].m_w),
        "final_accuracy": float(both.iloc[-1].accuracy),
        "final_gsi5": float(both.iloc[-1].gsi5),
        # A drift crossing alone establishes suppression of the weak mode's *rate*.
        # A response crossing is outcome suppression; the causal starvation label is
        # reserved for runs that also pass the weak-only learnability gate.
        "phase": (
            (
                (
                    "transfer_then_starvation"
                    if weak_only_learnable
                    else "transfer_then_outcome_crossing_unlearnable"
                )
                if math.isfinite(tau_star_response)
                else "transfer_then_suppression"
            )
            if math.isfinite(tau_star_drift)
            else (
                "transfer_throughout"
                if d_w_equal_time[-1] > 0
                else "suppression_throughout"
            )
        ),
    }


def _write_enl_crossover_report(summary: pd.DataFrame, run_dir: Path) -> None:
    """Report crossing counts alongside means so 'never crossed' stays visible.

    ``mean_confidence_interval`` drops non-finite values, so a configuration in
    which some seeds never cross would otherwise be summarized by the mean of the
    seeds that did.  The counts make that explicit.
    """
    rows: list[dict[str, Any]] = []
    group_columns = ["model_kind", "rho", "lag_separation", "regime"]
    for keys, group in summary.groupby(group_columns, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, key_values))
        row["n_seeds"] = int(group.seed.nunique())
        row["n_tail_single_crossover"] = int(
            group.tail_single_transfer_to_suppression.sum()
        )
        row["n_tail_response_equality"] = int(
            group.tail_response_equality_reached.sum()
        )
        row["n_tail_strict_starvation"] = int(
            group.tail_strict_outcome_starvation.sum()
        )
        row["n_tail_causal_starvation_certified"] = int(
            group.tail_causal_starvation_certified.sum()
        )
        row["max_abs_projection_residual"] = float(
            group.max_equal_time_projection_residual.max()
        )
        for label, column in (("drift", "tau_star_drift"), ("response", "tau_star_response")):
            values = group[column].to_numpy(dtype=float)
            row[f"n_{label}_crossed"] = int(np.isfinite(values).sum())
            row[f"n_{label}_never_crossed"] = int(np.isinf(values).sum())
            row[f"n_{label}_undecidable"] = int(np.isnan(values).sum())
            mean, low, high = mean_confidence_interval(values)
            row[f"tau_star_{label}_mean"] = mean
            row[f"tau_star_{label}_ci95_low"] = low
            row[f"tau_star_{label}_ci95_high"] = high
        row["max_abs_tau_star_identity_gap"] = float(
            np.nanmax(group.abs_tau_star_error.to_numpy(dtype=float))
        )
        row["max_reconstruction_error"] = float(
            group.max_decomposition_reconstruction_error.max()
        )
        row["phases"] = "|".join(sorted(set(group.phase)))
        rows.append(row)
    pd.DataFrame(rows).to_csv(run_dir / "crossover.csv", index=False)


def run_enl(config: dict[str, Any]) -> Path:
    """E-NL: measure the transfer-to-starvation crossover mechanism.

    Requires the lockstep paired trainer and computes the universal direct-autograd
    response drift at every logged point. Produces an empirical trajectory
    certificate for Theorems B/C; it does not independently prove the rank-one
    derivative bounds or the general DMFT conjecture.
    """
    run_dir = create_run_directory(config)
    device = resolve_device(str(config.get("experiment", {}).get("device", "auto")))
    task, training, model_config = config["task"], config["training"], config["model"]
    requested_mode = str(training.get("paired_mode", "lockstep"))
    if requested_mode != "lockstep":
        raise ValueError(
            "E-NL requires training.paired_mode: lockstep; the sequential trainer "
            f"cannot form the paired crossover decomposition (received {requested_mode!r})."
        )
    training = {
        **training,
        "paired_mode": "lockstep",
        "exact_response_drift": True,
    }
    kinds = model_config.get("kinds", [model_config.get("kind", "tanh")])
    trajectories: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for kind in kinds:
        for rho in task.get("rho_values", [4]):
            for lag in task.get("lag_separations", [2]):
                for regime in task.get("regimes", ["positive"]):
                    spec = _task_spec(task, float(rho), int(lag), str(regime))
                    for seed in training.get("seeds", [0, 1]):
                        both, weak = (
                            batch.to(device) for batch in make_paired_task(spec, int(seed))
                        )
                        history, _ = train_paired(
                            model_config, both, weak, training, {"method": "erm"},
                            seed=int(seed), kind=str(kind),
                        )
                        metadata = {
                            "model_kind": kind, "rho": float(rho),
                            "lag_separation": int(lag), "regime": regime, "seed": int(seed),
                        }
                        annotated = _annotate(history, **metadata)
                        trajectories.extend(annotated)
                        summaries.append({
                            **metadata,
                            **_enl_summary(
                                pd.DataFrame(annotated),
                                beta=float(task.get("beta", 0.5)),
                                phase_delay=float(task.get("phase_delay", 1.0)),
                            ),
                        })
                        pd.DataFrame(trajectories).to_csv(
                            run_dir / "trajectories.csv", index=False
                        )
                        pd.DataFrame(summaries).to_csv(run_dir / "summary.csv", index=False)
    summary = pd.DataFrame(summaries)
    _write_aggregate(
        summary, ["model_kind", "rho", "lag_separation", "regime"],
        [
            "tau_star_drift", "tau_star_drift_projected", "tau_star_response",
            "drift_leads_response_by", "tail_positive_area", "tail_negative_area",
            "tail_area_margin", "max_equal_time_projection_residual",
            "weak_auc_gap", "final_both_m_s", "final_both_m_w", "final_weak_m_w",
            "final_accuracy", "final_gsi5",
        ],
        run_dir / "aggregate.csv",
    )
    _write_enl_crossover_report(summary, run_dir)
    plot_enl(pd.DataFrame(trajectories), summary, run_dir)
    return run_dir


def _e2_points(task: Mapping[str, Any]) -> list[tuple[float, int]]:
    return [
        (float(point["rho"]), int(point["lag_separation"]))
        for point in task.get("points", [])
    ]


def _write_finite_n_geometry_check(run_dir: Path, task: Mapping[str, Any], model: Mapping[str, Any]) -> None:
    torch.manual_seed(0)
    sequence_length = int(task.get('sequence_length', 20))
    spec = _task_spec(task, rho=2.0, lag=min(2, sequence_length - 1), regime='positive')
    dense = DenseLinearRNN(width=8, bulk_gain=float(model.get('bulk_gain', 0.8)))
    empirical, _ = gradient_gram(dense.mode_responses(spec), dense)
    analytic = exact_dense_linear_geometry(dense, spec)
    difference = empirical.detach() - analytic.detach()
    relative_error = float(difference.norm() / analytic.detach().norm().clamp_min(1e-12))
    payload = {
        'relative_error': relative_error,
        'autograd_geometry': empirical.detach().tolist(),
        'analytic_geometry': analytic.detach().tolist(),
        'width': dense.width,
    }
    (run_dir / 'finite_n_geometry_check.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _mean_closure(history: pd.DataFrame, condition: str) -> pd.DataFrame:
    columns = [
        "m_s", "m_w", "G_ss", "G_sw", "G_ww", "A_ss", "A_sw", "A_ww",
        "margin_mean", "margin_std", "margin_q10", "margin_q50", "margin_q90", "gsi5",
    ]
    return (
        history[history.condition == condition]
        .groupby("tau", as_index=False)[columns]
        .mean()
        .sort_values("tau")
    )


def _relative_trajectory_rmse(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    difference = np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    scale = float(np.sqrt(np.mean(np.square(observed))))
    return rmse, rmse / max(scale, 1e-8)


def _write_e2_convergence(summary: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    metrics = ["trajectory_nrmse", "geometry_nrmse", "margin_nrmse", "gsi_rmse"]
    rows: list[dict[str, Any]] = []
    group_columns = ["rho", "lag_separation", "regime", "condition"]
    for keys, group in summary.groupby(group_columns):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        for metric in metrics:
            by_width = group.groupby("width", as_index=False)[metric].mean().sort_values("width")
            finite = by_width[np.isfinite(by_width[metric]) & (by_width[metric] > 0)]
            slope = float("nan")
            if len(finite) >= 2:
                slope = float(
                    np.polyfit(np.log(finite["width"]), np.log(finite[metric]), deg=1)[0]
                )
            rows.append(
                {
                    **dict(zip(group_columns, key_values)),
                    "metric": metric,
                    "n_widths": int(len(by_width)),
                    "log_log_slope": slope,
                    "smallest_width_error": float(by_width.iloc[0][metric]),
                    "largest_width_error": float(by_width.iloc[-1][metric]),
                    "decreases_end_to_end": bool(
                        by_width.iloc[-1][metric] < by_width.iloc[0][metric]
                    ),
                }
            )
    convergence = pd.DataFrame(rows)
    convergence.to_csv(run_dir / "convergence.csv", index=False)
    return convergence


def _write_e2_acceptance(
    run_dir: Path,
    summary: pd.DataFrame,
    convergence: pd.DataFrame,
    closure_seeds: set[int],
    evaluation_seeds: set[int],
) -> None:
    geometry = json.loads((run_dir / "finite_n_geometry_check.json").read_text())
    mode_rows = convergence[convergence["metric"] == "trajectory_nrmse"]
    payload = {
        "closure_and_evaluation_seeds_disjoint": closure_seeds.isdisjoint(evaluation_seeds),
        "finite_n_geometry_relative_error": float(geometry["relative_error"]),
        "max_projected_identity_absolute_error": float(
            summary["max_identity_absolute_error"].max()
        ),
        "all_summary_diagnostics_finite": bool(
            np.isfinite(
                summary[
                    [
                        "trajectory_rmse", "trajectory_nrmse", "geometry_nrmse",
                        "margin_nrmse", "gsi_rmse", "max_identity_absolute_error",
                    ]
                ].to_numpy()
            ).all()
        ),
        "mode_error_decreases_for_every_point_and_condition": bool(
            len(mode_rows) > 0
            and mode_rows["decreases_end_to_end"].all()
            and (mode_rows["log_log_slope"] < 0).all()
        ),
        "at_least_three_widths": bool(summary["width"].nunique() >= 3),
    }
    payload["passed"] = bool(
        payload["closure_and_evaluation_seeds_disjoint"]
        and payload["finite_n_geometry_relative_error"] < 1e-5
        and payload["max_projected_identity_absolute_error"] < 1e-4
        and payload["all_summary_diagnostics_finite"]
        and payload["mode_error_decreases_for_every_point_and_condition"]
        and payload["at_least_three_widths"]
    )
    (run_dir / "e2_acceptance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_e2(config: dict[str, Any]) -> Path:
    run_dir = create_run_directory(config)
    _write_finite_n_geometry_check(run_dir, config['task'], config['model'])
    device = resolve_device(str(config.get("experiment", {}).get("device", "auto")))
    task, training, base_model, closure_config = (
        config["task"], config["training"], config["model"], config["closure"]
    )
    all_trajectories: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    closure_seeds = {int(seed) for seed in closure_config.get(
        "seeds", range(int(closure_config.get("particles", 16)))
    )}
    evaluation_seeds = {int(seed) for seed in training.get("seeds", [0, 1, 2])}
    if not closure_seeds.isdisjoint(evaluation_seeds):
        raise ValueError("E2 closure.seeds and training.seeds must be disjoint.")
    for rho, lag in _e2_points(task):
        for regime in task.get("regimes", ["positive"]):
            spec = _task_spec(task, rho, lag, str(regime))
            closure_model = {**base_model, "width": int(closure_config.get("width", 256))}
            closure_rows: list[dict[str, Any]] = []
            for particle, seed in enumerate(sorted(closure_seeds)):
                both, weak = (batch.to(device) for batch in make_paired_task(spec, int(seed)))
                history, _ = train_paired(
                    closure_model, both, weak, training, {"method": "erm"}, seed=int(seed)
                )
                closure_rows.extend(_annotate(history, particle=particle, closure_seed=int(seed)))
            closure_frame = pd.DataFrame(closure_rows)
            theory_by_condition: dict[str, pd.DataFrame] = {}
            field_spec = replace(
                spec, n_samples=int(closure_config.get("field_samples", spec.n_samples))
            )
            reference_both, reference_weak = make_paired_task(
                field_spec, int(closure_config.get("field_seed", 1000003))
            )
            reference = {"both": reference_both, "weak_only": reference_weak}
            for condition in ["both", "weak_only"]:
                mean = _mean_closure(closure_frame, condition)
                geometry = np.stack(
                    [np.array([[row.G_ss, row.G_sw], [row.G_sw, row.G_ww]]) for row in mean.itertuples()]
                )
                batch = reference[condition]
                coordinates = np.stack((batch.z_s.numpy(), batch.z_w.numpy()), axis=1)
                modes = integrate_projected_flow(
                    mean[["m_s", "m_w"]].iloc[0].to_numpy(),
                    coordinates,
                    mean.tau.to_numpy(),
                    geometry,
                )
                theory = mean.copy()
                theory["m_s"], theory["m_w"] = modes[:, 0], modes[:, 1]
                # Named `closure_reference`, not `particle_closure` and not
                # anything containing "dmft": this is a mean over trained finite
                # networks fed through integrate_projected_flow. It is a numerical
                # approximation used as a reference, not a solved theory. Existing
                # run directories keep their original label and are not rewritten.
                theory["condition"], theory["source"] = condition, "closure_reference"
                theory["width"], theory["seed"] = int(closure_config.get("width", 256)), -1
                theory["rho"], theory["lag_separation"], theory["regime"] = rho, lag, regime
                theory_by_condition[condition] = theory
                all_trajectories.extend(theory.to_dict("records"))
            for width in base_model.get("widths", [128, 512, 2048]):
                model_config = {**base_model, "width": int(width)}
                for seed in training.get("seeds", [0, 1, 2]):
                    both, weak = (batch.to(device) for batch in make_paired_task(spec, int(seed)))
                    history, _ = train_paired(
                        model_config,
                        both,
                        weak,
                        training,
                        {"method": "erm"},
                        seed=int(seed),
                        identity_steps=int(training.get("identity_steps", 5)),
                    )
                    network = pd.DataFrame(history)
                    network["source"], network["width"], network["seed"] = "network", int(width), int(seed)
                    network["rho"], network["lag_separation"], network["regime"] = rho, lag, regime
                    all_trajectories.extend(network.to_dict("records"))
                    for condition in ["both", "weak_only"]:
                        observed = network[network.condition == condition].sort_values("tau")
                        theory = theory_by_condition[condition]
                        predicted_s = np.interp(observed.tau, theory.tau, theory.m_s)
                        predicted_w = np.interp(observed.tau, theory.tau, theory.m_w)
                        observed_modes = observed[["m_s", "m_w"]].to_numpy()
                        predicted_modes = np.stack((predicted_s, predicted_w), axis=1)
                        rmse, nrmse = _relative_trajectory_rmse(observed_modes, predicted_modes)
                        strong_rmse, _ = _relative_trajectory_rmse(
                            observed.m_s.to_numpy(), predicted_s
                        )
                        weak_rmse, _ = _relative_trajectory_rmse(
                            observed.m_w.to_numpy(), predicted_w
                        )
                        geometry_columns = ["G_ss", "G_sw", "G_ww"]
                        predicted_geometry = np.stack(
                            [
                                np.interp(observed.tau, theory.tau, theory[column])
                                for column in geometry_columns
                            ],
                            axis=1,
                        )
                        geometry_rmse, geometry_nrmse = _relative_trajectory_rmse(
                            observed[geometry_columns].to_numpy(), predicted_geometry
                        )
                        margin_columns = [
                            "margin_mean", "margin_std", "margin_q10", "margin_q50", "margin_q90"
                        ]
                        predicted_margins = np.stack(
                            [
                                np.interp(observed.tau, theory.tau, theory[column])
                                for column in margin_columns
                            ],
                            axis=1,
                        )
                        margin_rmse, margin_nrmse = _relative_trajectory_rmse(
                            observed[margin_columns].to_numpy(), predicted_margins
                        )
                        predicted_gsi = np.interp(observed.tau, theory.tau, theory.gsi5)
                        gsi_rmse, _ = _relative_trajectory_rmse(
                            observed.gsi5.to_numpy(), predicted_gsi
                        )
                        identity = observed.projected_identity_relative_error.to_numpy()
                        identity_absolute = observed.projected_identity_absolute_error.to_numpy()
                        summaries.append({
                            "rho": rho, "lag_separation": lag, "regime": regime,
                            "condition": condition, "width": int(width), "seed": int(seed),
                            "trajectory_rmse": rmse,
                            "trajectory_nrmse": nrmse,
                            "strong_mode_rmse": strong_rmse,
                            "weak_mode_rmse": weak_rmse,
                            "geometry_rmse": geometry_rmse,
                            "geometry_nrmse": geometry_nrmse,
                            "margin_rmse": margin_rmse,
                            "margin_nrmse": margin_nrmse,
                            "gsi_rmse": gsi_rmse,
                            "max_identity_absolute_error": float(np.nanmax(identity_absolute)),
                            "max_identity_relative_error": float(np.nanmax(identity)),
                        })
                    pd.DataFrame(all_trajectories).to_csv(run_dir / "trajectories.csv", index=False)
                    pd.DataFrame(summaries).to_csv(run_dir / "summary.csv", index=False)
    trajectories, summary = pd.DataFrame(all_trajectories), pd.DataFrame(summaries)
    _write_aggregate(
        summary, ['rho', 'lag_separation', 'regime', 'condition', 'width'],
        [
            'trajectory_rmse', 'trajectory_nrmse', 'strong_mode_rmse', 'weak_mode_rmse',
            'geometry_nrmse', 'margin_nrmse', 'gsi_rmse',
            'max_identity_absolute_error', 'max_identity_relative_error',
        ], run_dir / 'aggregate.csv',
    )
    convergence = _write_e2_convergence(summary, run_dir)
    _write_e2_acceptance(run_dir, summary, convergence, closure_seeds, evaluation_seeds)
    plot_e2(trajectories, summary[summary.condition == "both"], run_dir)
    return run_dir


_E2R_BLOCKED_CHECKS = {
    "check_b_weak_only_reduction": [1, 2],
    "check_d_mse_solver": [1, 2],
    "check_e_internal_convergence": [2, 3],
    "check_f_finite_width_against_frozen_prediction": [1, 2, 3, 4],
}


def _e2r_check_a(task: Mapping[str, Any], model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compare the zero-disorder solver against finite networks at several widths.

    Two errors are reported per width and learning rate.

    ``exact_seeded_relative_error``
        Solver seeded from the network's own initial inner products.  The reduction
        is exact at every width, so the only residual is the O(eta) gap between
        gradient flow and discrete SGD.  It should track the learning rate and be
        essentially width-independent.

    ``wide_limit_relative_error``
        Solver seeded from the wide-limit initial conditions (unit input Gram, unit
        readout norm, zero modes).  This one *should* shrink with width, because it
        measures concentration of the initial inner products.
    """
    sequence_length = int(task.get("sequence_length", 5))
    spec_kwargs = dict(
        sequence_length=sequence_length, bulk_gain=0.0, lag_separation=0,
        rho=float(task.get("rho", 2.0)), cue_noise=float(task.get("cue_noise", 0.0)),
    )
    rows: list[dict[str, Any]] = []
    for width in model.get("widths", [32, 64, 128, 256]):
        for learning_rate in task.get("learning_rates", [0.04, 0.02, 0.01]):
            tau_max = float(task.get("tau_max", 2.0))
            steps = int(round(tau_max / learning_rate))
            data_spec = SyntheticTaskSpec(
                sequence_length=sequence_length,
                n_samples=int(task.get("n_samples", 2048)),
                rho=float(task.get("rho", 2.0)), lag_separation=0,
                cue_noise=float(task.get("cue_noise", 0.0)), background_noise=0.0,
            )
            both, _ = make_paired_task(data_spec, int(task.get("seed", 0)))
            seed_everything(int(task.get("seed", 0)))
            network = DenseLinearRNN(width=int(width), bulk_gain=0.0)
            with torch.no_grad():
                b_s = network.input[:, 0].clone()
                b_w = network.input[:, 1].clone()
                c = network.readout.clone()
            shared = dict(tau_max=tau_max, dtau=min(learning_rate / 50, 1e-3), **spec_kwargs)
            exact = solve_zero_disorder(
                DMFTSpec(
                    **shared,
                    initial_mode=(float(c @ b_s), float(c @ b_w)),
                    initial_input_gram=(
                        (float(b_s @ b_s), float(b_s @ b_w)),
                        (float(b_s @ b_w), float(b_w @ b_w)),
                    ),
                    initial_readout_norm_sq=float(c @ c),
                )
            )
            wide = solve_zero_disorder(DMFTSpec(**shared))

            optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
            observed = []
            for _ in range(steps + 1):
                observed.append(network.mode_responses(data_spec).detach().numpy().copy())
                optimizer.zero_grad(set_to_none=True)
                loss, _ = training_objective(network, both, {"method": "erm"})
                loss.backward()
                optimizer.step()
            observed = np.asarray(observed)
            grid = np.arange(steps + 1) * learning_rate
            scale = max(float(np.abs(observed).max()), 1e-8)

            def relative(solution) -> float:
                predicted = np.stack(
                    [np.interp(grid, solution.tau, solution.m_s),
                     np.interp(grid, solution.tau, solution.m_w)], axis=1
                )
                return float(np.abs(observed - predicted).max() / scale)

            rows.append({
                "check": "check_a_zero_disorder",
                "width": int(width),
                "learning_rate": float(learning_rate),
                "recurrent_block_inert": float(network.recurrent.detach().abs().max()) == 0.0,
                "exact_seeded_relative_error": relative(exact),
                "wide_limit_relative_error": relative(wide),
            })
    return rows


def _e2r_check_c(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compare the frozen-geometry solver against ``integrate_projected_flow``."""
    rows: list[dict[str, Any]] = []
    for geometry in task.get(
        "frozen_geometries", [[[1.0, 0.0], [0.0, 1.0]], [[1.8, 0.2], [0.2, 1.1]]]
    ):
        matrix = np.asarray(geometry, dtype=float)
        spec = DMFTSpec(
            frozen_geometry=((matrix[0, 0], matrix[0, 1]), (matrix[1, 0], matrix[1, 1])),
            rho=float(task.get("rho", 2.0)), cue_noise=float(task.get("cue_noise", 0.0)),
            tau_max=float(task.get("tau_max", 2.0)), dtau=0.005,
            quadrature_size=int(task.get("quadrature_size", 61)),
        )
        solution = solve_frozen_geometry(spec)
        coordinates, weights = cue_quadrature(spec)
        counts = np.maximum((weights / weights.max() * 4000).astype(int), 1)
        empirical = np.repeat(coordinates, counts, axis=0)
        reference = integrate_projected_flow(
            np.zeros(2), empirical, solution.tau,
            np.repeat(matrix[None, :, :], len(solution.tau), axis=0),
        )
        scale = max(float(np.abs(reference).max()), 1e-8)
        rows.append({
            "check": "check_c_frozen_geometry",
            "geometry": json.dumps(matrix.tolist()),
            "relative_error": float(np.abs(solution.modes - reference).max() / scale),
            "geometry_held_constant": bool(
                np.allclose(solution.geometry[0], solution.geometry[-1])
            ),
        })
    return rows


def _write_e2r_acceptance(run_dir: Path, checks: pd.DataFrame) -> dict[str, Any]:
    """Write an acceptance record that cannot pass, and say why.

    Only checks A and C are implementable without the closure derivation.  The rest
    carry the literal string ``"blocked"`` and the top-level ``passed`` is ``false``.
    An acceptance record that refuses to pass is the correct output for this stage;
    a passing one would misrepresent an unproved theory as validated.
    """
    check_a = checks[checks["check"] == "check_a_zero_disorder"]
    check_c = checks[checks["check"] == "check_c_frozen_geometry"]

    widths = sorted(check_a["width"].unique())
    by_width = check_a.groupby("width")["wide_limit_relative_error"].mean()
    slopes: dict[str, float] = {}
    for rate, group in check_a.groupby("learning_rate"):
        ordered = group.sort_values("width")
        slopes[str(rate)] = float(ordered["exact_seeded_relative_error"].mean())
    rates = sorted(check_a["learning_rate"].unique())
    by_rate = check_a.groupby("learning_rate")["exact_seeded_relative_error"].mean()
    discretization_slope = (
        float(np.polyfit(np.log(rates), np.log(by_rate.loc[rates].to_numpy()), 1)[0])
        if len(rates) >= 2 else float("nan")
    )

    payload: dict[str, Any] = {
        "check_a_zero_disorder": {
            "implemented": True,
            "recurrent_block_inert_everywhere": bool(check_a["recurrent_block_inert"].all()),
            "max_exact_seeded_relative_error": float(
                check_a["exact_seeded_relative_error"].max()
            ),
            "discretization_log_log_slope_vs_learning_rate": discretization_slope,
            "wide_limit_error_by_width": {str(k): float(v) for k, v in by_width.items()},
            "wide_limit_error_decreases_end_to_end": bool(
                len(widths) >= 2 and by_width.loc[widths[-1]] < by_width.loc[widths[0]]
            ),
            "wide_limit_error_monotone_in_width": bool(
                len(widths) >= 2
                and all(
                    by_width.loc[widths[index + 1]] < by_width.loc[widths[index]]
                    for index in range(len(widths) - 1)
                )
            ),
            "wide_limit_caveat": (
                "Single initialization seed. The wide-limit error measures "
                "concentration of the initial inner products, so intermediate "
                "widths fluctuate and non-monotonicity here is sampling noise "
                "rather than a systematic effect. Report the end-to-end trend only, "
                "or average over seeds before claiming a rate."
            ),
            "mean_exact_seeded_error_by_learning_rate": slopes,
        },
        "check_c_frozen_geometry": {
            "implemented": True,
            "max_relative_error": float(check_c["relative_error"].max()),
            "geometry_held_constant": bool(check_c["geometry_held_constant"].all()),
        },
    }
    for name, obligations in _E2R_BLOCKED_CHECKS.items():
        payload[name] = "blocked"
    payload["blocked_on"] = {
        name: {
            "obligations": obligations,
            "source": 'research_scope/e2_theorem.md § "Proof obligations"',
        }
        for name, obligations in _E2R_BLOCKED_CHECKS.items()
    }
    payload["passed"] = False
    payload["passed_reason"] = (
        "Deliberately false. Only the zero-disorder and frozen-geometry special "
        "cases are implementable without the joint cross-entropy recurrent "
        "mean-field derivation. Until obligations 1-4 of "
        'research_scope/e2_theorem.md § "Proof obligations" are discharged there is '
        "no independent solver to freeze a prediction from, so nothing in this "
        "phase validates the joint mean-field theory."
    )
    (run_dir / "e2r_acceptance.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def run_e2r(config: dict[str, Any]) -> Path:
    """E2-R: exercise the implementable solver special cases and record the blocks.

    This experiment cannot pass its own acceptance record, by construction.  It
    exists to verify the two exact special cases against machinery the repository
    already trusts, and to make the remaining blocks legible as an artifact rather
    than as an omission.
    """
    run_dir = create_run_directory(config)
    task, model = config.get("task", {}), config.get("model", {})
    rows = _e2r_check_a(task, model) + _e2r_check_c(task)
    checks = pd.DataFrame(rows)
    checks.to_csv(run_dir / "checks.csv", index=False)
    payload = _write_e2r_acceptance(run_dir, checks)
    plot_e2r(checks, run_dir)
    (run_dir / "blocked_checks.txt").write_text(
        "\n".join(
            f"{name}: blocked on obligations "
            f"{', '.join(str(number) for number in obligations)} of "
            'research_scope/e2_theorem.md § "Proof obligations"'
            for name, obligations in _E2R_BLOCKED_CHECKS.items()
        )
        + "\n",
        encoding="utf-8",
    )
    assert payload["passed"] is False  # invariant: this record must never pass
    return run_dir


def run_e3(config: dict[str, Any]) -> Path:
    run_dir = create_run_directory(config)
    device = resolve_device(str(config.get("experiment", {}).get("device", "auto")))
    task, training, model_config = config["task"], config["training"], config["model"]
    mitigation_config = config.get("mitigation", {})
    trajectories: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    kinds = model_config.get("kinds", [model_config.get("kind", "low_rank_linear")])
    for kind in kinds:
        for method in mitigation_config.get("methods", ["erm"]):
            mitigation = {**mitigation_config, "method": method}
            for rho in task.get("rho_values", [4, 8]):
                for lag in task.get("lag_separations", [4, 8]):
                    for regime in task.get("regimes", ["positive"]):
                        spec = _task_spec(task, float(rho), int(lag), str(regime))
                        for seed in training.get("seeds", [0, 1, 2, 3, 4]):
                            both, weak = (batch.to(device) for batch in make_paired_task(spec, int(seed)))
                            history, _ = train_paired(
                                model_config, both, weak, training, mitigation,
                                seed=int(seed), kind=str(kind),
                            )
                            metadata = {
                                "model_kind": kind, "method": method, "rho": float(rho),
                                "lag_separation": int(lag), "regime": regime, "seed": int(seed),
                            }
                            annotated = _annotate(history, **metadata)
                            trajectories.extend(annotated)
                            summaries.append({
                                **metadata,
                                **_paired_summary(
                                    pd.DataFrame(annotated),
                                    beta=float(task.get("beta", 0.5)),
                                    phase_delay=float(task.get("phase_delay", 1.0)),
                                ),
                            })
                            pd.DataFrame(trajectories).to_csv(run_dir / "trajectories.csv", index=False)
                            pd.DataFrame(summaries).to_csv(run_dir / "summary.csv", index=False)
    summary = pd.DataFrame(summaries)
    _write_aggregate(
        summary, ['model_kind', 'method', 'rho', 'lag_separation', 'regime'],
        [
            'delta_tw', 'weak_auc_gap', 'final_both_m_s', 'final_both_m_w',
            'final_accuracy', 'final_gsi5', 'mean_chi',
        ],
        run_dir / 'aggregate.csv',
    )
    plot_e3(summary, run_dir)
    return run_dir
