from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .data.synthetic import SyntheticTaskSpec, make_paired_task
from .dmft import DMFTSpec, cue_quadrature, solve_frozen_geometry, solve_zero_disorder
from .losses import training_objective
from .metrics import (
    causal_metrics,
    classify_causal_regime,
    gsi5,
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
    initial_frozen_empirical_kernel,
    integrate_frozen_logistic_sgd,
    integrate_projected_flow,
    paired_initial_response_jet,
)
from .training import initialize_paired_models, train_paired
from .utils import (
    atomic_torch_save,
    create_run_directory,
    executable_source_fingerprint,
    resolve_device,
    seed_everything,
    sha256_file,
)


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


def _enl_seed_pairs(
    training: Mapping[str, Any],
) -> list[tuple[int, int, int | None]]:
    """Normalize legacy coupled seeds or explicit E-NL data/model seed pairs.

    The third tuple entry is the legacy scalar seed. It remains present only for
    coupled runs so historical CSV consumers keep their original ``seed`` column;
    independent runs are identified unambiguously by ``data_seed`` and
    ``model_seed`` instead.
    """
    raw_pairs = training.get("seed_pairs")
    if raw_pairs is None:
        raw_seeds = training.get("seeds", [0, 1])
        if not isinstance(raw_seeds, list) or not raw_seeds:
            raise ValueError("E-NL training.seeds must be a non-empty list.")
        seeds = [int(seed) for seed in raw_seeds]
        return [(seed, seed, seed) for seed in seeds]

    if "seeds" in training:
        raise ValueError(
            "E-NL training.seed_pairs cannot be combined with training.seeds; "
            "remove the coupled legacy list to make the design unambiguous."
        )
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("E-NL training.seed_pairs must be a non-empty list.")

    normalized: list[tuple[int, int, int | None]] = []
    seen: set[tuple[int, int]] = set()
    for index, pair in enumerate(raw_pairs):
        if not isinstance(pair, Mapping) or not {"data_seed", "model_seed"} <= set(pair):
            raise ValueError(
                "Each E-NL seed pair must map both data_seed and model_seed "
                f"(invalid entry at index {index})."
            )
        data_seed = int(pair["data_seed"])
        model_seed = int(pair["model_seed"])
        key = (data_seed, model_seed)
        if key in seen:
            raise ValueError(f"Duplicate E-NL seed pair {key} at index {index}.")
        seen.add(key)
        normalized.append((data_seed, model_seed, None))
    return normalized


def _replicate_design(group: pd.DataFrame) -> dict[str, Any]:
    """Describe replication without treating crossed seed reuse as independence."""
    if {"data_seed", "model_seed"} <= set(group.columns):
        n_pairs = int(
            group[["data_seed", "model_seed"]].drop_duplicates().shape[0]
        )
        n_data = int(group.data_seed.nunique())
        n_model = int(group.model_seed.nunique())
    elif "seed" in group:
        n_pairs = n_data = n_model = int(group.seed.nunique())
    else:
        n_pairs = n_data = n_model = len(group)
    seed_reuse = n_data < n_pairs or n_model < n_pairs
    return {
        "n_seeds": n_pairs,
        "n_seed_pairs": n_pairs,
        "n_data_seeds": n_data,
        "n_model_seeds": n_model,
        "seed_reuse": seed_reuse,
        "ci95_scope": (
            "descriptive_only_seed_reuse"
            if seed_reuse
            else "independent_pair_replicates"
        ),
    }


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
        replicate_design = _replicate_design(group)
        row.update(replicate_design)
        for metric in metrics:
            mean, low, high = mean_confidence_interval(group[metric].to_numpy())
            if replicate_design["seed_reuse"]:
                # Reusing a data or model seed creates crossed dependence between
                # pair rows. A row-wise Student-t interval would be anti-conservative;
                # a hierarchical/two-way analysis belongs in the confirmatory report.
                low = high = float("nan")
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
        "tail_strict_outcome_suppression": (
            finite_step_certificate.strict_outcome_suppression
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
        "weak_only_learnable": weak_only_learnable,
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
        replicate_design = _replicate_design(group)
        row.update(replicate_design)
        row["n_tail_single_crossover"] = int(
            group.tail_single_transfer_to_suppression.sum()
        )
        row["n_tail_response_equality"] = int(
            group.tail_response_equality_reached.sum()
        )
        # Historical summaries retain the pre-repair column name. Read it only as
        # a compatibility alias; all newly emitted summaries use "suppression".
        strict_outcome_column = (
            "tail_strict_outcome_suppression"
            if "tail_strict_outcome_suppression" in group.columns
            else "tail_strict_outcome_starvation"
        )
        row["n_tail_strict_suppression"] = int(
            group[strict_outcome_column].sum()
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
            if replicate_design["seed_reuse"]:
                low = high = float("nan")
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


def _frozen_kernel_payload(kernel: Any) -> dict[str, Any]:
    return {
        "ordinary_logits": kernel.ordinary_logits.detach().cpu(),
        "signed_logits": kernel.signed_logits.detach().cpu(),
        "response": kernel.response.detach().cpu(),
        "logit_jacobian": kernel.logit_jacobian.detach().cpu(),
        "signed_logit_jacobian": kernel.signed_logit_jacobian.detach().cpu(),
        "response_jacobian": kernel.response_jacobian.detach().cpu(),
        "signed_ntk": kernel.signed_ntk.detach().cpu(),
        "response_cross_kernel": kernel.response_cross_kernel.detach().cpu(),
        "parameter_names": kernel.parameter_names,
        "parameter_shapes": kernel.parameter_shapes,
    }


def _frozen_trajectory_rows(
    trajectory: Any,
    *,
    condition: str,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, step in enumerate(trajectory.steps.tolist()):
        signed_logits = trajectory.signed_logits[index]
        weights = torch.sigmoid(-signed_logits)
        rows.append(
            {
                **metadata,
                "step": int(step),
                "tau": float(trajectory.tau[index]),
                "condition": condition,
                "m_w": float(trajectory.response[index]),
                "direct_drift_w": float(trajectory.response_drift[index]),
                "loss": float(torch.nn.functional.softplus(-signed_logits).mean()),
                "accuracy": float((signed_logits >= 0).to(torch.float64).mean()),
                "gsi5": gsi5(weights.cpu().numpy()),
            }
        )
    return rows


def _frozen_enl_summary(
    both_trajectory: Any,
    weak_trajectory: Any,
    *,
    beta: float,
) -> dict[str, Any]:
    tau = both_trajectory.tau.cpu().numpy()
    both_response = both_trajectory.response.cpu().numpy()
    weak_response = weak_trajectory.response.cpu().numpy()
    response_gap = both_response - weak_response
    drift_gap = (
        both_trajectory.response_drift - weak_trajectory.response_drift
    ).cpu().numpy()
    tau_star_drift = sign_crossing_time(tau, drift_gap)
    tau_star_response = sign_crossing_time(tau, response_gap)
    causal = causal_metrics(tau, both_response, weak_response, beta)
    weak_only_learnable = bool(
        math.isfinite(causal.weak_hitting_time)
        and causal.weak_hitting_time > float(tau[0])
    )
    finite_step = discrete_crossover_certificate(
        response_gap,
        weak_only_learnable=weak_only_learnable,
        tolerance=1e-10,
    )
    phase = (
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
            "transfer_throughout" if drift_gap[-1] > 0 else "suppression_throughout"
        )
    )
    return {
        **causal.__dict__,
        "tau_star_drift": tau_star_drift,
        "tau_star_response": tau_star_response,
        "drift_crossed": bool(math.isfinite(tau_star_drift)),
        "response_crossed": bool(math.isfinite(tau_star_response)),
        "n_sign_changes_d_w": n_sign_changes(drift_gap),
        "initial_d_w": float(drift_gap[0]),
        "final_d_w": float(drift_gap[-1]),
        "final_both_m_w": float(both_response[-1]),
        "final_weak_m_w": float(weak_response[-1]),
        "final_response_gap": float(response_gap[-1]),
        "weak_only_learnable": weak_only_learnable,
        "tail_single_transfer_to_suppression": finite_step.single_transfer_to_suppression,
        "tail_response_equality_reached": finite_step.response_equality_reached,
        "tail_strict_outcome_suppression": finite_step.strict_outcome_suppression,
        "tail_causal_starvation_certified": finite_step.causal_starvation_certified,
        "tail_positive_area": finite_step.positive_area,
        "tail_negative_area": finite_step.negative_tail_area,
        "tail_area_margin": finite_step.tail_area_margin,
        "phase": phase,
    }


def run_enl_preflight(config: dict[str, Any]) -> Path:
    """Freeze full empirical-NTK predictions without running optimization.

    This stage constructs shared initial models, differentiates only initialization
    objects, integrates the frozen tangent recursion, and hashes every prediction
    artifact. It never creates an optimizer or evaluates a trained parameter state.
    """
    task, training, model_config = config["task"], config["training"], config["model"]
    if not bool(training.get("full_batch", True)):
        raise ValueError("E-NL preflight requires full_batch: true.")
    if float(training.get("weight_decay", 0.0)) != 0.0:
        raise ValueError("E-NL preflight requires weight_decay: 0.")
    if training.get("gradient_clip") is not None:
        raise ValueError("E-NL preflight is incompatible with gradient_clip.")
    if str(training.get("objective", "cross_entropy")) != "cross_entropy":
        raise ValueError("E-NL preflight requires the cross_entropy objective.")

    run_dir = create_run_directory(config)
    artifact_dir = run_dir / "kernels"
    artifact_dir.mkdir()
    device = resolve_device(str(config.get("experiment", {}).get("device", "auto")))
    steps = int(training.get("steps", 1000))
    learning_rate = float(training.get("learning_rate", 0.01))
    log_every = int(training.get("log_every", 10))
    kinds = model_config.get("kinds", [model_config.get("kind", "tanh")])
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    max_initial_drift_error = 0.0
    all_finite = True
    state_unchanged = True
    gradients_unpopulated = True

    for kind in kinds:
        for rho in task.get("rho_values", [4]):
            for lag in task.get("lag_separations", [2]):
                for regime in task.get("regimes", ["positive"]):
                    spec = _task_spec(task, float(rho), int(lag), str(regime))
                    for data_seed, model_seed, legacy_seed in _enl_seed_pairs(training):
                        both, weak = (
                            batch.to(device)
                            for batch in make_paired_task(spec, data_seed)
                        )
                        both_model, weak_model = initialize_paired_models(
                            model_config, both, seed=model_seed, kind=str(kind)
                        )
                        before_both = {
                            name: value.detach().clone()
                            for name, value in both_model.state_dict().items()
                        }
                        before_weak = {
                            name: value.detach().clone()
                            for name, value in weak_model.state_dict().items()
                        }
                        jet = paired_initial_response_jet(
                            both_model, weak_model, both, weak
                        )
                        both_kernel = initial_frozen_empirical_kernel(both_model, both)
                        weak_kernel = initial_frozen_empirical_kernel(weak_model, weak)
                        both_trajectory = integrate_frozen_logistic_sgd(
                            both_kernel,
                            steps=steps,
                            learning_rate=learning_rate,
                            log_every=log_every,
                        )
                        weak_trajectory = integrate_frozen_logistic_sgd(
                            weak_kernel,
                            steps=steps,
                            learning_rate=learning_rate,
                            log_every=log_every,
                        )

                        kernel_initial_drift = float(
                            both_trajectory.response_drift[0]
                            - weak_trajectory.response_drift[0]
                        )
                        initial_drift_error = abs(kernel_initial_drift - float(jet.d_0))
                        max_initial_drift_error = max(
                            max_initial_drift_error, initial_drift_error
                        )
                        finite_tensors = (
                            both_kernel.signed_ntk,
                            weak_kernel.signed_ntk,
                            both_trajectory.response,
                            weak_trajectory.response,
                        )
                        all_finite = all_finite and all(
                            bool(torch.isfinite(value).all()) for value in finite_tensors
                        )
                        state_unchanged = state_unchanged and all(
                            torch.equal(value, both_model.state_dict()[name])
                            for name, value in before_both.items()
                        ) and all(
                            torch.equal(value, weak_model.state_dict()[name])
                            for name, value in before_weak.items()
                        )
                        gradients_unpopulated = gradients_unpopulated and all(
                            parameter.grad is None
                            for model in (both_model, weak_model)
                            for parameter in model.parameters()
                        )

                        metadata: dict[str, Any] = {
                            "model_kind": str(kind),
                            "rho": float(rho),
                            "lag_separation": int(lag),
                            "regime": str(regime),
                            "data_seed": data_seed,
                            "model_seed": model_seed,
                        }
                        if legacy_seed is not None:
                            metadata["seed"] = legacy_seed
                        both_rows = _frozen_trajectory_rows(
                            both_trajectory, condition="both", metadata=metadata
                        )
                        weak_rows = _frozen_trajectory_rows(
                            weak_trajectory, condition="weak_only", metadata=metadata
                        )
                        for both_row, weak_row in zip(both_rows, weak_rows):
                            both_row["response_gap"] = (
                                both_row["m_w"] - weak_row["m_w"]
                            )
                            both_row["d_w_equal_time_exact"] = (
                                both_row["direct_drift_w"]
                                - weak_row["direct_drift_w"]
                            )
                        prediction_rows.extend(both_rows)
                        prediction_rows.extend(weak_rows)

                        summary = {
                            **metadata,
                            **_frozen_enl_summary(
                                both_trajectory,
                                weak_trajectory,
                                beta=float(task.get("beta", 0.5)),
                            ),
                            "initial_delta_w": float(jet.delta_0),
                            "initial_j_w": float(jet.j_0),
                            "initial_drift_reconstruction_error": initial_drift_error,
                            "both_ntk_min_eigenvalue": float(
                                torch.linalg.eigvalsh(
                                    both_kernel.signed_ntk.to(torch.float64)
                                ).min()
                            ),
                            "weak_ntk_min_eigenvalue": float(
                                torch.linalg.eigvalsh(
                                    weak_kernel.signed_ntk.to(torch.float64)
                                ).min()
                            ),
                        }
                        summary_rows.append(summary)

                        record_id = (
                            f"{kind}_rho-{float(rho):g}_lag-{int(lag)}_{regime}"
                            f"_data-{data_seed}_model-{model_seed}"
                        )
                        artifact_path = artifact_dir / f"{record_id}.pt"
                        payload = {
                            "schema_version": "frozen-empirical-ntk-v1",
                            "metadata": metadata,
                            "task": {
                                "sequence_length": spec.sequence_length,
                                "n_samples": spec.n_samples,
                                "rho": spec.rho,
                                "lag_separation": spec.lag_separation,
                                "regime": spec.regime,
                                "cue_noise": spec.cue_noise,
                                "background_noise": spec.background_noise,
                            },
                            "both": _frozen_kernel_payload(both_kernel),
                            "weak_only": _frozen_kernel_payload(weak_kernel),
                            "initial_jet": {
                                "delta_0": jet.delta_0.detach().cpu(),
                                "d_0": jet.d_0.detach().cpu(),
                                "j_0": jet.j_0.detach().cpu(),
                            },
                        }
                        atomic_torch_save(payload, artifact_path)
                        manifest_records.append(
                            {
                                "record_id": record_id,
                                **metadata,
                                "artifact": artifact_path.relative_to(run_dir).as_posix(),
                                "artifact_sha256": sha256_file(artifact_path),
                                "artifact_bytes": artifact_path.stat().st_size,
                                "predicted_phase": summary["phase"],
                                "predicted_tau_star_drift": summary["tau_star_drift"],
                                "predicted_tau_star_response": summary["tau_star_response"],
                            }
                        )

    predictions = pd.DataFrame(prediction_rows)
    summaries = pd.DataFrame(summary_rows)
    predictions_path = run_dir / "predictions.csv"
    summary_path = run_dir / "prediction_summary.csv"
    aggregate_path = run_dir / "prediction_aggregate.csv"
    predictions.to_csv(predictions_path, index=False)
    summaries.to_csv(summary_path, index=False)
    _write_aggregate(
        summaries,
        ["model_kind", "rho", "lag_separation", "regime"],
        [
            "initial_d_w",
            "initial_j_w",
            "tau_star_drift",
            "tau_star_response",
            "final_response_gap",
            "tail_area_margin",
        ],
        aggregate_path,
    )
    acceptance = {
        "stage": "initialization_only_preflight",
        "optimizer_constructed": False,
        "training_trajectory_observed": False,
        "state_unchanged": state_unchanged,
        "parameter_gradients_unpopulated": gradients_unpopulated,
        "all_kernel_and_prediction_values_finite": all_finite,
        "max_initial_drift_reconstruction_error": max_initial_drift_error,
    }
    acceptance["passed"] = bool(
        state_unchanged
        and gradients_unpopulated
        and all_finite
        and max_initial_drift_error < 2e-6
    )
    acceptance_path = run_dir / "preflight_acceptance.json"
    acceptance_path.write_text(json.dumps(acceptance, indent=2), encoding="utf-8")

    source_sha256, source_file_count = executable_source_fingerprint()
    tracked_files = [
        predictions_path,
        summary_path,
        aggregate_path,
        acceptance_path,
        run_dir / "config.resolved.yaml",
        run_dir / "environment.json",
    ]
    manifest = {
        "schema_version": "enl-ntk-preflight-v2",
        "source_sha256": source_sha256,
        "source_file_count": source_file_count,
        "prediction_sha256": sha256_file(predictions_path),
        "files": [
            {
                "path": path.relative_to(run_dir).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in tracked_files
        ],
        "records": manifest_records,
        "preflight_passed": acceptance["passed"],
    }
    # JSON has no Infinity literal. Crossing event booleans and the hashed summary
    # retain the decision; null marks a non-finite convenience copy in the manifest.
    for record in manifest["records"]:
        for key in ("predicted_tau_star_drift", "predicted_tau_star_response"):
            if not math.isfinite(float(record[key])):
                record[key] = None
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest_digest = sha256_file(manifest_path)
    (run_dir / "manifest.sha256").write_text(
        f"{manifest_digest}  manifest.json\n", encoding="utf-8"
    )
    return run_dir


def _manifest_member(root: Path, raw_path: Any) -> Path:
    """Resolve one manifest member without permitting traversal or absolute paths."""
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Preflight manifest paths must be non-empty strings.")
    member = PurePosixPath(raw_path)
    if member.is_absolute() or ".." in member.parts or "." in member.parts:
        raise ValueError(f"Unsafe preflight manifest path: {raw_path!r}.")
    candidate = (root / Path(*member.parts)).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"Preflight manifest path escapes its directory: {raw_path!r}.")
    return candidate


def _enl_record_key(values: Mapping[str, Any]) -> tuple[str, float, int, str, int, int]:
    return (
        str(values["model_kind"]),
        float(values["rho"]),
        int(values["lag_separation"]),
        str(values["regime"]),
        int(values["data_seed"]),
        int(values["model_seed"]),
    )


def _expected_enl_record_keys(config: Mapping[str, Any]) -> set[tuple[str, float, int, str, int, int]]:
    task = config["task"]
    model = config["model"]
    training = config["training"]
    kinds = model.get("kinds", [model.get("kind", "tanh")])
    return {
        (str(kind), float(rho), int(lag), str(regime), data_seed, model_seed)
        for kind in kinds
        for rho in task.get("rho_values", [4])
        for lag in task.get("lag_separations", [2])
        for regime in task.get("regimes", ["positive"])
        for data_seed, model_seed, _ in _enl_seed_pairs(training)
    }


def _file_signature(path: Path) -> tuple[str, int]:
    return sha256_file(path), path.stat().st_size


def _verify_enl_preflight(
    config: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], pd.DataFrame, pd.DataFrame, dict[Path, tuple[str, int]]]:
    """Verify a frozen preflight before any optimizer or output directory exists."""
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("E-NL evaluation requires an evaluation mapping.")
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("E-NL evaluation requires a training mapping.")
    if training.get("paired_mode") != "lockstep":
        raise ValueError("E-NL evaluation requires training.paired_mode='lockstep'.")
    if training.get("exact_response_drift") is not True:
        raise ValueError("E-NL evaluation requires training.exact_response_drift=true.")
    if "preflight_dir" not in evaluation or "manifest_sha256" not in evaluation:
        raise ValueError(
            "E-NL evaluation requires evaluation.preflight_dir and an externally "
            "pinned evaluation.manifest_sha256."
        )
    preflight_dir = Path(str(evaluation["preflight_dir"])).expanduser().resolve()
    manifest_path = preflight_dir / "manifest.json"
    sidecar_path = preflight_dir / "manifest.sha256"
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise ValueError(f"Incomplete E-NL preflight directory: {preflight_dir}.")

    sidecar_fields = sidecar_path.read_text(encoding="utf-8").strip().split()
    if (
        len(sidecar_fields) != 2
        or sidecar_fields[1] != "manifest.json"
        or len(sidecar_fields[0]) != 64
        or any(character not in "0123456789abcdef" for character in sidecar_fields[0])
    ):
        raise ValueError("Malformed preflight manifest.sha256 sidecar.")
    actual_manifest_sha256 = sha256_file(manifest_path)
    pinned_manifest_sha256 = str(evaluation["manifest_sha256"]).lower()
    if sidecar_fields[0] != actual_manifest_sha256:
        raise ValueError("Preflight manifest hash does not match manifest.sha256.")
    if pinned_manifest_sha256 != actual_manifest_sha256:
        raise ValueError("Preflight manifest hash does not match the externally pinned hash.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "enl-ntk-preflight-v2":
        raise ValueError("Unsupported E-NL preflight manifest schema.")
    if manifest.get("preflight_passed") is not True:
        raise ValueError("E-NL preflight acceptance did not pass.")

    source_sha256, source_file_count = executable_source_fingerprint()
    if (
        manifest.get("source_sha256") != source_sha256
        or int(manifest.get("source_file_count", -1)) != source_file_count
    ):
        raise ValueError("Executable source does not match the frozen E-NL preflight.")

    snapshots: dict[Path, tuple[str, int]] = {
        manifest_path: _file_signature(manifest_path),
        sidecar_path: _file_signature(sidecar_path),
    }
    file_entries = manifest.get("files")
    if not isinstance(file_entries, list):
        raise ValueError("Preflight manifest files must be a list.")
    declared_paths: set[str] = set()
    required_paths = {
        "predictions.csv",
        "prediction_summary.csv",
        "prediction_aggregate.csv",
        "preflight_acceptance.json",
        "config.resolved.yaml",
        "environment.json",
    }
    for entry in file_entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Every preflight file entry must be a mapping.")
        raw_path = str(entry.get("path", ""))
        if raw_path in declared_paths:
            raise ValueError(f"Duplicate preflight file entry: {raw_path!r}.")
        declared_paths.add(raw_path)
        path = _manifest_member(preflight_dir, raw_path)
        if not path.is_file():
            raise ValueError(f"Missing preflight file: {raw_path}.")
        signature = _file_signature(path)
        if signature != (str(entry.get("sha256")), int(entry.get("bytes", -1))):
            raise ValueError(f"Preflight file hash/size mismatch: {raw_path}.")
        snapshots[path] = signature
    missing_paths = required_paths - declared_paths
    if missing_paths:
        raise ValueError(f"Preflight manifest omits required files: {sorted(missing_paths)}.")

    predictions_path = preflight_dir / "predictions.csv"
    if manifest.get("prediction_sha256") != sha256_file(predictions_path):
        raise ValueError("Frozen prediction SHA-256 does not match predictions.csv.")
    acceptance = json.loads(
        (preflight_dir / "preflight_acceptance.json").read_text(encoding="utf-8")
    )
    acceptance_contract = {
        "stage": "initialization_only_preflight",
        "optimizer_constructed": False,
        "training_trajectory_observed": False,
        "state_unchanged": True,
        "parameter_gradients_unpopulated": True,
        "all_kernel_and_prediction_values_finite": True,
        "passed": True,
    }
    for key, expected in acceptance_contract.items():
        if acceptance.get(key) != expected:
            raise ValueError(f"Preflight acceptance contract failed at {key!r}.")

    frozen_config = load_config(preflight_dir / "config.resolved.yaml")
    for section in ("task", "model", "training", "held_out_protocol"):
        if config.get(section) != frozen_config.get(section):
            raise ValueError(f"Evaluation {section} does not match the frozen preflight config.")

    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Preflight manifest records must be a non-empty list.")
    record_ids: set[str] = set()
    record_keys: set[tuple[str, float, int, str, int, int]] = set()
    artifact_paths: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Every preflight record must be a mapping.")
        record_id = str(record.get("record_id", ""))
        if not record_id or record_id in record_ids:
            raise ValueError(f"Missing or duplicate preflight record_id: {record_id!r}.")
        record_ids.add(record_id)
        key = _enl_record_key(record)
        if key in record_keys:
            raise ValueError(f"Duplicate preflight record key: {key!r}.")
        record_keys.add(key)
        raw_artifact = str(record.get("artifact", ""))
        if raw_artifact in artifact_paths or raw_artifact in declared_paths:
            raise ValueError(f"Duplicate preflight artifact path: {raw_artifact!r}.")
        artifact_paths.add(raw_artifact)
        artifact = _manifest_member(preflight_dir, raw_artifact)
        if not artifact.is_file():
            raise ValueError(f"Missing frozen kernel artifact: {raw_artifact}.")
        signature = _file_signature(artifact)
        if signature != (
            str(record.get("artifact_sha256")),
            int(record.get("artifact_bytes", -1)),
        ):
            raise ValueError(f"Frozen kernel hash/size mismatch: {raw_artifact}.")
        snapshots[artifact] = signature

    expected_keys = _expected_enl_record_keys(config)
    if record_keys != expected_keys:
        raise ValueError("Manifest records do not exactly match the frozen protocol seed pairs.")

    predictions = pd.read_csv(predictions_path)
    prediction_summary = pd.read_csv(preflight_dir / "prediction_summary.csv")
    identity_columns = [
        "model_kind", "rho", "lag_separation", "regime", "data_seed", "model_seed"
    ]
    trajectory_columns = identity_columns + [
        "step", "tau", "condition", "m_w", "direct_drift_w"
    ]
    if not set(trajectory_columns) <= set(predictions.columns):
        raise ValueError("Frozen predictions.csv is missing required columns.")
    if not set(identity_columns + ["phase"]) <= set(prediction_summary.columns):
        raise ValueError("Frozen prediction_summary.csv is missing required columns.")
    prediction_keys = {
        _enl_record_key(row) for row in predictions[identity_columns].to_dict("records")
    }
    summary_keys = {
        _enl_record_key(row)
        for row in prediction_summary[identity_columns].to_dict("records")
    }
    if prediction_keys != record_keys or summary_keys != record_keys:
        raise ValueError("Frozen CSV record identities do not match the manifest.")
    if prediction_summary.duplicated(identity_columns).any():
        raise ValueError("Frozen prediction summary contains duplicate records.")

    training = config["training"]
    steps = int(training.get("steps", 1000))
    log_every = int(training.get("log_every", 10))
    expected_steps = list(range(0, steps + 1, log_every))
    if expected_steps[-1] != steps:
        expected_steps.append(steps)
    for key, group in predictions.groupby(identity_columns, dropna=False):
        if set(group["condition"]) != {"both", "weak_only"}:
            raise ValueError(f"Frozen prediction conditions are incomplete for {key!r}.")
        for condition in ("both", "weak_only"):
            condition_rows = group[group.condition == condition]
            if condition_rows.step.tolist() != expected_steps:
                raise ValueError(f"Frozen prediction grid mismatch for {key!r}/{condition}.")
            expected_tau = np.asarray(expected_steps) * float(training["learning_rate"])
            if not np.allclose(
                condition_rows.tau.to_numpy(), expected_tau, rtol=0.0, atol=1e-12
            ):
                raise ValueError(f"Frozen prediction tau grid mismatch for {key!r}/{condition}.")

    return preflight_dir, manifest, predictions, prediction_summary, snapshots


def _strict_json_value(value: Any) -> Any:
    """Convert NumPy values and non-finite floats into strict JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _write_strict_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_strict_json_value(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _binary_accuracy(group: pd.DataFrame, column: str) -> dict[str, Any]:
    correct = int(group[column].sum())
    total = int(len(group))
    return {"correct": correct, "total": total, "accuracy": correct / total}


def _two_way_bootstrap_accuracy(
    group: pd.DataFrame,
    column: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Resample data- and model-seed clusters independently on a crossed design.

    Rows sharing either seed are dependent. The two-way pigeonhole bootstrap
    resamples each axis, then averages the resulting crossed cells. When multiple
    rows occupy a cell (the overall tanh+GRU result), their cell mean is retained.
    """
    required = {"data_seed", "model_seed", column}
    if not required <= set(group.columns):
        raise ValueError(f"Two-way bootstrap is missing columns: {sorted(required - set(group))}.")
    if replicates < 1:
        raise ValueError("Two-way bootstrap requires at least one replicate.")
    matrix = group.pivot_table(
        index="data_seed", columns="model_seed", values=column, aggfunc="mean"
    ).sort_index().sort_index(axis=1)
    if matrix.empty or matrix.isna().any().any():
        raise ValueError("Two-way bootstrap requires a complete crossed seed factorial.")
    values = matrix.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        data_indices = rng.integers(0, values.shape[0], size=values.shape[0])
        model_indices = rng.integers(0, values.shape[1], size=values.shape[1])
        estimates[index] = float(values[np.ix_(data_indices, model_indices)].mean())
    return {
        "method": "two_way_pigeonhole_bootstrap",
        "n_data_seeds": int(values.shape[0]),
        "n_model_seeds": int(values.shape[1]),
        "replicates": int(replicates),
        "seed": int(seed),
        "point_accuracy": float(values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
    }


def _timing_metrics(group: pd.DataFrame, prefix: str) -> dict[str, Any]:
    errors = group[f"{prefix}_time_error"].dropna().to_numpy(dtype=float)
    result: dict[str, Any] = {
        "finite_pair_count": int(len(errors)),
        "mae": float(np.mean(np.abs(errors))) if len(errors) else float("nan"),
        "bias": float(np.mean(errors)) if len(errors) else float("nan"),
        "correlation": float("nan"),
    }
    finite = group.dropna(subset=[f"predicted_{prefix}_time", f"observed_{prefix}_time"])
    if len(finite) >= 2:
        predicted = finite[f"predicted_{prefix}_time"].to_numpy(dtype=float)
        observed = finite[f"observed_{prefix}_time"].to_numpy(dtype=float)
        if np.std(predicted) > 0 and np.std(observed) > 0:
            result["correlation"] = float(np.corrcoef(predicted, observed)[0, 1])
    return result


def run_enl_evaluate(config: dict[str, Any]) -> Path:
    """Train and score only records from an externally pinned E-NL preflight.

    Verification precedes optimizer construction and run-directory creation. Frozen
    predictions are loaded from hashed CSV bytes and are never reconstructed from
    kernels. Every preflight input is rechecked after training to detect mutation.
    """
    preflight_dir, manifest, predictions, predicted_summary, snapshots = (
        _verify_enl_preflight(config)
    )
    run_dir = create_run_directory(config)
    device = resolve_device(str(config.get("experiment", {}).get("device", "auto")))
    task = config["task"]
    training = dict(config["training"])
    model_config = config["model"]

    identity_columns = [
        "model_kind", "rho", "lag_separation", "regime", "data_seed", "model_seed"
    ]
    key_to_record_id = {
        _enl_record_key(record): str(record["record_id"])
        for record in manifest["records"]
    }
    predictions = predictions.copy()
    predicted_summary = predicted_summary.copy()
    predictions["record_id"] = [
        key_to_record_id[_enl_record_key(row)] for row in predictions.to_dict("records")
    ]
    predicted_summary["record_id"] = [
        key_to_record_id[_enl_record_key(row)]
        for row in predicted_summary.to_dict("records")
    ]

    trajectories: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for record in manifest["records"]:
        spec = _task_spec(
            task,
            float(record["rho"]),
            int(record["lag_separation"]),
            str(record["regime"]),
        )
        both, weak = (
            batch.to(device)
            for batch in make_paired_task(spec, int(record["data_seed"]))
        )
        history, _ = train_paired(
            model_config,
            both,
            weak,
            training,
            {"method": "erm"},
            seed=int(record["model_seed"]),
            kind=str(record["model_kind"]),
        )
        metadata = {
            "record_id": str(record["record_id"]),
            "model_kind": str(record["model_kind"]),
            "rho": float(record["rho"]),
            "lag_separation": int(record["lag_separation"]),
            "regime": str(record["regime"]),
            "data_seed": int(record["data_seed"]),
            "model_seed": int(record["model_seed"]),
        }
        annotated = _annotate(history, **metadata)
        trajectories.extend(annotated)
        summaries.append(
            {
                **metadata,
                **_enl_summary(
                    pd.DataFrame(annotated),
                    beta=float(task.get("beta", 0.5)),
                    phase_delay=float(task.get("phase_delay", 1.0)),
                ),
            }
        )

    observed_trajectories = pd.DataFrame(trajectories)
    observed_summary = pd.DataFrame(summaries)
    observed_trajectories.to_csv(run_dir / "trajectories.csv", index=False)
    observed_summary.to_csv(run_dir / "summary.csv", index=False)
    _write_aggregate(
        observed_summary,
        ["model_kind", "rho", "lag_separation", "regime"],
        [
            "tau_star_drift", "tau_star_response", "tail_positive_area",
            "tail_negative_area", "tail_area_margin", "weak_auc_gap",
            "final_both_m_w", "final_weak_m_w", "final_accuracy", "final_gsi5",
        ],
        run_dir / "aggregate.csv",
    )
    _write_enl_crossover_report(observed_summary, run_dir)

    summary_fields = [
        "record_id", "phase", "drift_crossed", "response_crossed",
        "tail_causal_starvation_certified", "weak_only_learnable",
        "tau_star_drift", "tau_star_response",
    ]
    comparison = predicted_summary[summary_fields].merge(
        observed_summary[summary_fields],
        on="record_id",
        suffixes=("_predicted", "_observed"),
        validate="one_to_one",
    )
    comparison = comparison.merge(
        observed_summary[["record_id", *identity_columns]],
        on="record_id",
        validate="one_to_one",
    )
    comparison["phase_correct"] = (
        comparison.phase_predicted == comparison.phase_observed
    )
    comparison["drift_crossing_correct"] = (
        comparison.drift_crossed_predicted == comparison.drift_crossed_observed
    )
    comparison["response_crossing_correct"] = (
        comparison.response_crossed_predicted == comparison.response_crossed_observed
    )
    comparison["certificate_correct"] = (
        comparison.tail_causal_starvation_certified_predicted
        == comparison.tail_causal_starvation_certified_observed
    )
    comparison["learnability_correct"] = (
        comparison.weak_only_learnable_predicted
        == comparison.weak_only_learnable_observed
    )
    for label, column in (
        ("drift", "tau_star_drift"), ("response", "tau_star_response")
    ):
        predicted_time = comparison[f"{column}_predicted"].to_numpy(dtype=float)
        observed_time = comparison[f"{column}_observed"].to_numpy(dtype=float)
        finite_pair = np.isfinite(predicted_time) & np.isfinite(observed_time)
        comparison[f"predicted_{label}_time"] = np.where(
            np.isfinite(predicted_time), predicted_time, np.nan
        )
        comparison[f"observed_{label}_time"] = np.where(
            np.isfinite(observed_time), observed_time, np.nan
        )
        time_error = np.full(predicted_time.shape, np.nan, dtype=float)
        time_error[finite_pair] = (
            predicted_time[finite_pair] - observed_time[finite_pair]
        )
        comparison[f"{label}_time_error"] = time_error

    trajectory_scores: list[dict[str, Any]] = []
    trajectory_key = ["record_id", "step", "condition"]
    predicted_core = predictions[
        trajectory_key
        + [
            "tau", "m_w", "direct_drift_w", "response_gap",
            "d_w_equal_time_exact",
        ]
    ]
    observed_core = observed_trajectories[
        trajectory_key + ["tau", "m_w", "direct_drift_w", "d_w_equal_time_exact"]
    ]
    aligned = predicted_core.merge(
        observed_core,
        on=trajectory_key,
        suffixes=("_predicted", "_observed"),
        validate="one_to_one",
    )
    if len(aligned) != len(predicted_core) or len(aligned) != len(observed_core):
        raise RuntimeError("Observed trajectory grid does not exactly match frozen predictions.")
    if not np.allclose(
        aligned.tau_predicted.to_numpy(),
        aligned.tau_observed.to_numpy(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Observed tau grid does not match frozen predictions.")
    for record_id, group in aligned.groupby("record_id", sort=False):
        both_rows = group[group.condition == "both"].sort_values("step")
        weak_rows = group[group.condition == "weak_only"].sort_values("step")
        if not np.array_equal(both_rows.step.to_numpy(), weak_rows.step.to_numpy()):
            raise RuntimeError(f"Paired score grids differ for record {record_id}.")
        predicted_gap = (
            both_rows.m_w_predicted.to_numpy() - weak_rows.m_w_predicted.to_numpy()
        )
        observed_gap = (
            both_rows.m_w_observed.to_numpy() - weak_rows.m_w_observed.to_numpy()
        )
        predicted_drift = (
            both_rows.direct_drift_w_predicted.to_numpy()
            - weak_rows.direct_drift_w_predicted.to_numpy()
        )
        observed_drift = both_rows.d_w_equal_time_exact_observed.to_numpy()
        if not np.allclose(
            predicted_gap, both_rows.response_gap.to_numpy(), rtol=1e-10, atol=1e-10
        ) or not np.allclose(
            predicted_drift,
            both_rows.d_w_equal_time_exact_predicted.to_numpy(),
            rtol=1e-10,
            atol=1e-10,
        ):
            raise RuntimeError(f"Frozen pair-level columns are inconsistent for {record_id}.")
        if not np.allclose(
            both_rows.m_w_predicted.iloc[0], both_rows.m_w_observed.iloc[0],
            rtol=2e-5, atol=2e-6,
        ) or not np.allclose(
            predicted_drift[0], observed_drift[0], rtol=2e-5, atol=2e-6
        ):
            raise RuntimeError(
                f"Initialization mismatch for {record_id}; config/seed equivalence failed."
            )
        metadata = observed_summary[observed_summary.record_id == record_id].iloc[0]
        trajectory_scores.append(
            {
                "record_id": record_id,
                **{column: metadata[column] for column in identity_columns},
                "n_trajectory_points": int(len(predicted_gap)),
                "response_gap_rmse": float(
                    np.sqrt(np.mean(np.square(predicted_gap - observed_gap)))
                ),
                "drift_rmse": float(
                    np.sqrt(np.mean(np.square(predicted_drift - observed_drift)))
                ),
                "response_gap_sign_agreement": float(
                    np.mean(np.sign(predicted_gap) == np.sign(observed_gap))
                ),
                "drift_sign_agreement": float(
                    np.mean(np.sign(predicted_drift) == np.sign(observed_drift))
                ),
                "both_m_w_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                both_rows.m_w_predicted.to_numpy()
                                - both_rows.m_w_observed.to_numpy()
                            )
                        )
                    )
                ),
                "weak_m_w_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                weak_rows.m_w_predicted.to_numpy()
                                - weak_rows.m_w_observed.to_numpy()
                            )
                        )
                    )
                ),
            }
        )
    trajectory_score_frame = pd.DataFrame(trajectory_scores)
    comparison = comparison.merge(
        trajectory_score_frame.drop(columns=identity_columns),
        on="record_id",
        validate="one_to_one",
    )
    comparison.to_csv(run_dir / "scores.csv", index=False)

    primary_columns = {
        "phase": "phase_correct",
        "drift_crossing": "drift_crossing_correct",
        "response_crossing": "response_crossing_correct",
        "causal_certificate": "certificate_correct",
        "weak_only_learnability": "learnability_correct",
    }
    held_out_protocol = config.get("held_out_protocol", {})
    if held_out_protocol and not isinstance(held_out_protocol, Mapping):
        raise ValueError("held_out_protocol must be a mapping when provided.")
    configured_primary = (
        held_out_protocol.get("primary_outputs", list(primary_columns))
        if isinstance(held_out_protocol, Mapping)
        else list(primary_columns)
    )
    if not isinstance(configured_primary, list) or not configured_primary:
        raise ValueError("held_out_protocol.primary_outputs must be a non-empty list.")
    acceptance_primary = [str(name) for name in configured_primary]
    unknown_primary = set(acceptance_primary) - set(primary_columns)
    if unknown_primary:
        raise ValueError(f"Unknown held-out primary outputs: {sorted(unknown_primary)}.")

    inference_config = (
        held_out_protocol.get("reuse_aware_inference", {})
        if isinstance(held_out_protocol, Mapping)
        else {}
    )
    if not isinstance(inference_config, Mapping):
        raise ValueError("held_out_protocol.reuse_aware_inference must be a mapping.")
    bootstrap_replicates = int(inference_config.get("bootstrap_replicates", 5000))
    bootstrap_seed = int(inference_config.get("bootstrap_seed", 2027))
    classifications: dict[str, Any] = {
        "overall": {
            name: _binary_accuracy(comparison, column)
            for name, column in primary_columns.items()
        },
        "by_model": {},
        "reuse_aware_two_way_bootstrap": {
            "overall": {
                name: _two_way_bootstrap_accuracy(
                    comparison,
                    column,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + index,
                )
                for index, (name, column) in enumerate(primary_columns.items())
            },
            "by_model": {},
        },
    }
    timing: dict[str, Any] = {"overall": {}, "by_model": {}}
    trajectory_metrics: dict[str, Any] = {"by_model": {}}
    for label in ("drift", "response"):
        timing["overall"][label] = _timing_metrics(comparison, label)
    for model_index, (model_kind, group) in enumerate(comparison.groupby("model_kind")):
        classifications["by_model"][str(model_kind)] = {
            name: _binary_accuracy(group, column)
            for name, column in primary_columns.items()
        }
        classifications["reuse_aware_two_way_bootstrap"]["by_model"][
            str(model_kind)
        ] = {
            name: _two_way_bootstrap_accuracy(
                group,
                column,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + 100 * (model_index + 1) + index,
            )
            for index, (name, column) in enumerate(primary_columns.items())
        }
        timing["by_model"][str(model_kind)] = {
            label: _timing_metrics(group, label) for label in ("drift", "response")
        }
        trajectory_metrics["by_model"][str(model_kind)] = {
            column: float(group[column].mean())
            for column in (
                "response_gap_rmse", "drift_rmse", "response_gap_sign_agreement",
                "drift_sign_agreement", "both_m_w_rmse", "weak_m_w_rmse",
            )
        }

    acceptance_config = (
        held_out_protocol.get("acceptance", {})
        if isinstance(held_out_protocol, Mapping)
        else {}
    )
    if not acceptance_config:
        acceptance_config = config["evaluation"].get("acceptance", {})
    if not isinstance(acceptance_config, Mapping):
        raise ValueError("Held-out acceptance thresholds must be a mapping.")
    minimum_overall = float(acceptance_config.get("minimum_overall_accuracy", 0.75))
    minimum_by_model = float(acceptance_config.get("minimum_model_accuracy", 0.625))
    required_coverage = float(acceptance_config.get("required_prediction_coverage", 1.0))
    minimum_bootstrap_low_raw = acceptance_config.get(
        "minimum_two_way_bootstrap_ci95_low"
    )
    minimum_bootstrap_low = (
        float(minimum_bootstrap_low_raw)
        if minimum_bootstrap_low_raw is not None
        else None
    )
    coverage = len(comparison) / len(manifest["records"])
    checks = {
        "prediction_coverage": coverage >= required_coverage,
        **{
            f"overall_{name}": classifications["overall"][name]["accuracy"]
            >= minimum_overall
            for name in acceptance_primary
        },
    }
    for model_kind, model_metrics in classifications["by_model"].items():
        for name in acceptance_primary:
            checks[f"{model_kind}_{name}"] = (
                model_metrics[name]["accuracy"] >= minimum_by_model
            )
    if minimum_bootstrap_low is not None:
        overall_bootstrap = classifications["reuse_aware_two_way_bootstrap"]["overall"]
        for name in acceptance_primary:
            checks[f"overall_{name}_bootstrap_ci95_low"] = (
                overall_bootstrap[name]["ci95_low"] >= minimum_bootstrap_low
            )
    thresholds: dict[str, Any] = {
        "minimum_overall_accuracy": minimum_overall,
        "minimum_model_accuracy": minimum_by_model,
        "required_prediction_coverage": required_coverage,
    }
    if minimum_bootstrap_low is not None:
        thresholds["minimum_two_way_bootstrap_ci95_low"] = minimum_bootstrap_low
    evaluation_acceptance = {
        "stage": "held_out_falsification_evaluation",
        "manifest_sha256": sha256_file(preflight_dir / "manifest.json"),
        "prediction_sha256": manifest["prediction_sha256"],
        "accepted_primary_outputs": acceptance_primary,
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
    }
    metrics = {
        "status": (
            held_out_protocol.get("status", "held_out_falsification_evaluation")
            if isinstance(held_out_protocol, Mapping)
            else "held_out_falsification_evaluation"
        ),
        "manifest_sha256": evaluation_acceptance["manifest_sha256"],
        "prediction_sha256": manifest["prediction_sha256"],
        "n_records": int(len(comparison)),
        "classifications": classifications,
        "conditional_crossing_time": timing,
        "trajectory_diagnostics": trajectory_metrics,
        "acceptance": evaluation_acceptance,
    }
    _write_strict_json(run_dir / "metrics.json", metrics)
    _write_strict_json(run_dir / "evaluation_acceptance.json", evaluation_acceptance)

    changed = [
        str(path) for path, signature in snapshots.items()
        if not path.is_file() or _file_signature(path) != signature
    ]
    if changed:
        raise RuntimeError(f"Frozen preflight inputs changed during evaluation: {changed}.")
    provenance = {
        "preflight_dir": str(preflight_dir),
        "manifest_sha256": evaluation_acceptance["manifest_sha256"],
        "prediction_sha256": manifest["prediction_sha256"],
        "verified_file_count": len(snapshots),
        "source_sha256": manifest["source_sha256"],
        "frozen_inputs_unchanged": True,
    }
    _write_strict_json(run_dir / "provenance.json", provenance)
    return run_dir


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
                    for data_seed, model_seed, legacy_seed in _enl_seed_pairs(training):
                        both, weak = (
                            batch.to(device)
                            for batch in make_paired_task(spec, data_seed)
                        )
                        history, _ = train_paired(
                            model_config, both, weak, training, {"method": "erm"},
                            seed=model_seed, kind=str(kind),
                        )
                        metadata = {
                            "model_kind": kind,
                            "rho": float(rho),
                            "lag_separation": int(lag),
                            "regime": regime,
                            "data_seed": data_seed,
                            "model_seed": model_seed,
                        }
                        if legacy_seed is not None:
                            metadata["seed"] = legacy_seed
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
