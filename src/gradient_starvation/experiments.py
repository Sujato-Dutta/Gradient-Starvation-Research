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
from .metrics import causal_metrics, mean_confidence_interval
from .plotting import plot_e1, plot_e2, plot_e3
from .models.recurrent import DenseLinearRNN
from .theory import exact_dense_linear_geometry, gradient_gram, integrate_projected_flow
from .training import train_paired
from .utils import create_run_directory, resolve_device


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


def _paired_summary(frame: pd.DataFrame, beta: float, phase_delay: float) -> dict[str, Any]:
    both = frame[frame.condition == "both"].sort_values("tau")
    weak = frame[frame.condition == "weak_only"].sort_values("tau")
    if not np.allclose(both.tau.to_numpy(), weak.tau.to_numpy()):
        raise RuntimeError("Paired trajectories do not share the same optimization-time grid.")
    causal = causal_metrics(both.tau.to_numpy(), both.m_w.to_numpy(), weak.m_w.to_numpy(), beta)
    if math.isfinite(causal.delta_tw):
        phase = "delayed_weak" if causal.delta_tw > phase_delay else "jointly_learned"
    elif math.isfinite(causal.weak_hitting_time) and not math.isfinite(causal.both_hitting_time):
        phase = "strongly_starved"
    else:
        phase = "indeterminate"
    return {
        **causal.__dict__,
        "phase": phase,
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
    boundary_path = task.get('theory_boundary_file')
    boundary = pd.read_csv(boundary_path) if boundary_path else None
    plot_e1(summary, run_dir, boundary)
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
                theory["condition"], theory["source"] = condition, "particle_closure"
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
