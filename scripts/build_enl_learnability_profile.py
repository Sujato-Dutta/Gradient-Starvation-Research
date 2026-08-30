#!/usr/bin/env python3
"""Build the E-NL weak-only reach audit from tracked completed-run summaries.

This script performs no training and does not read model checkpoints or raw
trajectories. It derives a post-hoc threshold-sensitivity visualization from the
tracked per-seed terminal weak-only responses in the final E-NL summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


MODEL_ORDER = ("tanh", "gru")
MODEL_COLORS = {"tanh": "#0072B2", "gru": "#D55E00"}
REQUIRED_SUMMARY_COLUMNS = {
    "model_kind",
    "rho",
    "lag_separation",
    "regime",
    "seed",
    "final_weak_m_w",
    "weak_hitting_time",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(path: Path, repository_root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        display_path = str(resolved.relative_to(repository_root))
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a top-level mapping.")
    return payload


def _validate_and_derive(
    summary_path: Path, config_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    config = _load_config(config_path)
    task = config.get("task", {})
    training = config.get("training", {})
    beta = float(task["beta"])
    horizon = float(training["steps"]) * float(training["learning_rate"])
    expected_seeds = [int(seed) for seed in training["seeds"]]
    expected_models = [str(kind) for kind in config.get("model", {}).get("kinds", [])]
    if expected_models != list(MODEL_ORDER):
        raise ValueError(
            f"Expected model order {list(MODEL_ORDER)}, found {expected_models}."
        )

    source = pd.read_csv(summary_path)
    missing = REQUIRED_SUMMARY_COLUMNS - set(source.columns)
    if missing:
        raise ValueError(f"Summary is missing required columns: {sorted(missing)}.")
    if set(source["model_kind"]) != set(MODEL_ORDER):
        raise ValueError("Summary must contain exactly the tanh and gru model families.")
    if source.duplicated(["model_kind", "seed"]).any():
        raise ValueError("Summary contains duplicate model_kind/seed records.")

    records = source[
        [
            "model_kind",
            "rho",
            "lag_separation",
            "regime",
            "seed",
            "final_weak_m_w",
            "weak_hitting_time",
        ]
    ].copy()
    records = records.rename(columns={"final_weak_m_w": "B_W_H_terminal"})
    records["seed"] = records["seed"].astype(int)
    records["beta"] = beta
    records["terminal_reach_at_beta"] = records["B_W_H_terminal"] >= beta
    records["first_hit_by_horizon"] = (
        np.isfinite(records["weak_hitting_time"].to_numpy(dtype=float))
        & (records["weak_hitting_time"].to_numpy(dtype=float) > 0.0)
        & (records["weak_hitting_time"].to_numpy(dtype=float) <= horizon)
    )

    for model_kind in MODEL_ORDER:
        model_records = records[records["model_kind"] == model_kind]
        seeds = sorted(model_records["seed"].tolist())
        if seeds != sorted(expected_seeds):
            raise ValueError(
                f"{model_kind} seeds {seeds} do not match config seeds "
                f"{sorted(expected_seeds)}."
            )
        values = model_records["B_W_H_terminal"].to_numpy(dtype=float)
        if len(values) != len(expected_seeds) or not np.isfinite(values).all():
            raise ValueError(f"{model_kind} terminal weak responses are incomplete.")

    records["model_kind"] = pd.Categorical(
        records["model_kind"], categories=MODEL_ORDER, ordered=True
    )
    records = records.sort_values(["model_kind", "seed"]).reset_index(drop=True)

    summary_rows: list[dict[str, Any]] = []
    for model_kind in MODEL_ORDER:
        group = records[records["model_kind"] == model_kind]
        values = group["B_W_H_terminal"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "model_kind": model_kind,
                "n_seeds": len(group),
                "beta": beta,
                "B_W_H_terminal_min": float(values.min()),
                "B_W_H_terminal_median": float(np.median(values)),
                "B_W_H_terminal_mean": float(values.mean()),
                "B_W_H_terminal_max": float(values.max()),
                "n_terminal_reach_at_beta": int(group["terminal_reach_at_beta"].sum()),
                "fraction_terminal_reach_at_beta": float(
                    group["terminal_reach_at_beta"].mean()
                ),
                "n_first_hit_by_horizon": int(group["first_hit_by_horizon"].sum()),
            }
        )
    return records, pd.DataFrame(summary_rows), beta, horizon


def _plot_profile(records: pd.DataFrame, beta: float, output_base: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), constrained_layout=True)

    left = axes[0]
    for model_index, model_kind in enumerate(MODEL_ORDER):
        values = records.loc[
            records["model_kind"] == model_kind, "B_W_H_terminal"
        ].to_numpy(dtype=float)
        jitter = np.linspace(-0.13, 0.13, len(values))
        left.scatter(
            np.full(len(values), model_index) + jitter,
            values,
            s=38,
            color=MODEL_COLORS[model_kind],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            label=f"{model_kind} (n={len(values)})",
        )
        left.hlines(
            np.median(values),
            model_index - 0.22,
            model_index + 0.22,
            color=MODEL_COLORS[model_kind],
            linewidth=2.0,
        )
    left.axhline(beta, color="black", linestyle="--", linewidth=1.2, label=rf"$\beta={beta:g}$")
    left.set_yscale("log")
    left.set_xticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    left.set_ylabel(r"Terminal weak-only response $B_W(H)$")
    left.set_title("Per-seed terminal reach")
    left.grid(axis="y", which="both", alpha=0.18)
    left.legend(fontsize=8, loc="center right")

    right = axes[1]
    all_values = records["B_W_H_terminal"].to_numpy(dtype=float)
    lower = max(float(all_values.min()) * 0.55, 1e-4)
    upper = float(all_values.max()) * 1.25
    thresholds = np.geomspace(lower, upper, 500)
    for model_kind in MODEL_ORDER:
        values = records.loc[
            records["model_kind"] == model_kind, "B_W_H_terminal"
        ].to_numpy(dtype=float)
        reach_fraction = np.asarray([(values >= threshold).mean() for threshold in thresholds])
        right.step(
            thresholds,
            reach_fraction,
            where="post",
            color=MODEL_COLORS[model_kind],
            linewidth=2.0,
            label=model_kind,
        )
    right.axvline(beta, color="black", linestyle="--", linewidth=1.2)
    right.set_xscale("log")
    right.set_ylim(-0.03, 1.03)
    right.set_xlabel(r"Candidate threshold $\beta$")
    right.set_ylabel(r"Fraction with $B_W(H)\geq\beta$")
    right.set_title("Post-hoc terminal-reach profile")
    right.grid(alpha=0.18)
    right.legend(fontsize=8)

    fig.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "Creator": "scripts/build_enl_learnability_profile.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        output_base.with_suffix(".png"),
        dpi=200,
        bbox_inches="tight",
        metadata={"Software": "scripts/build_enl_learnability_profile.py"},
    )
    plt.close(fig)


def build(
    summary_path: Path,
    config_path: Path,
    output_dir: Path,
) -> None:
    summary_path = summary_path.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, aggregate, beta, horizon = _validate_and_derive(
        summary_path, config_path
    )
    records_path = output_dir / "learnability_profile.csv"
    aggregate_path = output_dir / "learnability_profile_summary.csv"
    figure_base = output_dir / "enl_learnability_profile"
    provenance_path = output_dir / "learnability_profile_provenance.json"

    records.to_csv(records_path, index=False, float_format="%.12g")
    aggregate.to_csv(aggregate_path, index=False, float_format="%.12g")
    _plot_profile(records, beta, figure_base)

    repository_root = Path(__file__).resolve().parents[1]
    provenance = {
        "schema_version": "enl-learnability-profile-v1",
        "analysis_status": "post_hoc_threshold_sensitivity_audit",
        "new_training_outcomes_generated": False,
        "definition": {
            "B_W_H_terminal": "M_w^W(H), the terminal weak-only response",
            "profile": "For each candidate beta, the fraction of seeds with B_W_H_terminal >= beta",
            "scope": "Terminal-response sensitivity diagnostic; not a replacement for the first-hitting-time learnability gate in nonmonotone trajectories.",
        },
        "beta_used_in_original_protocol": beta,
        "optimization_horizon": horizon,
        "input_signatures": [
            _signature(summary_path, repository_root),
            _signature(config_path, repository_root),
        ],
        "generator": _signature(Path(__file__).resolve(), repository_root),
        "outputs": [
            _signature(records_path, repository_root),
            _signature(aggregate_path, repository_root),
            _signature(figure_base.with_suffix(".pdf"), repository_root),
            _signature(figure_base.with_suffix(".png"), repository_root),
        ],
        "repository_relative_output_dir": str(output_dir.resolve().relative_to(repository_root)),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    artifact_dir = root / "paper" / "artifacts" / "enl_tanh_crossover-20260824-141207"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=artifact_dir / "summary.csv")
    parser.add_argument("--config", type=Path, default=artifact_dir / "config.resolved.yaml")
    parser.add_argument("--output-dir", type=Path, default=artifact_dir)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build(args.summary, args.config, args.output_dir)
    print(
        "E-NL learnability profile built from tracked completed-run summaries; "
        "no training executed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
