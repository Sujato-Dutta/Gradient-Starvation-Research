"""Aggregate the eight frozen CIFAR B/W seed blocks without replacing failures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_cifar_preregistration import validate_protocol  # noqa: E402


PAIRED_METRICS = (
    "signed_normalized_deficit_auc",
    "positive_normalized_deficit_auc",
    "maximum_material_deficit_duration_fraction",
    "original_head_neutral_accuracy_gap_B_minus_W",
    "original_head_random_accuracy_gap_B_minus_W",
    "original_head_consistent_accuracy_gap_B_minus_W",
    "original_head_conflict_accuracy_gap_B_minus_W",
    "fresh_head_neutral_accuracy_gap_B_minus_W",
)

ARM_METRICS = tuple(
    f"original_head_{mode}_accuracy_{condition}"
    for mode in ("neutral", "random", "consistent", "conflict")
    for condition in ("both", "weak_only")
)

# Backward-compatible public name used by existing analysis/tests.
METRICS = PAIRED_METRICS


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(config: dict, run_paths: list[Path], *, config_sha256: str) -> dict:
    """Return per-block records and CIs only if all eight are complete."""
    expected = list(config["seed_blocks"]["confirmation"])
    if len(expected) != 8 or len(set(expected)) != 8:
        raise ValueError("Expected eight distinct confirmation seeds.")
    observed = {}
    for path in run_paths:
        metadata_path = path / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError(f"Missing metadata.json: {path}.")
        metadata = json.loads(metadata_path.read_text())
        seed = metadata.get("seed")
        if seed not in expected or seed in observed:
            raise ValueError(f"Unknown or duplicate confirmation seed: {seed}.")
        observed[seed] = (path, metadata)
    rows = []
    for seed in expected:
        if seed not in observed:
            rows.append({"seed": seed, "status": "missing"})
            continue
        path, metadata = observed[seed]
        if metadata.get("config_sha256") != config_sha256:
            raise ValueError(f"Frozen config digest mismatch for seed {seed}.")
        for key, expected_value in (
            ("source_commit", config["source_commit"]),
            ("environment_sha256", config["environment_sha256"]),
            ("dataset_archive_sha256", config["dataset"]["dataset_archive_sha256"]),
        ):
            if metadata.get(key) != expected_value:
                raise ValueError(f"Provenance {key} mismatch for seed {seed}.")
        if metadata.get("status") != "complete":
            rows.append({"seed": seed, "status": "incomplete"})
            continue
        if (metadata.get("last_update") != config["training"]["horizon_updates"] or
                metadata.get("checkpoint_count") != len(config["training"]["checkpoint_updates"]) + 1):
            raise ValueError(f"Checkpoint/horizon contract failed for seed {seed}.")
        trace_path = path / "trace.json"
        checkpoint_manifest_path = path / "checkpoints.json"
        if (not trace_path.is_file() or not checkpoint_manifest_path.is_file() or
                _digest(trace_path) != metadata.get("trace_sha256") or
                _digest(checkpoint_manifest_path) != metadata.get("checkpoint_manifest_sha256")):
            raise ValueError(f"Trace or checkpoint manifest missing/tampered for seed {seed}.")
        trace = json.loads(trace_path.read_text())
        checkpoints = json.loads(checkpoint_manifest_path.read_text())
        expected_steps = [0, *config["training"]["checkpoint_updates"]]
        if ([row.get("step") for row in trace] != expected_steps or
                [row.get("step") for row in checkpoints] != expected_steps):
            raise ValueError(f"Checkpoint grid differs from frozen protocol for seed {seed}.")
        for checkpoint in checkpoints:
            artifact = (path / checkpoint["path"]).resolve()
            if (not artifact.is_relative_to(path.resolve()) or not artifact.is_file() or
                    _digest(artifact) != checkpoint.get("file_sha256")):
                raise ValueError(f"Checkpoint missing/tampered for seed {seed}.")
        summary_path = path / "summary.json"
        if not summary_path.is_file() or _digest(summary_path) != metadata.get("summary_sha256"):
            raise ValueError(f"Summary missing or tampered for seed {seed}.")
        summary = json.loads(summary_path.read_text())
        if summary.get("seed") != seed:
            raise ValueError(f"Summary seed mismatch for seed {seed}.")
        e4 = summary.get("e4_lite", {})
        row = {
            "seed": seed, "status": "complete",
            "weak_only_gate_passed": bool(summary["weak_only_gate_passed"]),
            "primary_causal_certificate": bool(summary["primary_causal_certificate"]),
            "weak_only_first_hit_right_censored": bool(
                summary["weak_only_first_hit_right_censored"]),
            "weak_only_first_hit_update": summary["weak_only_first_hit_update"],
            **{key: float(summary[key]) for key in (*PAIRED_METRICS, *ARM_METRICS)
               if key in summary},
            "fresh_head_neutral_accuracy_gap_B_minus_W": float(
                e4["fresh_head_neutral_accuracy_gap_B_minus_W"]),
        }
        if set((*PAIRED_METRICS, *ARM_METRICS)) - set(row):
            raise ValueError(f"Required paired metric missing for seed {seed}.")
        if not all(math.isfinite(row[key])
                   for key in (*PAIRED_METRICS, *ARM_METRICS)):
            raise ValueError(f"Nonfinite paired metric for seed {seed}.")
        if row["primary_causal_certificate"] and not row["weak_only_gate_passed"]:
            raise ValueError(f"Ungated causal certificate for seed {seed}.")
        rows.append(row)
    complete = [row for row in rows if row["status"] == "complete"]
    all_complete = len(complete) == 8
    intervals = {}
    if all_complete:
        critical = float(student_t.ppf(0.975, df=7))
        for key in (*PAIRED_METRICS, *ARM_METRICS):
            values = np.asarray([row[key] for row in complete], dtype=float)
            mean = float(values.mean())
            half_width = critical * float(values.std(ddof=1)) / math.sqrt(8)
            intervals[key] = {"mean": mean, "ci95_low": mean - half_width,
                              "ci95_high": mean + half_width, "n": 8,
                              "unit": "paired_seed_block"}
    return {
        "status": "complete_eight_block_descriptive_analysis" if all_complete
                  else "incomplete_no_confirmatory_interval",
        "expected_seed_count": 8,
        "completed_seed_count": len(complete),
        "missing_or_incomplete_seeds": [row["seed"] for row in rows
                                        if row["status"] != "complete"],
        "weak_only_gate_pass_count": sum(row["weak_only_gate_passed"] for row in complete),
        "primary_causal_certificate_count": sum(row["primary_causal_certificate"]
                                                 for row in complete),
        "per_seed": rows,
        "seed_block_95_percent_t_intervals": intervals,
        "paired_95_percent_t_intervals": {
            key: intervals[key] for key in PAIRED_METRICS
        } if all_complete else {},
        "interpretation_limit": "Descriptive paired seed-block intervals; no oral-acceptance or universal-mechanism claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("run_directories", type=Path, nargs="*")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Aggregate output already exists; do not overwrite frozen analysis.")
    try:
        config = json.loads(args.config.read_text())
        validate_protocol(config, require_frozen=True)
        result = aggregate(config, args.run_directories,
                           config_sha256=_digest(args.config))
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(2, f"CIFAR aggregation failed: {exc}\n")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "paired_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    columns = ("seed", "status", "weak_only_gate_passed",
               "primary_causal_certificate", "weak_only_first_hit_right_censored",
               "weak_only_first_hit_update", *PAIRED_METRICS, *ARM_METRICS)
    with (args.output / "per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["per_seed"])
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "completed_seed_count": result["completed_seed_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
