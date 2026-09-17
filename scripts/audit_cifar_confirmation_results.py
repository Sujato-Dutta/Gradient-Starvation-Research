"""Independently audit the compact frozen CIFAR confirmation evidence.

This script intentionally recomputes the headline statistics from each run's
raw ``summary.json``/``trace.json`` rather than trusting the aggregate file.
It also verifies the hashes and frozen snapshots that remain available in the
compact archive. Model checkpoint payloads are not present in that archive, so
their manifest hashes can only be checked against run metadata here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _interval(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise AssertionError("non-finite value in audited metric")
    mean = float(array.mean())
    half_width = float(
        student_t.ppf(0.975, df=len(array) - 1)
        * array.std(ddof=1)
        / math.sqrt(len(array))
    )
    return {
        "mean": mean,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "n": len(array),
    }


def audit(evidence_root: Path) -> dict[str, Any]:
    config_path = evidence_root / "configs/cifar_matched_preregistration.frozen.json"
    environment_path = evidence_root / "research_scope/cifar_dgx_environment_v2.json"
    config = _read_json(config_path)
    config_sha256 = _sha256(config_path)
    environment_sha256 = _sha256(environment_path)
    seeds = config["seed_blocks"]["confirmation"]
    expected_steps = [0, *config["training"]["checkpoint_updates"]]

    values: dict[str, list[float]] = {
        "signed_normalized_deficit_auc": [],
        "maximum_material_deficit_duration_fraction": [],
        "gap_at_weak_only_first_hit": [],
        "final_brier_skill_gap_B_minus_W": [],
        "original_head_neutral_accuracy_gap_B_minus_W": [],
        "original_head_random_accuracy_gap_B_minus_W": [],
        "original_head_consistent_accuracy_gap_B_minus_W": [],
        "original_head_conflict_accuracy_gap_B_minus_W": [],
        "original_head_neutral_accuracy_both": [],
        "original_head_neutral_accuracy_weak_only": [],
        "fresh_head_neutral_accuracy_both": [],
        "fresh_head_neutral_accuracy_weak_only": [],
        "fresh_head_neutral_accuracy_gap_B_minus_W": [],
        "fresh_head_gap_recovery": [],
    }
    per_seed: list[dict[str, Any]] = []
    unique_order_hashes: set[str] = set()
    unique_split_hashes: set[str] = set()
    unique_initial_hashes: set[str] = set()

    for seed in seeds:
        run = evidence_root / "results" / f"cifar_matched_frozen_v1_seed{seed}"
        metadata_path = run / "metadata.json"
        summary_path = run / "summary.json"
        trace_path = run / "trace.json"
        checkpoints_path = run / "checkpoints.json"
        metadata = _read_json(metadata_path)
        summary = _read_json(summary_path)
        trace = _read_json(trace_path)
        checkpoints = _read_json(checkpoints_path)

        assert metadata["status"] == "complete"
        assert metadata["seed"] == summary["seed"] == seed
        assert metadata["source_commit"] == config["source_commit"]
        assert metadata["config_sha256"] == config_sha256
        assert metadata["environment_sha256"] == environment_sha256
        assert metadata["dataset_archive_sha256"] == config["dataset"]["dataset_archive_sha256"]
        assert _sha256(trace_path) == metadata["trace_sha256"]
        assert _sha256(summary_path) == metadata["summary_sha256"]
        assert _sha256(checkpoints_path) == metadata["checkpoint_manifest_sha256"]
        assert (run / "preregistration.frozen.json").read_bytes() == config_path.read_bytes()
        assert (run / "environment.locked.json").read_bytes() == environment_path.read_bytes()
        assert [row["step"] for row in trace] == expected_steps
        assert [row["step"] for row in checkpoints] == expected_steps
        assert metadata["checkpoint_count"] == len(expected_steps)
        assert checkpoints[0]["weights_sha256_both"] == checkpoints[0]["weights_sha256_weak_only"]
        assert checkpoints[0]["weights_sha256_both"] == summary["training"]["initial_weights_sha256"]
        assert checkpoints[-1]["weights_sha256_both"] == summary["training"]["final_weights_sha256_both"]
        assert checkpoints[-1]["weights_sha256_weak_only"] == summary["training"]["final_weights_sha256_weak_only"]

        gaps = np.asarray(
            [row["both"]["brier_skill"] - row["weak_only"]["brier_skill"] for row in trace],
            dtype=float,
        )
        assert abs(gaps[0]) <= config["response"]["gap_tolerance"]
        assert np.all(gaps[1:] < -config["response"]["gap_tolerance"])
        assert np.allclose(gaps, summary["checkpoint_gap"], rtol=0.0, atol=1e-12)
        assert summary["weak_only_gate_passed"]
        assert summary["primary_causal_certificate"]

        original_gap = summary["original_head_neutral_accuracy_gap_B_minus_W"]
        fresh_b = summary["e4_lite"]["both"]["evaluation"]["accuracy"]
        fresh_w = summary["e4_lite"]["weak_only"]["evaluation"]["accuracy"]
        fresh_gap = summary["e4_lite"]["fresh_head_neutral_accuracy_gap_B_minus_W"]
        assert abs((fresh_b - fresh_w) - fresh_gap) <= 1e-12

        row_values = {
            "signed_normalized_deficit_auc": summary["signed_normalized_deficit_auc"],
            "maximum_material_deficit_duration_fraction": summary["maximum_material_deficit_duration_fraction"],
            "gap_at_weak_only_first_hit": summary["gap_at_weak_only_first_hit"],
            "final_brier_skill_gap_B_minus_W": float(gaps[-1]),
            "original_head_neutral_accuracy_gap_B_minus_W": original_gap,
            "original_head_random_accuracy_gap_B_minus_W": summary["original_head_random_accuracy_gap_B_minus_W"],
            "original_head_consistent_accuracy_gap_B_minus_W": summary["original_head_consistent_accuracy_gap_B_minus_W"],
            "original_head_conflict_accuracy_gap_B_minus_W": summary["original_head_conflict_accuracy_gap_B_minus_W"],
            "original_head_neutral_accuracy_both": summary["original_head_neutral_accuracy_both"],
            "original_head_neutral_accuracy_weak_only": summary["original_head_neutral_accuracy_weak_only"],
            "fresh_head_neutral_accuracy_both": fresh_b,
            "fresh_head_neutral_accuracy_weak_only": fresh_w,
            "fresh_head_neutral_accuracy_gap_B_minus_W": fresh_gap,
            "fresh_head_gap_recovery": fresh_gap - original_gap,
        }
        for key, value in row_values.items():
            values[key].append(float(value))

        unique_order_hashes.add(summary["training"]["training_image_order_sha256"])
        unique_split_hashes.add(summary["training"]["split_indices_sha256"])
        unique_initial_hashes.add(summary["training"]["initial_weights_sha256"])
        per_seed.append(
            {
                "seed": seed,
                "weak_only_gate_passed": summary["weak_only_gate_passed"],
                "weak_only_first_hit_update": summary["weak_only_first_hit_update"],
                "primary_causal_certificate": summary["primary_causal_certificate"],
                **row_values,
            }
        )

    aggregate = _read_json(
        evidence_root / "results/cifar_matched_frozen_v1_aggregate/paired_summary.json"
    )
    assert aggregate["completed_seed_count"] == len(seeds)
    assert aggregate["weak_only_gate_pass_count"] == len(seeds)
    assert aggregate["primary_causal_certificate_count"] == len(seeds)
    for key, interval in aggregate["seed_block_95_percent_t_intervals"].items():
        if key in values:
            independently_computed = _interval(values[key])
            assert abs(independently_computed["mean"] - interval["mean"]) <= 1e-12
            assert abs(independently_computed["ci95_low"] - interval["ci95_low"]) <= 1e-12
            assert abs(independently_computed["ci95_high"] - interval["ci95_high"]) <= 1e-12

    stderr_files = sorted((evidence_root / "results/slurm").glob("*.err"))
    assert len(stderr_files) == len(seeds)
    assert all(path.stat().st_size == 0 for path in stderr_files)

    return {
        "status": "audit_passed",
        "evidence_root": str(evidence_root),
        "source_commit": config["source_commit"],
        "config_sha256": config_sha256,
        "environment_sha256": environment_sha256,
        "dataset_archive_sha256": config["dataset"]["dataset_archive_sha256"],
        "seed_count": len(seeds),
        "weak_only_gate_pass_count": sum(row["weak_only_gate_passed"] for row in per_seed),
        "primary_causal_certificate_count": sum(row["primary_causal_certificate"] for row in per_seed),
        "all_post_initialization_checkpoint_gaps_negative": True,
        "unique_training_order_hash_count": len(unique_order_hashes),
        "unique_split_hash_count": len(unique_split_hashes),
        "unique_initialization_hash_count": len(unique_initial_hashes),
        "slurm_stderr_files_empty": True,
        "compact_archive_contains_model_payloads": False,
        "metrics": {key: _interval(metric_values) for key, metric_values in values.items()},
        "per_seed": per_seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.evidence_root)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
