"""Fail closed on an unfrozen or inconsistent CIFAR confirmation protocol."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path


REQUIRED_EVALUATION = {
    "neutral_core_accuracy",
    "random_cue_accuracy",
    "conflict_accuracy_average_nine_wrong_codes_within_image",
    "weak_response_gap_trajectory",
    "signed_and_positive_part_weak_auc",
    "persistent_suppression_duration",
    "E4_lite_final_checkpoint_balanced_fresh_linear_head",
}


def validate_protocol(spec: dict, *, require_frozen: bool = False) -> list[str]:
    """Return unfinished freeze gates; raise for structurally unsafe protocols."""
    if spec.get("schema_version") != "cifar-matched-v1":
        raise ValueError("Unsupported CIFAR preregistration schema.")
    if spec.get("status") not in {"draft_do_not_run_confirmation", "frozen"}:
        raise ValueError("Unknown or missing preregistration status.")
    seeds = spec["seed_blocks"]
    groups = [seeds["weak_only_calibration"], seeds["debug_pilot_exploratory"], seeds["confirmation"]]
    if len(groups[2]) != 8 or any(len(group) != len(set(group)) for group in groups):
        raise ValueError("Confirmation requires exactly eight unique paired seed blocks.")
    if len(set().union(*map(set, groups))) != sum(map(len, groups)):
        raise ValueError("Calibration, pilot and confirmation seeds must be disjoint.")
    data = spec["dataset"]
    if data["name"] != "CIFAR-10" or data["core_height"] != 32 or data["core_width"] != 32:
        raise ValueError("The experiment must preserve the full CIFAR-10 core.")
    if data["ribbon_height"] <= 0 or data["weak_arm_cue_after_normalization"] != 0:
        raise ValueError("W must have an exactly zero-valued appended cue channel.")
    if data["train_per_class"] + data["validation_per_class"] + data["diagnostic_per_class"] != 5000:
        raise ValueError("Class-stratified training splits must partition each original class.")
    if data["diagnostic_probe_fit_per_class"] + data["diagnostic_evaluation_per_class"] != data["diagnostic_per_class"]:
        raise ValueError("Probe-fit and diagnostic-evaluation sets must be disjoint.")
    if spec["model"]["architecture_count"] != 1:
        raise ValueError("The requested confirmation uses one architecture only.")
    training = spec["training"]
    for field in ("shared_initialization", "shared_training_image_order", "shared_augmentation_randomness", "shared_optimizer_configuration"):
        if training[field] is not True:
            raise ValueError(f"Pairing invariant failed: {field}.")
    if not REQUIRED_EVALUATION.issubset(spec["evaluation"]):
        raise ValueError("A required behavioral or E4-lite endpoint is missing.")
    if spec["response"]["primary_certificate"] != "nondegenerate_W_first_hit_and_negative_B_minus_W_gap_at_that_hit":
        raise ValueError("Primary certificate must match the current manuscript definition.")
    pending = []
    for section, fields in (
        (spec, ("source_commit", "environment_sha256", "weak_only_calibration_manifest_sha256", "cue_only_feasibility_manifest_sha256", "frozen_utc")),
        (data, ("dataset_archive_sha256", "cue_amplitude")),
        (spec["model"], ("architecture_code_sha256",)),
        (training, ("learning_rate", "weight_decay", "horizon_updates", "checkpoint_updates")),
        (spec["response"], ("beta", "scale_S")),
        (spec["external_execution"], ("slurm_resources",)),
    ):
        pending.extend(field for field in fields if section.get(field) is None)
    if spec["status"] == "frozen" and pending:
        raise ValueError(f"Frozen protocol has unresolved fields: {', '.join(pending)}")
    if spec["status"] == "frozen":
        e4 = spec.get("e4_lite")
        if not isinstance(e4, dict):
            raise ValueError("Frozen protocol must specify E4-lite head fitting.")
        grid = e4.get("probe_l2_grid")
        iterations = e4.get("max_iter")
        if (not isinstance(grid, list) or not grid or
                any(not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value < 0 for value in grid) or
                grid != sorted(set(grid)) or
                not isinstance(iterations, int) or iterations <= 0):
            raise ValueError("E4-lite needs a unique sorted nonnegative L2 grid and positive max_iter.")
        for field, value, length in (
            ("source_commit", spec["source_commit"], 40),
            ("environment_sha256", spec["environment_sha256"], 64),
            ("weak_only_calibration_manifest_sha256", spec["weak_only_calibration_manifest_sha256"], 64),
            ("cue_only_feasibility_manifest_sha256", spec["cue_only_feasibility_manifest_sha256"], 64),
            ("dataset_archive_sha256", data["dataset_archive_sha256"], 64),
            ("architecture_code_sha256", spec["model"]["architecture_code_sha256"], 64),
        ):
            if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
                raise ValueError(f"{field} must be a lowercase hexadecimal digest of length {length}.")
        try:
            frozen_time = datetime.fromisoformat(spec["frozen_utc"].replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("frozen_utc must be an ISO-8601 timestamp.") from exc
        if frozen_time.tzinfo is None:
            raise ValueError("frozen_utc must specify a timezone.")
        for field, value in (
            ("cue_amplitude", data["cue_amplitude"]),
            ("learning_rate", training["learning_rate"]),
            ("beta", spec["response"]["beta"]),
            ("scale_S", spec["response"]["scale_S"]),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive.")
        if data["cue_amplitude"] > 1:
            raise ValueError("cue_amplitude must not exceed one with the stated pixel construction.")
        if spec["response"]["scale_S"] < spec["response"]["weak_only_scale_minimum"]:
            raise ValueError("W-only Brier-skill scale failed its preregistered minimum.")
        weight_decay = training["weight_decay"]
        if not isinstance(weight_decay, (int, float)) or not math.isfinite(weight_decay) or not 0 <= weight_decay < 1:
            raise ValueError("weight_decay must lie in [0, 1).")
        horizon = training["horizon_updates"]
        checkpoints = training["checkpoint_updates"]
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon_updates must be a positive integer.")
        if not isinstance(checkpoints, list) or not checkpoints or any(
            not isinstance(step, int) for step in checkpoints
        ) or checkpoints != sorted(set(checkpoints)) or checkpoints[0] <= 0 or checkpoints[-1] != horizon:
            raise ValueError("Checkpoint updates must be unique increasing positive integers ending at H.")
        resources = spec["external_execution"]["slurm_resources"]
        if (not isinstance(resources, dict) or
                resources.get("partition") != "gpu_student" or
                not isinstance(resources.get("gpus"), int) or resources["gpus"] != 1 or
                not isinstance(resources.get("gres"), str) or
                re.fullmatch(r"gpu:a100_1g\.5gb:1", resources["gres"]) is None or
                not isinstance(resources.get("cpus_per_task"), int) or
                resources["cpus_per_task"] < 1 or
                not isinstance(resources.get("memory_gb"), int) or
                resources["memory_gb"] < 1 or
                not isinstance(resources.get("time_limit"), str) or
                re.fullmatch(r"\d{2}:\d{2}:\d{2}", resources["time_limit"]) is None):
            raise ValueError(
                "Frozen DGX resources must fully specify one gpu_student "
                "A100 1g.5gb allocation, CPU, memory and time."
            )
    if require_frozen and (spec["status"] != "frozen" or pending):
        raise ValueError(f"Confirmation blocked: protocol is not frozen; pending {', '.join(pending)}")
    return pending


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.path.read_text())
    try:
        pending = validate_protocol(spec, require_frozen=args.require_frozen)
    except (KeyError, ValueError) as exc:
        parser.exit(2, f"Preregistration check failed: {exc}\n")
    print(f"Protocol structurally valid; status={spec['status']}; pending={pending}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
