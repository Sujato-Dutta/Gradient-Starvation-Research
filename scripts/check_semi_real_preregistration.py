#!/usr/bin/env python3
"""Validate the frozen semi-real design and fail closed on outcome authorization."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SHA256 = re.compile(r"[0-9a-f]{64}")
EXPECTED_DATASETS = {
    "mnist": (3, 8),
    "fashion_mnist": (0, 6),
}
EXPECTED_PRIMARY = [
    "weak_core_response_auc_gap",
    "causal_starvation_certificate",
    "weak_only_learnability",
]


class SemiRealPreregistrationError(ValueError):
    pass


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SemiRealPreregistrationError(f"Duplicate key: {key!r}.")
        result[key] = value
    return result


def load_preregistration(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SemiRealPreregistrationError(f"Non-finite value: {value}.")
            ),
        )
    except SemiRealPreregistrationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SemiRealPreregistrationError(f"Cannot parse {path}: {error}") from error
    if not isinstance(value, dict):
        raise SemiRealPreregistrationError("Preregistration must be a mapping.")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SemiRealPreregistrationError(message)


def validate_preregistration(value: dict[str, Any]) -> dict[str, Any]:
    _require(
        value.get("schema_version") == "semi-real-preregistration-v1",
        "Unsupported semi-real preregistration schema.",
    )
    _require(value.get("freeze_revision") == 1, "Unexpected freeze revision.")
    _require(
        value.get("status") == "design_frozen_execution_blocked",
        "The current artifact must remain design-frozen and execution-blocked.",
    )
    _require(value.get("execution_authorized") is False, "Outcome execution is forbidden.")

    datasets = value.get("datasets")
    _require(isinstance(datasets, list) and len(datasets) == 2, "Exactly two datasets are required.")
    names: set[str] = set()
    for dataset in datasets:
        _require(isinstance(dataset, dict), "Every dataset entry must be a mapping.")
        name = dataset.get("name")
        _require(name in EXPECTED_DATASETS and name not in names, "Dataset identities are invalid.")
        names.add(name)
        negative, positive = EXPECTED_DATASETS[str(name)]
        _require(
            (dataset.get("negative_class"), dataset.get("positive_class"))
            == (negative, positive),
            f"Frozen class mapping changed for {name}.",
        )
        for field in ("train_per_class", "probe_per_class", "evaluation_per_class"):
            _require(isinstance(dataset.get(field), int) and dataset[field] > 0, f"Invalid {name}.{field}.")
        _require(dataset.get("download") is False, "Outcome configuration must not download data.")
    _require(names == set(EXPECTED_DATASETS), "Dataset set is incomplete.")

    preprocessing = value.get("preprocessing")
    _require(isinstance(preprocessing, dict), "preprocessing must be a mapping.")
    _require(preprocessing.get("augmentation") == "none", "Confirmatory augmentation is forbidden.")
    _require(preprocessing.get("value_map") == "x/127.5-1", "Pixel scaling changed.")
    _require(
        preprocessing.get("resize") == {"height": 14, "width": 14, "mode": "area"},
        "Frozen resize contract changed.",
    )
    _require(
        preprocessing.get("probe_and_evaluation_disjoint_from_training") is True
        and preprocessing.get("probe_and_evaluation_disjoint_from_each_other") is True,
        "Probe/evaluation split isolation is mandatory.",
    )

    intervention = value.get("intervention")
    _require(isinstance(intervention, dict), "intervention must be a mapping.")
    _require(intervention.get("cue_strength") == 4.0, "Generated cue strength changed.")
    _require(
        intervention.get("cue_patch") == {"top": 0, "left": 0, "height": 2, "width": 2},
        "Generated cue support changed.",
    )
    checks = intervention.get("preservation_checks")
    _require(isinstance(checks, list) and len(checks) == 5, "Intervention checks are incomplete.")

    response = value.get("response_functionals")
    _require(isinstance(response, dict), "response_functionals must be a mapping.")
    _require(response.get("common_across_conditions") is True, "Response functionals must be common.")
    _require(response.get("primary_derivative") == "exact full-batch direct autograd Lie derivative", "Exact drift contract changed.")
    _require(response.get("two_mode_Gg_promotable") is False, "Nonlinear Gg promotion is forbidden.")

    optimization = value.get("optimization")
    _require(isinstance(optimization, dict), "optimization must be a mapping.")
    expected_optimization = {
        "objective": "mean_binary_cross_entropy_with_logits",
        "optimizer": "sgd",
        "full_batch": True,
        "learning_rate": 0.01,
        "steps": 500,
        "log_every": 5,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "gradient_clip": None,
        "paired_mode": "lockstep_shared_initialization",
    }
    _require(optimization == expected_optimization, "Frozen optimization contract changed.")

    seed_design = value.get("seed_design")
    _require(isinstance(seed_design, dict), "seed_design must be a mapping.")
    data_seeds = seed_design.get("data_seeds")
    model_seeds = seed_design.get("model_seeds")
    _require(isinstance(data_seeds, list) and len(data_seeds) == len(set(data_seeds)) == 4, "Four unique data seeds are required.")
    _require(isinstance(model_seeds, list) and len(model_seeds) == len(set(model_seeds)) == 8, "Eight unique model seeds are required.")
    _require(seed_design.get("crossed") is True, "The seed design must remain crossed.")
    _require(seed_design.get("engineering_seeds_excluded") is True, "Engineering seeds must be excluded.")

    _require(value.get("primary_endpoints") == EXPECTED_PRIMARY, "Primary endpoints changed.")
    inference = value.get("inference")
    _require(isinstance(inference, dict), "inference must be a mapping.")
    _require(inference.get("method") == "two_way_pigeonhole_bootstrap", "Reuse-aware inference changed.")
    _require(inference.get("bootstrap_replicates") == 5000, "Bootstrap replicate count changed.")

    gates = value.get("authorization_gates")
    _require(isinstance(gates, dict), "authorization_gates must be a mapping.")
    required_gate_names = {
        "reviewed_clean_git_source",
        "committed_preregistration_git_sha",
        "vision_environment_lock_path",
        "vision_environment_lock_sha256",
        "mnist_raw_data_sha256",
        "fashion_mnist_raw_data_sha256",
        "selection_manifest_sha256",
        "untouched_seed_attestation",
        "submission_claim_set_allows_new_outcomes",
    }
    _require(set(gates) == required_gate_names, "Authorization gate schema changed.")
    sha_fields = [name for name in gates if name.endswith("sha256")]
    resolved_sha_fields = [
        name for name in sha_fields
        if isinstance(gates[name], str) and SHA256.fullmatch(gates[name]) is not None
    ]
    authorized = bool(
        gates.get("reviewed_clean_git_source") is True
        and isinstance(gates.get("committed_preregistration_git_sha"), str)
        and len(gates["committed_preregistration_git_sha"]) == 40
        and isinstance(gates.get("vision_environment_lock_path"), str)
        and len(resolved_sha_fields) == len(sha_fields)
        and gates.get("untouched_seed_attestation") is True
        and gates.get("submission_claim_set_allows_new_outcomes") is True
    )
    _require(not authorized, "Resolved gates require a reviewed authorization transition, not this blocked schema.")
    blocked_reasons = value.get("blocked_reasons")
    _require(isinstance(blocked_reasons, list) and len(blocked_reasons) >= 6, "Blocked reasons are incomplete.")
    return {
        "valid": True,
        "execution_authorized": False,
        "dataset_count": len(datasets),
        "record_count": len(datasets) * len(data_seeds) * len(model_seeds),
        "unresolved_sha256_gates": len(sha_fields) - len(resolved_sha_fields),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("configs/semi_real_preregistration.yaml"),
    )
    parser.add_argument("--require-authorized", action="store_true")
    arguments = parser.parse_args()
    try:
        report = validate_preregistration(load_preregistration(arguments.path))
        if arguments.require_authorized and not report["execution_authorized"]:
            raise SemiRealPreregistrationError("Semi-real outcome generation remains blocked.")
    except SemiRealPreregistrationError as error:
        print(f"semi-real preregistration: FAIL: {error}")
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
