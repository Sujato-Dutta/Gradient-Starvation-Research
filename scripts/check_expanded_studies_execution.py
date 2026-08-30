#!/usr/bin/env python3
"""Strict validator for the frozen expanded-studies execution contract.

The mandatory ``contract`` command uses only the Python standard library.  The
optional preflight command may reconstruct initialization-only models, while
authorization and artifact commands inspect bound artifacts.  This file never
creates a seal, optimizer, training trajectory, or scientific outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs" / "expanded_studies_execution.yaml"
CONTRACT_SEMANTIC_SHA256 = "0565e4cceb4a029e8206aa74f289082363a6b55e94bda1cd778c2f6e20fd51f5"
HISTORICAL_SHA256 = "481508ee5381caf6a279d7c992e8e77e5f04679f013309f35c1b5b1a023ce439"
ENVIRONMENT_LOCK_SHA256 = "651ba112e67b38f80eb60d26ef92a348b3e3228fbb0a7aae8a0f24930b823c1c"

EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "study_id",
    "contract_revision",
    "status",
    "execution_authorized",
    "publication_scope",
    "evidence_label",
    "local_workspace",
    "historical_contract",
    "execution_environment",
    "shared_task",
    "seed_design",
    "scientific_record_design",
    "nonlinear_beta_study",
    "cdc_tradeoff_study",
    "inference",
    "preflight",
    "authorization",
    "artifacts",
    "engineering_smoke",
    "prohibited_tuning",
    "prohibited_promotions",
}
EXPECTED_CELLS = [
    {"cell_id": "rho3_lag1", "rho": 3.0, "lag_separation": 1},
    {"cell_id": "rho4_lag2", "rho": 4.0, "lag_separation": 2},
    {"cell_id": "rho5_lag3", "rho": 5.0, "lag_separation": 3},
]
EXPECTED_SHARED_TASK = {
    "sequence_length": 20,
    "n_samples": 256,
    "regime": "positive",
    "cue_noise": 0.1,
    "background_noise": 0.0,
    "cells": EXPECTED_CELLS,
}
EXPECTED_SEED_DESIGN = {
    "data_seeds": [301, 302, 303, 304],
    "model_seeds": [3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017],
    "crossed": True,
    "candidate_seeds_require_untouched_attestation": True,
    "engineering_and_historical_seeds_excluded": True,
    "reuse_across_cells": True,
}
EXPECTED_STUDY_A_OPTIMIZATION = {
    "objective": "cross_entropy",
    "optimizer": "sgd",
    "full_batch": True,
    "steps": 1000,
    "learning_rate": 0.01,
    "log_every": 5,
    "weight_decay": 0.0,
    "gradient_clip": None,
    "paired_mode": "lockstep",
    "exact_response_drift": True,
}
EXPECTED_STUDY_B_OPTIMIZATION = {
    "objective": "cross_entropy",
    "optimizer": "sgd",
    "full_batch": True,
    "steps": 1000,
    "learning_rate": 0.01,
    "log_every": 5,
    "weight_decay": 0.0,
    "gradient_clip": None,
    "paired_mode": "lockstep",
    "max_alpha": None,
    "feasibility_epsilon": 1e-10,
    "ema_decay": 0.9,
}
EXPECTED_METHODS = [
    {"method_id": "erm", "publication_label": "ERM", "uses_weak_only_shadow": False},
    {
        "method_id": "counterfactual_drift",
        "publication_label": "CDC",
        "uses_weak_only_shadow": True,
    },
    {
        "method_id": "bloop",
        "publication_label": "Bloop-style shadow-target rescue",
        "uses_weak_only_shadow": True,
    },
    {
        "method_id": "pcgrad",
        "publication_label": "PCGrad-style shadow-target rescue",
        "uses_weak_only_shadow": True,
    },
]
EXPECTED_INITIALIZATION_MANIFEST = {
    "schema_version": "expanded-studies-initialization-manifest-v1",
    "file_name": "initialization_manifest.json",
    "manifest_digest_semantics": "sha256_of_exact_raw_file_bytes",
    "seal_binding_name": "initialization_manifest_sha256",
    "final_run_copy_semantics": (
        "byte_for_byte_identical_to_authorized_preflight_initialization_manifest"
    ),
    "top_level_keys": [
        "schema_version",
        "study_a_initializations",
        "study_b_base_initializations",
        "study_b_method_references",
    ],
    "exact_top_level_keys_required": True,
    "study_a_entries_key": "study_a_initializations",
    "study_a_entry_keys": [
        "initialization_id",
        "model_kind",
        "cell_id",
        "data_seed",
        "model_seed",
        "paired_task_sha256",
        "shared_initial_state_sha256",
        "paired_state_bitwise_equal",
        "parameter_gradients_absent",
    ],
    "study_a_entry_id_format": "esa-{model_kind}-{cell_id}-d{data_seed}-m{model_seed}",
    "study_a_entry_order": ["model_kind", "cell_id", "data_seed", "model_seed"],
    "study_b_base_entries_key": "study_b_base_initializations",
    "study_b_base_entry_keys": [
        "base_initialization_id",
        "model_kind",
        "cell_id",
        "data_seed",
        "model_seed",
        "paired_task_sha256",
        "shared_initial_state_sha256",
        "parameter_gradients_absent",
    ],
    "study_b_base_entry_id_format": "esb-base-{cell_id}-d{data_seed}-m{model_seed}",
    "study_b_base_entry_order": ["model_kind", "cell_id", "data_seed", "model_seed"],
    "study_b_method_references_key": "study_b_method_references",
    "study_b_method_reference_keys": [
        "method_record_id",
        "model_kind",
        "cell_id",
        "method_id",
        "data_seed",
        "model_seed",
        "base_initialization_id",
        "paired_task_sha256",
        "shared_initial_state_sha256",
        "state_bitwise_equal_to_base",
        "parameter_gradients_absent",
    ],
    "study_b_method_reference_id_format": (
        "esb-{method_id}-{cell_id}-d{data_seed}-m{model_seed}"
    ),
    "study_b_method_reference_order": [
        "model_kind",
        "cell_id",
        "method_id",
        "data_seed",
        "model_seed",
    ],
    "expected_study_a_initialization_count": 192,
    "expected_study_b_base_initialization_count": 64,
    "expected_total_unique_base_initialization_count": 256,
    "expected_study_b_method_reference_count": 256,
    "expected_study_b_method_references_per_base": 4,
    "expected_total_serialized_entry_count": 512,
    "base_initialization_ids_unique_and_study_disjoint": True,
    "entry_arrays_must_equal_complete_ordered_factorials": True,
    "missing_duplicate_or_extra_entries_invalidate_manifest": True,
    "study_b_methods_share_one_base_state": True,
    "study_b_reference_rule": {
        "method_ids_in_order": ["erm", "counterfactual_drift", "bloop", "pcgrad"],
        "exactly_one_reference_per_method_record": True,
        "exactly_four_method_references_per_base": True,
        "reference_coordinates_must_equal_base_coordinates": True,
        "reference_task_digest_must_equal_base_task_digest": True,
        "reference_state_digest_must_equal_base_state_digest": True,
        "all_four_methods_must_reference_one_base_initialization_id": True,
    },
    "digest_fields": ["paired_task_sha256", "shared_initial_state_sha256"],
    "sha256_value_format": "exactly_64_lowercase_hexadecimal_characters",
    "task_tensor_names": ["both.x", "weak_only.x", "y", "signed_labels", "z_s", "z_w"],
    "task_tensor_order": "exact_task_tensor_names_list_order",
    "state_tensor_scope": (
        "all_state_dict_tensor_values_including_parameters_and_persistent_buffers"
    ),
    "state_tensor_order": "lexicographic_state_dict_name",
    "tensor_digest_entry_keys": ["name", "dtype", "shape", "data_hex"],
    "tensor_digest_entry_semantics": {
        "name": "exact_tensor_name_utf8_string",
        "dtype": "numpy_dtype_str_after_explicit_little_endian_normalization",
        "shape": "json_array_of_nonnegative_base10_integer_dimensions",
        "data_hex": "lowercase_hex_of_little_endian_c_contiguous_raw_tensor_bytes",
    },
    "tensor_digest_serialization": (
        "sorted_key_compact_utf8_json_array_allow_nan_false_with_single_trailing_lf"
    ),
    "tensor_digest_semantics": "sha256_of_canonical_json_bytes_of_ordered_tensor_entries",
    "paired_task_digest_semantics": (
        "one_entry_per_task_tensor_in_exact_task_tensor_names_list_order"
    ),
    "shared_initial_state_digest_semantics": (
        "one_entry_per_state_dict_tensor_in_lexicographic_name_order"
    ),
    "state_equality_semantics": (
        "identical_ordered_names_dtypes_shapes_and_little_endian_c_order_raw_bytes"
    ),
    "paired_state_bitwise_equality_required": True,
    "study_b_method_state_bitwise_equality_to_base_required": True,
    "parameter_gradients_must_be_absent": True,
    "parameter_gradients_absent_semantics": (
        "every_named_parameter_grad_is_none_before_and_after_digest_and_reference_creation"
    ),
}
EXPECTED_PREFLIGHT_FILES = [
    "execution-contract.json",
    "historical-contract.json",
    "environment-lock.txt",
    "source_manifest.json",
    "record_manifest.json",
    "bootstrap_draw_manifest.json",
    "initialization_manifest.json",
    "preflight.manifest.json",
    "preflight.manifest.sha256",
]
EXPECTED_RUN_FILES = [
    "contract.resolved.json",
    "historical-contract.json",
    "environment-lock.txt",
    "preflight.manifest.json",
    "preflight.manifest.sha256",
    "authorization.seal.json",
    "source_manifest.json",
    "record_manifest.json",
    "bootstrap_draw_manifest.json",
    "initialization_manifest.json",
    "environment.json",
    "provenance.json",
    "study_a_first_hit_profile.csv",
    "study_a_record_summary.json",
    "study_a_inference.json",
    "study_a_claims.json",
    "study_b_record_summary.csv",
    "study_b_record_summary.json",
    "study_b_diagnostics.csv",
    "study_b_cost.csv",
    "study_b_inference.json",
    "study_b_claims.json",
    "study_b_pareto_trajectory.json",
    "study_b_pareto_final.json",
    "resource_usage.json",
    "artifact_manifest.json",
    "artifact_manifest.sha256",
]
EXPECTED_ATTESTATION = (
    "I attest that no scientific data seed or model seed in this contract has been used "
    "to inspect expanded-study optimizer outcomes before this authorization."
)
EXPECTED_AUTHORIZATION = (
    "I explicitly authorize generation of the 192 nonlinear records and 256 method records "
    "bound by this seal."
)


class ExpandedStudiesExecutionError(RuntimeError):
    """Fail-closed contract or optional delegate validation error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExpandedStudiesExecutionError(message)


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ExpandedStudiesExecutionError(f"Duplicate JSON key: {key!r}.")
        value[key] = item
    return value


def load_strict_json(path: Path) -> dict[str, Any]:
    """Load a JSON-subset YAML mapping, rejecting duplicates and NaN/Infinity."""
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ExpandedStudiesExecutionError(f"Non-finite JSON value: {token}.")
            ),
        )
    except ExpandedStudiesExecutionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExpandedStudiesExecutionError(f"Cannot parse {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} must contain one JSON mapping.")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the frozen finite, sorted, compact JSON representation."""
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ExpandedStudiesExecutionError(f"Value is not canonical finite JSON: {error}") from error
    return (text + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ExpandedStudiesExecutionError(f"Cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _semantic_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    _require(isinstance(item, Mapping), f"{key} must be a mapping.")
    return item


def _parse_exact_lock(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ExpandedStudiesExecutionError(f"Cannot read environment lock {path}: {error}") from error
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        _require(
            line.count("==") == 1
            and not any(token in line for token in (";", " @ ", "[", "]")),
            f"Environment lock line {line_number} is not one exact name==version pin.",
        )
        name, version = line.split("==", 1)
        normalized = name.strip().lower().replace("_", "-")
        _require(normalized != "" and version.strip() != "", f"Invalid lock line {line_number}.")
        _require(normalized not in packages, f"Duplicate environment package: {normalized}.")
        packages[normalized] = version.strip()
    _require(packages, "Environment lock contains no package pins.")
    return packages


def expected_record_ids(contract: Mapping[str, Any]) -> dict[str, list[str]]:
    """Construct the complete ordered Study-A and Study-B scientific ID sets."""
    seeds = _mapping(contract, "seed_design")
    design = _mapping(contract, "scientific_record_design")
    study_a_design = _mapping(design, "study_a")
    study_b_design = _mapping(design, "study_b")
    study_a = _mapping(contract, "nonlinear_beta_study")
    study_b = _mapping(contract, "cdc_tradeoff_study")

    study_a_ids = [
        str(study_a_design["record_id_format"]).format(
            model_kind=model_kind,
            cell_id=cell_id,
            data_seed=data_seed,
            model_seed=model_seed,
        )
        for model_kind in study_a["model_kinds"]
        for cell_id in study_a["cells"]
        for data_seed in seeds["data_seeds"]
        for model_seed in seeds["model_seeds"]
    ]
    study_b_ids = [
        str(study_b_design["record_id_format"]).format(
            model_kind=model_kind,
            cell_id=cell_id,
            method_id=method["method_id"],
            data_seed=data_seed,
            model_seed=model_seed,
        )
        for model_kind in study_b["model_kinds"]
        for cell_id in study_b["cells"]
        for method in study_b["methods"]
        for data_seed in seeds["data_seeds"]
        for model_seed in seeds["model_seeds"]
    ]
    _require(
        len(study_a_ids) == len(set(study_a_ids)) == 192,
        "Study A must have exactly 192 unique scientific record IDs.",
    )
    _require(
        len(study_b_ids) == len(set(study_b_ids)) == 256,
        "Study B must have exactly 256 unique scientific method-record IDs.",
    )
    _require(set(study_a_ids).isdisjoint(study_b_ids), "Study A and Study B IDs overlap.")
    return {"study_a": study_a_ids, "study_b": study_b_ids}


def expected_initialization_bindings(
    contract: Mapping[str, Any],
    *,
    record_ids: Mapping[str, list[str]],
) -> dict[str, Any]:
    """Derive the exact ordered base-initialization IDs and Study-B references."""
    seeds = _mapping(contract, "seed_design")
    design = _mapping(contract, "scientific_record_design")
    study_a_design = _mapping(design, "study_a")
    study_b_design = _mapping(design, "study_b")
    study_a = _mapping(contract, "nonlinear_beta_study")
    study_b = _mapping(contract, "cdc_tradeoff_study")
    preflight = _mapping(contract, "preflight")
    manifest = _mapping(preflight, "initialization_manifest")

    _require(
        manifest["study_a_entry_id_format"] == study_a_design["record_id_format"],
        "Study A initialization IDs must be the exact scientific record IDs.",
    )
    _require(
        manifest["study_b_method_reference_id_format"]
        == study_b_design["record_id_format"],
        "Study B initialization references must use the exact method-record IDs.",
    )

    study_a_initialization_ids = [
        str(manifest["study_a_entry_id_format"]).format(
            model_kind=model_kind,
            cell_id=cell_id,
            data_seed=data_seed,
            model_seed=model_seed,
        )
        for model_kind in study_a["model_kinds"]
        for cell_id in study_a["cells"]
        for data_seed in seeds["data_seeds"]
        for model_seed in seeds["model_seeds"]
    ]
    study_b_base_initialization_ids = [
        str(manifest["study_b_base_entry_id_format"]).format(
            model_kind=model_kind,
            cell_id=cell_id,
            data_seed=data_seed,
            model_seed=model_seed,
        )
        for model_kind in study_b["model_kinds"]
        for cell_id in study_b["cells"]
        for data_seed in seeds["data_seeds"]
        for model_seed in seeds["model_seeds"]
    ]
    study_b_method_references = [
        {
            "method_record_id": str(manifest["study_b_method_reference_id_format"]).format(
                model_kind=model_kind,
                cell_id=cell_id,
                method_id=method["method_id"],
                data_seed=data_seed,
                model_seed=model_seed,
            ),
            "model_kind": model_kind,
            "cell_id": cell_id,
            "method_id": method["method_id"],
            "data_seed": data_seed,
            "model_seed": model_seed,
            "base_initialization_id": str(
                manifest["study_b_base_entry_id_format"]
            ).format(
                model_kind=model_kind,
                cell_id=cell_id,
                data_seed=data_seed,
                model_seed=model_seed,
            ),
        }
        for model_kind in study_b["model_kinds"]
        for cell_id in study_b["cells"]
        for method in study_b["methods"]
        for data_seed in seeds["data_seeds"]
        for model_seed in seeds["model_seeds"]
    ]

    _require(
        study_a_initialization_ids == record_ids["study_a"]
        and len(study_a_initialization_ids)
        == len(set(study_a_initialization_ids))
        == manifest["expected_study_a_initialization_count"]
        == 192,
        "Initialization manifest must bind exactly 192 ordered unique Study-A bases.",
    )
    _require(
        len(study_b_base_initialization_ids)
        == len(set(study_b_base_initialization_ids))
        == manifest["expected_study_b_base_initialization_count"]
        == 64,
        "Initialization manifest must bind exactly 64 ordered unique Study-B bases.",
    )
    all_base_ids = study_a_initialization_ids + study_b_base_initialization_ids
    _require(
        len(all_base_ids)
        == len(set(all_base_ids))
        == manifest["expected_total_unique_base_initialization_count"]
        == 256,
        "Initialization manifest must bind exactly 256 study-disjoint unique bases.",
    )
    _require(
        [item["method_record_id"] for item in study_b_method_references]
        == record_ids["study_b"]
        and len(study_b_method_references)
        == manifest["expected_study_b_method_reference_count"]
        == 256,
        "Initialization manifest must bind every ordered Study-B method record once.",
    )
    _require(
        len(all_base_ids) + len(study_b_method_references)
        == manifest["expected_total_serialized_entry_count"]
        == 512,
        "Initialization manifest serialized entry count changed.",
    )

    expected_method_ids = [method["method_id"] for method in study_b["methods"]]
    _require(
        expected_method_ids == manifest["study_b_reference_rule"]["method_ids_in_order"]
        and len(expected_method_ids)
        == manifest["expected_study_b_method_references_per_base"]
        == 4,
        "Study B initialization method-reference order or arity changed.",
    )
    methods_by_base = {base_id: [] for base_id in study_b_base_initialization_ids}
    for reference in study_b_method_references:
        base_id = reference["base_initialization_id"]
        _require(base_id in methods_by_base, "Study B method references an unknown base.")
        methods_by_base[base_id].append(reference["method_id"])
    _require(
        all(method_ids == expected_method_ids for method_ids in methods_by_base.values()),
        "Every Study-B base must be shared by exactly the four frozen methods in order.",
    )

    return {
        "study_a_initialization_ids": study_a_initialization_ids,
        "study_b_base_initialization_ids": study_b_base_initialization_ids,
        "study_b_method_references": study_b_method_references,
        "total_unique_base_initialization_count": len(all_base_ids),
    }


def _validate_study_a(contract: Mapping[str, Any]) -> int:
    study = _mapping(contract, "nonlinear_beta_study")
    _require(study["model_kinds"] == ["tanh", "gru"], "Study A model kinds changed.")
    _require(study["width"] == 32, "Study A width changed.")
    _require(
        study["cells"] == ["rho3_lag1", "rho4_lag2", "rho5_lag3"],
        "Study A cells changed.",
    )
    _require(study["optimization"] == EXPECTED_STUDY_A_OPTIMIZATION, "Study A optimizer changed.")
    _require(study["beta_values"] == [0.25, 0.5, 1.0, 2.0], "Study A beta grid changed.")
    _require(
        study["logging"]
        == {
            "logged_step_start": 0,
            "logged_step_end": 1000,
            "logged_step_stride": 5,
            "logged_point_count_per_arm": 201,
            "terminal_step_required": True,
            "arms": ["both", "weak_only"],
        },
        "Study A logging grid changed.",
    )
    _require(
        study["first_hit_rule"]
        == {
            "all_four_betas_required": True,
            "delay_tolerance": 0.25,
            "first_upward_hit": True,
            "linear_interpolation_between_logged_points": True,
            "terminal_threshold_replacement_forbidden": True,
            "initial_hit_is_degenerate": True,
            "nonhit_is_unlearnable": True,
        },
        "Study A prospective first-hit rule changed.",
    )
    _require(
        study["outcome_certificate"]
        == {
            "input": "discrete_logged_response_gap_trajectory",
            "response_gap": "both_weak_response_minus_weak_only_weak_response",
            "criterion": "strict_outcome_suppression",
            "implementation_binding": "gradient_starvation.theory.discrete_crossover_certificate",
            "increment_definition": "increment[k]=response_gap[k+1]-response_gap[k]",
            "crossover_index": "first_k_with_increment[k]<-tolerance",
            "strict_transfer_pattern": (
                "crossover_index>0_and_all_prior_increments>tolerance_and_all_increments_"
                "from_crossover_onward<-tolerance"
            ),
            "post_peak_strict_suppression": (
                "at_least_one_response_gap_after_crossover<response_gap[0]-tolerance"
            ),
            "strict_outcome_suppression_definition": (
                "strict_transfer_pattern_and_post_peak_strict_suppression"
            ),
            "interpolation_forbidden": True,
            "tolerance": 1e-10,
            "computed_once_per_record": True,
            "independent_of_beta": True,
            "reuse_same_boolean_for_all_beta_certificates": True,
        },
        "Study A discrete beta-independent outcome certificate changed.",
    )
    _require(
        study["all_beta_certificate"]["missing_beta_invalidates_certificate"] is True
        and len(study["all_beta_certificate"]["requires"]) == 5,
        "Study A all-beta certificate is incomplete.",
    )
    claim = _mapping(study, "aggregate_claim")
    _require(
        claim["architecture_estimand"]
        == "equal_weight_macro_average_of_three_fixed_cell_prevalences"
        and claim["raw_bootstrap_ci95_lower_strictly_greater_than"] == 0.5
        and claim["one_sided_null"]
        == "equal_cell_macro_all_beta_certificate_prevalence<=0.5"
        and claim["holm_family"] == ["tanh", "gru"]
        and claim["holm_alpha"] == 0.05
        and claim["overall_claim_requires_both_architectures"] is True,
        "Study A beta-robust architecture decision rule changed.",
    )
    seeds = _mapping(contract, "seed_design")
    count = (
        len(study["model_kinds"])
        * len(study["cells"])
        * len(seeds["data_seeds"])
        * len(seeds["model_seeds"])
    )
    _require(count == study["record_count"] == 192, "Study A record count changed.")
    return count


def _validate_study_b(contract: Mapping[str, Any]) -> int:
    study = _mapping(contract, "cdc_tradeoff_study")
    _require(study["model_kinds"] == ["tanh"], "Study B model kind changed.")
    _require(study["width"] == 64, "Study B width changed.")
    _require(study["cells"] == ["rho4_lag2", "rho5_lag3"], "Study B cells changed.")
    _require(study["methods"] == EXPECTED_METHODS, "Study B method IDs or style labels changed.")
    _require(
        study["baseline_fidelity"]
        == {
            "bloop_is_canonical": False,
            "pcgrad_is_canonical": False,
            "reason": (
                "Both are response-gradient shadow-target variants sharing CDC's target; "
                "neither has canonical algorithm/reference parity."
            ),
            "canonical_name_promotion_forbidden": True,
        },
        "Study B noncanonical comparator disclosure changed.",
    )
    _require(study["optimization"] == EXPECTED_STUDY_B_OPTIMIZATION, "Study B optimizer changed.")
    _require(
        study["weak_rescue"]["primary_score"] == "negative_absolute_weak_auc_gap"
        and study["weak_rescue"]["formula"] == "-abs(weak_auc_gap)"
        and study["weak_rescue"]["higher_is_better"] is True
        and study["weak_rescue"]["beta_first_hit_delay_report_beta"] == 0.5,
        "Study B weak-rescue endpoint changed.",
    )
    _require(
        study["strong_preservation"]["reference"]
        == "paired ERM at identical task/data/model seed"
        and study["strong_preservation"]["primary_trajectory_deviation"]
        == "absolute_strong_response_auc_difference_from_erm"
        and study["strong_preservation"]["primary_final_deviation"]
        == "absolute_final_strong_response_difference_from_erm",
        "Study B paired-ERM strong endpoints changed.",
    )
    diagnostics = _mapping(study, "diagnostics")
    _require(
        diagnostics["condition"] == "both"
        and diagnostics["denominator"] == "all_201_logged_both_rows"
        and diagnostics["logged_row_count"] == 201
        and diagnostics["logged_step_end"] == 1000
        and diagnostics["terminal_step_included"] is True
        and diagnostics["erm_status"] == diagnostics["erm_serialized_value"] == "N/A",
        "Study B diagnostic denominator or ERM N/A rule changed.",
    )
    cost = _mapping(study, "cost")
    _require(
        cost["worker_isolation"] == "one_fresh_isolated_worker_process_per_whole_method_record"
        and cost["wall_time_scope"]
        == (
            "parent_prelaunch_through_sample_after_worker_exit_and_bundle_validation_"
            "before_final_resources_and_complete_marker_writes"
        )
        and cost["peak_rss_output_unit"] == "bytes"
        and cost["macos_to_bytes_multiplier"] == 1
        and cost["linux_to_bytes_multiplier"] == 1024
        and cost["cuda_status_for_this_cpu_contract"] == "N/A",
        "Study B isolated cost telemetry contract changed.",
    )
    scope = _mapping(study, "primary_claim_scope")
    _require(
        scope["estimand"] == "equal_weight_two_cell_macro"
        and scope["cell_weights"] == {"rho4_lag2": 0.5, "rho5_lag3": 0.5}
        and scope["fixed_cell_intervals_are_descriptive"] is True
        and scope["fixed_cell_intervals_cannot_establish_comparator_win"] is True,
        "Study B primary claim scope changed.",
    )
    families = _mapping(study, "test_families")
    _require(set(families) == {"weak_equivalence", "trajectory_superiority", "final_superiority"}, "Study B test families changed.")
    _require(
        families["weak_equivalence"]["procedure"] == "TOST"
        and families["weak_equivalence"]["family_size"] == 4
        and len(families["weak_equivalence"]["tests"]) == 4
        and families["weak_equivalence"]["multiplicity"] == "Holm"
        and families["weak_equivalence"]["alpha"] == 0.05,
        "Study B four-test weak-equivalence family changed.",
    )
    for family_name in ("trajectory_superiority", "final_superiority"):
        family = families[family_name]
        _require(
            family["family_size"] == 2
            and len(family["tests"]) == 2
            and family["multiplicity"] == "Holm"
            and family["alpha"] == 0.05,
            f"Study B {family_name} family changed.",
        )
    rule = _mapping(study, "tradeoff_rule")
    _require(
        rule["weak_rescue_equivalence_margin"] == 0.05
        and rule["all_raw_ci_and_holm_conditions_required"] is True
        and rule["both_strong_endpoints_required"] is True
        and len(rule["joint_cdc_beats_comparator_requires"]) == 8,
        "Study B final comparator-win conjunction changed.",
    )
    frontiers = _mapping(study, "pareto_frontiers")
    _require(
        frontiers["scope"] == "equal_weight_two_cell_macro"
        and frontiers["trajectory_frontier"]["dimensions"] == 2
        and frontiers["final_frontier"]["dimensions"] == 2
        and frontiers["separate_frontiers_required"] is True,
        "Study B separate two-dimensional Pareto frontiers changed.",
    )
    seeds = _mapping(contract, "seed_design")
    count = (
        len(study["model_kinds"])
        * len(study["cells"])
        * len(study["methods"])
        * len(seeds["data_seeds"])
        * len(seeds["model_seeds"])
    )
    _require(count == study["method_record_count"] == 256, "Study B record count changed.")
    return count


def _validate_inference(contract: Mapping[str, Any]) -> None:
    inference = _mapping(contract, "inference")
    _require(
        inference["seed_reuse_method"] == "two_way_pigeonhole_bootstrap"
        and inference["bootstrap_replicates"] == 5000
        and inference["bootstrap_generator"] == "numpy.random.default_rng"
        and inference["bootstrap_seed"] == 7301
        and inference["draw_algorithm"]
        == "two_sequential_numpy_Generator.integers_calls_data_axis_then_model_axis"
        and inference["draw_call_order"] == ["data_indices", "model_indices"]
        and inference["data_index_draw"]
        == {
            "call": "rng.integers(low=0,high=4,size=(5000,4),dtype=numpy.int64,endpoint=False)",
            "shape": [5000, 4],
            "value_range": [0, 3],
            "sha256_c_contiguous_little_endian_int64_bytes": (
                "643a1476acefa5b6be29979ebc992b7d8331581ebd154996c3cd755695f55152"
            ),
        }
        and inference["model_index_draw"]
        == {
            "call": "rng.integers(low=0,high=8,size=(5000,8),dtype=numpy.int64,endpoint=False)",
            "shape": [5000, 8],
            "value_range": [0, 7],
            "sha256_c_contiguous_little_endian_int64_bytes": (
                "18840767f3b7f2e66b38c09d07874e84c102c96fc8ff653882e2ab4228495e35"
            ),
        }
        and inference["draw_hash_serialization"]
        == "numpy_int64_cast_to_explicit_little_endian_then_C_order_raw_bytes"
        and inference["data_index_draw_count_per_replicate"] == 4
        and inference["model_index_draw_count_per_replicate"] == 8
        and inference["draws_generated_once"] is True
        and inference["draw_reuse"]
        == "reuse_the_same_data_and_model_index_draws_across_all_cells_architectures_methods_and_endpoints",
        "Shared two-way pigeonhole bootstrap calls, hashes, or draw reuse changed.",
    )
    _require(
        inference["confidence_interval"]
        == {
            "method": "raw_bootstrap_percentile",
            "lower_quantile": 0.025,
            "upper_quantile": 0.975,
            "quantile_method": "linear",
            "confidence_level": 0.95,
        },
        "Raw percentile-linear 95% interval changed.",
    )
    p_values = inference["one_sided_p_values"]
    _require(
        p_values["method"] == "centered_bootstrap_null_tail"
        and p_values["center_each_bootstrap_distribution_at_observed_estimate"] is True
        and p_values["plus_one_correction"] is True
        and p_values["denominator"] == 5001,
        "Centered-bootstrap one-sided p-value rule changed.",
    )
    _require(
        inference["holm"]
        == {
            "alpha": 0.05,
            "step_down": True,
            "ties": "stable_preregistered_test_order",
            "families_kept_separate": True,
        },
        "Holm multiplicity procedure changed.",
    )


def _validate_execution_boundaries(contract: Mapping[str, Any]) -> None:
    environment = _mapping(contract, "execution_environment")
    _require(
        environment["device"] == "cpu"
        and environment["dtype"] == "float32"
        and environment["deterministic_algorithms"] is True
        and environment["deterministic_warn_only"] is False
        and environment["environment_lock_sha256"] == ENVIRONMENT_LOCK_SHA256
        and environment["cuda_applicable"] is False
        and environment["cuda_status"] == "not_applicable_cpu_contract",
        "Deterministic CPU/float32 execution environment changed.",
    )
    preflight = _mapping(contract, "preflight")
    _require(
        preflight["source_files"]
        == [
            "run_experiment.py",
            "scripts/run_expanded_studies.py",
            "scripts/check_expanded_studies_execution.py",
        ]
        and preflight["source_globs"] == ["src/**/*.py"]
        and preflight["execution_contract_binding"]
        == {
            "source_path": "configs/expanded_studies_execution.yaml",
            "preflight_copy_path": "execution-contract.json",
            "copy_semantics": "byte_for_byte_identical_raw_file_bytes",
            "digest_semantics": "sha256_of_raw_file_bytes",
            "seal_binding_name": "execution_contract_sha256",
        }
        and preflight["bootstrap_draw_manifest_schema_version"]
        == "expanded-studies-bootstrap-draw-manifest-v1"
        and preflight["bootstrap_draw_manifest_requires_exact_array_hashes"] is True
        and preflight["initialization_manifest"] == EXPECTED_INITIALIZATION_MANIFEST
        and preflight["required_files"] == EXPECTED_PREFLIGHT_FILES
        and preflight["model_construction_for_initialization_hashing_permitted"] is True
        and preflight["model_construction_scope"]
        == "permitted_only_to_hash_shared_initial_state_for_initialization_manifest"
        and preflight[
            "model_parameter_or_buffer_mutation_during_initialization_hashing_forbidden"
        ]
        is True
        and preflight["scientific_data_forward_pass_forbidden"] is True
        and preflight["optimizer_construction_forbidden"] is True
        and preflight["training_forbidden"] is True
        and preflight["model_outcome_inspection_forbidden"] is True
        and preflight["scientific_output_creation_forbidden"] is True,
        "Outcome-free preflight source scope, initialization-only model boundary, or exact artifacts changed.",
    )
    authorization = _mapping(contract, "authorization")
    _require(
        authorization["external_seal_required"] is True
        and authorization["code_must_not_create_seal"] is True
        and authorization["execution_contract_sha256_semantics"]
        == (
            "sha256_of_exact_raw_bytes_of_preflight_execution-contract.json_which_must_be_"
            "byte_for_byte_identical_to_configs/expanded_studies_execution.yaml"
        )
        and authorization["untouched_seed_attestation_text"] == EXPECTED_ATTESTATION
        and authorization["explicit_authorization_text"] == EXPECTED_AUTHORIZATION
        and authorization["fail_before_model_and_optimizer_construction"] is True
        and authorization["fail_before_model_and_optimizer_construction_scope"]
        == (
            "all_post_preflight_model_construction_and_all_optimizer_construction_with_"
            "preflight_initialization_hashing_as_the_only_model_construction_exception"
        )
        and authorization["fail_before_scientific_output_directory_creation"] is True
        and authorization["seal_does_not_mutate_contract_execution_authorized"] is True,
        "External authorization-seal boundary changed.",
    )
    _require(
        authorization["required_bindings"]
        == [
            "execution_contract_sha256",
            "preflight_manifest_sha256",
            "historical_contract_sha256",
            "source_manifest_sha256",
            "source_aggregate_sha256",
            "environment_lock_sha256",
            "record_manifest_sha256",
            "bootstrap_draw_manifest_sha256",
            "initialization_manifest_sha256",
            "study_a_expected_record_ids",
            "study_b_expected_record_ids",
        ],
        "Authorization seal bindings changed.",
    )
    artifacts = _mapping(contract, "artifacts")
    _require(artifacts["required_final_files"] == EXPECTED_RUN_FILES, "Required run artifacts changed.")
    _require(
        artifacts["study_a_required_record_files"]
        == ["trajectory.csv", "summary.json", "resources.json", "complete.json"]
        and artifacts["study_b_required_method_record_files"]
        == ["trajectory.csv", "summary.json", "resources.json", "complete.json"],
        "Per-record artifact protocol changed.",
    )
    smoke = _mapping(contract, "engineering_smoke")
    scientific_seeds = set(contract["seed_design"]["data_seeds"]) | set(
        contract["seed_design"]["model_seeds"]
    )
    _require(
        smoke["data_seed"] == 900201
        and smoke["model_seed"] == 900202
        and smoke["data_seed"] not in scientific_seeds
        and smoke["model_seed"] not in scientific_seeds
        and smoke["steps"] == 2
        and smoke["log_every"] == 1
        and smoke["scientific_id_generation_forbidden"] is True
        and smoke["scientific_acceptance_forbidden"] is True
        and smoke["evidence_eligibility"] is False,
        "Engineering smoke exclusion changed.",
    )


def _validate_linked_files(contract: Mapping[str, Any], repository_root: Path) -> None:
    historical = _mapping(contract, "historical_contract")
    _require(
        historical
        == {
            "path": "configs/expanded_nonlinear_cdc_preregistration.yaml",
            "schema_version": "expanded-nonlinear-cdc-preregistration-v1",
            "sha256": HISTORICAL_SHA256,
            "must_remain_execution_blocked": True,
            "must_remain_execution_authorized_false": True,
        },
        "Historical blocked preregistration binding changed.",
    )
    historical_path = repository_root / historical["path"]
    _require(
        historical_path.is_file() and not historical_path.is_symlink(),
        f"Historical preregistration is missing or symlinked: {historical_path}",
    )
    _require(
        _sha256_file(historical_path) == HISTORICAL_SHA256,
        "Historical blocked preregistration SHA-256 drifted.",
    )
    historical_value = load_strict_json(historical_path)
    _require(
        historical_value.get("schema_version") == historical["schema_version"]
        and historical_value.get("status") == "design_frozen_execution_blocked"
        and historical_value.get("execution_authorized") is False,
        "Historical preregistration is no longer the blocked v1 artifact.",
    )

    environment = _mapping(contract, "execution_environment")
    lock_path = repository_root / environment["environment_lock_path"]
    _require(
        lock_path.is_file() and not lock_path.is_symlink(),
        f"Environment lock is missing or symlinked: {lock_path}",
    )
    _require(_sha256_file(lock_path) == ENVIRONMENT_LOCK_SHA256, "Environment lock SHA-256 drifted.")
    packages = _parse_exact_lock(lock_path)
    _require(packages.get("numpy") == "2.5.2", "Environment lock must pin numpy==2.5.2.")
    _require(packages.get("torch") == "2.13.0", "Environment lock must pin torch==2.13.0.")


def validate_execution_contract(
    contract: Mapping[str, Any],
    *,
    repository_root: Path = ROOT,
    verify_linked_files: bool = True,
) -> dict[str, Any]:
    """Validate every field, list order, and nested value in the frozen contract.

    Explicit checks provide targeted failures for the load-bearing rules.  The
    canonical whole-document fingerprint then rejects every other key addition,
    deletion, reorder, or value change, including prose that limits promotions.
    """
    _require(isinstance(contract, Mapping), "Execution contract must be a mapping.")
    _require(set(contract) == EXPECTED_TOP_LEVEL_KEYS, "Execution contract top-level schema changed.")
    _require(contract["schema_version"] == "expanded-studies-execution-v1", "Schema changed.")
    _require(contract["study_id"] == "expanded-nonlinear-beta-cdc-tradeoff-v1", "Study ID changed.")
    _require(contract["contract_revision"] == 3, "Contract revision changed.")
    _require(
        contract["status"] == "external_authorization_required"
        and contract["execution_authorized"] is False,
        "Contract must remain external_authorization_required and unauthorized.",
    )
    _require(contract["shared_task"] == EXPECTED_SHARED_TASK, "Frozen task cells or data settings changed.")
    _require(contract["seed_design"] == EXPECTED_SEED_DESIGN, "Frozen scientific seeds changed.")

    study_a_count = _validate_study_a(contract)
    study_b_count = _validate_study_b(contract)
    record_ids = expected_record_ids(contract)
    initialization_bindings = expected_initialization_bindings(
        contract,
        record_ids=record_ids,
    )
    design = _mapping(contract, "scientific_record_design")
    _require(
        design["study_a"]["all_beta_values_required_per_record"] == [0.25, 0.5, 1.0, 2.0]
        and design["study_a"]["beta_is_within_record_endpoint_not_record_axis"] is True
        and design["study_a"]["expected_record_count"] == study_a_count
        and design["study_b"]["expected_method_record_count"] == study_b_count
        and design["expected_total_scientific_records"] == study_a_count + study_b_count == 448,
        "Scientific record manifest design changed.",
    )
    smoke = contract["engineering_smoke"]
    forbidden_tokens = {str(smoke["data_seed"]), str(smoke["model_seed"])}
    _require(
        not any(token in record_id for token in forbidden_tokens for record_id in record_ids["study_a"] + record_ids["study_b"]),
        "Engineering smoke seed leaked into a scientific record ID.",
    )
    _validate_inference(contract)
    _validate_execution_boundaries(contract)

    semantic_sha256 = _semantic_sha256(contract)
    _require(
        semantic_sha256 == CONTRACT_SEMANTIC_SHA256,
        (
            "Execution contract differs from the fully frozen "
            "expanded-studies-execution-v1 document: expected "
            f"{CONTRACT_SEMANTIC_SHA256}, got {semantic_sha256}."
        ),
    )
    if verify_linked_files:
        _validate_linked_files(contract, repository_root)

    return {
        "valid": True,
        "schema_version": contract["schema_version"],
        "semantic_sha256": semantic_sha256,
        "study_a_record_count": study_a_count,
        "study_b_method_record_count": study_b_count,
        "total_scientific_record_count": study_a_count + study_b_count,
        "total_unique_base_initialization_count": initialization_bindings[
            "total_unique_base_initialization_count"
        ],
        "execution_authorized": False,
        "authorization_status": "external_seal_required",
    }


def load_execution_contract(
    path: Path,
    *,
    repository_root: Path = ROOT,
    verify_linked_files: bool = True,
) -> dict[str, Any]:
    contract = load_strict_json(path)
    validate_execution_contract(
        contract,
        repository_root=repository_root,
        verify_linked_files=verify_linked_files,
    )
    return contract


def _lazy_delegate(function_name: str) -> Callable[..., Any]:
    source_root = str(ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    try:
        module = importlib.import_module("gradient_starvation.expanded_studies")
    except (ImportError, ModuleNotFoundError) as error:
        raise ExpandedStudiesExecutionError(
            "Optional expanded-studies execution delegates are not importable; "
            "the contract remains validation-only."
        ) from error
    function = getattr(module, function_name, None)
    _require(callable(function), f"Optional delegate {function_name!r} is unavailable.")
    return function


def _mapping_report(value: Any, *, mode: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        report = dict(value)
        _require(report.get("valid") is True, f"Optional {mode} delegate reported invalid output.")
        return report
    if mode == "preflight" and isinstance(getattr(value, "manifest", None), Mapping):
        manifest = value.manifest
        return {
            "valid": True,
            "ready_for_authorization": manifest.get("ready_for_authorization"),
            "blockers": manifest.get("blockers"),
            "execution_authorized": False,
        }
    if mode == "authorize" and isinstance(getattr(value, "seal", None), Mapping):
        return {
            "valid": True,
            "external_authorization_verified": True,
            "execution_authorized": True,
        }
    raise ExpandedStudiesExecutionError(f"Optional {mode} delegate returned no JSON report.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate frozen expanded-study execution inputs.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    contract = subparsers.add_parser("contract", help="Validate every frozen contract field.")
    contract.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)

    preflight = subparsers.add_parser("preflight", help="Lazily validate an existing preflight.")
    preflight.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    preflight.add_argument("--preflight", type=Path, required=True)
    preflight.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Validate hashes while permitting truthfully reported authorization blockers.",
    )

    authorize = subparsers.add_parser(
        "authorize",
        help="Lazily verify a supplied external seal; constructs no model or optimizer.",
    )
    authorize.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    authorize.add_argument("--preflight", type=Path, required=True)
    authorize.add_argument("--seal", type=Path, required=True)

    artifacts = subparsers.add_parser(
        "artifacts",
        help="Lazily rehash and recompute an existing complete run.",
    )
    artifacts.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.mode == "contract":
            contract = load_strict_json(arguments.contract)
            report = validate_execution_contract(contract, repository_root=ROOT)
            report["execution_contract_sha256"] = _sha256_file(arguments.contract)
        elif arguments.mode == "preflight":
            load_execution_contract(arguments.contract, repository_root=ROOT)
            verify_preflight = _lazy_delegate("verify_preflight")
            verified = verify_preflight(
                arguments.contract,
                arguments.preflight,
                repository_root=ROOT,
                require_ready=not arguments.allow_blocked,
            )
            report = _mapping_report(verified, mode="preflight")
        elif arguments.mode == "authorize":
            load_execution_contract(arguments.contract, repository_root=ROOT)
            verify_authorization = _lazy_delegate("verify_authorization")
            authorized = verify_authorization(
                arguments.contract,
                arguments.preflight,
                arguments.seal,
                repository_root=ROOT,
            )
            report = _mapping_report(authorized, mode="authorize")
        else:
            validate_run_artifacts = _lazy_delegate("validate_run_artifacts")
            report = _mapping_report(
                validate_run_artifacts(arguments.run_dir, repository_root=ROOT),
                mode="artifacts",
            )
    except (ExpandedStudiesExecutionError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"expanded-studies execution: FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
