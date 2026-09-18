#!/usr/bin/env python3
"""Validate frozen claims, provenance links, and fail-closed readiness gates.

The ``.yaml`` inputs intentionally use JSON syntax, a YAML 1.2 subset. The
validator therefore runs with only the Python standard library before project
dependencies are installed. A valid but not-ready state exits zero;
``--require-ready`` makes any non-passing mandatory gate an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ALLOWED_READINESS_STATUSES = {
    "pass",
    "fail",
    "blocked",
    "unknown",
    "not_applicable",
}
THEORY_IDS = [f"T{index}" for index in range(1, 21)]
INCLUDED_THEORY_IDS = [*[f"T{index}" for index in range(1, 17)], "T19", "T20"]
BLOCKED_THEORY_IDS = ["T17", "T18"]
CROSSING_ENDPOINTS = ["drift_crossing", "response_crossing"]
CIFAR_PRIMARY_ENDPOINTS = [
    "weak_only_gate",
    "at_hit_causal_certificate",
    "behavioral_accuracy",
    "e4_lite",
]
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
EXECUTION_SOURCE_COMMIT = "9ebbc617a9aaad5c3c19e9d5fd4fa08f2d3517d0"
CIFAR_SOURCE_COMMIT = "0726aea12947a30eb60ae32fc9c1eb0a7a122660"
ANONYMOUS_CIFAR_SUPPLEMENT_SHA256 = "da4acec99abdfc4994c81fb27cca140a3937120dbf56eb14eacaf00e6b0464b8"
CANONICAL_SOURCE_BRANCH = "codex/gradient-starvation-research"
FINDINGS_FIRST_HEADLINE_LABEL = "reviewed findings-first causal-gradient-starvation synthesis"
FINDINGS_FIRST_HEADLINE_INTERPRETATION = (
    "Weak learning failure alone is not gradient starvation. Exact finite-width "
    "response dynamics and the matched BOTH/WEAK intervention separate cross-entropy "
    "weighting and response-specific geometry, rate suppression, outcome suppression, "
    "weak-only gate failure, and causal starvation. T19 and the initialization-jet "
    "results provide only local or assumption-scoped signs; they do not supply a global "
    "positive-lag parameter-level predictor. That predictor remains unresolved, as do "
    "general positive-disorder closure and nonlinear kernel-stability claims. A frozen "
    "eight-block CIFAR-10 matched intervention now establishes one controlled existence "
    "result: all eight weak-only controls learn, while all eight cue-present partners "
    "satisfy the preregistered causal-starvation certificate and lose neutral/conflict "
    "performance. This one artificial cue and one CNN do not establish prevalence, "
    "architecture universality, or the continuous-time tail-area theorem."
)
BLOOP_COMPARATOR_LABEL = "Bloop-style shadow-target rescue"
PCGRAD_COMPARATOR_LABEL = "PCGrad-style shadow-target rescue"
E28_SCOPE_LIMIT = (
    f"The frozen joint tradeoff is true only versus the {BLOOP_COMPARATOR_LABEL} and "
    f"false versus the {PCGRAD_COMPARATOR_LABEL}; both comparators are non-canonical, "
    "and no broad CDC superiority is supported."
)
READINESS_DECISION_REASON = (
    "E26 is negative/indeterminate because weak-only learnability is 0/32 in each "
    "dataset; E27 is negative and supports no architecture ranking; E28 is "
    f"comparator-restricted mixed, with the frozen joint tradeoff true only versus the {BLOOP_COMPARATOR_LABEL} "
    f"and false versus the {PCGRAD_COMPARATOR_LABEL}, and both comparators are non-canonical. "
    "Initialization-jet information is local or assumption-scoped, and the stronger "
    "global positive-lag parameter-level predictor remains unresolved. E29 is a positive "
    "frozen CIFAR-10 matched confirmation: 8/8 weak-only gates and 8/8 causal certificates, "
    "with persistent neutral/conflict deficits and only partial E4-lite recovery. It remains "
    "limited to one artificial cue and one CNN. Submission remains no-go because the remaining "
    "blockers are AI-use/submission-form consistency; author/account/policy obligations; "
    "historical replayability; a real sequential benchmark; canonical-baseline parity; "
    "and independent scientific/statistical review."
)
PROMOTION_SCOPE = (
    "E20 restricted drift/response crossing classification; E26 negative/indeterminate "
    "semi-real result with 0/32 weak-only learnability per dataset; E27 negative "
    "beta-robust result without architecture ranking; E28 comparator-restricted mixed "
    f"result, with the frozen joint tradeoff true only versus the non-canonical {BLOOP_COMPARATOR_LABEL} "
    f"and false versus the non-canonical {PCGRAD_COMPARATOR_LABEL}; E29 frozen CIFAR-10 "
    "matched confirmation with 8/8 causal certificates, restricted to one artificial cue "
    "and one CNN architecture"
)
KERNEL_REFRAMING_CRITERION = (
    "Either a nonvacuous initialization-time movement certificate is established or all "
    "frozen-kernel language is explicitly framed as an empirical event heuristic."
)
KERNEL_REFRAMING_EVIDENCE = (
    "Empirical reframing alternative satisfied: paper/main.tex, README.md, and the frozen "
    "claim surfaces frame trained-network use of the initialization-frozen surrogate only "
    "as an empirical event heuristic. Exactness is limited to the time-varying finite-width "
    "flow and the surrogate's own recursion. No tanh/GRU kernel-movement certificate or "
    "global positive-lag parameter-level predictor is claimed; T16 remains conditional and "
    "T18 remains open/blocked."
)

# This ordered mapping is the readiness schema. Gates cannot be deleted, added,
# reordered, or made optional without changing this reviewed validator.
CANONICAL_GATE_REQUIREMENTS = {
    "claims_frozen": True,
    "official_2027_requirements_sourced": True,
    "official_style_package_integrated": True,
    "submission_main_text_at_most_9_pages": True,
    "double_blind_anonymity_audit": True,
    "mandatory_ai_use_statement": True,
    "reproducibility_statement": False,
    "author_openreview_profiles_and_order": True,
    "coauthor_quota_and_reciprocal_reviewing": True,
    "dual_submission_compliance": True,
    "clean_git_source_release": True,
    "exact_environment_lock": True,
    "historical_empirical_replayability": True,
    "compact_archive_integrity_verifier": True,
    "evaluator_fail_closed_regression_tests": True,
    "post_change_full_test_suite": True,
    "fresh_breadth_preregistration": True,
    "fresh_breadth_validation": True,
    "kernel_certificate_or_empirical_reframing": True,
    "sequential_benchmark": True,
    "baseline_parity_audit": True,
    "clean_clone_artifact_reproduction": True,
    "independent_scientific_and_statistical_review": True,
}
EXPECTED_GATE_STATUSES = {
    "claims_frozen": "pass",
    "official_2027_requirements_sourced": "pass",
    "official_style_package_integrated": "pass",
    "submission_main_text_at_most_9_pages": "pass",
    "double_blind_anonymity_audit": "pass",
    "mandatory_ai_use_statement": "unknown",
    "reproducibility_statement": "pass",
    "author_openreview_profiles_and_order": "unknown",
    "coauthor_quota_and_reciprocal_reviewing": "unknown",
    "dual_submission_compliance": "unknown",
    "clean_git_source_release": "pass",
    "exact_environment_lock": "pass",
    "historical_empirical_replayability": "fail",
    "compact_archive_integrity_verifier": "pass",
    "evaluator_fail_closed_regression_tests": "pass",
    "post_change_full_test_suite": "pass",
    "fresh_breadth_preregistration": "pass",
    "fresh_breadth_validation": "pass",
    "kernel_certificate_or_empirical_reframing": "pass",
    "sequential_benchmark": "fail",
    "baseline_parity_audit": "blocked",
    "clean_clone_artifact_reproduction": "pass",
    "independent_scientific_and_statistical_review": "blocked",
}
NEW_OUTCOME_GATE_IDS = [
    "clean_git_source_release",
    "exact_environment_lock",
    "fresh_breadth_preregistration",
]
HISTORICAL_ARCHIVE_ROOTS = {
    "E19": "paper/artifacts/enl_ntk_pilot-20260825",
    "E20": "paper/artifacts/enl_ntk_crossing_factorial-20260825",
}
FROZEN_ACCEPTANCE_CONTRACTS = {
    "E19": {
        "accepted_primary_outputs": [
            "phase",
            "drift_crossing",
            "response_crossing",
            "causal_certificate",
            "weak_only_learnability",
        ],
        "acceptance_contract": {
            "total_records": 16,
            "records_by_model": {"tanh": 8, "gru": 8},
            "required_thresholds": {
                "minimum_model_accuracy": 0.625,
                "minimum_overall_accuracy": 0.75,
                "required_prediction_coverage": 1.0,
            },
        },
    },
    "E20": {
        "accepted_primary_outputs": CROSSING_ENDPOINTS,
        "acceptance_contract": {
            "total_records": 64,
            "records_by_model": {"tanh": 32, "gru": 32},
            "required_thresholds": {
                "minimum_model_accuracy": 0.625,
                "minimum_overall_accuracy": 0.75,
                "minimum_two_way_bootstrap_ci95_low": 0.5,
                "required_prediction_coverage": 1.0,
            },
        },
    },
}
HISTORICAL_RESULT_CONTRACTS = {
    "E19": {
        "role": "sealed broad falsification pilot",
        "correct_counts": {
            "phase": "12/16",
            "drift_crossing": "14/16",
            "response_crossing": "14/16",
            "causal_certificate": "14/16",
            "weak_only_learnability": "12/16",
        },
        "failure": "tanh weak-only-learnability correctness was 4/8, below the preregistered 5/8 threshold",
        "broad_hypothesis_accepted": False,
    },
    "E20": {
        "role": "fresh restricted crossing-event factorial",
        "design": "4 data seeds by 8 model seeds for tanh and GRU at one frozen task/width cell",
        "correct_counts": {
            "response_crossing": "64/64",
            "drift_crossing": "60/64",
            "tanh_drift_crossing": "32/32",
            "gru_drift_crossing": "28/32",
        },
        "two_way_bootstrap_ci95_lower": {
            "response_crossing": 1.0,
            "drift_crossing": 0.8125,
        },
        "restricted_hypothesis_accepted": True,
        "scope_limit": "Only drift- and response-crossing event classification in this frozen cell is supported.",
    },
}
COMPLETED_ARCHIVE_CONTRACTS = {
    "semi-real-generated-cue-v1": {
        "claim_ids": ["E26"],
        "archive_type": "completed-run-compact-archive-v1",
        "compact_archive_root": "paper/artifacts/semi-real-generated-cue-v1-20260830",
        "source_provenance_class": "clean_git_source",
        "git_commit": EXECUTION_SOURCE_COMMIT,
        "git_dirty": False,
        "source_sha256": "02ba4ce4d6aff3ff726d9e798606aed29cff1276a6837e10ff43fb37824ca042",
        "source_file_count": 22,
        "source_snapshot_archived": False,
        "source_reconstructible": True,
        "declared_file_count": 207,
        "omitted_file_count": 192,
        "archive_manifest_sha256": "307aafb10168413113adb7cc97386133679c4d64689ce50671b907b74a8b188f",
        "artifact_manifest_sha256": "3ef7a7f2ce08857bace1451318a226a50df16807c526bdaca262e463646a0c72",
        "central_indexed_file_count": 18,
        "central_indexed_bytes": 559939,
    },
    "expanded-nonlinear-beta-cdc-tradeoff-v1": {
        "claim_ids": ["E27", "E28"],
        "archive_type": "completed-run-compact-archive-v1",
        "compact_archive_root": "paper/artifacts/expanded-studies-v1-20260830",
        "source_provenance_class": "clean_git_source",
        "git_commit": EXECUTION_SOURCE_COMMIT,
        "git_dirty": False,
        "source_sha256": "7f43b602b71199367938c1df51aa201a7d601e57ebb47835bdb5c9282b42f84e",
        "source_file_count": 23,
        "source_snapshot_archived": False,
        "source_reconstructible": True,
        "declared_file_count": 1817,
        "omitted_file_count": 1792,
        "archive_manifest_sha256": "97ea2ad410b238190f845a4fb6a6baaf378fc4eddc946b5e4996fc5b85a58606",
        "artifact_manifest_sha256": "54e4c4acd7d0e1b177f6c8da05bcb1ba848397d44056c447843a1df778564a96",
        "central_indexed_file_count": 28,
        "central_indexed_bytes": 2800551,
    },
}
CLAIM_TO_COMPLETED_ARCHIVE = {
    "E26": "semi-real-generated-cue-v1",
    "E27": "expanded-nonlinear-beta-cdc-tradeoff-v1",
    "E28": "expanded-nonlinear-beta-cdc-tradeoff-v1",
}
CANONICAL_REGENERATION_INTERPRETATION = (
    "Regeneration under current or future source is a new study, not a replay "
    "of either historical study."
)
ZERO_CERTIFICATE_INTERPRETATION = (
    "Zero causal certificates must not be interpreted as evidence of no starvation "
    "because the weak-only learnability gate was 0/32 in each dataset."
)


class ManifestError(ValueError):
    """Raised when a claim/readiness manifest violates its frozen contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"Duplicate JSON/YAML key: {key!r}.")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ManifestError(f"Non-finite JSON constant {value!r} is forbidden.")


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except ManifestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"Cannot parse {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestError(f"{path} must contain a top-level mapping.")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"{label} must be a lowercase 64-character SHA-256 digest.",
    )
    return value


def _indexed_digest(study: dict[str, Any], path: str) -> str:
    entries = study.get("archived_files")
    _require(isinstance(entries, list), "Provenance archived_files must be a list.")
    matches = [
        entry for entry in entries if isinstance(entry, dict) and entry.get("path") == path
    ]
    _require(len(matches) == 1, f"Provenance must index {path} exactly once.")
    return _require_sha256(matches[0].get("sha256"), f"{path}.sha256")


def _provenance_studies(
    provenance: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    _require(provenance.get("schema_version") == 3, "Unsupported provenance schema; schema 3 is required.")

    historical = provenance.get("frozen_empirical_ntk_studies")
    _require(isinstance(historical, list), "Historical provenance studies must be a list.")
    historical_by_id: dict[str, dict[str, Any]] = {}
    for index, study in enumerate(historical):
        _require(isinstance(study, dict), f"Historical provenance study {index} must be a mapping.")
        claim_id = study.get("claim_id")
        _require(claim_id in HISTORICAL_ARCHIVE_ROOTS, f"Unexpected historical claim_id: {claim_id!r}.")
        _require(claim_id not in historical_by_id, f"Duplicate provenance claim_id: {claim_id}.")
        historical_by_id[claim_id] = study
    _require(
        list(historical_by_id) == ["E19", "E20"],
        "Historical provenance must contain exactly E19 and E20 in order.",
    )

    completed = provenance.get("completed_run_archives")
    _require(isinstance(completed, list), "Completed-run provenance archives must be a list.")
    completed_by_id: dict[str, dict[str, Any]] = {}
    assigned_claims: list[str] = []
    for index, archive in enumerate(completed):
        _require(isinstance(archive, dict), f"Completed archive {index} must be a mapping.")
        archive_id = archive.get("study_id")
        _require(
            archive_id in COMPLETED_ARCHIVE_CONTRACTS,
            f"Unexpected completed archive study_id: {archive_id!r}.",
        )
        _require(archive_id not in completed_by_id, f"Duplicate completed archive: {archive_id}.")
        contract = COMPLETED_ARCHIVE_CONTRACTS[archive_id]
        for field in (
            "claim_ids",
            "archive_type",
            "compact_archive_root",
            "source_provenance_class",
            "git_commit",
            "git_dirty",
            "source_sha256",
            "source_file_count",
            "source_snapshot_archived",
            "source_reconstructible",
            "declared_file_count",
            "omitted_file_count",
        ):
            _require(
                archive.get(field) == contract[field],
                f"Completed provenance {archive_id}.{field} is not canonical.",
            )
        entries = archive.get("archived_files")
        _require(isinstance(entries, list), f"Completed provenance {archive_id}.archived_files must be a list.")
        _require(
            len(entries) == contract["central_indexed_file_count"]
            and all(isinstance(entry, dict) and isinstance(entry.get("bytes"), int) for entry in entries)
            and sum(entry["bytes"] for entry in entries) == contract["central_indexed_bytes"],
            f"Completed provenance {archive_id} central index count/bytes are not canonical.",
        )
        root = contract["compact_archive_root"]
        _require(
            _indexed_digest(archive, f"{root}/archive_manifest.json")
            == contract["archive_manifest_sha256"],
            f"Completed provenance {archive_id} archive-manifest digest is not canonical.",
        )
        _require(
            _indexed_digest(archive, f"{root}/artifact_manifest.json")
            == contract["artifact_manifest_sha256"],
            f"Completed provenance {archive_id} artifact-manifest digest is not canonical.",
        )
        assigned_claims.extend(archive["claim_ids"])
        completed_by_id[archive_id] = archive
    _require(
        list(completed_by_id) == list(COMPLETED_ARCHIVE_CONTRACTS),
        "Completed provenance must contain exactly the semi-real and shared expanded archives.",
    )
    _require(
        assigned_claims == ["E26", "E27", "E28"] and len(set(assigned_claims)) == 3,
        "Completed provenance claim_ids must assign exactly E26, E27, and E28 once.",
    )
    return historical_by_id, completed_by_id


def _expected_completed_evidence(archive_id: str) -> dict[str, Any]:
    contract = COMPLETED_ARCHIVE_CONTRACTS[archive_id]
    return {
        "archive_study_id": archive_id,
        "claim_ids": contract["claim_ids"],
        "archive_type": contract["archive_type"],
        "compact_archive_root": contract["compact_archive_root"],
        "archive_manifest_sha256": contract["archive_manifest_sha256"],
        "artifact_manifest_sha256": contract["artifact_manifest_sha256"],
        "source_provenance_class": contract["source_provenance_class"],
        "git_commit": contract["git_commit"],
        "git_dirty": contract["git_dirty"],
        "source_sha256": contract["source_sha256"],
        "source_file_count": contract["source_file_count"],
        "source_snapshot_archived": contract["source_snapshot_archived"],
        "source_reconstructible": contract["source_reconstructible"],
        "declared_file_count": contract["declared_file_count"],
        "omitted_file_count": contract["omitted_file_count"],
        "central_indexed_file_count": contract["central_indexed_file_count"],
        "central_indexed_bytes": contract["central_indexed_bytes"],
    }


def _validate_venue_and_theory(claims: dict[str, Any]) -> None:
    venue = claims.get("venue")
    _require(isinstance(venue, dict), "venue must be a mapping.")
    _require(venue.get("target") == "ICLR 2027", "Target venue must be ICLR 2027.")
    _require(
        venue.get("official_venue_requirements_verified") is True,
        "Official venue requirements must be verified from current official sources.",
    )
    sources = venue.get("official_sources")
    _require(isinstance(sources, list) and len(sources) >= 3, "Official sources are incomplete.")
    for index, source in enumerate(sources):
        _require(isinstance(source, dict), f"official_sources[{index}] must be a mapping.")
        url = source.get("url")
        _require(
            isinstance(url, str) and url.startswith("https://iclr.cc/Conferences/2027/"),
            f"official_sources[{index}] must be an official ICLR 2027 HTTPS URL.",
        )

    headline = claims.get("headline_claim")
    _require(isinstance(headline, dict), "headline_claim must be a mapping.")
    _require(
        headline.get("label") == FINDINGS_FIRST_HEADLINE_LABEL
        and headline.get("interpretation") == FINDINGS_FIRST_HEADLINE_INTERPRETATION,
        "The reviewed findings-first headline label or interpretation is stale.",
    )
    _require(
        headline.get("included_theory_claim_ids") == INCLUDED_THEORY_IDS,
        "The included theorem package must be exactly T1--T16 plus T19--T20.",
    )
    _require(
        headline.get("blocked_or_excluded_theory_claim_ids") == BLOCKED_THEORY_IDS,
        "The blocked/excluded theory set must be exactly T17--T18.",
    )
    theory_claims = claims.get("theory_claims")
    _require(isinstance(theory_claims, list), "theory_claims must be a list.")
    ids = [entry.get("id") for entry in theory_claims if isinstance(entry, dict)]
    _require(ids == THEORY_IDS, "Theory claims must appear once each in T1--T20 order.")
    theory_by_id = {entry.get("id"): entry for entry in theory_claims if isinstance(entry, dict)}
    _require(
        theory_by_id["T17"].get("status") == "blocked_conjecture"
        and theory_by_id["T18"].get("status") == "open_blocked",
        "T17 and T18 must remain blocked.",
    )
    _require(
        theory_by_id["T19"].get("status") == "proved_with_assumptions"
        and theory_by_id["T20"].get("status") == "proved_with_assumptions",
        "T19 and T20 must remain assumption-scoped proved claims.",
    )


def _validate_completed_execution(claims: dict[str, Any]) -> None:
    _require("prospective_designs" not in claims, "Stale prospective_designs are forbidden after completed execution.")
    execution = claims.get("completed_externally_sealed_execution")
    _require(isinstance(execution, dict), "completed_externally_sealed_execution must be a mapping.")
    _require(
        execution.get("empirical_claim_ids_assigned") is True
        and execution.get("outcomes_generated") is True
        and execution.get("execution_authorized") is True
        and execution.get("repository_head_was_clean_at_execution") is True
        and execution.get("execution_source_commit") == EXECUTION_SOURCE_COMMIT,
        "Completed execution must retain assigned claims, generated outcomes, authorization, and clean source commit.",
    )
    programs = execution.get("programs")
    _require(isinstance(programs, list), "Completed execution programs must be a list.")
    _require(
        [program.get("id") for program in programs if isinstance(program, dict)]
        == ["semi_real_generated_cue", "expanded_nonlinear_cdc", "cifar_matched_confirmation"],
        "Completed execution must contain exactly the two historical programs and CIFAR confirmation in order.",
    )
    semi_real, expanded, cifar = programs
    _require(
        semi_real.get("status") == "completed_failed_negative_indeterminate"
        and semi_real.get("claim_ids") == ["E26"]
        and semi_real.get("outcome_records") == 64
        and semi_real.get("archive_study_id") == "semi-real-generated-cue-v1"
        and semi_real.get("externally_sealed_preoutcome") is True
        and semi_real.get("human_authorization_seal_archived") is True,
        "Semi-real completed execution state is not canonical.",
    )
    _require(
        expanded.get("status") == "completed_mixed"
        and expanded.get("claim_ids") == ["E27", "E28"]
        and expanded.get("outcome_records") == 448
        and expanded.get("study_a_records") == 192
        and expanded.get("study_b_method_records") == 256
        and expanded.get("archive_study_id") == "expanded-nonlinear-beta-cdc-tradeoff-v1"
        and expanded.get("canonical_named_baseline_count") == 0
        and expanded.get("externally_sealed_preoutcome") is True
        and expanded.get("human_authorization_seal_archived") is True
        and expanded.get("baseline_parity_audit_complete") is False,
        "Expanded completed execution state is not canonical.",
    )
    _require(
        cifar.get("status") == "completed_passed_confirmatory"
        and cifar.get("claim_ids") == ["E29"]
        and cifar.get("paired_seed_blocks") == 8
        and cifar.get("source_commit") == CIFAR_SOURCE_COMMIT
        and cifar.get("externally_frozen_preoutcome") is True
        and cifar.get("preflight_passed") is True
        and cifar.get("all_array_tasks_completed") is True,
        "CIFAR matched confirmation execution state is not canonical.",
    )


def _validate_historical_study(
    study_id: str,
    study: dict[str, Any],
    central: dict[str, Any],
) -> None:
    evidence = study.get("evidence")
    _require(isinstance(evidence, dict), f"{study_id}.evidence must be a mapping.")
    archive_root = HISTORICAL_ARCHIVE_ROOTS[study_id]
    _require(
        evidence.get("compact_archive_root") == archive_root == central.get("compact_archive_root"),
        f"{study_id} compact archive root is detached from provenance.",
    )
    for field in ("manifest_sha256", "prediction_sha256", "source_sha256"):
        claim_digest = _require_sha256(evidence.get(field), f"{study_id}.evidence.{field}")
        central_digest = _require_sha256(central.get(field), f"provenance.{study_id}.{field}")
        _require(claim_digest == central_digest, f"{study_id}.{field} does not match central provenance.")
    archive_manifest_sha256 = _require_sha256(
        evidence.get("archive_manifest_sha256"), f"{study_id}.evidence.archive_manifest_sha256"
    )
    _require(
        archive_manifest_sha256 == _indexed_digest(central, f"{archive_root}/archive_manifest.json"),
        f"{study_id} archive-manifest digest does not match central provenance.",
    )
    _require(
        evidence.get("verified_frozen_file_count") == central.get("verified_frozen_file_count"),
        f"{study_id} verified-file count does not match central provenance.",
    )
    for field in ("source_provenance_class", "source_snapshot_archived", "source_reconstructible"):
        _require(evidence.get(field) == central.get(field), f"{study_id}.{field} does not match central provenance.")
    _require(
        evidence.get("source_provenance_class") == "historical_dirty_unreconstructible"
        and evidence.get("source_snapshot_archived") is False
        and evidence.get("source_reconstructible") is False,
        f"{study_id} must disclose unavailable, unreconstructible dirty source.",
    )
    frozen = FROZEN_ACCEPTANCE_CONTRACTS[study_id]
    historical_result = HISTORICAL_RESULT_CONTRACTS[study_id]
    for field, expected in historical_result.items():
        _require(
            study.get(field) == expected,
            f"{study_id}.{field} differs from the unchanged historical result contract.",
        )
    _require(
        study.get("records") == frozen["acceptance_contract"]["total_records"],
        f"{study_id} record count differs from the frozen acceptance contract.",
    )
    _require(study.get("primary_outputs") == frozen["accepted_primary_outputs"], f"{study_id} primary outputs differ from the frozen acceptance contract.")
    _require(study.get("acceptance_contract") == frozen["acceptance_contract"], f"{study_id} claim acceptance contract is not canonical.")
    _require(central.get("accepted_primary_outputs") == frozen["accepted_primary_outputs"], f"{study_id} primary outputs do not match canonical central provenance.")
    _require(central.get("acceptance_contract") == frozen["acceptance_contract"], f"{study_id} central acceptance contract is not canonical.")
    _require(central.get("evaluation_passed") is (study_id == "E20"), f"{study_id} evaluation pass/fail state does not match the frozen claim.")


def _validate_e26(study: dict[str, Any]) -> None:
    expected_datasets = {
        "MNIST": {
            "records": 32,
            "weak_only_learnable_count": 0,
            "causal_certificate_count": 0,
            "mean_weak_auc_gap": {
                "point_estimate": -0.0012362842136667493,
                "ci95": [-0.0034405428466004646, 0.0010159591371180453],
            },
        },
        "FashionMNIST": {
            "records": 32,
            "weak_only_learnable_count": 0,
            "causal_certificate_count": 0,
            "mean_weak_auc_gap": {
                "point_estimate": -0.00010921728159018996,
                "ci95": [-0.004129384520886106, 0.003182670049955049],
            },
        },
    }
    _require(
        study.get("status") == "failed_negative_indeterminate"
        and study.get("records") == 64
        and study.get("overall_hypothesis_accepted") is False
        and study.get("datasets") == expected_datasets,
        "E26 negative/indeterminate verdict and exact frozen values are not canonical.",
    )
    _require(
        study.get("zero_certificate_interpretation") == ZERO_CERTIFICATE_INTERPRETATION,
        "E26 must forbid interpreting zero certificates as no starvation when learnability is 0/32.",
    )


def _validate_e27(study: dict[str, Any]) -> None:
    expected_architectures = {
        "tanh": {
            "beta_robust_architecture_claim": False,
            "point_estimate": 0.2916666666666667,
            "ci95": [0.16666666666666666, 0.4166666666666667],
            "holm_adjusted_p_value": 1.0,
        },
        "gru": {
            "beta_robust_architecture_claim": False,
            "point_estimate": 0.0,
            "ci95": [0.0, 0.0],
            "holm_adjusted_p_value": 1.0,
        },
    }
    _require(
        study.get("status") == "failed_negative"
        and study.get("records") == 192
        and study.get("overall_beta_robust_claim") is False
        and study.get("architecture_ranking_performed") is False
        and study.get("architectures") == expected_architectures,
        "E27 negative beta-robust verdict and exact frozen values are not canonical.",
    )
    _require("no architecture ranking" in study.get("scope_limit", ""), "E27 must prohibit an architecture ranking.")


def _validate_e28(study: dict[str, Any]) -> None:
    expected_comparators = [
        {
            "id": "bloop",
            "label": BLOOP_COMPARATOR_LABEL,
            "canonical": False,
            "cdc_beats_comparator_joint_tradeoff": True,
            "cdc_minus_comparator_weak_rescue": {
                "point_estimate": 0.002198259399210656,
                "ci95": [0.0009779187294930126, 0.003277366267916477],
            },
            "comparator_minus_cdc_trajectory_deviation": {
                "point_estimate": 3.4651361294978416,
                "ci95": [3.3443827215131754, 3.5849715262780952],
            },
            "comparator_minus_cdc_final_deviation": {
                "point_estimate": 0.6422362388111651,
                "ci95": [0.6288088704226539, 0.6563168084365315],
            },
        },
        {
            "id": "pcgrad",
            "label": PCGRAD_COMPARATOR_LABEL,
            "canonical": False,
            "cdc_beats_comparator_joint_tradeoff": False,
            "cdc_minus_comparator_weak_rescue": {
                "point_estimate": -0.00015351238407674823,
                "ci95": [-0.00047909348592838785, 0.00013248012419808214],
            },
            "comparator_minus_cdc_trajectory_deviation": {
                "point_estimate": -0.48709889128076556,
                "ci95": [-0.6897378944428417, -0.27379327373499074],
            },
            "comparator_minus_cdc_final_deviation": {
                "point_estimate": -0.2565436437726021,
                "ci95": [-0.3016841153614223, -0.2087651835754514],
            },
        },
    ]
    _require(
        study.get("status") == "passed_restricted_mixed"
        and study.get("method_records") == 256
        and study.get("comparators") == expected_comparators
        and study.get("broad_cdc_superiority_supported") is False
        and study.get("scope_limit") == E28_SCOPE_LIMIT,
        "E28 mixed comparator verdict, exact full-label scope, canonical:false labels, or frozen values are not canonical.",
    )


def _validate_e29(study: dict[str, Any]) -> None:
    expected_metrics = {
        "signed_normalized_deficit_auc": {
            "mean": 0.48870154668887456,
            "ci95": [0.4134652840634009, 0.5639378093143482],
        },
        "maximum_material_deficit_duration_fraction": {
            "mean": 0.9829501353141692,
            "ci95": [0.975690854764631, 0.9902094158637074],
        },
        "original_head_neutral_accuracy_gap_B_minus_W": {
            "mean": -0.19894999265670776,
            "ci95": [-0.22594634589590384, -0.17195363941751168],
        },
        "original_head_random_accuracy_gap_B_minus_W": {
            "mean": -0.27559999376535416,
            "ci95": [-0.28722279389842154, -0.2639771936322868],
        },
        "original_head_consistent_accuracy_gap_B_minus_W": {
            "mean": 0.16825001686811447,
            "ci95": [0.15797709192234982, 0.17852294181387912],
        },
        "original_head_conflict_accuracy_gap_B_minus_W": {
            "mean": -0.3188333362340927,
            "ci95": [-0.32958905569404773, -0.3080776167741377],
        },
        "fresh_head_neutral_accuracy_gap_B_minus_W": {
            "mean": -0.1075499951839447,
            "ci95": [-0.11137304047692884, -0.10372694989096057],
        },
        "fresh_head_gap_recovery": {
            "mean": 0.09139999747276306,
            "ci95": [0.06529960806819932, 0.1175003868773268],
        },
    }
    expected_evidence = {
        "source_commit": CIFAR_SOURCE_COMMIT,
        "config_sha256": "8f662f9982f8445bc144a021b9717b2986ee7d4f7d19c03c9a1868f5aa1e2fea",
        "environment_sha256": "1c1e2d126093f449e176a9ef22e637a8a52ceb0ac0574028f7930e0e2266a034",
        "dataset_archive_sha256": "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce",
        "compact_archive_sha256": "4310e61a825b9be939063e59ab27f229a3b4545c819b2de2c610ef1620e7bea8",
        "audit_metrics_path": "research_scope/cifar_confirmation_audit_metrics_20260917.json",
        "audit_metrics_sha256": "88cf4aa2ce37aaa7c41123e2f5807e23602ed5914f35b13af7dd4285b07263c7",
        "audit_report_path": "research_scope/cifar_confirmation_result_audit_20260917.md",
        "audit_report_sha256": "1323d05cfddda32c15f23de0e7bc964e2350c29213666c4f569ea965d1acbd6d",
        "compact_archive_contains_model_payloads": False,
        "full_model_payload_retention": "DGX working storage; external durable backup pending",
    }
    scope_limit = (
        "One artificial ribbon cue and one CNN architecture establish a controlled "
        "existence and diagnostic result, not prevalence, architecture universality, "
        "or the continuous-time tail-area theorem."
    )
    _require(
        study.get("status") == "passed_confirmatory"
        and study.get("role") == "frozen one-architecture CIFAR-10 matched confirmation"
        and study.get("paired_seed_blocks") == 8
        and study.get("weak_only_gate_pass_count") == 8
        and study.get("primary_causal_certificate_count") == 8
        and study.get("all_post_initialization_checkpoint_gaps_negative") is True
        and study.get("metrics") == expected_metrics
        and study.get("scope_limit") == scope_limit
        and study.get("evidence") == expected_evidence,
        "E29 CIFAR confirmatory verdict, exact values, scope, or evidence is not canonical.",
    )


def validate_claim_set(claims: dict[str, Any], provenance: dict[str, Any]) -> None:
    _require(claims.get("schema_version") == "submission-claim-set-v2", "Unsupported submission claim-set schema.")
    _require(claims.get("freeze_revision") == 5, "Submission claim-set freeze revision must be 5.")
    source_of_truth = claims.get("source_of_truth")
    _require(isinstance(source_of_truth, dict), "source_of_truth must be a mapping.")
    _require(
        source_of_truth.get("provenance_manifest") == "paper/artifacts/provenance_manifest.json"
        and source_of_truth.get("repository_head_at_evidence_freeze") == EXECUTION_SOURCE_COMMIT
        and source_of_truth.get("branch") == CANONICAL_SOURCE_BRANCH,
        "Evidence freeze must point to schema-3 provenance, the clean execution source commit, and the canonical reviewed branch.",
    )
    freeze_policy = claims.get("freeze_policy")
    _require(isinstance(freeze_policy, dict), "freeze_policy must be a mapping.")
    _require(freeze_policy.get("claims_frozen_before_new_confirmatory_outcomes") is True, "Claims must be frozen before new confirmatory outcomes.")
    _require(isinstance(freeze_policy.get("new_confirmatory_outcomes_allowed_now"), bool), "new_confirmatory_outcomes_allowed_now must be boolean.")

    _validate_venue_and_theory(claims)
    _validate_completed_execution(claims)
    historical_by_id, completed_by_id = _provenance_studies(provenance)

    empirical = claims.get("empirical_claims")
    _require(isinstance(empirical, dict), "empirical_claims must be a mapping.")
    _require(empirical.get("submission_primary_claim_id") == "E29", "E29 must be the submission primary empirical claim.")
    _require(empirical.get("submission_primary_endpoints") == CIFAR_PRIMARY_ENDPOINTS, "Submission primary endpoints must match the frozen CIFAR confirmation.")
    studies = empirical.get("studies")
    _require(isinstance(studies, list), "empirical_claims.studies must be a list.")
    study_ids = [study.get("id") for study in studies if isinstance(study, dict)]
    _require(study_ids == ["E19", "E20", "E26", "E27", "E28", "E29"], "Frozen studies must be exactly E19, E20, E26, E27, E28, and E29 in order.")
    studies_by_id = {study["id"]: study for study in studies}
    _require(studies_by_id["E19"].get("status") == "failed" and studies_by_id["E19"].get("broad_hypothesis_accepted") is False, "E19 must remain failed.")
    _require(studies_by_id["E20"].get("status") == "passed_restricted" and studies_by_id["E20"].get("primary_outputs") == CROSSING_ENDPOINTS, "E20 must remain a restricted crossing-event success.")
    for study_id in ("E19", "E20"):
        _validate_historical_study(study_id, studies_by_id[study_id], historical_by_id[study_id])

    _validate_e26(studies_by_id["E26"])
    _validate_e27(studies_by_id["E27"])
    _validate_e28(studies_by_id["E28"])
    _validate_e29(studies_by_id["E29"])
    for study_id in ("E26", "E27", "E28"):
        archive_id = CLAIM_TO_COMPLETED_ARCHIVE[study_id]
        _require(archive_id in completed_by_id, f"{study_id} completed archive is missing from central provenance.")
        _require(
            studies_by_id[study_id].get("evidence") == _expected_completed_evidence(archive_id),
            f"{study_id} evidence does not match its exact schema-3 completed archive contract.",
        )
    _require(
        studies_by_id["E27"]["evidence"] == studies_by_id["E28"]["evidence"],
        "E27 and E28 must map to the same shared expanded archive.",
    )

    limitation = claims.get("historical_source_limitation")
    _require(isinstance(limitation, dict), "historical_source_limitation is mandatory.")
    _require(limitation.get("applies_to_claim_ids") == ["E19", "E20"], "Historical source limitation must apply exactly to E19 and E20.")
    _require(
        limitation.get("exact_dirty_source_bytes_available") is False
        and limitation.get("reconstruction_possible_from_digest") is False
        and limitation.get("clean_checkout_can_audit_compact_outcomes") is True
        and limitation.get("clean_checkout_can_replay_historical_studies") is False,
        "Historical source limitation must preserve unavailable and unreplayable source truth.",
    )
    _require(limitation.get("regeneration_class") == "new_study", "Historical regeneration_class must be exactly new_study.")
    _require(limitation.get("required_interpretation") == CANONICAL_REGENERATION_INTERPRETATION, "Historical regeneration interpretation must use the canonical non-replay wording.")

    secondary = empirical.get("secondary_non_promotable_outputs")
    _require(isinstance(secondary, list) and secondary, "Secondary outputs must be listed.")
    _require(all(isinstance(item, dict) and item.get("promotion_allowed") is False for item in secondary), "Every secondary output must remain non-promotable.")


def _require_evidence_terms(gate: dict[str, Any], terms: tuple[str, ...]) -> None:
    evidence = gate.get("evidence")
    _require(isinstance(evidence, str), f"{gate.get('id')}.evidence must be text.")
    _require(all(term in evidence for term in terms), f"{gate.get('id')} evidence is stale or incomplete.")


def _validate_findings_first_documents(repo_root: Path) -> None:
    expected_digests = {
        "README.md": "6f537939b80026fbd9d008ca28979a77313794fed87ee5700a8a0ceb76abcff7",
        "paper/main.tex": "b1fb927f8af8cf863923a0ba0a44e039a6c0c0b0257e7496efc133671275bdb5",
    }
    for relative_path, expected_digest in expected_digests.items():
        document = repo_root / relative_path
        try:
            actual_digest = hashlib.sha256(document.read_bytes()).hexdigest()
        except OSError as error:
            raise ManifestError(
                f"Findings-first evidence document is unavailable: {relative_path}."
            ) from error
        _require(
            actual_digest == expected_digest,
            f"Findings-first document digest is stale: {relative_path}.",
        )


def validate_readiness(
    readiness: dict[str, Any],
    claims: dict[str, Any],
    *,
    framing_root: Path | None = None,
) -> list[str]:
    _require(readiness.get("schema_version") == "submission-readiness-v1", "Unsupported submission readiness schema.")
    _require(
        readiness.get("decision_reason") == READINESS_DECISION_REASON,
        "Readiness decision_reason is stale or includes an inaccurate blocker summary.",
    )
    _require(set(readiness.get("allowed_statuses", [])) == ALLOWED_READINESS_STATUSES, "Readiness status enumeration is incomplete or changed.")
    gates = readiness.get("gates")
    _require(isinstance(gates, list) and gates, "gates must be a non-empty list.")
    gate_ids = [gate.get("id") for gate in gates if isinstance(gate, dict)]
    _require(gate_ids == list(CANONICAL_GATE_REQUIREMENTS), "Readiness gates were deleted, added, or reordered relative to the canonical schema.")

    gates_by_id: dict[str, dict[str, Any]] = {}
    blocking: list[str] = []
    for index, gate in enumerate(gates):
        _require(isinstance(gate, dict), f"gates[{index}] must be a mapping.")
        gate_id = gate["id"]
        _require(gate_id not in gates_by_id, f"Duplicate readiness gate: {gate_id}.")
        gates_by_id[gate_id] = gate
        status = gate.get("status")
        _require(status in ALLOWED_READINESS_STATUSES, f"Invalid status for {gate_id}: {status!r}.")
        _require(status != "not_applicable", f"Gate {gate_id} cannot be made not_applicable in schema v1.")
        _require(gate.get("required_for_submission") is CANONICAL_GATE_REQUIREMENTS[gate_id], f"{gate_id}.required_for_submission differs from the canonical schema.")
        _require(status == EXPECTED_GATE_STATUSES[gate_id], f"{gate_id}.status differs from the frozen readiness state.")
        if CANONICAL_GATE_REQUIREMENTS[gate_id] and status != "pass":
            blocking.append(gate_id)

    _require(
        claims["venue"]["official_venue_requirements_verified"]
        is (gates_by_id["official_2027_requirements_sourced"]["status"] == "pass"),
        "Claim-set and readiness venue-verification states disagree.",
    )
    _require_evidence_terms(gates_by_id["claims_frozen"], ("revision 5", "E19", "E20", "E26", "E27", "E28", "E29"))
    _require_evidence_terms(
        gates_by_id["double_blind_anonymity_audit"],
        ("cifar-confirmation-anonymous-v1.zip", ANONYMOUS_CIFAR_SUPPLEMENT_SHA256),
    )
    supplement = Path(__file__).resolve().parents[1] / "paper" / "artifacts" / "cifar-confirmation-anonymous-v1.zip"
    try:
        supplement_digest = hashlib.sha256(supplement.read_bytes()).hexdigest()
    except OSError as error:
        raise ManifestError("Anonymous CIFAR supplement is unavailable.") from error
    _require(
        supplement_digest == ANONYMOUS_CIFAR_SUPPLEMENT_SHA256,
        "Anonymous CIFAR supplement digest is stale.",
    )
    _require_evidence_terms(gates_by_id["clean_git_source_release"], ("2798c45", "integrated release base"))
    _require_evidence_terms(gates_by_id["historical_empirical_replayability"], ("E19/E20", "E26--E28", "E29", "full checkpoints remain on DGX"))
    _require_evidence_terms(gates_by_id["compact_archive_integrity_verifier"], ("four historical physical archives", "67 files", "3510433 bytes", "E29 anonymous archive"))
    _require_evidence_terms(gates_by_id["fresh_breadth_preregistration"], (EXECUTION_SOURCE_COMMIT, CIFAR_SOURCE_COMMIT, "passed no-training preflight"))
    fresh_validation = gates_by_id["fresh_breadth_validation"]
    _require(
        fresh_validation.get("criterion")
        == "Fresh untouched cells and seeds execute the frozen Study-A beta-robust endpoint, Study-B comparator-specific joint tradeoff, semi-real generated-cue outcomes, and eight-block CIFAR-10 matched confirmation under preregistered gates, whether claims support, falsify, or restrict promotion.",
        "fresh_breadth_validation criterion must name the actual frozen endpoints.",
    )
    _require_evidence_terms(
        fresh_validation,
        (
            "E26",
            "E27",
            "E28",
            "comparator-restricted mixed",
            BLOOP_COMPARATOR_LABEL,
            PCGRAD_COMPARATOR_LABEL,
            "Both E28 comparators are non-canonical",
            "no broad CDC superiority",
            "E29",
            "8/8 weak-only gates",
            "one artificial cue and one CNN",
        ),
    )
    kernel_reframing = gates_by_id["kernel_certificate_or_empirical_reframing"]
    _require(
        kernel_reframing.get("criterion") == KERNEL_REFRAMING_CRITERION
        and kernel_reframing.get("evidence") == KERNEL_REFRAMING_EVIDENCE,
        "Kernel gate must pass only through the canonical empirical-event-heuristic reframing with no tanh/GRU certificate or global positive-lag predictor claim.",
    )
    _validate_findings_first_documents(
        (framing_root or Path(__file__).resolve().parents[1]).resolve()
    )
    _require_evidence_terms(gates_by_id["sequential_benchmark"], ("E29", "not a real sequential benchmark", "0/32"))
    _require_evidence_terms(gates_by_id["baseline_parity_audit"], ("Bloop-style shadow-target rescue", "PCGrad-style shadow-target rescue", "canonical:false", "broad CDC superiority"))
    _require_evidence_terms(gates_by_id["clean_clone_artifact_reproduction"], ("2798c45", "362 passed", "8 skipped", "16-page"))

    go_no_go = readiness.get("go_no_go")
    _require(isinstance(go_no_go, dict), "go_no_go must be a mapping.")
    outcome = go_no_go.get("new_confirmatory_outcomes")
    _require(isinstance(outcome, dict), "new_confirmatory_outcomes decision is missing.")
    outcome_blocking = [gate_id for gate_id in NEW_OUTCOME_GATE_IDS if gates_by_id[gate_id]["status"] != "pass"]
    expected_outcome_decision = "no_go" if outcome_blocking else "go"
    _require(outcome.get("decision") == expected_outcome_decision, "New-outcome decision does not match its canonical gates.")
    _require(outcome.get("blocking_gate_ids") == outcome_blocking, "New-outcome blocking_gate_ids do not match current gate statuses.")
    _require(
        claims["freeze_policy"]["new_confirmatory_outcomes_allowed_now"] is (expected_outcome_decision == "go"),
        "Claim-set outcome permission disagrees with readiness gates.",
    )
    promotion = go_no_go.get("claim_promotion")
    _require(isinstance(promotion, dict) and promotion.get("decision") == "no_go", "Schema-v1 claim promotion must remain no_go.")
    _require(
        promotion.get("allowed_current_empirical_scope") == PROMOTION_SCOPE,
        "Claim-promotion allowed scope is not canonical.",
    )
    prohibited = promotion.get("prohibited_until_new_evidence")
    _require(
        isinstance(prohibited, list)
        and "architecture ranking" in prohibited
        and "broad CDC superiority or canonical-baseline superiority" in prohibited,
        "Claim-promotion prohibitions must retain architecture-ranking and broad-CDC limits.",
    )

    computed_ready = not blocking
    _require(readiness.get("overall_submission_ready") is computed_ready, "overall_submission_ready does not match the mandatory gates.")
    expected_decision = "go" if computed_ready else "no_go"
    _require(readiness.get("overall_decision") == expected_decision, "overall_decision does not match the mandatory gates.")
    _require(go_no_go.get("submission", {}).get("decision") == expected_decision, "go_no_go.submission.decision does not match the mandatory gates.")
    return blocking


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-set", type=Path, default=root / "research_scope" / "submission_claim_set.yaml")
    parser.add_argument("--readiness", type=Path, default=root / "research_scope" / "submission_readiness.yaml")
    parser.add_argument("--provenance", type=Path, default=root / "paper" / "artifacts" / "provenance_manifest.json")
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Return nonzero unless every mandatory submission gate passes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        claims = _load_mapping(args.claim_set)
        readiness = _load_mapping(args.readiness)
        provenance = _load_mapping(args.provenance)
        validate_claim_set(claims, provenance)
        blocking = validate_readiness(readiness, claims)
    except ManifestError as error:
        print(f"submission readiness validation failed: {error}", file=sys.stderr)
        return 1

    print(
        "submission manifests valid; "
        f"overall_submission_ready={not blocking}; "
        f"blocking_required_gates={len(blocking)}"
    )
    if blocking:
        print("blocking gates: " + ", ".join(blocking))
    if args.require_ready and blocking:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
