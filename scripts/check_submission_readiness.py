#!/usr/bin/env python3
"""Validate frozen claims, provenance links, and fail-closed readiness gates.

The ``.yaml`` inputs intentionally use JSON syntax, a YAML 1.2 subset. The
validator therefore runs with only the Python standard library before project
dependencies are installed. A valid but not-ready state exits zero;
``--require-ready`` makes any non-passing mandatory gate an error.
"""

from __future__ import annotations

import argparse
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
INCLUDED_THEORY_IDS = [
    *[f"T{index}" for index in range(1, 17)],
    "T19",
    "T20",
]
BLOCKED_THEORY_IDS = ["T17", "T18"]
PRIMARY_ENDPOINTS = ["drift_crossing", "response_crossing"]
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

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
NEW_OUTCOME_GATE_IDS = [
    "clean_git_source_release",
    "exact_environment_lock",
    "fresh_breadth_preregistration",
]
CLAIM_ARCHIVE_ROOTS = {
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
        "accepted_primary_outputs": PRIMARY_ENDPOINTS,
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
CANONICAL_REGENERATION_INTERPRETATION = (
    "Regeneration under current or future source is a new study, not a replay "
    "of either historical study."
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


def _provenance_studies(provenance: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _require(provenance.get("schema_version") == 2, "Unsupported provenance schema.")
    studies = provenance.get("frozen_empirical_ntk_studies")
    _require(isinstance(studies, list), "Provenance studies must be a list.")
    studies_by_id: dict[str, dict[str, Any]] = {}
    for index, study in enumerate(studies):
        _require(isinstance(study, dict), f"Provenance study {index} must be a mapping.")
        claim_id = study.get("claim_id")
        _require(claim_id in CLAIM_ARCHIVE_ROOTS, f"Unexpected provenance claim_id: {claim_id!r}.")
        _require(claim_id not in studies_by_id, f"Duplicate provenance claim_id: {claim_id}.")
        studies_by_id[claim_id] = study
    _require(
        set(studies_by_id) == set(CLAIM_ARCHIVE_ROOTS),
        "Provenance must contain exactly E19 and E20.",
    )
    return studies_by_id


def _archive_manifest_digest(study: dict[str, Any], archive_root: str) -> str:
    entries = study.get("archived_files")
    _require(isinstance(entries, list), "Provenance archived_files must be a list.")
    target = f"{archive_root}/archive_manifest.json"
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("path") == target
    ]
    _require(len(matches) == 1, f"Provenance must index {target} exactly once.")
    return _require_sha256(matches[0].get("sha256"), f"{target}.sha256")


def validate_claim_set(
    claims: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    _require(
        claims.get("schema_version") == "submission-claim-set-v1",
        "Unsupported submission claim-set schema.",
    )
    freeze_policy = claims.get("freeze_policy")
    _require(isinstance(freeze_policy, dict), "freeze_policy must be a mapping.")
    _require(
        freeze_policy.get("claims_frozen_before_new_confirmatory_outcomes") is True,
        "Claims must be frozen before new confirmatory outcomes.",
    )
    _require(
        isinstance(freeze_policy.get("new_confirmatory_outcomes_allowed_now"), bool),
        "new_confirmatory_outcomes_allowed_now must be boolean.",
    )

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
            isinstance(url, str)
            and url.startswith("https://iclr.cc/Conferences/2027/"),
            f"official_sources[{index}] must be an official ICLR 2027 HTTPS URL.",
        )

    headline = claims.get("headline_claim")
    _require(isinstance(headline, dict), "headline_claim must be a mapping.")
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
    theory_by_id = {
        entry.get("id"): entry for entry in theory_claims if isinstance(entry, dict)
    }
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

    prospective = claims.get("prospective_designs")
    _require(isinstance(prospective, dict), "prospective_designs must be a mapping.")
    _require(
        prospective.get("empirical_claim_ids_assigned") is False
        and prospective.get("outcomes_generated") is False
        and prospective.get("execution_authorized") is False,
        "Prospective designs must remain outcome-free, unauthorized, and outside E-claims.",
    )
    designs = prospective.get("designs")
    _require(isinstance(designs, list), "prospective_designs.designs must be a list.")
    prospective_by_id = {
        design.get("id"): design for design in designs if isinstance(design, dict)
    }
    _require(
        set(prospective_by_id) == {"semi_real_generated_cue", "expanded_nonlinear_cdc"},
        "The prospective design set must contain exactly the two frozen blocked studies.",
    )
    _require(
        all(
            design.get("status") == "blocked_unexecuted"
            and design.get("outcome_records") == 0
            for design in prospective_by_id.values()
        ),
        "Every prospective design must remain blocked, unexecuted, and outcome-free.",
    )

    empirical = claims.get("empirical_claims")
    _require(isinstance(empirical, dict), "empirical_claims must be a mapping.")
    _require(
        empirical.get("submission_primary_endpoints") == PRIMARY_ENDPOINTS,
        "Submission primary endpoints must be exactly drift_crossing and response_crossing.",
    )
    studies = empirical.get("studies")
    _require(isinstance(studies, list) and len(studies) == 2, "Exactly E19 and E20 must be frozen.")
    studies_by_id = {
        study.get("id"): study for study in studies if isinstance(study, dict)
    }
    _require(set(studies_by_id) == {"E19", "E20"}, "Frozen studies must be E19 and E20.")
    _require(studies_by_id["E19"].get("status") == "failed", "E19 must remain failed.")
    _require(
        studies_by_id["E19"].get("broad_hypothesis_accepted") is False,
        "E19 broad acceptance must be false.",
    )
    _require(
        studies_by_id["E20"].get("status") == "passed_restricted",
        "E20 must remain a restricted success.",
    )
    _require(
        studies_by_id["E20"].get("primary_outputs") == PRIMARY_ENDPOINTS,
        "E20 primary outputs must be exactly the two crossing events.",
    )

    provenance_by_id = _provenance_studies(provenance)
    for study_id, study in studies_by_id.items():
        evidence = study.get("evidence")
        _require(isinstance(evidence, dict), f"{study_id}.evidence must be a mapping.")
        central = provenance_by_id[study_id]
        archive_root = CLAIM_ARCHIVE_ROOTS[study_id]
        _require(
            evidence.get("compact_archive_root") == archive_root
            == central.get("compact_archive_root"),
            f"{study_id} compact archive root is detached from provenance.",
        )
        for field in ("manifest_sha256", "prediction_sha256", "source_sha256"):
            claim_digest = _require_sha256(evidence.get(field), f"{study_id}.evidence.{field}")
            central_digest = _require_sha256(central.get(field), f"provenance.{study_id}.{field}")
            _require(
                claim_digest == central_digest,
                f"{study_id}.{field} does not match central provenance.",
            )
        archive_manifest_sha256 = _require_sha256(
            evidence.get("archive_manifest_sha256"),
            f"{study_id}.evidence.archive_manifest_sha256",
        )
        _require(
            archive_manifest_sha256 == _archive_manifest_digest(central, archive_root),
            f"{study_id} archive-manifest digest does not match central provenance.",
        )
        _require(
            evidence.get("verified_frozen_file_count")
            == central.get("verified_frozen_file_count"),
            f"{study_id} verified-file count does not match central provenance.",
        )
        for field in (
            "source_provenance_class",
            "source_snapshot_archived",
            "source_reconstructible",
        ):
            _require(
                evidence.get(field) == central.get(field),
                f"{study_id}.{field} does not match central provenance.",
            )
        _require(
            evidence.get("source_provenance_class")
            == "historical_dirty_unreconstructible"
            and evidence.get("source_snapshot_archived") is False
            and evidence.get("source_reconstructible") is False,
            f"{study_id} must disclose unavailable, unreconstructible dirty source.",
        )
        frozen_acceptance = FROZEN_ACCEPTANCE_CONTRACTS[study_id]
        _require(
            study.get("records")
            == frozen_acceptance["acceptance_contract"]["total_records"],
            f"{study_id} record count differs from the frozen acceptance contract.",
        )
        _require(
            study.get("primary_outputs")
            == frozen_acceptance["accepted_primary_outputs"],
            f"{study_id} primary outputs differ from the frozen acceptance contract.",
        )
        _require(
            study.get("acceptance_contract")
            == frozen_acceptance["acceptance_contract"],
            f"{study_id} claim acceptance contract is not canonical.",
        )
        _require(
            central.get("accepted_primary_outputs")
            == frozen_acceptance["accepted_primary_outputs"],
            f"{study_id} primary outputs do not match canonical central provenance.",
        )
        _require(
            central.get("acceptance_contract")
            == frozen_acceptance["acceptance_contract"],
            f"{study_id} central acceptance contract is not canonical.",
        )
        expected_passed = study_id == "E20"
        _require(
            central.get("evaluation_passed") is expected_passed,
            f"{study_id} evaluation pass/fail state does not match the frozen claim.",
        )

    limitation = claims.get("historical_source_limitation")
    _require(isinstance(limitation, dict), "historical_source_limitation is mandatory.")
    _require(
        limitation.get("applies_to_claim_ids") == ["E19", "E20"],
        "Historical source limitation must apply exactly to E19 and E20.",
    )
    _require(
        limitation.get("exact_dirty_source_bytes_available") is False
        and limitation.get("reconstruction_possible_from_digest") is False
        and limitation.get("clean_checkout_can_audit_compact_outcomes") is True
        and limitation.get("clean_checkout_can_replay_historical_studies") is False,
        "Historical source limitation must preserve unavailable and unreplayable source truth.",
    )
    _require(
        limitation.get("regeneration_class") == "new_study",
        "Historical regeneration_class must be exactly new_study.",
    )
    _require(
        limitation.get("required_interpretation")
        == CANONICAL_REGENERATION_INTERPRETATION,
        "Historical regeneration interpretation must use the canonical non-replay wording.",
    )

    secondary = empirical.get("secondary_non_promotable_outputs")
    _require(isinstance(secondary, list) and secondary, "Secondary outputs must be listed.")
    _require(
        all(
            isinstance(item, dict) and item.get("promotion_allowed") is False
            for item in secondary
        ),
        "Every secondary output must remain non-promotable.",
    )


def validate_readiness(readiness: dict[str, Any], claims: dict[str, Any]) -> list[str]:
    _require(
        readiness.get("schema_version") == "submission-readiness-v1",
        "Unsupported submission readiness schema.",
    )
    _require(
        set(readiness.get("allowed_statuses", [])) == ALLOWED_READINESS_STATUSES,
        "Readiness status enumeration is incomplete or changed.",
    )
    gates = readiness.get("gates")
    _require(isinstance(gates, list) and gates, "gates must be a non-empty list.")
    gate_ids = [gate.get("id") for gate in gates if isinstance(gate, dict)]
    _require(
        gate_ids == list(CANONICAL_GATE_REQUIREMENTS),
        "Readiness gates were deleted, added, or reordered relative to the canonical schema.",
    )

    gates_by_id: dict[str, dict[str, Any]] = {}
    blocking: list[str] = []
    for index, gate in enumerate(gates):
        _require(isinstance(gate, dict), f"gates[{index}] must be a mapping.")
        gate_id = gate["id"]
        _require(gate_id not in gates_by_id, f"Duplicate readiness gate: {gate_id}.")
        gates_by_id[gate_id] = gate
        status = gate.get("status")
        _require(
            status in ALLOWED_READINESS_STATUSES,
            f"Invalid status for {gate_id}: {status!r}.",
        )
        _require(
            status != "not_applicable",
            f"Gate {gate_id} cannot be made not_applicable in schema v1.",
        )
        expected_required = CANONICAL_GATE_REQUIREMENTS[gate_id]
        _require(
            gate.get("required_for_submission") is expected_required,
            f"{gate_id}.required_for_submission differs from the canonical schema.",
        )
        if expected_required and status != "pass":
            blocking.append(gate_id)

    venue_gate_passed = (
        gates_by_id["official_2027_requirements_sourced"]["status"] == "pass"
    )
    _require(
        claims["venue"]["official_venue_requirements_verified"] is venue_gate_passed,
        "Claim-set and readiness venue-verification states disagree.",
    )

    go_no_go = readiness.get("go_no_go")
    _require(isinstance(go_no_go, dict), "go_no_go must be a mapping.")
    outcome = go_no_go.get("new_confirmatory_outcomes")
    _require(isinstance(outcome, dict), "new_confirmatory_outcomes decision is missing.")
    outcome_blocking = [
        gate_id
        for gate_id in NEW_OUTCOME_GATE_IDS
        if gates_by_id[gate_id]["status"] != "pass"
    ]
    expected_outcome_decision = "no_go" if outcome_blocking else "go"
    _require(
        outcome.get("decision") == expected_outcome_decision,
        "New-outcome decision does not match its canonical gates.",
    )
    _require(
        outcome.get("blocking_gate_ids") == outcome_blocking,
        "New-outcome blocking_gate_ids do not match current gate statuses.",
    )
    _require(
        claims["freeze_policy"]["new_confirmatory_outcomes_allowed_now"]
        is (expected_outcome_decision == "go"),
        "Claim-set outcome permission disagrees with readiness gates.",
    )
    _require(
        go_no_go.get("claim_promotion", {}).get("decision") == "no_go",
        "Schema-v1 claim promotion must remain no_go.",
    )

    computed_ready = not blocking
    _require(
        readiness.get("overall_submission_ready") is computed_ready,
        "overall_submission_ready does not match the mandatory gates.",
    )
    expected_decision = "go" if computed_ready else "no_go"
    _require(
        readiness.get("overall_decision") == expected_decision,
        "overall_decision does not match the mandatory gates.",
    )
    submission_decision = go_no_go.get("submission", {}).get("decision")
    _require(
        submission_decision == expected_decision,
        "go_no_go.submission.decision does not match the mandatory gates.",
    )
    return blocking


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claim-set",
        type=Path,
        default=root / "research_scope" / "submission_claim_set.yaml",
    )
    parser.add_argument(
        "--readiness",
        type=Path,
        default=root / "research_scope" / "submission_readiness.yaml",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=root / "paper" / "artifacts" / "provenance_manifest.json",
    )
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
