from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.check_submission_readiness import (
    ManifestError,
    validate_claim_set,
    validate_readiness,
)


ROOT = Path(__file__).resolve().parents[1]


def _payloads() -> tuple[dict, dict, dict]:
    claims = json.loads(
        (ROOT / "research_scope/submission_claim_set.yaml").read_text(encoding="utf-8")
    )
    readiness = json.loads(
        (ROOT / "research_scope/submission_readiness.yaml").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (ROOT / "paper/artifacts/provenance_manifest.json").read_text(encoding="utf-8")
    )
    return claims, readiness, provenance


def test_current_claim_and_readiness_manifests_are_consistent_and_not_ready():
    claims, readiness, provenance = _payloads()

    validate_claim_set(claims, provenance)
    blocking = validate_readiness(readiness, claims)

    assert blocking
    assert "clean_git_source_release" in blocking
    assert readiness["overall_submission_ready"] is False


def test_readiness_rejects_deleted_mandatory_gate():
    claims, readiness, _ = _payloads()
    readiness["gates"] = [
        gate for gate in readiness["gates"] if gate["id"] != "historical_empirical_replayability"
    ]

    with pytest.raises(ManifestError, match="deleted, added, or reordered"):
        validate_readiness(readiness, claims)


def test_readiness_rejects_mandatory_gate_downgraded_to_optional():
    claims, readiness, _ = _payloads()
    mutated = copy.deepcopy(readiness)
    gate = next(
        gate
        for gate in mutated["gates"]
        if gate["id"] == "historical_empirical_replayability"
    )
    gate["required_for_submission"] = False

    with pytest.raises(ManifestError, match="differs from the canonical schema"):
        validate_readiness(mutated, claims)


def test_claim_evidence_digest_must_match_verified_provenance():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    e19 = next(study for study in mutated["empirical_claims"]["studies"] if study["id"] == "E19")
    e19["evidence"]["manifest_sha256"] = "a" * 64

    with pytest.raises(ManifestError, match="does not match central provenance"):
        validate_claim_set(mutated, provenance)


def test_claim_set_cannot_erase_historical_source_limitation():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    mutated["historical_source_limitation"]["reconstruction_possible_from_digest"] = True

    with pytest.raises(ManifestError, match="unavailable and unreplayable"):
        validate_claim_set(mutated, provenance)


def test_new_outcome_blocker_list_is_derived_from_canonical_gates():
    claims, readiness, _ = _payloads()
    mutated = copy.deepcopy(readiness)
    mutated["go_no_go"]["new_confirmatory_outcomes"]["blocking_gate_ids"] = []

    with pytest.raises(ManifestError, match="blocking_gate_ids"):
        validate_readiness(mutated, claims)


def test_claim_set_rejects_negated_new_study_interpretation():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    mutated["historical_source_limitation"]["required_interpretation"] = (
        "Regeneration is not a new study and may be represented as a replay."
    )

    with pytest.raises(ManifestError, match="canonical non-replay wording"):
        validate_claim_set(mutated, provenance)