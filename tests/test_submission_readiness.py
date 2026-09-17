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


def _study(claims: dict, claim_id: str) -> dict:
    return next(
        study for study in claims["empirical_claims"]["studies"]
        if study["id"] == claim_id
    )


def _gate(readiness: dict, gate_id: str) -> dict:
    return next(gate for gate in readiness["gates"] if gate["id"] == gate_id)


def test_current_claim_and_readiness_manifests_are_consistent_and_not_ready():
    claims, readiness, provenance = _payloads()

    validate_claim_set(claims, provenance)
    blocking = validate_readiness(readiness, claims)

    assert claims["schema_version"] == "submission-claim-set-v2"
    assert claims["freeze_revision"] == 5
    assert provenance["schema_version"] == 3
    assert [
        study["id"] for study in claims["empirical_claims"]["studies"]
    ] == ["E19", "E20", "E26", "E27", "E28", "E29"]
    assert claims["empirical_claims"]["submission_primary_claim_id"] == "E29"
    assert blocking
    assert "clean_git_source_release" in blocking
    assert readiness["overall_submission_ready"] is False
    assert readiness["overall_decision"] == "no_go"
    assert readiness["go_no_go"]["new_confirmatory_outcomes"]["blocking_gate_ids"] == [
        "clean_git_source_release"
    ]


def test_claim_set_rejects_wrong_evidence_freeze_branch():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    mutated["source_of_truth"]["branch"] = "main"

    with pytest.raises(ManifestError, match="canonical reviewed branch"):
        validate_claim_set(mutated, provenance)


def test_claim_set_rejects_stale_findings_first_headline():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    mutated["headline_claim"]["interpretation"] = (
        "A global positive-lag parameter-level predictor is solved."
    )

    with pytest.raises(ManifestError, match="findings-first headline"):
        validate_claim_set(mutated, provenance)


def test_readiness_rejects_stale_decision_reason():
    claims, readiness, _ = _payloads()
    mutated = copy.deepcopy(readiness)
    mutated["decision_reason"] = "The passing full suite is still a blocker."

    with pytest.raises(ManifestError, match="decision_reason is stale"):
        validate_readiness(mutated, claims)


def test_e26_e29_exact_membership_values_and_shared_archive_are_frozen():
    claims, _, _ = _payloads()
    e26 = _study(claims, "E26")
    e27 = _study(claims, "E27")
    e28 = _study(claims, "E28")
    e29 = _study(claims, "E29")

    assert e26["status"] == "failed_negative_indeterminate"
    assert e26["records"] == 64
    assert e26["datasets"]["MNIST"]["records"] == 32
    assert e26["datasets"]["FashionMNIST"]["records"] == 32
    assert e26["datasets"]["MNIST"]["mean_weak_auc_gap"] == {
        "point_estimate": -0.0012362842136667493,
        "ci95": [-0.0034405428466004646, 0.0010159591371180453],
    }
    assert e27["status"] == "failed_negative"
    assert e27["records"] == 192
    assert e28["status"] == "passed_restricted_mixed"
    assert e28["method_records"] == 256
    assert e27["evidence"] == e28["evidence"]
    assert e27["evidence"]["archive_study_id"] == (
        "expanded-nonlinear-beta-cdc-tradeoff-v1"
    )
    assert e27["evidence"]["claim_ids"] == ["E27", "E28"]
    assert e29["status"] == "passed_confirmatory"
    assert e29["weak_only_gate_pass_count"] == 8
    assert e29["primary_causal_certificate_count"] == 8
    assert e29["metrics"]["signed_normalized_deficit_auc"]["mean"] == pytest.approx(
        0.48870154668887456
    )


def test_completed_program_state_and_readiness_states_are_frozen():
    claims, readiness, _ = _payloads()
    execution = claims["completed_externally_sealed_execution"]

    assert execution["empirical_claim_ids_assigned"] is True
    assert execution["outcomes_generated"] is True
    assert execution["execution_authorized"] is True
    assert execution["repository_head_was_clean_at_execution"] is True
    assert execution["execution_source_commit"] == (
        "9ebbc617a9aaad5c3c19e9d5fd4fa08f2d3517d0"
    )
    assert _gate(readiness, "fresh_breadth_preregistration")["status"] == "pass"
    assert _gate(readiness, "fresh_breadth_validation")["status"] == "pass"
    assert _gate(readiness, "sequential_benchmark")["status"] == "fail"
    assert _gate(readiness, "clean_git_source_release")["status"] == "blocked"


def test_readiness_rejects_deleted_mandatory_gate():
    claims, readiness, _ = _payloads()
    readiness["gates"] = [
        gate for gate in readiness["gates"]
        if gate["id"] != "historical_empirical_replayability"
    ]

    with pytest.raises(ManifestError, match="deleted, added, or reordered"):
        validate_readiness(readiness, claims)


def test_readiness_rejects_mandatory_gate_downgraded_to_optional():
    claims, readiness, _ = _payloads()
    mutated = copy.deepcopy(readiness)
    _gate(mutated, "historical_empirical_replayability")[
        "required_for_submission"
    ] = False

    with pytest.raises(ManifestError, match="differs from the canonical schema"):
        validate_readiness(mutated, claims)


def test_claim_evidence_digest_must_match_verified_provenance():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    _study(mutated, "E19")["evidence"]["manifest_sha256"] = "a" * 64

    with pytest.raises(ManifestError, match="does not match central provenance"):
        validate_claim_set(mutated, provenance)


def test_claim_set_requires_schema_3_provenance():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(provenance)
    mutated["schema_version"] = 2

    with pytest.raises(ManifestError, match="schema 3 is required"):
        validate_claim_set(claims, mutated)


def test_claim_set_rejects_missing_e28_membership():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    mutated["empirical_claims"]["studies"] = [
        study for study in mutated["empirical_claims"]["studies"]
        if study["id"] != "E28"
    ]

    with pytest.raises(ManifestError, match="exactly E19, E20, E26, E27, E28, and E29"):
        validate_claim_set(mutated, provenance)


def test_claim_set_rejects_promoted_cifar_scope_or_mutated_result():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    e29 = _study(mutated, "E29")
    e29["scope_limit"] = "This proves universal gradient starvation."

    with pytest.raises(ManifestError, match="E29 CIFAR confirmatory"):
        validate_claim_set(mutated, provenance)


def test_provenance_rejects_detaching_e27_e28_shared_archive_membership():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(provenance)
    shared = next(
        archive for archive in mutated["completed_run_archives"]
        if archive["study_id"] == "expanded-nonlinear-beta-cdc-tradeoff-v1"
    )
    shared["claim_ids"] = ["E27"]

    with pytest.raises(ManifestError, match=r"claim_ids is not canonical"):
        validate_claim_set(claims, mutated)


def test_claim_set_rejects_detaching_e28_from_shared_archive():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    _study(mutated, "E28")["evidence"]["archive_study_id"] = (
        "semi-real-generated-cue-v1"
    )

    with pytest.raises(ManifestError, match="E28 evidence does not match"):
        validate_claim_set(mutated, provenance)


def test_claim_set_rejects_e26_negative_verdict_promotion():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    _study(mutated, "E26")["overall_hypothesis_accepted"] = True

    with pytest.raises(ManifestError, match="E26 negative/indeterminate verdict"):
        validate_claim_set(mutated, provenance)


def test_claim_set_rejects_e27_negative_verdict_promotion():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    _study(mutated, "E27")["overall_beta_robust_claim"] = True

    with pytest.raises(ManifestError, match="E27 negative beta-robust verdict"):
        validate_claim_set(mutated, provenance)


def test_claim_set_rejects_e28_pcgrad_failure_promotion():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    e28 = _study(mutated, "E28")
    pcgrad = next(
        comparator for comparator in e28["comparators"]
        if comparator["id"] == "pcgrad"
    )
    pcgrad["cdc_beats_comparator_joint_tradeoff"] = True

    with pytest.raises(ManifestError, match="E28 mixed comparator verdict"):
        validate_claim_set(mutated, provenance)


@pytest.mark.parametrize("comparator_id", ["bloop", "pcgrad"])
def test_claim_set_requires_noncanonical_comparator_labels(comparator_id: str):
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    e28 = _study(mutated, "E28")
    comparator = next(
        item for item in e28["comparators"] if item["id"] == comparator_id
    )
    comparator["canonical"] = True

    with pytest.raises(ManifestError, match="canonical:false labels"):
        validate_claim_set(mutated, provenance)


@pytest.mark.parametrize(
    ("comparator_id", "short_label"),
    [("bloop", "Bloop-style"), ("pcgrad", "PCGrad-style")],
)
def test_claim_set_rejects_shortened_comparator_labels(
    comparator_id: str, short_label: str
):
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    e28 = _study(mutated, "E28")
    comparator = next(
        item for item in e28["comparators"] if item["id"] == comparator_id
    )
    comparator["label"] = short_label

    with pytest.raises(ManifestError, match="exact full-label scope"):
        validate_claim_set(mutated, provenance)


def test_readiness_rejects_global_predictor_promotion_in_kernel_gate():
    claims, readiness, _ = _payloads()
    mutated = copy.deepcopy(readiness)
    _gate(mutated, "kernel_certificate_or_empirical_reframing")["evidence"] = (
        "A global positive-lag parameter-level predictor is established."
    )

    with pytest.raises(ManifestError, match="canonical empirical-event-heuristic"):
        validate_readiness(mutated, claims)


@pytest.mark.parametrize("relative_path", ["README.md", "paper/main.tex"])
def test_readiness_rejects_drifted_findings_first_document(
    tmp_path: Path, relative_path: str
):
    claims, readiness, _ = _payloads()
    framing_root = tmp_path / "framing"
    for source_relative_path in ("README.md", "paper/main.tex"):
        destination = framing_root / source_relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / source_relative_path).read_bytes())
    drifted_document = framing_root / relative_path
    drifted_document.write_bytes(
        drifted_document.read_bytes()
        + b"\nExact trained-network kernel dynamics are now claimed.\n"
    )

    with pytest.raises(ManifestError, match="Findings-first document digest is stale"):
        validate_readiness(readiness, claims, framing_root=framing_root)


def test_claim_set_rejects_stale_unexecuted_state():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    mutated["completed_externally_sealed_execution"]["outcomes_generated"] = False

    with pytest.raises(ManifestError, match="Completed execution must retain"):
        validate_claim_set(mutated, provenance)


def test_claim_set_rejects_dirty_completed_source():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    _study(mutated, "E26")["evidence"]["git_dirty"] = True

    with pytest.raises(ManifestError, match="E26 evidence does not match"):
        validate_claim_set(mutated, provenance)


def test_claim_set_cannot_erase_historical_source_limitation():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    mutated["historical_source_limitation"][
        "reconstruction_possible_from_digest"
    ] = True

    with pytest.raises(ManifestError, match="unavailable and unreplayable"):
        validate_claim_set(mutated, provenance)


def test_claim_set_cannot_extend_historical_limitation_to_completed_claims():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    mutated["historical_source_limitation"]["applies_to_claim_ids"].append("E26")

    with pytest.raises(ManifestError, match="exactly to E19 and E20"):
        validate_claim_set(mutated, provenance)


def test_new_outcome_blocker_list_is_derived_from_canonical_gates():
    claims, readiness, _ = _payloads()
    mutated = copy.deepcopy(readiness)
    mutated["go_no_go"]["new_confirmatory_outcomes"]["blocking_gate_ids"] = []

    with pytest.raises(ManifestError, match="blocking_gate_ids"):
        validate_readiness(mutated, claims)


def test_readiness_rejects_fresh_validation_status_tampering():
    claims, readiness, _ = _payloads()
    mutated = copy.deepcopy(readiness)
    _gate(mutated, "fresh_breadth_validation")["status"] = "blocked"

    with pytest.raises(ManifestError, match="status differs from the frozen"):
        validate_readiness(mutated, claims)


def test_claim_set_rejects_negated_new_study_interpretation():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    mutated["historical_source_limitation"]["required_interpretation"] = (
        "Regeneration is not a new study and may be represented as a replay."
    )

    with pytest.raises(ManifestError, match="canonical non-replay wording"):
        validate_claim_set(mutated, provenance)


def test_claim_set_rejects_e19_historical_result_tampering():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    e19 = _study(mutated, "E19")
    e19["correct_counts"]["weak_only_learnability"] = "16/16"
    e19["failure"] = "rewritten historical result"

    with pytest.raises(ManifestError, match="unchanged historical result contract"):
        validate_claim_set(mutated, provenance)


def test_claim_set_rejects_e20_historical_result_tampering():
    claims, _, provenance = _payloads()
    mutated = copy.deepcopy(claims)
    e20 = _study(mutated, "E20")
    e20["restricted_hypothesis_accepted"] = False
    e20["correct_counts"]["response_crossing"] = "0/64"

    with pytest.raises(ManifestError, match="unchanged historical result contract"):
        validate_claim_set(mutated, provenance)
