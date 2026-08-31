from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from verify_release_artifacts import (
    GIT_EXECUTABLE,
    VerificationError,
    git_executable_source_fingerprint,
    validate_source_provenance,
    verify_archive_manifest,
    verify_completed_run_archive,
    verify_release_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_ARCHIVE_ROOTS = (
    "paper/artifacts/enl_ntk_pilot-20260825",
    "paper/artifacts/enl_ntk_crossing_factorial-20260825",
)
COMPLETED_ARCHIVES = {
    "semi-real-generated-cue-v1": "paper/artifacts/semi-real-generated-cue-v1-20260830",
    "expanded-nonlinear-beta-cdc-tradeoff-v1": "paper/artifacts/expanded-studies-v1-20260830",
}
ARCHIVE_ROOTS = (*HISTORICAL_ARCHIVE_ROOTS, *COMPLETED_ARCHIVES.values())


def _copy_release_tree(tmp_path: Path) -> Path:
    release_root = tmp_path / "release"
    artifacts_root = release_root / "paper" / "artifacts"
    artifacts_root.mkdir(parents=True)
    shutil.copy2(
        ROOT / "paper" / "artifacts" / "provenance_manifest.json",
        artifacts_root / "provenance_manifest.json",
    )
    for archive_root in ARCHIVE_ROOTS:
        shutil.copytree(ROOT / archive_root, release_root / archive_root)
    return release_root


def _rewrite_archive_member_signature(
    release_root: Path,
    archive_root: str,
    relative_path: str,
) -> None:
    manifest_path = release_root / archive_root / "archive_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    member = release_root / archive_root / relative_path
    matches = [
        entry
        for entry in manifest["archived_files"]
        if entry["archive_relative_path"] == relative_path
    ]
    assert len(matches) == 1
    matches[0]["sha256"] = hashlib.sha256(member.read_bytes()).hexdigest()
    matches[0]["bytes"] = member.stat().st_size
    manifest["archived_file_bytes"] = sum(
        entry["bytes"] for entry in manifest["archived_files"]
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _rewrite_completed_artifact_signature(
    release_root: Path,
    archive_root: str,
    relative_path: str,
) -> None:
    archive_dir = release_root / archive_root
    artifact_manifest_path = archive_dir / "artifact_manifest.json"
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    member = archive_dir / relative_path
    matches = [entry for entry in artifact_manifest["files"] if entry["path"] == relative_path]
    assert len(matches) == 1
    matches[0]["sha256"] = hashlib.sha256(member.read_bytes()).hexdigest()
    matches[0]["size"] = member.stat().st_size
    artifact_manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2) + "\n", encoding="utf-8"
    )
    artifact_sha256 = hashlib.sha256(artifact_manifest_path.read_bytes()).hexdigest()
    (archive_dir / "artifact_manifest.sha256").write_text(
        f"{artifact_sha256}  artifact_manifest.json\n", encoding="utf-8"
    )
    for member_path in (relative_path, "artifact_manifest.json", "artifact_manifest.sha256"):
        _rewrite_archive_member_signature(release_root, archive_root, member_path)
    archive_manifest_path = archive_dir / "archive_manifest.json"
    archive_manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
    archive_manifest["full_run_commitment"]["artifact_manifest_sha256"] = artifact_sha256
    archive_manifest["full_run_commitment"]["artifact_manifest_bytes"] = artifact_manifest_path.stat().st_size
    archive_manifest_path.write_text(
        json.dumps(archive_manifest, indent=2) + "\n", encoding="utf-8"
    )


def test_real_compact_archives_verify_without_project_dependencies():
    report = verify_release_artifacts(ROOT)

    assert report.study_count == 5
    assert report.archive_count == 4
    assert report.file_count == 67
    assert report.byte_count > 0
    assert report.historical_unreconstructible_study_count == 2
    assert report.git_tracking_checked is False


def test_release_verifier_rejects_stale_central_interpretation(tmp_path):
    release_root = _copy_release_tree(tmp_path)
    provenance_path = release_root / "paper/artifacts/provenance_manifest.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["interpretation"] = (
        "Only the historical theorem-aligned source is reconstructible."
    )
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(
        VerificationError, match="Central provenance interpretation is stale"
    ):
        verify_release_artifacts(release_root, source_repo_root=ROOT)


def test_release_verifier_rejects_tampered_archived_bytes(tmp_path):
    release_root = _copy_release_tree(tmp_path)
    scores = (
        release_root
        / "paper/artifacts/enl_ntk_pilot-20260825/evaluation/scores.csv"
    )
    scores.write_bytes(scores.read_bytes() + b"\n")

    with pytest.raises(VerificationError, match="hash/size mismatch"):
        verify_release_artifacts(release_root)


def test_release_verifier_rejects_tampered_completed_archive_member(tmp_path):
    release_root = _copy_release_tree(tmp_path)
    member = release_root / COMPLETED_ARCHIVES["expanded-nonlinear-beta-cdc-tradeoff-v1"] / "study_b_claims.json"
    member.write_bytes(member.read_bytes() + b"\n")

    with pytest.raises(VerificationError, match="hash/size mismatch"):
        verify_release_artifacts(release_root, source_repo_root=ROOT)


@pytest.mark.parametrize(
    ("study_index", "claim_ids"),
    [
        (0, ["E29"]),
        (1, ["E27", "E27"]),
    ],
)
def test_release_verifier_rejects_completed_claim_membership_and_duplicates(
    tmp_path, study_index, claim_ids
):
    release_root = _copy_release_tree(tmp_path)
    provenance_path = release_root / "paper/artifacts/provenance_manifest.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["completed_run_archives"][study_index]["claim_ids"] = claim_ids
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(VerificationError, match="claim membership"):
        verify_release_artifacts(release_root, source_repo_root=ROOT)


def test_completed_archive_rejects_omitted_count_tampering(tmp_path):
    release_root = _copy_release_tree(tmp_path)
    study_id = "semi-real-generated-cue-v1"
    archive_root = COMPLETED_ARCHIVES[study_id]
    manifest_path = release_root / archive_root / "archive_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["full_run_commitment"]["omitted_file_count"] = 191
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(VerificationError, match="commitment is not exact"):
        verify_completed_run_archive(
            release_root,
            archive_root,
            expected_study_id=study_id,
            source_repo_root=ROOT,
        )


def test_completed_archive_rejects_coherently_rehashed_semi_result(tmp_path):
    release_root = _copy_release_tree(tmp_path)
    study_id = "semi-real-generated-cue-v1"
    archive_root = COMPLETED_ARCHIVES[study_id]
    acceptance_path = release_root / archive_root / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    acceptance["overall_passed"] = True
    acceptance_path.write_text(json.dumps(acceptance, indent=2) + "\n", encoding="utf-8")
    _rewrite_completed_artifact_signature(release_root, archive_root, "acceptance.json")

    with pytest.raises(VerificationError, match="overall result changed"):
        verify_completed_run_archive(
            release_root,
            archive_root,
            expected_study_id=study_id,
            source_repo_root=ROOT,
        )


def test_completed_archive_rejects_coherently_rehashed_expanded_result(tmp_path):
    release_root = _copy_release_tree(tmp_path)
    study_id = "expanded-nonlinear-beta-cdc-tradeoff-v1"
    archive_root = COMPLETED_ARCHIVES[study_id]
    inference_path = release_root / archive_root / "study_a_inference.json"
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    inference["architectures"]["tanh"]["equal_weight_three_cell_macro"]["point_estimate"] = 1.0
    inference_path.write_text(json.dumps(inference, indent=2) + "\n", encoding="utf-8")
    _rewrite_completed_artifact_signature(
        release_root, archive_root, "study_a_inference.json"
    )

    with pytest.raises(VerificationError, match="frozen estimate changed"):
        verify_completed_run_archive(
            release_root,
            archive_root,
            expected_study_id=study_id,
            source_repo_root=ROOT,
        )


def test_completed_archive_rejects_unbound_trajectory_or_final_proof(tmp_path):
    release_root = _copy_release_tree(tmp_path)
    study_id = "expanded-nonlinear-beta-cdc-tradeoff-v1"
    archive_root = COMPLETED_ARCHIVES[study_id]
    inference_path = release_root / archive_root / "study_b_inference.json"
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    inference["test_families"]["trajectory_superiority"][0]["observed"] = -999.0
    inference_path.write_text(json.dumps(inference, indent=2) + "\n", encoding="utf-8")
    _rewrite_completed_artifact_signature(
        release_root, archive_root, "study_b_inference.json"
    )

    with pytest.raises(VerificationError, match="not bound to its frozen estimand"):
        verify_completed_run_archive(
            release_root,
            archive_root,
            expected_study_id=study_id,
            source_repo_root=ROOT,
        )


def test_completed_archive_rejects_incomplete_cost_rows(tmp_path):
    release_root = _copy_release_tree(tmp_path)
    study_id = "expanded-nonlinear-beta-cdc-tradeoff-v1"
    archive_root = COMPLETED_ARCHIVES[study_id]
    cost_path = release_root / archive_root / "study_b_cost.csv"
    lines = cost_path.read_text(encoding="utf-8").splitlines()
    first_row = lines[1].split(",")
    lines[1] = ",".join([first_row[0], *("" for _ in first_row[1:])])
    cost_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _rewrite_completed_artifact_signature(
        release_root, archive_root, "study_b_cost.csv"
    )

    with pytest.raises(VerificationError, match="row 0 is incomplete"):
        verify_completed_run_archive(
            release_root,
            archive_root,
            expected_study_id=study_id,
            source_repo_root=ROOT,
        )


def test_completed_archive_rejects_unknown_authorization_schema(tmp_path):
    release_root = _copy_release_tree(tmp_path)
    study_id = "expanded-nonlinear-beta-cdc-tradeoff-v1"
    archive_root = COMPLETED_ARCHIVES[study_id]
    archive_dir = release_root / archive_root
    authorization_path = archive_dir / "authorization.seal.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["schema_version"] = "unknown-authorization-v999"
    authorization_path.write_text(
        json.dumps(authorization, indent=2) + "\n", encoding="utf-8"
    )
    authorization_sha = hashlib.sha256(authorization_path.read_bytes()).hexdigest()
    provenance_path = archive_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["authorization_seal_sha256"] = authorization_sha
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    _rewrite_completed_artifact_signature(
        release_root, archive_root, "authorization.seal.json"
    )
    _rewrite_completed_artifact_signature(
        release_root, archive_root, "provenance.json"
    )

    with pytest.raises(VerificationError, match="authorization-seal schema"):
        verify_completed_run_archive(
            release_root,
            archive_root,
            expected_study_id=study_id,
            source_repo_root=ROOT,
        )


def test_archive_verifier_rejects_path_traversal(tmp_path):
    release_root = _copy_release_tree(tmp_path)
    archive_root = ARCHIVE_ROOTS[0]
    manifest_path = release_root / archive_root / "archive_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["archived_files"][0]["archive_relative_path"] = "../escape.yaml"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(VerificationError, match="Unsafe"):
        verify_archive_manifest(release_root, archive_root)


def test_archive_verifier_rejects_duplicate_member_paths(tmp_path):
    release_root = _copy_release_tree(tmp_path)
    archive_root = ARCHIVE_ROOTS[0]
    manifest_path = release_root / archive_root / "archive_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    duplicate = manifest["archived_files"][0]["archive_relative_path"]
    manifest["archived_files"][1]["archive_relative_path"] = duplicate
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(VerificationError, match="Duplicate archive member path"):
        verify_archive_manifest(release_root, archive_root)


def test_archive_verifier_derives_pass_from_all_acceptance_checks(tmp_path):
    release_root = _copy_release_tree(tmp_path)
    archive_root = ARCHIVE_ROOTS[0]
    acceptance_path = release_root / archive_root / "evaluation/evaluation_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    assert acceptance["passed"] is False
    assert not all(acceptance["checks"].values())
    acceptance["passed"] = True
    acceptance_path.write_text(json.dumps(acceptance, indent=2) + "\n", encoding="utf-8")
    _rewrite_archive_member_signature(
        release_root, archive_root, "evaluation/evaluation_acceptance.json"
    )

    with pytest.raises(VerificationError, match="does not equal all declared checks"):
        verify_archive_manifest(release_root, archive_root)


def test_archive_verifier_rejects_broadened_primary_outputs(tmp_path):
    release_root = _copy_release_tree(tmp_path)
    archive_root = ARCHIVE_ROOTS[1]
    acceptance_path = release_root / archive_root / "evaluation/evaluation_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    acceptance["accepted_primary_outputs"].append("causal_certificate")
    acceptance_path.write_text(json.dumps(acceptance, indent=2) + "\n", encoding="utf-8")
    _rewrite_archive_member_signature(
        release_root, archive_root, "evaluation/evaluation_acceptance.json"
    )

    with pytest.raises(VerificationError, match="disagree with central provenance"):
        verify_archive_manifest(
            release_root,
            archive_root,
            expected_primary_outputs=("drift_crossing", "response_crossing"),
            expected_evaluation_passed=True,
        )


def test_source_provenance_classes_fail_closed():
    provenance = json.loads(
        (ROOT / "paper/artifacts/provenance_manifest.json").read_text(encoding="utf-8")
    )
    historical = provenance["frozen_empirical_ntk_studies"][0]
    assert validate_source_provenance(historical) == "historical_dirty_unreconstructible"

    false_replay_claim = copy.deepcopy(historical)
    false_replay_claim["source_reconstructible"] = True
    with pytest.raises(VerificationError, match="cannot claim reconstructibility"):
        validate_source_provenance(false_replay_claim)

    commit = subprocess.check_output(
        [GIT_EXECUTABLE, "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    source_sha256, source_file_count = git_executable_source_fingerprint(ROOT, commit)
    clean = {
        "source_provenance_class": "clean_git_source",
        "source_sha256": source_sha256,
        "source_file_count": source_file_count,
        "git_commit": commit,
        "git_dirty": False,
        "source_snapshot_archived": False,
        "source_reconstructible": True,
    }
    assert validate_source_provenance(clean, repo_root=ROOT) == "clean_git_source"

    with pytest.raises(VerificationError, match="repository-backed commit verification"):
        validate_source_provenance(clean)

    nonexistent_commit = {**clean, "git_commit": "f" * 40}
    with pytest.raises(VerificationError, match="does not exist as a commit"):
        validate_source_provenance(nonexistent_commit, repo_root=ROOT)

    wrong_digest = {**clean, "source_sha256": "a" * 64}
    with pytest.raises(VerificationError, match="do not match source bytes"):
        validate_source_provenance(wrong_digest, repo_root=ROOT)

    unknown = {**clean, "source_provenance_class": "digest_only"}
    with pytest.raises(VerificationError, match="Unknown source_provenance_class"):
        validate_source_provenance(unknown, repo_root=ROOT)


def test_archive_verifier_rejects_coherently_rehashed_model_family_deletion(tmp_path):
    release_root = _copy_release_tree(tmp_path)
    archive_root = ARCHIVE_ROOTS[1]
    archive_dir = release_root / archive_root

    preflight_path = archive_dir / "preflight/manifest.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    for record in preflight["records"]:
        if record["model_kind"] == "gru":
            record["model_kind"] = "tanh"
    preflight_path.write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    preflight_sha256 = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    (archive_dir / "preflight/manifest.sha256").write_text(
        f"{preflight_sha256}  manifest.json\n", encoding="utf-8"
    )

    provenance_path = archive_dir / "evaluation/provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["manifest_sha256"] = preflight_sha256
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    acceptance_path = archive_dir / "evaluation/evaluation_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    acceptance["manifest_sha256"] = preflight_sha256
    acceptance["checks"] = {
        key: value
        for key, value in acceptance["checks"].items()
        if not key.startswith("gru_")
    }
    acceptance_path.write_text(json.dumps(acceptance, indent=2) + "\n", encoding="utf-8")

    metrics_path = archive_dir / "evaluation/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["manifest_sha256"] = preflight_sha256
    metrics["acceptance"] = acceptance
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    for relative_path in (
        "preflight/manifest.json",
        "preflight/manifest.sha256",
        "evaluation/provenance.json",
        "evaluation/evaluation_acceptance.json",
        "evaluation/metrics.json",
    ):
        _rewrite_archive_member_signature(release_root, archive_root, relative_path)

    with pytest.raises(VerificationError, match="model-family record counts"):
        verify_archive_manifest(
            release_root,
            archive_root,
            expected_primary_outputs=("drift_crossing", "response_crossing"),
            expected_records_by_model={"tanh": 32, "gru": 32},
            expected_thresholds={
                "minimum_model_accuracy": 0.625,
                "minimum_overall_accuracy": 0.75,
                "minimum_two_way_bootstrap_ci95_low": 0.5,
                "required_prediction_coverage": 1.0,
            },
            expected_evaluation_passed=True,
        )


def test_archive_verifier_rejects_coherently_rehashed_bootstrap_deletion(tmp_path):
    release_root = _copy_release_tree(tmp_path)
    archive_root = ARCHIVE_ROOTS[1]
    archive_dir = release_root / archive_root

    acceptance_path = archive_dir / "evaluation/evaluation_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    acceptance["thresholds"].pop("minimum_two_way_bootstrap_ci95_low")
    acceptance["checks"] = {
        key: value
        for key, value in acceptance["checks"].items()
        if not key.endswith("_bootstrap_ci95_low")
    }
    acceptance_path.write_text(json.dumps(acceptance, indent=2) + "\n", encoding="utf-8")

    metrics_path = archive_dir / "evaluation/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["acceptance"] = acceptance
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    for relative_path in (
        "evaluation/evaluation_acceptance.json",
        "evaluation/metrics.json",
    ):
        _rewrite_archive_member_signature(release_root, archive_root, relative_path)

    with pytest.raises(VerificationError, match="thresholds disagree"):
        verify_archive_manifest(
            release_root,
            archive_root,
            expected_primary_outputs=("drift_crossing", "response_crossing"),
            expected_records_by_model={"tanh": 32, "gru": 32},
            expected_thresholds={
                "minimum_model_accuracy": 0.625,
                "minimum_overall_accuracy": 0.75,
                "minimum_two_way_bootstrap_ci95_low": 0.5,
                "required_prediction_coverage": 1.0,
            },
            expected_evaluation_passed=True,
        )