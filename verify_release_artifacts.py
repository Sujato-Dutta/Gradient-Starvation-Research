#!/usr/bin/env python3
"""Read-only verification for compact claim-bearing release artifacts.

This script intentionally uses only the Python standard library. It verifies the
central provenance index, both compact archive manifests, every archived byte,
the initialization-only and evaluation seals, and the disclosed source-provenance
class. It does not import project code, regenerate outcomes, or claim that missing
historical source bytes can be reconstructed from their digests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
GIT_EXECUTABLE = shutil.which("git") or next(
    (path for path in ("/usr/bin/git", "/opt/homebrew/bin/git", "/usr/local/bin/git") if Path(path).is_file()),
    "git",
)
COMPLETED_ARCHIVE_SCHEMA = "completed-run-compact-archive-v1"
COMPLETED_SOURCE_COMMIT = "9ebbc617a9aaad5c3c19e9d5fd4fa08f2d3517d0"
CANONICAL_PROVENANCE_INTERPRETATION = (
    "The final theorem-aligned E-NL source is reconstructible and content-addressed "
    "through commit 2d0ee83/tag theorem-aligned-v1. The clean E26--E28 execution source "
    "is separately reconstructible from commit "
    "9ebbc617a9aaad5c3c19e9d5fd4fa08f2d3517d0. In contrast, E19/E20 retain auditable "
    "compact outcomes and seals but their dirty executed Python source bytes are "
    "unavailable and unreconstructible; E19 failed its broad acceptance rule and E20 "
    "passed only the restricted drift/response crossing-classification endpoints. Source "
    "reconstructibility is not replay completeness: the E26 archive omits 192 records/** "
    "files and the shared E27/E28 archive omits 1,792 records/** files; hashes cannot "
    "recover those omitted bytes, so neither compact archive is a self-contained replay "
    "package. Manifest integrity establishes identity, not scientific validity: it does "
    "not prove kernel stability, causal prediction, calibrated crossing times, event "
    "prevalence, architecture ranking, canonical baseline parity, or broad CDC superiority. "
    "Historical provenance limitations for E1, E2, E2-R, exploratory E-NL, and E3 remain "
    "unchanged."
)
COMPLETED_ARCHIVE_CONTRACTS = {
    "semi-real-generated-cue-v1": {
        "claim_ids": ("E26",),
        "archive_root": "paper/artifacts/semi-real-generated-cue-v1-20260830",
        "source_schema": "semi-real-source-manifest-v1",
        "source_sha256": "02ba4ce4d6aff3ff726d9e798606aed29cff1276a6837e10ff43fb37824ca042",
        "source_file_count": 22,
        "artifact_schema": "semi-real-artifact-manifest-v1",
        "declared_file_count": 207,
        "omitted_file_count": 192,
        "required_members": {
            "acceptance.json", "artifact_manifest.json", "artifact_manifest.sha256",
            "authorization.seal.json", "contract.resolved.json", "environment-lock.txt",
            "environment.json", "inference.json", "preflight.manifest.json",
            "preflight.manifest.sha256", "provenance.json", "raw_data_manifest.json",
            "record_summary.csv", "record_summary.json", "resource_usage.json",
            "selection_manifest.json", "source_manifest.json",
        },
    },
    "expanded-nonlinear-beta-cdc-tradeoff-v1": {
        "claim_ids": ("E27", "E28"),
        "archive_root": "paper/artifacts/expanded-studies-v1-20260830",
        "source_schema": "expanded-studies-source-manifest-v1",
        "source_sha256": "7f43b602b71199367938c1df51aa201a7d601e57ebb47835bdb5c9282b42f84e",
        "source_file_count": 23,
        "artifact_schema": "expanded-studies-artifact-manifest-v1",
        "declared_file_count": 1817,
        "omitted_file_count": 1792,
        "required_members": {
            "artifact_manifest.json", "artifact_manifest.sha256", "authorization.seal.json",
            "bootstrap_draw_manifest.json", "contract.resolved.json", "environment-lock.txt",
            "environment.json", "historical-contract.json", "initialization_manifest.json",
            "preflight.manifest.json", "preflight.manifest.sha256", "provenance.json",
            "record_manifest.json", "resource_usage.json", "source_manifest.json",
            "study_a_claims.json", "study_a_first_hit_profile.csv", "study_a_inference.json",
            "study_a_record_summary.json", "study_b_claims.json", "study_b_cost.csv",
            "study_b_diagnostics.csv", "study_b_inference.json", "study_b_pareto_final.json",
            "study_b_pareto_trajectory.json", "study_b_record_summary.csv",
            "study_b_record_summary.json",
        },
    },
}
SOURCE_PROVENANCE_CLASSES = {
    "historical_dirty_unreconstructible",
    "clean_git_source",
}
REQUIRED_ARCHIVE_MEMBERS = {
    "preflight/config.resolved.yaml",
    "preflight/manifest.json",
    "preflight/manifest.sha256",
    "preflight/preflight_acceptance.json",
    "evaluation/config.resolved.yaml",
    "evaluation/metrics.json",
    "evaluation/evaluation_acceptance.json",
    "evaluation/provenance.json",
    "evaluation/scores.csv",
}
PREFLIGHT_ACCEPTANCE_CONTRACT = {
    "stage": "initialization_only_preflight",
    "optimizer_constructed": False,
    "training_trajectory_observed": False,
    "state_unchanged": True,
    "parameter_gradients_unpopulated": True,
    "all_kernel_and_prediction_values_finite": True,
    "passed": True,
}
FROZEN_STUDY_CONTRACTS = {
    "E19": {
        "accepted_primary_outputs": (
            "phase",
            "drift_crossing",
            "response_crossing",
            "causal_certificate",
            "weak_only_learnability",
        ),
        "acceptance_contract": {
            "total_records": 16,
            "records_by_model": {"tanh": 8, "gru": 8},
            "required_thresholds": {
                "minimum_model_accuracy": 0.625,
                "minimum_overall_accuracy": 0.75,
                "required_prediction_coverage": 1.0,
            },
        },
        "evaluation_passed": False,
    },
    "E20": {
        "accepted_primary_outputs": ("drift_crossing", "response_crossing"),
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
        "evaluation_passed": True,
    },
}
RELEASE_CONTRACT_PATHS = {
    ".github/workflows/tests.yml",
    ".python-version",
    "README.md",
    "requirements.txt",
    "requirements-lock-py312.txt",
    "research_scope/submission_claim_set.yaml",
    "research_scope/submission_readiness.yaml",
    "scripts/check_environment_lock.py",
    "scripts/check_submission_readiness.py",
    "tests/test_enl_evaluation_contract.py",
    "tests/test_environment_lock.py",
    "tests/test_release_artifacts.py",
    "tests/test_submission_readiness.py",
    "verify_release_artifacts.py",
}


class VerificationError(ValueError):
    """Raised when a release artifact violates the integrity contract."""


@dataclass(frozen=True)
class FileSignature:
    sha256: str
    bytes: int


@dataclass(frozen=True)
class ArchiveVerification:
    archive_root: str
    member_signatures: dict[str, FileSignature]
    archived_file_bytes: int
    preflight_manifest_sha256: str
    prediction_sha256: str
    source_sha256: str
    verified_frozen_file_count: int
    evaluation_passed: bool
    accepted_primary_outputs: tuple[str, ...]


@dataclass(frozen=True)
class CompletedArchiveVerification:
    archive_root: str
    study_id: str
    claim_ids: tuple[str, ...]
    member_signatures: dict[str, FileSignature]
    archived_file_bytes: int
    source_sha256: str
    source_file_count: int
    declared_file_count: int
    omitted_file_count: int


@dataclass(frozen=True)
class ReleaseVerification:
    provenance_path: str
    study_count: int
    archive_count: int
    file_count: int
    byte_count: int
    historical_unreconstructible_study_count: int
    git_tracking_checked: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise VerificationError(f"Non-finite JSON constant {value!r} is forbidden.")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except VerificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"Cannot parse {label} at {path}: {error}") from error
    _require(isinstance(payload, dict), f"{label} must be a JSON object.")
    return payload


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"{label} must be a lowercase 64-character SHA-256 digest.",
    )
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a nonnegative integer.",
    )
    return value


def _relative_parts(raw_path: Any, label: str) -> tuple[str, ...]:
    _require(isinstance(raw_path, str) and raw_path, f"{label} must be a non-empty string.")
    _require("\\" not in raw_path and "\x00" not in raw_path, f"Unsafe {label}: {raw_path!r}.")
    path = PurePosixPath(raw_path)
    _require(not path.is_absolute(), f"Unsafe absolute {label}: {raw_path!r}.")
    _require(
        path.as_posix() == raw_path
        and raw_path not in {".", ".."}
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"Unsafe non-canonical or traversing {label}: {raw_path!r}.",
    )
    return path.parts


def _candidate(root: Path, raw_path: Any, label: str) -> Path:
    return root.joinpath(*_relative_parts(raw_path, label))


def _regular_file(root: Path, raw_path: Any, label: str) -> Path:
    _require(root.is_dir(), f"Verification root is not a directory: {root}.")
    candidate = _candidate(root, raw_path, label)
    cursor = root
    for part in _relative_parts(raw_path, label):
        cursor = cursor / part
        _require(not cursor.is_symlink(), f"Symlinks are forbidden for {label}: {raw_path!r}.")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise VerificationError(f"Missing or escaping {label}: {raw_path!r}.") from error
    _require(resolved.is_file(), f"Expected regular file for {label}: {raw_path!r}.")
    return resolved


def _directory(root: Path, raw_path: Any, label: str) -> Path:
    candidate = _candidate(root, raw_path, label)
    cursor = root
    for part in _relative_parts(raw_path, label):
        cursor = cursor / part
        _require(not cursor.is_symlink(), f"Symlinks are forbidden for {label}: {raw_path!r}.")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise VerificationError(f"Missing or escaping {label}: {raw_path!r}.") from error
    _require(resolved.is_dir(), f"Expected directory for {label}: {raw_path!r}.")
    return resolved


def _file_signature(path: Path) -> FileSignature:
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise VerificationError(f"Cannot read release artifact {path}: {error}") from error
    return FileSignature(digest.hexdigest(), byte_count)


def _declared_signature(entry: dict[str, Any], label: str) -> FileSignature:
    return FileSignature(
        _require_sha256(entry.get("sha256"), f"{label}.sha256"),
        _require_nonnegative_int(entry.get("bytes"), f"{label}.bytes"),
    )


def _verify_signature(path: Path, expected: FileSignature, label: str) -> FileSignature:
    actual = _file_signature(path)
    _require(
        actual == expected,
        f"Artifact hash/size mismatch for {label}: expected {expected}, got {actual}.",
    )
    return actual


def git_executable_source_fingerprint(repo_root: Path, commit: str) -> tuple[str, int]:
    """Reproduce executable_source_fingerprint from source bytes stored in a commit."""
    _require(
        isinstance(commit, str) and GIT_COMMIT_PATTERN.fullmatch(commit) is not None,
        "Git source fingerprint requires a lowercase 40-character commit ID.",
    )
    repo_root = repo_root.resolve(strict=True)
    try:
        subprocess.run(
            [GIT_EXECUTABLE, "-C", str(repo_root), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        listing = subprocess.run(
            [
                GIT_EXECUTABLE,
                "-C",
                str(repo_root),
                "ls-tree",
                "-r",
                "-z",
                "--name-only",
                commit,
                "--",
                "run_experiment.py",
                "src",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise VerificationError(
            f"clean_git_source commit does not exist as a commit: {commit}."
        ) from error

    listed_paths = [
        value.decode("utf-8") for value in listing.split(b"\0") if value
    ]
    source_paths = [
        "run_experiment.py",
        *sorted(
            path
            for path in listed_paths
            if path.startswith("src/") and path.endswith(".py")
        ),
    ]
    _require(
        "run_experiment.py" in listed_paths,
        f"Commit {commit} does not contain run_experiment.py.",
    )
    digest = hashlib.sha256()
    for source_path in source_paths:
        try:
            source_bytes = subprocess.run(
                [
                    GIT_EXECUTABLE,
                    "-C",
                    str(repo_root),
                    "cat-file",
                    "blob",
                    f"{commit}:{source_path}",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise VerificationError(
                f"Cannot reconstruct {source_path} from clean source commit {commit}."
            ) from error
        digest.update(source_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_bytes)
        digest.update(b"\0")
    return digest.hexdigest(), len(source_paths)


def validate_source_provenance(
    study: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> str:
    """Validate and, for clean source, reconstruct the declared source identity."""
    provenance_class = study.get("source_provenance_class")
    _require(
        provenance_class in SOURCE_PROVENANCE_CLASSES,
        f"Unknown source_provenance_class: {provenance_class!r}.",
    )
    source_sha256 = _require_sha256(study.get("source_sha256"), "study.source_sha256")
    git_commit = study.get("git_commit")
    _require(
        isinstance(git_commit, str) and GIT_COMMIT_PATTERN.fullmatch(git_commit) is not None,
        "study.git_commit must be a lowercase 40-character Git object ID.",
    )
    _require(isinstance(study.get("git_dirty"), bool), "study.git_dirty must be boolean.")
    _require(
        isinstance(study.get("source_snapshot_archived"), bool),
        "study.source_snapshot_archived must be boolean.",
    )
    _require(
        isinstance(study.get("source_reconstructible"), bool),
        "study.source_reconstructible must be boolean.",
    )

    if provenance_class == "historical_dirty_unreconstructible":
        _require(study["git_dirty"] is True, "Historical dirty source must record git_dirty=true.")
        _require(
            study["source_snapshot_archived"] is False,
            "Historical dirty source cannot claim an archived source snapshot.",
        )
        _require(
            study["source_reconstructible"] is False,
            "Historical dirty source cannot claim reconstructibility.",
        )
        limitation = study.get("source_limitation")
        _require(
            isinstance(limitation, str) and len(limitation.strip()) >= 40,
            "Historical dirty source requires an explicit source_limitation.",
        )
    else:
        _require(study["git_dirty"] is False, "clean_git_source requires git_dirty=false.")
        _require(
            study["source_reconstructible"] is True,
            "clean_git_source requires source_reconstructible=true.",
        )
        _require(
            repo_root is not None,
            "clean_git_source requires repository-backed commit verification.",
        )
        declared_count = _require_nonnegative_int(
            study.get("source_file_count"), "study.source_file_count"
        )
        reconstructed_sha256, reconstructed_count = git_executable_source_fingerprint(
            repo_root, git_commit
        )
        _require(
            (source_sha256, declared_count)
            == (reconstructed_sha256, reconstructed_count),
            "clean_git_source digest/count do not match source bytes in the declared commit.",
        )
    return provenance_class


def verify_archive_manifest(
    repo_root: Path,
    archive_root: str,
    *,
    expected_primary_outputs: tuple[str, ...] | None = None,
    expected_records_by_model: dict[str, int] | None = None,
    expected_thresholds: dict[str, float] | None = None,
    expected_evaluation_passed: bool | None = None,
) -> ArchiveVerification:
    """Verify one compact archive and return its semantic seals."""
    repo_root = repo_root.resolve(strict=True)
    archive_dir = _directory(repo_root, archive_root, "archive root")
    manifest_path = _regular_file(archive_dir, "archive_manifest.json", "archive manifest")
    manifest = _load_json(manifest_path, "archive manifest")
    _require(manifest.get("schema_version") == 1, "Unsupported archive manifest schema.")
    _require(
        manifest.get("archive_root") == archive_root,
        f"Archive root declaration mismatch for {archive_root}.",
    )
    entries = manifest.get("archived_files")
    _require(isinstance(entries, list), "archive_manifest.archived_files must be a list.")

    member_signatures: dict[str, FileSignature] = {}
    source_paths: set[str] = set()
    byte_total = 0
    for index, raw_entry in enumerate(entries):
        _require(isinstance(raw_entry, dict), f"Archive entry {index} must be a mapping.")
        source_path = raw_entry.get("source_path")
        _relative_parts(source_path, f"archive entry {index} source_path")
        _require(source_path not in source_paths, f"Duplicate archive source path: {source_path!r}.")
        source_paths.add(source_path)

        relative_path = raw_entry.get("archive_relative_path")
        _relative_parts(relative_path, f"archive entry {index} archive_relative_path")
        _require(
            relative_path not in member_signatures,
            f"Duplicate archive member path: {relative_path!r}.",
        )
        expected = _declared_signature(raw_entry, f"archive entry {index}")
        member = _regular_file(archive_dir, relative_path, "archive member")
        member_signatures[relative_path] = _verify_signature(member, expected, relative_path)
        byte_total += expected.bytes

    _require(
        REQUIRED_ARCHIVE_MEMBERS <= set(member_signatures),
        "Compact archive omits required members: "
        f"{sorted(REQUIRED_ARCHIVE_MEMBERS - set(member_signatures))}.",
    )
    _require(
        manifest.get("archived_file_count") == len(entries),
        "archive_manifest.archived_file_count is inconsistent.",
    )
    _require(
        manifest.get("archived_file_bytes") == byte_total,
        "archive_manifest.archived_file_bytes is inconsistent.",
    )

    preflight_manifest_path = _regular_file(
        archive_dir, "preflight/manifest.json", "sealed preflight manifest"
    )
    preflight_manifest = _load_json(preflight_manifest_path, "sealed preflight manifest")
    _require(
        preflight_manifest.get("schema_version") == "enl-ntk-preflight-v2",
        "Unsupported sealed preflight schema.",
    )
    _require(preflight_manifest.get("preflight_passed") is True, "Sealed preflight did not pass.")
    prediction_sha256 = _require_sha256(
        preflight_manifest.get("prediction_sha256"), "preflight prediction_sha256"
    )
    source_sha256 = _require_sha256(
        preflight_manifest.get("source_sha256"), "preflight source_sha256"
    )
    preflight_manifest_sha256 = _file_signature(preflight_manifest_path).sha256

    sidecar_path = _regular_file(
        archive_dir, "preflight/manifest.sha256", "preflight manifest sidecar"
    )
    try:
        sidecar_fields = sidecar_path.read_text(encoding="utf-8").strip().split()
    except (OSError, UnicodeError) as error:
        raise VerificationError(f"Cannot read preflight manifest sidecar: {error}") from error
    _require(
        sidecar_fields == [preflight_manifest_sha256, "manifest.json"],
        "Preflight manifest sidecar does not seal manifest.json.",
    )

    preflight_files = preflight_manifest.get("files")
    records = preflight_manifest.get("records")
    _require(isinstance(preflight_files, list), "Sealed preflight files must be a list.")
    _require(isinstance(records, list) and records, "Sealed preflight records must be non-empty.")
    preflight_file_signatures: dict[str, FileSignature] = {}
    for index, raw_entry in enumerate(preflight_files):
        _require(isinstance(raw_entry, dict), f"Preflight file entry {index} must be a mapping.")
        raw_path = raw_entry.get("path")
        _relative_parts(raw_path, f"preflight file entry {index} path")
        _require(
            raw_path not in preflight_file_signatures,
            f"Duplicate sealed preflight file path: {raw_path!r}.",
        )
        preflight_file_signatures[raw_path] = _declared_signature(
            raw_entry, f"preflight file entry {index}"
        )
    for compact_name in ("config.resolved.yaml", "preflight_acceptance.json"):
        _require(compact_name in preflight_file_signatures, f"Preflight omits {compact_name}.")
        compact_signature = member_signatures[f"preflight/{compact_name}"]
        _require(
            compact_signature == preflight_file_signatures[compact_name],
            f"Compact preflight copy disagrees with sealed declaration for {compact_name}.",
        )

    preflight_acceptance = _load_json(
        _regular_file(
            archive_dir,
            "preflight/preflight_acceptance.json",
            "preflight acceptance",
        ),
        "preflight acceptance",
    )
    for field, expected in PREFLIGHT_ACCEPTANCE_CONTRACT.items():
        _require(
            preflight_acceptance.get(field) == expected,
            f"Preflight acceptance contract failed at {field!r}.",
        )

    verified_frozen_file_count = 2 + len(preflight_files) + len(records)
    evaluation_provenance = _load_json(
        _regular_file(
            archive_dir, "evaluation/provenance.json", "evaluation provenance"
        ),
        "evaluation provenance",
    )
    _require(
        evaluation_provenance.get("frozen_inputs_unchanged") is True,
        "Evaluation does not certify frozen inputs unchanged.",
    )
    _require(
        evaluation_provenance.get("manifest_sha256") == preflight_manifest_sha256,
        "Evaluation provenance manifest digest mismatch.",
    )
    _require(
        evaluation_provenance.get("prediction_sha256") == prediction_sha256,
        "Evaluation provenance prediction digest mismatch.",
    )
    _require(
        evaluation_provenance.get("source_sha256") == source_sha256,
        "Evaluation provenance source digest mismatch.",
    )
    _require(
        evaluation_provenance.get("verified_file_count") == verified_frozen_file_count,
        "Evaluation provenance verified-file count mismatch.",
    )

    evaluation_acceptance = _load_json(
        _regular_file(
            archive_dir,
            "evaluation/evaluation_acceptance.json",
            "evaluation acceptance",
        ),
        "evaluation acceptance",
    )
    _require(
        evaluation_acceptance.get("stage") == "held_out_falsification_evaluation",
        "Unexpected evaluation-acceptance stage.",
    )
    _require(
        evaluation_acceptance.get("manifest_sha256") == preflight_manifest_sha256,
        "Evaluation acceptance manifest digest mismatch.",
    )
    _require(
        evaluation_acceptance.get("prediction_sha256") == prediction_sha256,
        "Evaluation acceptance prediction digest mismatch.",
    )
    evaluation_passed = evaluation_acceptance.get("passed")
    _require(isinstance(evaluation_passed, bool), "evaluation_acceptance.passed must be boolean.")
    checks = evaluation_acceptance.get("checks")
    _require(
        isinstance(checks, dict)
        and checks
        and all(isinstance(key, str) and isinstance(value, bool) for key, value in checks.items()),
        "evaluation_acceptance.checks must be a non-empty boolean mapping.",
    )
    _require(
        evaluation_passed is all(checks.values()),
        "evaluation_acceptance.passed does not equal all declared checks.",
    )
    if expected_evaluation_passed is not None:
        _require(
            evaluation_passed is expected_evaluation_passed,
            "Evaluation pass/fail result disagrees with central provenance.",
        )

    raw_primary_outputs = evaluation_acceptance.get("accepted_primary_outputs")
    if raw_primary_outputs is None:
        raw_primary_outputs = []
    _require(
        isinstance(raw_primary_outputs, list)
        and all(isinstance(item, str) and item for item in raw_primary_outputs)
        and len(raw_primary_outputs) == len(set(raw_primary_outputs)),
        "accepted_primary_outputs must be a duplicate-free string list when present.",
    )
    if expected_primary_outputs is None:
        effective_primary_outputs = tuple(raw_primary_outputs)
    else:
        _require(
            expected_primary_outputs
            and all(isinstance(item, str) and item for item in expected_primary_outputs)
            and len(expected_primary_outputs) == len(set(expected_primary_outputs)),
            "Central accepted_primary_outputs must be a non-empty duplicate-free tuple.",
        )
        if "accepted_primary_outputs" in evaluation_acceptance:
            _require(
                tuple(raw_primary_outputs) == expected_primary_outputs,
                "Evaluation accepted_primary_outputs disagree with central provenance.",
            )
        effective_primary_outputs = expected_primary_outputs

    if effective_primary_outputs:
        actual_records_by_model: dict[str, int] = {}
        for index, record in enumerate(records):
            _require(isinstance(record, dict), f"Preflight record {index} must be a mapping.")
            model_kind = record.get("model_kind")
            _require(
                isinstance(model_kind, str) and model_kind,
                f"Preflight record {index} has no model_kind.",
            )
            actual_records_by_model[model_kind] = (
                actual_records_by_model.get(model_kind, 0) + 1
            )
        if expected_records_by_model is not None:
            _require(
                actual_records_by_model == expected_records_by_model,
                "Preflight model-family record counts disagree with the frozen contract.",
            )
            model_kinds = set(expected_records_by_model)
        else:
            model_kinds = set(actual_records_by_model)

        thresholds = evaluation_acceptance.get("thresholds")
        _require(isinstance(thresholds, dict), "evaluation_acceptance.thresholds must be a mapping.")
        if expected_thresholds is not None:
            _require(
                thresholds == expected_thresholds,
                "Evaluation thresholds disagree with the frozen acceptance contract.",
            )
            effective_thresholds = expected_thresholds
        else:
            effective_thresholds = thresholds

        expected_checks = {"prediction_coverage"}
        for output in effective_primary_outputs:
            expected_checks.add(f"overall_{output}")
            expected_checks.update(f"{model_kind}_{output}" for model_kind in model_kinds)
        if effective_thresholds.get("minimum_two_way_bootstrap_ci95_low") is not None:
            expected_checks.update(
                f"overall_{output}_bootstrap_ci95_low"
                for output in effective_primary_outputs
            )
        _require(
            set(checks) == expected_checks,
            "Evaluation acceptance checks do not exactly match the frozen primary outputs: "
            f"missing={sorted(expected_checks - set(checks))}, "
            f"extra={sorted(set(checks) - expected_checks)}.",
        )

    metrics = _load_json(
        _regular_file(archive_dir, "evaluation/metrics.json", "evaluation metrics"),
        "evaluation metrics",
    )
    _require(
        metrics.get("manifest_sha256") == preflight_manifest_sha256
        and metrics.get("prediction_sha256") == prediction_sha256,
        "Evaluation metrics digest identity mismatch.",
    )
    _require(
        metrics.get("acceptance") == evaluation_acceptance,
        "Evaluation metrics and acceptance seal disagree.",
    )

    return ArchiveVerification(
        archive_root=archive_root,
        member_signatures=member_signatures,
        archived_file_bytes=byte_total,
        preflight_manifest_sha256=preflight_manifest_sha256,
        prediction_sha256=prediction_sha256,
        source_sha256=source_sha256,
        verified_frozen_file_count=verified_frozen_file_count,
        evaluation_passed=evaluation_passed,
        accepted_primary_outputs=effective_primary_outputs,
    )


def _read_sidecar(path: Path, expected_digest: str, expected_name: str, label: str) -> None:
    try:
        fields = path.read_text(encoding="utf-8").strip().split()
    except (OSError, UnicodeError) as error:
        raise VerificationError(f"Cannot read {label}: {error}") from error
    _require(fields == [expected_digest, expected_name], f"{label} does not seal {expected_name}.")


def _require_exact_keys(mapping: Any, expected: set[str], label: str) -> dict[str, Any]:
    _require(isinstance(mapping, dict), f"{label} must be a mapping.")
    _require(
        set(mapping) == expected,
        f"{label} keys are not exact: missing={sorted(expected - set(mapping))}, "
        f"extra={sorted(set(mapping) - expected)}.",
    )
    return mapping


def _verify_clean_source_manifest(
    source_repo_root: Path,
    archive_dir: Path,
    member_signatures: dict[str, FileSignature],
    contract: dict[str, Any],
) -> tuple[str, int]:
    source_path = _regular_file(archive_dir, "source_manifest.json", "source manifest")
    source = _load_json(source_path, "source manifest")
    _require(source.get("schema_version") == contract["source_schema"], "Unexpected source-manifest schema.")
    _require(source.get("hash_algorithm") == "sha256", "Source manifest must use SHA-256.")
    entries = source.get("files")
    _require(isinstance(entries, list), "source_manifest.files must be a list.")
    _require(len(entries) == contract["source_file_count"], "Source-manifest file count is not frozen.")
    commit = COMPLETED_SOURCE_COMMIT
    try:
        subprocess.run(
            [GIT_EXECUTABLE, "-C", str(source_repo_root), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise VerificationError(f"Clean source commit is unavailable: {commit}.") from error

    seen: set[str] = set()
    for index, raw_entry in enumerate(entries):
        _require(isinstance(raw_entry, dict), f"Source entry {index} must be a mapping.")
        raw_path = raw_entry.get("path")
        _relative_parts(raw_path, f"source entry {index} path")
        _require(raw_path not in seen, f"Duplicate source-manifest path: {raw_path!r}.")
        seen.add(raw_path)
        expected = FileSignature(
            _require_sha256(raw_entry.get("sha256"), f"source entry {index}.sha256"),
            _require_nonnegative_int(raw_entry.get("size"), f"source entry {index}.size"),
        )
        try:
            blob = subprocess.run(
                [GIT_EXECUTABLE, "-C", str(source_repo_root), "cat-file", "blob", f"{commit}:{raw_path}"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise VerificationError(f"Cannot reconstruct {raw_path} from clean commit {commit}.") from error
        actual = FileSignature(hashlib.sha256(blob).hexdigest(), len(blob))
        _require(actual == expected, f"Source-manifest blob mismatch for {raw_path}.")

    source_sha256 = _require_sha256(source.get("aggregate_sha256"), "source aggregate_sha256")
    _require(source_sha256 == contract["source_sha256"], "Source aggregate is not the frozen clean-source digest.")
    canonical_entries = (json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    _require(hashlib.sha256(canonical_entries).hexdigest() == source_sha256, "Source aggregate does not match the source-manifest entries.")
    _require(member_signatures["source_manifest.json"] == _file_signature(source_path), "Source manifest is not sealed by the archive.")
    return source_sha256, len(entries)


def _verify_completed_archive_structure(
    repo_root: Path,
    source_repo_root: Path,
    archive_root: str,
    expected_study_id: str,
) -> tuple[dict[str, Any], dict[str, FileSignature], dict[str, Any], Path, str, int]:
    contract = COMPLETED_ARCHIVE_CONTRACTS[expected_study_id]
    archive_dir = _directory(repo_root, archive_root, "completed-run archive root")
    manifest_path = _regular_file(archive_dir, "archive_manifest.json", "completed-run archive manifest")
    manifest = _load_json(manifest_path, "completed-run archive manifest")
    _require(manifest.get("schema_version") == COMPLETED_ARCHIVE_SCHEMA, "Unsupported completed-run archive schema.")
    _require(manifest.get("archive_type") == "completed_run_compact", "Unexpected completed-run archive type.")
    _require(manifest.get("archive_root") == archive_root == contract["archive_root"], "Completed-run archive root mismatch.")
    _require(manifest.get("study_id") == expected_study_id, "Completed-run study ID mismatch.")
    _require(manifest.get("claim_ids") == list(contract["claim_ids"]), "Completed-run claim membership is not frozen.")
    _require(manifest.get("clean_source_commit") == COMPLETED_SOURCE_COMMIT, "Completed-run source commit is not frozen.")

    entries = manifest.get("archived_files")
    _require(isinstance(entries, list), "completed archive archived_files must be a list.")
    member_signatures: dict[str, FileSignature] = {}
    source_paths: set[str] = set()
    byte_total = 0
    for index, raw_entry in enumerate(entries):
        _require(isinstance(raw_entry, dict), f"Completed archive entry {index} must be a mapping.")
        source_path = raw_entry.get("source_path")
        _relative_parts(source_path, f"completed archive entry {index} source_path")
        _require(source_path not in source_paths, f"Duplicate completed archive source path: {source_path!r}.")
        source_paths.add(source_path)
        relative_path = raw_entry.get("archive_relative_path")
        _relative_parts(relative_path, f"completed archive entry {index} archive_relative_path")
        _require(relative_path not in member_signatures, f"Duplicate completed archive member path: {relative_path!r}.")
        expected = _declared_signature(raw_entry, f"completed archive entry {index}")
        member_signatures[relative_path] = _verify_signature(
            _regular_file(archive_dir, relative_path, "completed archive member"), expected, relative_path
        )
        byte_total += expected.bytes
    _require(manifest.get("archived_file_count") == len(entries), "Completed archive file count is inconsistent.")
    _require(manifest.get("archived_file_bytes") == byte_total, "Completed archive byte count is inconsistent.")
    _require(set(member_signatures) == contract["required_members"], "Completed archive member set is not frozen.")

    physical_paths: set[str] = set()
    for candidate in archive_dir.rglob("*"):
        _require(not candidate.is_symlink(), f"Symlinks are forbidden in completed archive: {candidate}.")
        if candidate.is_file():
            physical_paths.add(candidate.relative_to(archive_dir).as_posix())
    expected_physical = {"archive_manifest.json", *member_signatures}
    _require(
        physical_paths == expected_physical,
        f"Completed archive physical members are not exact: missing={sorted(expected_physical - physical_paths)}, "
        f"extra={sorted(physical_paths - expected_physical)}.",
    )

    artifact_path = _regular_file(archive_dir, "artifact_manifest.json", "full artifact manifest")
    artifact = _load_json(artifact_path, "full artifact manifest")
    _require(artifact.get("schema_version") == contract["artifact_schema"], "Unexpected full artifact-manifest schema.")
    _require(artifact.get("hash_algorithm") == "sha256", "Full artifact manifest must use SHA-256.")
    artifact_entries = artifact.get("files")
    _require(isinstance(artifact_entries, list), "artifact_manifest.files must be a list.")
    artifact_signatures: dict[str, FileSignature] = {}
    for index, raw_entry in enumerate(artifact_entries):
        _require(isinstance(raw_entry, dict), f"Artifact-manifest entry {index} must be a mapping.")
        raw_path = raw_entry.get("path")
        _relative_parts(raw_path, f"artifact-manifest entry {index} path")
        _require(raw_path not in artifact_signatures, f"Duplicate full artifact path: {raw_path!r}.")
        artifact_signatures[raw_path] = FileSignature(
            _require_sha256(raw_entry.get("sha256"), f"artifact entry {index}.sha256"),
            _require_nonnegative_int(raw_entry.get("size"), f"artifact entry {index}.size"),
        )
    declared_count = contract["declared_file_count"]
    omitted_count = contract["omitted_file_count"]
    _require(artifact.get("file_count") == len(artifact_entries) == declared_count, "Full artifact file count is not frozen.")
    omitted_paths = {path for path in artifact_signatures if path.startswith("records/")}
    _require(len(omitted_paths) == omitted_count, "records/** omitted count is not frozen.")
    top_level = set(artifact_signatures) - omitted_paths
    _require(all("/" not in path for path in top_level), "Only records/** may be omitted from a completed archive.")
    _require(set(member_signatures) == top_level | {"artifact_manifest.json", "artifact_manifest.sha256"}, "Completed archive members do not exactly match top-level artifacts.")
    for path in top_level:
        _require(member_signatures[path] == artifact_signatures[path], f"Archive and full manifest disagree for {path}.")

    artifact_signature = _file_signature(artifact_path)
    _require(member_signatures["artifact_manifest.json"] == artifact_signature, "Artifact manifest is not archive-sealed.")
    _read_sidecar(
        _regular_file(archive_dir, "artifact_manifest.sha256", "artifact manifest sidecar"),
        artifact_signature.sha256,
        "artifact_manifest.json",
        "Artifact manifest sidecar",
    )
    commitment = manifest.get("full_run_commitment")
    _require(isinstance(commitment, dict), "full_run_commitment must be a mapping.")
    _require(
        commitment == {
            "artifact_manifest_bytes": artifact_signature.bytes,
            "artifact_manifest_sha256": artifact_signature.sha256,
            "declared_file_count": declared_count,
            "omitted_bytes_archived": False,
            "omitted_bytes_recoverable_from_hashes": False,
            "omitted_file_count": omitted_count,
            "omitted_path_prefix": "records/",
        },
        "Full-run compact commitment is not exact.",
    )
    limitation = manifest.get("limitation")
    _require(
        isinstance(limitation, str)
        and "omitted bytes cannot be recovered from hashes" in limitation
        and "not a self-contained replay package" in limitation,
        "Completed archive must disclose that omitted bytes are unrecoverable and replay is not self-contained.",
    )

    preflight_path = _regular_file(archive_dir, "preflight.manifest.json", "completed preflight manifest")
    preflight_sha = _file_signature(preflight_path).sha256
    _read_sidecar(
        _regular_file(archive_dir, "preflight.manifest.sha256", "completed preflight sidecar"),
        preflight_sha,
        "preflight.manifest.json",
        "Completed preflight sidecar",
    )
    source_sha256, source_count = _verify_clean_source_manifest(
        source_repo_root, archive_dir, member_signatures, contract
    )
    return manifest, member_signatures, artifact, archive_dir, source_sha256, source_count


def _require_clean_git_state(payload: dict[str, Any], label: str) -> None:
    _require(isinstance(payload, dict), f"{label} must be a mapping.")
    _require(
        payload == {"clean": True, "commit": COMPLETED_SOURCE_COMMIT, **({"status_porcelain": []} if "status_porcelain" in payload else {})},
        f"{label} does not bind the frozen clean commit/state.",
    )


def _verify_common_completed_seals(
    archive_dir: Path,
    member_signatures: dict[str, FileSignature],
    study_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    preflight = _load_json(_regular_file(archive_dir, "preflight.manifest.json", "preflight"), "preflight")
    authorization = _load_json(_regular_file(archive_dir, "authorization.seal.json", "authorization"), "authorization")
    provenance = _load_json(_regular_file(archive_dir, "provenance.json", "provenance"), "provenance")
    environment = _load_json(_regular_file(archive_dir, "environment.json", "runtime environment"), "runtime environment")
    _require(preflight.get("study_id") == study_id, "Preflight study ID mismatch.")
    expected_authorization_schema = {
        "semi-real-generated-cue-v1": "semi-real-authorization-seal-v1",
        "expanded-nonlinear-beta-cdc-tradeoff-v1": "expanded-studies-authorization-seal-v1",
    }[study_id]
    _require(authorization.get("schema_version") == expected_authorization_schema, "Unsupported completed-run authorization-seal schema.")
    _require(authorization.get("study_id") == study_id, "Authorization study ID mismatch.")
    if "study_id" in provenance:
        _require(provenance.get("study_id") == study_id, "Provenance study ID mismatch.")
    _require_clean_git_state(preflight.get("git"), "Preflight Git state")
    _require_clean_git_state(environment.get("git"), "Runtime Git state")
    if "expected_git_state" in authorization:
        _require_clean_git_state(authorization.get("expected_git_state"), "Authorization expected Git state")
    else:
        _require_clean_git_state(authorization.get("git"), "Authorization Git state")
    authorization_flag = authorization.get("authorization", {}).get("authorized")
    attestation = authorization.get("untouched_seed_attestation", authorization.get("attestation", {})).get("attested")
    _require(authorization_flag is True and attestation is True, "Scientific authorization/untouched-seed attestation is incomplete.")
    bindings = authorization.get("bindings")
    _require(isinstance(bindings, dict), "Authorization bindings must be a mapping.")
    expected_bindings = {
        "execution_contract_sha256": member_signatures["contract.resolved.json"].sha256,
        "preflight_manifest_sha256": member_signatures["preflight.manifest.json"].sha256,
        "source_manifest_sha256": member_signatures["source_manifest.json"].sha256,
        "environment_lock_sha256": member_signatures["environment-lock.txt"].sha256,
    }
    for key, value in expected_bindings.items():
        _require(bindings.get(key) == value, f"Authorization binding mismatch for {key}.")
    _require(provenance.get("authorization_seal_sha256") == member_signatures["authorization.seal.json"].sha256, "Provenance authorization binding mismatch.")
    for key, value in expected_bindings.items():
        if key != "source_manifest_sha256" or key in provenance:
            _require(provenance.get(key) == value, f"Provenance binding mismatch for {key}.")
    _require(preflight.get("ready_for_authorization") is True, "Preflight was not ready for authorization.")
    _require(preflight.get("optimizer_constructed") is False, "Preflight constructed an optimizer.")
    _require(preflight.get("model_outcomes_inspected") is False, "Preflight inspected model outcomes.")
    return preflight, authorization, provenance


def _verify_semi_real_semantics(
    archive_dir: Path,
    member_signatures: dict[str, FileSignature],
) -> None:
    preflight, authorization, provenance = _verify_common_completed_seals(
        archive_dir, member_signatures, "semi-real-generated-cue-v1"
    )
    _require(preflight.get("schema_version") == "semi-real-preflight-v1", "Unexpected semi-real preflight schema.")
    _require(preflight.get("publication_scope") == "semi-real", "Semi-real publication scope mismatch.")
    bindings = authorization["bindings"]
    source = _load_json(_regular_file(archive_dir, "source_manifest.json", "source manifest"), "source manifest")
    _require(
        preflight.get("source_manifest")
        == {
            "aggregate_sha256": source.get("aggregate_sha256"),
            "file_count": 22,
            "path": "source_manifest.json",
            "sha256": member_signatures["source_manifest.json"].sha256,
            "size": member_signatures["source_manifest.json"].bytes,
        },
        "Semi-real preflight source-manifest binding changed.",
    )
    _require(
        preflight.get("contract", {}).get("sha256") == member_signatures["contract.resolved.json"].sha256
        and preflight.get("contract", {}).get("size") == member_signatures["contract.resolved.json"].bytes
        and preflight.get("environment_lock") == {
            "path": "environment-lock.txt",
            "sha256": member_signatures["environment-lock.txt"].sha256,
            "size": member_signatures["environment-lock.txt"].bytes,
        }
        and preflight.get("raw_data_manifest", {}).get("sha256") == member_signatures["raw_data_manifest.json"].sha256
        and preflight.get("selection_manifest", {}).get("sha256") == member_signatures["selection_manifest.json"].sha256,
        "Semi-real preflight file bindings changed.",
    )
    historical_sha = _require_sha256(preflight.get("historical_contract", {}).get("sha256"), "semi-real historical-contract digest")
    _require(
        bindings.get("historical_contract_sha256") == provenance.get("historical_contract_sha256") == historical_sha,
        "Semi-real historical-contract bindings disagree.",
    )
    _require(bindings.get("source_aggregate_sha256") == source.get("aggregate_sha256"), "Authorization source aggregate mismatch.")
    _require(provenance.get("source_aggregate_sha256") == source.get("aggregate_sha256"), "Provenance source aggregate mismatch.")
    for name, binding_key in (
        ("raw_data_manifest.json", "raw_data_manifest_sha256"),
        ("selection_manifest.json", "selection_manifest_sha256"),
    ):
        expected = member_signatures[name].sha256
        _require(bindings.get(binding_key) == expected and provenance.get(binding_key) == expected, f"Semi-real binding mismatch for {name}.")
    _require(bindings.get("raw_dataset_aggregate_sha256") == provenance.get("raw_dataset_aggregate_sha256") == preflight.get("raw_data_manifest", {}).get("dataset_aggregate_sha256"), "Raw-dataset aggregate bindings disagree.")

    expected_ids = [
        f"sr-{dataset}-d{data_seed}-m{model_seed}"
        for dataset in ("mnist", "fashion_mnist")
        for data_seed in range(4101, 4105)
        for model_seed in range(5101, 5109)
    ]
    _require(preflight.get("expected_record_ids") == expected_ids, "Semi-real preflight record matrix is not frozen.")
    _require(bindings.get("expected_record_ids") == expected_ids, "Semi-real authorization record matrix is not frozen.")
    _require(provenance.get("expected_record_ids") == expected_ids, "Semi-real provenance record matrix is not frozen.")

    summaries = _load_json(_regular_file(archive_dir, "record_summary.json", "semi-real summaries"), "semi-real summaries")
    _require(summaries.get("schema_version") == "semi-real-record-summary-collection-v1", "Unexpected semi-real summary schema.")
    records = summaries.get("records")
    _require(isinstance(records, list) and len(records) == 64, "Semi-real summary must contain exactly 64 records.")
    _require([record.get("record_id") for record in records] == expected_ids, "Semi-real summary record membership/order is not frozen.")
    counts = {dataset: {"records": 0, "causal": 0, "learnable": 0} for dataset in ("mnist", "fashion_mnist")}
    expected_record_bindings = {
        "authorization_seal_sha256": member_signatures["authorization.seal.json"].sha256,
        "execution_contract_sha256": member_signatures["contract.resolved.json"].sha256,
        "preflight_manifest_sha256": member_signatures["preflight.manifest.json"].sha256,
        "raw_dataset_aggregate_sha256": provenance["raw_dataset_aggregate_sha256"],
        "selection_manifest_sha256": member_signatures["selection_manifest.json"].sha256,
        "source_aggregate_sha256": source["aggregate_sha256"],
    }
    intervention_contract = {
        "core_channel_bitwise_equal": True,
        "labels_equal": True,
        "only_generated_channel_differs": True,
        "sample_ids_equal": True,
        "weak_generated_channel_all_zero": True,
    }
    secondary_contract = {
        "gsi5": "missing_secondary_not_defined_for_evaluation_matrix",
        "gsi5_available": False,
        "gsi5_required_for_acceptance": False,
    }
    for record in records:
        dataset = record.get("dataset")
        _require(dataset in counts, "Unknown semi-real dataset.")
        _require(record.get("schema_version") == "semi-real-record-summary-v1", "Unexpected semi-real record schema.")
        _require(record.get("publication_scope") == "semi-real", "Semi-real record publication scope mismatch.")
        _require(record.get("input_bindings") == expected_record_bindings, "Semi-real record input bindings mismatch.")
        _require(record.get("intervention_checks") == intervention_contract, "Semi-real intervention checks are incomplete.")
        _require(record.get("initial_response_gaps") == {"core_both_minus_weak": 0.0, "cue_both_minus_weak": 0.0}, "Semi-real initial response gaps are not exactly zero.")
        _require(record.get("all_initial_response_gaps_exactly_zero") is True, "Semi-real initial equality flag is false.")
        _require(record.get("secondary_endpoint_status") == secondary_contract, "Semi-real GSI-5 limitation changed.")
        primary = record.get("primary_and_trajectory")
        _require(isinstance(primary, dict), "Semi-real primary result missing.")
        counts[dataset]["records"] += 1
        counts[dataset]["causal"] += primary.get("causal_certificate") is True
        counts[dataset]["learnable"] += primary.get("weak_only_learnable") is True
    _require(counts == {"mnist": {"records": 32, "causal": 0, "learnable": 0}, "fashion_mnist": {"records": 32, "causal": 0, "learnable": 0}}, "Semi-real record outcomes are not frozen.")

    inference = _load_json(_regular_file(archive_dir, "inference.json", "semi-real inference"), "semi-real inference")
    _require(
        {key: inference.get(key) for key in ("schema_version", "method", "row_iid_intervals_used", "bootstrap_replicates", "bootstrap_seed")}
        == {"schema_version": "semi-real-inference-v1", "method": "two_way_pigeonhole_bootstrap", "row_iid_intervals_used": False, "bootstrap_replicates": 5000, "bootstrap_seed": 6201},
        "Semi-real inference contract changed.",
    )
    expected_estimates = {
        "mnist": (-0.0012362842136667493, -0.0034405428466004646, 0.0010159591371180453),
        "fashion_mnist": (-0.00010921728159018996, -0.004129384520886106, 0.003182670049955049),
    }
    for dataset, expected in expected_estimates.items():
        result = inference.get("datasets", {}).get(dataset, {})
        _require(
            {key: result.get(key) for key in ("record_count", "data_seed_count", "model_seed_count", "causal_certificate_count", "weak_only_learnable_count")}
            == {"record_count": 32, "data_seed_count": 4, "model_seed_count": 8, "causal_certificate_count": 0, "weak_only_learnable_count": 0},
            f"Semi-real inference counts changed for {dataset}.",
        )
        estimate = result.get("metrics", {}).get("mean_weak_auc_gap", {})
        _require((estimate.get("point_estimate"), estimate.get("ci95_low"), estimate.get("ci95_high")) == expected, f"Semi-real frozen estimate changed for {dataset}.")
    acceptance = _load_json(_regular_file(archive_dir, "acceptance.json", "semi-real acceptance"), "semi-real acceptance")
    _require(acceptance.get("schema_version") == "semi-real-acceptance-v1", "Unexpected semi-real acceptance schema.")
    _require(acceptance.get("publication_scope") == "semi-real" and acceptance.get("gsi5_status") == "missing_secondary_not_defined_for_evaluation_matrix", "Semi-real acceptance scope/limitation changed.")
    _require(
        acceptance.get("thresholds")
        == {
            "all_datasets_must_pass": True,
            "all_initial_response_gaps_exactly_zero": True,
            "all_intervention_checks_required": True,
            "minimum_causal_certificate_count_per_dataset": 16,
            "minimum_weak_only_learnable_count_per_dataset": 24,
            "required_records_per_dataset": 32,
            "required_total_records": 64,
            "weak_auc_gap_ci95_lower_strictly_greater_than": 0.0,
        },
        "Semi-real acceptance thresholds changed.",
    )
    _require(set(acceptance.get("datasets", {})) == {"mnist", "fashion_mnist"}, "Semi-real acceptance dataset set changed.")
    _require(acceptance.get("overall_passed") is False, "Semi-real frozen overall result changed.")
    for dataset in ("mnist", "fashion_mnist"):
        result = acceptance.get("datasets", {}).get(dataset, {})
        _require(
            {key: result.get(key) for key in (
                "record_count", "causal_certificate_count", "weak_only_learnable_count",
                "coverage_passed", "intervention_checks_passed",
                "initial_response_equality_passed", "causal_certificate_passed",
                "weak_only_learnability_passed", "weak_auc_ci95_low",
                "weak_auc_ci_lower_passed", "passed",
            )}
            == {
                "record_count": 32,
                "causal_certificate_count": 0,
                "weak_only_learnable_count": 0,
                "coverage_passed": True,
                "intervention_checks_passed": True,
                "initial_response_equality_passed": True,
                "causal_certificate_passed": False,
                "weak_only_learnability_passed": False,
                "weak_auc_ci95_low": expected_estimates[dataset][1],
                "weak_auc_ci_lower_passed": False,
                "passed": False,
            },
            f"Semi-real acceptance result changed for {dataset}.",
        )


def _expanded_expected_ids() -> tuple[list[str], list[str]]:
    study_a = [
        f"esa-{architecture}-{cell}-d{data_seed}-m{model_seed}"
        for architecture in ("tanh", "gru")
        for cell in ("rho3_lag1", "rho4_lag2", "rho5_lag3")
        for data_seed in range(301, 305)
        for model_seed in range(3010, 3018)
    ]
    study_b = [
        f"esb-{method}-{cell}-d{data_seed}-m{model_seed}"
        for cell in ("rho4_lag2", "rho5_lag3")
        for method in ("erm", "counterfactual_drift", "bloop", "pcgrad")
        for data_seed in range(301, 305)
        for model_seed in range(3010, 3018)
    ]
    return study_a, study_b


def _complete_csv_rows(
    path: Path,
    label: str,
    expected_columns: tuple[str, ...],
    numeric_columns: tuple[str, ...],
) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(tuple(reader.fieldnames or ()) == expected_columns, f"{label} columns are not exact.")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise VerificationError(f"Cannot parse {label}: {error}") from error
    _require(rows, f"{label} must contain records.")
    for row_index, row in enumerate(rows):
        _require(
            set(row) == set(expected_columns)
            and all(isinstance(row[column], str) and row[column].strip() for column in expected_columns),
            f"{label} row {row_index} is incomplete.",
        )
        for column in numeric_columns:
            if row[column] != "N/A":
                try:
                    value = float(row[column])
                except ValueError as error:
                    raise VerificationError(f"{label} row {row_index} has nonnumeric {column}.") from error
                _require(math.isfinite(value), f"{label} row {row_index} has non-finite {column}.")
        _require(
            row["record_id"]
            == f"esb-{row['method_id']}-{row['cell_id']}-d{row['data_seed']}-m{row['model_seed']}",
            f"{label} row {row_index} metadata do not match its record ID.",
        )
    return rows


def _require_frozen_estimate(container: dict[str, Any], key: str, expected: tuple[float, float, float], label: str) -> None:
    estimate = container.get(key, {}).get("equal_weight_two_cell_macro", {})
    _require(
        (estimate.get("point_estimate"), estimate.get("ci95_low"), estimate.get("ci95_high")) == expected
        and estimate.get("bootstrap_replicates") == 5000
        and estimate.get("confidence_interval") == "raw_percentile_linear_95_percent",
        f"Frozen estimate changed for {label}.",
    )


def _verify_expanded_semantics(
    archive_dir: Path,
    member_signatures: dict[str, FileSignature],
) -> None:
    preflight, authorization, provenance = _verify_common_completed_seals(
        archive_dir, member_signatures, "expanded-nonlinear-beta-cdc-tradeoff-v1"
    )
    _require(preflight.get("schema_version") == "expanded-studies-preflight-v1", "Unexpected expanded preflight schema.")
    _require(preflight.get("execution_authorized") is False and preflight.get("training_performed") is False, "Expanded preflight must remain pre-authorization and training-free.")
    _require(provenance.get("git_commit") == COMPLETED_SOURCE_COMMIT, "Expanded provenance commit mismatch.")
    bindings = authorization["bindings"]
    for name, key in (
        ("historical-contract.json", "historical_contract_sha256"),
        ("record_manifest.json", "record_manifest_sha256"),
        ("bootstrap_draw_manifest.json", "bootstrap_draw_manifest_sha256"),
        ("initialization_manifest.json", "initialization_manifest_sha256"),
    ):
        expected = member_signatures[name].sha256
        _require(bindings.get(key) == expected and provenance.get(key) == expected, f"Expanded binding mismatch for {name}.")
    source = _load_json(_regular_file(archive_dir, "source_manifest.json", "source manifest"), "source manifest")
    preflight_files = preflight.get("files")
    _require(isinstance(preflight_files, list), "Expanded preflight files must be a list.")
    preflight_signatures: dict[str, FileSignature] = {}
    for index, entry in enumerate(preflight_files):
        _require(isinstance(entry, dict), f"Expanded preflight file {index} must be a mapping.")
        path = entry.get("path")
        if path == "execution-contract.json":
            archive_path = "contract.resolved.json"
        else:
            archive_path = path
        _require(archive_path in member_signatures, f"Expanded preflight file is not archived: {path!r}.")
        _require(path not in preflight_signatures, f"Duplicate expanded preflight file: {path!r}.")
        preflight_signatures[path] = FileSignature(
            _require_sha256(entry.get("sha256"), f"expanded preflight file {index}.sha256"),
            _require_nonnegative_int(entry.get("size"), f"expanded preflight file {index}.size"),
        )
        _require(preflight_signatures[path] == member_signatures[archive_path], f"Expanded preflight binding mismatch for {path}.")
    _require(
        set(preflight_signatures)
        == {
            "execution-contract.json", "historical-contract.json", "environment-lock.txt",
            "source_manifest.json", "record_manifest.json", "bootstrap_draw_manifest.json",
            "initialization_manifest.json",
        },
        "Expanded preflight file set is not exact.",
    )
    _require(bindings.get("source_aggregate_sha256") == provenance.get("source_aggregate_sha256") == source.get("aggregate_sha256"), "Expanded source aggregate bindings disagree.")

    expected_a, expected_b = _expanded_expected_ids()
    record_manifest = _load_json(_regular_file(archive_dir, "record_manifest.json", "record manifest"), "record manifest")
    _require(record_manifest.get("schema_version") == "expanded-studies-record-manifest-v1", "Unexpected expanded record-manifest schema.")
    _require(record_manifest.get("study_a_expected_record_ids") == expected_a and record_manifest.get("study_a_record_count") == 192, "Study A record matrix is not frozen.")
    _require(record_manifest.get("study_b_expected_record_ids") == expected_b and record_manifest.get("study_b_method_record_count") == 256, "Study B record matrix is not frozen.")
    _require(record_manifest.get("total_scientific_record_count") == 448 and len(set(expected_a + expected_b)) == 448, "Expanded total record count/membership is not exact.")
    for payload, label in ((preflight, "preflight"), (bindings, "authorization"), (provenance, "provenance")):
        _require(payload.get("study_a_expected_record_ids") == expected_a, f"Expanded {label} Study A membership changed.")
        _require(payload.get("study_b_expected_record_ids") == expected_b, f"Expanded {label} Study B membership changed.")
    for filename, expected_ids, schema in (
        ("study_a_record_summary.json", expected_a, "expanded-studies-study-a-record-collection-v1"),
        ("study_b_record_summary.json", expected_b, "expanded-studies-study-b-record-collection-v1"),
    ):
        summaries = _load_json(_regular_file(archive_dir, filename, filename), filename)
        _require(summaries.get("schema_version") == schema, f"Unexpected {filename} schema.")
        records = summaries.get("records")
        _require(isinstance(records, list) and [record.get("record_id") for record in records] == expected_ids, f"{filename} record membership/order changed.")
    cost_rows = _complete_csv_rows(
        _regular_file(archive_dir, "study_b_cost.csv", "Study B cost CSV"),
        "Study B cost CSV",
        (
            "record_id", "method_id", "cell_id", "data_seed", "model_seed",
            "paired_wall_time_ratio_to_erm", "process_peak_rss_bytes",
            "static_parameter_bytes", "cuda_status",
        ),
        ("paired_wall_time_ratio_to_erm", "process_peak_rss_bytes", "static_parameter_bytes"),
    )
    diagnostic_rows = _complete_csv_rows(
        _regular_file(archive_dir, "study_b_diagnostics.csv", "Study B diagnostics CSV"),
        "Study B diagnostics CSV",
        (
            "record_id", "method_id", "cell_id", "data_seed", "model_seed",
            "diagnostic_status", "target_attainment_fraction", "feasibility_fraction",
            "cap_binding_fraction", "max_absolute_strong_drift_change",
        ),
        (
            "target_attainment_fraction", "feasibility_fraction", "cap_binding_fraction",
            "max_absolute_strong_drift_change",
        ),
    )
    _require([row["record_id"] for row in cost_rows] == expected_b, "Study B cost reporting is incomplete.")
    _require([row["record_id"] for row in diagnostic_rows] == expected_b, "Study B diagnostics are incomplete.")

    draws = _load_json(_regular_file(archive_dir, "bootstrap_draw_manifest.json", "bootstrap draws"), "bootstrap draws")
    _require(
        {key: draws.get(key) for key in ("schema_version", "generator", "seed", "replicates", "draws_generated_once", "draw_call_order", "reuse")}
        == {
            "schema_version": "expanded-studies-bootstrap-draw-manifest-v1",
            "generator": "numpy.random.default_rng",
            "seed": 7301,
            "replicates": 5000,
            "draws_generated_once": True,
            "draw_call_order": ["data_indices", "model_indices"],
            "reuse": "reuse_the_same_data_and_model_index_draws_across_all_cells_architectures_methods_and_endpoints",
        }
        and draws.get("data_index_draw", {}).get("shape") == [5000, 4]
        and draws.get("data_index_draw", {}).get("value_range") == [0, 3]
        and draws.get("model_index_draw", {}).get("shape") == [5000, 8]
        and draws.get("model_index_draw", {}).get("value_range") == [0, 7],
        "Expanded shared bootstrap-draw contract changed.",
    )
    draw_sha = member_signatures["bootstrap_draw_manifest.json"].sha256

    inference_a = _load_json(_regular_file(archive_dir, "study_a_inference.json", "Study A inference"), "Study A inference")
    claims_a = _load_json(_regular_file(archive_dir, "study_a_claims.json", "Study A claims"), "Study A claims")
    _require(inference_a.get("schema_version") == "expanded-studies-study-a-inference-v1" and inference_a.get("method") == "shared_two_way_pigeonhole_bootstrap" and inference_a.get("bootstrap_draw_manifest_sha256") == draw_sha, "Study A inference contract changed.")
    _require(set(inference_a.get("architectures", {})) == {"tanh", "gru"}, "Study A architecture set changed.")
    expected_a_estimates = {
        "tanh": (0.2916666666666667, 0.16666666666666666, 0.4166666666666667),
        "gru": (0.0, 0.0, 0.0),
    }
    derived_a: dict[str, dict[str, bool]] = {}
    for architecture, expected in expected_a_estimates.items():
        result = inference_a.get("architectures", {}).get(architecture, {})
        estimate = result.get("equal_weight_three_cell_macro", {})
        _require((estimate.get("point_estimate"), estimate.get("ci95_low"), estimate.get("ci95_high")) == expected, f"Study A frozen estimate changed for {architecture}.")
        test = result.get("one_sided_test", {})
        _require(
            {key: test.get(key) for key in ("test_id", "alternative", "null_boundary", "observed", "family_alpha", "holm_adjusted_p_value", "holm_rejected")}
            == {
                "test_id": architecture,
                "alternative": "greater",
                "null_boundary": 0.5,
                "observed": expected[0],
                "family_alpha": 0.05,
                "holm_adjusted_p_value": 1.0,
                "holm_rejected": False,
            },
            f"Study A Holm result changed for {architecture}.",
        )
        ci_pass = estimate.get("ci95_low") > 0.5
        holm_pass = test.get("holm_rejected") is True
        derived_a[architecture] = {
            "beta_robust_architecture_claim": ci_pass and holm_pass,
            "holm_adjusted_one_sided_test_rejects": holm_pass,
            "raw_ci95_lower_strictly_greater_than_0.5": ci_pass,
        }
    _require(
        inference_a.get("holm_family")
        == [inference_a["architectures"][architecture]["one_sided_test"] for architecture in ("tanh", "gru")],
        "Study A Holm family does not exactly match architecture tests.",
    )
    _require(claims_a.get("schema_version") == "expanded-studies-study-a-claims-v1", "Unexpected Study A claims schema.")
    _require(claims_a.get("architecture_ranking_performed") is False, "Study A must not perform architecture ranking.")
    _require(claims_a.get("architectures") == derived_a and claims_a.get("overall_claim") is all(item["beta_robust_architecture_claim"] for item in derived_a.values()) is False, "Study A frozen semantic outcome changed.")

    inference_b = _load_json(_regular_file(archive_dir, "study_b_inference.json", "Study B inference"), "Study B inference")
    claims_b = _load_json(_regular_file(archive_dir, "study_b_claims.json", "Study B claims"), "Study B claims")
    _require(inference_b.get("schema_version") == "expanded-studies-study-b-inference-v1" and inference_b.get("method") == "shared_two_way_pigeonhole_bootstrap" and inference_b.get("bootstrap_draw_manifest_sha256") == draw_sha, "Study B inference contract changed.")
    _require(inference_b.get("cost_complete") is True, "Study B cost reporting is incomplete.")
    _require(inference_b.get("diagnostics_complete") == {"bloop": True, "counterfactual_drift": True, "erm": True, "pcgrad": True}, "Study B diagnostics are incomplete.")
    expected_b = {
        "bloop": {
            "weak": (0.002198259399210656, 0.0009779187294930126, 0.003277366267916477),
            "trajectory": (3.4651361294978416, 3.3443827215131754, 3.5849715262780952),
            "final": (0.6422362388111651, 0.6288088704226539, 0.6563168084365315),
            "claim": True,
        },
        "pcgrad": {
            "weak": (-0.00015351238407674823, -0.00047909348592838785, 0.00013248012419808214),
            "trajectory": (-0.48709889128076556, -0.6897378944428417, -0.27379327373499074),
            "final": (-0.2565436437726021, -0.3016841153614223, -0.2087651835754514),
            "claim": False,
        },
    }
    test_families = inference_b.get("test_families", {})
    _require_exact_keys(test_families, {"trajectory_superiority", "final_superiority", "weak_equivalence_tost"}, "Study B test families")
    raw_trajectory_tests = test_families["trajectory_superiority"]
    raw_final_tests = test_families["final_superiority"]
    raw_weak_tests = test_families["weak_equivalence_tost"]
    _require(isinstance(raw_trajectory_tests, list) and len(raw_trajectory_tests) == 2, "Study B trajectory test family is not exact.")
    _require(isinstance(raw_final_tests, list) and len(raw_final_tests) == 2, "Study B final test family is not exact.")
    _require(isinstance(raw_weak_tests, list) and len(raw_weak_tests) == 4, "Study B weak TOST family is not exact.")
    trajectory_tests = {item.get("comparator"): item for item in raw_trajectory_tests if isinstance(item, dict)}
    final_tests = {item.get("comparator"): item for item in raw_final_tests if isinstance(item, dict)}
    weak_tests = {(item.get("comparator"), item.get("boundary")): item for item in raw_weak_tests if isinstance(item, dict)}
    _require(set(trajectory_tests) == set(expected_b) and set(final_tests) == set(expected_b), "Study B superiority test membership is not exact.")
    _require(set(weak_tests) == {(comparator, boundary) for comparator in expected_b for boundary in ("lower", "upper")}, "Study B weak TOST membership is not exact.")
    derived_b: dict[str, Any] = {}
    for comparator, expected in expected_b.items():
        comparison = inference_b.get("comparisons", {}).get(comparator, {})
        _require_frozen_estimate(comparison, "cdc_minus_comparator_weak_rescue", expected["weak"], f"{comparator} weak rescue")
        _require_frozen_estimate(comparison, "comparator_minus_cdc_trajectory_deviation", expected["trajectory"], f"{comparator} trajectory")
        _require_frozen_estimate(comparison, "comparator_minus_cdc_final_deviation", expected["final"], f"{comparator} final")
        expected_holm_rejected = expected["claim"]
        expected_holm_p = 0.0003999200159968006 if comparator == "bloop" else 1.0
        for test, observed, label in (
            (trajectory_tests[comparator], expected["trajectory"][0], "trajectory"),
            (final_tests[comparator], expected["final"][0], "final"),
        ):
            _require(
                {key: test.get(key) for key in ("comparator", "test_id", "alternative", "null_boundary", "observed", "family_alpha", "holm_adjusted_p_value", "holm_rejected")}
                == {
                    "comparator": comparator,
                    "test_id": comparator,
                    "alternative": "greater",
                    "null_boundary": 0.0,
                    "observed": observed,
                    "family_alpha": 0.05,
                    "holm_adjusted_p_value": expected_holm_p,
                    "holm_rejected": expected_holm_rejected,
                },
                f"Study B {label} test is not bound to its frozen estimand for {comparator}.",
            )
        for boundary, alternative, null_boundary in (
            ("lower", "greater", -0.05),
            ("upper", "less", 0.05),
        ):
            test = weak_tests[(comparator, boundary)]
            _require(
                {key: test.get(key) for key in ("comparator", "test_id", "boundary", "alternative", "null_boundary", "observed", "family_alpha", "holm_adjusted_p_value", "holm_rejected")}
                == {
                    "comparator": comparator,
                    "test_id": f"{comparator}:{boundary}",
                    "boundary": boundary,
                    "alternative": alternative,
                    "null_boundary": null_boundary,
                    "observed": expected["weak"][0],
                    "family_alpha": 0.05,
                    "holm_adjusted_p_value": 0.0007998400319936012,
                    "holm_rejected": True,
                },
                f"Study B weak TOST is not bound to its frozen estimand for {comparator}:{boundary}.",
            )
        weak_inside = expected["weak"][1] >= -0.05 and expected["weak"][2] <= 0.05
        weak_holm = all(weak_tests[(comparator, boundary)].get("holm_rejected") is True for boundary in ("lower", "upper"))
        trajectory_positive = expected["trajectory"][1] > 0.0
        final_positive = expected["final"][1] > 0.0
        components = {
            "both_weak_tost_tests_pass_holm": weak_holm,
            "complete_cost_reporting": inference_b["cost_complete"] is True,
            "complete_diagnostics": inference_b["diagnostics_complete"][comparator] is True and inference_b["diagnostics_complete"]["counterfactual_drift"] is True,
            "final_raw_ci_lower_strictly_positive": final_positive,
            "final_test_passes_holm": final_tests[comparator].get("holm_rejected") is True,
            "trajectory_raw_ci_lower_strictly_positive": trajectory_positive,
            "trajectory_test_passes_holm": trajectory_tests[comparator].get("holm_rejected") is True,
            "weak_rescue_equivalence_raw_ci_inside_margin": weak_inside,
        }
        derived_b[comparator] = {"cdc_beats_comparator_joint_tradeoff": all(components.values()), "components": components}
        _require(derived_b[comparator]["cdc_beats_comparator_joint_tradeoff"] is expected["claim"], f"Study B frozen claim changed for {comparator}.")
    _require(claims_b.get("schema_version") == "expanded-studies-study-b-claims-v1" and claims_b.get("comparators") == derived_b, "Study B claims do not equal derived frozen outcomes.")
    _require(claims_b.get("instantaneous_metric_used_as_trajectory_or_final_proof") is False, "Instantaneous metrics cannot prove trajectory/final claims.")

    endpoints = inference_b.get("method_endpoints", {})
    for filename, schema, strong_key, endpoint_key in (
        ("study_b_pareto_trajectory.json", "expanded-studies-study-b-pareto-trajectory-v1", "strong_trajectory_deviation", "trajectory_deviation"),
        ("study_b_pareto_final.json", "expanded-studies-study-b-pareto-final-v1", "strong_final_deviation", "final_deviation"),
    ):
        pareto = _load_json(_regular_file(archive_dir, filename, filename), filename)
        _require(pareto.get("schema_version") == schema and pareto.get("scope") == "equal_weight_two_cell_macro" and pareto.get("superiority_test") is False, f"{filename} cannot substitute for superiority tests.")
        expected_strong = {method: values[endpoint_key]["equal_weight_two_cell_macro"]["point_estimate"] for method, values in endpoints.items()}
        expected_weak = {method: values["weak_rescue"]["equal_weight_two_cell_macro"]["point_estimate"] for method, values in endpoints.items()}
        _require(pareto.get(strong_key) == expected_strong and pareto.get("weak_rescue") == expected_weak, f"{filename} endpoints disagree with inference.")


def verify_completed_run_archive(
    repo_root: Path,
    archive_root: str,
    *,
    expected_study_id: str,
    source_repo_root: Path | None = None,
) -> CompletedArchiveVerification:
    """Verify one completed-run compact archive without applying ENL-NTK semantics."""
    repo_root = repo_root.resolve(strict=True)
    source_repo_root = (source_repo_root or repo_root).resolve(strict=True)
    _require(expected_study_id in COMPLETED_ARCHIVE_CONTRACTS, f"Unknown completed-run study: {expected_study_id!r}.")
    manifest, members, _artifact, archive_dir, source_sha256, source_count = _verify_completed_archive_structure(
        repo_root, source_repo_root, archive_root, expected_study_id
    )
    if expected_study_id == "semi-real-generated-cue-v1":
        _verify_semi_real_semantics(archive_dir, members)
    else:
        _verify_expanded_semantics(archive_dir, members)
    contract = COMPLETED_ARCHIVE_CONTRACTS[expected_study_id]
    return CompletedArchiveVerification(
        archive_root=archive_root,
        study_id=expected_study_id,
        claim_ids=contract["claim_ids"],
        member_signatures=members,
        archived_file_bytes=manifest["archived_file_bytes"],
        source_sha256=source_sha256,
        source_file_count=source_count,
        declared_file_count=contract["declared_file_count"],
        omitted_file_count=contract["omitted_file_count"],
    )


def _verify_git_tracking(repo_root: Path, relative_paths: set[str]) -> None:
    try:
        top_level = subprocess.run(
            [GIT_EXECUTABLE, "-C", str(repo_root), "rev-parse", "--show-toplevel"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise VerificationError("Git tracking was required but no Git worktree was available.") from error
    _require(
        Path(top_level).resolve() == repo_root.resolve(),
        "--repo-root is not the Git worktree root.",
    )
    try:
        result = subprocess.run(
            [GIT_EXECUTABLE, "-C", str(repo_root), "ls-files", "-z", "--cached", "--", *sorted(relative_paths)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise VerificationError("Could not query Git-tracked release files.") from error
    tracked = {
        value.decode("utf-8")
        for value in result.stdout.split(b"\0")
        if value
    }
    missing = sorted(relative_paths - tracked)
    _require(not missing, f"Release files are not Git tracked: {missing}.")


def verify_release_artifacts(
    repo_root: Path,
    provenance_path: str = "paper/artifacts/provenance_manifest.json",
    *,
    require_git_tracked: bool = False,
    source_repo_root: Path | None = None,
) -> ReleaseVerification:
    """Verify all compact studies indexed by the central provenance manifest."""
    repo_root = repo_root.resolve(strict=True)
    source_repo_root = (source_repo_root or repo_root).resolve(strict=True)
    provenance_file = _regular_file(repo_root, provenance_path, "provenance manifest")
    provenance = _load_json(provenance_file, "provenance manifest")
    _require(provenance.get("schema_version") == 3, "Unsupported provenance schema.")
    _require(provenance.get("schema_v3_update_timestamp") is None, "Schema-v3 update must not invent a wall-clock timestamp.")
    schema_v3_note = provenance.get("schema_v3_update_note")
    _require(
        isinstance(schema_v3_note, str)
        and "two already-created completed-run compact archives" in schema_v3_note
        and "no wall-clock update time is asserted" in schema_v3_note,
        "Schema-v3 update note is missing or untruthful.",
    )
    _require(
        provenance.get("interpretation") == CANONICAL_PROVENANCE_INTERPRETATION,
        "Central provenance interpretation is stale or incomplete; source reconstructibility, replay completeness, and scientific-validity limits must use the canonical wording.",
    )
    studies = provenance.get("frozen_empirical_ntk_studies")
    _require(isinstance(studies, list) and studies, "No frozen empirical-NTK studies declared.")
    completed_studies = provenance.get("completed_run_archives")
    _require(
        isinstance(completed_studies, list) and len(completed_studies) == 2,
        "Central provenance must declare exactly two completed-run archives.",
    )

    verified_signatures: dict[str, FileSignature] = {
        provenance_path: _file_signature(provenance_file)
    }
    git_paths: set[str] = {provenance_path, *RELEASE_CONTRACT_PATHS}
    archive_roots: set[str] = set()
    claim_ids: set[str] = set()
    historical_count = 0

    for study_index, raw_study in enumerate(studies):
        _require(isinstance(raw_study, dict), f"Study {study_index} must be a mapping.")
        provenance_class = validate_source_provenance(raw_study, repo_root=repo_root)
        historical_count += provenance_class == "historical_dirty_unreconstructible"
        claim_id = raw_study.get("claim_id")
        _require(isinstance(claim_id, str) and claim_id, f"Study {study_index} has no claim_id.")
        _require(claim_id not in claim_ids, f"Duplicate study claim_id: {claim_id!r}.")
        claim_ids.add(claim_id)
        archive_root = raw_study.get("compact_archive_root")
        _relative_parts(archive_root, f"study {study_index} compact_archive_root")
        _require(archive_root not in archive_roots, f"Duplicate compact archive root: {archive_root!r}.")
        archive_roots.add(archive_root)
        frozen_contract = FROZEN_STUDY_CONTRACTS.get(claim_id)
        _require(frozen_contract is not None, f"Unknown frozen study claim_id: {claim_id!r}.")
        expected_primary_outputs = frozen_contract["accepted_primary_outputs"]
        _require(
            raw_study.get("accepted_primary_outputs") == list(expected_primary_outputs),
            f"Study {claim_id}.accepted_primary_outputs is not canonical.",
        )
        acceptance_contract = frozen_contract["acceptance_contract"]
        _require(
            raw_study.get("acceptance_contract") == acceptance_contract,
            f"Study {claim_id}.acceptance_contract is not canonical.",
        )
        expected_evaluation_passed = frozen_contract["evaluation_passed"]
        _require(
            raw_study.get("evaluation_passed") is expected_evaluation_passed,
            f"Study {claim_id}.evaluation_passed is not canonical.",
        )

        archive = verify_archive_manifest(
            repo_root,
            archive_root,
            expected_primary_outputs=expected_primary_outputs,
            expected_records_by_model=acceptance_contract["records_by_model"],
            expected_thresholds=acceptance_contract["required_thresholds"],
            expected_evaluation_passed=expected_evaluation_passed,
        )
        _require(
            archive.accepted_primary_outputs == expected_primary_outputs,
            f"Archive primary outputs disagree with frozen contract for {archive_root}.",
        )
        _require(
            raw_study.get("manifest_sha256") == archive.preflight_manifest_sha256,
            f"Central manifest digest mismatch for {archive_root}.",
        )
        _require(
            raw_study.get("prediction_sha256") == archive.prediction_sha256,
            f"Central prediction digest mismatch for {archive_root}.",
        )
        _require(
            raw_study.get("source_sha256") == archive.source_sha256,
            f"Central source digest mismatch for {archive_root}.",
        )
        _require(
            raw_study.get("verified_frozen_file_count")
            == archive.verified_frozen_file_count,
            f"Central verified-file count mismatch for {archive_root}.",
        )
        verdict = raw_study.get("verdict")
        _require(isinstance(verdict, str) and verdict, f"Study {study_index} has no verdict.")
        if archive.evaluation_passed:
            _require(verdict.startswith("passed"), f"Passing archive has inconsistent verdict: {verdict!r}.")
        else:
            _require(verdict.startswith("failed"), f"Failed archive has inconsistent verdict: {verdict!r}.")

        central_entries = raw_study.get("archived_files")
        _require(isinstance(central_entries, list), f"Study {study_index}.archived_files must be a list.")
        central_signatures: dict[str, FileSignature] = {}
        archive_prefix = f"{archive_root}/"
        for entry_index, raw_entry in enumerate(central_entries):
            _require(
                isinstance(raw_entry, dict),
                f"Study {study_index} archive entry {entry_index} must be a mapping.",
            )
            raw_path = raw_entry.get("path")
            _relative_parts(raw_path, f"study {study_index} archived path")
            _require(
                raw_path.startswith(archive_prefix),
                f"Central archive path escapes declared root: {raw_path!r}.",
            )
            _require(raw_path not in central_signatures, f"Duplicate central archive path: {raw_path!r}.")
            expected = _declared_signature(
                raw_entry, f"study {study_index} archive entry {entry_index}"
            )
            actual = _verify_signature(
                _regular_file(repo_root, raw_path, "central archive member"),
                expected,
                raw_path,
            )
            central_signatures[raw_path] = actual
            if raw_path in verified_signatures:
                _require(
                    verified_signatures[raw_path] == actual,
                    f"Conflicting duplicate release file declaration: {raw_path!r}.",
                )
            verified_signatures[raw_path] = actual
            git_paths.add(raw_path)

        expected_central_paths = {f"{archive_root}/archive_manifest.json"} | {
            f"{archive_root}/{relative_path}"
            for relative_path in archive.member_signatures
        }
        _require(
            set(central_signatures) == expected_central_paths,
            f"Central archive index is not exact for {archive_root}: "
            f"missing={sorted(expected_central_paths - set(central_signatures))}, "
            f"extra={sorted(set(central_signatures) - expected_central_paths)}.",
        )
        for relative_path, signature in archive.member_signatures.items():
            central_path = f"{archive_root}/{relative_path}"
            _require(
                central_signatures[central_path] == signature,
                f"Central and local archive signatures disagree for {central_path}.",
            )

    _require(
        claim_ids == set(FROZEN_STUDY_CONTRACTS),
        "Central provenance must contain exactly the frozen E19/E20 study contracts.",
    )

    completed_study_ids: set[str] = set()
    completed_claim_ids: set[str] = set()
    for study_index, raw_study in enumerate(completed_studies):
        _require(isinstance(raw_study, dict), f"Completed study {study_index} must be a mapping.")
        study_id = raw_study.get("study_id")
        _require(study_id in COMPLETED_ARCHIVE_CONTRACTS, f"Unknown completed-run study_id: {study_id!r}.")
        _require(study_id not in completed_study_ids, f"Duplicate completed-run study_id: {study_id!r}.")
        completed_study_ids.add(study_id)
        contract = COMPLETED_ARCHIVE_CONTRACTS[study_id]
        raw_claims = raw_study.get("claim_ids")
        _require(raw_claims == list(contract["claim_ids"]), f"Completed study {study_id} claim membership is not canonical.")
        _require(len(raw_claims) == len(set(raw_claims)), f"Duplicate claim membership within {study_id}.")
        overlap = completed_claim_ids.intersection(raw_claims)
        _require(not overlap, f"Duplicate completed-run claim membership: {sorted(overlap)}.")
        _require(not claim_ids.intersection(raw_claims), f"Claim appears in historical and completed collections: {raw_claims}.")
        completed_claim_ids.update(raw_claims)
        _require(raw_study.get("archive_type") == COMPLETED_ARCHIVE_SCHEMA, f"Completed study {study_id} archive type mismatch.")
        archive_root = raw_study.get("compact_archive_root")
        _require(archive_root == contract["archive_root"], f"Completed study {study_id} archive root is not canonical.")
        _relative_parts(archive_root, f"completed study {study_index} compact_archive_root")
        _require(archive_root not in archive_roots, f"Duplicate compact archive root: {archive_root!r}.")
        archive_roots.add(archive_root)
        _require(
            {key: raw_study.get(key) for key in ("source_provenance_class", "git_commit", "git_dirty", "source_sha256", "source_file_count", "source_snapshot_archived", "source_reconstructible")}
            == {
                "source_provenance_class": "clean_git_source",
                "git_commit": COMPLETED_SOURCE_COMMIT,
                "git_dirty": False,
                "source_sha256": contract["source_sha256"],
                "source_file_count": contract["source_file_count"],
                "source_snapshot_archived": False,
                "source_reconstructible": True,
            },
            f"Completed study {study_id} clean-source declaration is not canonical.",
        )
        _require(raw_study.get("declared_file_count") == contract["declared_file_count"], f"Completed study {study_id} full file count changed.")
        _require(raw_study.get("omitted_file_count") == contract["omitted_file_count"], f"Completed study {study_id} omitted count changed.")
        limitation = raw_study.get("limitation")
        _require(
            isinstance(limitation, str)
            and "omitted bytes cannot be recovered from hashes" in limitation
            and "not a self-contained replay package" in limitation,
            f"Completed study {study_id} limitation is incomplete.",
        )

        archive = verify_completed_run_archive(
            repo_root,
            archive_root,
            expected_study_id=study_id,
            source_repo_root=source_repo_root,
        )
        _require(archive.claim_ids == contract["claim_ids"], f"Completed archive claims disagree for {study_id}.")
        _require((archive.source_sha256, archive.source_file_count) == (contract["source_sha256"], contract["source_file_count"]), f"Completed archive source identity disagrees for {study_id}.")
        _require((archive.declared_file_count, archive.omitted_file_count) == (contract["declared_file_count"], contract["omitted_file_count"]), f"Completed archive file commitment disagrees for {study_id}.")

        central_entries = raw_study.get("archived_files")
        _require(isinstance(central_entries, list), f"Completed study {study_id}.archived_files must be a list.")
        central_signatures: dict[str, FileSignature] = {}
        archive_prefix = f"{archive_root}/"
        for entry_index, raw_entry in enumerate(central_entries):
            _require(isinstance(raw_entry, dict), f"Completed study {study_id} archive entry {entry_index} must be a mapping.")
            raw_path = raw_entry.get("path")
            _relative_parts(raw_path, f"completed study {study_id} archived path")
            _require(raw_path.startswith(archive_prefix), f"Completed central path escapes declared root: {raw_path!r}.")
            _require(raw_path not in central_signatures, f"Duplicate completed central archive path: {raw_path!r}.")
            expected = _declared_signature(raw_entry, f"completed study {study_id} archive entry {entry_index}")
            actual = _verify_signature(_regular_file(repo_root, raw_path, "completed central archive member"), expected, raw_path)
            central_signatures[raw_path] = actual
            _require(raw_path not in verified_signatures, f"Conflicting duplicate release file declaration: {raw_path!r}.")
            verified_signatures[raw_path] = actual
            git_paths.add(raw_path)
        expected_central_paths = {f"{archive_root}/archive_manifest.json"} | {
            f"{archive_root}/{relative_path}" for relative_path in archive.member_signatures
        }
        _require(
            set(central_signatures) == expected_central_paths,
            f"Completed central archive index is not exact for {archive_root}: "
            f"missing={sorted(expected_central_paths - set(central_signatures))}, "
            f"extra={sorted(set(central_signatures) - expected_central_paths)}.",
        )
        for relative_path, signature in archive.member_signatures.items():
            central_path = f"{archive_root}/{relative_path}"
            _require(central_signatures[central_path] == signature, f"Completed central/local signatures disagree for {central_path}.")

    _require(completed_study_ids == set(COMPLETED_ARCHIVE_CONTRACTS), "Central provenance must contain exactly the two completed-run studies.")
    _require(completed_claim_ids == {"E26", "E27", "E28"}, "Central provenance must contain exactly completed claims E26/E27/E28.")
    if require_git_tracked:
        _verify_git_tracking(repo_root, git_paths)

    total_archives = len(studies) + len(completed_studies)
    total_studies = len(claim_ids) + len(completed_claim_ids)
    return ReleaseVerification(
        provenance_path=provenance_path,
        study_count=total_studies,
        archive_count=total_archives,
        file_count=len(verified_signatures),
        byte_count=sum(signature.bytes for signature in verified_signatures.values()),
        historical_unreconstructible_study_count=historical_count,
        git_tracking_checked=require_git_tracked,
    )


def build_parser() -> argparse.ArgumentParser:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--provenance",
        default="paper/artifacts/provenance_manifest.json",
        help="Repository-relative central provenance manifest path.",
    )
    parser.add_argument(
        "--require-git-tracked",
        action="store_true",
        help="Also require the verifier, central manifest, and every indexed file to be tracked by Git.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_release_artifacts(
            args.repo_root,
            args.provenance,
            require_git_tracked=args.require_git_tracked,
        )
    except (OSError, VerificationError) as error:
        print(f"release artifact verification failed: {error}", file=sys.stderr)
        return 1

    print(
        "release artifact integrity verified: "
        f"archives={report.archive_count}, studies={report.study_count}, "
        f"files={report.file_count}, bytes={report.byte_count}, "
        f"git_tracking_checked={report.git_tracking_checked}"
    )
    if report.historical_unreconstructible_study_count:
        print(
            "disclosed limitation: "
            f"{report.historical_unreconstructible_study_count} study/studies have "
            "historical_dirty_unreconstructible source; integrity verification does "
            "not make them replayable"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
