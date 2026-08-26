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
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
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
class ReleaseVerification:
    provenance_path: str
    study_count: int
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
            ["git", "-C", str(repo_root), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        listing = subprocess.run(
            [
                "git",
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
                    "git",
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


def _verify_git_tracking(repo_root: Path, relative_paths: set[str]) -> None:
    try:
        top_level = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
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
            ["git", "-C", str(repo_root), "ls-files", "-z", "--cached", "--", *sorted(relative_paths)],
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
) -> ReleaseVerification:
    """Verify all compact studies indexed by the central provenance manifest."""
    repo_root = repo_root.resolve(strict=True)
    provenance_file = _regular_file(repo_root, provenance_path, "provenance manifest")
    provenance = _load_json(provenance_file, "provenance manifest")
    _require(provenance.get("schema_version") == 2, "Unsupported provenance schema.")
    studies = provenance.get("frozen_empirical_ntk_studies")
    _require(isinstance(studies, list) and studies, "No frozen empirical-NTK studies declared.")

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
    if require_git_tracked:
        _verify_git_tracking(repo_root, git_paths)

    return ReleaseVerification(
        provenance_path=provenance_path,
        study_count=len(studies),
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
        f"studies={report.study_count}, files={report.file_count}, "
        f"bytes={report.byte_count}, git_tracking_checked={report.git_tracking_checked}"
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
