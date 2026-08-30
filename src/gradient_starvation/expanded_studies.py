"""Auditable revision-3 execution for the frozen expanded synthetic studies.

The contract deliberately separates initialization-only preflight, external human
authorization, isolated scientific workers, and deterministic analysis.  This
module never creates an authorization seal.  A scientific model or optimizer is
unreachable until the supplied seal and every bound preflight byte have passed
verification.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import resource
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from .data.synthetic import SyntheticBatch, SyntheticTaskSpec, make_paired_task
from .metrics import n_sign_changes, sign_crossing_time
from .study_metrics import (
    SHADOW_METHOD_PUBLICATION_LABELS,
    first_hitting_profile,
    pareto_frontier,
)
from .theory import discrete_crossover_certificate
from .training import initialize_paired_models, train_paired


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_RELATIVE_ROOT = Path("data/expanded_studies_execution_v1")
CONTRACT_SEMANTIC_SHA256 = (
    "0565e4cceb4a029e8206aa74f289082363a6b55e94bda1cd778c2f6e20fd51f5"
)
HISTORICAL_SHA256 = (
    "481508ee5381caf6a279d7c992e8e77e5f04679f013309f35c1b5b1a023ce439"
)
ENVIRONMENT_LOCK_SHA256 = (
    "651ba112e67b38f80eb60d26ef92a348b3e3228fbb0a7aae8a0f24930b823c1c"
)
SHA256_LENGTH = 64
_WORKER_SCRIPT_RELATIVE_PATH = "scripts/run_expanded_studies.py"
_WORKER_LAUNCH_CONTEXT_ENV = "GRADIENT_STARVATION_EXPANDED_WORKER_CONTEXT"
_WORKER_LAUNCH_CONTEXT_SCHEMA = "expanded-studies-worker-launch-context-v1"
_WORKER_LAUNCH_EVIDENCE_SCHEMA = "expanded-studies-worker-launch-evidence-v1"
_WORKER_WALL_TIME_SCOPE = (
    "worker_entry_through_sample_after_trajectory_and_summary_serialization_"
    "before_preliminary_resources_write"
)
_PARENT_WALL_TIME_SCOPE = (
    "parent_prelaunch_through_sample_after_worker_exit_and_bundle_validation_"
    "before_final_resources_and_complete_marker_writes"
)

_PREFLIGHT_FILES = [
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
_RUN_FILES = [
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
_RECORD_FILES = ["trajectory.csv", "summary.json", "resources.json", "complete.json"]
_NONFINAL_RECORD_FILES = ["trajectory.csv", "summary.json", "resources.json"]

_METADATA_FIELDS = [
    "record_id",
    "study",
    "model_kind",
    "cell_id",
    "method_id",
    "data_seed",
    "model_seed",
    "condition",
]
_NUMERIC_TRAJECTORY_FIELDS = [
    "step",
    "tau",
    "loss",
    "accuracy",
    "m_s",
    "m_w",
    "drift_s",
    "drift_w",
    "direct_drift_s",
    "direct_drift_w",
    "projected_drift_residual_s",
    "projected_drift_residual_w",
    "mode_residual_rms",
    "projected_identity_absolute_error",
    "projected_identity_relative_error",
    "g_s",
    "g_w",
    "A_ss",
    "A_sw",
    "A_ww",
    "G_ss",
    "G_sw",
    "G_ww",
    "chi_w_from_s",
    "margin_mean",
    "margin_std",
    "margin_q10",
    "margin_q50",
    "margin_q90",
    "gsi5",
    "d_w_matched",
    "t_geom",
    "s_ce",
    "t_geom_self",
    "t_geom_cross",
    "decomposition_reconstruction_error",
    "matched_weak_only_m_w",
    "d_w_equal_time",
    "d_w_equal_time_projected",
    "d_w_equal_time_exact",
    "equal_time_projection_residual",
    "equal_time_cross_transport",
    "equal_time_geometry_a",
    "equal_time_field_a",
    "equal_time_geometry_b",
    "equal_time_field_b",
    "equal_time_reconstruction_error_a",
    "equal_time_reconstruction_error_b",
    "cdc_target_weak_drift",
    "cdc_weak_drift_before",
    "cdc_weak_drift_after",
    "cdc_strong_drift_before",
    "cdc_strong_drift_after",
    "cdc_deficit",
    "cdc_alpha",
    "cdc_uncapped_alpha",
    "cdc_target_residual",
    "cdc_protected_norm_sq",
    "cdc_correction_norm",
    "cdc_strong_drift_change",
]
_BOOLEAN_TRAJECTORY_FIELDS = [
    "equal_time_dominance_ordering_invariant",
    "cdc_cap_binding",
    "cdc_feasible",
    "cdc_target_met",
]
TRAJECTORY_FIELDS = [
    *_METADATA_FIELDS,
    *_NUMERIC_TRAJECTORY_FIELDS,
    *_BOOLEAN_TRAJECTORY_FIELDS,
    "nonfinite_fields",
]
_BASE_NUMERIC_FIELDS = frozenset(
    {
        "step",
        "tau",
        "loss",
        "accuracy",
        "m_s",
        "m_w",
        "drift_s",
        "drift_w",
        "mode_residual_rms",
        "g_s",
        "g_w",
        "A_ss",
        "A_sw",
        "A_ww",
        "G_ss",
        "G_sw",
        "G_ww",
        "chi_w_from_s",
        "margin_mean",
        "margin_std",
        "margin_q10",
        "margin_q50",
        "margin_q90",
        "gsi5",
    }
)
_DIRECT_NUMERIC_FIELDS = frozenset(
    {
        "direct_drift_s",
        "direct_drift_w",
        "projected_drift_residual_s",
        "projected_drift_residual_w",
        "projected_identity_absolute_error",
        "projected_identity_relative_error",
    }
)
_LOCKSTEP_PROJECTED_NUMERIC_FIELDS = frozenset(
    {
        "d_w_matched",
        "t_geom",
        "s_ce",
        "t_geom_self",
        "t_geom_cross",
        "decomposition_reconstruction_error",
        "matched_weak_only_m_w",
        "d_w_equal_time",
        "d_w_equal_time_projected",
        "equal_time_cross_transport",
        "equal_time_geometry_a",
        "equal_time_field_a",
        "equal_time_geometry_b",
        "equal_time_field_b",
        "equal_time_reconstruction_error_a",
        "equal_time_reconstruction_error_b",
    }
)
_LOCKSTEP_DIRECT_NUMERIC_FIELDS = frozenset(
    {"d_w_equal_time_exact", "equal_time_projection_residual"}
)
_SHADOW_NUMERIC_FIELDS = frozenset(
    {
        "cdc_target_weak_drift",
        "cdc_weak_drift_before",
        "cdc_weak_drift_after",
        "cdc_strong_drift_before",
        "cdc_strong_drift_after",
        "cdc_deficit",
        "cdc_alpha",
        "cdc_uncapped_alpha",
        "cdc_target_residual",
        "cdc_protected_norm_sq",
        "cdc_correction_norm",
        "cdc_strong_drift_change",
    }
)
_LOCKSTEP_BOOLEAN_FIELDS = frozenset(
    {"equal_time_dominance_ordering_invariant"}
)
_SHADOW_BOOLEAN_FIELDS = frozenset(
    {"cdc_cap_binding", "cdc_feasible", "cdc_target_met"}
)
_STUDY_B_SHADOW_METHODS = frozenset(
    {"counterfactual_drift", "bloop", "pcgrad"}
)


class ExpandedStudiesError(RuntimeError):
    """Fail-closed expanded-study preflight, execution, or analysis error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExpandedStudiesError(message)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExpandedStudiesError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ExpandedStudiesError(f"Non-finite JSON value: {token}.")
            ),
        )
    except ExpandedStudiesError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExpandedStudiesError(f"Cannot parse {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} must contain one JSON mapping.")
    return value


def _strict_json_text(text: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ExpandedStudiesError(f"Non-finite JSON value: {token}.")
            ),
        )
    except ExpandedStudiesError:
        raise
    except (TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise ExpandedStudiesError(f"Cannot parse {label}: {error}") from error
    _require(isinstance(value, dict), f"{label} must contain one JSON mapping.")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        _require(math.isfinite(number), "Non-finite JSON values are forbidden.")
        return number
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            _json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ExpandedStudiesError(f"Value is not strict finite JSON: {error}") from error
    return (text + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ExpandedStudiesError(f"Cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _descriptor(path: Path, *, name: str | None = None) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"Missing or symlinked file: {path}")
    return {
        "path": name or path.name,
        "size": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _verify_descriptor(directory: Path, descriptor: Mapping[str, Any]) -> Path:
    _require(
        set(descriptor) == {"path", "size", "sha256"},
        "File descriptor schema changed.",
    )
    path = _safe_member(directory, descriptor["path"], "manifest member")
    _require(path.is_file() and not path.is_symlink(), f"Referenced file is missing: {path}")
    _require(type(descriptor["size"]) is int and descriptor["size"] >= 0, "Invalid file size.")
    _require(_is_sha256(descriptor["sha256"]), "Invalid file SHA-256.")
    _require(path.stat().st_size == descriptor["size"], f"File size drift: {path.name}")
    _require(_sha256_file(path) == descriptor["sha256"], f"File hash drift: {path.name}")
    return path


def _worker_executable_binding(
    source_manifest: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    """Resolve the only production worker from the sealed source manifest."""
    _require(
        set(source_manifest)
        == {"schema_version", "hash_algorithm", "files", "aggregate_sha256"}
        and source_manifest["schema_version"]
        == "expanded-studies-source-manifest-v1"
        and source_manifest["hash_algorithm"] == "sha256"
        and isinstance(source_manifest["files"], list),
        "Source manifest schema changed.",
    )
    matches = [
        item
        for item in source_manifest["files"]
        if isinstance(item, Mapping)
        and item.get("path") == _WORKER_SCRIPT_RELATIVE_PATH
    ]
    _require(
        len(matches) == 1,
        "Sealed source manifest must contain exactly one expanded-study worker script.",
    )
    descriptor = dict(matches[0])
    script = _verify_descriptor(root, descriptor)
    expected_script = root / _WORKER_SCRIPT_RELATIVE_PATH
    _require(
        script == expected_script
        and _descriptor(script, name=_WORKER_SCRIPT_RELATIVE_PATH) == descriptor,
        "Current worker script descriptor differs from the sealed source manifest.",
    )
    return {
        "interpreter_path": sys.executable,
        "script_path": str(script),
        "script_descriptor": descriptor,
    }


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_or_verify(path: Path, value: bytes) -> None:
    if path.exists():
        _require(path.is_file() and not path.is_symlink(), f"Unsafe existing artifact: {path}")
        _require(path.read_bytes() == value, f"Existing artifact drift: {path}")
    else:
        _atomic_write(path, value)


def _write_sidecar(manifest_path: Path) -> None:
    _atomic_write(
        manifest_path.with_suffix(".sha256"),
        f"{_sha256_file(manifest_path)}  {manifest_path.name}\n".encode("ascii"),
    )


def _verify_sidecar(manifest_path: Path) -> None:
    _require(
        manifest_path.is_file() and not manifest_path.is_symlink(),
        f"Manifest is missing or symlinked: {manifest_path}",
    )
    sidecar = manifest_path.with_suffix(".sha256")
    _require(sidecar.is_file() and not sidecar.is_symlink(), f"Missing sidecar: {sidecar}")
    expected = f"{_sha256_file(manifest_path)}  {manifest_path.name}\n".encode("ascii")
    _require(sidecar.read_bytes() == expected, f"Manifest sidecar drift: {sidecar}")


def _relative_parts(raw: Any, label: str) -> tuple[str, ...]:
    _require(isinstance(raw, str) and raw, f"{label} must be a nonempty string.")
    _require("\\" not in raw and "\x00" not in raw, f"Unsafe {label}: {raw!r}.")
    path = PurePosixPath(raw)
    _require(
        not path.is_absolute()
        and path.as_posix() == raw
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"Unsafe {label}: {raw!r}.",
    )
    return path.parts


def _safe_member(root: Path, raw: Any, label: str) -> Path:
    candidate = root.joinpath(*_relative_parts(raw, label))
    cursor = root
    for part in _relative_parts(raw, label):
        cursor /= part
        _require(not cursor.is_symlink(), f"Symlinked {label} is forbidden: {raw!r}.")
    return candidate


def _parse_exact_lock(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ExpandedStudiesError(f"Cannot read environment lock {path}: {error}") from error
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        _require(
            line.count("==") == 1 and not any(token in line for token in (";", " @ ", "[", "]")),
            f"Environment lock line {line_number} is not one exact name==version pin.",
        )
        name, version = line.split("==", 1)
        normalized = name.strip().lower().replace("_", "-")
        _require(normalized and version.strip(), f"Invalid lock line {line_number}.")
        _require(normalized not in packages, f"Duplicate locked package: {normalized}.")
        packages[normalized] = version.strip()
    _require(bool(packages), "Environment lock has no exact package pins.")
    return packages


def _load_contract(
    path: Path, *, repository_root: Path = REPOSITORY_ROOT
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    _require(path.is_file() and not path.is_symlink(), f"Missing or symlinked contract: {path}")
    contract = _strict_json(path)
    _require(
        contract.get("schema_version") == "expanded-studies-execution-v1"
        and contract.get("study_id") == "expanded-nonlinear-beta-cdc-tradeoff-v1"
        and contract.get("contract_revision") == 3
        and contract.get("status") == "external_authorization_required"
        and contract.get("execution_authorized") is False,
        "Unsupported or authorization-mutated expanded-study contract.",
    )
    observed = _sha256_bytes(_canonical_json_bytes(contract))
    _require(
        observed == CONTRACT_SEMANTIC_SHA256,
        f"Expanded-study contract semantic SHA-256 drifted: {observed}.",
    )
    _require(
        contract["local_workspace"]["root"] == WORKSPACE_RELATIVE_ROOT.as_posix(),
        "Expanded-study workspace root changed.",
    )
    _require(
        contract["cdc_tradeoff_study"]["cost"]["wall_time_scope"]
        == _PARENT_WALL_TIME_SCOPE,
        "Study-B contract wall-time scope differs from the finalized cost telemetry.",
    )
    historical = root / contract["historical_contract"]["path"]
    lock = root / contract["execution_environment"]["environment_lock_path"]
    _require(historical.is_file() and not historical.is_symlink(), "Historical contract is missing or symlinked.")
    _require(lock.is_file() and not lock.is_symlink(), "Environment lock is missing or symlinked.")
    _require(
        contract["historical_contract"]["sha256"]
        == _sha256_file(historical)
        == HISTORICAL_SHA256,
        "Historical blocked contract SHA-256 drifted.",
    )
    historical_value = _strict_json(historical)
    _require(
        historical_value.get("schema_version") == contract["historical_contract"]["schema_version"]
        and historical_value.get("status") == "design_frozen_execution_blocked"
        and historical_value.get("execution_authorized") is False,
        "Historical preregistration is no longer blocked and unauthorized.",
    )
    _require(
        contract["execution_environment"]["environment_lock_sha256"]
        == _sha256_file(lock)
        == ENVIRONMENT_LOCK_SHA256,
        "Environment lock SHA-256 drifted.",
    )
    pins = _parse_exact_lock(lock)
    for package, version in contract["execution_environment"]["required_exact_packages"].items():
        _require(pins.get(package) == version, f"Environment lock must pin {package}=={version}.")
    _require(contract["preflight"]["required_files"] == _PREFLIGHT_FILES, "Preflight inventory changed.")
    _require(contract["artifacts"]["required_final_files"] == _RUN_FILES, "Run inventory changed.")
    _require(
        contract["artifacts"]["study_a_required_record_files"] == _RECORD_FILES
        and contract["artifacts"]["study_b_required_method_record_files"] == _RECORD_FILES,
        "Record artifact protocol changed.",
    )
    return contract


def _expected_record_ids(contract: Mapping[str, Any]) -> dict[str, list[str]]:
    seeds = contract["seed_design"]
    design = contract["scientific_record_design"]
    study_a = contract["nonlinear_beta_study"]
    study_b = contract["cdc_tradeoff_study"]
    a = [
        design["study_a"]["record_id_format"].format(
            model_kind=kind,
            cell_id=cell,
            data_seed=data_seed,
            model_seed=model_seed,
        )
        for kind in study_a["model_kinds"]
        for cell in study_a["cells"]
        for data_seed in seeds["data_seeds"]
        for model_seed in seeds["model_seeds"]
    ]
    b = [
        design["study_b"]["record_id_format"].format(
            model_kind=kind,
            cell_id=cell,
            method_id=method["method_id"],
            data_seed=data_seed,
            model_seed=model_seed,
        )
        for kind in study_b["model_kinds"]
        for cell in study_b["cells"]
        for method in study_b["methods"]
        for data_seed in seeds["data_seeds"]
        for model_seed in seeds["model_seeds"]
    ]
    _require(len(a) == len(set(a)) == 192, "Study A must contain exactly 192 unique IDs.")
    _require(len(b) == len(set(b)) == 256, "Study B must contain exactly 256 unique IDs.")
    _require(set(a).isdisjoint(b), "Study record IDs overlap.")
    return {"study_a": a, "study_b": b}


def _record_coordinates(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    ids = _expected_record_ids(contract)
    seeds = contract["seed_design"]
    study_a = contract["nonlinear_beta_study"]
    study_b = contract["cdc_tradeoff_study"]
    result: dict[str, dict[str, Any]] = {}
    index = 0
    for kind in study_a["model_kinds"]:
        for cell in study_a["cells"]:
            for data_seed in seeds["data_seeds"]:
                for model_seed in seeds["model_seeds"]:
                    result[ids["study_a"][index]] = {
                        "record_id": ids["study_a"][index],
                        "study": "study_a",
                        "model_kind": kind,
                        "cell_id": cell,
                        "method_id": None,
                        "data_seed": int(data_seed),
                        "model_seed": int(model_seed),
                    }
                    index += 1
    index = 0
    for kind in study_b["model_kinds"]:
        for cell in study_b["cells"]:
            for method in study_b["methods"]:
                for data_seed in seeds["data_seeds"]:
                    for model_seed in seeds["model_seeds"]:
                        result[ids["study_b"][index]] = {
                            "record_id": ids["study_b"][index],
                            "study": "study_b",
                            "model_kind": kind,
                            "cell_id": cell,
                            "method_id": method["method_id"],
                            "data_seed": int(data_seed),
                            "model_seed": int(model_seed),
                        }
                        index += 1
    _require(len(result) == 448, "Record coordinate map is incomplete.")
    return result


def _git_executable() -> str:
    executable = shutil.which("git")
    if executable is None:
        for candidate in (
            Path("/usr/bin/git"),
            Path("/opt/homebrew/bin/git"),
            Path("/Library/Developer/CommandLineTools/usr/bin/git"),
        ):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                executable = str(candidate)
                break
    _require(executable is not None, "git is required for auditable execution.")
    return str(executable)


def _git_output(root: Path, arguments: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            [_git_executable(), *arguments],
            cwd=root,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExpandedStudiesError(f"Cannot inspect git state: {error}") from error


def _git_state(root: Path) -> dict[str, Any]:
    commit = _git_output(root, ["rev-parse", "HEAD"])
    status = _git_output(root, ["status", "--porcelain", "--untracked-files=all"])
    _require(
        len(commit) == 40 and all(character in "0123456789abcdef" for character in commit),
        "Git HEAD is not one lowercase 40-character commit.",
    )
    return {"commit": commit, "clean": not bool(status), "status_porcelain": status.splitlines()}


def _workspace_path(path: Path, root: Path, *, role: str) -> Path:
    root = root.resolve()
    supplied = Path(path)
    _require(".." not in supplied.parts, f"{role} cannot contain parent traversal: {supplied}")
    candidate = Path(os.path.abspath(root / supplied if not supplied.is_absolute() else supplied))
    workspace = root / WORKSPACE_RELATIVE_ROOT
    _require(
        candidate != workspace and workspace in candidate.parents,
        f"{role} must be a strict descendant of {workspace}.",
    )
    cursor = root
    for component in candidate.relative_to(root).parts:
        cursor /= component
        _require(not cursor.is_symlink(), f"Symlinked {role.lower()} component: {cursor}")
    tracked = subprocess.run(
        [_git_executable(), "ls-files", "--", WORKSPACE_RELATIVE_ROOT.as_posix()],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    _require(tracked.returncode == 0 and not tracked.stdout.strip(), "Workspace contains tracked paths.")
    ignored = subprocess.run(
        [_git_executable(), "check-ignore", "--quiet", "--", candidate.relative_to(root).as_posix()],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    _require(ignored.returncode == 0, f"{candidate} must remain Git-ignored.")
    return candidate


def _tree_without_symlinks(path: Path, *, role: str) -> None:
    if not path.exists():
        return
    _require(path.is_dir() and not path.is_symlink(), f"{role} must be a non-symlink directory.")
    for descendant in path.rglob("*"):
        _require(not descendant.is_symlink(), f"Symlinked {role.lower()} artifact: {descendant}")


def _current_environment(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    expected = contract["execution_environment"]
    _require(platform.python_version() == expected["python_version"], "Python version does not match the lock.")
    pins = _parse_exact_lock(root / expected["environment_lock_path"])
    installed: dict[str, str] = {}
    for package, required in pins.items():
        try:
            observed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise ExpandedStudiesError(f"Locked package is not installed: {package}") from error
        _require(observed == required, f"Installed {package}=={observed}; locked {required}.")
        installed[package] = observed
    return {
        "python_version": platform.python_version(),
        "locked_packages": {name: installed[name] for name in sorted(installed)},
        "device": "cpu",
        "dtype": "float32",
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "intraop_threads": 1,
        "interop_threads": 1,
        "cuda_status": "not_applicable_cpu_contract",
    }


def _configure_determinism(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    environment = _current_environment(contract, root)
    expected = contract["execution_environment"]
    _require(
        expected["device"] == "cpu"
        and expected["dtype"] == "float32"
        and expected["cuda_applicable"] is False,
        "Only the frozen CPU/float32 contract is supported.",
    )
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(int(expected["intraop_threads"]))
    desired_interop = int(expected["interop_threads"])
    if torch.get_num_interop_threads() != desired_interop:
        try:
            torch.set_num_interop_threads(desired_interop)
        except RuntimeError as error:
            if torch.get_num_interop_threads() != desired_interop:
                raise ExpandedStudiesError("Cannot enforce the frozen inter-op thread count.") from error
    torch.use_deterministic_algorithms(True, warn_only=False)
    _require(torch.are_deterministic_algorithms_enabled(), "Deterministic algorithms are not enabled.")
    _require(torch.get_default_dtype() == torch.float32, "Default dtype is not float32.")
    return environment


def _source_manifest(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    relative = {str(path) for path in contract["preflight"]["source_files"]}
    for pattern in contract["preflight"]["source_globs"]:
        relative.update(
            path.relative_to(root).as_posix()
            for path in root.glob(str(pattern))
            if path.is_file()
        )
    entries: list[dict[str, Any]] = []
    for name in sorted(relative):
        path = root / name
        _require(
            path.is_file() and not path.is_symlink(),
            f"Frozen source file is missing or symlinked: {name}",
        )
        entries.append(_descriptor(path, name=name))
    return {
        "schema_version": "expanded-studies-source-manifest-v1",
        "hash_algorithm": "sha256",
        "files": entries,
        "aggregate_sha256": _sha256_bytes(_canonical_json_bytes(entries)),
    }


def _little_endian_array(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    dtype = array.dtype
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<")
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def _tensor_entry(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    array = _little_endian_array(tensor)
    return {
        "name": name,
        "dtype": array.dtype.str,
        "shape": [int(value) for value in array.shape],
        "data_hex": array.tobytes(order="C").hex(),
    }


def _tensor_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json_bytes(list(entries)))


def _task_entries(both: SyntheticBatch, weak: SyntheticBatch) -> list[dict[str, Any]]:
    return [
        _tensor_entry("both.x", both.x),
        _tensor_entry("weak_only.x", weak.x),
        _tensor_entry("y", both.y),
        _tensor_entry("signed_labels", both.signed_labels),
        _tensor_entry("z_s", both.z_s),
        _tensor_entry("z_w", both.z_w),
    ]


def _state_entries(model: torch.nn.Module) -> list[dict[str, Any]]:
    return [
        _tensor_entry(name, value)
        for name, value in sorted(model.state_dict().items(), key=lambda item: item[0])
    ]


def _grads_absent(*models: torch.nn.Module) -> bool:
    return all(parameter.grad is None for model in models for parameter in model.parameters())


def _task_spec(contract: Mapping[str, Any], cell_id: str, *, smoke: bool = False) -> SyntheticTaskSpec:
    cell = next((item for item in contract["shared_task"]["cells"] if item["cell_id"] == cell_id), None)
    _require(cell is not None, f"Unknown task cell: {cell_id}")
    shared = contract["shared_task"]
    if smoke:
        values = contract["engineering_smoke"]
        return SyntheticTaskSpec(
            sequence_length=int(values["sequence_length"]),
            n_samples=int(values["n_samples"]),
            rho=float(cell["rho"]),
            lag_separation=min(int(cell["lag_separation"]), int(values["sequence_length"]) - 1),
            regime=str(shared["regime"]),
            cue_noise=float(shared["cue_noise"]),
            background_noise=float(shared["background_noise"]),
        )
    return SyntheticTaskSpec(
        sequence_length=int(shared["sequence_length"]),
        n_samples=int(shared["n_samples"]),
        rho=float(cell["rho"]),
        lag_separation=int(cell["lag_separation"]),
        regime=str(shared["regime"]),
        cue_noise=float(shared["cue_noise"]),
        background_noise=float(shared["background_noise"]),
    )


def _initialization_manifest(contract: Mapping[str, Any]) -> dict[str, Any]:
    seeds = contract["seed_design"]
    study_a = contract["nonlinear_beta_study"]
    study_b = contract["cdc_tradeoff_study"]
    init_contract = contract["preflight"]["initialization_manifest"]
    a_entries: list[dict[str, Any]] = []
    b_entries: list[dict[str, Any]] = []
    base_lookup: dict[tuple[str, str, int, int], dict[str, Any]] = {}

    for kind in study_a["model_kinds"]:
        for cell in study_a["cells"]:
            spec = _task_spec(contract, str(cell))
            for data_seed in seeds["data_seeds"]:
                both, weak = make_paired_task(spec, int(data_seed))
                task_digest = _tensor_digest(_task_entries(both, weak))
                for model_seed in seeds["model_seeds"]:
                    models = initialize_paired_models(
                        {"width": int(study_a["width"])},
                        both,
                        seed=int(model_seed),
                        kind=str(kind),
                    )
                    both_model, weak_model = models
                    _require(_grads_absent(*models), "Preflight model gradients were populated.")
                    before_both = _state_entries(both_model)
                    before_weak = _state_entries(weak_model)
                    _require(before_both == before_weak, "Study-A paired initialization is not bitwise equal.")
                    state_digest = _tensor_digest(before_both)
                    _require(
                        _state_entries(both_model) == before_both
                        and _state_entries(weak_model) == before_weak
                        and _grads_absent(*models),
                        "Initialization hashing mutated state or gradients.",
                    )
                    record_id = init_contract["study_a_entry_id_format"].format(
                        model_kind=kind,
                        cell_id=cell,
                        data_seed=data_seed,
                        model_seed=model_seed,
                    )
                    a_entries.append(
                        {
                            "initialization_id": record_id,
                            "model_kind": str(kind),
                            "cell_id": str(cell),
                            "data_seed": int(data_seed),
                            "model_seed": int(model_seed),
                            "paired_task_sha256": task_digest,
                            "shared_initial_state_sha256": state_digest,
                            "paired_state_bitwise_equal": True,
                            "parameter_gradients_absent": True,
                        }
                    )
                    del both_model, weak_model

    for kind in study_b["model_kinds"]:
        for cell in study_b["cells"]:
            spec = _task_spec(contract, str(cell))
            for data_seed in seeds["data_seeds"]:
                both, weak = make_paired_task(spec, int(data_seed))
                task_digest = _tensor_digest(_task_entries(both, weak))
                for model_seed in seeds["model_seeds"]:
                    models = initialize_paired_models(
                        {"width": int(study_b["width"])},
                        both,
                        seed=int(model_seed),
                        kind=str(kind),
                    )
                    both_model, weak_model = models
                    _require(_grads_absent(*models), "Preflight model gradients were populated.")
                    before_both = _state_entries(both_model)
                    before_weak = _state_entries(weak_model)
                    _require(before_both == before_weak, "Study-B paired initialization is not bitwise equal.")
                    state_digest = _tensor_digest(before_both)
                    _require(
                        _state_entries(both_model) == before_both
                        and _state_entries(weak_model) == before_weak
                        and _grads_absent(*models),
                        "Initialization hashing mutated state or gradients.",
                    )
                    base_id = init_contract["study_b_base_entry_id_format"].format(
                        model_kind=kind,
                        cell_id=cell,
                        data_seed=data_seed,
                        model_seed=model_seed,
                    )
                    entry = {
                        "base_initialization_id": base_id,
                        "model_kind": str(kind),
                        "cell_id": str(cell),
                        "data_seed": int(data_seed),
                        "model_seed": int(model_seed),
                        "paired_task_sha256": task_digest,
                        "shared_initial_state_sha256": state_digest,
                        "parameter_gradients_absent": True,
                    }
                    b_entries.append(entry)
                    base_lookup[(str(kind), str(cell), int(data_seed), int(model_seed))] = entry
                    del both_model, weak_model

    references: list[dict[str, Any]] = []
    for kind in study_b["model_kinds"]:
        for cell in study_b["cells"]:
            for method in study_b["methods"]:
                for data_seed in seeds["data_seeds"]:
                    for model_seed in seeds["model_seeds"]:
                        base = base_lookup[(str(kind), str(cell), int(data_seed), int(model_seed))]
                        references.append(
                            {
                                "method_record_id": init_contract[
                                    "study_b_method_reference_id_format"
                                ].format(
                                    model_kind=kind,
                                    cell_id=cell,
                                    method_id=method["method_id"],
                                    data_seed=data_seed,
                                    model_seed=model_seed,
                                ),
                                "model_kind": str(kind),
                                "cell_id": str(cell),
                                "method_id": str(method["method_id"]),
                                "data_seed": int(data_seed),
                                "model_seed": int(model_seed),
                                "base_initialization_id": base["base_initialization_id"],
                                "paired_task_sha256": base["paired_task_sha256"],
                                "shared_initial_state_sha256": base["shared_initial_state_sha256"],
                                "state_bitwise_equal_to_base": True,
                                "parameter_gradients_absent": True,
                            }
                        )
    manifest = {
        "schema_version": init_contract["schema_version"],
        "study_a_initializations": a_entries,
        "study_b_base_initializations": b_entries,
        "study_b_method_references": references,
    }
    _validate_initialization_manifest(contract, manifest)
    return manifest


def _validate_initialization_manifest(
    contract: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    init = contract["preflight"]["initialization_manifest"]
    _require(set(manifest) == set(init["top_level_keys"]), "Initialization manifest keys changed.")
    _require(manifest["schema_version"] == init["schema_version"], "Initialization schema changed.")
    a = manifest["study_a_initializations"]
    b = manifest["study_b_base_initializations"]
    references = manifest["study_b_method_references"]
    _require(
        isinstance(a, list)
        and len(a) == init["expected_study_a_initialization_count"] == 192
        and all(isinstance(item, Mapping) for item in a),
        "Initialization manifest must contain 192 Study-A bases.",
    )
    _require(
        isinstance(b, list)
        and len(b) == init["expected_study_b_base_initialization_count"] == 64
        and all(isinstance(item, Mapping) for item in b),
        "Initialization manifest must contain 64 Study-B bases.",
    )
    _require(
        isinstance(references, list)
        and len(references) == init["expected_study_b_method_reference_count"] == 256
        and all(isinstance(item, Mapping) for item in references),
        "Initialization manifest must contain 256 Study-B references.",
    )
    ids = _expected_record_ids(contract)
    coordinates = _record_coordinates(contract)
    _require([item.get("initialization_id") for item in a] == ids["study_a"], "Study-A initialization order changed.")
    _require([item.get("method_record_id") for item in references] == ids["study_b"], "Study-B reference order changed.")
    for item, record_id in zip(a, ids["study_a"]):
        _require(set(item) == set(init["study_a_entry_keys"]), "Study-A initialization entry keys changed.")
        coordinate = coordinates[record_id]
        _require(
            all(item[field] == coordinate[field] for field in ("model_kind", "cell_id", "data_seed", "model_seed"))
            and item["paired_state_bitwise_equal"] is True
            and item["parameter_gradients_absent"] is True,
            "Invalid Study-A initialization coordinate or audit.",
        )
    study_b = contract["cdc_tradeoff_study"]
    seeds = contract["seed_design"]
    expected_bases = [
        {
            "base_initialization_id": init["study_b_base_entry_id_format"].format(
                model_kind=kind,
                cell_id=cell,
                data_seed=data_seed,
                model_seed=model_seed,
            ),
            "model_kind": str(kind),
            "cell_id": str(cell),
            "data_seed": int(data_seed),
            "model_seed": int(model_seed),
        }
        for kind in study_b["model_kinds"]
        for cell in study_b["cells"]
        for data_seed in seeds["data_seeds"]
        for model_seed in seeds["model_seeds"]
    ]
    for item, expected in zip(b, expected_bases):
        _require(set(item) == set(init["study_b_base_entry_keys"]), "Study-B base entry keys changed.")
        _require(
            all(item[field] == value for field, value in expected.items())
            and item["parameter_gradients_absent"] is True,
            "Invalid Study-B base coordinate or gradient audit.",
        )
    for item, record_id in zip(references, ids["study_b"]):
        _require(set(item) == set(init["study_b_method_reference_keys"]), "Study-B reference keys changed.")
        coordinate = coordinates[record_id]
        _require(
            all(
                item[field] == coordinate[field]
                for field in ("model_kind", "cell_id", "method_id", "data_seed", "model_seed")
            )
            and item["state_bitwise_equal_to_base"] is True
            and item["parameter_gradients_absent"] is True,
            "Invalid Study-B reference coordinate or audit.",
        )
    for item in [*a, *b, *references]:
        for field in init["digest_fields"]:
            _require(_is_sha256(item[field]), f"Invalid initialization digest field {field}.")
    bases = {item["base_initialization_id"]: item for item in b}
    _require(len(bases) == 64, "Duplicate Study-B base initialization ID.")
    counts = {key: 0 for key in bases}
    for item in references:
        base = bases.get(item["base_initialization_id"])
        _require(base is not None, "Study-B method references an unknown base.")
        _require(
            all(item[key] == base[key] for key in ("model_kind", "cell_id", "data_seed", "model_seed", "paired_task_sha256", "shared_initial_state_sha256")),
            "Study-B method reference differs from its base.",
        )
        counts[item["base_initialization_id"]] += 1
    _require(all(value == 4 for value in counts.values()), "Every Study-B base must have four references.")


def _record_manifest(contract: Mapping[str, Any]) -> dict[str, Any]:
    ids = _expected_record_ids(contract)
    return {
        "schema_version": contract["preflight"]["record_manifest_schema_version"],
        "study_a_expected_record_ids": ids["study_a"],
        "study_b_expected_record_ids": ids["study_b"],
        "study_a_record_count": len(ids["study_a"]),
        "study_b_method_record_count": len(ids["study_b"]),
        "total_scientific_record_count": len(ids["study_a"]) + len(ids["study_b"]),
    }


def _bootstrap_draws(
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    inference = contract["inference"]
    replicates = int(inference["bootstrap_replicates"])
    rng = np.random.default_rng(int(inference["bootstrap_seed"]))
    data = rng.integers(0, 4, size=(replicates, 4), dtype=np.int64, endpoint=False)
    model = rng.integers(0, 8, size=(replicates, 8), dtype=np.int64, endpoint=False)
    data_le = np.ascontiguousarray(data.astype(np.dtype("<i8"), copy=False))
    model_le = np.ascontiguousarray(model.astype(np.dtype("<i8"), copy=False))
    data_hash = _sha256_bytes(data_le.tobytes(order="C"))
    model_hash = _sha256_bytes(model_le.tobytes(order="C"))
    _require(
        data_hash == inference["data_index_draw"]["sha256_c_contiguous_little_endian_int64_bytes"],
        "Data-axis bootstrap draw hash changed.",
    )
    _require(
        model_hash == inference["model_index_draw"]["sha256_c_contiguous_little_endian_int64_bytes"],
        "Model-axis bootstrap draw hash changed.",
    )
    manifest = {
        "schema_version": contract["preflight"]["bootstrap_draw_manifest_schema_version"],
        "generator": inference["bootstrap_generator"],
        "seed": int(inference["bootstrap_seed"]),
        "replicates": replicates,
        "draw_call_order": list(inference["draw_call_order"]),
        "data_index_draw": {
            "shape": [int(value) for value in data.shape],
            "dtype": "<i8",
            "value_range": [int(data.min()), int(data.max())],
            "sha256": data_hash,
        },
        "model_index_draw": {
            "shape": [int(value) for value in model.shape],
            "dtype": "<i8",
            "value_range": [int(model.min()), int(model.max())],
            "sha256": model_hash,
        },
        "draws_generated_once": True,
        "reuse": inference["draw_reuse"],
    }
    return data, model, manifest


@dataclass
class VerifiedPreflight:
    contract: dict[str, Any]
    manifest: dict[str, Any]
    source_manifest: dict[str, Any]
    record_manifest: dict[str, Any]
    bootstrap_manifest: dict[str, Any]
    initialization_manifest: dict[str, Any]
    contract_path: Path
    preflight_path: Path


@dataclass
class AuthorizedExecution:
    preflight: VerifiedPreflight
    seal: dict[str, Any]
    seal_path: Path


def build_preflight(
    contract_path: Path,
    *,
    output_directory: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Build the outcome-free initialization manifest and authorization inputs."""
    root = Path(repository_root).resolve()
    contract_path = Path(contract_path)
    if not contract_path.is_absolute():
        contract_path = root / contract_path
    contract = _load_contract(contract_path, repository_root=root)
    output = _workspace_path(output_directory, root, role="Preflight output")
    _require(not output.exists(), f"Preflight output already exists: {output}")
    environment = _configure_determinism(contract, root)
    source = _source_manifest(contract, root)
    git = _git_state(root)
    records = _record_manifest(contract)
    _, _, bootstrap = _bootstrap_draws(contract)
    initializations = _initialization_manifest(contract)
    blockers = [] if git["clean"] else ["source_worktree_is_not_clean"]

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        _atomic_write(staging / "execution-contract.json", contract_path.read_bytes())
        historical = root / contract["historical_contract"]["path"]
        lock = root / contract["execution_environment"]["environment_lock_path"]
        _atomic_write(staging / "historical-contract.json", historical.read_bytes())
        _atomic_write(staging / "environment-lock.txt", lock.read_bytes())
        _atomic_write(staging / "source_manifest.json", _canonical_json_bytes(source))
        _atomic_write(staging / "record_manifest.json", _canonical_json_bytes(records))
        _atomic_write(staging / "bootstrap_draw_manifest.json", _canonical_json_bytes(bootstrap))
        _atomic_write(staging / "initialization_manifest.json", _canonical_json_bytes(initializations))
        payload_names = _PREFLIGHT_FILES[:-2]
        manifest = {
            "schema_version": contract["preflight"]["schema_version"],
            "study_id": contract["study_id"],
            "contract_revision": contract["contract_revision"],
            "publication_scope": contract["publication_scope"],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_construction_scope": "initialization_hashing_only",
            "model_constructed": True,
            "model_forward_called": False,
            "optimizer_constructed": False,
            "training_performed": False,
            "model_outcomes_inspected": False,
            "scientific_output_created": False,
            "environment": environment,
            "git": git,
            "files": [_descriptor(staging / name, name=name) for name in payload_names],
            "study_a_expected_record_ids": records["study_a_expected_record_ids"],
            "study_b_expected_record_ids": records["study_b_expected_record_ids"],
            "ready_for_authorization": not blockers,
            "blockers": blockers,
            "execution_authorized": False,
        }
        manifest_path = staging / "preflight.manifest.json"
        _atomic_write(manifest_path, _canonical_json_bytes(manifest))
        _write_sidecar(manifest_path)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "preflight.manifest.json"


def _manifest_path(path: Path, root: Path, *, role: str) -> Path:
    candidate = _workspace_path(path, root, role=role)
    return candidate / "preflight.manifest.json" if candidate.is_dir() else candidate


def _verify_preflight_document(
    contract_path: Path,
    preflight_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    require_ready: bool,
    recompute_initializations: bool,
) -> VerifiedPreflight:
    """Verify preflight bytes, optionally rebuilding initialization-only state."""
    root = Path(repository_root).resolve()
    contract_path = Path(contract_path)
    if not contract_path.is_absolute():
        contract_path = root / contract_path
    contract = _load_contract(contract_path, repository_root=root)
    environment = _configure_determinism(contract, root)
    manifest_path = _manifest_path(preflight_path, root, role="Preflight manifest")
    directory = manifest_path.parent
    _tree_without_symlinks(directory, role="Preflight")
    _require(
        {path.name for path in directory.iterdir()} == set(_PREFLIGHT_FILES),
        "Preflight directory has missing or extra files.",
    )
    _verify_sidecar(manifest_path)
    manifest = _strict_json(manifest_path)
    expected_keys = {
        "schema_version",
        "study_id",
        "contract_revision",
        "publication_scope",
        "created_utc",
        "model_construction_scope",
        "model_constructed",
        "model_forward_called",
        "optimizer_constructed",
        "training_performed",
        "model_outcomes_inspected",
        "scientific_output_created",
        "environment",
        "git",
        "files",
        "study_a_expected_record_ids",
        "study_b_expected_record_ids",
        "ready_for_authorization",
        "blockers",
        "execution_authorized",
    }
    _require(set(manifest) == expected_keys, "Preflight manifest schema is not exact.")
    _validate_timestamp(manifest["created_utc"], "Preflight created_utc")
    _require(
        manifest["schema_version"] == contract["preflight"]["schema_version"]
        and manifest["study_id"] == contract["study_id"]
        and manifest["contract_revision"] == contract["contract_revision"]
        and manifest["publication_scope"] == contract["publication_scope"],
        "Preflight identity changed.",
    )
    _require(
        manifest["model_construction_scope"] == "initialization_hashing_only"
        and manifest["model_constructed"] is True
        and manifest["model_forward_called"] is False
        and manifest["optimizer_constructed"] is False
        and manifest["training_performed"] is False
        and manifest["model_outcomes_inspected"] is False
        and manifest["scientific_output_created"] is False
        and manifest["execution_authorized"] is False,
        "Preflight crossed the initialization-only boundary.",
    )
    _require(manifest["environment"] == environment, "Preflight environment drifted.")
    file_entries = manifest["files"]
    _require(
        isinstance(file_entries, list)
        and len(file_entries) == 7
        and all(isinstance(entry, Mapping) for entry in file_entries),
        "Preflight file list changed.",
    )
    _require([entry.get("path") for entry in file_entries] == _PREFLIGHT_FILES[:-2], "Preflight file order changed.")
    verified = {entry["path"]: _verify_descriptor(directory, entry) for entry in file_entries}
    _require(
        verified["execution-contract.json"].read_bytes() == contract_path.read_bytes(),
        "Preflight contract copy is not byte-for-byte identical.",
    )
    historical = root / contract["historical_contract"]["path"]
    lock = root / contract["execution_environment"]["environment_lock_path"]
    _require(verified["historical-contract.json"].read_bytes() == historical.read_bytes(), "Historical copy drifted.")
    _require(verified["environment-lock.txt"].read_bytes() == lock.read_bytes(), "Environment lock copy drifted.")

    source = _strict_json(verified["source_manifest.json"])
    records = _strict_json(verified["record_manifest.json"])
    bootstrap = _strict_json(verified["bootstrap_draw_manifest.json"])
    initializations = _strict_json(verified["initialization_manifest.json"])
    _require(_canonical_json_bytes(source) == _canonical_json_bytes(_source_manifest(contract, root)), "Source manifest is not reproducible.")
    expected_records = _record_manifest(contract)
    _require(_canonical_json_bytes(records) == _canonical_json_bytes(expected_records), "Record manifest changed.")
    _, _, expected_bootstrap = _bootstrap_draws(contract)
    _require(_canonical_json_bytes(bootstrap) == _canonical_json_bytes(expected_bootstrap), "Bootstrap draw manifest changed.")
    _validate_initialization_manifest(contract, initializations)
    if recompute_initializations:
        expected_initializations = _initialization_manifest(contract)
        _require(
            _canonical_json_bytes(initializations) == _canonical_json_bytes(expected_initializations),
            "Initialization task/state manifest is not reproducible.",
        )
    _require(
        manifest["study_a_expected_record_ids"] == expected_records["study_a_expected_record_ids"]
        and manifest["study_b_expected_record_ids"] == expected_records["study_b_expected_record_ids"],
        "Preflight record IDs changed.",
    )
    current_git = _git_state(root)
    _require(manifest["git"] == current_git, "Git state changed after preflight.")
    truthful_blockers = [] if current_git["clean"] else ["source_worktree_is_not_clean"]
    _require(
        manifest["blockers"] == truthful_blockers
        and manifest["ready_for_authorization"] is (not truthful_blockers),
        "Preflight blockers are not truthful.",
    )
    if require_ready:
        _require(manifest["ready_for_authorization"] is True and not manifest["blockers"], "Preflight is not authorization-ready.")
    return VerifiedPreflight(
        contract=contract,
        manifest=manifest,
        source_manifest=source,
        record_manifest=records,
        bootstrap_manifest=bootstrap,
        initialization_manifest=initializations,
        contract_path=contract_path,
        preflight_path=manifest_path,
    )


def verify_preflight(
    contract_path: Path,
    preflight_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    require_ready: bool = True,
) -> VerifiedPreflight:
    """Fully verify preflight, including initialization-only task/model digests."""
    return _verify_preflight_document(
        contract_path,
        preflight_path,
        repository_root=repository_root,
        require_ready=require_ready,
        recompute_initializations=True,
    )


def _validate_timestamp(value: Any, label: str) -> None:
    _require(isinstance(value, str), f"{label} must be an ISO-8601 string.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ExpandedStudiesError(f"{label} is not valid ISO-8601.") from error
    _require(parsed.tzinfo is not None, f"{label} must include a timezone.")


def _authorization_bindings(preflight: VerifiedPreflight) -> dict[str, Any]:
    directory = preflight.preflight_path.parent
    contract = preflight.contract
    return {
        "execution_contract_sha256": _sha256_file(directory / "execution-contract.json"),
        "preflight_manifest_sha256": _sha256_file(preflight.preflight_path),
        "historical_contract_sha256": _sha256_file(directory / "historical-contract.json"),
        "source_manifest_sha256": _sha256_file(directory / "source_manifest.json"),
        "source_aggregate_sha256": preflight.source_manifest["aggregate_sha256"],
        "environment_lock_sha256": _sha256_file(directory / "environment-lock.txt"),
        "record_manifest_sha256": _sha256_file(directory / "record_manifest.json"),
        "bootstrap_draw_manifest_sha256": _sha256_file(directory / "bootstrap_draw_manifest.json"),
        "initialization_manifest_sha256": _sha256_file(directory / "initialization_manifest.json"),
        "study_a_expected_record_ids": preflight.record_manifest["study_a_expected_record_ids"],
        "study_b_expected_record_ids": preflight.record_manifest["study_b_expected_record_ids"],
    }


def _validate_seal_document(
    contract: Mapping[str, Any],
    seal: Mapping[str, Any],
    expected_bindings: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require(
        set(seal)
        == {"schema_version", "study_id", "issued_utc", "attestation", "authorization", "git", "bindings"},
        "Authorization seal schema is not exact.",
    )
    _require(
        seal["schema_version"] == contract["authorization"]["seal_schema_version"]
        and seal["study_id"] == contract["study_id"],
        "Authorization seal identity changed.",
    )
    _validate_timestamp(seal["issued_utc"], "Authorization issued_utc")
    _require(
        seal["attestation"]
        == {
            "attested": True,
            "text": contract["authorization"]["untouched_seed_attestation_text"],
        },
        "Untouched-seed attestation is absent or changed.",
    )
    _require(
        seal["authorization"]
        == {
            "authorized": True,
            "text": contract["authorization"]["explicit_authorization_text"],
        },
        "Explicit scientific authorization is absent or changed.",
    )
    git = seal["git"]
    _require(
        isinstance(git, Mapping)
        and set(git) == {"commit", "clean"}
        and git["clean"] is True
        and isinstance(git["commit"], str)
        and len(git["commit"]) == 40
        and all(character in "0123456789abcdef" for character in git["commit"]),
        "Authorization seal must bind one clean lowercase 40-character commit.",
    )
    _require(
        list(seal["bindings"].keys()) == contract["authorization"]["required_bindings"]
        and seal["bindings"] == expected_bindings,
        "Authorization seal does not bind the exact preflight and record IDs.",
    )
    return git


def verify_authorization(
    contract_path: Path,
    preflight_path: Path,
    seal_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> AuthorizedExecution:
    """Verify a supplied external seal; this function never creates one."""
    root = Path(repository_root).resolve()
    preflight = _verify_preflight_document(
        contract_path,
        preflight_path,
        repository_root=root,
        require_ready=True,
        recompute_initializations=False,
    )
    seal_path = _workspace_path(seal_path, root, role="Authorization seal")
    _require(seal_path.is_file() and not seal_path.is_symlink(), f"Authorization seal is missing: {seal_path}")
    seal = _strict_json(seal_path)
    expected = _authorization_bindings(preflight)
    sealed_git = _validate_seal_document(preflight.contract, seal, expected)
    current_git = _git_state(root)
    _require(
        current_git["clean"] is True
        and current_git["commit"] == sealed_git["commit"]
        and preflight.manifest["git"] == current_git,
        "Seal, preflight, and current source do not identify one clean commit.",
    )
    return AuthorizedExecution(preflight=preflight, seal=seal, seal_path=seal_path)


def _input_bindings(authorized: AuthorizedExecution) -> dict[str, Any]:
    preflight = authorized.preflight
    bindings = _authorization_bindings(preflight)
    return {
        "execution_contract_sha256": bindings["execution_contract_sha256"],
        "preflight_manifest_sha256": bindings["preflight_manifest_sha256"],
        "authorization_seal_sha256": _sha256_file(authorized.seal_path),
        "source_aggregate_sha256": bindings["source_aggregate_sha256"],
        "record_manifest_sha256": bindings["record_manifest_sha256"],
        "bootstrap_draw_manifest_sha256": bindings["bootstrap_draw_manifest_sha256"],
        "initialization_manifest_sha256": bindings["initialization_manifest_sha256"],
    }


def _initialization_for_record(
    manifest: Mapping[str, Any], coordinate: Mapping[str, Any]
) -> Mapping[str, Any]:
    key = "study_a_initializations" if coordinate["study"] == "study_a" else "study_b_method_references"
    identifier = "initialization_id" if coordinate["study"] == "study_a" else "method_record_id"
    matches = [item for item in manifest[key] if item[identifier] == coordinate["record_id"]]
    _require(len(matches) == 1, f"Missing/duplicate initialization for {coordinate['record_id']}.")
    return matches[0]


def _trajectory_applicability(
    coordinate: Mapping[str, Any], condition: str, step: int
) -> tuple[frozenset[str], frozenset[str]]:
    """Return the explicit finite/present fields for one producer row."""
    _require(condition in {"both", "weak_only"}, "Unknown trajectory condition.")
    _require(type(step) is int and step >= 0, "Trajectory step must be nonnegative.")
    study = str(coordinate["study"])
    raw_method = coordinate.get("method_id")
    method = "" if raw_method is None else str(raw_method)
    numeric = set(_BASE_NUMERIC_FIELDS)
    booleans: set[str] = set()

    if study == "study_a":
        _require(method == "", "Study-A rows cannot carry a method ID.")
        numeric.update(_DIRECT_NUMERIC_FIELDS)
        if condition == "both":
            numeric.update(_LOCKSTEP_PROJECTED_NUMERIC_FIELDS)
            numeric.update(_LOCKSTEP_DIRECT_NUMERIC_FIELDS)
            booleans.update(_LOCKSTEP_BOOLEAN_FIELDS)
    elif study == "study_b" and method == "erm":
        # The frozen exact_response_drift=false ERM producer computes direct
        # autograd only for identity_steps=0, hence only on the logged step 0.
        direct_available = step == 0
        if direct_available:
            numeric.update(_DIRECT_NUMERIC_FIELDS)
        if condition == "both":
            numeric.update(_LOCKSTEP_PROJECTED_NUMERIC_FIELDS)
            booleans.update(_LOCKSTEP_BOOLEAN_FIELDS)
            if direct_available:
                numeric.update(_LOCKSTEP_DIRECT_NUMERIC_FIELDS)
    elif study == "study_b" and method in _STUDY_B_SHADOW_METHODS:
        if condition == "both":
            numeric.update(_SHADOW_NUMERIC_FIELDS)
            booleans.update(_SHADOW_BOOLEAN_FIELDS)
    elif study == "engineering_smoke" and method == "erm":
        numeric.update(_DIRECT_NUMERIC_FIELDS)
        if condition == "both":
            numeric.update(_LOCKSTEP_PROJECTED_NUMERIC_FIELDS)
            numeric.update(_LOCKSTEP_DIRECT_NUMERIC_FIELDS)
            booleans.update(_LOCKSTEP_BOOLEAN_FIELDS)
    elif study == "engineering_smoke" and method == "counterfactual_drift":
        if condition == "both":
            numeric.update(_SHADOW_NUMERIC_FIELDS)
            booleans.update(_SHADOW_BOOLEAN_FIELDS)
    else:
        raise ExpandedStudiesError(
            f"Unsupported trajectory producer: study={study!r}, method={method!r}."
        )
    return frozenset(numeric), frozenset(booleans)


def _structural_na_annotation(
    applicable_numeric: frozenset[str], applicable_booleans: frozenset[str]
) -> str:
    structural_na = (
        set(_NUMERIC_TRAJECTORY_FIELDS) - set(applicable_numeric)
    ) | (set(_BOOLEAN_TRAJECTORY_FIELDS) - set(applicable_booleans))
    return ",".join(f"{field}=N/A" for field in sorted(structural_na))


def _normalize_history(
    history: Sequence[Mapping[str, Any]], coordinate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in history:
        condition = str(raw.get("condition", ""))
        _require(condition in {"both", "weak_only"}, "Unknown trajectory condition.")
        try:
            step_number = float(raw["step"])
        except (KeyError, TypeError, ValueError) as error:
            raise ExpandedStudiesError("Trajectory step is missing or invalid.") from error
        _require(
            math.isfinite(step_number) and step_number.is_integer() and step_number >= 0,
            "Trajectory step must be one finite nonnegative integer.",
        )
        step = int(step_number)
        applicable_numeric, applicable_booleans = _trajectory_applicability(
            coordinate, condition, step
        )
        row: dict[str, Any] = {
            "record_id": coordinate["record_id"],
            "study": coordinate["study"],
            "model_kind": coordinate["model_kind"],
            "cell_id": coordinate["cell_id"],
            "method_id": coordinate["method_id"] or "",
            "data_seed": int(coordinate["data_seed"]),
            "model_seed": int(coordinate["model_seed"]),
            "condition": condition,
        }
        for field in _NUMERIC_TRAJECTORY_FIELDS:
            if field in applicable_numeric:
                _require(
                    field in raw,
                    f"Applicable trajectory field is missing: {field}.",
                )
                try:
                    value = float(raw[field])
                except (TypeError, ValueError) as error:
                    raise ExpandedStudiesError(
                        f"Applicable trajectory field is invalid: {field}."
                    ) from error
                _require(
                    math.isfinite(value),
                    f"Applicable trajectory field is non-finite: {field}.",
                )
                if field == "step":
                    _require(value.is_integer(), "Trajectory step must be an exact integer.")
                    row[field] = int(value)
                else:
                    row[field] = value
            else:
                if field in raw:
                    try:
                        structural_value = float(raw[field])
                    except (TypeError, ValueError) as error:
                        raise ExpandedStudiesError(
                            f"Structurally N/A trajectory field is invalid: {field}."
                        ) from error
                    _require(
                        math.isnan(structural_value),
                        f"Structurally N/A trajectory field is populated: {field}.",
                    )
                row[field] = None
        for field in _BOOLEAN_TRAJECTORY_FIELDS:
            if field in applicable_booleans:
                _require(
                    field in raw and isinstance(raw[field], (bool, np.bool_)),
                    f"Applicable trajectory boolean is missing or invalid: {field}.",
                )
                row[field] = bool(raw[field])
            else:
                _require(
                    field not in raw,
                    f"Structurally N/A trajectory boolean is populated: {field}.",
                )
                row[field] = None
        row["nonfinite_fields"] = _structural_na_annotation(
            applicable_numeric, applicable_booleans
        )
        result.append(row)
    condition_order = {"both": 0, "weak_only": 1}
    result.sort(key=lambda item: (condition_order.get(item["condition"], 99), int(item["step"])))
    _require(
        len({(row["condition"], row["step"]) for row in result}) == len(result),
        "Trajectory contains duplicate condition/step rows.",
    )
    return result


def _format_csv(value: Any) -> str | int:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        _require(math.isfinite(number), "Non-finite CSV values are forbidden.")
        return format(number, ".17g")
    return str(value)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _format_csv(row.get(field)) for field in fields})
    return output.getvalue().encode("utf-8")


def _read_trajectory(path: Path) -> list[dict[str, Any]]:
    try:
        raw_bytes = path.read_bytes()
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(reader.fieldnames == TRAJECTORY_FIELDS, "Trajectory CSV schema changed.")
            rows: list[dict[str, Any]] = []
            for raw in reader:
                _require(
                    None not in raw
                    and all(raw.get(field) is not None for field in TRAJECTORY_FIELDS),
                    "Trajectory row has missing or extra columns.",
                )
                _require(
                    all(raw[field] != "" for field in _METADATA_FIELDS if field != "method_id"),
                    "Trajectory mandatory metadata cannot be blank.",
                )
                _require(
                    raw["study"] in {"study_a", "study_b"}
                    and raw["condition"] in {"both", "weak_only"}
                    and (
                        (raw["study"] == "study_a" and raw["method_id"] == "")
                        or (raw["study"] == "study_b" and raw["method_id"] != "")
                    ),
                    "Trajectory study, method, or condition metadata is invalid.",
                )
                row: dict[str, Any] = {field: raw[field] for field in _METADATA_FIELDS}
                row["data_seed"] = int(row["data_seed"])
                row["model_seed"] = int(row["model_seed"])
                step_value = float(raw["step"])
                _require(
                    math.isfinite(step_value)
                    and step_value.is_integer()
                    and step_value >= 0,
                    "Trajectory step must be one finite nonnegative integer.",
                )
                applicable_numeric, applicable_booleans = _trajectory_applicability(
                    {
                        "study": raw["study"],
                        "method_id": raw["method_id"] or None,
                    },
                    raw["condition"],
                    int(step_value),
                )
                for field in _NUMERIC_TRAJECTORY_FIELDS:
                    text = raw[field]
                    if field in applicable_numeric:
                        _require(
                            text != "",
                            f"Applicable trajectory field is blank: {field}.",
                        )
                        value = float(text)
                        _require(
                            math.isfinite(value),
                            f"Applicable trajectory field is non-finite: {field}.",
                        )
                        if field == "step":
                            _require(value.is_integer(), "Trajectory step must be an exact integer.")
                            row[field] = int(value)
                        else:
                            row[field] = value
                    else:
                        _require(
                            text == "",
                            f"Structurally N/A trajectory field is populated: {field}.",
                        )
                        row[field] = None
                for field in _BOOLEAN_TRAJECTORY_FIELDS:
                    text = raw[field]
                    if field in applicable_booleans:
                        _require(
                            text in {"true", "false"},
                            f"Applicable trajectory boolean is missing or invalid: {field}.",
                        )
                        row[field] = text == "true"
                    else:
                        _require(
                            text == "",
                            f"Structurally N/A trajectory boolean is populated: {field}.",
                        )
                        row[field] = None
                expected_annotation = _structural_na_annotation(
                    applicable_numeric, applicable_booleans
                )
                _require(
                    raw["nonfinite_fields"] == expected_annotation,
                    "Trajectory structural N/A declaration is not exact.",
                )
                row["nonfinite_fields"] = expected_annotation
                rows.append(row)
    except ExpandedStudiesError:
        raise
    except (OSError, UnicodeError, csv.Error, KeyError, TypeError, ValueError) as error:
        raise ExpandedStudiesError(f"Cannot parse trajectory {path}: {error}") from error
    _require(
        len({(row["condition"], row["step"]) for row in rows}) == len(rows),
        "Trajectory contains duplicate condition/step rows.",
    )
    _require(
        raw_bytes == _csv_bytes(rows, TRAJECTORY_FIELDS),
        "Trajectory CSV is not in canonical finite/structural-N/A form.",
    )
    return rows


def _status(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "undefined"
    if math.isinf(value):
        return "right_censored" if value > 0 else "negative_infinity"
    return "finite"


def _finite_or_none(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(float(value)) else None


def _outcome_certificate_dict(certificate: Any) -> dict[str, Any]:
    """Serialize every nullable certificate value with an explicit status."""
    result = asdict(certificate)
    for field in ("drift_crossover_step", "peak_step", "response_equality_step"):
        value = result[field]
        result[f"{field}_status"] = _status(None if value is None else float(value))
        result[field] = None if value is None else int(value)
    for field in ("positive_area", "negative_tail_area", "tail_area_margin"):
        value = float(result[field])
        result[f"{field}_status"] = _status(value)
        result[field] = _finite_or_none(value)
    return result


def _validated_arms(
    rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any], study: str
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], np.ndarray]:
    both = sorted((row for row in rows if row["condition"] == "both"), key=lambda row: row["step"])
    weak = sorted((row for row in rows if row["condition"] == "weak_only"), key=lambda row: row["step"])
    block = contract["nonlinear_beta_study"] if study == "study_a" else contract["cdc_tradeoff_study"]
    optimization = block["optimization"]
    expected = list(range(0, int(optimization["steps"]) + 1, int(optimization["log_every"])))
    if expected[-1] != int(optimization["steps"]):
        expected.append(int(optimization["steps"]))
    _require(
        len(both) == len(weak) == 201
        and [row["step"] for row in both] == expected
        and [row["step"] for row in weak] == expected,
        "Trajectory does not contain both exact 201-point arms.",
    )
    times = np.asarray([row["tau"] for row in both], dtype=float)
    expected_times = np.asarray(expected, dtype=float) * float(optimization["learning_rate"])
    _require(
        np.array_equal(times, expected_times)
        and np.array_equal(np.asarray([row["tau"] for row in weak], dtype=float), expected_times),
        "Trajectory tau grid changed.",
    )
    return both, weak, times


def _profile_dict(point: Any) -> dict[str, Any]:
    raw = asdict(point)
    raw["both_hitting_time_status"] = _status(float(point.both_hitting_time))
    raw["weak_hitting_time_status"] = _status(float(point.weak_hitting_time))
    raw["gated_delta_tw_status"] = _status(float(point.gated_delta_tw))
    raw["both_hitting_time"] = _finite_or_none(float(point.both_hitting_time))
    raw["weak_hitting_time"] = _finite_or_none(float(point.weak_hitting_time))
    raw["gated_delta_tw"] = _finite_or_none(float(point.gated_delta_tw))
    return raw


def _summarize_record(
    contract: Mapping[str, Any],
    coordinate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    input_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    for row in rows:
        _require(
            row["record_id"] == coordinate["record_id"]
            and row["study"] == coordinate["study"]
            and row["model_kind"] == coordinate["model_kind"]
            and row["cell_id"] == coordinate["cell_id"]
            and (row["method_id"] or None) == coordinate["method_id"]
            and row["data_seed"] == coordinate["data_seed"]
            and row["model_seed"] == coordinate["model_seed"],
            "Trajectory metadata differs from record coordinates.",
        )
    both, weak, times = _validated_arms(rows, contract, coordinate["study"])
    both_weak = np.asarray([row["m_w"] for row in both], dtype=float)
    weak_weak = np.asarray([row["m_w"] for row in weak], dtype=float)
    both_strong = np.asarray([row["m_s"] for row in both], dtype=float)
    gap = both_weak - weak_weak
    if coordinate["study"] == "study_a":
        study = contract["nonlinear_beta_study"]
        tail = discrete_crossover_certificate(
            gap,
            weak_only_learnable=False,
            tolerance=float(study["outcome_certificate"]["tolerance"]),
        )
        outcome = bool(tail.strict_outcome_suppression)
        points = first_hitting_profile(
            times,
            both_weak,
            weak_weak,
            study["beta_values"],
            delay_tolerance=float(study["first_hit_rule"]["delay_tolerance"]),
            outcome_suppression_certified=outcome,
        )
        all_beta = bool(outcome and all(point.weak_only_learnable for point in points))
        exact_drift = np.asarray([row["d_w_equal_time_exact"] for row in both], dtype=float)
        projection = np.asarray([row["equal_time_projection_residual"] for row in both], dtype=float)
        _require(np.isfinite(exact_drift).all() and np.isfinite(projection).all(), "Study-A exact drift diagnostics are incomplete.")
        drift_crossing = float(sign_crossing_time(times, exact_drift))
        response_crossing = float(sign_crossing_time(times, gap))
        return {
            "schema_version": "expanded-studies-study-a-record-summary-v1",
            **dict(coordinate),
            "publication_scope": contract["publication_scope"],
            "input_bindings": dict(input_bindings),
            "logged_points_per_arm": 201,
            "outcome_suppression": outcome,
            "outcome_certificate": _outcome_certificate_dict(tail),
            "first_hit_profile": [_profile_dict(point) for point in points],
            "all_beta_robust_causal_certificate": all_beta,
            "weak_auc_gap": float(points[0].weak_auc_gap),
            "exact_drift_crossing_tau": _finite_or_none(drift_crossing),
            "exact_drift_crossing_status": _status(drift_crossing),
            "response_crossing_tau": _finite_or_none(response_crossing),
            "response_crossing_status": _status(response_crossing),
            "exact_drift_sign_change_count": int(n_sign_changes(exact_drift)),
            "response_sign_change_count": int(n_sign_changes(gap)),
            "max_absolute_projection_residual": float(np.max(np.abs(projection))),
            "final_both_weak_response": float(both_weak[-1]),
            "final_weak_only_weak_response": float(weak_weak[-1]),
        }

    study = contract["cdc_tradeoff_study"]
    method = str(coordinate["method_id"])
    point = first_hitting_profile(
        times,
        both_weak,
        weak_weak,
        [float(study["weak_rescue"]["beta_first_hit_delay_report_beta"])],
        delay_tolerance=0.0,
    )[0]
    if method == "erm":
        diagnostics: dict[str, Any] = {
            "status": "N/A",
            "reason": study["diagnostics"]["erm_reason"],
            "target_attainment_fraction": "N/A",
            "feasibility_fraction": "N/A",
            "cap_binding_fraction": "N/A",
            "max_absolute_strong_drift_change": "N/A",
            "row_count": 201,
            "complete": True,
        }
    else:
        for field in ("cdc_target_met", "cdc_feasible", "cdc_cap_binding", "cdc_strong_drift_change"):
            _require(all(row[field] is not None for row in both), f"Shadow diagnostic {field} is incomplete.")
        diagnostics = {
            "status": "available",
            "reason": None,
            "target_attainment_fraction": float(np.mean([bool(row["cdc_target_met"]) for row in both])),
            "feasibility_fraction": float(np.mean([bool(row["cdc_feasible"]) for row in both])),
            "cap_binding_fraction": float(np.mean([bool(row["cdc_cap_binding"]) for row in both])),
            "max_absolute_strong_drift_change": float(max(abs(float(row["cdc_strong_drift_change"])) for row in both)),
            "row_count": len(both),
            "complete": len(both) == 201,
        }
    label = next(item["publication_label"] for item in study["methods"] if item["method_id"] == method)
    if method in SHADOW_METHOD_PUBLICATION_LABELS:
        _require(label == SHADOW_METHOD_PUBLICATION_LABELS[method], "Publication label drifted.")
    return {
        "schema_version": "expanded-studies-study-b-record-summary-v1",
        **dict(coordinate),
        "publication_scope": contract["publication_scope"],
        "publication_label": label,
        "input_bindings": dict(input_bindings),
        "logged_points_per_arm": 201,
        "weak_auc_gap": float(point.weak_auc_gap),
        "weak_rescue_score": -abs(float(point.weak_auc_gap)),
        "beta_first_hit_delay": _finite_or_none(float(point.gated_delta_tw)),
        "beta_first_hit_delay_status": _status(float(point.gated_delta_tw)),
        "final_both_weak_response": float(both_weak[-1]),
        "strong_response_auc": float(np.trapezoid(both_strong, times)),
        "final_strong_response": float(both_strong[-1]),
        "shadow_diagnostics": diagnostics,
    }


def _peak_rss() -> tuple[int, str, int]:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return raw, "bytes", 1
    return raw * 1024, "KiB", 1024


def _record_directory(run_directory: Path, record_id: str) -> Path:
    return run_directory / "records" / record_id


def _complete_marker(
    directory: Path,
    coordinate: Mapping[str, Any],
    input_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "expanded-studies-record-complete-v1",
        "record_id": coordinate["record_id"],
        "study": coordinate["study"],
        "coordinates": dict(coordinate),
        "input_bindings": dict(input_bindings),
        "files": [
            _descriptor(directory / name, name=name)
            for name in ("trajectory.csv", "summary.json", "resources.json")
        ],
    }


def _consume_worker_launch_context(
    record_id: str, root: Path
) -> dict[str, Any]:
    """Consume and validate the parent-only launch context before scientific work."""
    raw = os.environ.pop(_WORKER_LAUNCH_CONTEXT_ENV, None)
    _require(
        isinstance(raw, str) and raw and len(raw) <= 65536,
        "Isolated worker launch context is missing or invalid.",
    )
    context = _strict_json_text(raw, "isolated worker launch context")
    expected_keys = {
        "schema_version",
        "token",
        "token_sha256",
        "record_id",
        "parent_pid",
        "repository_root",
        "interpreter_path",
        "script_path",
        "script_descriptor",
        "input_bindings",
        "argv",
    }
    _require(set(context) == expected_keys, "Worker launch context schema is not exact.")
    _require(
        context["schema_version"] == _WORKER_LAUNCH_CONTEXT_SCHEMA
        and _is_sha256(context["token"])
        and _is_sha256(context["token_sha256"])
        and context["token_sha256"]
        == _sha256_bytes(str(context["token"]).encode("ascii"))
        and context["record_id"] == record_id
        and type(context["parent_pid"]) is int
        and context["parent_pid"] > 0
        and context["repository_root"] == str(root)
        and context["interpreter_path"] == sys.executable
        and context["script_path"] == str(root / _WORKER_SCRIPT_RELATIVE_PATH)
        and isinstance(context["script_descriptor"], Mapping)
        and isinstance(context["input_bindings"], Mapping)
        and isinstance(context["argv"], list)
        and all(isinstance(item, str) for item in context["argv"]),
        "Worker launch context identity is invalid.",
    )
    _require(
        list(sys.argv) == context["argv"],
        "Worker argv differs from the parent-issued launch context.",
    )
    _require(
        os.getppid() == context["parent_pid"] and os.getpid() != context["parent_pid"],
        "Worker process ancestry differs from the parent-issued launch context.",
    )
    script = _verify_descriptor(root, context["script_descriptor"])
    _require(
        script == root / _WORKER_SCRIPT_RELATIVE_PATH
        and _descriptor(script, name=_WORKER_SCRIPT_RELATIVE_PATH)
        == context["script_descriptor"],
        "Worker script differs from the parent-issued launch descriptor.",
    )
    return context


def _launch_evidence(
    context: Mapping[str, Any], worker_pid: int
) -> dict[str, Any]:
    descriptor = dict(context["script_descriptor"])
    return {
        "schema_version": _WORKER_LAUNCH_EVIDENCE_SCHEMA,
        "record_id": context["record_id"],
        "token_sha256": context["token_sha256"],
        "parent_pid": int(context["parent_pid"]),
        "worker_pid": int(worker_pid),
        "interpreter_path": context["interpreter_path"],
        "script_path": context["script_path"],
        "script_descriptor": descriptor,
        "script_sha256": descriptor["sha256"],
        "isolation_verified": True,
    }


def _validate_launch_evidence(
    evidence: Mapping[str, Any],
    coordinate: Mapping[str, Any],
    worker_binding: Mapping[str, Any],
    *,
    expected_token_sha256: str | None = None,
    expected_parent_pid: int | None = None,
    expected_worker_pid: int | None = None,
) -> None:
    expected_keys = {
        "schema_version",
        "record_id",
        "token_sha256",
        "parent_pid",
        "worker_pid",
        "interpreter_path",
        "script_path",
        "script_descriptor",
        "script_sha256",
        "isolation_verified",
    }
    _require(set(evidence) == expected_keys, "Worker launch evidence schema changed.")
    _require(
        evidence["schema_version"] == _WORKER_LAUNCH_EVIDENCE_SCHEMA
        and evidence["record_id"] == coordinate["record_id"]
        and _is_sha256(evidence["token_sha256"])
        and type(evidence["parent_pid"]) is int
        and evidence["parent_pid"] > 0
        and type(evidence["worker_pid"]) is int
        and evidence["worker_pid"] > 0
        and evidence["worker_pid"] != evidence["parent_pid"]
        and evidence["interpreter_path"] == worker_binding["interpreter_path"]
        and evidence["script_path"] == worker_binding["script_path"]
        and evidence["script_descriptor"] == worker_binding["script_descriptor"]
        and evidence["script_sha256"]
        == worker_binding["script_descriptor"]["sha256"]
        and evidence["isolation_verified"] is True,
        "Worker launch evidence is inconsistent with the sealed executable.",
    )
    if expected_token_sha256 is not None:
        _require(
            evidence["token_sha256"] == expected_token_sha256,
            "Worker launch token digest differs from the parent-issued token.",
        )
    if expected_parent_pid is not None:
        _require(
            evidence["parent_pid"] == expected_parent_pid,
            "Worker launch evidence has the wrong parent PID.",
        )
    if expected_worker_pid is not None:
        _require(
            evidence["worker_pid"] == expected_worker_pid,
            "Worker launch evidence has the wrong worker PID.",
        )


def _validate_resource_telemetry(resources: Mapping[str, Any]) -> None:
    _require(
        type(resources["worker_pid"]) is int and resources["worker_pid"] > 0,
        "Invalid worker PID.",
    )
    _require(
        type(resources["worker_wall_time_seconds"]) in {int, float}
        and math.isfinite(float(resources["worker_wall_time_seconds"]))
        and float(resources["worker_wall_time_seconds"]) > 0,
        "Invalid worker wall time.",
    )
    _require(
        type(resources["process_peak_rss_bytes"]) is int
        and resources["process_peak_rss_bytes"] > 0,
        "Invalid peak RSS.",
    )
    _require(
        type(resources["static_parameter_bytes"]) is int
        and resources["static_parameter_bytes"] > 0,
        "Invalid static parameter bytes.",
    )
    _require(
        (resources["rss_input_unit"], resources["rss_multiplier_to_bytes"])
        in {("bytes", 1), ("KiB", 1024)},
        "Invalid RSS normalization.",
    )
    _require(
        resources["cuda_peak_allocated_bytes"] is None
        and resources["cuda_status"] == "N/A",
        "CPU CUDA telemetry must be N/A.",
    )


def _validate_worker_resources(
    resources: Mapping[str, Any],
    coordinate: Mapping[str, Any],
    worker_binding: Mapping[str, Any],
    *,
    launch_context: Mapping[str, Any] | None = None,
    observed_worker_pid: int | None = None,
) -> None:
    expected_keys = {
        "schema_version",
        "finalization_status",
        "record_id",
        "study",
        "worker_pid",
        "launch_evidence",
        "worker_wall_time_seconds",
        "worker_wall_time_scope",
        "process_peak_rss_bytes",
        "rss_input_unit",
        "rss_multiplier_to_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_status",
        "static_parameter_bytes",
    }
    _require(set(resources) == expected_keys, "Preliminary worker resource schema changed.")
    _require(
        resources["schema_version"]
        == "expanded-studies-record-worker-resources-v1"
        and resources["finalization_status"] == "worker_nonfinal"
        and resources["record_id"] == coordinate["record_id"]
        and resources["study"] == coordinate["study"]
        and resources["worker_wall_time_scope"] == _WORKER_WALL_TIME_SCOPE
        and isinstance(resources["launch_evidence"], Mapping),
        "Preliminary worker resource identity changed.",
    )
    _validate_resource_telemetry(resources)
    expected_token_sha256 = None
    expected_parent_pid = None
    if launch_context is not None:
        expected_token_sha256 = str(launch_context["token_sha256"])
        expected_parent_pid = int(launch_context["parent_pid"])
    _validate_launch_evidence(
        resources["launch_evidence"],
        coordinate,
        worker_binding,
        expected_token_sha256=expected_token_sha256,
        expected_parent_pid=expected_parent_pid,
        expected_worker_pid=observed_worker_pid,
    )
    _require(
        resources["worker_pid"] == resources["launch_evidence"]["worker_pid"],
        "Worker PID differs between resources and launch evidence.",
    )


def _validate_final_resources(
    resources: Mapping[str, Any],
    coordinate: Mapping[str, Any],
    worker_binding: Mapping[str, Any],
    *,
    launch_context: Mapping[str, Any] | None = None,
    observed_worker_pid: int | None = None,
) -> None:
    expected_keys = {
        "schema_version",
        "finalization_status",
        "record_id",
        "study",
        "worker_pid",
        "launch_evidence",
        "worker_wall_time_seconds",
        "worker_wall_time_scope",
        "parent_pid",
        "parent_wall_time_seconds",
        "parent_wall_time_scope",
        "wall_time_seconds",
        "process_peak_rss_bytes",
        "rss_input_unit",
        "rss_multiplier_to_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_status",
        "static_parameter_bytes",
    }
    _require(set(resources) == expected_keys, "Final record resource schema changed.")
    _require(
        resources["schema_version"] == "expanded-studies-record-resources-v2"
        and resources["finalization_status"] == "parent_finalized"
        and resources["record_id"] == coordinate["record_id"]
        and resources["study"] == coordinate["study"]
        and resources["worker_wall_time_scope"] == _WORKER_WALL_TIME_SCOPE
        and resources["parent_wall_time_scope"] == _PARENT_WALL_TIME_SCOPE
        and isinstance(resources["launch_evidence"], Mapping),
        "Final record resource identity changed.",
    )
    _validate_resource_telemetry(resources)
    for field in ("parent_wall_time_seconds", "wall_time_seconds"):
        _require(
            type(resources[field]) in {int, float}
            and math.isfinite(float(resources[field]))
            and float(resources[field]) > 0,
            f"Invalid resource field: {field}",
        )
    expected_token_sha256 = None
    expected_parent_pid = None
    if launch_context is not None:
        expected_token_sha256 = str(launch_context["token_sha256"])
        expected_parent_pid = int(launch_context["parent_pid"])
    _validate_launch_evidence(
        resources["launch_evidence"],
        coordinate,
        worker_binding,
        expected_token_sha256=expected_token_sha256,
        expected_parent_pid=expected_parent_pid,
        expected_worker_pid=observed_worker_pid,
    )
    _require(
        resources["worker_pid"] == resources["launch_evidence"]["worker_pid"]
        and type(resources["parent_pid"]) is int
        and resources["parent_pid"] > 0
        and resources["parent_pid"]
        == resources["launch_evidence"]["parent_pid"]
        and float(resources["wall_time_seconds"])
        == float(resources["parent_wall_time_seconds"])
        and float(resources["parent_wall_time_seconds"])
        >= float(resources["worker_wall_time_seconds"]),
        "Parent finalization evidence or wall-time scopes are inconsistent.",
    )


def _write_worker_payload(
    run_directory: Path,
    coordinate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> Path:
    """Write only the worker's nonfinal trajectory and summary payload."""
    directory = _record_directory(run_directory, str(coordinate["record_id"]))
    _require(not directory.exists(), f"Record directory already exists: {directory}")
    directory.mkdir()
    _atomic_write(directory / "trajectory.csv", _csv_bytes(rows, TRAJECTORY_FIELDS))
    _atomic_write(directory / "summary.json", _canonical_json_bytes(summary))
    return directory


def run_record_worker(
    contract_path: Path,
    preflight_path: Path,
    seal_path: Path,
    record_id: str,
    *,
    output_directory: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Execute one parent-issued authorized record in this isolated process."""
    worker_started = time.perf_counter()
    root = Path(repository_root).resolve()
    launch_context = _consume_worker_launch_context(record_id, root)
    contract = _load_contract(Path(contract_path), repository_root=root)
    _configure_determinism(contract, root)
    # Context and model-free authorization verification both precede task/model work.
    authorized = verify_authorization(
        contract_path,
        preflight_path,
        seal_path,
        repository_root=root,
    )
    worker_binding = _worker_executable_binding(
        authorized.preflight.source_manifest, root
    )
    _require(
        launch_context["interpreter_path"] == worker_binding["interpreter_path"]
        and launch_context["script_path"] == worker_binding["script_path"]
        and launch_context["script_descriptor"]
        == worker_binding["script_descriptor"]
        and launch_context["input_bindings"] == _input_bindings(authorized),
        "Worker launch context differs from the sealed authorization inputs.",
    )
    coordinates = _record_coordinates(contract)
    _require(record_id in coordinates, f"Unknown scientific record ID: {record_id}")
    coordinate = coordinates[record_id]
    run_directory = _workspace_path(output_directory, root, role="Scientific run output")
    _tree_without_symlinks(run_directory, role="Scientific run output")
    _require(
        run_directory.is_dir()
        and (run_directory / "records").is_dir()
        and not (run_directory / "records").is_symlink(),
        "Parent-created scientific run/records directories are missing.",
    )
    record_directory = _record_directory(run_directory, record_id)
    _require(not record_directory.exists(), f"Record output already exists: {record_id}")

    spec = _task_spec(contract, str(coordinate["cell_id"]))
    both, weak = make_paired_task(spec, int(coordinate["data_seed"]))
    initialization = _initialization_for_record(
        authorized.preflight.initialization_manifest, coordinate
    )
    _require(
        _tensor_digest(_task_entries(both, weak)) == initialization["paired_task_sha256"],
        "Worker task digest differs from the authorized initialization.",
    )
    study = (
        contract["nonlinear_beta_study"]
        if coordinate["study"] == "study_a"
        else contract["cdc_tradeoff_study"]
    )
    initial_models = initialize_paired_models(
        {"width": int(study["width"])},
        both,
        seed=int(coordinate["model_seed"]),
        kind=str(coordinate["model_kind"]),
    )
    state_entries = _state_entries(initial_models[0])
    _require(
        state_entries == _state_entries(initial_models[1])
        and _tensor_digest(state_entries) == initialization["shared_initial_state_sha256"]
        and _grads_absent(*initial_models),
        "Worker initialization differs from the authorized bitwise-shared state.",
    )
    del initial_models

    training = dict(study["optimization"])
    method = "erm" if coordinate["study"] == "study_a" else str(coordinate["method_id"])
    mitigation = {
        "method": method,
        "objective": "cross_entropy",
        "max_alpha": training.get("max_alpha"),
        "feasibility_epsilon": training.get("feasibility_epsilon", 1e-10),
        "ema_decay": training.get("ema_decay", 0.9),
    }
    history, states = train_paired(
        {"width": int(study["width"])},
        both,
        weak,
        training,
        mitigation,
        seed=int(coordinate["model_seed"]),
        kind=str(coordinate["model_kind"]),
    )
    rows = _normalize_history(history, coordinate)
    bindings = _input_bindings(authorized)
    summary = _summarize_record(contract, coordinate, rows, bindings)
    static_bytes = int(
        sum(tensor.numel() * tensor.element_size() for tensor in states["both"].values())
    )
    directory = _write_worker_payload(run_directory, coordinate, rows, summary)
    peak_bytes, rss_input_unit, multiplier = _peak_rss()
    resources = {
        "schema_version": "expanded-studies-record-worker-resources-v1",
        "finalization_status": "worker_nonfinal",
        "record_id": record_id,
        "study": coordinate["study"],
        "worker_pid": os.getpid(),
        "launch_evidence": _launch_evidence(launch_context, os.getpid()),
        "worker_wall_time_seconds": time.perf_counter() - worker_started,
        "worker_wall_time_scope": _WORKER_WALL_TIME_SCOPE,
        "process_peak_rss_bytes": peak_bytes,
        "rss_input_unit": rss_input_unit,
        "rss_multiplier_to_bytes": multiplier,
        "cuda_peak_allocated_bytes": None,
        "cuda_status": "N/A",
        "static_parameter_bytes": static_bytes,
    }
    _validate_worker_resources(
        resources,
        coordinate,
        worker_binding,
        launch_context=launch_context,
        observed_worker_pid=os.getpid(),
    )
    _atomic_write(directory / "resources.json", _canonical_json_bytes(resources))
    _require(
        {path.name for path in directory.iterdir()} == set(_NONFINAL_RECORD_FILES),
        "Worker emitted files outside the nonfinal three-file bundle.",
    )
    return directory


def _load_worker_bundle(
    run_directory: Path,
    contract: Mapping[str, Any],
    coordinate: Mapping[str, Any],
    expected_bindings: Mapping[str, Any],
    worker_binding: Mapping[str, Any],
    launch_context: Mapping[str, Any],
    observed_worker_pid: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    directory = _record_directory(run_directory, str(coordinate["record_id"]))
    _require(
        directory.is_dir() and not directory.is_symlink(),
        f"Worker did not create a record bundle: {coordinate['record_id']}",
    )
    _require(
        {path.name for path in directory.iterdir()} == set(_NONFINAL_RECORD_FILES),
        f"Worker bundle is not the exact nonfinal three-file inventory: {coordinate['record_id']}",
    )
    rows = _read_trajectory(directory / "trajectory.csv")
    summary = _strict_json(directory / "summary.json")
    resources = _strict_json(directory / "resources.json")
    recomputed = _summarize_record(contract, coordinate, rows, expected_bindings)
    _require(
        _canonical_json_bytes(summary) == _canonical_json_bytes(recomputed),
        f"Worker record summary is not reproducible: {coordinate['record_id']}",
    )
    _validate_worker_resources(
        resources,
        coordinate,
        worker_binding,
        launch_context=launch_context,
        observed_worker_pid=observed_worker_pid,
    )
    return summary, rows, resources


def _finalize_worker_bundle(
    run_directory: Path,
    coordinate: Mapping[str, Any],
    input_bindings: Mapping[str, Any],
    worker_binding: Mapping[str, Any],
    launch_context: Mapping[str, Any],
    observed_worker_pid: int,
    preliminary_resources: Mapping[str, Any],
    parent_started: float,
) -> None:
    """Write final resources once, then create the sole completion marker."""
    directory = _record_directory(run_directory, str(coordinate["record_id"]))
    _require(
        {path.name for path in directory.iterdir()} == set(_NONFINAL_RECORD_FILES),
        f"Cannot finalize a noncanonical worker bundle: {coordinate['record_id']}",
    )
    _validate_worker_resources(
        preliminary_resources,
        coordinate,
        worker_binding,
        launch_context=launch_context,
        observed_worker_pid=observed_worker_pid,
    )
    _require(
        os.getpid() == launch_context["parent_pid"],
        "Only the worker's issuing parent may finalize the record.",
    )
    parent_elapsed = time.perf_counter() - parent_started
    resources = dict(preliminary_resources)
    resources.update(
        {
            "schema_version": "expanded-studies-record-resources-v2",
            "finalization_status": "parent_finalized",
            "parent_pid": os.getpid(),
            "parent_wall_time_seconds": parent_elapsed,
            "parent_wall_time_scope": _PARENT_WALL_TIME_SCOPE,
            "wall_time_seconds": parent_elapsed,
        }
    )
    _validate_final_resources(
        resources,
        coordinate,
        worker_binding,
        launch_context=launch_context,
        observed_worker_pid=observed_worker_pid,
    )
    _atomic_write(directory / "resources.json", _canonical_json_bytes(resources))
    marker = _complete_marker(directory, coordinate, input_bindings)
    _atomic_write(directory / "complete.json", _canonical_json_bytes(marker))
    _require(
        {path.name for path in directory.iterdir()} == set(_RECORD_FILES),
        f"Final record inventory changed: {coordinate['record_id']}",
    )


def _load_record(
    run_directory: Path,
    contract: Mapping[str, Any],
    coordinate: Mapping[str, Any],
    expected_bindings: Mapping[str, Any],
    worker_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    directory = _record_directory(run_directory, str(coordinate["record_id"]))
    _require(directory.is_dir() and not directory.is_symlink(), f"Missing record: {coordinate['record_id']}")
    _require({path.name for path in directory.iterdir()} == set(_RECORD_FILES), f"Record file inventory changed: {coordinate['record_id']}")
    marker = _strict_json(directory / "complete.json")
    _require(
        set(marker) == {"schema_version", "record_id", "study", "coordinates", "input_bindings", "files"}
        and marker["schema_version"] == "expanded-studies-record-complete-v1"
        and marker["record_id"] == coordinate["record_id"]
        and marker["study"] == coordinate["study"]
        and marker["coordinates"] == coordinate
        and marker["input_bindings"] == expected_bindings
        and isinstance(marker["files"], list)
        and all(isinstance(item, Mapping) for item in marker["files"]),
        f"Completion marker binding changed: {coordinate['record_id']}",
    )
    _require([item.get("path") for item in marker["files"]] == ["trajectory.csv", "summary.json", "resources.json"], "Completion marker file order changed.")
    for descriptor in marker["files"]:
        _verify_descriptor(directory, descriptor)
    rows = _read_trajectory(directory / "trajectory.csv")
    summary = _strict_json(directory / "summary.json")
    resources = _strict_json(directory / "resources.json")
    recomputed = _summarize_record(contract, coordinate, rows, expected_bindings)
    _require(_canonical_json_bytes(summary) == _canonical_json_bytes(recomputed), f"Record summary is not reproducible: {coordinate['record_id']}")
    _validate_final_resources(resources, coordinate, worker_binding)
    return summary, rows, resources


def _validate_runtime_environment_document(
    runtime: Mapping[str, Any],
    *,
    expected_environment: Mapping[str, Any],
    expected_git: Mapping[str, Any],
) -> None:
    _require(
        set(runtime) == {"schema_version", "created_utc", "environment", "git", "command"}
        and runtime["schema_version"] == "expanded-studies-runtime-environment-v1"
        and runtime["environment"] == expected_environment
        and runtime["git"] == expected_git
        and isinstance(runtime["command"], list)
        and all(isinstance(item, str) for item in runtime["command"])
        and isinstance(runtime["git"], Mapping)
        and set(runtime["git"]) == {"commit", "clean", "status_porcelain"}
        and runtime["git"]["clean"] is True
        and runtime["git"]["status_porcelain"] == [],
        "Runtime environment/git state is inconsistent with the authorized preflight.",
    )
    _validate_timestamp(runtime["created_utc"], "Runtime created_utc")


def _copy_run_provenance(
    authorized: AuthorizedExecution, run_directory: Path, root: Path
) -> dict[str, Any]:
    preflight = authorized.preflight
    directory = preflight.preflight_path.parent
    mappings = {
        directory / "execution-contract.json": run_directory / "contract.resolved.json",
        directory / "historical-contract.json": run_directory / "historical-contract.json",
        directory / "environment-lock.txt": run_directory / "environment-lock.txt",
        preflight.preflight_path: run_directory / "preflight.manifest.json",
        preflight.preflight_path.with_suffix(".sha256"): run_directory / "preflight.manifest.sha256",
        authorized.seal_path: run_directory / "authorization.seal.json",
        directory / "source_manifest.json": run_directory / "source_manifest.json",
        directory / "record_manifest.json": run_directory / "record_manifest.json",
        directory / "bootstrap_draw_manifest.json": run_directory / "bootstrap_draw_manifest.json",
        directory / "initialization_manifest.json": run_directory / "initialization_manifest.json",
    }
    for source, destination in mappings.items():
        _write_or_verify(destination, source.read_bytes())
    bindings = _input_bindings(authorized)
    provenance = {
        "schema_version": "expanded-studies-provenance-v1",
        "study_id": preflight.contract["study_id"],
        "publication_scope": preflight.contract["publication_scope"],
        **bindings,
        "historical_contract_sha256": _sha256_file(directory / "historical-contract.json"),
        "environment_lock_sha256": _sha256_file(directory / "environment-lock.txt"),
        "study_a_expected_record_ids": preflight.record_manifest["study_a_expected_record_ids"],
        "study_b_expected_record_ids": preflight.record_manifest["study_b_expected_record_ids"],
        "git_commit": authorized.seal["git"]["commit"],
    }
    _write_or_verify(run_directory / "provenance.json", _canonical_json_bytes(provenance))
    environment_path = run_directory / "environment.json"
    current_environment = _current_environment(preflight.contract, root)
    current_git = _git_state(root)
    if environment_path.exists():
        _validate_runtime_environment_document(
            _strict_json(environment_path),
            expected_environment=current_environment,
            expected_git=current_git,
        )
    else:
        runtime = {
            "schema_version": "expanded-studies-runtime-environment-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "environment": current_environment,
            "git": current_git,
            "command": list(sys.argv),
        }
        _atomic_write(environment_path, _canonical_json_bytes(runtime))
    return provenance


def _new_worker_launch_context(
    record_id: str,
    root: Path,
    worker_binding: Mapping[str, Any],
    input_bindings: Mapping[str, Any],
    argv: Sequence[str],
    used_token_sha256: set[str],
) -> dict[str, Any]:
    while True:
        token = secrets.token_hex(32)
        token_sha256 = _sha256_bytes(token.encode("ascii"))
        if token_sha256 not in used_token_sha256:
            break
    return {
        "schema_version": _WORKER_LAUNCH_CONTEXT_SCHEMA,
        "token": token,
        "token_sha256": token_sha256,
        "record_id": record_id,
        "parent_pid": os.getpid(),
        "repository_root": str(root),
        "interpreter_path": worker_binding["interpreter_path"],
        "script_path": worker_binding["script_path"],
        "script_descriptor": dict(worker_binding["script_descriptor"]),
        "input_bindings": dict(input_bindings),
        "argv": list(argv),
    }


def _validate_existing_records(
    output: Path,
    contract: Mapping[str, Any],
    coordinates: Mapping[str, Mapping[str, Any]],
    expected_ids: Sequence[str],
    input_bindings: Mapping[str, Any],
    worker_binding: Mapping[str, Any],
) -> set[str]:
    """Validate all completed resume records before any new record is launched."""
    records_root = output / "records"
    if not records_root.exists():
        return set()
    _require(
        records_root.is_dir() and not records_root.is_symlink(),
        "Partial-run records must be a non-symlink directory.",
    )
    entries = {path.name: path for path in records_root.iterdir()}
    extras = set(entries) - set(expected_ids)
    _require(not extras, f"Unexpected record directories: {sorted(extras)}")
    token_hashes: set[str] = set()
    for record_id in expected_ids:
        directory = entries.get(record_id)
        if directory is None:
            continue
        _require(
            directory.is_dir() and not directory.is_symlink(),
            f"Record path is not a non-symlink directory: {record_id}",
        )
        observed_files = {path.name for path in directory.iterdir()}
        if observed_files != set(_RECORD_FILES):
            raise ExpandedStudiesError(
                "Nonrecoverable incomplete record "
                f"{record_id}: expected the exact finalized four-file inventory, "
                f"found {sorted(observed_files)}. The record will not be rerun or accepted."
            )
        _, _, resources = _load_record(
            output,
            contract,
            coordinates[record_id],
            input_bindings,
            worker_binding,
        )
        token_sha256 = str(resources["launch_evidence"]["token_sha256"])
        _require(
            token_sha256 not in token_hashes,
            "Completed records reuse a worker launch token digest.",
        )
        token_hashes.add(token_sha256)
    return token_hashes


def _validate_partial_run_inventory(output: Path) -> None:
    if not output.exists():
        return
    _require(
        output.is_dir() and not output.is_symlink(),
        "Partial scientific run output must be a non-symlink directory.",
    )
    allowed = set(_RUN_FILES) | {"records"}
    entries = {path.name: path for path in output.iterdir()}
    extras = set(entries) - allowed
    _require(not extras, f"Unexpected top-level partial-run entries: {sorted(extras)}")
    for name, path in entries.items():
        if name == "records":
            _require(path.is_dir() and not path.is_symlink(), "Partial-run records must be a directory.")
        else:
            _require(path.is_file() and not path.is_symlink(), f"Partial-run artifact must be a file: {name}")


def run_scientific_studies(
    contract_path: Path,
    preflight_path: Path,
    seal_path: Path,
    *,
    output_directory: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Run all 448 records through the one sealed, fresh-worker protocol."""
    root = Path(repository_root).resolve()
    contract = _load_contract(Path(contract_path), repository_root=root)
    _configure_determinism(contract, root)
    # Seal and exact source-bound worker verification precede scientific output.
    authorized = verify_authorization(contract_path, preflight_path, seal_path, repository_root=root)
    worker_binding = _worker_executable_binding(
        authorized.preflight.source_manifest, root
    )
    output = _workspace_path(output_directory, root, role="Scientific run output")
    _tree_without_symlinks(output, role="Scientific run output")
    _validate_partial_run_inventory(output)
    if (output / "artifact_manifest.json").exists():
        validate_run_artifacts(output, repository_root=root)
        return output

    bindings = _input_bindings(authorized)
    coordinates = _record_coordinates(contract)
    expected_ids = list(authorized.preflight.record_manifest["study_a_expected_record_ids"]) + list(
        authorized.preflight.record_manifest["study_b_expected_record_ids"]
    )
    used_token_sha256 = _validate_existing_records(
        output,
        contract,
        coordinates,
        expected_ids,
        bindings,
        worker_binding,
    )

    output.mkdir(parents=True, exist_ok=True)
    (output / "records").mkdir(exist_ok=True)
    provenance = _copy_run_provenance(authorized, output, root)
    _require(
        all(provenance[key] == value for key, value in bindings.items()),
        "Run provenance differs from the authorized input bindings.",
    )

    for record_id in expected_ids:
        coordinate = coordinates[record_id]
        directory = _record_directory(output, record_id)
        if directory.exists():
            continue  # Every existing directory was fully validated before this loop.
        _require(
            _worker_executable_binding(authorized.preflight.source_manifest, root)
            == worker_binding,
            "Sealed worker executable changed before record launch.",
        )
        command = [
            worker_binding["interpreter_path"],
            worker_binding["script_path"],
            "worker",
            "--contract",
            str(Path(contract_path)),
            "--preflight",
            str(Path(preflight_path)),
            "--seal",
            str(Path(seal_path)),
            "--record-id",
            record_id,
            "--output-directory",
            str(output),
            "--repository-root",
            str(root),
        ]
        launch_context = _new_worker_launch_context(
            record_id,
            root,
            worker_binding,
            bindings,
            command[1:],
            used_token_sha256,
        )
        token_sha256 = str(launch_context["token_sha256"])
        environment = os.environ.copy()
        environment[_WORKER_LAUNCH_CONTEXT_ENV] = (
            _canonical_json_bytes(launch_context).decode("utf-8").rstrip("\n")
        )
        parent_started = time.perf_counter()
        try:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate()
        except OSError as error:
            raise ExpandedStudiesError(f"Cannot launch isolated worker for {record_id}: {error}") from error
        _require(
            process.returncode == 0,
            f"Worker failed for {record_id}: {stderr.strip() or stdout.strip()}",
        )
        _, _, preliminary_resources = _load_worker_bundle(
            output,
            contract,
            coordinate,
            bindings,
            worker_binding,
            launch_context,
            process.pid,
        )
        _finalize_worker_bundle(
            output,
            coordinate,
            bindings,
            worker_binding,
            launch_context,
            process.pid,
            preliminary_resources,
            parent_started,
        )
        _, _, final_resources = _load_record(
            output, contract, coordinate, bindings, worker_binding
        )
        _require(
            final_resources["launch_evidence"]["token_sha256"] == token_sha256,
            "Final record lost its parent-issued launch token binding.",
        )
        used_token_sha256.add(token_sha256)
        del launch_context["token"]
        environment.pop(_WORKER_LAUNCH_CONTEXT_ENV, None)

    summaries: list[dict[str, Any]] = []
    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    resources: list[dict[str, Any]] = []
    for record_id in expected_ids:
        summary, rows, record_resources = _load_record(
            output,
            contract,
            coordinates[record_id],
            bindings,
            worker_binding,
        )
        summaries.append(summary)
        rows_by_id[record_id] = rows
        resources.append(record_resources)
    _require(
        len(used_token_sha256) == len(expected_ids),
        "Every scientific record must have one unique worker launch token digest.",
    )
    for name, value in _analysis_artifacts(contract, summaries, rows_by_id, resources).items():
        _atomic_write(output / name, value)
    _write_artifact_manifest(output)
    validate_run_artifacts(output, repository_root=root)
    return output


def _bootstrap_values(
    matrix: np.ndarray, data_draws: np.ndarray, model_draws: np.ndarray
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    _require(values.shape == (4, 8) and np.isfinite(values).all(), "Bootstrap requires a complete finite 4x8 matrix.")
    return values[data_draws[:, :, None], model_draws[:, None, :]].mean(axis=(1, 2))


def _estimate(point: float, replicates: np.ndarray) -> dict[str, Any]:
    _require(math.isfinite(point) and np.isfinite(replicates).all(), "Inference values must be finite.")
    return {
        "point_estimate": float(point),
        "ci95_low": float(np.quantile(replicates, 0.025, method="linear")),
        "ci95_high": float(np.quantile(replicates, 0.975, method="linear")),
        "bootstrap_replicates": int(len(replicates)),
        "confidence_interval": "raw_percentile_linear_95_percent",
    }


def _centered_p(
    replicates: np.ndarray,
    observed: float,
    boundary: float,
    *,
    alternative: str,
) -> dict[str, Any]:
    centered = replicates - float(observed)
    distance = float(observed) - float(boundary)
    if alternative == "greater":
        count = int(np.count_nonzero(centered >= distance))
    elif alternative == "less":
        count = int(np.count_nonzero(centered <= distance))
    else:
        raise ExpandedStudiesError(f"Unknown one-sided alternative: {alternative}")
    return {
        "method": "centered_bootstrap_null_tail_plus_one",
        "alternative": alternative,
        "null_boundary": float(boundary),
        "observed": float(observed),
        "extreme_replicate_count": count,
        "p_value": float((1 + count) / (len(replicates) + 1)),
    }


def _holm(tests: Sequence[Mapping[str, Any]], *, alpha: float = 0.05) -> list[dict[str, Any]]:
    indexed = [(index, float(test["p_value"])) for index, test in enumerate(tests)]
    _require(all(math.isfinite(value) and 0 <= value <= 1 for _, value in indexed), "Holm p-values must lie in [0,1].")
    ordered = sorted(indexed, key=lambda item: (item[1], item[0]))
    rejected: dict[int, bool] = {}
    adjusted: dict[int, float] = {}
    running_adjusted = 0.0
    still_rejecting = True
    total = len(ordered)
    for rank, (index, p_value) in enumerate(ordered, 1):
        threshold = alpha / (total - rank + 1)
        reject = bool(still_rejecting and p_value <= threshold)
        if not reject:
            still_rejecting = False
        rejected[index] = reject
        running_adjusted = max(running_adjusted, (total - rank + 1) * p_value)
        adjusted[index] = min(1.0, running_adjusted)
    return [
        {
            **dict(test),
            "holm_rank": next(rank for rank, (index, _) in enumerate(ordered, 1) if index == original),
            "holm_adjusted_p_value": adjusted[original],
            "holm_rejected": rejected[original],
            "family_alpha": float(alpha),
        }
        for original, test in enumerate(tests)
    ]


def _matrix(
    records: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    value: Any,
) -> np.ndarray:
    data_seeds = [int(item) for item in contract["seed_design"]["data_seeds"]]
    model_seeds = [int(item) for item in contract["seed_design"]["model_seeds"]]
    lookup = {(int(item["data_seed"]), int(item["model_seed"])): item for item in records}
    expected = {(data, model) for data in data_seeds for model in model_seeds}
    _require(set(lookup) == expected and len(records) == 32, "Crossed 4x8 factorial is incomplete.")
    return np.asarray(
        [[float(value(lookup[(data, model)])) for model in model_seeds] for data in data_seeds],
        dtype=float,
    )


def _study_a_inference(
    contract: Mapping[str, Any], summaries: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    data_draws, model_draws, draw_manifest = _bootstrap_draws(contract)
    study = contract["nonlinear_beta_study"]
    architectures: dict[str, Any] = {}
    raw_tests: list[dict[str, Any]] = []
    macro_replicates: dict[str, np.ndarray] = {}
    for kind in study["model_kinds"]:
        cells: dict[str, Any] = {}
        cell_replicates: list[np.ndarray] = []
        for cell in study["cells"]:
            group = [item for item in summaries if item["model_kind"] == kind and item["cell_id"] == cell]
            matrix = _matrix(group, contract, lambda item: bool(item["all_beta_robust_causal_certificate"]))
            replicates = _bootstrap_values(matrix, data_draws, model_draws)
            cells[cell] = _estimate(float(matrix.mean()), replicates)
            cell_replicates.append(replicates)
        macro = np.mean(np.stack(cell_replicates), axis=0)
        point = float(np.mean([cells[cell]["point_estimate"] for cell in study["cells"]]))
        macro_replicates[str(kind)] = macro
        test = _centered_p(macro, point, 0.5, alternative="greater")
        test["test_id"] = str(kind)
        raw_tests.append(test)
        architectures[str(kind)] = {
            "fixed_cells": cells,
            "equal_weight_three_cell_macro": _estimate(point, macro),
            "one_sided_test": test,
        }
    holm = _holm(raw_tests, alpha=float(study["aggregate_claim"]["holm_alpha"]))
    for item in holm:
        architectures[item["test_id"]]["one_sided_test"] = item
    claims_by_architecture: dict[str, Any] = {}
    for kind in study["model_kinds"]:
        aggregate = architectures[str(kind)]
        raw_pass = aggregate["equal_weight_three_cell_macro"]["ci95_low"] > 0.5
        holm_pass = aggregate["one_sided_test"]["holm_rejected"]
        claims_by_architecture[str(kind)] = {
            "raw_ci95_lower_strictly_greater_than_0.5": raw_pass,
            "holm_adjusted_one_sided_test_rejects": holm_pass,
            "beta_robust_architecture_claim": bool(raw_pass and holm_pass),
        }
    claims = {
        "schema_version": "expanded-studies-study-a-claims-v1",
        "architectures": claims_by_architecture,
        "overall_claim": all(item["beta_robust_architecture_claim"] for item in claims_by_architecture.values()),
        "architecture_ranking_performed": False,
    }
    inference = {
        "schema_version": "expanded-studies-study-a-inference-v1",
        "method": "shared_two_way_pigeonhole_bootstrap",
        "bootstrap_draw_manifest_sha256": _sha256_bytes(_canonical_json_bytes(draw_manifest)),
        "architectures": architectures,
        "holm_family": holm,
    }
    return inference, claims


def _study_b_derived(
    contract: Mapping[str, Any], summaries: Sequence[Mapping[str, Any]], resources: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    methods = [item["method_id"] for item in contract["cdc_tradeoff_study"]["methods"]]
    grouped: dict[tuple[str, int, int], dict[str, Mapping[str, Any]]] = {}
    for item in summaries:
        key = (str(item["cell_id"]), int(item["data_seed"]), int(item["model_seed"]))
        grouped.setdefault(key, {})[str(item["method_id"])] = item
    _require(len(grouped) == 64 and all(set(value) == set(methods) for value in grouped.values()), "Study-B one-to-one four-arm pairing is incomplete.")
    result: list[dict[str, Any]] = []
    for item in summaries:
        key = (str(item["cell_id"]), int(item["data_seed"]), int(item["model_seed"]))
        erm = grouped[key]["erm"]
        resource_value = resources[str(item["record_id"])]
        erm_resource = resources[str(erm["record_id"])]
        method_wall_time = float(resource_value["wall_time_seconds"])
        erm_wall_time = float(erm_resource["wall_time_seconds"])
        _require(
            math.isfinite(method_wall_time)
            and method_wall_time > 0
            and math.isfinite(erm_wall_time)
            and erm_wall_time > 0,
            "Study-B paired wall-time ratio requires positive finite elapsed times.",
        )
        result.append(
            {
                **dict(item),
                "absolute_strong_response_auc_difference_from_erm": abs(float(item["strong_response_auc"]) - float(erm["strong_response_auc"])),
                "absolute_final_strong_response_difference_from_erm": abs(float(item["final_strong_response"]) - float(erm["final_strong_response"])),
                "paired_wall_time_ratio_to_erm": method_wall_time / erm_wall_time,
                "process_peak_rss_bytes": int(resource_value["process_peak_rss_bytes"]),
                "static_parameter_bytes": int(resource_value["static_parameter_bytes"]),
                "cuda_status": resource_value["cuda_status"],
            }
        )
    return result


def _macro_difference(
    contract: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    left_method: str,
    right_method: str,
    value: Any,
    data_draws: np.ndarray,
    model_draws: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    cells = contract["cdc_tradeoff_study"]["cells"]
    fixed: dict[str, Any] = {}
    replicate_values: list[np.ndarray] = []
    for cell in cells:
        left = [item for item in records if item["cell_id"] == cell and item["method_id"] == left_method]
        right = [item for item in records if item["cell_id"] == cell and item["method_id"] == right_method]
        left_matrix = _matrix(left, contract, value)
        right_matrix = _matrix(right, contract, value)
        matrix = left_matrix - right_matrix
        replicates = _bootstrap_values(matrix, data_draws, model_draws)
        fixed[str(cell)] = _estimate(float(matrix.mean()), replicates)
        replicate_values.append(replicates)
    macro = np.mean(np.stack(replicate_values), axis=0)
    point = float(np.mean([fixed[str(cell)]["point_estimate"] for cell in cells]))
    return {"fixed_cells_descriptive": fixed, "equal_weight_two_cell_macro": _estimate(point, macro)}, macro


def _method_macro(
    contract: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    method: str,
    value: Any,
    data_draws: np.ndarray,
    model_draws: np.ndarray,
) -> dict[str, Any]:
    cells = contract["cdc_tradeoff_study"]["cells"]
    fixed: dict[str, Any] = {}
    replicate_values: list[np.ndarray] = []
    for cell in cells:
        group = [item for item in records if item["cell_id"] == cell and item["method_id"] == method]
        matrix = _matrix(group, contract, value)
        replicates = _bootstrap_values(matrix, data_draws, model_draws)
        fixed[str(cell)] = _estimate(float(matrix.mean()), replicates)
        replicate_values.append(replicates)
    macro = np.mean(np.stack(replicate_values), axis=0)
    point = float(np.mean([fixed[str(cell)]["point_estimate"] for cell in cells]))
    return {"fixed_cells_descriptive": fixed, "equal_weight_two_cell_macro": _estimate(point, macro)}


def _study_b_inference(
    contract: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    study = contract["cdc_tradeoff_study"]
    data_draws, model_draws, draw_manifest = _bootstrap_draws(contract)
    methods = [item["method_id"] for item in study["methods"]]
    method_estimates: dict[str, Any] = {}
    for method in methods:
        method_estimates[method] = {
            "weak_rescue": _method_macro(contract, records, method, lambda item: item["weak_rescue_score"], data_draws, model_draws),
            "trajectory_deviation": _method_macro(contract, records, method, lambda item: item["absolute_strong_response_auc_difference_from_erm"], data_draws, model_draws),
            "final_deviation": _method_macro(contract, records, method, lambda item: item["absolute_final_strong_response_difference_from_erm"], data_draws, model_draws),
        }
    comparisons: dict[str, Any] = {}
    weak_tests: list[dict[str, Any]] = []
    trajectory_tests: list[dict[str, Any]] = []
    final_tests: list[dict[str, Any]] = []
    margin = float(study["tradeoff_rule"]["weak_rescue_equivalence_margin"])
    for comparator in study["comparators"]:
        weak, weak_reps = _macro_difference(
            contract, records, "counterfactual_drift", comparator,
            lambda item: item["weak_rescue_score"], data_draws, model_draws,
        )
        trajectory, trajectory_reps = _macro_difference(
            contract, records, comparator, "counterfactual_drift",
            lambda item: item["absolute_strong_response_auc_difference_from_erm"], data_draws, model_draws,
        )
        final, final_reps = _macro_difference(
            contract, records, comparator, "counterfactual_drift",
            lambda item: item["absolute_final_strong_response_difference_from_erm"], data_draws, model_draws,
        )
        weak_point = weak["equal_weight_two_cell_macro"]["point_estimate"]
        lower = _centered_p(weak_reps, weak_point, -margin, alternative="greater")
        lower.update({"test_id": f"{comparator}:lower", "comparator": comparator, "boundary": "lower"})
        upper = _centered_p(weak_reps, weak_point, margin, alternative="less")
        upper.update({"test_id": f"{comparator}:upper", "comparator": comparator, "boundary": "upper"})
        trajectory_point = trajectory["equal_weight_two_cell_macro"]["point_estimate"]
        trajectory_test = _centered_p(trajectory_reps, trajectory_point, 0.0, alternative="greater")
        trajectory_test.update({"test_id": comparator, "comparator": comparator})
        final_point = final["equal_weight_two_cell_macro"]["point_estimate"]
        final_test = _centered_p(final_reps, final_point, 0.0, alternative="greater")
        final_test.update({"test_id": comparator, "comparator": comparator})
        weak_tests.extend((lower, upper))
        trajectory_tests.append(trajectory_test)
        final_tests.append(final_test)
        comparisons[str(comparator)] = {
            "cdc_minus_comparator_weak_rescue": weak,
            "comparator_minus_cdc_trajectory_deviation": trajectory,
            "comparator_minus_cdc_final_deviation": final,
        }
    # The loops above already follow the frozen [bloop lower, bloop upper, pcgrad lower, pcgrad upper] order.
    weak_holm = _holm(weak_tests, alpha=0.05)
    trajectory_holm = _holm(trajectory_tests, alpha=0.05)
    final_holm = _holm(final_tests, alpha=0.05)
    diagnostics_complete = {
        method: all(
            item["shadow_diagnostics"]["complete"] is True
            and item["shadow_diagnostics"]["row_count"] == 201
            for item in records
            if item["method_id"] == method
        )
        for method in methods
    }
    cost_complete = all(
        math.isfinite(float(item["paired_wall_time_ratio_to_erm"]))
        and item["process_peak_rss_bytes"] > 0
        and item["cuda_status"] == "N/A"
        for item in records
    )
    claims: dict[str, Any] = {
        "schema_version": "expanded-studies-study-b-claims-v1",
        "comparators": {},
        "instantaneous_metric_used_as_trajectory_or_final_proof": False,
    }
    weak_by_id = {item["test_id"]: item for item in weak_holm}
    trajectory_by_id = {item["test_id"]: item for item in trajectory_holm}
    final_by_id = {item["test_id"]: item for item in final_holm}
    for comparator in study["comparators"]:
        values = comparisons[comparator]
        weak_ci = values["cdc_minus_comparator_weak_rescue"]["equal_weight_two_cell_macro"]
        trajectory_ci = values["comparator_minus_cdc_trajectory_deviation"]["equal_weight_two_cell_macro"]
        final_ci = values["comparator_minus_cdc_final_deviation"]["equal_weight_two_cell_macro"]
        components = {
            "weak_rescue_equivalence_raw_ci_inside_margin": weak_ci["ci95_low"] >= -margin and weak_ci["ci95_high"] <= margin,
            "both_weak_tost_tests_pass_holm": weak_by_id[f"{comparator}:lower"]["holm_rejected"] and weak_by_id[f"{comparator}:upper"]["holm_rejected"],
            "trajectory_raw_ci_lower_strictly_positive": trajectory_ci["ci95_low"] > 0,
            "trajectory_test_passes_holm": trajectory_by_id[comparator]["holm_rejected"],
            "final_raw_ci_lower_strictly_positive": final_ci["ci95_low"] > 0,
            "final_test_passes_holm": final_by_id[comparator]["holm_rejected"],
            "complete_diagnostics": diagnostics_complete["counterfactual_drift"] and diagnostics_complete[comparator],
            "complete_cost_reporting": cost_complete,
        }
        claims["comparators"][comparator] = {
            "components": components,
            "cdc_beats_comparator_joint_tradeoff": all(components.values()),
        }
    weak_points = {
        method: method_estimates[method]["weak_rescue"]["equal_weight_two_cell_macro"]["point_estimate"]
        for method in methods
    }
    trajectory_points = {
        method: method_estimates[method]["trajectory_deviation"]["equal_weight_two_cell_macro"]["point_estimate"]
        for method in methods
    }
    final_points = {
        method: method_estimates[method]["final_deviation"]["equal_weight_two_cell_macro"]["point_estimate"]
        for method in methods
    }
    trajectory_frontier = {
        "schema_version": "expanded-studies-study-b-pareto-trajectory-v1",
        "scope": "equal_weight_two_cell_macro",
        "weak_rescue": weak_points,
        "strong_trajectory_deviation": trajectory_points,
        "frontier": pareto_frontier(weak_points, trajectory_points),
        "superiority_test": False,
    }
    final_frontier = {
        "schema_version": "expanded-studies-study-b-pareto-final-v1",
        "scope": "equal_weight_two_cell_macro",
        "weak_rescue": weak_points,
        "strong_final_deviation": final_points,
        "frontier": pareto_frontier(weak_points, final_points),
        "superiority_test": False,
    }
    inference = {
        "schema_version": "expanded-studies-study-b-inference-v1",
        "method": "shared_two_way_pigeonhole_bootstrap",
        "bootstrap_draw_manifest_sha256": _sha256_bytes(_canonical_json_bytes(draw_manifest)),
        "method_endpoints": method_estimates,
        "comparisons": comparisons,
        "test_families": {
            "weak_equivalence_tost": weak_holm,
            "trajectory_superiority": trajectory_holm,
            "final_superiority": final_holm,
        },
        "diagnostics_complete": diagnostics_complete,
        "cost_complete": cost_complete,
    }
    return inference, claims, trajectory_frontier, final_frontier


def _flatten_study_b(item: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = item["shadow_diagnostics"]
    return {
        "record_id": item["record_id"],
        "model_kind": item["model_kind"],
        "cell_id": item["cell_id"],
        "method_id": item["method_id"],
        "publication_label": item["publication_label"],
        "data_seed": item["data_seed"],
        "model_seed": item["model_seed"],
        "weak_auc_gap": item["weak_auc_gap"],
        "weak_rescue_score": item["weak_rescue_score"],
        "beta_first_hit_delay": item["beta_first_hit_delay"],
        "beta_first_hit_delay_status": item["beta_first_hit_delay_status"],
        "final_both_weak_response": item["final_both_weak_response"],
        "strong_response_auc": item["strong_response_auc"],
        "final_strong_response": item["final_strong_response"],
        "absolute_strong_response_auc_difference_from_erm": item["absolute_strong_response_auc_difference_from_erm"],
        "absolute_final_strong_response_difference_from_erm": item["absolute_final_strong_response_difference_from_erm"],
        "diagnostic_status": diagnostics["status"],
        "target_attainment_fraction": diagnostics["target_attainment_fraction"],
        "feasibility_fraction": diagnostics["feasibility_fraction"],
        "cap_binding_fraction": diagnostics["cap_binding_fraction"],
        "max_absolute_strong_drift_change": diagnostics["max_absolute_strong_drift_change"],
        "paired_wall_time_ratio_to_erm": item["paired_wall_time_ratio_to_erm"],
        "process_peak_rss_bytes": item["process_peak_rss_bytes"],
        "static_parameter_bytes": item["static_parameter_bytes"],
        "cuda_status": item["cuda_status"],
    }


def _analysis_artifacts(
    contract: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    rows_by_id: Mapping[str, Sequence[Mapping[str, Any]]],
    resources: Sequence[Mapping[str, Any]],
) -> dict[str, bytes]:
    del rows_by_id  # Record summaries were deterministically recomputed from these rows.
    ids = _expected_record_ids(contract)
    order = {record_id: index for index, record_id in enumerate(ids["study_a"] + ids["study_b"])}
    ordered = sorted(summaries, key=lambda item: order[str(item["record_id"])])
    study_a = [item for item in ordered if item["study"] == "study_a"]
    study_b = [item for item in ordered if item["study"] == "study_b"]
    _require(len(study_a) == 192 and len(study_b) == 256, "Analysis record counts are incomplete.")
    profile_rows: list[dict[str, Any]] = []
    for summary in study_a:
        for point in summary["first_hit_profile"]:
            profile_rows.append(
                {
                    "record_id": summary["record_id"],
                    "model_kind": summary["model_kind"],
                    "cell_id": summary["cell_id"],
                    "data_seed": summary["data_seed"],
                    "model_seed": summary["model_seed"],
                    **point,
                    "outcome_suppression": summary["outcome_suppression"],
                    "all_beta_robust_causal_certificate": summary["all_beta_robust_causal_certificate"],
                }
            )
    a_inference, a_claims = _study_a_inference(contract, study_a)
    resources_by_id = {str(item["record_id"]): item for item in resources}
    _require(len(resources_by_id) == 448, "Resource records are incomplete.")
    derived_b = _study_b_derived(contract, study_b, resources_by_id)
    b_inference, b_claims, pareto_trajectory, pareto_final = _study_b_inference(contract, derived_b)
    flat_b = [_flatten_study_b(item) for item in derived_b]
    diagnostics_fields = [
        "record_id", "method_id", "cell_id", "data_seed", "model_seed",
        "diagnostic_status", "target_attainment_fraction", "feasibility_fraction",
        "cap_binding_fraction", "max_absolute_strong_drift_change",
    ]
    cost_fields = [
        "record_id", "method_id", "cell_id", "data_seed", "model_seed",
        "paired_wall_time_ratio_to_erm", "process_peak_rss_bytes",
        "static_parameter_bytes", "cuda_status",
    ]
    resource_usage = {
        "schema_version": "expanded-studies-resource-usage-v1",
        "record_count": len(resources_by_id),
        "total_parent_wall_time_seconds": float(sum(float(item["parent_wall_time_seconds"]) for item in resources_by_id.values())),
        "maximum_process_peak_rss_bytes": max(int(item["process_peak_rss_bytes"]) for item in resources_by_id.values()),
        "records": {record_id: resources_by_id[record_id] for record_id in ids["study_a"] + ids["study_b"]},
        "cuda_status": "N/A",
    }
    return {
        "study_a_first_hit_profile.csv": _csv_bytes(profile_rows, list(profile_rows[0])),
        "study_a_record_summary.json": _canonical_json_bytes({"schema_version": "expanded-studies-study-a-record-collection-v1", "records": study_a}),
        "study_a_inference.json": _canonical_json_bytes(a_inference),
        "study_a_claims.json": _canonical_json_bytes(a_claims),
        "study_b_record_summary.csv": _csv_bytes(flat_b, list(flat_b[0])),
        "study_b_record_summary.json": _canonical_json_bytes({"schema_version": "expanded-studies-study-b-record-collection-v1", "records": derived_b}),
        "study_b_diagnostics.csv": _csv_bytes([{field: item[field] for field in diagnostics_fields} for item in flat_b], diagnostics_fields),
        "study_b_cost.csv": _csv_bytes([{field: item[field] for field in cost_fields} for item in flat_b], cost_fields),
        "study_b_inference.json": _canonical_json_bytes(b_inference),
        "study_b_claims.json": _canonical_json_bytes(b_claims),
        "study_b_pareto_trajectory.json": _canonical_json_bytes(pareto_trajectory),
        "study_b_pareto_final.json": _canonical_json_bytes(pareto_final),
        "resource_usage.json": _canonical_json_bytes(resource_usage),
    }


def _write_artifact_manifest(directory: Path) -> None:
    manifest_path = directory / "artifact_manifest.json"
    sidecar_path = directory / "artifact_manifest.sha256"
    files = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        if path in {manifest_path, sidecar_path}:
            continue
        _require(not path.is_symlink(), f"Symlinked run artifact: {path}")
        if path.is_file():
            files.append(_descriptor(path, name=path.relative_to(directory).as_posix()))
    manifest = {
        "schema_version": "expanded-studies-artifact-manifest-v1",
        "hash_algorithm": "sha256",
        "file_count": len(files),
        "files": files,
    }
    _atomic_write(manifest_path, _canonical_json_bytes(manifest))
    _write_sidecar(manifest_path)


def _copied_preflight(
    run_directory: Path,
    contract: Mapping[str, Any],
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = run_directory / "preflight.manifest.json"
    _verify_sidecar(manifest_path)
    preflight = _strict_json(manifest_path)
    expected_keys = {
        "schema_version",
        "study_id",
        "contract_revision",
        "publication_scope",
        "created_utc",
        "model_construction_scope",
        "model_constructed",
        "model_forward_called",
        "optimizer_constructed",
        "training_performed",
        "model_outcomes_inspected",
        "scientific_output_created",
        "environment",
        "git",
        "files",
        "study_a_expected_record_ids",
        "study_b_expected_record_ids",
        "ready_for_authorization",
        "blockers",
        "execution_authorized",
    }
    _require(set(preflight) == expected_keys, "Copied preflight manifest schema is not exact.")
    _validate_timestamp(preflight["created_utc"], "Copied preflight created_utc")
    _require(
        preflight["schema_version"] == contract["preflight"]["schema_version"]
        and preflight["study_id"] == contract["study_id"]
        and preflight["contract_revision"] == contract["contract_revision"]
        and preflight["publication_scope"] == contract["publication_scope"],
        "Copied preflight identity changed.",
    )
    _require(
        preflight["model_construction_scope"] == "initialization_hashing_only"
        and preflight["model_constructed"] is True
        and preflight["model_forward_called"] is False
        and preflight["optimizer_constructed"] is False
        and preflight["training_performed"] is False
        and preflight["model_outcomes_inspected"] is False
        and preflight["scientific_output_created"] is False
        and preflight["execution_authorized"] is False,
        "Copied preflight audit crossed the initialization-only boundary.",
    )
    expected_environment = _current_environment(contract, root)
    _require(preflight["environment"] == expected_environment, "Copied preflight environment drifted.")
    git = preflight["git"]
    _require(
        isinstance(git, Mapping)
        and set(git) == {"commit", "clean", "status_porcelain"}
        and git["clean"] is True
        and git["status_porcelain"] == []
        and isinstance(git["commit"], str)
        and len(git["commit"]) == 40
        and all(character in "0123456789abcdef" for character in git["commit"]),
        "Copied preflight git audit is not one clean commit.",
    )

    source = _strict_json(run_directory / "source_manifest.json")
    records = _strict_json(run_directory / "record_manifest.json")
    bootstrap = _strict_json(run_directory / "bootstrap_draw_manifest.json")
    initializations = _strict_json(run_directory / "initialization_manifest.json")
    _validate_initialization_manifest(contract, initializations)
    expected_records = _record_manifest(contract)
    _require(_canonical_json_bytes(records) == _canonical_json_bytes(expected_records), "Copied record manifest changed.")
    _, _, expected_bootstrap = _bootstrap_draws(contract)
    _require(_canonical_json_bytes(bootstrap) == _canonical_json_bytes(expected_bootstrap), "Copied bootstrap manifest changed.")
    _require(
        _canonical_json_bytes(source) == _canonical_json_bytes(_source_manifest(contract, root)),
        "Copied source manifest is not reproducible.",
    )
    _require(
        preflight["study_a_expected_record_ids"] == expected_records["study_a_expected_record_ids"]
        and preflight["study_b_expected_record_ids"] == expected_records["study_b_expected_record_ids"],
        "Copied preflight record IDs changed.",
    )
    mapping = {
        "execution-contract.json": "contract.resolved.json",
        "historical-contract.json": "historical-contract.json",
        "environment-lock.txt": "environment-lock.txt",
        "source_manifest.json": "source_manifest.json",
        "record_manifest.json": "record_manifest.json",
        "bootstrap_draw_manifest.json": "bootstrap_draw_manifest.json",
        "initialization_manifest.json": "initialization_manifest.json",
    }
    file_entries = preflight["files"]
    _require(
        isinstance(file_entries, list)
        and len(file_entries) == len(mapping)
        and all(isinstance(entry, Mapping) for entry in file_entries)
        and [entry.get("path") for entry in file_entries] == list(mapping),
        "Copied preflight file declarations changed.",
    )
    for entry in file_entries:
        _require(
            set(entry) == {"path", "size", "sha256"}
            and type(entry["size"]) is int
            and entry["size"] >= 0
            and _is_sha256(entry["sha256"]),
            "Copied preflight file descriptor schema changed.",
        )
        path = run_directory / mapping[entry["path"]]
        _require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == entry["size"]
            and _sha256_file(path) == entry["sha256"],
            f"Copied preflight-bound file drift: {entry['path']}",
        )
    _require(
        (run_directory / "historical-contract.json").read_bytes()
        == (root / contract["historical_contract"]["path"]).read_bytes(),
        "Copied historical contract drifted from the frozen repository file.",
    )
    _require(
        (run_directory / "environment-lock.txt").read_bytes()
        == (root / contract["execution_environment"]["environment_lock_path"]).read_bytes(),
        "Copied environment lock drifted from the frozen repository file.",
    )
    _require(
        preflight["ready_for_authorization"] is True and preflight["blockers"] == [],
        "Copied preflight was not authorization-ready.",
    )
    return preflight, source, records, bootstrap, initializations


def validate_run_artifacts(
    run_directory: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Rehash and deterministically recompute a complete authorized 448-record run."""
    root = Path(repository_root).resolve()
    run_directory = _workspace_path(run_directory, root, role="Scientific run directory")
    _tree_without_symlinks(run_directory, role="Scientific run directory")
    _require(run_directory.is_dir(), f"Run directory is missing: {run_directory}")
    _require(
        {path.name for path in run_directory.iterdir()} == set(_RUN_FILES) | {"records"},
        "Top-level run artifact inventory has missing or extra entries.",
    )
    manifest_path = run_directory / "artifact_manifest.json"
    _verify_sidecar(manifest_path)
    artifact_manifest = _strict_json(manifest_path)
    _require(
        set(artifact_manifest) == {"schema_version", "hash_algorithm", "file_count", "files"}
        and artifact_manifest["schema_version"] == "expanded-studies-artifact-manifest-v1"
        and artifact_manifest["hash_algorithm"] == "sha256",
        "Artifact manifest schema changed.",
    )
    declared: dict[str, Mapping[str, Any]] = {}
    for entry in artifact_manifest["files"]:
        path_name = entry.get("path")
        _require(path_name not in declared, f"Duplicate artifact path: {path_name}")
        declared[str(path_name)] = entry
        _verify_descriptor(run_directory, entry)
    actual = {
        path.relative_to(run_directory).as_posix()
        for path in run_directory.rglob("*")
        if path.is_file() and path not in {manifest_path, run_directory / "artifact_manifest.sha256"}
    }
    _require(set(declared) == actual and artifact_manifest["file_count"] == len(actual), "Artifact manifest scope is not exact.")

    contract = _load_contract(run_directory / "contract.resolved.json", repository_root=root)
    preflight, source, records, bootstrap, initializations = _copied_preflight(
        run_directory, contract, root
    )
    del bootstrap, initializations
    worker_binding = _worker_executable_binding(source, root)
    expected_bindings = {
        "execution_contract_sha256": _sha256_file(run_directory / "contract.resolved.json"),
        "preflight_manifest_sha256": _sha256_file(run_directory / "preflight.manifest.json"),
        "historical_contract_sha256": _sha256_file(run_directory / "historical-contract.json"),
        "source_manifest_sha256": _sha256_file(run_directory / "source_manifest.json"),
        "source_aggregate_sha256": source["aggregate_sha256"],
        "environment_lock_sha256": _sha256_file(run_directory / "environment-lock.txt"),
        "record_manifest_sha256": _sha256_file(run_directory / "record_manifest.json"),
        "bootstrap_draw_manifest_sha256": _sha256_file(run_directory / "bootstrap_draw_manifest.json"),
        "initialization_manifest_sha256": _sha256_file(run_directory / "initialization_manifest.json"),
        "study_a_expected_record_ids": records["study_a_expected_record_ids"],
        "study_b_expected_record_ids": records["study_b_expected_record_ids"],
    }
    seal = _strict_json(run_directory / "authorization.seal.json")
    sealed_git = _validate_seal_document(contract, seal, expected_bindings)
    provenance = _strict_json(run_directory / "provenance.json")
    input_bindings = {
        "execution_contract_sha256": expected_bindings["execution_contract_sha256"],
        "preflight_manifest_sha256": expected_bindings["preflight_manifest_sha256"],
        "authorization_seal_sha256": _sha256_file(run_directory / "authorization.seal.json"),
        "source_aggregate_sha256": expected_bindings["source_aggregate_sha256"],
        "record_manifest_sha256": expected_bindings["record_manifest_sha256"],
        "bootstrap_draw_manifest_sha256": expected_bindings["bootstrap_draw_manifest_sha256"],
        "initialization_manifest_sha256": expected_bindings["initialization_manifest_sha256"],
    }
    expected_provenance = {
        "schema_version": "expanded-studies-provenance-v1",
        "study_id": contract["study_id"],
        "publication_scope": contract["publication_scope"],
        **input_bindings,
        "historical_contract_sha256": expected_bindings["historical_contract_sha256"],
        "environment_lock_sha256": expected_bindings["environment_lock_sha256"],
        "study_a_expected_record_ids": records["study_a_expected_record_ids"],
        "study_b_expected_record_ids": records["study_b_expected_record_ids"],
        "git_commit": sealed_git["commit"],
    }
    _require(provenance == expected_provenance, "Run provenance bindings are inconsistent.")
    runtime = _strict_json(run_directory / "environment.json")
    _validate_runtime_environment_document(
        runtime,
        expected_environment=preflight["environment"],
        expected_git=preflight["git"],
    )
    _require(
        preflight["git"]["commit"] == sealed_git["commit"],
        "Runtime environment/git, preflight, and seal bindings are inconsistent.",
    )

    coordinates = _record_coordinates(contract)
    expected_ids = records["study_a_expected_record_ids"] + records["study_b_expected_record_ids"]
    records_root = run_directory / "records"
    _require(
        records_root.is_dir()
        and {path.name for path in records_root.iterdir()} == set(expected_ids)
        and all(path.is_dir() and not path.is_symlink() for path in records_root.iterdir()),
        "Record directory factorial is missing, duplicate, or extra.",
    )
    summaries: list[dict[str, Any]] = []
    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    resource_values: list[dict[str, Any]] = []
    launch_token_sha256: set[str] = set()
    for record_id in expected_ids:
        summary, rows, record_resources = _load_record(
            run_directory,
            contract,
            coordinates[record_id],
            input_bindings,
            worker_binding,
        )
        token_sha256 = str(record_resources["launch_evidence"]["token_sha256"])
        _require(
            token_sha256 not in launch_token_sha256,
            "Scientific records reuse a worker launch token digest.",
        )
        launch_token_sha256.add(token_sha256)
        summaries.append(summary)
        rows_by_id[record_id] = rows
        resource_values.append(record_resources)
    _require(
        len(launch_token_sha256) == len(expected_ids),
        "Every scientific record must have unique sealed-script launch evidence.",
    )
    recomputed = _analysis_artifacts(contract, summaries, rows_by_id, resource_values)
    for name, value in recomputed.items():
        _require((run_directory / name).read_bytes() == value, f"Derived artifact differs from deterministic recomputation: {name}")
    claims_a = _strict_json(run_directory / "study_a_claims.json")
    claims_b = _strict_json(run_directory / "study_b_claims.json")
    return {
        "valid": True,
        "schema_version": contract["schema_version"],
        "study_a_record_count": 192,
        "study_b_method_record_count": 256,
        "total_scientific_record_count": 448,
        "artifact_file_count": artifact_manifest["file_count"],
        "study_a_overall_claim": claims_a["overall_claim"],
        "study_b_comparator_claims": {
            name: value["cdc_beats_comparator_joint_tradeoff"]
            for name, value in claims_b["comparators"].items()
        },
    }


def run_engineering_smoke(
    contract_path: Path,
    *,
    output_directory: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Run only the excluded two-step toy ERM/CDC engineering check."""
    root = Path(repository_root).resolve()
    contract = _load_contract(Path(contract_path), repository_root=root)
    _configure_determinism(contract, root)
    output = _workspace_path(output_directory, root, role="Engineering smoke output")
    _require(not output.exists(), f"Engineering smoke output already exists: {output}")
    smoke = contract["engineering_smoke"]
    scientific = set(contract["seed_design"]["data_seeds"]) | set(contract["seed_design"]["model_seeds"])
    _require(smoke["data_seed"] not in scientific and smoke["model_seed"] not in scientific, "Smoke seeds overlap scientific seeds.")
    cell = "rho4_lag2"
    spec = _task_spec(contract, cell, smoke=True)
    both, weak = make_paired_task(spec, int(smoke["data_seed"]))
    all_rows: list[dict[str, Any]] = []
    method_summaries: list[dict[str, Any]] = []
    training = {
        "objective": "cross_entropy",
        "optimizer": "sgd",
        "full_batch": True,
        "steps": int(smoke["steps"]),
        "learning_rate": 0.01,
        "log_every": int(smoke["log_every"]),
        "weight_decay": 0.0,
        "gradient_clip": None,
        "paired_mode": "lockstep",
        "exact_response_drift": True,
    }
    for method in smoke["methods"]:
        mitigation = {
            "method": method,
            "objective": "cross_entropy",
            "feasibility_epsilon": 1e-10,
            "max_alpha": None,
            "ema_decay": 0.9,
        }
        history, _ = train_paired(
            {"width": int(smoke["width"])},
            both,
            weak,
            training,
            mitigation,
            seed=int(smoke["model_seed"]),
            kind="tanh",
        )
        coordinate = {
            "record_id": f"engineering-smoke-{method}",
            "study": "engineering_smoke",
            "model_kind": "tanh",
            "cell_id": "toy",
            "method_id": method,
            "data_seed": int(smoke["data_seed"]),
            "model_seed": int(smoke["model_seed"]),
        }
        rows = _normalize_history(history, coordinate)
        all_rows.extend(rows)
        arms = {condition: [row for row in rows if row["condition"] == condition] for condition in ("both", "weak_only")}
        method_summaries.append(
            {
                "method": method,
                "logged_rows": len(rows),
                "final_both_weak_response": arms["both"][-1]["m_w"],
                "final_weak_only_weak_response": arms["weak_only"][-1]["m_w"],
            }
        )
    summary = {
        "schema_version": "expanded-studies-engineering-smoke-v1",
        "scope": smoke["scope_label"],
        "scientific_record": False,
        "acceptance_eligible": False,
        "evidence_eligibility": False,
        "scientific_id_generated": False,
        "scientific_seeds_used": False,
        "data_source": smoke["task"],
        "data_seed": int(smoke["data_seed"]),
        "model_seed": int(smoke["model_seed"]),
        "methods": method_summaries,
    }
    output.mkdir(parents=True)
    _atomic_write(output / "smoke.trajectory.csv", _csv_bytes(all_rows, TRAJECTORY_FIELDS))
    _atomic_write(output / "smoke.summary.json", _canonical_json_bytes(summary))
    _write_artifact_manifest(output)
    return output


__all__ = [
    "AuthorizedExecution",
    "ExpandedStudiesError",
    "VerifiedPreflight",
    "build_preflight",
    "run_engineering_smoke",
    "run_scientific_studies",
    "validate_run_artifacts",
    "verify_authorization",
    "verify_preflight",
]
