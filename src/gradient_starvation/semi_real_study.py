"""Auditable execution machinery for the frozen semi-real generated-cue study.

The historical v1 preregistration remains blocked.  This module separates three
operations deliberately:

* preflight may acquire data only when its caller passes ``allow_download=True``;
  it hashes inputs and freezes selected indices but never constructs a model;
* scientific execution requires an external human authorization seal whose
  bindings are verified before output, model, or optimizer construction;
* engineering smoke execution uses generated data and excluded seeds and is
  permanently ineligible for scientific acceptance.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import resource
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import causal_metrics, n_sign_changes, sign_crossing_time
from .semi_real import (
    SemiRealBatch,
    SemiRealCNN,
    SemiRealProbe,
    SemiRealTaskSpec,
    make_paired_semi_real_task,
    make_semi_real_probe,
    prepare_core_images,
    semi_real_evaluation_accuracies,
    semi_real_statistics,
)
from .theory import discrete_crossover_certificate


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SEMI_REAL_WORKSPACE_RELATIVE_ROOT = Path("data/semi_real_generated_cue_v1")
CONTRACT_SEMANTIC_SHA256 = "6c7315503c5729caae20867b5aaaedfa9e937e5e7e8c92d692215efece058af7"
SHA256_LENGTH = 64
RECORD_TRAJECTORY_FIELDS = [
    "record_id",
    "dataset",
    "data_seed",
    "model_seed",
    "condition",
    "step",
    "tau",
    "loss",
    "accuracy",
    "core_response",
    "core_gain",
    "cue_response",
    "cue_gain",
    "core_drift",
    "cue_drift",
]
PREFLIGHT_MANIFEST_KEYS = {
    "schema_version",
    "study_id",
    "publication_scope",
    "created_utc",
    "download_requested",
    "model_constructed",
    "optimizer_constructed",
    "model_outcomes_inspected",
    "contract",
    "historical_contract",
    "environment_lock",
    "environment",
    "source_manifest",
    "raw_data_manifest",
    "selection_manifest",
    "git",
    "expected_record_ids",
    "ready_for_authorization",
    "blockers",
}


class SemiRealStudyError(RuntimeError):
    """Fail-closed error raised by semi-real preflight, execution, or analysis."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SemiRealStudyError(message)


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SemiRealStudyError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def load_strict_json(path: Path) -> dict[str, Any]:
    """Load one strict JSON mapping, rejecting duplicate keys and NaN/Infinity."""
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SemiRealStudyError(f"Non-finite JSON value: {token}.")
            ),
        )
    except SemiRealStudyError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SemiRealStudyError(f"Cannot parse {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} must contain a JSON mapping.")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON serialization used by every content-addressed manifest."""
    return (
        json.dumps(
            _json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_safe(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise SemiRealStudyError(f"Cannot hash {path}: {error}") from error
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    """Return the MD5 used by frozen official torchvision resource metadata."""
    digest = hashlib.md5()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise SemiRealStudyError(f"Cannot verify official archive {path}: {error}") from error
    return digest.hexdigest()


def _file_descriptor(path: Path, *, relative_path: str | None = None) -> dict[str, Any]:
    _require(path.is_file(), f"Required file is missing: {path}")
    _require(not path.is_symlink(), f"Symlinked scientific input is forbidden: {path}")
    return {
        "path": relative_path if relative_path is not None else path.name,
        "size": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
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
        return
    _atomic_write_bytes(path, value)


def _parse_exact_lock(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise SemiRealStudyError(f"Cannot read environment lock {path}: {error}") from error
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        _require(
            line.count("==") == 1 and not any(token in line for token in (";", " @ ", "[", "]")),
            f"Environment lock line {line_number} is not one exact name==version pin.",
        )
        name, version = line.split("==", 1)
        normalized = name.strip().lower().replace("_", "-")
        _require(normalized and version.strip(), f"Invalid environment lock line {line_number}.")
        _require(normalized not in packages, f"Duplicate environment lock package: {normalized}.")
        packages[normalized] = version.strip()
    return packages


def _semantic_contract_sha256(contract: Mapping[str, Any]) -> str:
    return _sha256_bytes(canonical_json_bytes(contract))


def validate_execution_contract(
    contract: Mapping[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    verify_linked_files: bool = True,
) -> dict[str, Any]:
    """Validate every contract field by a canonical whole-document fingerprint.

    The semantic fingerprint covers the entire mapping, including all nested fields
    and list order.  Consequently an added, removed, or changed material field is
    rejected rather than being silently ignored by a partial validator.
    """
    _require(isinstance(contract, Mapping), "Execution contract must be a mapping.")
    _require(
        contract.get("schema_version") == "semi-real-execution-v1",
        "Unsupported semi-real execution schema.",
    )
    _require(contract.get("study_id") == "semi-real-generated-cue-v1", "Study ID changed.")
    _require(contract.get("publication_scope") == "semi-real", "Publication scope changed.")
    semantic_sha256 = _semantic_contract_sha256(contract)
    _require(
        semantic_sha256 == CONTRACT_SEMANTIC_SHA256,
        "Execution contract differs from the fully validated semi-real-execution-v1 contract.",
    )
    _require(
        contract["local_workspace"]["root"]
        == SEMI_REAL_WORKSPACE_RELATIVE_ROOT.as_posix(),
        "Semi-real local workspace root changed.",
    )

    historical = contract["historical_contract"]
    environment = contract["execution_environment"]
    if verify_linked_files:
        _require_workspace_is_git_ignored(repository_root)
        historical_path = repository_root / str(historical["path"])
        _require(
            sha256_file(historical_path) == historical["sha256"],
            "Historical blocked v1 preregistration hash drifted.",
        )
        historical_value = load_strict_json(historical_path)
        _require(
            historical_value.get("schema_version") == "semi-real-preregistration-v1"
            and historical_value.get("execution_authorized") is False
            and historical_value.get("status") == "design_frozen_execution_blocked",
            "Historical v1 contract is no longer the blocked artifact.",
        )
        lock_path = repository_root / str(environment["environment_lock_path"])
        _require(
            sha256_file(lock_path) == environment["environment_lock_sha256"],
            "Semi-real environment lock hash drifted.",
        )
        pins = _parse_exact_lock(lock_path)
        for name, version in environment["required_exact_packages"].items():
            _require(pins.get(name) == version, f"Environment lock must pin {name}=={version}.")

    expected_records = expected_record_ids(contract)
    return {
        "valid": True,
        "schema_version": contract["schema_version"],
        "publication_scope": "semi-real",
        "semantic_sha256": semantic_sha256,
        "record_count": len(expected_records),
        "execution_authorized": False,
        "authorization_status": "external_seal_required",
    }


def load_execution_contract(
    path: Path, *, repository_root: Path = REPOSITORY_ROOT
) -> dict[str, Any]:
    contract = load_strict_json(path)
    validate_execution_contract(contract, repository_root=repository_root)
    return contract


def expected_record_ids(contract: Mapping[str, Any]) -> list[str]:
    template = str(contract["seed_design"]["record_id_format"])
    identifiers = [
        template.format(dataset=dataset["name"], data_seed=data_seed, model_seed=model_seed)
        for dataset in contract["datasets"]
        for data_seed in contract["seed_design"]["data_seeds"]
        for model_seed in contract["seed_design"]["model_seeds"]
    ]
    _require(len(identifiers) == len(set(identifiers)) == 64, "Expected record IDs are not 64 unique IDs.")
    return identifiers


def expected_record_coordinates(
    contract: Mapping[str, Any]
) -> dict[str, tuple[str, int, int]]:
    template = str(contract["seed_design"]["record_id_format"])
    coordinates = {
        template.format(dataset=dataset["name"], data_seed=data_seed, model_seed=model_seed): (
            str(dataset["name"]),
            int(data_seed),
            int(model_seed),
        )
        for dataset in contract["datasets"]
        for data_seed in contract["seed_design"]["data_seeds"]
        for model_seed in contract["seed_design"]["model_seeds"]
    }
    _require(len(coordinates) == 64, "Expected record coordinate map is incomplete.")
    return coordinates


def _git_executable() -> str:
    executable = shutil.which("git")
    if executable:
        return executable
    fallback = Path("/usr/bin/git")
    _require(fallback.is_file(), "git is required for auditable preflight.")
    return str(fallback)


def _git_output(repository_root: Path, arguments: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            [_git_executable(), *arguments],
            cwd=repository_root,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SemiRealStudyError(f"Cannot inspect git state: {error}") from error


def git_state(repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    commit = _git_output(repository_root, ["rev-parse", "HEAD"])
    status = _git_output(repository_root, ["status", "--porcelain", "--untracked-files=all"])
    _require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "Invalid git commit.")
    return {"commit": commit, "clean": not bool(status), "status_porcelain": status.splitlines()}


def _current_environment(
    contract: Mapping[str, Any], repository_root: Path = REPOSITORY_ROOT
) -> dict[str, Any]:
    expected = contract["execution_environment"]
    _require(platform.python_version() == expected["python_version"], "Python version does not match lock.")
    lock_path = repository_root / expected["environment_lock_path"]
    pins = _parse_exact_lock(lock_path)
    verified: dict[str, str] = {}
    for package, required in pins.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise SemiRealStudyError(f"Locked package is not installed: {package}") from error
        _require(installed == required, f"Installed {package}=={installed}; locked {required}.")
        verified[package] = installed
    for package, required in expected["required_exact_packages"].items():
        _require(verified.get(package) == required, f"Required exact package changed: {package}.")
    return {
        "python": platform.python_version(),
        "locked_package_count": len(verified),
        "locked_packages_verified": True,
        "required_packages": {
            package: verified[package]
            for package in sorted(expected["required_exact_packages"])
        },
        "device": "cpu",
        "dtype": "float32",
        "deterministic_algorithms": True,
        "intraop_threads": 1,
        "interop_threads": 1,
    }


def build_source_manifest(
    contract: Mapping[str, Any], repository_root: Path = REPOSITORY_ROOT
) -> dict[str, Any]:
    relative_paths = {str(relative) for relative in contract["preflight"]["source_files"]}
    for pattern in contract["preflight"]["source_globs"]:
        relative_paths.update(
            path.relative_to(repository_root).as_posix()
            for path in repository_root.glob(str(pattern))
            if path.is_file()
        )
    entries: list[dict[str, Any]] = []
    for relative in sorted(relative_paths):
        path = repository_root / relative
        entries.append(_file_descriptor(path, relative_path=relative))
    _require(entries, "Executable source manifest is empty.")
    return {
        "schema_version": "semi-real-source-manifest-v1",
        "hash_algorithm": "sha256",
        "files": entries,
        "aggregate_sha256": _sha256_bytes(canonical_json_bytes(entries)),
    }


def build_raw_data_manifest(
    contract: Mapping[str, Any], data_root: Path
) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    _require(not data_root.is_symlink(), f"Data root symlink is forbidden: {data_root}")
    root = data_root.resolve()
    for dataset in contract["datasets"]:
        dataset_directory = root / dataset["torchvision_directory"]
        raw_directory = dataset_directory / "raw"
        _require(
            dataset_directory.is_dir()
            and raw_directory.is_dir()
            and not dataset_directory.is_symlink()
            and not raw_directory.is_symlink(),
            f"Official raw data directory is missing or symlinked: {raw_directory}",
        )
        files: list[dict[str, Any]] = []
        for path in sorted(raw_directory.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_symlink():
                raise SemiRealStudyError(f"Symlinked raw data is forbidden: {path}")
            if path.is_file():
                files.append(_file_descriptor(path, relative_path=path.relative_to(root).as_posix()))
        _require(files, f"No raw files found for {dataset['name']}.")
        names = {Path(item["path"]).name for item in files}
        missing = set(dataset["required_extracted_raw_files"]) - names
        _require(not missing, f"{dataset['name']} is missing extracted raw files: {sorted(missing)}")
        official_archives: list[dict[str, Any]] = []
        for archive in dataset["official_archives"]:
            archive_path = raw_directory / archive["filename"]
            _require(
                archive["filename"] in names
                and archive_path.is_file()
                and not archive_path.is_symlink(),
                f"{dataset['name']} official archive is missing: {archive['filename']}",
            )
            observed_md5 = md5_file(archive_path)
            _require(
                observed_md5 == archive["md5"],
                f"{dataset['name']} official archive checksum failed: {archive['filename']}",
            )
            official_archives.append(
                {
                    "filename": archive["filename"],
                    "official_md5": archive["md5"],
                    "verified": True,
                }
            )
        datasets.append(
            {
                "name": dataset["name"],
                "official_archives": official_archives,
                "files": files,
                "aggregate_sha256": _sha256_bytes(canonical_json_bytes(files)),
            }
        )
    return {
        "schema_version": "semi-real-raw-data-manifest-v1",
        "hash_algorithm": "sha256",
        "file_scope": contract["preflight"]["raw_file_scope"],
        "datasets": datasets,
    }


@dataclass(frozen=True)
class OfficialSplit:
    """Official split decoded only from content-addressed extracted IDX bytes."""

    data: torch.Tensor
    targets: torch.Tensor


def _decode_idx_images(path: Path, expected_shape: Sequence[int]) -> torch.Tensor:
    _require(path.is_file() and not path.is_symlink(), f"Required IDX image file is missing: {path}")
    payload = path.read_bytes()
    _require(len(payload) >= 16, f"Truncated IDX image header: {path}")
    magic, count, rows, columns = struct.unpack(">IIII", payload[:16])
    _require(magic == 2051, f"Unexpected IDX image magic in {path}.")
    _require([rows, columns] == list(expected_shape), f"Unexpected IDX image shape in {path}.")
    expected_size = 16 + count * rows * columns
    _require(len(payload) == expected_size, f"IDX image byte count changed in {path}.")
    values = np.frombuffer(payload, dtype=np.uint8, offset=16).copy()
    return torch.from_numpy(values.reshape(count, rows, columns)).contiguous()


def _decode_idx_labels(path: Path) -> torch.Tensor:
    _require(path.is_file() and not path.is_symlink(), f"Required IDX label file is missing: {path}")
    payload = path.read_bytes()
    _require(len(payload) >= 8, f"Truncated IDX label header: {path}")
    magic, count = struct.unpack(">II", payload[:8])
    _require(magic == 2049, f"Unexpected IDX label magic in {path}.")
    _require(len(payload) == 8 + count, f"IDX label byte count changed in {path}.")
    return torch.from_numpy(np.frombuffer(payload, dtype=np.uint8, offset=8).copy()).long()


def _load_official_datasets(
    contract: Mapping[str, Any], data_root: Path, *, download: bool
) -> dict[str, dict[str, OfficialSplit]]:
    """Optionally acquire with torchvision, then always decode sealed raw IDX files.

    Legacy ``processed/*.pt`` caches are never read, so they cannot override the
    raw files bound into the preflight and authorization seal.
    """
    if download:
        try:
            from torchvision.datasets.utils import download_and_extract_archive
        except ImportError as error:  # pragma: no cover - exact lock catches this
            raise SemiRealStudyError("The exact torchvision acquisition utility is unavailable.") from error
        for specification in contract["datasets"]:
            dataset_directory = data_root / specification["torchvision_directory"]
            _require(
                not dataset_directory.exists() or not dataset_directory.is_symlink(),
                f"Refusing symlinked acquisition directory: {dataset_directory}",
            )
            raw_directory = dataset_directory / "raw"
            raw_directory.mkdir(parents=True, exist_ok=True)
            _require(not raw_directory.is_symlink(), f"Refusing symlinked raw directory: {raw_directory}")
            for archive in specification["official_archives"]:
                errors: list[str] = []
                for mirror in specification["official_download_mirrors"]:
                    try:
                        download_and_extract_archive(
                            f"{mirror}{archive['filename']}",
                            download_root=str(raw_directory),
                            filename=archive["filename"],
                            md5=archive["md5"],
                        )
                        break
                    except Exception as error:  # pragma: no cover - network/provider dependent
                        errors.append(str(error))
                else:
                    raise SemiRealStudyError(
                        f"Official acquisition failed for {specification['name']}/"
                        f"{archive['filename']}: {' | '.join(errors)}"
                    )
                _require(
                    md5_file(raw_directory / archive["filename"]) == archive["md5"],
                    f"Acquired official archive checksum failed: {archive['filename']}",
                )

    loaded: dict[str, dict[str, OfficialSplit]] = {}
    for specification in contract["datasets"]:
        raw_directory = data_root / specification["torchvision_directory"] / "raw"
        shape = specification["raw_image_shape"]
        loaded[specification["name"]] = {}
        filenames = {
            "train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
            "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
        }
        for split, (image_name, label_name) in filenames.items():
            images = _decode_idx_images(raw_directory / image_name, shape)
            labels = _decode_idx_labels(raw_directory / label_name)
            _require(len(images) == len(labels), f"Official {specification['name']} {split} count mismatch.")
            loaded[specification["name"]][split] = OfficialSplit(images, labels)
    return loaded


def _stable_sample_id(dataset: str, split: str, source_index: int) -> str:
    return f"{dataset}:{split}:{int(source_index):08d}"


def deterministic_class_order(
    *,
    study_id: str,
    dataset: str,
    split: str,
    data_seed: int,
    source_class: int,
    candidate_indices: Iterable[int],
) -> list[int]:
    """Version-independent seeded ordering defined entirely by SHA-256."""
    keyed: list[tuple[str, int]] = []
    for source_index in candidate_indices:
        index = int(source_index)
        key = "\0".join(
            [study_id, dataset, split, str(int(data_seed)), str(int(source_class)), str(index)]
        ).encode("utf-8")
        keyed.append((hashlib.sha256(key).hexdigest(), index))
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [index for _, index in keyed]


def _class_entry(
    dataset: str,
    split: str,
    source_class: int,
    binary_label: int,
    source_indices: Sequence[int],
) -> dict[str, Any]:
    indices = [int(index) for index in source_indices]
    return {
        "binary_label": int(binary_label),
        "source_class": int(source_class),
        "source_indices": indices,
        "stable_ids": [_stable_sample_id(dataset, split, index) for index in indices],
    }


def _selected_tensor_binding(
    split: OfficialSplit, source_indices: Sequence[int]
) -> dict[str, Any]:
    selected = torch.tensor([int(index) for index in source_indices], dtype=torch.long)
    raw_bytes = split.data[selected].contiguous().numpy().tobytes(order="C")
    label_bytes = np.asarray(
        split.targets[selected].tolist(), dtype=">i8"
    ).tobytes(order="C")
    return {
        "selected_raw_uint8_size": len(raw_bytes),
        "selected_raw_uint8_sha256": _sha256_bytes(raw_bytes),
        "selected_source_labels_size": len(label_bytes),
        "selected_source_labels_sha256": _sha256_bytes(label_bytes),
    }


def build_selection_manifest(
    contract: Mapping[str, Any],
    datasets: Mapping[str, Mapping[str, Any]],
    *,
    execution_contract_sha256: str,
) -> dict[str, Any]:
    selections: list[dict[str, Any]] = []
    for specification in contract["datasets"]:
        name = str(specification["name"])
        targets_by_split = {
            split: torch.as_tensor(datasets[name][split].targets, dtype=torch.long)
            for split in ("train", "test")
        }
        for data_seed in contract["seed_design"]["data_seeds"]:
            ordered: dict[tuple[str, int], list[int]] = {}
            for split in ("train", "test"):
                targets = targets_by_split[split]
                for source_class in (specification["negative_class"], specification["positive_class"]):
                    candidates = torch.nonzero(
                        targets == int(source_class), as_tuple=False
                    ).flatten().tolist()
                    ordered[(split, int(source_class))] = deterministic_class_order(
                        study_id=str(contract["study_id"]),
                        dataset=name,
                        split=split,
                        data_seed=int(data_seed),
                        source_class=int(source_class),
                        candidate_indices=candidates,
                    )
            role_entries: list[dict[str, Any]] = []
            for role in contract["selection"]["role_order"]:
                split = str(specification[f"{role}_source_split"])
                count = int(specification[f"{role}_per_class"])
                class_entries: list[dict[str, Any]] = []
                for binary_label, source_class in enumerate(
                    (specification["negative_class"], specification["positive_class"])
                ):
                    class_order = ordered[(split, int(source_class))]
                    if role == "evaluation":
                        start = int(specification["probe_per_class"])
                    else:
                        start = 0
                    chosen = class_order[start : start + count]
                    _require(len(chosen) == count, f"Too few {name} {split} class {source_class} samples.")
                    class_entries.append(
                        _class_entry(name, split, int(source_class), binary_label, chosen)
                    )
                ordered_indices = [
                    index for class_value in class_entries for index in class_value["source_indices"]
                ]
                role_entries.append(
                    {
                        "role": role,
                        "official_split": split,
                        "classes": class_entries,
                        **_selected_tensor_binding(datasets[name][split], ordered_indices),
                    }
                )
            selections.append(
                {"dataset": name, "data_seed": int(data_seed), "roles": role_entries}
            )

    overlap_disclosure = _cross_seed_overlap_disclosure(contract, selections)
    manifest = {
        "schema_version": contract["selection"]["manifest_schema"],
        "study_id": contract["study_id"],
        "execution_contract_sha256": execution_contract_sha256,
        "algorithm": contract["selection"]["algorithm"],
        "cross_data_seed_overlap": contract["selection"]["cross_data_seed_overlap"],
        "selection_count": len(selections),
        "selections": selections,
        "cross_seed_overlap_disclosure": overlap_disclosure,
    }
    validate_selection_manifest(contract, manifest)
    return manifest


def _selection_role_ids(selection: Mapping[str, Any], role: str) -> set[str]:
    role_value = next(item for item in selection["roles"] if item["role"] == role)
    return {
        stable_id
        for class_value in role_value["classes"]
        for stable_id in class_value["stable_ids"]
    }


def _cross_seed_overlap_disclosure(
    contract: Mapping[str, Any], selections: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    disclosures: list[dict[str, Any]] = []
    seeds = list(contract["seed_design"]["data_seeds"])
    roles = list(contract["selection"]["role_order"])
    for dataset in contract["datasets"]:
        mapping = {
            int(selection["data_seed"]): selection
            for selection in selections
            if selection["dataset"] == dataset["name"]
        }
        for first_index, first_seed in enumerate(seeds):
            for second_seed in seeds[first_index + 1 :]:
                role_pairs = [
                    {
                        "role_a": role_a,
                        "role_b": role_b,
                        "overlap_count": len(
                            _selection_role_ids(mapping[first_seed], role_a)
                            & _selection_role_ids(mapping[second_seed], role_b)
                        ),
                    }
                    for role_a in roles
                    for role_b in roles
                ]
                full_a = set().union(
                    *(_selection_role_ids(mapping[first_seed], role) for role in roles)
                )
                full_b = set().union(
                    *(_selection_role_ids(mapping[second_seed], role) for role in roles)
                )
                disclosures.append(
                    {
                        "dataset": dataset["name"],
                        "data_seed_a": int(first_seed),
                        "data_seed_b": int(second_seed),
                        "full_selection_overlap_count": len(full_a & full_b),
                        "role_pair_overlap_counts": role_pairs,
                    }
                )
    return disclosures


def validate_selection_manifest(
    contract: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    expected_top = {
        "schema_version",
        "study_id",
        "execution_contract_sha256",
        "algorithm",
        "cross_data_seed_overlap",
        "selection_count",
        "selections",
        "cross_seed_overlap_disclosure",
    }
    _require(set(manifest) == expected_top, "Selection manifest schema is not exact.")
    _require(manifest["schema_version"] == "semi-real-selection-manifest-v1", "Selection schema changed.")
    _require(manifest["study_id"] == contract["study_id"], "Selection study ID changed.")
    contract_digest = manifest["execution_contract_sha256"]
    _require(
        isinstance(contract_digest, str)
        and len(contract_digest) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in contract_digest),
        "Selection execution-contract binding is not one SHA-256 digest.",
    )
    _require(manifest["algorithm"] == contract["selection"]["algorithm"], "Selection algorithm changed.")
    _require(
        manifest["cross_data_seed_overlap"] == contract["selection"]["cross_data_seed_overlap"],
        "Cross-seed overlap disclosure changed.",
    )
    selections = manifest["selections"]
    _require(isinstance(selections, list) and len(selections) == manifest["selection_count"] == 8, "Exactly 2x4 selections are required.")
    expected_keys = {
        (dataset["name"], int(seed))
        for dataset in contract["datasets"]
        for seed in contract["seed_design"]["data_seeds"]
    }
    seen: set[tuple[str, int]] = set()
    for selection in selections:
        _require(set(selection) == {"dataset", "data_seed", "roles"}, "Selection entry schema changed.")
        key = (str(selection["dataset"]), int(selection["data_seed"]))
        _require(key in expected_keys and key not in seen, f"Unexpected/duplicate selection: {key}")
        seen.add(key)
        specification = next(item for item in contract["datasets"] if item["name"] == key[0])
        roles = selection["roles"]
        _require(
            [item.get("role") for item in roles] == contract["selection"]["role_order"],
            f"Role ordering changed for {key}.",
        )
        role_id_sets: dict[str, set[str]] = {}
        for role_value in roles:
            _require(
                set(role_value)
                == {
                    "role",
                    "official_split",
                    "classes",
                    "selected_raw_uint8_size",
                    "selected_raw_uint8_sha256",
                    "selected_source_labels_size",
                    "selected_source_labels_sha256",
                },
                "Role schema changed.",
            )
            role = role_value["role"]
            split = specification[f"{role}_source_split"]
            _require(role_value["official_split"] == split, f"Official split changed for {key}/{role}.")
            classes = role_value["classes"]
            _require(isinstance(classes, list) and len(classes) == 2, "Each role requires two classes.")
            expected_count = int(specification[f"{role}_per_class"])
            ids: list[str] = []
            for binary_label, class_value in enumerate(classes):
                _require(
                    set(class_value) == {"binary_label", "source_class", "source_indices", "stable_ids"},
                    "Selection class schema changed.",
                )
                source_class = (
                    specification["negative_class"] if binary_label == 0 else specification["positive_class"]
                )
                _require(
                    class_value["binary_label"] == binary_label
                    and class_value["source_class"] == source_class,
                    f"Class mapping changed for {key}/{role}.",
                )
                indices = class_value["source_indices"]
                stable_ids = class_value["stable_ids"]
                _require(
                    isinstance(indices, list)
                    and isinstance(stable_ids, list)
                    and len(indices) == len(stable_ids) == expected_count,
                    f"Cardinality changed for {key}/{role}/class{binary_label}.",
                )
                _require(
                    all(isinstance(index, int) and index >= 0 for index in indices)
                    and len(indices) == len(set(indices)),
                    "Selected source indices must be unique non-negative integers.",
                )
                _require(
                    stable_ids == [_stable_sample_id(key[0], split, index) for index in indices],
                    "Split-qualified stable IDs do not match selected indices.",
                )
                ids.extend(stable_ids)
            _require(len(ids) == len(set(ids)), f"Duplicate stable IDs in {key}/{role}.")
            expected_examples = 2 * expected_count
            expected_raw_size = expected_examples * int(specification["raw_image_shape"][0]) * int(
                specification["raw_image_shape"][1]
            )
            for digest_field in (
                "selected_raw_uint8_sha256",
                "selected_source_labels_sha256",
            ):
                digest = role_value[digest_field]
                _require(
                    isinstance(digest, str)
                    and len(digest) == SHA256_LENGTH
                    and all(character in "0123456789abcdef" for character in digest),
                    f"Invalid selected tensor digest for {key}/{role}.",
                )
            _require(
                role_value["selected_raw_uint8_size"] == expected_raw_size
                and role_value["selected_source_labels_size"] == expected_examples * 8,
                f"Selected tensor size changed for {key}/{role}.",
            )
            role_id_sets[role] = set(ids)
        _require(
            not (role_id_sets["train"] & role_id_sets["probe"])
            and not (role_id_sets["train"] & role_id_sets["evaluation"])
            and not (role_id_sets["probe"] & role_id_sets["evaluation"]),
            f"Cross-role overlap detected in {key}.",
        )
    _require(seen == expected_keys, "Selection factorial is incomplete.")

    recomputed_overlap = _cross_seed_overlap_disclosure(contract, selections)
    _require(
        manifest["cross_seed_overlap_disclosure"] == recomputed_overlap,
        "Cross-data-seed role-pair overlap disclosure is incomplete or incorrect.",
    )
    return {"valid": True, "selection_count": len(selections)}


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_workspace_is_git_ignored(
    repository_root: Path,
    *,
    candidate: Path | None = None,
) -> None:
    repository = _lexical_absolute(repository_root)
    workspace_relative = SEMI_REAL_WORKSPACE_RELATIVE_ROOT
    checked_relative = (
        workspace_relative / ".git-ignore-policy-probe"
        if candidate is None
        else _lexical_absolute(candidate).relative_to(repository)
    )
    try:
        tracked = subprocess.run(
            [
                _git_executable(),
                "ls-files",
                "--",
                workspace_relative.as_posix(),
            ],
            cwd=repository,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        _require(
            tracked.returncode == 0,
            "Cannot inspect tracked paths in the semi-real local workspace.",
        )
        _require(
            not tracked.stdout.strip(),
            "data/semi_real_generated_cue_v1/ must contain no tracked paths so local "
            "artifacts do not affect git cleanliness.",
        )
        ignored = subprocess.run(
            [
                _git_executable(),
                "check-ignore",
                "--quiet",
                "--",
                checked_relative.as_posix(),
            ],
            cwd=repository,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as error:
        raise SemiRealStudyError(f"Cannot verify the semi-real Git-ignored workspace: {error}") from error
    _require(
        ignored.returncode == 0,
        f"{checked_relative.as_posix()} must remain Git-ignored so local artifacts "
        "do not affect git cleanliness.",
    )


def _require_workspace_subpath(
    path: Path,
    repository_root: Path,
    *,
    path_role: str,
) -> Path:
    """Require one non-symlink path strictly below the ignored local workspace."""
    supplied = Path(path)
    _require(
        ".." not in supplied.parts,
        f"{path_role} cannot contain parent-directory traversal: {supplied}",
    )
    repository = _lexical_absolute(repository_root)
    workspace = repository / SEMI_REAL_WORKSPACE_RELATIVE_ROOT
    candidate = _lexical_absolute(supplied)
    _require(
        candidate != workspace and workspace in candidate.parents,
        f"{path_role} must be a strict descendant of the Git-ignored workspace "
        f"{workspace}; ignored local artifacts do not affect git cleanliness.",
    )
    cursor = repository
    for component in candidate.relative_to(repository).parts:
        cursor /= component
        _require(
            not cursor.is_symlink(),
            f"Symlinked {path_role.lower()} path component is forbidden: {cursor}",
        )
    _require_workspace_is_git_ignored(repository, candidate=candidate)
    return candidate


def _require_existing_tree_without_symlinks(path: Path, *, path_role: str) -> None:
    if not path.exists():
        return
    _require(
        path.is_dir() and not path.is_symlink(),
        f"Existing {path_role.lower()} must be a non-symlink directory: {path}",
    )
    for descendant in path.rglob("*"):
        _require(
            not descendant.is_symlink(),
            f"Symlinked {path_role.lower()} artifact is forbidden: {descendant}",
        )


def _manifest_reference(path: Path, *, name: str | None = None) -> dict[str, Any]:
    return _file_descriptor(path, relative_path=name or path.name)


def _write_manifest_sidecar(manifest_path: Path) -> None:
    digest = sha256_file(manifest_path)
    _atomic_write_bytes(
        manifest_path.with_suffix(".sha256"),
        f"{digest}  {manifest_path.name}\n".encode("ascii"),
    )


def _verify_manifest_sidecar(manifest_path: Path) -> None:
    _require(
        manifest_path.is_file() and not manifest_path.is_symlink(),
        f"Manifest is missing or symlinked: {manifest_path}",
    )
    sidecar = manifest_path.with_suffix(".sha256")
    _require(sidecar.is_file() and not sidecar.is_symlink(), f"Manifest sidecar is missing: {sidecar}")
    expected = f"{sha256_file(manifest_path)}  {manifest_path.name}\n".encode("ascii")
    _require(sidecar.read_bytes() == expected, f"Manifest sidecar hash drift: {manifest_path}")


def build_preflight(
    contract_path: Path,
    *,
    data_root: Path,
    output_directory: Path,
    allow_download: bool = False,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Freeze all non-outcome inputs; never construct a model or optimizer."""
    contract = load_execution_contract(contract_path, repository_root=repository_root)
    data_root = _require_workspace_subpath(
        data_root,
        repository_root,
        path_role="Data root",
    )
    output_directory = _require_workspace_subpath(
        output_directory,
        repository_root,
        path_role="Preflight output",
    )
    _require_existing_tree_without_symlinks(data_root, path_role="Data root")
    _require(not output_directory.exists(), f"Preflight output already exists: {output_directory}")
    environment = contract["execution_environment"]
    lock_path = repository_root / environment["environment_lock_path"]
    _current_environment(contract, repository_root)
    if allow_download:
        datasets = _load_official_datasets(contract, data_root, download=True)
        raw_manifest = build_raw_data_manifest(contract, data_root)
    else:
        raw_manifest = build_raw_data_manifest(contract, data_root)
        datasets = _load_official_datasets(contract, data_root, download=False)
    source_manifest = build_source_manifest(contract, repository_root)
    contract_sha256 = sha256_file(contract_path)
    selection_manifest = build_selection_manifest(
        contract, datasets, execution_contract_sha256=contract_sha256
    )
    current_git = git_state(repository_root)
    expected_ids = expected_record_ids(contract)
    blockers: list[str] = []
    if not current_git["clean"]:
        blockers.append("source_worktree_is_not_clean")
    preflight = {
        "schema_version": contract["preflight"]["schema_version"],
        "study_id": contract["study_id"],
        "publication_scope": "semi-real",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "download_requested": bool(allow_download),
        "model_constructed": False,
        "optimizer_constructed": False,
        "model_outcomes_inspected": False,
        "contract": {
            **_manifest_reference(contract_path, name="execution-contract.json"),
            "semantic_sha256": CONTRACT_SEMANTIC_SHA256,
        },
        "historical_contract": _manifest_reference(
            repository_root / contract["historical_contract"]["path"],
            name="historical-contract.json",
        ),
        "environment_lock": _manifest_reference(lock_path, name="environment-lock.txt"),
        "environment": _current_environment(contract, repository_root),
        "source_manifest": {
            "path": "source_manifest.json",
            "size": len(canonical_json_bytes(source_manifest)),
            "sha256": _sha256_bytes(canonical_json_bytes(source_manifest)),
            "aggregate_sha256": source_manifest["aggregate_sha256"],
            "file_count": len(source_manifest["files"]),
        },
        "raw_data_manifest": {
            "path": "raw_data_manifest.json",
            "size": len(canonical_json_bytes(raw_manifest)),
            "sha256": _sha256_bytes(canonical_json_bytes(raw_manifest)),
            "dataset_aggregate_sha256": {
                item["name"]: item["aggregate_sha256"] for item in raw_manifest["datasets"]
            },
        },
        "selection_manifest": {
            "path": "selection_manifest.json",
            "size": len(canonical_json_bytes(selection_manifest)),
            "sha256": _sha256_bytes(canonical_json_bytes(selection_manifest)),
            "selection_count": 8,
        },
        "git": current_git,
        "expected_record_ids": expected_ids,
        "ready_for_authorization": not blockers,
        "blockers": blockers,
    }

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.staging-", dir=output_directory.parent)
    )
    try:
        _atomic_write_bytes(staging / "execution-contract.json", contract_path.read_bytes())
        historical_path = repository_root / contract["historical_contract"]["path"]
        _atomic_write_bytes(staging / "historical-contract.json", historical_path.read_bytes())
        _atomic_write_bytes(staging / "environment-lock.txt", lock_path.read_bytes())
        _atomic_write_bytes(staging / "source_manifest.json", canonical_json_bytes(source_manifest))
        _atomic_write_bytes(staging / "raw_data_manifest.json", canonical_json_bytes(raw_manifest))
        _atomic_write_bytes(staging / "selection_manifest.json", canonical_json_bytes(selection_manifest))
        manifest_path = staging / "preflight.manifest.json"
        _atomic_write_bytes(manifest_path, canonical_json_bytes(preflight))
        _write_manifest_sidecar(manifest_path)
        os.replace(staging, output_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_directory / "preflight.manifest.json"


@dataclass
class VerifiedPreflight:
    contract: dict[str, Any]
    manifest: dict[str, Any]
    source_manifest: dict[str, Any]
    raw_manifest: dict[str, Any]
    selection_manifest: dict[str, Any]
    datasets: dict[str, dict[str, Any]]
    contract_path: Path
    preflight_path: Path
    data_root: Path


def _verify_reference(directory: Path, reference: Mapping[str, Any]) -> Path:
    path = directory / str(reference["path"])
    _require(path.is_file() and not path.is_symlink(), f"Referenced preflight file is missing: {path}")
    _require(path.stat().st_size == reference["size"], f"Referenced preflight file size drift: {path}")
    _require(sha256_file(path) == reference["sha256"], f"Referenced preflight file hash drift: {path}")
    return path


def verify_preflight(
    contract_path: Path,
    preflight_path: Path,
    *,
    data_root: Path,
    repository_root: Path = REPOSITORY_ROOT,
    require_ready: bool = True,
) -> VerifiedPreflight:
    contract = load_execution_contract(contract_path, repository_root=repository_root)
    data_root = _require_workspace_subpath(
        data_root,
        repository_root,
        path_role="Data root",
    )
    preflight_path = _require_workspace_subpath(
        preflight_path,
        repository_root,
        path_role="Preflight manifest",
    )
    _require_existing_tree_without_symlinks(data_root, path_role="Data root")
    _require_existing_tree_without_symlinks(preflight_path.parent, path_role="Preflight")
    _verify_manifest_sidecar(preflight_path)
    preflight = load_strict_json(preflight_path)
    _require(set(preflight) == PREFLIGHT_MANIFEST_KEYS, "Preflight manifest schema is not exact.")
    _require(preflight["schema_version"] == "semi-real-preflight-v1", "Preflight schema changed.")
    _require(preflight["study_id"] == contract["study_id"], "Preflight study ID changed.")
    _require(preflight["publication_scope"] == "semi-real", "Preflight scope changed.")
    _require(
        preflight["model_constructed"] is False
        and preflight["optimizer_constructed"] is False
        and preflight["model_outcomes_inspected"] is False,
        "Preflight must be outcome-free.",
    )
    _require(preflight["expected_record_ids"] == expected_record_ids(contract), "Preflight record IDs changed.")
    _require(preflight["contract"]["sha256"] == sha256_file(contract_path), "Preflight contract binding changed.")
    _require(
        preflight["contract"]["semantic_sha256"] == CONTRACT_SEMANTIC_SHA256,
        "Preflight semantic contract binding changed.",
    )
    directory = preflight_path.parent
    copied_contract = _verify_reference(directory, preflight["contract"])
    _require(copied_contract.read_bytes() == contract_path.read_bytes(), "Preflight contract copy drifted.")
    copied_historical = _verify_reference(directory, preflight["historical_contract"])
    historical_path = repository_root / contract["historical_contract"]["path"]
    _require(copied_historical.read_bytes() == historical_path.read_bytes(), "Historical copy drifted.")
    copied_lock = _verify_reference(directory, preflight["environment_lock"])
    lock_path = repository_root / contract["execution_environment"]["environment_lock_path"]
    _require(copied_lock.read_bytes() == lock_path.read_bytes(), "Environment lock copy drifted.")
    _require(preflight["environment"] == _current_environment(contract, repository_root), "Current environment drifted.")

    source_path = _verify_reference(directory, preflight["source_manifest"])
    raw_path = _verify_reference(directory, preflight["raw_data_manifest"])
    selection_path = _verify_reference(directory, preflight["selection_manifest"])
    source_manifest = load_strict_json(source_path)
    raw_manifest = load_strict_json(raw_path)
    selection_manifest = load_strict_json(selection_path)
    current_source = build_source_manifest(contract, repository_root)
    _require(
        canonical_json_bytes(source_manifest) == canonical_json_bytes(current_source),
        "Executable source hash drifted after preflight.",
    )
    current_raw = build_raw_data_manifest(contract, data_root)
    _require(
        canonical_json_bytes(raw_manifest) == canonical_json_bytes(current_raw),
        "Raw dataset file hash drifted after preflight.",
    )
    datasets = _load_official_datasets(contract, data_root, download=False)
    current_selection = build_selection_manifest(
        contract,
        datasets,
        execution_contract_sha256=sha256_file(contract_path),
    )
    _require(
        canonical_json_bytes(selection_manifest) == canonical_json_bytes(current_selection),
        "Selected-index manifest drifted or is not reproducible from official labels.",
    )
    _require(
        preflight["source_manifest"]["aggregate_sha256"] == source_manifest["aggregate_sha256"],
        "Source aggregate binding changed.",
    )
    raw_aggregates = {item["name"]: item["aggregate_sha256"] for item in raw_manifest["datasets"]}
    _require(
        preflight["raw_data_manifest"]["dataset_aggregate_sha256"] == raw_aggregates,
        "Raw dataset aggregate bindings changed.",
    )
    current_git = git_state(repository_root)
    _require(preflight["git"] == current_git, "Git state changed after preflight.")
    if require_ready:
        _require(preflight["ready_for_authorization"] is True, "Preflight is not ready for authorization.")
        _require(preflight["blockers"] == [], "Ready preflight cannot contain blockers.")
        _require(current_git["clean"] is True, "Scientific execution requires a clean worktree.")
    return VerifiedPreflight(
        contract=contract,
        manifest=preflight,
        source_manifest=source_manifest,
        raw_manifest=raw_manifest,
        selection_manifest=selection_manifest,
        datasets=datasets,
        contract_path=contract_path,
        preflight_path=preflight_path,
        data_root=data_root,
    )


@dataclass
class AuthorizedExecution:
    preflight: VerifiedPreflight
    seal: dict[str, Any]
    seal_path: Path


def _validate_timestamp(value: Any) -> None:
    _require(isinstance(value, str), "Authorization timestamp must be an ISO-8601 string.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SemiRealStudyError("Authorization timestamp is not valid ISO-8601.") from error
    _require(parsed.tzinfo is not None, "Authorization timestamp must include a timezone.")


def _validate_seal_document(
    contract: Mapping[str, Any],
    seal: Mapping[str, Any],
    expected_bindings: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected_top = {
        "schema_version",
        "study_id",
        "authorization_kind",
        "bindings",
        "expected_git_state",
        "untouched_seed_attestation",
        "authorization",
    }
    _require(set(seal) == expected_top, "Authorization seal schema is not exact.")
    _require(
        seal["schema_version"] == contract["authorization"]["seal_schema_version"]
        and seal["study_id"] == contract["study_id"]
        and seal["authorization_kind"] == "explicit_human_scientific_execution",
        "Authorization seal identity or kind changed.",
    )
    _require(
        set(seal["bindings"]) == set(contract["authorization"]["required_bindings"])
        and seal["bindings"] == expected_bindings,
        "Authorization seal does not bind the exact execution inputs.",
    )
    expected_git = seal["expected_git_state"]
    _require(
        set(expected_git) == {"clean", "commit"}
        and expected_git["clean"] is True
        and isinstance(expected_git["commit"], str)
        and len(expected_git["commit"]) == 40
        and all(character in "0123456789abcdef" for character in expected_git["commit"]),
        "Authorization seal must require one clean lowercase 40-character commit.",
    )
    _require(
        seal["untouched_seed_attestation"]
        == {
            "attested": True,
            "text": contract["authorization"]["untouched_seed_attestation_text"],
        },
        "Untouched-seed attestation is absent or changed.",
    )
    authorization = seal["authorization"]
    _require(
        set(authorization) == {"authorized", "text", "timestamp_utc"}
        and authorization["authorized"] is True
        and authorization["text"] == contract["authorization"]["explicit_authorization_text"],
        "Explicit user authorization is absent or changed.",
    )
    _validate_timestamp(authorization["timestamp_utc"])
    return expected_git


def verify_authorization(
    contract_path: Path,
    preflight_path: Path,
    seal_path: Path,
    *,
    data_root: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> AuthorizedExecution:
    """Verify the external seal before any model, optimizer, or output is created."""
    preflight = verify_preflight(
        contract_path,
        preflight_path,
        data_root=data_root,
        repository_root=repository_root,
        require_ready=True,
    )
    seal_path = _require_workspace_subpath(
        seal_path,
        repository_root,
        path_role="Authorization seal",
    )
    seal = load_strict_json(seal_path)
    contract = preflight.contract
    raw_aggregates = {
        item["name"]: item["aggregate_sha256"] for item in preflight.raw_manifest["datasets"]
    }
    expected_bindings = {
        "execution_contract_sha256": sha256_file(contract_path),
        "preflight_manifest_sha256": sha256_file(preflight_path),
        "historical_contract_sha256": contract["historical_contract"]["sha256"],
        "source_manifest_sha256": preflight.manifest["source_manifest"]["sha256"],
        "source_aggregate_sha256": preflight.source_manifest["aggregate_sha256"],
        "environment_lock_sha256": contract["execution_environment"]["environment_lock_sha256"],
        "raw_data_manifest_sha256": preflight.manifest["raw_data_manifest"]["sha256"],
        "raw_dataset_aggregate_sha256": raw_aggregates,
        "selection_manifest_sha256": preflight.manifest["selection_manifest"]["sha256"],
        "expected_record_ids": expected_record_ids(contract),
    }
    expected_git = _validate_seal_document(contract, seal, expected_bindings)
    current_git = git_state(repository_root)
    _require(
        current_git["clean"] is True and current_git["commit"] == expected_git["commit"],
        "Current git state is not the sealed clean commit.",
    )
    _require(
        preflight.manifest["git"]["clean"] is True
        and preflight.manifest["git"]["commit"] == expected_git["commit"],
        "Preflight was not produced from the sealed clean commit.",
    )
    return AuthorizedExecution(preflight=preflight, seal=seal, seal_path=seal_path)


def _configure_determinism(contract: Mapping[str, Any]) -> None:
    environment = contract["execution_environment"]
    _require(environment["device"] == "cpu" and environment["dtype"] == "float32", "Only CPU float32 is authorized.")
    torch.set_num_threads(int(environment["intraop_threads"]))
    desired_interop = int(environment["interop_threads"])
    if torch.get_num_interop_threads() != desired_interop:
        try:
            torch.set_num_interop_threads(desired_interop)
        except RuntimeError as error:
            if torch.get_num_interop_threads() != desired_interop:
                raise SemiRealStudyError("Cannot enforce the frozen inter-op thread count.") from error
    torch.use_deterministic_algorithms(True, warn_only=False)
    _require(torch.are_deterministic_algorithms_enabled(), "Deterministic algorithms are not enabled.")


def _find_dataset_specification(contract: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return next(item for item in contract["datasets"] if item["name"] == name)


def _find_selection(manifest: Mapping[str, Any], dataset: str, data_seed: int) -> Mapping[str, Any]:
    matches = [
        item
        for item in manifest["selections"]
        if item["dataset"] == dataset and item["data_seed"] == int(data_seed)
    ]
    _require(len(matches) == 1, f"Missing/duplicate selection for {dataset}/{data_seed}.")
    return matches[0]


def _role_tensors(
    verified: VerifiedPreflight,
    dataset_specification: Mapping[str, Any],
    selection: Mapping[str, Any],
    role: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    role_value = next(item for item in selection["roles"] if item["role"] == role)
    split = str(role_value["official_split"])
    dataset = verified.datasets[dataset_specification["name"]][split]
    raw = torch.as_tensor(dataset.data)
    targets = torch.as_tensor(dataset.targets, dtype=torch.long)
    _require(raw.dtype == torch.uint8, "Official pixels changed dtype after preflight.")
    indices = [index for item in role_value["classes"] for index in item["source_indices"]]
    expected_labels = [
        item["binary_label"] for item in role_value["classes"] for _ in item["source_indices"]
    ]
    selected = torch.tensor(indices, dtype=torch.long)
    tensor_binding = _selected_tensor_binding(dataset, indices)
    _require(
        all(role_value[field] == value for field, value in tensor_binding.items()),
        f"Selected raw tensor/label binding drifted for {dataset_specification['name']}/{role}.",
    )
    selected_targets = targets[selected]
    expected_source_classes = torch.tensor(
        [item["source_class"] for item in role_value["classes"] for _ in item["source_indices"]],
        dtype=torch.long,
    )
    _require(torch.equal(selected_targets, expected_source_classes), "Selected source labels drifted.")
    labels = torch.tensor(expected_labels, dtype=torch.long)
    core = prepare_core_images(raw[selected], int(verified.contract["preprocessing"]["resize"]["height"]))
    return core.to(dtype=torch.float32, device="cpu"), labels, selected


def _audit_intervention(both: SemiRealBatch, weak: SemiRealBatch) -> dict[str, bool]:
    patch = both.spec.cue_patch_size
    expected = both.signed_labels[:, None, None] * both.spec.cue_strength
    both_channel = both.x[:, 1]
    exact_patch = torch.equal(both_channel[:, :patch, :patch], expected.expand(-1, patch, patch))
    outside = both_channel.clone()
    outside[:, :patch, :patch] = 0
    exact_generated_channel = bool(exact_patch and torch.count_nonzero(outside).item() == 0)
    checks = {
        "labels_equal": torch.equal(both.y, weak.y),
        "sample_ids_equal": torch.equal(both.sample_ids, weak.sample_ids),
        "core_channel_bitwise_equal": torch.equal(both.x[:, 0], weak.x[:, 0]),
        "weak_generated_channel_all_zero": torch.count_nonzero(weak.x[:, 1]).item() == 0,
        "only_generated_channel_differs": bool(
            torch.equal(both.x[:, 0], weak.x[:, 0]) and exact_generated_channel
        ),
    }
    _require(all(checks.values()), f"Intervention preservation audit failed: {checks}")
    return checks


def _record_data(
    verified: VerifiedPreflight, dataset_name: str, data_seed: int
) -> tuple[SemiRealBatch, SemiRealBatch, SemiRealProbe, SemiRealProbe, dict[str, bool], dict[str, int]]:
    contract = verified.contract
    specification = _find_dataset_specification(contract, dataset_name)
    selection = _find_selection(verified.selection_manifest, dataset_name, data_seed)
    spec = SemiRealTaskSpec(
        dataset_name=dataset_name,
        negative_class=int(specification["negative_class"]),
        positive_class=int(specification["positive_class"]),
        cue_strength=float(contract["intervention"]["cue_strength"]),
        image_size=int(contract["preprocessing"]["resize"]["height"]),
        cue_patch_size=int(contract["intervention"]["cue_patch"]["height"]),
    )
    train_core, train_labels, train_ids = _role_tensors(
        verified, specification, selection, "train"
    )
    probe_core, probe_labels, probe_ids = _role_tensors(
        verified, specification, selection, "probe"
    )
    evaluation_core, evaluation_labels, evaluation_ids = _role_tensors(
        verified, specification, selection, "evaluation"
    )
    stable_sets = {
        role: _selection_role_ids(selection, role)
        for role in contract["selection"]["role_order"]
    }
    _require(
        not (stable_sets["train"] & stable_sets["probe"])
        and not (stable_sets["train"] & stable_sets["evaluation"])
        and not (stable_sets["probe"] & stable_sets["evaluation"]),
        "Split-qualified role IDs overlap.",
    )
    both, weak = make_paired_semi_real_task(train_core, train_labels, train_ids, spec)
    checks = _audit_intervention(both, weak)
    probe = make_semi_real_probe(probe_core, probe_labels, probe_ids, spec)
    evaluation = make_semi_real_probe(
        evaluation_core, evaluation_labels, evaluation_ids, spec
    )
    counts = {
        "train": len(train_labels),
        "probe": len(probe_labels),
        "evaluation": len(evaluation_labels),
    }
    return both, weak, probe, evaluation, checks, counts


def _finite_scalar(value: torch.Tensor, name: str) -> float:
    result = float(value.detach().cpu().item())
    _require(math.isfinite(result), f"Non-finite training diagnostic: {name}")
    return result


def _train_pair(
    *,
    both_batch: SemiRealBatch,
    weak_batch: SemiRealBatch,
    probe: SemiRealProbe,
    model_seed: int,
    channels: tuple[int, int],
    learning_rate: float,
    steps: int,
    log_every: int,
    record_metadata: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], SemiRealCNN, SemiRealCNN, dict[str, float]]:
    """Train copied models with independent full-batch SGD in serial lockstep."""
    torch.manual_seed(int(model_seed))
    both_model = SemiRealCNN(channels=channels).to(device="cpu", dtype=torch.float32)
    weak_model = copy.deepcopy(both_model)
    for both_parameter, weak_parameter in zip(both_model.parameters(), weak_model.parameters()):
        _require(torch.equal(both_parameter, weak_parameter), "Copied initialization is not bitwise equal.")
    optimizers = {
        "both": torch.optim.SGD(
            both_model.parameters(), lr=learning_rate, momentum=0.0, weight_decay=0.0
        ),
        "weak_only": torch.optim.SGD(
            weak_model.parameters(), lr=learning_rate, momentum=0.0, weight_decay=0.0
        ),
    }
    arms = {
        "both": (both_model, both_batch),
        "weak_only": (weak_model, weak_batch),
    }
    initial: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    for step in range(steps + 1):
        if step % log_every == 0:
            for condition in ("both", "weak_only"):
                model, batch = arms[condition]
                _require(
                    all(parameter.grad is None for parameter in model.parameters()),
                    "Diagnostic logging encountered residual optimizer gradients.",
                )
                statistics = semi_real_statistics(model, batch, probe)
                _require(
                    all(parameter.grad is None for parameter in model.parameters()),
                    "Direct-autograd diagnostics changed optimizer gradients.",
                )
                core_response = _finite_scalar(statistics.core_response, "core_response")
                cue_response = _finite_scalar(statistics.cue_response, "cue_response")
                if step == 0:
                    initial[condition] = {"core": core_response, "cue": cue_response}
                rows.append(
                    {
                        **record_metadata,
                        "condition": condition,
                        "step": int(step),
                        "tau": float(step * learning_rate),
                        "loss": _finite_scalar(statistics.loss, "loss"),
                        "accuracy": _finite_scalar(statistics.accuracy, "accuracy"),
                        "core_response": core_response,
                        "core_gain": core_response - initial[condition]["core"],
                        "cue_response": cue_response,
                        "cue_gain": cue_response - initial[condition]["cue"],
                        "core_drift": _finite_scalar(statistics.core_drift, "core_drift"),
                        "cue_drift": _finite_scalar(statistics.cue_drift, "cue_drift"),
                    }
                )
        if step == steps:
            break
        for condition in ("both", "weak_only"):
            model, batch = arms[condition]
            optimizer = optimizers[condition]
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x)
            loss = F.binary_cross_entropy_with_logits(logits, batch.y.to(logits.dtype))
            _require(torch.isfinite(loss).item(), "Training loss became non-finite.")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    expected_points = steps // log_every + 1
    _require(len(rows) == 2 * expected_points, "Logged trajectory point count changed.")
    _require(
        initial["both"]["core"] == initial["weak_only"]["core"]
        and initial["both"]["cue"] == initial["weak_only"]["cue"],
        "Shared initialization did not produce exact common-probe response equality.",
    )
    initial_gaps = {
        "core_both_minus_weak": initial["both"]["core"] - initial["weak_only"]["core"],
        "cue_both_minus_weak": initial["both"]["cue"] - initial["weak_only"]["cue"],
    }
    return rows, both_model, weak_model, initial_gaps


def _piecewise_linear_signed_areas(times: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    positive = 0.0
    negative = 0.0
    for index in range(len(times) - 1):
        width = float(times[index + 1] - times[index])
        left = float(values[index])
        right = float(values[index + 1])
        if left >= 0 and right >= 0:
            positive += width * (left + right) / 2.0
        elif left <= 0 and right <= 0:
            negative += width * (-left - right) / 2.0
        else:
            left_width = width * abs(left) / (abs(left) + abs(right))
            right_width = width - left_width
            if left > 0:
                positive += left_width * left / 2.0
                negative += right_width * (-right) / 2.0
            else:
                negative += left_width * (-left) / 2.0
                positive += right_width * right / 2.0
    return float(positive), float(negative)


def summarize_paired_trajectory(
    rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    conditions = [row.get("condition") for row in rows]
    _require(
        set(conditions) == {"both", "weak_only"},
        "Trajectory must contain exactly the BOTH and WEAK_ONLY arms.",
    )
    by_arm = {
        condition: sorted(
            [row for row in rows if row["condition"] == condition], key=lambda row: row["step"]
        )
        for condition in ("both", "weak_only")
    }
    optimization = contract["optimization"]
    expected_steps = np.arange(
        int(optimization["logged_step_start"]),
        int(optimization["logged_step_end"]) + 1,
        int(optimization["log_every"]),
        dtype=int,
    )
    _require(
        len(expected_steps) == int(optimization["logged_point_count"])
        and len(by_arm["both"]) == len(by_arm["weak_only"]) == len(expected_steps)
        and len(rows) == 2 * len(expected_steps),
        "Paired trajectory point count is off contract.",
    )
    both_steps = np.asarray([row["step"] for row in by_arm["both"]], dtype=int)
    weak_steps = np.asarray([row["step"] for row in by_arm["weak_only"]], dtype=int)
    times = np.asarray([row["tau"] for row in by_arm["both"]], dtype=float)
    weak_times = np.asarray([row["tau"] for row in by_arm["weak_only"]], dtype=float)
    expected_times = expected_steps.astype(float) * float(optimization["learning_rate"])
    _require(
        np.array_equal(both_steps, expected_steps)
        and np.array_equal(weak_steps, expected_steps)
        and np.array_equal(times, expected_times)
        and np.array_equal(weak_times, expected_times),
        "Trajectory must use exactly steps 0,5,...,500 and tau=step*learning_rate.",
    )
    for condition, arm_rows in by_arm.items():
        initial_core = float(arm_rows[0]["core_response"])
        initial_cue = float(arm_rows[0]["cue_response"])
        _require(
            all(
                float(row["core_gain"]) == float(row["core_response"]) - initial_core
                and float(row["cue_gain"]) == float(row["cue_response"]) - initial_cue
                for row in arm_rows
            ),
            f"{condition} gains are not exact gains from its logged shared initialization.",
        )
    both_gain = np.asarray([row["core_gain"] for row in by_arm["both"]], dtype=float)
    weak_gain = np.asarray([row["core_gain"] for row in by_arm["weak_only"]], dtype=float)
    both_drift = np.asarray([row["core_drift"] for row in by_arm["both"]], dtype=float)
    weak_drift = np.asarray([row["core_drift"] for row in by_arm["weak_only"]], dtype=float)
    _require(
        np.isfinite(both_gain).all()
        and np.isfinite(weak_gain).all()
        and np.isfinite(both_drift).all()
        and np.isfinite(weak_drift).all(),
        "Trajectory contains non-finite scientific values.",
    )
    endpoint = contract["primary_endpoints"]
    beta = float(endpoint["weak_only_learnability"]["target_core_gain"])
    horizon = float(endpoint["weak_only_learnability"]["horizon_tau"])
    causal = causal_metrics(times, both_gain, weak_gain, beta)
    weak_learnable = bool(
        math.isfinite(causal.weak_hitting_time)
        and causal.weak_hitting_time > float(times[0])
        and causal.weak_hitting_time <= horizon
    )
    response_gap = both_gain - weak_gain
    drift_gap = both_drift - weak_drift
    tolerance = float(endpoint["outcome_suppression"]["tolerance"])
    tail = discrete_crossover_certificate(
        response_gap, weak_only_learnable=weak_learnable, tolerance=tolerance
    )
    outcome_suppression = bool(tail.strict_outcome_suppression)
    causal_certificate = bool(outcome_suppression and weak_learnable)
    _require(
        causal_certificate == tail.causal_starvation_certified,
        "Tail certificate and explicit causal gate disagree.",
    )
    positive_drift_area, negative_drift_area = _piecewise_linear_signed_areas(times, drift_gap)
    drift_crossing = float(sign_crossing_time(times, drift_gap))
    response_crossing = float(sign_crossing_time(times, response_gap))
    drift_sign_changes = int(n_sign_changes(drift_gap))
    response_sign_changes = int(n_sign_changes(response_gap))

    def hit_status(value: float) -> str:
        if math.isfinite(value):
            return "hit_at_initialization" if value <= times[0] else "finite_nondegenerate"
        return "not_hit_by_horizon"

    def crossing_status(value: float) -> str:
        if math.isfinite(value):
            return "finite"
        return "undecidable" if math.isnan(value) else "never"

    return {
        "weak_auc_gap": float(causal.weak_auc_gap),
        "both_first_hit_tau": float(causal.both_hitting_time),
        "both_first_hit_status": hit_status(causal.both_hitting_time),
        "weak_only_first_hit_tau": float(causal.weak_hitting_time),
        "weak_only_first_hit_status": hit_status(causal.weak_hitting_time),
        "weak_only_learnable": weak_learnable,
        "outcome_suppression": outcome_suppression,
        "causal_certificate": causal_certificate,
        "response_gap_min": float(response_gap.min()),
        "response_gap_final": float(response_gap[-1]),
        "tail_certificate": asdict(tail),
        "first_drift_crossing_tau": drift_crossing,
        "first_drift_crossing_status": crossing_status(drift_crossing),
        "first_response_crossing_tau": response_crossing,
        "first_response_crossing_status": crossing_status(response_crossing),
        "drift_sign_change_count": drift_sign_changes,
        "response_sign_change_count": response_sign_changes,
        "drift_multiple_sign_changes": bool(drift_sign_changes > 1),
        "response_multiple_sign_changes": bool(response_sign_changes > 1),
        "positive_drift_area": positive_drift_area,
        "negative_drift_area": negative_drift_area,
        "final_both_core_gain": float(both_gain[-1]),
        "final_weak_only_core_gain": float(weak_gain[-1]),
        "endpoint_signs": {
            "weak_auc_integrand": "weak_only_minus_both",
            "response_gap": "both_minus_weak_only",
            "drift_gap": "both_minus_weak_only",
        },
    }


def _process_peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _record_input_bindings_from_authorization(
    authorized: AuthorizedExecution,
) -> dict[str, Any]:
    verified = authorized.preflight
    return {
        "execution_contract_sha256": sha256_file(verified.contract_path),
        "preflight_manifest_sha256": sha256_file(verified.preflight_path),
        "authorization_seal_sha256": sha256_file(authorized.seal_path),
        "source_aggregate_sha256": verified.source_manifest["aggregate_sha256"],
        "raw_dataset_aggregate_sha256": verified.manifest["raw_data_manifest"][
            "dataset_aggregate_sha256"
        ],
        "selection_manifest_sha256": verified.manifest["selection_manifest"]["sha256"],
    }


def _run_one_record(
    authorized: AuthorizedExecution,
    *,
    dataset_name: str,
    data_seed: int,
    model_seed: int,
    record_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    both, weak, probe, evaluation, checks, counts = _record_data(
        authorized.preflight, dataset_name, data_seed
    )
    contract = authorized.preflight.contract
    optimization = contract["optimization"]
    metadata = {
        "record_id": record_id,
        "dataset": dataset_name,
        "data_seed": int(data_seed),
        "model_seed": int(model_seed),
    }
    rows, both_model, weak_model, initial_gaps = _train_pair(
        both_batch=both,
        weak_batch=weak,
        probe=probe,
        model_seed=int(model_seed),
        channels=tuple(contract["model"]["hidden_channels"]),
        learning_rate=float(optimization["learning_rate"]),
        steps=int(optimization["steps"]),
        log_every=int(optimization["log_every"]),
        record_metadata=metadata,
    )
    evaluation_values = {
        "both": semi_real_evaluation_accuracies(both_model, evaluation),
        "weak_only": semi_real_evaluation_accuracies(weak_model, evaluation),
    }
    trajectory_summary = summarize_paired_trajectory(rows, contract)
    summary = {
        "schema_version": "semi-real-record-summary-v1",
        **metadata,
        "publication_scope": "semi-real",
        "input_bindings": _record_input_bindings_from_authorization(authorized),
        "sample_counts": counts,
        "intervention_checks": checks,
        "initial_response_gaps": initial_gaps,
        "all_initial_response_gaps_exactly_zero": all(value == 0.0 for value in initial_gaps.values()),
        "primary_and_trajectory": trajectory_summary,
        "evaluation_accuracy": evaluation_values,
        "secondary_endpoint_status": {
            "gsi5": contract["evaluation"]["gsi5"]["artifact_marker"],
            "gsi5_available": False,
            "gsi5_required_for_acceptance": False,
        },
        "resources": {
            "wall_time_seconds": time.perf_counter() - started,
            "process_peak_rss_bytes": _process_peak_rss_bytes(),
        },
    }
    return rows, summary


def _format_csv_value(value: Any) -> str | int:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return "" if not math.isfinite(numeric) else format(numeric, ".17g")
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _format_csv_value(row.get(field)) for field in fields})
    return output.getvalue().encode("utf-8")


def _record_directory(run_directory: Path, record_id: str) -> Path:
    return run_directory / "records" / record_id


def _write_record(
    run_directory: Path,
    record_id: str,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    directory = _record_directory(run_directory, record_id)
    directory.mkdir(parents=True, exist_ok=True)
    trajectory_path = directory / "trajectory.csv"
    summary_path = directory / "summary.json"
    marker_path = directory / "complete.json"
    _require(not marker_path.exists(), f"Record already has a completion marker: {record_id}")
    _atomic_write_bytes(trajectory_path, _csv_bytes(rows, RECORD_TRAJECTORY_FIELDS))
    _atomic_write_bytes(summary_path, canonical_json_bytes(summary))
    marker = {
        "schema_version": "semi-real-record-complete-v1",
        "record_id": record_id,
        "files": [
            _file_descriptor(summary_path, relative_path="summary.json"),
            _file_descriptor(trajectory_path, relative_path="trajectory.csv"),
        ],
    }
    _atomic_write_bytes(marker_path, canonical_json_bytes(marker))


def _read_record_trajectory(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(reader.fieldnames == RECORD_TRAJECTORY_FIELDS, "Trajectory CSV schema changed.")
            rows: list[dict[str, Any]] = []
            for raw in reader:
                row: dict[str, Any] = {
                    "record_id": raw["record_id"],
                    "dataset": raw["dataset"],
                    "data_seed": int(raw["data_seed"]),
                    "model_seed": int(raw["model_seed"]),
                    "condition": raw["condition"],
                    "step": int(raw["step"]),
                }
                for field in RECORD_TRAJECTORY_FIELDS[6:]:
                    row[field] = float(raw[field])
                    _require(math.isfinite(row[field]), f"Non-finite trajectory value: {field}")
                rows.append(row)
    except (OSError, UnicodeError, csv.Error, TypeError, ValueError) as error:
        raise SemiRealStudyError(f"Cannot parse trajectory {path}: {error}") from error
    _require(
        len({(row["condition"], row["step"]) for row in rows}) == len(rows),
        "Trajectory contains duplicate arm/step rows.",
    )
    return rows


def _load_completed_record(
    run_directory: Path,
    record_id: str,
    contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    directory = _record_directory(run_directory, record_id)
    marker_path = directory / "complete.json"
    if not marker_path.exists():
        return None
    _require(directory.is_dir() and not directory.is_symlink(), f"Unsafe record directory: {directory}")
    marker = load_strict_json(marker_path)
    _require(
        set(marker) == {"schema_version", "record_id", "files"}
        and marker["schema_version"] == "semi-real-record-complete-v1"
        and marker["record_id"] == record_id,
        f"Invalid completion marker for {record_id}.",
    )
    expected_names = {"complete.json", "summary.json", "trajectory.csv"}
    actual_names = {path.name for path in directory.iterdir() if path.is_file()}
    _require(actual_names == expected_names, f"Record {record_id} has missing or extra files.")
    _require(len(marker["files"]) == 2, f"Record {record_id} marker file list changed.")
    for descriptor in marker["files"]:
        _verify_reference(directory, descriptor)
    summary = load_strict_json(directory / "summary.json")
    required_summary_keys = {
        "schema_version",
        "record_id",
        "dataset",
        "data_seed",
        "model_seed",
        "publication_scope",
        "input_bindings",
        "sample_counts",
        "intervention_checks",
        "initial_response_gaps",
        "all_initial_response_gaps_exactly_zero",
        "primary_and_trajectory",
        "evaluation_accuracy",
        "secondary_endpoint_status",
        "resources",
    }
    _require(
        set(summary) == required_summary_keys
        and summary.get("schema_version") == "semi-real-record-summary-v1"
        and summary.get("record_id") == record_id,
        f"Record summary schema/ID mismatch: {record_id}",
    )
    provenance = load_strict_json(run_directory / "provenance.json")
    expected_input_bindings = {
        field: provenance[field]
        for field in (
            "execution_contract_sha256",
            "preflight_manifest_sha256",
            "authorization_seal_sha256",
            "source_aggregate_sha256",
            "raw_dataset_aggregate_sha256",
            "selection_manifest_sha256",
        )
    }
    _require(
        summary["input_bindings"] == expected_input_bindings,
        f"Completed record is not bound to this run's exact authorized inputs: {record_id}",
    )
    expected_dataset, expected_data_seed, expected_model_seed = expected_record_coordinates(contract)[
        record_id
    ]
    _require(
        summary["publication_scope"] == "semi-real"
        and (summary["dataset"], summary["data_seed"], summary["model_seed"])
        == (expected_dataset, expected_data_seed, expected_model_seed),
        f"Record ID is not bound to its frozen factorial coordinates: {record_id}",
    )
    dataset_specification = _find_dataset_specification(contract, expected_dataset)
    expected_counts = {
        role: 2 * int(dataset_specification[f"{role}_per_class"])
        for role in contract["selection"]["role_order"]
    }
    _require(summary["sample_counts"] == expected_counts, f"Frozen sample counts changed: {record_id}")
    intervention = summary["intervention_checks"]
    _require(
        set(intervention) == set(contract["intervention"]["required_checks"])
        and all(type(value) is bool for value in intervention.values()),
        f"Intervention audit schema/types changed: {record_id}",
    )
    expected_gsi = {
        "gsi5": contract["evaluation"]["gsi5"]["artifact_marker"],
        "gsi5_available": False,
        "gsi5_required_for_acceptance": False,
    }
    _require(
        summary["secondary_endpoint_status"] == expected_gsi,
        f"Fail-closed GSI-5 omission marker changed: {record_id}",
    )
    evaluation = summary["evaluation_accuracy"]
    _require(set(evaluation) == {"both", "weak_only"}, f"Evaluation arm schema changed: {record_id}")
    for arm in ("both", "weak_only"):
        _require(
            set(evaluation[arm]) == {"cue_free", "aligned", "conflicted"}
            and all(
                type(value) in {int, float} and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
                for value in evaluation[arm].values()
            ),
            f"Evaluation accuracy schema/range changed: {record_id}/{arm}",
        )
    resources = summary["resources"]
    _require(
        set(resources) == {"wall_time_seconds", "process_peak_rss_bytes"}
        and type(resources["wall_time_seconds"]) in {int, float}
        and math.isfinite(float(resources["wall_time_seconds"]))
        and float(resources["wall_time_seconds"]) >= 0.0
        and type(resources["process_peak_rss_bytes"]) is int
        and resources["process_peak_rss_bytes"] > 0,
        f"Resource metrics are invalid: {record_id}",
    )
    _require(
        set(summary["initial_response_gaps"])
        == {"core_both_minus_weak", "cue_both_minus_weak"}
        and type(summary["all_initial_response_gaps_exactly_zero"]) is bool,
        f"Initial response schema changed: {record_id}",
    )
    rows = _read_record_trajectory(directory / "trajectory.csv")
    expected_rows = 2 * int(contract["optimization"]["logged_point_count"])
    _require(len(rows) == expected_rows, f"Record trajectory row count changed: {record_id}")
    _require(
        all(
            row["record_id"] == record_id
            and row["dataset"] == summary["dataset"]
            and row["data_seed"] == summary["data_seed"]
            and row["model_seed"] == summary["model_seed"]
            for row in rows
        ),
        f"Trajectory metadata differs from summary: {record_id}",
    )
    recomputed = summarize_paired_trajectory(rows, contract)
    _require(
        canonical_json_bytes(recomputed)
        == canonical_json_bytes(summary["primary_and_trajectory"]),
        f"Record summary does not match deterministic trajectory analysis: {record_id}",
    )
    step_zero = {
        row["condition"]: row
        for row in rows
        if row["step"] == 0
    }
    _require(set(step_zero) == {"both", "weak_only"}, f"Missing step-zero rows: {record_id}")
    initial_gaps = {
        "core_both_minus_weak": step_zero["both"]["core_response"]
        - step_zero["weak_only"]["core_response"],
        "cue_both_minus_weak": step_zero["both"]["cue_response"]
        - step_zero["weak_only"]["cue_response"],
    }
    _require(
        initial_gaps == summary["initial_response_gaps"]
        and summary["all_initial_response_gaps_exactly_zero"]
        == all(value == 0.0 for value in initial_gaps.values()),
        f"Initial-response equality summary drift: {record_id}",
    )
    return summary


def _summary_flat_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    primary = summary["primary_and_trajectory"]
    evaluation = summary["evaluation_accuracy"]
    return {
        "record_id": summary["record_id"],
        "dataset": summary["dataset"],
        "data_seed": summary["data_seed"],
        "model_seed": summary["model_seed"],
        "weak_auc_gap": primary["weak_auc_gap"],
        "weak_only_learnable": primary["weak_only_learnable"],
        "outcome_suppression": primary["outcome_suppression"],
        "causal_certificate": primary["causal_certificate"],
        "both_first_hit_tau": primary["both_first_hit_tau"],
        "weak_only_first_hit_tau": primary["weak_only_first_hit_tau"],
        "first_drift_crossing_tau": primary["first_drift_crossing_tau"],
        "first_response_crossing_tau": primary["first_response_crossing_tau"],
        "drift_sign_change_count": primary["drift_sign_change_count"],
        "response_sign_change_count": primary["response_sign_change_count"],
        "positive_drift_area": primary["positive_drift_area"],
        "negative_drift_area": primary["negative_drift_area"],
        "both_cue_free_accuracy": evaluation["both"]["cue_free"],
        "both_aligned_accuracy": evaluation["both"]["aligned"],
        "both_conflicted_accuracy": evaluation["both"]["conflicted"],
        "weak_only_cue_free_accuracy": evaluation["weak_only"]["cue_free"],
        "weak_only_aligned_accuracy": evaluation["weak_only"]["aligned"],
        "weak_only_conflicted_accuracy": evaluation["weak_only"]["conflicted"],
        "wall_time_seconds": summary["resources"]["wall_time_seconds"],
        "process_peak_rss_bytes": summary["resources"]["process_peak_rss_bytes"],
    }


def two_way_pigeonhole_bootstrap(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    lower_quantile: float = 0.025,
    upper_quantile: float = 0.975,
) -> dict[str, Any]:
    matrix = np.asarray(values, dtype=float)
    _require(matrix.ndim == 2 and matrix.size > 0, "Two-way bootstrap requires one matrix.")
    _require(np.isfinite(matrix).all(), "Two-way bootstrap matrix contains non-finite values.")
    _require(replicates >= 1, "Two-way bootstrap requires at least one replicate.")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(replicates), dtype=float)
    for replicate in range(int(replicates)):
        data_indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        model_indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        estimates[replicate] = float(matrix[np.ix_(data_indices, model_indices)].mean())
    return {
        "point_estimate": float(matrix.mean()),
        "ci95_low": float(np.quantile(estimates, lower_quantile, method="linear")),
        "ci95_high": float(np.quantile(estimates, upper_quantile, method="linear")),
    }


def infer_complete_records(
    contract: Mapping[str, Any], summaries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected_ids = expected_record_ids(contract)
    identifiers = [str(summary["record_id"]) for summary in summaries]
    _require(len(identifiers) == len(set(identifiers)), "Duplicate record summaries are forbidden.")
    _require(set(identifiers) == set(expected_ids), "Inference requires exactly the expected 64 records.")
    coordinates = expected_record_coordinates(contract)
    _require(
        all(
            (str(summary["dataset"]), int(summary["data_seed"]), int(summary["model_seed"]))
            == coordinates[str(summary["record_id"])]
            for summary in summaries
        ),
        "Record IDs are not bound to their frozen dataset/data/model coordinates.",
    )
    inference = contract["inference"]
    data_seeds = list(contract["seed_design"]["data_seeds"])
    model_seeds = list(contract["seed_design"]["model_seeds"])
    dataset_results: dict[str, Any] = {}
    metric_paths = {
        "mean_weak_auc_gap": ("weak_auc_gap", float),
        "weak_only_learnability_rate": ("weak_only_learnable", lambda value: float(bool(value))),
        "causal_certificate_rate": ("causal_certificate", lambda value: float(bool(value))),
    }
    for dataset in contract["datasets"]:
        name = dataset["name"]
        group = [summary for summary in summaries if summary["dataset"] == name]
        _require(len(group) == 32, f"{name} requires all 4x8 records.")
        cell = {(int(item["data_seed"]), int(item["model_seed"])): item for item in group}
        _require(
            set(cell)
            == {(int(data_seed), int(model_seed)) for data_seed in data_seeds for model_seed in model_seeds},
            f"{name} crossed factorial is incomplete.",
        )
        metrics: dict[str, Any] = {}
        for label, (field, converter) in metric_paths.items():
            matrix = np.asarray(
                [
                    [converter(cell[(int(data_seed), int(model_seed))]["primary_and_trajectory"][field]) for model_seed in model_seeds]
                    for data_seed in data_seeds
                ],
                dtype=float,
            )
            metrics[label] = two_way_pigeonhole_bootstrap(
                matrix,
                replicates=int(inference["bootstrap_replicates"]),
                seed=int(inference["bootstrap_seed"]),
                lower_quantile=float(inference["ci"]["lower_quantile"]),
                upper_quantile=float(inference["ci"]["upper_quantile"]),
            )
        dataset_results[name] = {
            "record_count": 32,
            "data_seed_count": 4,
            "model_seed_count": 8,
            "weak_only_learnable_count": int(
                sum(bool(item["primary_and_trajectory"]["weak_only_learnable"]) for item in group)
            ),
            "causal_certificate_count": int(
                sum(bool(item["primary_and_trajectory"]["causal_certificate"]) for item in group)
            ),
            "metrics": metrics,
        }
    return {
        "schema_version": "semi-real-inference-v1",
        "method": "two_way_pigeonhole_bootstrap",
        "row_iid_intervals_used": False,
        "bootstrap_replicates": int(inference["bootstrap_replicates"]),
        "bootstrap_seed": int(inference["bootstrap_seed"]),
        "datasets": dataset_results,
    }


def acceptance_from_inference(
    contract: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    inference: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = contract["acceptance"]
    dataset_results: dict[str, Any] = {}
    for dataset in contract["datasets"]:
        name = dataset["name"]
        aggregate = inference["datasets"][name]
        group = [summary for summary in summaries if summary["dataset"] == name]
        coverage = len(group) == thresholds["required_records_per_dataset"]
        learnability = (
            aggregate["weak_only_learnable_count"]
            >= thresholds["minimum_weak_only_learnable_count_per_dataset"]
        )
        causal = (
            aggregate["causal_certificate_count"]
            >= thresholds["minimum_causal_certificate_count_per_dataset"]
        )
        auc_lower = aggregate["metrics"]["mean_weak_auc_gap"]["ci95_low"]
        auc = auc_lower > thresholds["weak_auc_gap_ci95_lower_strictly_greater_than"]
        intervention = all(all(item["intervention_checks"].values()) for item in group)
        initial = all(item["all_initial_response_gaps_exactly_zero"] for item in group)
        passed = bool(coverage and learnability and causal and auc and intervention and initial)
        dataset_results[name] = {
            "passed": passed,
            "coverage_passed": coverage,
            "weak_only_learnability_passed": learnability,
            "causal_certificate_passed": causal,
            "weak_auc_ci_lower_passed": auc,
            "intervention_checks_passed": intervention,
            "initial_response_equality_passed": initial,
            "record_count": len(group),
            "weak_only_learnable_count": aggregate["weak_only_learnable_count"],
            "causal_certificate_count": aggregate["causal_certificate_count"],
            "weak_auc_ci95_low": auc_lower,
        }
    return {
        "schema_version": "semi-real-acceptance-v1",
        "publication_scope": "semi-real",
        "thresholds": thresholds,
        "datasets": dataset_results,
        "overall_passed": all(item["passed"] for item in dataset_results.values()),
        "gsi5_status": contract["evaluation"]["gsi5"]["artifact_marker"],
    }


def _summary_artifact_values(
    contract: Mapping[str, Any], summaries: Sequence[Mapping[str, Any]]
) -> tuple[bytes, bytes, dict[str, Any], dict[str, Any]]:
    ordered = sorted(summaries, key=lambda item: expected_record_ids(contract).index(item["record_id"]))
    summary_json = canonical_json_bytes(
        {"schema_version": "semi-real-record-summary-collection-v1", "records": ordered}
    )
    flat = [_summary_flat_row(item) for item in ordered]
    summary_csv = _csv_bytes(flat, list(flat[0]))
    inference = infer_complete_records(contract, ordered)
    acceptance = acceptance_from_inference(contract, ordered, inference)
    return summary_json, summary_csv, inference, acceptance


def _write_artifact_manifest(run_directory: Path) -> None:
    manifest_path = run_directory / "artifact_manifest.json"
    sidecar_path = run_directory / "artifact_manifest.sha256"
    entries: list[dict[str, Any]] = []
    for path in sorted(run_directory.rglob("*"), key=lambda item: item.as_posix()):
        if path in {manifest_path, sidecar_path}:
            continue
        if path.is_symlink():
            raise SemiRealStudyError(f"Symlinked run artifact is forbidden: {path}")
        if path.is_file():
            entries.append(_file_descriptor(path, relative_path=path.relative_to(run_directory).as_posix()))
    manifest = {
        "schema_version": "semi-real-artifact-manifest-v1",
        "hash_algorithm": "sha256",
        "file_count": len(entries),
        "files": entries,
    }
    _atomic_write_bytes(manifest_path, canonical_json_bytes(manifest))
    _write_manifest_sidecar(manifest_path)


def _copy_preflight_provenance(
    authorized: AuthorizedExecution, run_directory: Path, repository_root: Path
) -> None:
    verified = authorized.preflight
    contract = verified.contract
    preflight_directory = verified.preflight_path.parent
    mappings = {
        verified.contract_path: run_directory / "contract.resolved.json",
        verified.preflight_path: run_directory / "preflight.manifest.json",
        verified.preflight_path.with_suffix(".sha256"): run_directory / "preflight.manifest.sha256",
        authorized.seal_path: run_directory / "authorization.seal.json",
        preflight_directory / "source_manifest.json": run_directory / "source_manifest.json",
        preflight_directory / "raw_data_manifest.json": run_directory / "raw_data_manifest.json",
        preflight_directory / "selection_manifest.json": run_directory / "selection_manifest.json",
        repository_root / contract["execution_environment"]["environment_lock_path"]: run_directory / "environment-lock.txt",
    }
    for source, destination in mappings.items():
        _write_or_verify(destination, source.read_bytes())
    provenance = {
        "schema_version": "semi-real-provenance-v1",
        "publication_scope": "semi-real",
        "execution_contract_sha256": sha256_file(verified.contract_path),
        "preflight_manifest_sha256": sha256_file(verified.preflight_path),
        "authorization_seal_sha256": sha256_file(authorized.seal_path),
        "historical_contract_sha256": contract["historical_contract"]["sha256"],
        "environment_lock_sha256": contract["execution_environment"]["environment_lock_sha256"],
        "source_manifest_sha256": verified.manifest["source_manifest"]["sha256"],
        "source_aggregate_sha256": verified.source_manifest["aggregate_sha256"],
        "raw_data_manifest_sha256": verified.manifest["raw_data_manifest"]["sha256"],
        "raw_dataset_aggregate_sha256": verified.manifest["raw_data_manifest"]["dataset_aggregate_sha256"],
        "selection_manifest_sha256": verified.manifest["selection_manifest"]["sha256"],
        "expected_record_ids": expected_record_ids(contract),
        "gsi5_status": contract["evaluation"]["gsi5"]["artifact_marker"],
    }
    _write_or_verify(run_directory / "provenance.json", canonical_json_bytes(provenance))
    environment_path = run_directory / "environment.json"
    if not environment_path.exists():
        runtime = {
            "schema_version": "semi-real-runtime-environment-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "environment": _current_environment(contract, repository_root),
            "git": git_state(repository_root),
            "command": sys.argv,
        }
        _atomic_write_bytes(environment_path, canonical_json_bytes(runtime))
    else:
        runtime = load_strict_json(environment_path)
        _require(runtime.get("environment") == _current_environment(contract, repository_root), "Resume environment drifted.")
        _require(runtime.get("git") == git_state(repository_root), "Resume git state drifted.")


def run_scientific_study(
    contract_path: Path,
    preflight_path: Path,
    seal_path: Path,
    *,
    data_root: Path,
    output_directory: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Execute all 64 records only after a separately supplied valid seal."""
    frozen_contract = load_execution_contract(contract_path, repository_root=repository_root)
    output_directory = _require_workspace_subpath(
        output_directory,
        repository_root,
        path_role="Scientific run output",
    )
    _require_existing_tree_without_symlinks(
        output_directory,
        path_role="Scientific run output",
    )
    _configure_determinism(frozen_contract)
    authorized = verify_authorization(
        contract_path,
        preflight_path,
        seal_path,
        data_root=data_root,
        repository_root=repository_root,
    )
    contract = authorized.preflight.contract
    if (output_directory / "artifact_manifest.json").exists():
        validate_run_artifacts(output_directory, repository_root=repository_root)
        return output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    records_directory = output_directory / "records"
    records_directory.mkdir(exist_ok=True)
    _require(
        records_directory.is_dir() and not records_directory.is_symlink(),
        f"Scientific records directory must not be symlinked: {records_directory}",
    )
    expected_ids = set(expected_record_ids(contract))
    for path in records_directory.iterdir():
        _require(
            path.is_dir() and not path.is_symlink() and path.name in expected_ids,
            f"Unexpected resume entry in records directory: {path.name}",
        )
        allowed_record_files = {"trajectory.csv", "summary.json", "complete.json"}
        _require(
            {item.name for item in path.iterdir()} <= allowed_record_files,
            f"Unexpected partial-record artifact: {path.name}",
        )
    _copy_preflight_provenance(authorized, output_directory, repository_root)
    completed_records = {
        record_id: summary
        for record_id in expected_record_ids(contract)
        if (summary := _load_completed_record(output_directory, record_id, contract))
        is not None
    }
    run_started = time.perf_counter()
    summaries: list[dict[str, Any]] = []
    template = contract["seed_design"]["record_id_format"]
    for dataset in contract["datasets"]:
        for data_seed in contract["seed_design"]["data_seeds"]:
            for model_seed in contract["seed_design"]["model_seeds"]:
                record_id = template.format(
                    dataset=dataset["name"], data_seed=data_seed, model_seed=model_seed
                )
                completed = completed_records.get(record_id)
                if completed is not None:
                    summaries.append(completed)
                    continue
                rows, summary = _run_one_record(
                    authorized,
                    dataset_name=dataset["name"],
                    data_seed=int(data_seed),
                    model_seed=int(model_seed),
                    record_id=record_id,
                )
                _write_record(output_directory, record_id, rows, summary)
                summaries.append(summary)

    # Recompute every external binding after the final optimizer step.  Output is
    # confined to the ignored local workspace, so artifacts cannot dirty source state.
    verify_authorization(
        contract_path,
        preflight_path,
        seal_path,
        data_root=data_root,
        repository_root=repository_root,
    )
    summary_json, summary_csv, inference, acceptance = _summary_artifact_values(contract, summaries)
    _atomic_write_bytes(output_directory / "record_summary.json", summary_json)
    _atomic_write_bytes(output_directory / "record_summary.csv", summary_csv)
    _atomic_write_bytes(output_directory / "inference.json", canonical_json_bytes(inference))
    _atomic_write_bytes(output_directory / "acceptance.json", canonical_json_bytes(acceptance))
    resources = {
        "schema_version": "semi-real-resource-usage-v1",
        "total_wall_time_seconds_this_invocation": time.perf_counter() - run_started,
        "process_peak_rss_bytes": max(
            _process_peak_rss_bytes(),
            max(item["resources"]["process_peak_rss_bytes"] for item in summaries),
        ),
        "records": {
            item["record_id"]: item["resources"] for item in sorted(summaries, key=lambda row: row["record_id"])
        },
    }
    _atomic_write_bytes(output_directory / "resource_usage.json", canonical_json_bytes(resources))
    _write_artifact_manifest(output_directory)
    validate_run_artifacts(output_directory, repository_root=repository_root)
    return output_directory


def validate_run_artifacts(
    run_directory: Path, *, repository_root: Path = REPOSITORY_ROOT
) -> dict[str, Any]:
    """Deterministically reject missing/duplicate records and any artifact hash drift."""
    run_directory = _require_workspace_subpath(
        run_directory,
        repository_root,
        path_role="Scientific run directory",
    )
    _require_existing_tree_without_symlinks(
        run_directory,
        path_role="Scientific run directory",
    )
    manifest_path = run_directory / "artifact_manifest.json"
    _verify_manifest_sidecar(manifest_path)
    manifest = load_strict_json(manifest_path)
    _require(
        set(manifest) == {"schema_version", "hash_algorithm", "file_count", "files"}
        and manifest["schema_version"] == "semi-real-artifact-manifest-v1"
        and manifest["hash_algorithm"] == "sha256",
        "Artifact manifest schema changed.",
    )
    listed = {item["path"]: item for item in manifest["files"]}
    _require(len(listed) == len(manifest["files"]) == manifest["file_count"], "Duplicate artifact paths.")
    all_run_paths = list(run_directory.rglob("*"))
    _require(
        not any(path.is_symlink() for path in all_run_paths),
        "Symlinked run artifacts or directories are forbidden.",
    )
    actual_paths = {
        path.relative_to(run_directory).as_posix()
        for path in all_run_paths
        if path.is_file()
        and path not in {manifest_path, run_directory / "artifact_manifest.sha256"}
    }
    _require(set(listed) == actual_paths, "Artifact manifest is incomplete or lists absent files.")
    for relative, descriptor in listed.items():
        path = run_directory / relative
        _require(not path.is_symlink(), f"Symlinked artifact is forbidden: {relative}")
        _require(path.stat().st_size == descriptor["size"], f"Artifact size drift: {relative}")
        _require(sha256_file(path) == descriptor["sha256"], f"Artifact hash drift: {relative}")

    contract_path = run_directory / "contract.resolved.json"
    contract = load_execution_contract(contract_path, repository_root=repository_root)
    required = set(contract["artifacts"]["required_final_files"])
    _require(required <= {path.name for path in run_directory.iterdir()}, "Required final artifact is missing.")
    provenance = load_strict_json(run_directory / "provenance.json")
    preflight_path = run_directory / "preflight.manifest.json"
    _verify_manifest_sidecar(preflight_path)
    preflight = load_strict_json(preflight_path)
    source_manifest_path = run_directory / "source_manifest.json"
    raw_manifest_path = run_directory / "raw_data_manifest.json"
    selection_manifest_path = run_directory / "selection_manifest.json"
    source_manifest = load_strict_json(source_manifest_path)
    raw_manifest = load_strict_json(raw_manifest_path)
    selection_manifest = load_strict_json(selection_manifest_path)
    validate_selection_manifest(contract, selection_manifest)
    _require(
        source_manifest["aggregate_sha256"]
        == _sha256_bytes(canonical_json_bytes(source_manifest["files"])),
        "Copied source aggregate is internally inconsistent.",
    )
    _require(
        set(raw_manifest) == {"schema_version", "hash_algorithm", "file_scope", "datasets"}
        and raw_manifest["schema_version"] == "semi-real-raw-data-manifest-v1"
        and raw_manifest["hash_algorithm"] == "sha256"
        and raw_manifest["file_scope"] == contract["preflight"]["raw_file_scope"],
        "Copied raw-data manifest schema changed.",
    )
    for dataset in raw_manifest["datasets"]:
        _require(
            set(dataset) == {"name", "official_archives", "files", "aggregate_sha256"},
            "Copied raw dataset entry schema changed.",
        )
        specification = _find_dataset_specification(contract, dataset["name"])
        expected_archives = [
            {
                "filename": archive["filename"],
                "official_md5": archive["md5"],
                "verified": True,
            }
            for archive in specification["official_archives"]
        ]
        _require(
            dataset["official_archives"] == expected_archives,
            f"Official archive identity changed for {dataset['name']}.",
        )
        _require(
            dataset["aggregate_sha256"]
            == _sha256_bytes(canonical_json_bytes(dataset["files"])),
            f"Copied raw aggregate is inconsistent for {dataset['name']}.",
        )
    raw_aggregates = {
        item["name"]: item["aggregate_sha256"] for item in raw_manifest["datasets"]
    }
    expected_provenance = {
        "schema_version": "semi-real-provenance-v1",
        "publication_scope": "semi-real",
        "execution_contract_sha256": sha256_file(contract_path),
        "preflight_manifest_sha256": sha256_file(preflight_path),
        "authorization_seal_sha256": sha256_file(run_directory / "authorization.seal.json"),
        "historical_contract_sha256": contract["historical_contract"]["sha256"],
        "environment_lock_sha256": sha256_file(run_directory / "environment-lock.txt"),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_aggregate_sha256": source_manifest["aggregate_sha256"],
        "raw_data_manifest_sha256": sha256_file(raw_manifest_path),
        "raw_dataset_aggregate_sha256": raw_aggregates,
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "expected_record_ids": expected_record_ids(contract),
        "gsi5_status": contract["evaluation"]["gsi5"]["artifact_marker"],
    }
    _require(provenance == expected_provenance, "Run provenance bindings are inconsistent.")
    _require(
        selection_manifest["execution_contract_sha256"]
        == expected_provenance["execution_contract_sha256"],
        "Copied selection manifest does not bind the resolved execution contract.",
    )
    _require(
        preflight["contract"]["sha256"] == expected_provenance["execution_contract_sha256"]
        and preflight["source_manifest"]["sha256"] == expected_provenance["source_manifest_sha256"]
        and preflight["raw_data_manifest"]["sha256"] == expected_provenance["raw_data_manifest_sha256"]
        and preflight["selection_manifest"]["sha256"] == expected_provenance["selection_manifest_sha256"],
        "Copied preflight does not bind the copied provenance manifests.",
    )
    seal = load_strict_json(run_directory / "authorization.seal.json")
    expected_seal_bindings = {
        "execution_contract_sha256": expected_provenance["execution_contract_sha256"],
        "preflight_manifest_sha256": expected_provenance["preflight_manifest_sha256"],
        "historical_contract_sha256": expected_provenance["historical_contract_sha256"],
        "source_manifest_sha256": expected_provenance["source_manifest_sha256"],
        "source_aggregate_sha256": expected_provenance["source_aggregate_sha256"],
        "environment_lock_sha256": expected_provenance["environment_lock_sha256"],
        "raw_data_manifest_sha256": expected_provenance["raw_data_manifest_sha256"],
        "raw_dataset_aggregate_sha256": raw_aggregates,
        "selection_manifest_sha256": expected_provenance["selection_manifest_sha256"],
        "expected_record_ids": expected_record_ids(contract),
    }
    expected_git = _validate_seal_document(contract, seal, expected_seal_bindings)
    _require(
        set(preflight) == PREFLIGHT_MANIFEST_KEYS
        and preflight["schema_version"] == contract["preflight"]["schema_version"]
        and preflight["study_id"] == contract["study_id"]
        and preflight["publication_scope"] == "semi-real"
        and preflight["ready_for_authorization"] is True
        and preflight["blockers"] == []
        and preflight["model_constructed"] is False
        and preflight["optimizer_constructed"] is False
        and preflight["model_outcomes_inspected"] is False
        and preflight["expected_record_ids"] == expected_record_ids(contract),
        "Copied preflight was not authorization-ready and outcome-free.",
    )
    runtime = load_strict_json(run_directory / "environment.json")
    _require(
        set(runtime) == {"schema_version", "created_utc", "environment", "git", "command"}
        and runtime["schema_version"] == "semi-real-runtime-environment-v1"
        and preflight["git"]["clean"] is True
        and runtime["git"]["clean"] is True
        and preflight["git"]["commit"]
        == runtime["git"]["commit"]
        == expected_git["commit"]
        and preflight["git"]["status_porcelain"] == []
        and runtime["git"]["status_porcelain"] == [],
        "Seal, preflight, and runtime git evidence do not identify one clean commit.",
    )

    expected_ids = expected_record_ids(contract)
    records_root = run_directory / "records"
    directory_ids = {path.name for path in records_root.iterdir() if path.is_dir()}
    _require(directory_ids == set(expected_ids), "Record directories are missing or duplicated/extra.")
    summaries: list[dict[str, Any]] = []
    for record_id in expected_ids:
        summary = _load_completed_record(run_directory, record_id, contract)
        _require(summary is not None, f"Record is incomplete: {record_id}")
        summaries.append(summary)
    summary_json, summary_csv, inference, acceptance = _summary_artifact_values(contract, summaries)
    _require((run_directory / "record_summary.json").read_bytes() == summary_json, "Combined summary JSON is not deterministic.")
    _require((run_directory / "record_summary.csv").read_bytes() == summary_csv, "Combined summary CSV is not deterministic.")
    _require((run_directory / "inference.json").read_bytes() == canonical_json_bytes(inference), "Inference artifact differs from deterministic recomputation.")
    _require((run_directory / "acceptance.json").read_bytes() == canonical_json_bytes(acceptance), "Acceptance artifact differs from deterministic recomputation.")
    resource_usage = load_strict_json(run_directory / "resource_usage.json")
    expected_record_resources = {
        summary["record_id"]: summary["resources"]
        for summary in sorted(summaries, key=lambda item: item["record_id"])
    }
    _require(
        set(resource_usage)
        == {
            "schema_version",
            "total_wall_time_seconds_this_invocation",
            "process_peak_rss_bytes",
            "records",
        }
        and resource_usage["schema_version"] == "semi-real-resource-usage-v1"
        and resource_usage["records"] == expected_record_resources
        and type(resource_usage["total_wall_time_seconds_this_invocation"]) in {int, float}
        and math.isfinite(float(resource_usage["total_wall_time_seconds_this_invocation"]))
        and float(resource_usage["total_wall_time_seconds_this_invocation"]) >= 0.0
        and type(resource_usage["process_peak_rss_bytes"]) is int
        and resource_usage["process_peak_rss_bytes"]
        >= max(value["process_peak_rss_bytes"] for value in expected_record_resources.values()),
        "Resource artifact does not exactly reconcile with all record summaries.",
    )
    return {
        "valid": True,
        "publication_scope": "semi-real",
        "record_count": len(summaries),
        "artifact_file_count": manifest["file_count"],
        "overall_accepted": acceptance["overall_passed"],
    }


def _toy_role(spec: SemiRealTaskSpec, *, offset: int, per_class: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = 2 * per_class
    values = torch.arange(count * 8 * 8, dtype=torch.int64).add(offset).remainder(256)
    raw = values.to(torch.uint8).reshape(count, 8, 8)
    labels = torch.tensor([0] * per_class + [1] * per_class, dtype=torch.long)
    identifiers = torch.arange(offset, offset + count, dtype=torch.long)
    return prepare_core_images(raw, spec.image_size), labels, identifiers


def run_engineering_smoke(
    contract_path: Path,
    *,
    output_directory: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Run an explicitly non-scientific toy smoke with excluded seeds only."""
    contract = load_execution_contract(contract_path, repository_root=repository_root)
    output_directory = _require_workspace_subpath(
        output_directory,
        repository_root,
        path_role="Engineering smoke output",
    )
    smoke = contract["engineering_smoke"]
    scientific_seeds = set(contract["seed_design"]["data_seeds"]) | set(
        contract["seed_design"]["model_seeds"]
    )
    _require(
        smoke["data_seed"] not in scientific_seeds and smoke["model_seed"] not in scientific_seeds,
        "Engineering smoke seeds overlap scientific seeds.",
    )
    _require(not output_directory.exists(), f"Smoke output already exists: {output_directory}")
    _configure_determinism(contract)
    spec = SemiRealTaskSpec(
        "mnist",
        3,
        8,
        cue_strength=float(contract["intervention"]["cue_strength"]),
        image_size=int(smoke["image_size"]),
        cue_patch_size=2,
    )
    train = _toy_role(spec, offset=int(smoke["data_seed"]), per_class=int(smoke["per_class"]))
    probe_values = _toy_role(spec, offset=int(smoke["data_seed"]) + 100, per_class=int(smoke["per_class"]))
    evaluation_values = _toy_role(spec, offset=int(smoke["data_seed"]) + 200, per_class=int(smoke["per_class"]))
    both, weak = make_paired_semi_real_task(*train, spec)
    checks = _audit_intervention(both, weak)
    probe = make_semi_real_probe(*probe_values, spec)
    evaluation = make_semi_real_probe(*evaluation_values, spec)
    metadata = {
        "record_id": "engineering-smoke-toy",
        "dataset": "generated_toy_not_mnist",
        "data_seed": int(smoke["data_seed"]),
        "model_seed": int(smoke["model_seed"]),
    }
    rows, both_model, weak_model, initial_gaps = _train_pair(
        both_batch=both,
        weak_batch=weak,
        probe=probe,
        model_seed=int(smoke["model_seed"]),
        channels=tuple(smoke["hidden_channels"]),
        learning_rate=float(contract["optimization"]["learning_rate"]),
        steps=int(smoke["steps"]),
        log_every=int(smoke["log_every"]),
        record_metadata=metadata,
    )
    summary = {
        "schema_version": "semi-real-engineering-smoke-v1",
        "scope": smoke["scope_label"],
        "scientific_record": False,
        "acceptance_eligible": False,
        "scientific_seeds_used": False,
        "data_source": "deterministically_generated_uint8_toy_images",
        "intervention_checks": checks,
        "initial_response_gaps": initial_gaps,
        "evaluation_accuracy": {
            "both": semi_real_evaluation_accuracies(both_model, evaluation),
            "weak_only": semi_real_evaluation_accuracies(weak_model, evaluation),
        },
        "gsi5_status": contract["evaluation"]["gsi5"]["artifact_marker"],
    }
    output_directory.mkdir(parents=True)
    _atomic_write_bytes(output_directory / "smoke.trajectory.csv", _csv_bytes(rows, RECORD_TRAJECTORY_FIELDS))
    _atomic_write_bytes(output_directory / "smoke.summary.json", canonical_json_bytes(summary))
    _write_artifact_manifest(output_directory)
    return output_directory
