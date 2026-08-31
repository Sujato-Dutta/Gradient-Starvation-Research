#!/usr/bin/env python3
"""Create deterministic compact archives from the two completed scientific runs.

The archive copies every top-level run file byte-for-byte and deliberately omits
``records/**``.  The copied full-run artifact manifest still commits to every
omitted record file; ``archive_manifest.json`` makes that limitation explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLEAN_SOURCE_COMMIT = "9ebbc617a9aaad5c3c19e9d5fd4fa08f2d3517d0"


@dataclass(frozen=True)
class ArchiveSpec:
    study_id: str
    claim_ids: tuple[str, ...]
    source_run: str
    archive_root: str
    expected_top_level_file_count: int
    expected_full_manifest_file_count: int
    expected_omitted_record_file_count: int
    expected_artifact_manifest_sha256: str
    limitation: str


SPECS = (
    ArchiveSpec(
        study_id="semi-real-generated-cue-v1",
        claim_ids=("E26",),
        source_run="data/semi_real_generated_cue_v1/scientific-run-9ebbc61",
        archive_root="paper/artifacts/semi-real-generated-cue-v1-20260830",
        expected_top_level_file_count=17,
        expected_full_manifest_file_count=207,
        expected_omitted_record_file_count=192,
        expected_artifact_manifest_sha256=(
            "3ef7a7f2ce08857bace1451318a226a50df16807c526bdaca262e463646a0c72"
        ),
        limitation=(
            "This compact archive preserves every top-level completed-run file and the "
            "full-run artifact manifest, but omits 192 records/** files. Their paths, "
            "hashes, and sizes remain committed by artifact_manifest.json; omitted "
            "bytes cannot be recovered from hashes, so this archive is auditable but "
            "not a self-contained replay package."
        ),
    ),
    ArchiveSpec(
        study_id="expanded-nonlinear-beta-cdc-tradeoff-v1",
        claim_ids=("E27", "E28"),
        source_run="data/expanded_studies_execution_v1/scientific-run-9ebbc61",
        archive_root="paper/artifacts/expanded-studies-v1-20260830",
        expected_top_level_file_count=27,
        expected_full_manifest_file_count=1817,
        expected_omitted_record_file_count=1792,
        expected_artifact_manifest_sha256=(
            "54e4c4acd7d0e1b177f6c8da05bcb1ba848397d44056c447843a1df778564a96"
        ),
        limitation=(
            "This compact archive preserves every top-level completed-run file and the "
            "full-run artifact manifest, but omits 1,792 records/** files. Their paths, "
            "hashes, and sizes remain committed by artifact_manifest.json; omitted "
            "bytes cannot be recovered from hashes, so this archive is auditable but "
            "not a self-contained replay package."
        ),
    ),
)


class PackagingError(ValueError):
    """Raised when a source run does not match the reviewed packaging contract."""


def _signature(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PackagingError(f"Expected a JSON object at {path}.")
    return payload


def package(spec: ArchiveSpec, *, root: Path) -> dict[str, Any]:
    source = root / spec.source_run
    destination = root / spec.archive_root
    if destination.exists():
        raise PackagingError(f"Refusing to overwrite existing archive: {destination}")
    if not source.is_dir():
        raise PackagingError(f"Missing completed source run: {source}")

    top_level_files = sorted(path for path in source.iterdir() if path.is_file())
    unexpected_directories = sorted(
        path.name for path in source.iterdir() if path.is_dir() and path.name != "records"
    )
    if unexpected_directories:
        raise PackagingError(f"Unexpected source-run directories: {unexpected_directories}")
    if len(top_level_files) != spec.expected_top_level_file_count:
        raise PackagingError(
            f"{spec.study_id} has {len(top_level_files)} top-level files; "
            f"expected {spec.expected_top_level_file_count}."
        )

    artifact_manifest_path = source / "artifact_manifest.json"
    artifact_sha256, artifact_bytes = _signature(artifact_manifest_path)
    if artifact_sha256 != spec.expected_artifact_manifest_sha256:
        raise PackagingError(f"{spec.study_id} artifact-manifest digest changed.")
    sidecar = (source / "artifact_manifest.sha256").read_text(encoding="utf-8").split()
    if sidecar != [artifact_sha256, "artifact_manifest.json"]:
        raise PackagingError(f"{spec.study_id} artifact-manifest sidecar is invalid.")

    full_manifest = _load_json(artifact_manifest_path)
    entries = full_manifest.get("files")
    if not isinstance(entries, list) or full_manifest.get("file_count") != len(entries):
        raise PackagingError(f"{spec.study_id} full artifact manifest is inconsistent.")
    if len(entries) != spec.expected_full_manifest_file_count:
        raise PackagingError(f"{spec.study_id} full artifact count changed.")
    record_entries = [
        entry for entry in entries
        if isinstance(entry, dict) and str(entry.get("path", "")).startswith("records/")
    ]
    if len(record_entries) != spec.expected_omitted_record_file_count:
        raise PackagingError(f"{spec.study_id} omitted record-file count changed.")
    declared_top_level = {
        entry["path"] for entry in entries
        if isinstance(entry, dict) and "/" not in str(entry.get("path", ""))
    }
    actual_manifest_members = {
        path.name for path in top_level_files
        if path.name not in {"artifact_manifest.json", "artifact_manifest.sha256"}
    }
    if declared_top_level != actual_manifest_members:
        raise PackagingError(
            f"{spec.study_id} top-level files disagree with artifact_manifest.json."
        )

    destination.mkdir(parents=True)
    archived_files: list[dict[str, Any]] = []
    archived_bytes = 0
    for source_file in top_level_files:
        target = destination / source_file.name
        shutil.copyfile(source_file, target)
        source_signature = _signature(source_file)
        if _signature(target) != source_signature:
            raise PackagingError(f"Byte-for-byte copy failed for {source_file}.")
        sha256, byte_count = source_signature
        archived_files.append(
            {
                "source_path": f"{spec.source_run}/{source_file.name}",
                "archive_relative_path": source_file.name,
                "sha256": sha256,
                "bytes": byte_count,
            }
        )
        archived_bytes += byte_count

    manifest = {
        "schema_version": "completed-run-compact-archive-v1",
        "archive_type": "completed_run_compact",
        "study_id": spec.study_id,
        "claim_ids": list(spec.claim_ids),
        "clean_source_commit": CLEAN_SOURCE_COMMIT,
        "source_run": spec.source_run,
        "archive_root": spec.archive_root,
        "archived_files": archived_files,
        "archived_file_count": len(archived_files),
        "archived_file_bytes": archived_bytes,
        "full_run_commitment": {
            "artifact_manifest_sha256": artifact_sha256,
            "artifact_manifest_bytes": artifact_bytes,
            "declared_file_count": len(entries),
            "omitted_path_prefix": "records/",
            "omitted_file_count": len(record_entries),
            "omitted_bytes_archived": False,
            "omitted_bytes_recoverable_from_hashes": False,
        },
        "limitation": spec.limitation,
    }
    (destination / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "study_id": spec.study_id,
        "claim_ids": list(spec.claim_ids),
        "archive_root": spec.archive_root,
        "archived_file_count": len(archived_files),
        "archived_file_bytes": archived_bytes,
        "omitted_file_count": len(record_entries),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    reports = [package(spec, root=args.repo_root.resolve()) for spec in SPECS]
    print(json.dumps({"archives": reports}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
