#!/usr/bin/env python3
"""Build or verify a deterministic, fail-closed anonymous CIFAR supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


FORBIDDEN = (
    b"se23uari141",
    b"prithviraj",
    b"/users/",
    b"/dgxa_home/",
    b"10.59.",
)
FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
TRACKED_FILES = (
    "configs/cifar_matched_preregistration.frozen.json",
    "research_scope/cifar_dgx_environment_v2.json",
    "research_scope/cifar_confirmation_audit_metrics_20260917.json",
    "research_scope/cifar_confirmation_result_audit_20260917.md",
    "scripts/aggregate_cifar_matched.py",
    "scripts/audit_cifar_confirmation_results.py",
    "scripts/check_cifar_preregistration.py",
    "scripts/freeze_cifar_preregistration.py",
    "scripts/plot_cifar_confirmation.py",
    "scripts/preflight_cifar_confirmation.py",
    "scripts/run_cifar_matched.py",
    "src/gradient_starvation/cifar_matched.py",
    "src/gradient_starvation/cifar_protocol.py",
    "tests/test_cifar_aggregate.py",
    "tests/test_cifar_freeze.py",
    "tests/test_cifar_matched.py",
    "tests/test_cifar_protocol.py",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _assert_anonymous(path: str, data: bytes) -> None:
    lowered = data.lower()
    hits = [token.decode("ascii") for token in FORBIDDEN if token in lowered]
    if hits:
        raise ValueError(f"identifying token(s) {hits} found in {path}")


def collect_files(repo_root: Path, evidence_root: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for relative in TRACKED_FILES:
        data = (repo_root / relative).read_bytes()
        _assert_anonymous(relative, data)
        files[relative] = data

    for source in sorted(evidence_root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(evidence_root)
        if relative.parts[:2] == ("results", "slurm"):
            if source.suffix.lower() != ".err" or source.stat().st_size != 0:
                continue
        elif source.suffix.lower() not in {".json", ".csv", ".md", ".txt"}:
            continue
        archive_path = (Path("evidence") / relative).as_posix()
        data = source.read_bytes()
        _assert_anonymous(archive_path, data)
        files[archive_path] = data
    if not any(path.endswith("paired_summary.json") for path in files):
        raise ValueError("paired_summary.json is missing from the evidence root")
    return files


def build(repo_root: Path, evidence_root: Path, output: Path) -> dict[str, object]:
    files = collect_files(repo_root, evidence_root)
    manifest: dict[str, object] = {
        "schema_version": "anonymous-cifar-supplement-v1",
        "source_commit": "0726aea12947a30eb60ae32fc9c1eb0a7a122660",
        "scope": "CIFAR-10 matched confirmation; model checkpoints and identifying Slurm stdout excluded; empty stderr sentinels retained",
        "file_count_excluding_manifest": len(files),
        "files": [
            {"path": path, "bytes": len(data), "sha256": _sha256(data)}
            for path, data in sorted(files.items())
        ],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    _assert_anonymous("MANIFEST.json", manifest_bytes)

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, data in [*sorted(files.items()), ("MANIFEST.json", manifest_bytes)]:
            info = zipfile.ZipInfo(path, FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    verify(output)
    return {
        "output": str(output),
        "sha256": _sha256(output.read_bytes()),
        "file_count": len(files) + 1,
    }


def verify(archive_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "MANIFEST.json" not in names:
            raise ValueError("archive has duplicate paths or no MANIFEST.json")
        manifest = json.loads(archive.read("MANIFEST.json"))
        expected = {entry["path"]: entry for entry in manifest["files"]}
        actual = set(names) - {"MANIFEST.json"}
        if actual != set(expected):
            raise ValueError("archive members do not match MANIFEST.json")
        for path in names:
            data = archive.read(path)
            _assert_anonymous(path, data)
            if path != "MANIFEST.json":
                entry = expected[path]
                if len(data) != entry["bytes"] or _sha256(data) != entry["sha256"]:
                    raise ValueError(f"size or digest mismatch for {path}")
        slurm_members = [path for path in names if path.startswith("evidence/results/slurm/")]
        if any(not path.endswith(".err") or archive.read(path) for path in slurm_members):
            raise ValueError("Only empty Slurm stderr sentinels are allowed")
    return {"sha256": _sha256(archive_path.read_bytes()), "file_count": len(names)}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=root / "results" / "cifar_confirmation_compact_dde8d35",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "paper" / "artifacts" / "cifar-confirmation-anonymous-v1.zip",
    )
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    result = verify(args.verify) if args.verify else build(args.repo_root, args.evidence_root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
