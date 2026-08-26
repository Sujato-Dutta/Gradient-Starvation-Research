#!/usr/bin/env python3
"""Validate the exact Python 3.12 environment lock without installing packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


EXACT_REQUIREMENT = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)==(?P<version>[A-Za-z0-9][A-Za-z0-9_.+!-]*)"
)
DIRECT_REQUIREMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
CANONICAL_PYTHON_VERSION = "3.12.4"
CANONICAL_LOCK_PATH = "requirements-lock-py312.txt"
CANONICAL_DIRECT_REQUIREMENTS_PATH = "requirements.txt"
CANONICAL_PYTHON_VERSION_PATH = ".python-version"


class EnvironmentLockError(ValueError):
    """Raised when the validated environment lock is incomplete or inconsistent."""


@dataclass(frozen=True)
class EnvironmentLockReport:
    python_version: str
    package_count: int
    lock_sha256: str
    direct_requirement_count: int


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _meaningful_lines(path: Path) -> list[str]:
    try:
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    except (OSError, UnicodeError) as error:
        raise EnvironmentLockError(f"Cannot read {path}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise EnvironmentLockError(f"Cannot hash {path}: {error}") from error
    return digest.hexdigest()


def verify_environment_lock(
    lock_path: Path,
    direct_requirements_path: Path,
    python_version_path: Path,
    *,
    expected_sha256: str | None = None,
) -> EnvironmentLockReport:
    try:
        python_version = python_version_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as error:
        raise EnvironmentLockError(
            f"Cannot read Python version pin {python_version_path}: {error}"
        ) from error
    if python_version != CANONICAL_PYTHON_VERSION:
        raise EnvironmentLockError(
            f"Python version must be exactly {CANONICAL_PYTHON_VERSION}, got {python_version!r}."
        )

    packages: dict[str, tuple[str, str]] = {}
    for line in _meaningful_lines(lock_path):
        match = EXACT_REQUIREMENT.fullmatch(line)
        if match is None:
            raise EnvironmentLockError(
                f"Lock entry is not an exact name==version pin: {line!r}."
            )
        original_name = match.group("name")
        canonical_name = _canonical_name(original_name)
        if canonical_name in packages:
            raise EnvironmentLockError(f"Duplicate normalized lock package: {original_name!r}.")
        packages[canonical_name] = (original_name, match.group("version"))

    direct_names: set[str] = set()
    for line in _meaningful_lines(direct_requirements_path):
        if DIRECT_REQUIREMENT.fullmatch(line) is None:
            raise EnvironmentLockError(
                f"Direct requirements must be bare package names for this check: {line!r}."
            )
        canonical_name = _canonical_name(line)
        if canonical_name in direct_names:
            raise EnvironmentLockError(f"Duplicate direct requirement: {line!r}.")
        direct_names.add(canonical_name)
    missing = sorted(direct_names - set(packages))
    if missing:
        raise EnvironmentLockError(
            f"Exact lock omits direct requirements: {missing}."
        )
    for tool in ("pip", "setuptools"):
        if tool not in packages:
            raise EnvironmentLockError(f"Exact lock omits installer tool {tool!r}.")

    lock_sha256 = _sha256(lock_path)
    if expected_sha256 is not None and lock_sha256 != expected_sha256:
        raise EnvironmentLockError(
            "Environment lock digest does not match submission_readiness.yaml: "
            f"expected {expected_sha256}, got {lock_sha256}."
        )
    return EnvironmentLockReport(
        python_version=python_version,
        package_count=len(packages),
        lock_sha256=lock_sha256,
        direct_requirement_count=len(direct_names),
    )


def _read_expected_metadata(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EnvironmentLockError(f"Cannot parse readiness metadata {path}: {error}") from error
    metadata = payload.get("validated_environment") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        raise EnvironmentLockError("submission_readiness.yaml omits validated_environment.")
    expected = {
        "python_version": CANONICAL_PYTHON_VERSION,
        "lock_path": CANONICAL_LOCK_PATH,
        "direct_requirements_path": CANONICAL_DIRECT_REQUIREMENTS_PATH,
        "python_version_path": CANONICAL_PYTHON_VERSION_PATH,
        "waterbirds_extras_included": False,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise EnvironmentLockError(
                f"validated_environment.{field} must be {value!r}."
            )
    digest = metadata.get("lock_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise EnvironmentLockError("validated_environment.lock_sha256 is malformed.")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repo_root.resolve()
    try:
        metadata = _read_expected_metadata(
            root / "research_scope" / "submission_readiness.yaml"
        )
        report = verify_environment_lock(
            root / CANONICAL_LOCK_PATH,
            root / CANONICAL_DIRECT_REQUIREMENTS_PATH,
            root / CANONICAL_PYTHON_VERSION_PATH,
            expected_sha256=metadata["lock_sha256"],
        )
    except EnvironmentLockError as error:
        print(f"environment lock validation failed: {error}", file=sys.stderr)
        return 1
    print(
        "environment lock valid: "
        f"python={report.python_version}, packages={report.package_count}, "
        f"direct_requirements={report.direct_requirement_count}, "
        f"sha256={report.lock_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
