from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts.check_environment_lock import (
    EnvironmentLockError,
    verify_environment_lock,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements-lock-py312.txt"
DIRECT = ROOT / "requirements.txt"
PYTHON_VERSION = ROOT / ".python-version"


def test_validated_python_environment_is_exact_and_content_addressed():
    readiness = json.loads(
        (ROOT / "research_scope/submission_readiness.yaml").read_text(encoding="utf-8")
    )
    report = verify_environment_lock(
        LOCK,
        DIRECT,
        PYTHON_VERSION,
        expected_sha256=readiness["validated_environment"]["lock_sha256"],
    )

    assert report.python_version == "3.12.4"
    assert report.package_count == 32
    assert report.direct_requirement_count == 10


def test_environment_lock_rejects_nonexact_requirement(tmp_path):
    lock = tmp_path / "lock.txt"
    shutil.copy2(LOCK, lock)
    lock.write_text(
        lock.read_text(encoding="utf-8").replace("torch==2.13.0", "torch>=2.13.0"),
        encoding="utf-8",
    )

    with pytest.raises(EnvironmentLockError, match="not an exact"):
        verify_environment_lock(lock, DIRECT, PYTHON_VERSION)


def test_environment_lock_rejects_normalized_duplicate(tmp_path):
    lock = tmp_path / "lock.txt"
    shutil.copy2(LOCK, lock)
    with lock.open("a", encoding="utf-8") as handle:
        handle.write("typing-extensions==4.16.0\n")

    with pytest.raises(EnvironmentLockError, match="Duplicate normalized"):
        verify_environment_lock(lock, DIRECT, PYTHON_VERSION)


def test_environment_lock_rejects_missing_direct_dependency(tmp_path):
    lock = tmp_path / "lock.txt"
    shutil.copy2(LOCK, lock)
    lock.write_text(
        lock.read_text(encoding="utf-8").replace("torch==2.13.0\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(EnvironmentLockError, match="omits direct requirements"):
        verify_environment_lock(lock, DIRECT, PYTHON_VERSION)


def test_environment_lock_rejects_readiness_digest_mismatch():
    with pytest.raises(EnvironmentLockError, match="digest does not match"):
        verify_environment_lock(
            LOCK,
            DIRECT,
            PYTHON_VERSION,
            expected_sha256="0" * 64,
        )
