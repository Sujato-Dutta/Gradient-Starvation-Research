from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts.package_anonymous_cifar_supplement import build, verify


def _fixture(root: Path, *, identifying: bool = False) -> tuple[Path, Path]:
    repo = root / "repo"
    evidence = root / "evidence"
    for relative in (
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
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("safe\n", encoding="utf-8")
    summary = evidence / "results" / "aggregate" / "paired_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        '{"path": "/dgxa_home/user/run"}\n' if identifying else '{"status": "complete"}\n',
        encoding="utf-8",
    )
    slurm = evidence / "results" / "slurm" / "job.err"
    slurm.parent.mkdir(parents=True)
    slurm.write_bytes(b"")
    (slurm.parent / "job.out").write_text(
        "/dgxa_home/user is deliberately excluded\n", encoding="utf-8"
    )
    return repo, evidence


def test_anonymous_package_is_deterministic_and_excludes_slurm(tmp_path: Path):
    repo, evidence = _fixture(tmp_path)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    assert build(repo, evidence, first)["sha256"] == build(repo, evidence, second)["sha256"]
    assert verify(first)["file_count"] > 1
    with zipfile.ZipFile(first) as archive:
        assert "evidence/results/slurm/job.err" in archive.namelist()
        assert "evidence/results/slurm/job.out" not in archive.namelist()


def test_anonymous_package_fails_closed_on_identifying_result(tmp_path: Path):
    repo, evidence = _fixture(tmp_path, identifying=True)
    with pytest.raises(ValueError, match="identifying token"):
        build(repo, evidence, tmp_path / "bad.zip")
