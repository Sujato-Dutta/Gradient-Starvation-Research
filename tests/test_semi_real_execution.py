"""Outcome-free and toy-only tests for the sealed semi-real execution layer."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import gradient_starvation.semi_real_study as study
from gradient_starvation.semi_real_study import (
    SemiRealStudyError,
    deterministic_class_order,
    load_execution_contract,
    run_engineering_smoke,
    run_scientific_study,
    summarize_paired_trajectory,
    two_way_pigeonhole_bootstrap,
    validate_execution_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "semi_real_execution.yaml"


def _contract() -> dict:
    return load_execution_contract(CONTRACT_PATH, repository_root=ROOT)


@pytest.fixture
def semi_real_workspace():
    workspace_root = ROOT / study.SEMI_REAL_WORKSPACE_RELATIVE_ROOT
    workspace_root.mkdir(parents=True, exist_ok=True)
    sandbox = Path(tempfile.mkdtemp(prefix="pytest-", dir=workspace_root))
    try:
        yield sandbox
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def test_execution_contract_is_exhaustive_and_historical_v1_remains_blocked():
    contract = _contract()
    report = validate_execution_contract(contract, repository_root=ROOT)
    historical = json.loads((ROOT / contract["historical_contract"]["path"]).read_text())

    assert report["valid"] is True
    assert report["execution_authorized"] is False
    assert report["record_count"] == 64
    assert historical["execution_authorized"] is False
    assert historical["status"] == "design_frozen_execution_blocked"
    assert contract["contract_revision"] == 2
    assert contract["local_workspace"] == {
        "root": "data/semi_real_generated_cue_v1",
        "required_path_roles": [
            "data",
            "preflight",
            "authorization_seal",
            "engineering_smoke",
            "scientific_run",
        ],
        "strict_descendants_only": True,
        "symlink_components_forbidden": True,
        "artifact_tree_symlinks_forbidden": True,
        "git_ignored_required": True,
        "tracked_workspace_paths_forbidden": True,
        "git_cleanliness_rationale": "ignored local artifacts do not affect git cleanliness",
    }

    changed = copy.deepcopy(contract)
    changed["local_workspace"]["root"] = "data/a-different-workspace"
    with pytest.raises(SemiRealStudyError, match="fully validated"):
        validate_execution_contract(changed, repository_root=ROOT)


def test_workspace_guard_accepts_only_ignored_non_symlink_descendants(
    semi_real_workspace,
):
    accepted = semi_real_workspace / "data"
    assert (
        study._require_workspace_subpath(
            accepted,
            ROOT,
            path_role="Test data",
        )
        == accepted
    )

    workspace_root = ROOT / study.SEMI_REAL_WORKSPACE_RELATIVE_ROOT
    for forbidden in (workspace_root, ROOT / "data" / "another-study" / "output"):
        with pytest.raises(SemiRealStudyError, match="strict descendant"):
            study._require_workspace_subpath(forbidden, ROOT, path_role="Test output")

    target = semi_real_workspace / "target"
    target.mkdir()
    linked_parent = semi_real_workspace / "linked-parent"
    linked_parent.symlink_to(target, target_is_directory=True)
    broken_parent = semi_real_workspace / "broken-parent"
    broken_parent.symlink_to(semi_real_workspace / "missing-target", target_is_directory=True)
    for forbidden in (linked_parent, linked_parent / "output", broken_parent / "output"):
        with pytest.raises(SemiRealStudyError, match="Symlinked"):
            study._require_workspace_subpath(forbidden, ROOT, path_role="Test output")


def test_git_ignore_guard_rejects_tracked_or_selectively_unignored_paths(
    semi_real_workspace,
    monkeypatch,
):
    candidate = semi_real_workspace / "run"
    tracked = study.subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="data/semi_real_generated_cue_v1/tracked-artifact\n",
        stderr="",
    )
    monkeypatch.setattr(study.subprocess, "run", lambda *args, **kwargs: tracked)
    with pytest.raises(SemiRealStudyError, match="no tracked paths"):
        study._require_workspace_is_git_ignored(ROOT, candidate=candidate)

    responses = iter(
        [
            study.subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            ),
            study.subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr=""
            ),
        ]
    )
    monkeypatch.setattr(
        study.subprocess,
        "run",
        lambda *args, **kwargs: next(responses),
    )
    with pytest.raises(SemiRealStudyError, match="must remain Git-ignored"):
        study._require_workspace_is_git_ignored(ROOT, candidate=candidate)


def test_pipeline_entry_points_reject_other_ignored_paths_before_processing(
    semi_real_workspace,
    monkeypatch,
):
    outside_workspace = ROOT / "data" / "another-study"

    with pytest.raises(SemiRealStudyError, match="strict descendant"):
        study.build_preflight(
            CONTRACT_PATH,
            data_root=outside_workspace / "data",
            output_directory=semi_real_workspace / "preflight",
            repository_root=ROOT,
        )
    with pytest.raises(SemiRealStudyError, match="strict descendant"):
        study.verify_preflight(
            CONTRACT_PATH,
            outside_workspace / "preflight.manifest.json",
            data_root=semi_real_workspace / "data",
            repository_root=ROOT,
        )

    def forbidden_determinism(*args, **kwargs):  # pragma: no cover - assertion helper
        pytest.fail("model setup was reached before workspace validation")

    monkeypatch.setattr(study, "_configure_determinism", forbidden_determinism)
    with pytest.raises(SemiRealStudyError, match="strict descendant"):
        run_scientific_study(
            CONTRACT_PATH,
            semi_real_workspace / "missing-preflight.json",
            semi_real_workspace / "missing-seal.json",
            data_root=semi_real_workspace / "missing-data",
            output_directory=outside_workspace / "run",
            repository_root=ROOT,
        )
    with pytest.raises(SemiRealStudyError, match="strict descendant"):
        run_engineering_smoke(
            CONTRACT_PATH,
            output_directory=outside_workspace / "smoke",
            repository_root=ROOT,
        )
    with pytest.raises(SemiRealStudyError, match="strict descendant"):
        study.validate_run_artifacts(outside_workspace / "run", repository_root=ROOT)


def test_scientific_resume_rejects_internal_symlink_before_model_setup(
    semi_real_workspace,
    tmp_path,
    monkeypatch,
):
    run_directory = semi_real_workspace / "run"
    run_directory.mkdir()
    external_records = tmp_path / "external-records"
    external_records.mkdir()
    (run_directory / "records").symlink_to(external_records, target_is_directory=True)

    def forbidden_determinism(*args, **kwargs):  # pragma: no cover - assertion helper
        pytest.fail("model setup was reached before resume-tree validation")

    monkeypatch.setattr(study, "_configure_determinism", forbidden_determinism)
    with pytest.raises(SemiRealStudyError, match="Symlinked"):
        run_scientific_study(
            CONTRACT_PATH,
            semi_real_workspace / "missing-preflight.json",
            semi_real_workspace / "missing-seal.json",
            data_root=semi_real_workspace / "missing-data",
            output_directory=run_directory,
            repository_root=ROOT,
        )
    assert list(external_records.iterdir()) == []


def test_source_manifest_covers_the_full_local_package_import_closure():
    contract = _contract()
    manifest = study.build_source_manifest(contract, ROOT)
    listed = {item["path"] for item in manifest["files"]}
    expected_package_files = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src" / "gradient_starvation").rglob("*.py")
    }
    assert expected_package_files <= listed
    assert "scripts/run_semi_real_study.py" in listed
    assert "scripts/check_semi_real_execution.py" in listed


def test_sha256_class_order_is_deterministic_and_seeded_without_torchvision():
    candidates = range(40)
    first = deterministic_class_order(
        study_id="toy-study",
        dataset="generated-toy",
        split="train",
        data_seed=900001,
        source_class=0,
        candidate_indices=candidates,
    )
    repeat = deterministic_class_order(
        study_id="toy-study",
        dataset="generated-toy",
        split="train",
        data_seed=900001,
        source_class=0,
        candidate_indices=reversed(range(40)),
    )
    different = deterministic_class_order(
        study_id="toy-study",
        dataset="generated-toy",
        split="train",
        data_seed=900002,
        source_class=0,
        candidate_indices=candidates,
    )
    assert first == repeat
    assert first != different
    assert sorted(first) == list(range(40))


def _write_idx_split(
    raw_directory: Path,
    split: str,
    images: np.ndarray,
    labels: np.ndarray,
) -> None:
    image_name, label_name = (
        ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
        if split == "train"
        else ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
    )
    raw_directory.mkdir(parents=True, exist_ok=True)
    image_header = struct.pack(">IIII", 2051, len(images), images.shape[1], images.shape[2])
    label_header = struct.pack(">II", 2049, len(labels))
    (raw_directory / image_name).write_bytes(image_header + images.tobytes())
    (raw_directory / label_name).write_bytes(label_header + labels.tobytes())


def _toy_official_splits(tmp_path: Path) -> tuple[dict, dict]:
    contract = copy.deepcopy(_contract())
    contract["seed_design"]["data_seeds"] = [900011, 900012, 900013, 900014]
    for dataset_index, specification in enumerate(contract["datasets"]):
        specification.update(
            {
                "train_per_class": 2,
                "probe_per_class": 1,
                "evaluation_per_class": 1,
                "raw_image_shape": [2, 2],
                "official_download_mirrors": ["https://example.invalid/"],
                "official_archives": [
                    {
                        "filename": archive["filename"],
                        "md5": hashlib.md5(
                            f"{specification['name']}:{archive['filename']}".encode()
                        ).hexdigest(),
                    }
                    for archive in specification["official_archives"]
                ],
            }
        )
        labels = np.asarray(
            [specification["negative_class"]] * 2
            + [specification["positive_class"]] * 2,
            dtype=np.uint8,
        )
        images = np.arange(16, dtype=np.uint8).reshape(4, 2, 2) + dataset_index * 32
        raw_directory = tmp_path / specification["torchvision_directory"] / "raw"
        _write_idx_split(raw_directory, "train", images, labels)
        _write_idx_split(raw_directory, "test", images + 16, labels)
        processed = tmp_path / specification["torchvision_directory"] / "processed"
        processed.mkdir(parents=True)
        (processed / "training.pt").write_bytes(b"poisoned-cache-must-never-be-read")
    loaded = study._load_official_datasets(contract, tmp_path, download=False)
    return contract, loaded


def test_offline_loader_decodes_hashed_idx_and_never_reads_processed_cache(tmp_path):
    contract, loaded = _toy_official_splits(tmp_path)

    for dataset_index, specification in enumerate(contract["datasets"]):
        split = loaded[specification["name"]]["train"]
        expected = torch.arange(16, dtype=torch.uint8).reshape(4, 2, 2) + dataset_index * 32
        assert torch.equal(split.data, expected)
        assert split.targets.tolist() == [
            specification["negative_class"],
            specification["negative_class"],
            specification["positive_class"],
            specification["positive_class"],
        ]


def test_offline_raw_manifest_requires_frozen_official_archive_checksums(tmp_path):
    contract, _ = _toy_official_splits(tmp_path)
    for specification in contract["datasets"]:
        raw_directory = tmp_path / specification["torchvision_directory"] / "raw"
        for archive in specification["official_archives"]:
            payload = f"{specification['name']}:{archive['filename']}".encode()
            (raw_directory / archive["filename"]).write_bytes(payload)
    manifest = study.build_raw_data_manifest(contract, tmp_path)
    assert all(
        all(item["verified"] is True for item in dataset["official_archives"])
        for dataset in manifest["datasets"]
    )

    first = contract["datasets"][0]
    bad_archive = tmp_path / first["torchvision_directory"] / "raw" / first["official_archives"][0]["filename"]
    bad_archive.write_bytes(b"not-the-official-archive")
    with pytest.raises(SemiRealStudyError, match="checksum failed"):
        study.build_raw_data_manifest(contract, tmp_path)


def test_explicit_acquisition_uses_resources_not_legacy_dataset_cache(tmp_path, monkeypatch):
    contract, _ = _toy_official_splits(tmp_path)
    calls: list[tuple[str, str, str]] = []

    def fake_download(url, download_root, *, filename, md5):
        calls.append((url, filename, md5))
        dataset_name = "fashion_mnist" if "FashionMNIST" in download_root else "mnist"
        payload = f"{dataset_name}:{filename}".encode()
        assert hashlib.md5(payload).hexdigest() == md5
        (Path(download_root) / filename).write_bytes(payload)

    import torchvision.datasets.utils as dataset_utils

    monkeypatch.setattr(dataset_utils, "download_and_extract_archive", fake_download)
    loaded = study._load_official_datasets(contract, tmp_path, download=True)

    assert len(calls) == 8
    assert set(loaded) == {"mnist", "fashion_mnist"}
    assert all("processed" not in url for url, _, _ in calls)
    for specification in contract["datasets"]:
        poisoned = (
            tmp_path
            / specification["torchvision_directory"]
            / "processed"
            / "training.pt"
        )
        assert poisoned.read_bytes() == b"poisoned-cache-must-never-be-read"


def test_selection_manifest_binds_tensors_and_discloses_all_role_pair_overlaps(tmp_path):
    contract, loaded = _toy_official_splits(tmp_path)
    manifest = study.build_selection_manifest(
        contract, loaded, execution_contract_sha256="0" * 64
    )
    report = study.validate_selection_manifest(contract, manifest)

    assert report == {"valid": True, "selection_count": 8}
    assert len(manifest["cross_seed_overlap_disclosure"]) == 12
    assert all(
        len(item["role_pair_overlap_counts"]) == 9
        for item in manifest["cross_seed_overlap_disclosure"]
    )
    assert all(
        role["selected_raw_uint8_size"] > 0
        and len(role["selected_raw_uint8_sha256"]) == 64
        for selection in manifest["selections"]
        for role in selection["roles"]
    )


def _toy_trajectory_rows() -> list[dict]:
    times = [0.0, 1.0, 2.0, 3.0]
    weak_gain = [0.0, 0.2, 0.6, 0.9]
    response_gap = [0.0, 0.4, 0.1, -0.2]
    both_gain = [weak + gap for weak, gap in zip(weak_gain, response_gap)]
    both_drift = [1.0, 0.5, -0.5, -1.0]
    weak_drift = [0.0, 0.0, 0.0, 0.0]
    rows: list[dict] = []
    for condition, gains, drifts in (
        ("both", both_gain, both_drift),
        ("weak_only", weak_gain, weak_drift),
    ):
        for step, (tau, gain, drift) in enumerate(zip(times, gains, drifts)):
            rows.append(
                {
                    "condition": condition,
                    "step": step,
                    "tau": tau,
                    "core_response": gain,
                    "core_gain": gain,
                    "cue_response": 0.0,
                    "cue_gain": 0.0,
                    "core_drift": drift,
                }
            )
    return rows


def _toy_trajectory_contract() -> dict:
    contract = copy.deepcopy(_contract())
    contract["optimization"].update(
        {
            "learning_rate": 1.0,
            "steps": 3,
            "log_every": 1,
            "logged_step_start": 0,
            "logged_step_end": 3,
            "logged_point_count": 4,
        }
    )
    return contract


def test_endpoint_summary_uses_frozen_signs_tail_gate_and_interpolation():
    contract = _toy_trajectory_contract()
    summary = summarize_paired_trajectory(_toy_trajectory_rows(), contract)

    assert summary["weak_only_learnable"] is True
    assert summary["weak_only_first_hit_tau"] == pytest.approx(1.75)
    assert summary["outcome_suppression"] is True
    assert summary["causal_certificate"] is True
    assert summary["first_drift_crossing_tau"] == pytest.approx(1.5)
    assert summary["first_response_crossing_tau"] == pytest.approx(2.0 + 1.0 / 3.0)
    assert summary["positive_drift_area"] > 0
    assert summary["negative_drift_area"] > 0
    assert summary["endpoint_signs"]["weak_auc_integrand"] == "weak_only_minus_both"
    assert summary["endpoint_signs"]["drift_gap"] == "both_minus_weak_only"

    off_grid = copy.deepcopy(_toy_trajectory_rows())
    off_grid[-1]["tau"] = 4.0
    with pytest.raises(SemiRealStudyError, match="exactly steps"):
        summarize_paired_trajectory(off_grid, contract)


def test_two_way_bootstrap_is_deterministic_and_resamples_both_axes():
    matrix = np.arange(32, dtype=float).reshape(4, 8)
    first = two_way_pigeonhole_bootstrap(matrix, replicates=200, seed=900003)
    second = two_way_pigeonhole_bootstrap(matrix, replicates=200, seed=900003)

    assert first == second
    assert first["point_estimate"] == pytest.approx(matrix.mean())
    assert first["ci95_low"] < first["point_estimate"] < first["ci95_high"]
    with pytest.raises(SemiRealStudyError, match="non-finite"):
        two_way_pigeonhole_bootstrap(
            np.asarray([[1.0, np.nan]]), replicates=10, seed=900003
        )


def test_seal_validator_rejects_identity_kind_and_git_drift():
    contract = _contract()
    bindings = {
        name: (
            study.expected_record_ids(contract)
            if name == "expected_record_ids"
            else ({"mnist": "a", "fashion_mnist": "b"}
                  if name == "raw_dataset_aggregate_sha256"
                  else "0" * 64)
        )
        for name in contract["authorization"]["required_bindings"]
    }
    seal = {
        "schema_version": contract["authorization"]["seal_schema_version"],
        "study_id": contract["study_id"],
        "authorization_kind": "explicit_human_scientific_execution",
        "bindings": bindings,
        "expected_git_state": {"clean": True, "commit": "a" * 40},
        "untouched_seed_attestation": {
            "attested": True,
            "text": contract["authorization"]["untouched_seed_attestation_text"],
        },
        "authorization": {
            "authorized": True,
            "text": contract["authorization"]["explicit_authorization_text"],
            "timestamp_utc": "2026-08-29T00:00:00+00:00",
        },
    }
    assert study._validate_seal_document(contract, seal, bindings)["clean"] is True

    for path, bad_value in (
        (("study_id",), "different-study"),
        (("authorization_kind",), "different-kind"),
        (("expected_git_state", "clean"), False),
    ):
        changed = copy.deepcopy(seal)
        if len(path) == 1:
            changed[path[0]] = bad_value
        else:
            changed[path[0]][path[1]] = bad_value
        with pytest.raises(SemiRealStudyError):
            study._validate_seal_document(contract, changed, bindings)


def test_scientific_run_fails_before_training_or_output_without_a_seal(
    semi_real_workspace,
    monkeypatch,
):
    def forbidden_training(*args, **kwargs):  # pragma: no cover - assertion helper
        pytest.fail("training was reached before authorization")

    monkeypatch.setattr(study, "_train_pair", forbidden_training)
    output = semi_real_workspace / "must-not-exist"
    with pytest.raises(SemiRealStudyError):
        run_scientific_study(
            CONTRACT_PATH,
            semi_real_workspace / "missing-preflight.json",
            semi_real_workspace / "missing-seal.json",
            data_root=semi_real_workspace / "missing-data",
            output_directory=output,
            repository_root=ROOT,
        )
    assert not output.exists()


def test_engineering_smoke_is_toy_only_and_acceptance_ineligible(semi_real_workspace):
    output = run_engineering_smoke(
        CONTRACT_PATH,
        output_directory=semi_real_workspace / "smoke",
        repository_root=ROOT,
    )
    summary = json.loads((output / "smoke.summary.json").read_text())
    contract = _contract()
    scientific = set(contract["seed_design"]["data_seeds"]) | set(
        contract["seed_design"]["model_seeds"]
    )

    assert summary["scope"] == "engineering_smoke_non_scientific_toy_data"
    assert summary["scientific_record"] is False
    assert summary["acceptance_eligible"] is False
    assert summary["scientific_seeds_used"] is False
    assert contract["engineering_smoke"]["data_seed"] not in scientific
    assert contract["engineering_smoke"]["model_seed"] not in scientific
    assert summary["gsi5_status"] == "missing_secondary_not_defined_for_evaluation_matrix"
    assert (output / "artifact_manifest.json").is_file()
    assert (output / "artifact_manifest.sha256").is_file()
