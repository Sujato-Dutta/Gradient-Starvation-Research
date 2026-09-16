import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.aggregate_cifar_matched import ARM_METRICS, METRICS, aggregate


SPEC = json.loads((Path(__file__).resolve().parents[1] /
                   "configs/cifar_matched_preregistration.draft.json").read_text())


def _fixture(tmp_path):
    config = copy.deepcopy(SPEC)
    config["source_commit"] = "1" * 40
    config["environment_sha256"] = "2" * 64
    config["dataset"]["dataset_archive_sha256"] = "3" * 64
    config["training"].update(horizon_updates=2, checkpoint_updates=[1, 2])
    paths = []
    for i, seed in enumerate(config["seed_blocks"]["confirmation"]):
        path = tmp_path / f"seed_{seed}"
        path.mkdir()
        summary = {
            "seed": seed, "weak_only_gate_passed": True,
            "primary_causal_certificate": i % 2 == 0,
            "weak_only_first_hit_right_censored": False,
            "weak_only_first_hit_update": 1,
            **{key: i / 100 for key in METRICS
               if key != "fresh_head_neutral_accuracy_gap_B_minus_W"},
            **{key: 0.5 + i / 100 for key in ARM_METRICS},
            "e4_lite": {"fresh_head_neutral_accuracy_gap_B_minus_W": i / 100},
        }
        summary_path = path / "summary.json"
        summary_path.write_text(json.dumps(summary))
        trace_path = path / "trace.json"
        trace_path.write_text(json.dumps([{"step": step} for step in (0, 1, 2)]))
        checkpoints_dir = path / "checkpoints"
        checkpoints_dir.mkdir()
        checkpoints = []
        for step in (0, 1, 2):
            artifact = checkpoints_dir / f"step_{step:06d}.pt"
            artifact.write_bytes(f"seed={seed},step={step}".encode())
            checkpoints.append({
                "step": step, "path": artifact.relative_to(path).as_posix(),
                "file_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            })
        checkpoint_manifest = path / "checkpoints.json"
        checkpoint_manifest.write_text(json.dumps(checkpoints))
        (path / "metadata.json").write_text(json.dumps({
            "seed": seed, "status": "complete", "config_sha256": "abc",
            "source_commit": config["source_commit"],
            "environment_sha256": config["environment_sha256"],
            "dataset_archive_sha256": config["dataset"]["dataset_archive_sha256"],
            "last_update": 2, "checkpoint_count": 3,
            "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            "checkpoint_manifest_sha256": hashlib.sha256(
                checkpoint_manifest.read_bytes()).hexdigest(),
            "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        }))
        paths.append(path)
    return config, paths


def test_all_eight_blocks_receive_paired_intervals(tmp_path):
    config, paths = _fixture(tmp_path)
    result = aggregate(config, paths, config_sha256="abc")
    assert result["status"] == "complete_eight_block_descriptive_analysis"
    assert result["completed_seed_count"] == 8
    assert result["primary_causal_certificate_count"] == 4
    assert set(result["paired_95_percent_t_intervals"]) == set(METRICS)
    assert set(result["seed_block_95_percent_t_intervals"]) == {
        *METRICS, *ARM_METRICS,
    }
    interval = result["paired_95_percent_t_intervals"]["signed_normalized_deficit_auc"]
    assert interval["mean"] == pytest.approx(0.035)
    assert interval["ci95_low"] < interval["mean"] < interval["ci95_high"]


def test_missing_or_incomplete_blocks_never_get_confirmation_interval(tmp_path):
    config, paths = _fixture(tmp_path)
    missing = aggregate(config, paths[:-1], config_sha256="abc")
    assert missing["status"] == "incomplete_no_confirmatory_interval"
    assert missing["missing_or_incomplete_seeds"] == [33008]
    assert missing["paired_95_percent_t_intervals"] == {}
    metadata_path = paths[-1] / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["status"] = "running_incomplete_if_interrupted"
    metadata_path.write_text(json.dumps(metadata))
    incomplete = aggregate(config, paths, config_sha256="abc")
    assert incomplete["missing_or_incomplete_seeds"] == [33008]
    assert incomplete["paired_95_percent_t_intervals"] == {}


def test_duplicate_and_tampered_runs_fail_closed(tmp_path):
    config, paths = _fixture(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        aggregate(config, [paths[0], paths[0]], config_sha256="abc")
    (paths[0] / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="tampered"):
        aggregate(config, paths, config_sha256="abc")


def test_tampered_checkpoint_fails_closed(tmp_path):
    config, paths = _fixture(tmp_path)
    (paths[0] / "checkpoints" / "step_000001.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint missing/tampered"):
        aggregate(config, paths, config_sha256="abc")
