import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradient_starvation.cifar_matched import (
    CIFAREvaluationDataset, PairedCIFARDataset, evaluate_behavior,
    split_for_seed, summarize_response, train_paired, weights_sha256,
)
from gradient_starvation.cifar_probe import fit_balanced_linear_probe
from gradient_starvation.cifar_protocol import hadamard_codes
from scripts.run_cifar_calibration import train_pilot
from scripts.run_cifar_matched import preflight, run
from scripts.check_cifar_preregistration import validate_protocol
from scripts.record_cifar_environment import environment_record


SPEC = json.loads((Path(__file__).resolve().parents[1] /
                   "configs/cifar_matched_preregistration.draft.json").read_text())


class TinyCIFAR:
    def __init__(self):
        self.targets = [label for label in range(10) for _ in range(5)]
        self.data = np.stack([
            np.full((32, 32, 3), (label * 21 + index) % 256, dtype=np.uint8)
            for label in range(10) for index in range(5)
        ])

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def tiny_spec(amplitude=0.5):
    config = copy.deepcopy(SPEC)
    config["dataset"].update(train_per_class=2, validation_per_class=1,
                             diagnostic_per_class=2, diagnostic_probe_fit_per_class=1,
                             diagnostic_evaluation_per_class=1, cue_amplitude=amplitude)
    config["model"].update(channels=[8, 8, 8, 8])
    config["training"].update(batch_size=10, learning_rate=0.01,
                              weight_decay=0.0, horizon_updates=2,
                              checkpoint_updates=[1, 2])
    config["response"].update(beta=0.5, scale_S=0.75)
    config["e4_lite"] = {"probe_l2_grid": [0.0], "max_iter": 5}
    return config


def test_pair_exact_ablation_and_common_augmentation():
    base = TinyCIFAR()
    paired = PairedCIFARDataset(base, np.array([0, 5, 10]), block_seed=33001,
                                training=True, reliability=0.9, amplitude=0.5)
    paired.set_epoch(3)
    x_b, x_w, label, image_id = paired[1]
    assert (label, image_id) == (1, 5)
    assert torch.equal(x_b[:, :32, :], x_w[:, :32, :])
    assert torch.count_nonzero(x_w[:, 32:, :]) == 0
    assert torch.count_nonzero(x_b[:, 32:, :]) == 3 * 16
    assert torch.equal(x_b, paired[1][0])


def test_evaluation_views_use_common_core_and_all_nine_wrong_codes():
    base = TinyCIFAR()
    common = dict(base=base, indices=np.array([0]), block_seed=33001, amplitude=0.5)
    neutral = CIFAREvaluationDataset(mode="neutral", **common)
    consistent = CIFAREvaluationDataset(mode="consistent", **common)
    random = CIFAREvaluationDataset(mode="random", **common)
    conflict = [CIFAREvaluationDataset(mode="conflict", wrong_index=i, **common)
                for i in range(9)]
    x_n, _ = neutral[0]
    assert torch.count_nonzero(x_n[:, 32:]) == 0
    all_views = [consistent, random, *conflict]
    assert all(torch.equal(view[0][0][:, :32], x_n[:, :32]) for view in all_views)
    assert len({view[0][0][:, 32:].numpy().tobytes() for view in conflict}) == 9
    assert all(not torch.equal(view[0][0][:, 32:], consistent[0][0][:, 32:])
               for view in conflict)
    assert hadamard_codes().shape == (10, 4, 4)
    with pytest.raises(ValueError, match="wrong_index"):
        CIFAREvaluationDataset(mode="conflict", wrong_index=9, **common)


def test_checkpoint_grid_certificate_and_gate_are_distinct():
    trace = [
        {"step": 0, "both": {"brier_skill": 0.0},
         "weak_only": {"brier_skill": 0.0, "accuracy": 0.1}},
        {"step": 1, "both": {"brier_skill": 0.5},
         "weak_only": {"brier_skill": 0.5, "accuracy": 0.7}},
        {"step": 2, "both": {"brier_skill": 0.4},
         "weak_only": {"brier_skill": 0.7, "accuracy": 0.9}},
    ]
    kwargs = dict(beta=0.5, scale=0.75, accuracy_gate=0.8, scale_gate=0.25,
                  gap_tolerance=1e-6, material_fraction=0.1)
    result = summarize_response(trace, **kwargs)
    assert result["weak_only_gate_passed"]
    assert result["weak_only_first_hit_update"] == 1
    assert result["gap_at_weak_only_first_hit"] == 0
    assert not result["primary_causal_certificate"]
    assert result["secondary_any_time_certificate"]
    assert result["minimum_gap"] == pytest.approx(-0.3)
    assert result["signed_normalized_deficit_auc"] == pytest.approx(0.1)
    crossing = [
        {"step": 0, "both": {"brier_skill": 0.5},
         "weak_only": {"brier_skill": 0.4, "accuracy": 0.5}},
        {"step": 1, "both": {"brier_skill": 0.5},
         "weak_only": {"brier_skill": 0.6, "accuracy": 0.9}},
    ]
    crossing_result = summarize_response(crossing, **{**kwargs, "scale": 1.0})
    assert crossing_result["signed_normalized_deficit_auc"] == pytest.approx(0)
    assert crossing_result["positive_normalized_deficit_auc"] == pytest.approx(0.025)
    no_gate = copy.deepcopy(trace)
    no_gate[-1]["weak_only"]["accuracy"] = 0.7
    assert not summarize_response(no_gate, **kwargs)["secondary_any_time_certificate"]
    with pytest.raises(ValueError, match="increasing"):
        summarize_response([trace[0], trace[0]], **kwargs)


def test_identical_input_null_reproduces_identical_models_and_trace():
    torch.set_num_threads(1)
    base = TinyCIFAR()
    trace, both, weak, record = train_paired(
        base, tiny_spec(amplitude=0.0), seed=33001, device=torch.device("cpu")
    )
    assert [row["step"] for row in trace] == [0, 1, 2]
    assert record["final_weights_sha256_both"] == record["final_weights_sha256_weak_only"]
    assert all(row["both"] == row["weak_only"] for row in trace)
    result = summarize_response(trace, beta=0.5, scale=0.75, accuracy_gate=0.8,
                                scale_gate=0.25, gap_tolerance=1e-6,
                                material_fraction=0.1)
    assert result["checkpoint_gap"] == [0.0, 0.0, 0.0]
    assert not result["primary_causal_certificate"]
    again, _, _, record_again = train_paired(
        base, tiny_spec(amplitude=0.0), seed=33001, device=torch.device("cpu")
    )
    assert record["training_image_order_sha256"] == record_again["training_image_order_sha256"]
    assert [row["both"] for row in trace] == [row["both"] for row in again]


def test_nonzero_pair_and_behavior_schema():
    torch.set_num_threads(1)
    base = TinyCIFAR()
    config = tiny_spec(amplitude=0.5)
    trace, both, weak, record = train_paired(
        base, config, seed=33001, device=torch.device("cpu")
    )
    assert record["initial_weights_sha256"]
    assert np.isfinite([row["both"]["brier_skill"] for row in trace]).all()
    splits = split_for_seed(base, config, 33001)
    before = weights_sha256(both)
    behavior = evaluate_behavior(both, base, splits["diagnostic_evaluation"],
                                 block_seed=33001, amplitude=0.5,
                                 device=torch.device("cpu"), batch_size=10)
    assert set(behavior) == {"neutral", "random", "consistent", "conflict"}
    assert all(set(value) == {"accuracy", "brier_skill", "cross_entropy"}
               for value in behavior.values())
    assert weights_sha256(both) == before
    assert weak is not both


def test_lockstep_weak_arm_reproduces_standalone_calibration_weights():
    torch.set_num_threads(1)
    base = TinyCIFAR()
    config = tiny_spec(amplitude=0.0)
    config["seed_blocks"]["confirmation"] = [13001]
    _, _, _, record = train_paired(base, config, seed=13001,
                                    device=torch.device("cpu"))
    _, standalone = train_pilot(
        base=base, config=config, condition="weak_only", seed=13001,
        amplitude=0.0, learning_rate=0.01, weight_decay=0.0,
        steps=2, eval_every=1, device=torch.device("cpu"),
    )
    assert record["final_weights_sha256_weak_only"] == weights_sha256(standalone)


def test_tiny_paired_run_writes_complete_artifact_contract(tmp_path):
    torch.set_num_threads(1)
    output = tmp_path / "paired"
    summary = run(base=TinyCIFAR(), config=tiny_spec(), output=output,
                  seed=33001, device=torch.device("cpu"), provenance={"test": True},
                  config_snapshot=b"frozen config", environment_snapshot=b"locked env")
    metadata = json.loads((output / "metadata.json").read_text())
    trace = json.loads((output / "trace.json").read_text())
    checkpoints = json.loads((output / "checkpoints.json").read_text())
    assert metadata["status"] == "complete" and metadata["checkpoint_count"] == 3
    assert metadata["peak_cuda_memory_bytes"] is None
    assert [row["step"] for row in trace] == [0, 1, 2]
    assert [row["step"] for row in checkpoints] == [0, 1, 2]
    assert all((output / row["path"]).stat().st_size > 0 for row in checkpoints)
    assert (output / "summary.json").is_file()
    assert (output / "preregistration.frozen.json").read_bytes() == b"frozen config"
    assert (output / "environment.locked.json").read_bytes() == b"locked env"
    assert set(summary["behavior_both"]) == {"neutral", "random", "consistent", "conflict"}
    assert set(summary["behavior_weak_only"]) == {"neutral", "random", "consistent", "conflict"}
    for mode in ("neutral", "random", "consistent", "conflict"):
        assert f"original_head_{mode}_accuracy_both" in summary
        assert f"original_head_{mode}_accuracy_weak_only" in summary
        assert f"original_head_{mode}_accuracy_gap_B_minus_W" in summary
    assert "e4_lite" in summary and "both" in summary["e4_lite"]
    assert summary["e4_lite"]["both"]["fit_size"] == 10
    with pytest.raises(FileExistsError):
        run(base=TinyCIFAR(), config=tiny_spec(), output=output,
            seed=33001, device=torch.device("cpu"), provenance={"test": True})


def test_balanced_linear_probe_is_fresh_and_fails_on_imbalanced_data():
    torch.set_num_threads(1)
    labels = torch.arange(10).repeat_interleave(3)
    features = torch.nn.functional.one_hot(labels, 10).float()
    result = fit_balanced_linear_probe(
        (features, labels), (features, labels), (features, labels),
        regularization_grid=(0.0, 0.01), max_iter=30,
    )
    assert result["evaluation"]["accuracy"] == 1.0
    assert result["fit_size"] == result["evaluation_size"] == 30
    assert result["selected_l2"] in (0.0, 0.01)
    with pytest.raises(ValueError, match="equal positive class counts"):
        fit_balanced_linear_probe((features[:-1], labels[:-1]),
                                  (features, labels), (features, labels))


def test_real_confirmation_cli_refuses_unfrozen_draft(tmp_path):
    with pytest.raises(ValueError, match="Confirmation blocked"):
        preflight(SPEC, config_path=tmp_path / "draft.json",
                  environment_path=tmp_path / "env.json",
                  data_root=tmp_path, seed=33001, device=torch.device("cpu"))


def test_frozen_checker_requires_explicit_e4_policy_and_slurm(monkeypatch, tmp_path):
    frozen = copy.deepcopy(SPEC)
    frozen.update(status="frozen", source_commit="0" * 40,
                  environment_sha256="0" * 64,
                  weak_only_calibration_manifest_sha256="0" * 64,
                  cue_only_feasibility_manifest_sha256="0" * 64,
                  frozen_utc="2026-09-14T00:00:00Z")
    frozen["dataset"].update(dataset_archive_sha256="0" * 64, cue_amplitude=0.5)
    frozen["model"]["architecture_code_sha256"] = "0" * 64
    frozen["training"].update(learning_rate=0.05, weight_decay=0.0005,
                              horizon_updates=4710, checkpoint_updates=[157, 4710])
    frozen["response"].update(beta=0.5, scale_S=0.75)
    frozen["external_execution"]["slurm_resources"] = {
        "partition": "gpu_student", "gpus": 1,
        "gres": "gpu:a100_1g.5gb:1", "cpus_per_task": 4,
        "memory_gb": 16, "time_limit": "02:00:00",
    }
    with pytest.raises(ValueError, match="E4-lite"):
        validate_protocol(frozen, require_frozen=True)
    frozen["e4_lite"] = {"probe_l2_grid": [0.0, 0.01], "max_iter": 100}
    assert validate_protocol(frozen, require_frozen=True) == []
    monkeypatch.setattr("scripts.run_cifar_matched.socket.gethostname", lambda: "dgxb")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="SLURM allocation"):
        preflight(frozen, config_path=tmp_path / "frozen.json",
                  environment_path=tmp_path / "env.json",
                  data_root=tmp_path, seed=33001, device=torch.device("cuda"))


def test_environment_capture_refuses_login_and_no_slurm(monkeypatch):
    monkeypatch.setattr("scripts.record_cifar_environment.socket.gethostname",
                        lambda: "dgx-login01")
    with pytest.raises(ValueError, match="login node"):
        environment_record()
    monkeypatch.setattr("scripts.record_cifar_environment.socket.gethostname",
                        lambda: "dgxb")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="gpu_student"):
        environment_record()
