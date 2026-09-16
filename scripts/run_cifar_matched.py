"""Run one frozen CIFAR B/W confirmation block on a SLURM compute node.

There is intentionally no draft/debug override. A frozen, committed protocol,
committed source, environment manifest, and verified dataset are mandatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from gradient_starvation.cifar_matched import (  # noqa: E402
    evaluate_behavior, split_for_seed, summarize_response, train_paired,
    weights_sha256,
)
from gradient_starvation.cifar_probe import evaluate_e4_lite  # noqa: E402
from scripts.check_cifar_preregistration import validate_protocol  # noqa: E402
from scripts.record_cifar_environment import runtime_fingerprint  # noqa: E402


SOURCE_PATHS = (
    "src/gradient_starvation/cifar_protocol.py",
    "src/gradient_starvation/cifar_matched.py",
    "src/gradient_starvation/cifar_probe.py",
    "scripts/check_cifar_preregistration.py",
    "scripts/run_cifar_matched.py",
    "scripts/aggregate_cifar_matched.py",
    "scripts/record_cifar_environment.py",
)


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def preflight(config: dict, *, config_path: Path, environment_path: Path,
              data_root: Path, seed: int, device: torch.device) -> dict:
    """Fail before B data or output creation if provenance is not locked."""
    validate_protocol(config, require_frozen=True)
    if seed not in config["seed_blocks"]["confirmation"]:
        raise ValueError("Seed is not one of the eight untouched confirmation blocks.")
    if socket.gethostname().startswith("dgx-login"):
        raise ValueError("CIFAR training is forbidden on the DGX login node.")
    if (not os.environ.get("SLURM_JOB_ID") or
            os.environ.get("SLURM_JOB_PARTITION") != "gpu_student"):
        raise ValueError("Confirmation requires an active gpu_student SLURM allocation.")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Confirmation requires a SLURM GPU compute allocation.")
    if _digest(environment_path) != config["environment_sha256"]:
        raise ValueError("Environment manifest SHA-256 differs from frozen protocol.")
    env = json.loads(environment_path.read_text())
    observed = runtime_fingerprint()
    if (env.get("schema_version") != "cifar-dgx-environment-v2" or
            env.get("runtime") != observed):
        raise ValueError(f"Runtime environment differs from manifest: {observed}.")
    if _digest(data_root / "cifar-10-python.tar.gz") != config["dataset"]["dataset_archive_sha256"]:
        raise ValueError("CIFAR-10 archive SHA-256 differs from frozen protocol.")
    if _digest(ROOT / SOURCE_PATHS[0]) != config["model"]["architecture_code_sha256"]:
        raise ValueError("Architecture/protocol code SHA-256 differs from frozen protocol.")
    source_commit = config["source_commit"]
    for relative in SOURCE_PATHS:
        if _git("show", f"{source_commit}:{relative}") != (ROOT / relative).read_bytes():
            raise ValueError(f"Source differs from reviewed commit: {relative}.")
    config_relative = config_path.resolve().relative_to(ROOT.resolve()).as_posix()
    if _git("show", f"HEAD:{config_relative}") != config_path.read_bytes():
        raise ValueError("Frozen preregistration is not committed unchanged at HEAD.")
    e4 = config.get("e4_lite")
    if not isinstance(e4, dict) or not e4.get("probe_l2_grid") or not isinstance(e4.get("max_iter"), int):
        raise ValueError("Frozen E4-lite grid and iteration budget are required.")
    return {"environment": observed, "source_commit": source_commit,
            "config_sha256": _digest(config_path),
            "environment_sha256": _digest(environment_path),
            "dataset_archive_sha256": _digest(data_root / "cifar-10-python.tar.gz")}


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(*, base: object, config: dict, output: Path, seed: int,
        device: torch.device, provenance: dict,
        config_snapshot: bytes | None = None,
        environment_snapshot: bytes | None = None) -> dict:
    """Train and evaluate; a missing complete status marks interrupted jobs."""
    output.mkdir(parents=True, exist_ok=False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if config_snapshot is not None:
        (output / "preregistration.frozen.json").write_bytes(config_snapshot)
    if environment_snapshot is not None:
        (output / "environment.locked.json").write_bytes(environment_snapshot)
    checkpoints_dir = output / "checkpoints"
    checkpoints_dir.mkdir()
    metadata = {"status": "running_incomplete_if_interrupted", "seed": seed,
                "condition_pair": ["both", "weak_only"], **provenance}
    _write_json(output / "metadata.json", metadata)
    checkpoint_records = []
    trace_so_far = []

    def save_checkpoint(step: int, both: torch.nn.Module,
                        weak: torch.nn.Module, row: dict) -> None:
        path = checkpoints_dir / f"step_{step:06d}.pt"
        torch.save({"step": step, "both": both.state_dict(),
                    "weak_only": weak.state_dict()}, path)
        checkpoint_records.append({
            "step": step, "path": path.relative_to(output).as_posix(),
            "file_sha256": _digest(path),
            "weights_sha256_both": weights_sha256(both),
            "weights_sha256_weak_only": weights_sha256(weak),
        })
        trace_so_far.append(row)
        _write_json(output / "trace.json", trace_so_far)
        _write_json(output / "checkpoints.json", checkpoint_records)

    trace, both, weak, training = train_paired(
        base, config, seed=seed, device=device, checkpoint=save_checkpoint,
    )
    assert trace == trace_so_far
    splits = split_for_seed(base, config, seed)
    response = config["response"]
    analysis = config["confirmatory_analysis"]
    summary = summarize_response(
        trace, beta=response["beta"], scale=response["scale_S"],
        accuracy_gate=response["weak_only_minimum_neutral_accuracy"],
        scale_gate=response["weak_only_scale_minimum"],
        gap_tolerance=response["gap_tolerance"],
        material_fraction=analysis["pointwise_deficit_fraction_of_S"],
    )
    summary["signed_auc_target_met"] = (
        summary["signed_normalized_deficit_auc"] >= analysis["normalized_signed_auc_target"]
    )
    summary["duration_target_met"] = (
        summary["maximum_material_deficit_duration_fraction"] >=
        analysis["contiguous_duration_fraction_of_H"]
    )
    for name, model in (("both", both), ("weak_only", weak)):
        summary[f"behavior_{name}"] = evaluate_behavior(
            model, base, splits["diagnostic_evaluation"], block_seed=seed,
            amplitude=config["dataset"]["cue_amplitude"], device=device,
        )
    for mode in ("neutral", "random", "consistent", "conflict"):
        both_accuracy = summary["behavior_both"][mode]["accuracy"]
        weak_accuracy = summary["behavior_weak_only"][mode]["accuracy"]
        summary[f"original_head_{mode}_accuracy_both"] = both_accuracy
        summary[f"original_head_{mode}_accuracy_weak_only"] = weak_accuracy
        summary[f"original_head_{mode}_accuracy_gap_B_minus_W"] = (
            both_accuracy - weak_accuracy
        )
    summary["behavior_harm_target_met"] = any(
        summary[name] <= -analysis["neutral_or_conflict_accuracy_harm_target"]
        for name in ("original_head_neutral_accuracy_gap_B_minus_W",
                     "original_head_conflict_accuracy_gap_B_minus_W")
    )
    e4 = config["e4_lite"]
    summary["e4_lite"] = evaluate_e4_lite(
        both, weak, base, splits, block_seed=seed,
        amplitude=config["dataset"]["cue_amplitude"], device=device,
        regularization_grid=tuple(e4["probe_l2_grid"]), max_iter=e4["max_iter"],
    )
    summary["training"] = training
    summary["peak_cuda_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda" else None
    )
    summary["seed"] = seed
    _write_json(output / "summary.json", summary)
    metadata.update({
        "status": "complete", "last_update": trace[-1]["step"],
        "checkpoint_count": len(checkpoint_records),
        "trace_sha256": _digest(output / "trace.json"),
        "checkpoint_manifest_sha256": _digest(output / "checkpoints.json"),
        "summary_sha256": _digest(output / "summary.json"),
        "training_image_order_sha256": training["training_image_order_sha256"],
        "peak_cuda_memory_bytes": summary["peak_cuda_memory_bytes"],
    })
    _write_json(output / "metadata.json", metadata)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output path exists; never overwrite a confirmation block.")
    try:
        config = json.loads(args.config.read_text())
        device = torch.device(args.device)
        provenance = preflight(
            config, config_path=args.config, environment_path=args.environment_manifest,
            data_root=args.data_root, seed=args.seed, device=device,
        )
        from torchvision.datasets import CIFAR10
        base = CIFAR10(root=str(args.data_root), train=True, download=False)
        summary = run(base=base, config=config, output=args.output,
                      seed=args.seed, device=device, provenance=provenance,
                      config_snapshot=args.config.read_bytes(),
                      environment_snapshot=args.environment_manifest.read_bytes())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"CIFAR confirmation blocked or failed: {exc}\n")
    print(json.dumps({"output": str(args.output), "seed": args.seed,
                      "primary_causal_certificate": summary["primary_causal_certificate"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
