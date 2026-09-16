"""Create the immutable CIFAR confirmation protocol from reviewed evidence.

This script does not run training. It refuses to freeze unless the pilot
manifests match the unchanged draft and protocol source, the execution source
is present byte-for-byte in a reviewed commit, and a DGX environment-v2
manifest is available.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_cifar_preregistration import validate_protocol  # noqa: E402
from scripts.run_cifar_matched import SOURCE_PATHS  # noqa: E402


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def build_frozen_protocol(
    draft: dict,
    weak_manifest: dict,
    cue_manifest: dict,
    *,
    source_commit: str,
    environment_sha256: str,
    weak_manifest_sha256: str,
    cue_manifest_sha256: str,
    architecture_sha256: str,
    frozen_utc: str,
) -> dict:
    """Populate all prospective choices without consulting a B outcome."""
    if draft.get("status") != "draft_do_not_run_confirmation":
        raise ValueError("Input protocol must be the unfrozen confirmation draft.")
    if (weak_manifest.get("status") != "exploratory_W_only_not_confirmation" or
            cue_manifest.get("status") != "exploratory_cue_only_not_confirmation"):
        raise ValueError("Unexpected Phase-1 pilot-manifest status.")
    if (weak_manifest.get("dataset_archive_sha256") !=
            cue_manifest.get("dataset_archive_sha256")):
        raise ValueError("W-only and cue-only pilots used different CIFAR archives.")
    if (weak_manifest.get("protocol_code_sha256") != architecture_sha256 or
            cue_manifest.get("protocol_code_sha256") != architecture_sha256):
        raise ValueError("Pilot protocol bytes differ from the reviewed architecture source.")
    candidate = weak_manifest.get("candidate_not_frozen", {})
    required = {
        "horizon_updates", "checkpoint_spacing_updates", "beta", "scale_S",
        "learning_rate", "weight_decay",
    }
    if set(candidate) != required:
        raise ValueError("W-only manifest does not contain the exact freeze candidate fields.")
    horizon = candidate["horizon_updates"]
    spacing = candidate["checkpoint_spacing_updates"]
    if (not isinstance(horizon, int) or not isinstance(spacing, int) or
            horizon <= 0 or spacing <= 0 or horizon % spacing):
        raise ValueError("Candidate horizon must be a positive multiple of checkpoint spacing.")
    amplitude = cue_manifest.get("candidate_cue_amplitude_not_frozen")
    if not isinstance(amplitude, (int, float)) or not 0 < amplitude <= 1:
        raise ValueError("Cue-only manifest lacks a valid amplitude candidate.")
    for name, value, length in (
        ("source_commit", source_commit, 40),
        ("environment_sha256", environment_sha256, 64),
        ("weak_manifest_sha256", weak_manifest_sha256, 64),
        ("cue_manifest_sha256", cue_manifest_sha256, 64),
        ("architecture_sha256", architecture_sha256, 64),
    ):
        if re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
            raise ValueError(f"{name} is not a lowercase hexadecimal digest.")

    frozen = copy.deepcopy(draft)
    frozen.update({
        "status": "frozen",
        "source_commit": source_commit,
        "environment_sha256": environment_sha256,
        "weak_only_calibration_manifest_sha256": weak_manifest_sha256,
        "cue_only_feasibility_manifest_sha256": cue_manifest_sha256,
        "frozen_utc": frozen_utc,
    })
    frozen["dataset"].update({
        "dataset_archive_sha256": weak_manifest["dataset_archive_sha256"],
        "cue_amplitude": amplitude,
    })
    frozen["model"]["architecture_code_sha256"] = architecture_sha256
    frozen["training"].update({
        "learning_rate": candidate["learning_rate"],
        "weight_decay": candidate["weight_decay"],
        "horizon_updates": horizon,
        "checkpoint_updates": list(range(spacing, horizon + 1, spacing)),
    })
    frozen["response"].update({
        "beta": candidate["beta"],
        "scale_S": candidate["scale_S"],
    })
    frozen["e4_lite"] = {
        "probe_l2_grid": [0.0, 0.001, 0.01],
        "max_iter": 100,
        "selection_metric": "validation_cross_entropy",
        "initialization": "all_zero_linear_softmax_head",
    }
    frozen["external_execution"]["slurm_resources"] = {
        "partition": "gpu_student",
        "gpus": 1,
        "gres": "gpu:a100_1g.5gb:1",
        "cpus_per_task": 4,
        "memory_gb": 16,
        "time_limit": "02:00:00",
    }
    validate_protocol(frozen, require_frozen=True)
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--weak-manifest", type=Path, required=True)
    parser.add_argument("--cue-manifest", type=Path, required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-utc", help="ISO-8601 time; default is current UTC")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Frozen protocol output already exists; do not overwrite it.")
    try:
        draft_bytes = args.draft.read_bytes()
        weak_bytes = args.weak_manifest.read_bytes()
        cue_bytes = args.cue_manifest.read_bytes()
        environment = json.loads(args.environment_manifest.read_text())
        weak = json.loads(weak_bytes)
        cue = json.loads(cue_bytes)
        if (weak.get("draft_config_sha256") != hashlib.sha256(draft_bytes).hexdigest() or
                cue.get("draft_config_sha256") != hashlib.sha256(draft_bytes).hexdigest()):
            raise ValueError("Pilot manifests do not reference the unchanged draft protocol.")
        if environment.get("schema_version") != "cifar-dgx-environment-v2":
            raise ValueError("Expected a CIFAR DGX environment-v2 manifest.")
        source_commit = args.source_commit.lower()
        _git("merge-base", "--is-ancestor", source_commit, "HEAD")
        for relative in SOURCE_PATHS:
            if _git("show", f"{source_commit}:{relative}") != (ROOT / relative).read_bytes():
                raise ValueError(f"Reviewed commit differs from current source: {relative}.")
        architecture_sha256 = _digest(ROOT / SOURCE_PATHS[0])
        frozen_utc = args.frozen_utc or datetime.now(timezone.utc).isoformat()
        frozen = build_frozen_protocol(
            json.loads(draft_bytes), weak, cue,
            source_commit=source_commit,
            environment_sha256=_digest(args.environment_manifest),
            weak_manifest_sha256=hashlib.sha256(weak_bytes).hexdigest(),
            cue_manifest_sha256=hashlib.sha256(cue_bytes).hexdigest(),
            architecture_sha256=architecture_sha256,
            frozen_utc=frozen_utc,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(frozen, indent=2, allow_nan=False) + "\n").encode()
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(args.output)
    except (OSError, KeyError, TypeError, ValueError,
            subprocess.CalledProcessError) as exc:
        parser.exit(2, f"CIFAR preregistration freeze failed: {exc}\n")
    print(json.dumps({"output": str(args.output),
                      "sha256": hashlib.sha256(payload).hexdigest(),
                      "source_commit": source_commit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
