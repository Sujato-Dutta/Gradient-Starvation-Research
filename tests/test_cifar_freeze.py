import json
from pathlib import Path

from scripts.check_cifar_preregistration import validate_protocol
from scripts.freeze_cifar_preregistration import build_frozen_protocol


ROOT = Path(__file__).resolve().parents[1]


def test_phase1_manifests_build_a_complete_frozen_protocol():
    draft = json.loads((ROOT / "configs/cifar_matched_preregistration.draft.json").read_text())
    weak = json.loads((ROOT / "research_scope/cifar_weak_only_calibration_manifest.json").read_text())
    cue = json.loads((ROOT / "research_scope/cifar_cue_only_feasibility_manifest.json").read_text())
    frozen = build_frozen_protocol(
        draft, weak, cue,
        source_commit="1" * 40,
        environment_sha256="2" * 64,
        weak_manifest_sha256="3" * 64,
        cue_manifest_sha256="4" * 64,
        architecture_sha256=weak["protocol_code_sha256"],
        frozen_utc="2026-09-16T00:00:00+00:00",
    )
    assert validate_protocol(frozen, require_frozen=True) == []
    assert frozen["training"]["checkpoint_updates"] == list(range(157, 4711, 157))
    assert frozen["dataset"]["cue_amplitude"] == 0.5
    assert frozen["external_execution"]["slurm_resources"]["time_limit"] == "02:00:00"
