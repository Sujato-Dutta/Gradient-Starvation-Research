import copy
import json
from pathlib import Path

import pytest

from scripts.check_cifar_preregistration import validate_protocol


SPEC = json.loads((Path(__file__).resolve().parents[1] / "configs/cifar_matched_preregistration.draft.json").read_text())


def test_current_draft_is_structurally_valid_but_blocks_confirmation():
    assert "beta" in validate_protocol(SPEC)
    with pytest.raises(ValueError, match="Confirmation blocked"):
        validate_protocol(SPEC, require_frozen=True)


def test_overlapping_seeds_and_missing_e4_fail_closed():
    overlapping = copy.deepcopy(SPEC)
    overlapping["seed_blocks"]["confirmation"][0] = 13001
    with pytest.raises(ValueError, match="disjoint"):
        validate_protocol(overlapping)
    missing = copy.deepcopy(SPEC)
    missing["evaluation"].remove("E4_lite_final_checkpoint_balanced_fresh_linear_head")
    with pytest.raises(ValueError, match="endpoint"):
        validate_protocol(missing)


def test_frozen_status_cannot_hide_unresolved_values():
    mutated = copy.deepcopy(SPEC)
    mutated["status"] = "frozen"
    with pytest.raises(ValueError, match="unresolved"):
        validate_protocol(mutated)
