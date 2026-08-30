from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.check_expanded_studies_preregistration import (
    ExpandedStudyPreregistrationError,
    load_preregistration,
    validate_preregistration,
)


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "configs/expanded_nonlinear_cdc_preregistration.yaml"


def test_expanded_design_is_valid_but_outcome_generation_is_blocked():
    report = validate_preregistration(load_preregistration(PREREGISTRATION))
    assert report == {
        "valid": True,
        "execution_authorized": False,
        "nonlinear_record_count": 192,
        "cdc_method_record_count": 256,
        "beta_count": 4,
        "canonical_named_baselines": 0,
    }


def test_expanded_design_rejects_beta_or_baseline_promotion_drift():
    original = load_preregistration(PREREGISTRATION)
    changed_beta = copy.deepcopy(original)
    changed_beta["nonlinear_beta_study"]["beta_values"] = [0.5]
    with pytest.raises(ExpandedStudyPreregistrationError, match="beta grid"):
        validate_preregistration(changed_beta)

    promoted = copy.deepcopy(original)
    promoted["cdc_tradeoff_study"]["methods"][2]["publication_label"] = "Bloop"
    with pytest.raises(ExpandedStudyPreregistrationError, match="style labels"):
        validate_preregistration(promoted)
