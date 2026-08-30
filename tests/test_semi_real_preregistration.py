from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.check_semi_real_preregistration import (
    SemiRealPreregistrationError,
    load_preregistration,
    validate_preregistration,
)


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "configs/semi_real_preregistration.yaml"


def test_semi_real_design_is_valid_but_outcomes_remain_blocked():
    report = validate_preregistration(load_preregistration(PREREGISTRATION))
    assert report == {
        "valid": True,
        "execution_authorized": False,
        "dataset_count": 2,
        "record_count": 64,
        "unresolved_sha256_gates": 4,
    }


def test_semi_real_design_rejects_endpoint_or_authorization_drift():
    original = load_preregistration(PREREGISTRATION)
    changed_endpoint = copy.deepcopy(original)
    changed_endpoint["primary_endpoints"] = ["exact_drift_crossing"]
    with pytest.raises(SemiRealPreregistrationError, match="Primary endpoints"):
        validate_preregistration(changed_endpoint)

    unauthorized = copy.deepcopy(original)
    unauthorized["execution_authorized"] = True
    with pytest.raises(SemiRealPreregistrationError, match="Outcome execution is forbidden"):
        validate_preregistration(unauthorized)
