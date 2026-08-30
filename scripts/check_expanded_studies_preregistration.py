#!/usr/bin/env python3
"""Validate the blocked nonlinear/beta and CDC tradeoff design freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class ExpandedStudyPreregistrationError(ValueError):
    pass


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ExpandedStudyPreregistrationError(f"Duplicate key: {key!r}.")
        value[key] = item
    return value


def load_preregistration(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ExpandedStudyPreregistrationError(f"Non-finite value: {item}.")
            ),
        )
    except ExpandedStudyPreregistrationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExpandedStudyPreregistrationError(f"Cannot parse {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExpandedStudyPreregistrationError("Preregistration must be a mapping.")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExpandedStudyPreregistrationError(message)


def validate_preregistration(value: dict[str, Any]) -> dict[str, Any]:
    _require(
        value.get("schema_version") == "expanded-nonlinear-cdc-preregistration-v1",
        "Unsupported expanded-study schema.",
    )
    _require(value.get("freeze_revision") == 1, "Unexpected freeze revision.")
    _require(
        value.get("status") == "design_frozen_execution_blocked"
        and value.get("execution_authorized") is False,
        "The expanded studies must remain execution-blocked.",
    )

    shared = value.get("shared_task")
    _require(isinstance(shared, dict), "shared_task must be a mapping.")
    cells = shared.get("cells")
    _require(isinstance(cells, list) and len(cells) == 3, "Exactly three task cells are frozen.")
    expected_cells = [
        {"cell_id": "rho3_lag1", "rho": 3.0, "lag_separation": 1},
        {"cell_id": "rho4_lag2", "rho": 4.0, "lag_separation": 2},
        {"cell_id": "rho5_lag3", "rho": 5.0, "lag_separation": 3},
    ]
    _require(cells == expected_cells, "Frozen nonlinear task cells changed.")

    seeds = value.get("seed_design")
    _require(isinstance(seeds, dict), "seed_design must be a mapping.")
    data_seeds = seeds.get("data_seeds")
    model_seeds = seeds.get("model_seeds")
    _require(isinstance(data_seeds, list) and len(data_seeds) == len(set(data_seeds)) == 4, "Four unique data seeds are required.")
    _require(isinstance(model_seeds, list) and len(model_seeds) == len(set(model_seeds)) == 8, "Eight unique model seeds are required.")
    _require(seeds.get("crossed") is True, "Seed axes must remain crossed.")
    _require(seeds.get("candidate_seeds_require_untouched_attestation") is True, "Seed attestation is mandatory.")

    nonlinear = value.get("nonlinear_beta_study")
    _require(isinstance(nonlinear, dict), "nonlinear_beta_study must be a mapping.")
    _require(nonlinear.get("model_kinds") == ["tanh", "gru"], "Nonlinear model set changed.")
    _require(nonlinear.get("cells") == [cell["cell_id"] for cell in expected_cells], "Nonlinear cell set changed.")
    _require(nonlinear.get("beta_values") == [0.25, 0.5, 1.0, 2.0], "Prospective beta grid changed.")
    semantics = nonlinear.get("beta_semantics")
    _require(isinstance(semantics, dict), "beta_semantics must be a mapping.")
    for field in (
        "prospective_grid",
        "first_hit_not_terminal_reach",
        "target_at_initialization_is_degenerate",
        "weak_only_nonhit_is_unlearnable",
        "causal_certificate_requires_outcome_suppression_and_non_degenerate_weak_first_hit",
        "no_beta_or_cell_selection_after_outcomes",
    ):
        _require(semantics.get(field) is True, f"Required beta semantic {field!r} changed.")
    nonlinear_records = len(nonlinear["model_kinds"]) * len(nonlinear["cells"]) * len(data_seeds) * len(model_seeds)
    _require(nonlinear.get("record_count") == nonlinear_records == 192, "Nonlinear record count is inconsistent.")

    cdc = value.get("cdc_tradeoff_study")
    _require(isinstance(cdc, dict), "cdc_tradeoff_study must be a mapping.")
    methods = cdc.get("methods")
    expected_methods = [
        {"method_id": "erm", "publication_label": "ERM", "uses_weak_only_shadow": False},
        {"method_id": "counterfactual_drift", "publication_label": "CDC", "uses_weak_only_shadow": True},
        {"method_id": "bloop", "publication_label": "Bloop-style shadow-target rescue", "uses_weak_only_shadow": True},
        {"method_id": "pcgrad", "publication_label": "PCGrad-style shadow-target rescue", "uses_weak_only_shadow": True},
    ]
    _require(methods == expected_methods, "Method IDs, information settings, or style labels changed.")
    fidelity = cdc.get("baseline_fidelity")
    _require(
        isinstance(fidelity, dict)
        and fidelity.get("bloop_is_canonical") is False
        and fidelity.get("pcgrad_is_canonical") is False
        and fidelity.get("canonical_name_promotion_forbidden") is True,
        "Canonical baseline fidelity must remain explicitly false.",
    )
    tradeoff = cdc.get("tradeoff_rule")
    _require(isinstance(tradeoff, dict), "tradeoff_rule must be a mapping.")
    _require(tradeoff.get("weak_rescue_equivalence_margin") == 0.05, "Weak equivalence margin changed.")
    _require(tradeoff.get("strong_superiority_requires_ci95_lower_strictly_positive") is True, "Strong superiority rule changed.")
    _require(tradeoff.get("no_dominance_claim_from_instantaneous_theorem_alone") is True, "Instantaneous-to-final promotion guard changed.")
    cdc_records = len(cdc["model_kinds"]) * len(cdc["cells"]) * len(data_seeds) * len(model_seeds) * len(methods)
    _require(cdc.get("method_record_count") == cdc_records == 256, "CDC method record count is inconsistent.")

    inference = value.get("inference")
    _require(isinstance(inference, dict), "inference must be a mapping.")
    _require(inference.get("seed_reuse_method") == "two_way_pigeonhole_bootstrap", "Reuse-aware inference changed.")
    _require(inference.get("bootstrap_replicates") == 5000, "Bootstrap replicate count changed.")
    _require(inference.get("equivalence") == "entire reuse-aware ci inside preregistered margin", "Equivalence rule changed.")
    _require(inference.get("joint_rule") == "intersection_union", "Joint tradeoff rule changed.")

    gates = value.get("authorization_gates")
    _require(isinstance(gates, dict), "authorization_gates must be a mapping.")
    _require(gates.get("reviewed_clean_git_source") is False, "Clean-source review has not passed.")
    _require(gates.get("committed_preregistration_git_sha") is None, "No preregistration commit is authorized.")
    _require(gates.get("environment_lock_sha256") is None, "Fresh environment digest must remain unresolved.")
    _require(gates.get("preflight_manifest_sha256") is None, "Fresh preflight digest must remain unresolved.")
    _require(gates.get("memory_telemetry_implemented") is False, "Peak-memory telemetry is still blocked.")
    _require(gates.get("untouched_seed_attestation") is False, "Untouched seeds have not been attested.")
    _require(gates.get("submission_claim_set_allows_new_outcomes") is False, "Claim gate still forbids outcomes.")
    reasons = value.get("blocked_reasons")
    _require(isinstance(reasons, list) and len(reasons) >= 7, "Blocked reasons are incomplete.")

    return {
        "valid": True,
        "execution_authorized": False,
        "nonlinear_record_count": nonlinear_records,
        "cdc_method_record_count": cdc_records,
        "beta_count": len(nonlinear["beta_values"]),
        "canonical_named_baselines": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("configs/expanded_nonlinear_cdc_preregistration.yaml"),
    )
    parser.add_argument("--require-authorized", action="store_true")
    arguments = parser.parse_args()
    try:
        report = validate_preregistration(load_preregistration(arguments.path))
        if arguments.require_authorized:
            raise ExpandedStudyPreregistrationError("Expanded study outcome generation remains blocked.")
    except ExpandedStudyPreregistrationError as error:
        print(f"expanded-study preregistration: FAIL: {error}")
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
