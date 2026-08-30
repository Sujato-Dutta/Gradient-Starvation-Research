#!/usr/bin/env python
"""Read-only validator for the semi-real execution contract and artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gradient_starvation.semi_real_study import (  # noqa: E402
    SemiRealStudyError,
    load_execution_contract,
    validate_execution_contract,
    validate_run_artifacts,
    verify_authorization,
    verify_preflight,
)

DEFAULT_CONTRACT = ROOT / "configs" / "semi_real_execution.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate semi-real execution inputs.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    contract = subparsers.add_parser("contract", help="Validate every frozen contract field.")
    contract.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)

    preflight = subparsers.add_parser("preflight", help="Recompute and validate a preflight.")
    preflight.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    preflight.add_argument("--preflight", type=Path, required=True)
    preflight.add_argument("--data-root", type=Path, required=True)
    preflight.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Validate hashes even when the preflight truthfully records a dirty source blocker.",
    )

    authorize = subparsers.add_parser(
        "authorize", help="Require a complete external seal; constructs no model or optimizer."
    )
    authorize.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    authorize.add_argument("--preflight", type=Path, required=True)
    authorize.add_argument("--seal", type=Path, required=True)
    authorize.add_argument("--data-root", type=Path, required=True)

    artifacts = subparsers.add_parser(
        "artifacts", help="Rehash a complete run and deterministically recompute inference."
    )
    artifacts.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.mode == "contract":
            value = load_execution_contract(arguments.contract, repository_root=ROOT)
            report = validate_execution_contract(value, repository_root=ROOT)
        elif arguments.mode == "preflight":
            verified = verify_preflight(
                arguments.contract,
                arguments.preflight,
                data_root=arguments.data_root,
                repository_root=ROOT,
                require_ready=not arguments.allow_blocked,
            )
            report = {
                "valid": True,
                "ready_for_authorization": verified.manifest["ready_for_authorization"],
                "blockers": verified.manifest["blockers"],
                "selection_count": verified.selection_manifest["selection_count"],
                "execution_authorized": False,
            }
        elif arguments.mode == "authorize":
            authorized = verify_authorization(
                arguments.contract,
                arguments.preflight,
                arguments.seal,
                data_root=arguments.data_root,
                repository_root=ROOT,
            )
            report = {
                "valid": True,
                "execution_authorized": True,
                "study_id": authorized.preflight.contract["study_id"],
                "record_count": len(authorized.seal["bindings"]["expected_record_ids"]),
            }
        else:
            report = validate_run_artifacts(arguments.run_dir, repository_root=ROOT)
    except (SemiRealStudyError, OSError, ValueError, KeyError) as error:
        print(f"semi-real execution: FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
