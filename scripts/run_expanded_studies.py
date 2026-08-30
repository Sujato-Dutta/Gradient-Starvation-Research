#!/usr/bin/env python3
"""Run expanded-study preflight, sealed execution, or excluded-seed smoke."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gradient_starvation.expanded_studies import (  # noqa: E402
    ExpandedStudiesError,
    build_preflight,
    run_engineering_smoke,
    run_record_worker,
    run_scientific_studies,
)

DEFAULT_CONTRACT = ROOT / "configs" / "expanded_studies_execution.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authorization-gated expanded nonlinear/beta/CDC study pipeline."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    preflight = subparsers.add_parser(
        "preflight",
        help="Freeze source, records, bootstrap draws, tasks, and initial states without optimization.",
    )
    preflight.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    preflight.add_argument("--output-dir", type=Path, required=True)

    execute = subparsers.add_parser(
        "run",
        help="Run all 448 scientific records only after external-seal verification.",
    )
    execute.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    execute.add_argument("--preflight", type=Path, required=True)
    execute.add_argument("--seal", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, required=True)

    smoke = subparsers.add_parser(
        "smoke",
        help="Run excluded-seed, two-step engineering checks that are ineligible as evidence.",
    )
    smoke.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    smoke.add_argument("--output-dir", type=Path, required=True)

    # Internal workers are still independently authorization-gated. Keeping this
    # as a CLI stage gives each method-record a fresh process and isolated RSS.
    worker = subparsers.add_parser(
        "worker", help="Internal authorization-gated isolated-record worker."
    )
    worker.add_argument("--contract", type=Path, required=True)
    worker.add_argument("--preflight", type=Path, required=True)
    worker.add_argument("--seal", type=Path, required=True)
    worker.add_argument("--record-id", required=True)
    worker.add_argument("--output-directory", type=Path, required=True)
    worker.add_argument("--repository-root", type=Path, default=ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.mode == "preflight":
            result = build_preflight(
                arguments.contract,
                output_directory=arguments.output_dir,
                repository_root=ROOT,
            )
            print(f"expanded-studies preflight: {result}")
        elif arguments.mode == "run":
            result = run_scientific_studies(
                arguments.contract,
                arguments.preflight,
                arguments.seal,
                output_directory=arguments.output_dir,
                repository_root=ROOT,
            )
            print(f"expanded-studies scientific run: {result}")
        elif arguments.mode == "smoke":
            result = run_engineering_smoke(
                arguments.contract,
                output_directory=arguments.output_dir,
                repository_root=ROOT,
            )
            print(f"expanded-studies engineering smoke (non-scientific): {result}")
        else:
            result = run_record_worker(
                arguments.contract,
                arguments.preflight,
                arguments.seal,
                arguments.record_id,
                output_directory=arguments.output_directory,
                repository_root=arguments.repository_root,
            )
            print(f"expanded-studies worker complete: {result}")
    except (ExpandedStudiesError, OSError, ValueError, RuntimeError) as error:
        print(f"expanded-studies {arguments.mode}: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
