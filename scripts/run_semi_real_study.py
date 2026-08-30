#!/usr/bin/env python
"""Run semi-real preflight, sealed execution, or excluded-seed toy smoke."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gradient_starvation.semi_real_study import (  # noqa: E402
    SemiRealStudyError,
    build_preflight,
    run_engineering_smoke,
    run_scientific_study,
)

DEFAULT_CONTRACT = ROOT / "configs" / "semi_real_execution.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auditable semi-real generated-cue study pipeline."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    preflight = subparsers.add_parser(
        "preflight", help="Hash inputs and freeze indices without constructing a model."
    )
    preflight.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    preflight.add_argument("--data-root", type=Path, required=True)
    preflight.add_argument("--output-dir", type=Path, required=True)
    preflight.add_argument(
        "--download",
        action="store_true",
        help="Explicitly permit official torchvision MNIST/FashionMNIST acquisition.",
    )

    execute = subparsers.add_parser(
        "run", help="Run all scientific records only after external seal verification."
    )
    execute.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    execute.add_argument("--preflight", type=Path, required=True)
    execute.add_argument("--seal", type=Path, required=True)
    execute.add_argument("--data-root", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, required=True)

    smoke = subparsers.add_parser(
        "smoke", help="Run generated toy data with excluded engineering seeds only."
    )
    smoke.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    smoke.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.mode == "preflight":
            result = build_preflight(
                arguments.contract,
                data_root=arguments.data_root,
                output_directory=arguments.output_dir,
                allow_download=arguments.download,
                repository_root=ROOT,
            )
            print(f"semi-real preflight: {result}")
        elif arguments.mode == "run":
            result = run_scientific_study(
                arguments.contract,
                arguments.preflight,
                arguments.seal,
                data_root=arguments.data_root,
                output_directory=arguments.output_dir,
                repository_root=ROOT,
            )
            print(f"semi-real scientific run: {result}")
        else:
            result = run_engineering_smoke(
                arguments.contract,
                output_directory=arguments.output_dir,
                repository_root=ROOT,
            )
            print(f"semi-real engineering smoke (non-scientific): {result}")
    except (SemiRealStudyError, OSError, ValueError) as error:
        print(f"semi-real {arguments.mode}: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
