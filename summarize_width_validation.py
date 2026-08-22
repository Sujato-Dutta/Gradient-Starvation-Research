#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from gradient_starvation.width_validation import summarize_width_runs


def _width_run(value: str) -> tuple[int, Path]:
    try:
        raw_width, raw_path = value.split("=", 1)
        return int(raw_width), Path(raw_path)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Expected WIDTH=RUN_DIRECTORY.") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine matched-seed E3 width checks.")
    parser.add_argument("--run", action="append", required=True, type=_width_run)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_dir = summarize_width_runs(dict(args.run), args.output)
    print(f"Combined width validation: {run_dir}")


if __name__ == "__main__":
    main()
