#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from gradient_starvation.config import load_config
from gradient_starvation.experiments import run_e1, run_e2, run_e3
from gradient_starvation.waterbirds import run_waterbirds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal gradient-starvation experiments.")
    parser.add_argument("experiment", choices=["e1", "e2", "e3", "waterbirds"])
    parser.add_argument("--config", required=True, help="Path to a YAML configuration.")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="Override a configuration value, e.g. --set training.steps=20.",
    )
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    runners = {"e1": run_e1, "e2": run_e2, "e3": run_e3, "waterbirds": run_waterbirds}
    run_dir = runners[args.experiment](config)
    print(f"Completed {args.experiment}. Results: {run_dir}")


if __name__ == "__main__":
    main()

