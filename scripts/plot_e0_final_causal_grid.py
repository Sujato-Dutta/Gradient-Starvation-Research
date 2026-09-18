"""Render the audited dense-E1 grid under the final causal definition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gradient_starvation.plotting import plot_final_causal_grid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "e0_consistency_audit-20260914" / "run_level.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper" / "figures" / "fig_e0_final_causal_grid",
    )
    args = parser.parse_args()
    records = pd.read_csv(args.input)
    plot_final_causal_grid(records, args.output)
    print(f"Saved {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
