"""Recompute E0 labels from local historical raw CSVs without retraining.

The source runs were executed from dirty historical trees, so this is an audit
of preserved outcome records, not a claim that their training code was replayed.
Run with `python scripts/audit_e0_consistency.py --output results/e0_audit`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gradient_starvation.causal_audit import audit_paired_response  # noqa: E402


SOURCES = (
    (
        "dense_e1", ROOT / "results/e1_dense_rerun-20260823-085142",
        ROOT / "results/e1_dense_rerun-20260823-085142", 0.25,
    ),
    (
        "nonlinear_enl", ROOT / "results/enl_tanh_crossover-20260824-141207",
        ROOT / "paper/artifacts/enl_tanh_crossover-20260824-141207", 0.5,
    ),
)
KEYS = ("model_kind", "rho", "lag_separation", "regime", "seed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_run(source: str, raw_directory: Path, summary_directory: Path, beta: float) -> pd.DataFrame:
    raw_path = raw_directory / "trajectories.csv"
    summary_path = summary_directory / "summary.csv"
    if not raw_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"E0 needs both {raw_path} and {summary_path}")
    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path)
    if "model_kind" not in raw:
        raw["model_kind"] = "dense_linear"
    if "model_kind" not in summary:
        summary["model_kind"] = "dense_linear"
    required = set(KEYS) | {"condition", "step", "tau", "m_w"}
    if required - set(raw):
        raise ValueError(f"Missing raw columns: {sorted(required - set(raw))}")
    if set(KEYS) - set(summary):
        raise ValueError("Summary lacks run identity columns.")
    if summary.duplicated(list(KEYS)).any():
        raise ValueError("Summary contains duplicate run identities.")
    indexed_summary = summary.set_index(list(KEYS))

    rows: list[dict] = []
    for key, group in raw.groupby(list(KEYS), sort=True, dropna=False):
        if set(group.condition) != {"both", "weak_only"}:
            raise ValueError(f"Missing or extra paired condition for {key}")
        both = group[group.condition == "both"].sort_values("step")
        weak = group[group.condition == "weak_only"].sort_values("step")
        if both.step.duplicated().any() or weak.step.duplicated().any():
            raise ValueError(f"Duplicate logged steps for {key}")
        if not np.array_equal(both.step.to_numpy(), weak.step.to_numpy()):
            raise ValueError(f"B/W logged steps differ for {key}")
        if not np.allclose(both.tau.to_numpy(), weak.tau.to_numpy(), rtol=0, atol=1e-10):
            raise ValueError(f"B/W time grids differ for {key}")
        if key not in indexed_summary.index:
            raise ValueError(f"No summary row for {key}")
        verdict = audit_paired_response(
            both.tau.to_numpy(), both.m_w.to_numpy(), weak.m_w.to_numpy(), beta=beta
        )
        stored = indexed_summary.loc[key]
        stored_auc = float(stored["weak_auc_gap"])
        if not math.isclose(
            verdict.signed_weak_auc_deficit, stored_auc, rel_tol=1e-8, abs_tol=1e-6
        ):
            raise ValueError(f"Stored AUC disagrees with raw trajectory for {key}")
        row = dict(zip(KEYS, key))
        row.update(verdict.as_dict())
        row.update(
            source=source,
            stored_weak_auc_gap=stored_auc,
            stored_weak_hit=float(stored["weak_hitting_time"]),
            stored_phase=str(stored["phase"]),
            stored_regime_class=str(stored.get("regime_class", "")),
        )
        if not math.isclose(
            verdict.weak_hit, row["stored_weak_hit"], rel_tol=1e-7, abs_tol=1e-6
        ) and not (
            math.isinf(verdict.weak_hit) and math.isinf(row["stored_weak_hit"])
        ):
            raise ValueError(f"Stored weak first hit disagrees with raw trajectory for {key}")
        rows.append(row)
    if len(rows) != len(summary):
        raise ValueError("Raw/summary run counts disagree.")
    return pd.DataFrame(rows)


def _count(group: pd.DataFrame, column: str) -> int:
    return int(group[column].eq(True).sum())  # noqa: E712


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be absent or empty; do not overwrite an audit.")
    frames = [audit_run(name, raw_path, summary_path, beta) for name, raw_path, summary_path, beta in SOURCES]
    combined = pd.concat(frames, ignore_index=True)
    aggregates: list[dict] = []
    for key, group in combined.groupby(["source", "model_kind", "regime", "lag_separation"], sort=True):
        aggregates.append({
            "source": key[0], "model_kind": key[1], "regime": key[2],
            "lag_separation": int(key[3]), "runs": len(group),
            "weak_gate_pass": _count(group, "weak_gate"),
            "initial_match": _count(group, "initial_match"),
            "at_hit_certificates": _count(group, "at_hit_certificate"),
            "any_time_certificates": _count(group, "any_time_certificate"),
            "positive_auc": int(group.signed_weak_auc_deficit.gt(0).sum()),
            "mean_auc": float(group.signed_weak_auc_deficit.mean()),
            "stored_phases": group.stored_phase.value_counts().to_dict(),
            "stored_regime_classes": group.stored_regime_class.value_counts().to_dict(),
        })
    inputs = {
        name: {
            "trajectories_sha256": _sha256(raw_path / "trajectories.csv"),
            "summary_sha256": _sha256(summary_path / "summary.csv"),
            "beta": beta,
            "historical_source_replayable": False,
        }
        for name, raw_path, summary_path, beta in SOURCES
    }
    args.output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output / "run_level.csv", index=False)
    report = {
        "scope": "Recalculation from extant local historical CSVs; no training replay",
        "interpolation": "piecewise-linear between logged checkpoints",
        "gap_tolerance": 1e-6,
        "initial_tolerance": 1e-5,
        "inputs": inputs,
        "aggregates": aggregates,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(aggregates, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
