"""Local replay checks; ignored historical raw CSVs are absent from clean CI."""

from pathlib import Path

import pandas as pd
import pytest

from gradient_starvation.causal_audit import audit_paired_response


ROOT = Path(__file__).resolve().parents[1]


def _audit_file(raw_path: Path, beta: float) -> pd.DataFrame:
    if not raw_path.is_file():
        pytest.skip("Historical raw trajectories are not included in a clean checkout")
    raw = pd.read_csv(raw_path)
    keys = ["rho", "lag_separation", "regime", "seed"]
    if "model_kind" in raw:
        keys.insert(0, "model_kind")
    rows = []
    for key, group in raw.groupby(keys):
        b = group[group.condition == "both"].sort_values("step")
        w = group[group.condition == "weak_only"].sort_values("step")
        assert b.step.tolist() == w.step.tolist()
        record = dict(zip(keys, key))
        record.update(audit_paired_response(
            b.tau.to_numpy(), b.m_w.to_numpy(), w.m_w.to_numpy(), beta=beta
        ).as_dict())
        rows.append(record)
    return pd.DataFrame(rows)


def test_local_dense_e1_replay_exposes_at_hit_vs_auc_difference():
    frame = _audit_file(
        ROOT / "results/e1_dense_rerun-20260823-085142/trajectories.csv", 0.25
    )
    positive = frame[frame.regime == "positive"]
    by_lag = positive.groupby("lag_separation").agg(
        gate=("weak_gate", "sum"), at_hit=("at_hit_certificate", "sum"),
        any_time=("any_time_certificate", "sum"),
    )
    assert by_lag.loc[0].to_dict() == {"gate": 32, "at_hit": 31, "any_time": 32}
    assert by_lag.loc[2].to_dict() == {"gate": 32, "at_hit": 32, "any_time": 32}
    assert by_lag.loc[4].to_dict() == {"gate": 32, "at_hit": 32, "any_time": 32}
    assert by_lag.loc[8].to_dict() == {"gate": 0, "at_hit": 0, "any_time": 0}


def test_local_nonlinear_replay_separates_tanh_subtypes_from_certificate():
    frame = _audit_file(
        ROOT / "results/enl_tanh_crossover-20260824-141207/trajectories.csv", 0.5
    )
    tanh = frame[frame.model_kind == "tanh"]
    gru = frame[frame.model_kind == "gru"]
    assert len(tanh) == len(gru) == 8
    assert tanh.weak_gate.sum() == 8
    assert tanh.at_hit_certificate.sum() == 8
    assert gru.weak_gate.sum() == 0
    assert gru.at_hit_certificate.sum() == 0
