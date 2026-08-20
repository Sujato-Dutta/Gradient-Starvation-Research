from __future__ import annotations

import copy
import time
from dataclasses import asdict
from typing import Any, Mapping

import numpy as np
import torch

from .data.synthetic import SyntheticBatch
from .losses import training_objective
from .metrics import gsi5
from .models.recurrent import RecurrentBinaryClassifier, build_model, synthetic_logits
from .theory import fixed_geometry_susceptibility, projected_statistics
from .utils import seed_everything


def _diagnostic_row(
    model: RecurrentBinaryClassifier,
    batch: SyntheticBatch,
    step: int,
    learning_rate: float,
    *,
    compute_direct_drift: bool,
) -> dict[str, Any]:
    stats = projected_statistics(
        model, batch, compute_direct_drift=compute_direct_drift, create_graph=False
    )
    margins = stats.margins.detach().cpu().numpy()
    weights = stats.sigmoid_weights.detach().cpu().numpy()
    geometry = stats.geometry.detach().cpu().numpy()
    sensitivity = stats.sensitivity.detach().cpu().numpy()
    field = stats.field.detach().cpu().numpy()
    mode = stats.mode.detach().cpu().numpy()
    drift = stats.predicted_drift.detach().cpu().numpy()
    direct = (
        stats.direct_drift.detach().cpu().numpy()
        if stats.direct_drift is not None
        else np.full(2, np.nan)
    )
    with torch.no_grad():
        logits = synthetic_logits(model, batch)
        accuracy = ((logits >= 0).long() == batch.y).float().mean().item()
    identity_denominator = max(float(np.linalg.norm(direct)), 1e-12)
    identity_error = float(np.linalg.norm(drift - direct) / identity_denominator)
    return {
        "step": step,
        "tau": step * learning_rate,
        "condition": batch.condition,
        "loss": float(stats.loss.detach()),
        "accuracy": accuracy,
        "m_s": float(mode[0]),
        "m_w": float(mode[1]),
        "drift_s": float(drift[0]),
        "drift_w": float(drift[1]),
        "direct_drift_s": float(direct[0]),
        "direct_drift_w": float(direct[1]),
        "projected_identity_relative_error": identity_error,
        "g_s": float(field[0]),
        "g_w": float(field[1]),
        "A_ss": float(sensitivity[0, 0]),
        "A_sw": float(sensitivity[0, 1]),
        "A_ww": float(sensitivity[1, 1]),
        "G_ss": float(geometry[0, 0]),
        "G_sw": float(geometry[0, 1]),
        "G_ww": float(geometry[1, 1]),
        "chi_w_from_s": float(fixed_geometry_susceptibility(stats).detach()),
        "margin_mean": float(margins.mean()),
        "margin_std": float(margins.std()),
        "margin_q10": float(np.quantile(margins, 0.10)),
        "margin_q50": float(np.quantile(margins, 0.50)),
        "margin_q90": float(np.quantile(margins, 0.90)),
        "gsi5": gsi5(weights),
    }


def train_single(
    model: RecurrentBinaryClassifier,
    batch: SyntheticBatch,
    training: Mapping[str, object],
    mitigation: Mapping[str, object],
    *,
    identity_steps: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    steps = int(training.get("steps", 1000))
    learning_rate = float(training.get("learning_rate", 0.01))
    log_every = int(training.get("log_every", 10))
    if not bool(training.get("full_batch", True)):
        raise NotImplementedError("Controlled synthetic experiments require full_batch: true.")
    optimizer = torch.optim.SGD(
        model.parameters(), learning_rate, weight_decay=float(training.get("weight_decay", 0.0))
    )
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(steps + 1):
        should_log = step % log_every == 0 or step == steps
        if should_log:
            row = _diagnostic_row(
                model,
                batch,
                step,
                learning_rate,
                compute_direct_drift=step <= identity_steps,
            )
            row["wall_seconds"] = time.perf_counter() - started
            history.append(row)
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss, _ = training_objective(model, batch, mitigation)
        loss.backward()
        gradient_clip = training.get("gradient_clip")
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
        optimizer.step()
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    return history, state


def train_paired(
    model_config: Mapping[str, object],
    both: SyntheticBatch,
    weak: SyntheticBatch,
    training: Mapping[str, object],
    mitigation: Mapping[str, object],
    *,
    seed: int,
    kind: str | None = None,
    identity_steps: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, torch.Tensor]]]:
    seed_everything(seed)
    device = both.x.device
    model = build_model(model_config, kind=kind).to(device)
    initial_state = copy.deepcopy(model.state_dict())
    both_history, both_state = train_single(
        model, both, training, mitigation, identity_steps=identity_steps
    )
    model.load_state_dict(initial_state)
    weak_history, weak_state = train_single(
        model, weak, training, mitigation, identity_steps=identity_steps
    )
    return both_history + weak_history, {"both": both_state, "weak_only": weak_state}
