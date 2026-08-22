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
from .theory import (
    counterfactual_drift_correction,
    fixed_geometry_susceptibility,
    projected_statistics,
)
from .utils import seed_everything


def _snapshot_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached, independent copy of the model parameters.

    ``clone()`` is required, not optional.  On CPU ``tensor.detach().cpu()``
    returns a tensor that *shares storage* with the live parameter, so a snapshot
    without the clone is silently invalidated by any later in-place mutation --
    including ``load_state_dict``.  The sequential paired trainer resets the model
    between conditions, so an aliased both-feature snapshot ended up holding the
    weak-only parameters instead.
    """
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


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
    identity_absolute_error = float(np.linalg.norm(drift - direct))
    # A one-sided denominator makes harmless roundoff look arbitrarily large
    # when the direct drift is near zero.  The symmetric scale remains a true
    # relative error while keeping zero-drift checks numerically meaningful.
    identity_denominator = max(
        float(np.linalg.norm(direct)), float(np.linalg.norm(drift)), 1e-8
    )
    identity_error = identity_absolute_error / identity_denominator
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
        "projected_identity_absolute_error": identity_absolute_error,
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
    return history, _snapshot_state(model)


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
    if str(mitigation.get("method", "erm")) == "counterfactual_drift":
        return _train_paired_counterfactual_drift(
            model_config,
            both,
            weak,
            training,
            mitigation,
            seed=seed,
            kind=kind,
        )

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


def _train_paired_counterfactual_drift(
    model_config: Mapping[str, object],
    both: SyntheticBatch,
    weak: SyntheticBatch,
    training: Mapping[str, object],
    mitigation: Mapping[str, object],
    *,
    seed: int,
    kind: str | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, torch.Tensor]]]:
    """Train a both-feature model against a simultaneous weak-only shadow.

    The both-feature update is the minimum-norm correction of the ERM update
    that matches the shadow model's instantaneous weak drift while preserving
    the both-feature strong drift.  Weight decay and gradient clipping are
    rejected because they would invalidate that guarantee after correction.
    """
    if not bool(training.get("full_batch", True)):
        raise NotImplementedError("Counterfactual drift correction requires full_batch: true.")
    if float(training.get("weight_decay", 0.0)) != 0.0:
        raise ValueError("counterfactual_drift requires weight_decay: 0 to preserve its guarantee.")
    if training.get("gradient_clip") is not None:
        raise ValueError("counterfactual_drift is incompatible with gradient_clip.")

    seed_everything(seed)
    device = both.x.device
    both_model = build_model(model_config, kind=kind).to(device)
    weak_model = build_model(model_config, kind=kind).to(device)
    weak_model.load_state_dict(copy.deepcopy(both_model.state_dict()))

    learning_rate = float(training.get("learning_rate", 0.01))
    steps = int(training.get("steps", 1000))
    log_every = int(training.get("log_every", 10))
    both_optimizer = torch.optim.SGD(both_model.parameters(), learning_rate)
    weak_optimizer = torch.optim.SGD(weak_model.parameters(), learning_rate)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for step in range(steps + 1):
        weak_stats = projected_statistics(
            weak_model, weak, compute_direct_drift=True, create_graph=False
        )
        if weak_stats.direct_drift is None:  # pragma: no cover - defensive
            raise RuntimeError("Weak-only target drift was not computed.")
        correction = counterfactual_drift_correction(
            both_model,
            both,
            weak_stats.direct_drift[1],
            feasibility_epsilon=float(mitigation.get("feasibility_epsilon", 1e-12)),
            max_alpha=(
                None
                if mitigation.get("max_alpha") is None
                else float(mitigation["max_alpha"])
            ),
        )

        should_log = step % log_every == 0 or step == steps
        if should_log:
            elapsed = time.perf_counter() - started
            both_row = _diagnostic_row(
                both_model,
                both,
                step,
                learning_rate,
                compute_direct_drift=False,
            )
            both_row.update(
                {
                    "wall_seconds": elapsed,
                    "cdc_target_weak_drift": float(correction.target_weak_drift),
                    "cdc_weak_drift_before": float(correction.weak_drift_before.detach()),
                    "cdc_weak_drift_after": float(correction.weak_drift_after.detach()),
                    "cdc_strong_drift_before": float(correction.strong_drift_before.detach()),
                    "cdc_strong_drift_after": float(correction.strong_drift_after.detach()),
                    "cdc_deficit": float(correction.deficit.detach()),
                    "cdc_alpha": float(correction.alpha.detach()),
                    "cdc_protected_norm_sq": float(correction.protected_norm_sq.detach()),
                    "cdc_correction_norm": float(correction.correction_norm.detach()),
                    "cdc_feasible": correction.feasible,
                    "cdc_target_met": bool(
                        correction.weak_drift_after.detach()
                        >= correction.target_weak_drift.detach() - 1e-6
                    ),
                    "cdc_strong_drift_change": float(
                        (correction.strong_drift_after - correction.strong_drift_before).detach()
                    ),
                }
            )
            weak_row = _diagnostic_row(
                weak_model,
                weak,
                step,
                learning_rate,
                compute_direct_drift=False,
            )
            weak_row["wall_seconds"] = elapsed
            history.extend((both_row, weak_row))

        if step == steps:
            break

        both_optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(both_model.parameters(), correction.gradients):
            parameter.grad = gradient.detach().clone()
        both_optimizer.step()

        weak_optimizer.zero_grad(set_to_none=True)
        weak_loss, _ = training_objective(weak_model, weak, {"method": "erm"})
        weak_loss.backward()
        weak_optimizer.step()

    states = {"both": _snapshot_state(both_model), "weak_only": _snapshot_state(weak_model)}
    return history, states
