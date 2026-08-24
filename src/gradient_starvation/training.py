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
    ProjectedStatistics,
    crossover_decomposition,
    drift_correction,
    equal_time_drift_difference,
    fixed_geometry_susceptibility,
    projected_statistics,
)
from .utils import seed_everything

#: Mitigation methods that require a matched weak-only shadow model, mapped to the
#: protection rule they apply.  All share the same causal target, so a comparison
#: across them isolates what is protected rather than what is aimed at.
_SHADOW_METHODS = {
    "counterfactual_drift": "strong_response",
    "loss_gradient_projection": "loss_gradient",
    "unconstrained_rescue": "unconstrained",
    "bloop": "bloop",
    "pcgrad": "pcgrad",
}


def _reject_mismatched_diagnostics(mitigation: Mapping[str, object]) -> None:
    """Refuse to log cross-entropy diagnostics for a run trained on another objective.

    ``projected_statistics`` computes the CE field, CE sensitivity, margin law and
    GSI-5.  Under ``objective: mse`` the parameters follow a different flow, so every
    one of those columns would describe a loss that is not being minimized -- silently,
    and under the same column names. Blocking is better than emitting mislabelled
    numbers; making the diagnostics objective-aware is the real fix and is not done
    here.
    """
    objective = str(mitigation.get("objective", "cross_entropy"))
    if objective != "cross_entropy":
        raise NotImplementedError(
            f"Diagnostic logging is cross-entropy specific, so training with "
            f"objective={objective!r} is refused rather than reported. The projected "
            "field, sensitivity, margin statistics and GSI-5 are all defined by the "
            "logistic loss; under another objective they would describe a loss that is "
            "not being minimized. The MSE objective itself is available through "
            "losses.base_objective for use outside the diagnostic pipeline."
        )


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
    return _row_from_statistics(stats, model, batch, step, learning_rate)


def _row_from_statistics(
    stats: ProjectedStatistics,
    model: RecurrentBinaryClassifier,
    batch: SyntheticBatch,
    step: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Render one diagnostic row from already-computed projected statistics.

    Split out of :func:`_diagnostic_row` so that callers needing the statistics
    themselves -- the lockstep trainer, which also forms the paired crossover
    decomposition -- do not have to recompute them.
    """
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
    projection_residual = (
        stats.projected_drift_residual.detach().cpu().numpy()
        if stats.projected_drift_residual is not None
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
        "projected_drift_residual_s": float(projection_residual[0]),
        "projected_drift_residual_w": float(projection_residual[1]),
        "mode_residual_rms": float(stats.mode_residual_rms.detach()),
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
    _reject_mismatched_diagnostics(mitigation)
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
                compute_direct_drift=(
                    bool(training.get("exact_response_drift", False))
                    or step <= identity_steps
                ),
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
    if str(mitigation.get("method", "erm")) in _SHADOW_METHODS:
        return _train_paired_counterfactual_drift(
            model_config,
            both,
            weak,
            training,
            mitigation,
            seed=seed,
            kind=kind,
        )
    if str(training.get("paired_mode", "sequential")) == "lockstep":
        return train_paired_lockstep(
            model_config,
            both,
            weak,
            training,
            mitigation,
            seed=seed,
            kind=kind,
            identity_steps=identity_steps,
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


def train_paired_lockstep(
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
    """Advance the both-feature and weak-only models together.

    Functionally equivalent to :func:`train_paired` in ``sequential`` mode -- both
    conditions get independent updates from a shared initialization -- but because
    the two models exist simultaneously, the paired crossover decomposition can be
    evaluated at every logged step.  The sequential path cannot do this: it
    finishes the both-feature run before the weak-only model exists.

    The equivalence is asserted by
    ``tests/test_training.py::test_lockstep_reproduces_sequential_erm_trajectories``.
    If that test ever fails, the divergence is a finding to investigate, not a
    tolerance to loosen.

    RNG note: this builds two models where the sequential path builds one, so it
    consumes an extra initialization draw.  That is harmless because the second
    draw is immediately overwritten by the shared initial state and no randomness
    is consumed during full-batch training.  The order matters, though -- the
    both-feature model must be constructed first so it receives exactly the draw
    the sequential path would have given it.
    """
    method = str(mitigation.get("method", "erm"))
    if method == "counterfactual_drift":
        raise ValueError(
            "counterfactual_drift already trains in lockstep; call train_paired "
            "without paired_mode: lockstep."
        )
    if not bool(training.get("full_batch", True)):
        raise NotImplementedError("Controlled synthetic experiments require full_batch: true.")
    _reject_mismatched_diagnostics(mitigation)

    seed_everything(seed)
    device = both.x.device
    both_model = build_model(model_config, kind=kind).to(device)
    shared_initial = copy.deepcopy(both_model.state_dict())
    weak_model = build_model(model_config, kind=kind).to(device)
    weak_model.load_state_dict(copy.deepcopy(shared_initial))

    steps = int(training.get("steps", 1000))
    learning_rate = float(training.get("learning_rate", 0.01))
    log_every = int(training.get("log_every", 10))
    weight_decay = float(training.get("weight_decay", 0.0))
    gradient_clip = training.get("gradient_clip")
    both_optimizer = torch.optim.SGD(
        both_model.parameters(), learning_rate, weight_decay=weight_decay
    )
    weak_optimizer = torch.optim.SGD(
        weak_model.parameters(), learning_rate, weight_decay=weight_decay
    )

    both_history: list[dict[str, Any]] = []
    weak_history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for step in range(steps + 1):
        should_log = step % log_every == 0 or step == steps
        if should_log:
            elapsed = time.perf_counter() - started
            direct = (
                bool(training.get("exact_response_drift", False))
                or step <= identity_steps
            )
            both_stats = projected_statistics(
                both_model, both, compute_direct_drift=direct, create_graph=False
            )
            weak_stats = projected_statistics(
                weak_model, weak, compute_direct_drift=direct, create_graph=False
            )
            both_row = _row_from_statistics(
                both_stats, both_model, both, step, learning_rate
            )
            weak_row = _row_from_statistics(
                weak_stats, weak_model, weak, step, learning_rate
            )
            crossover = crossover_decomposition(both_stats, weak_stats, both.z_w)
            equal_time = equal_time_drift_difference(both_stats, weak_stats)
            both_row.update(
                {
                    "wall_seconds": elapsed,
                    # Matched-state family: exact three-term identity, but NOT the
                    # derivative of the equal-time response gap.
                    "d_w_matched": float(crossover.d_w.detach()),
                    "t_geom": float(crossover.t_geom.detach()),
                    "s_ce": float(crossover.s_ce.detach()),
                    "t_geom_self": float(crossover.geometry_self_term.detach()),
                    "t_geom_cross": float(crossover.cross_transport_term.detach()),
                    "decomposition_reconstruction_error": float(
                        crossover.reconstruction_error.detach()
                    ),
                    "matched_weak_only_m_w": float(weak_stats.mode[1].detach()),
                    # Equal-time projected family.  The two additive orderings
                    # reconstruct this ``Gg`` difference exactly.  For nonlinear
                    # common nonlinear probe modes it need not equal the true gap derivative.
                    "d_w_equal_time": float(equal_time.d_w_equal_time.detach()),
                    "d_w_equal_time_projected": float(
                        equal_time.d_w_equal_time.detach()
                    ),
                    # Universal direct-autograd derivative from Theorem A.1. E-NL
                    # publication configurations require this finite value and use
                    # it, rather than the projected value, for crossover timing.
                    "d_w_equal_time_exact": (
                        float(equal_time.exact_d_w_equal_time.detach())
                        if equal_time.exact_d_w_equal_time is not None
                        else float("nan")
                    ),
                    "equal_time_projection_residual": (
                        float(equal_time.projection_residual.detach())
                        if equal_time.projection_residual is not None
                        else float("nan")
                    ),
                    "equal_time_cross_transport": float(
                        equal_time.cross_transport.detach()
                    ),
                    # Both exact orderings, because attribution between geometry and
                    # field is not unique and they can disagree on dominance.
                    "equal_time_geometry_a": float(
                        equal_time.geometry_difference_a.detach()
                    ),
                    "equal_time_field_a": float(equal_time.field_difference_a.detach()),
                    "equal_time_geometry_b": float(
                        equal_time.geometry_difference_b.detach()
                    ),
                    "equal_time_field_b": float(equal_time.field_difference_b.detach()),
                    "equal_time_reconstruction_error_a": float(
                        equal_time.reconstruction_error_a.detach()
                    ),
                    "equal_time_reconstruction_error_b": float(
                        equal_time.reconstruction_error_b.detach()
                    ),
                    "equal_time_dominance_ordering_invariant": (
                        equal_time.dominance_is_ordering_invariant
                    ),
                }
            )
            weak_row["wall_seconds"] = elapsed
            both_history.append(both_row)
            weak_history.append(weak_row)
        if step == steps:
            break

        for model, batch, optimizer in (
            (both_model, both, both_optimizer),
            (weak_model, weak, weak_optimizer),
        ):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = training_objective(model, batch, mitigation)
            loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            optimizer.step()

    states = {"both": _snapshot_state(both_model), "weak_only": _snapshot_state(weak_model)}
    # Concatenated in the same order the sequential path returns, so the two are
    # drop-in interchangeable for every downstream consumer.
    return both_history + weak_history, states


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

    When feasible and uncapped, the both-feature update is the unique minimum-norm
    correction of the ERM update that matches the shadow model's instantaneous weak
    drift while preserving the both-feature strong drift. Infeasible directions and
    binding caps are logged and may fail target attainment; instantaneous strong-
    drift preservation still holds. Cross-entropy diagnostics, zero weight decay,
    and no gradient clipping are required because post-correction transformations
    would invalidate the stated contract.
    """
    _reject_mismatched_diagnostics(mitigation)
    method = str(mitigation.get("method", "counterfactual_drift"))
    constraint = _SHADOW_METHODS[method]
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
    # Only the Bloop family carries state across steps; the others are memoryless.
    rescue_state: list[torch.Tensor] | None = (
        [torch.zeros_like(parameter) for parameter in both_model.parameters()]
        if constraint == "bloop"
        else None
    )

    for step in range(steps + 1):
        weak_stats = projected_statistics(
            weak_model, weak, compute_direct_drift=True, create_graph=False
        )
        if weak_stats.direct_drift is None:  # pragma: no cover - defensive
            raise RuntimeError("Weak-only target drift was not computed.")
        correction = drift_correction(
            both_model,
            both,
            weak_stats.direct_drift[1],
            constraint=constraint,
            feasibility_epsilon=float(mitigation.get("feasibility_epsilon", 1e-12)),
            max_alpha=(
                None
                if mitigation.get("max_alpha") is None
                else float(mitigation["max_alpha"])
            ),
            rescue_state=rescue_state,
            ema_decay=float(mitigation.get("ema_decay", 0.9)),
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
