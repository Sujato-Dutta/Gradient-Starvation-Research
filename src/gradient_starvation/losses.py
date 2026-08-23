from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from .data.synthetic import SyntheticBatch
from .models.recurrent import RecurrentBinaryClassifier, synthetic_logits
from .theory import fixed_geometry_susceptibility, projected_statistics


def base_objective(
    model: RecurrentBinaryClassifier,
    batch: SyntheticBatch,
    objective: str = "cross_entropy",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(logits, loss)`` for the requested fitting objective.

    ``cross_entropy``
        Binary logistic loss.  This is the objective the starvation theory is
        stated for, because the margin gate ``sigma(-z . m)`` is what produces
        cross-entropy suppression of the weak mode.
    ``mse``
        Squared error against the signed labels, ``mean (f(x) - y_signed)^2 / 2``.
        Provided so that regimes overlapping known low-rank RNN learning theory can
        be cross-checked, and because it has no saturating margin gate: the weak
        drift under MSE cannot be suppressed by margin growth, only by geometry.
        That makes it a useful contrast for the crossover mechanism.
    """
    logits = synthetic_logits(model, batch)
    if objective == "cross_entropy":
        return logits, F.binary_cross_entropy_with_logits(logits, batch.y.float())
    if objective == "mse":
        return logits, 0.5 * (logits - batch.signed_labels).square().mean()
    raise ValueError(
        f"Unknown objective: {objective!r}; expected 'cross_entropy' or 'mse'."
    )


def training_objective(
    model: RecurrentBinaryClassifier,
    batch: SyntheticBatch,
    mitigation: Mapping[str, object],
) -> tuple[torch.Tensor, dict[str, float]]:
    objective = str(mitigation.get("objective", "cross_entropy"))
    logits, ce = base_objective(model, batch, objective)
    method = str(mitigation.get("method", "erm"))
    regularizer = ce.new_zeros(())
    susceptibility = float("nan")

    if method == "spectral_decoupling":
        coefficient = float(
            mitigation.get("spectral_coefficient", mitigation.get("coefficient", 0.0))
        )
        regularizer = 0.5 * coefficient * logits.square().mean()
    elif method == "interaction":
        if batch.condition == "both":
            second_order = str(mitigation.get("mode", "proxy")) == "second_order"
            stats = projected_statistics(model, batch, create_graph=second_order)
            chi = fixed_geometry_susceptibility(stats)
            if not second_order:
                # Keep the loss/data interaction differentiable while treating geometry as a closure.
                detached_geometry = stats.geometry.detach()
                chi = (
                    detached_geometry[1, 0] * stats.sensitivity[0, 0]
                    + detached_geometry[1, 1] * stats.sensitivity[0, 1]
                )
            coefficient = float(mitigation.get("coefficient", 0.0))
            regularizer = coefficient * torch.relu(chi).square()
            susceptibility = float(chi.detach())
    elif method == "counterfactual_drift":
        raise ValueError(
            "counterfactual_drift requires paired both/weak-only training; use train_paired."
        )
    elif method != "erm":
        raise ValueError(f"Unknown mitigation method: {method}")

    return ce + regularizer, {
        "ce_loss": float(ce.detach()),
        "objective": objective,
        "regularizer": float(regularizer.detach()),
        "penalty_susceptibility": susceptibility,
    }
