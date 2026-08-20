from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from .data.synthetic import SyntheticBatch
from .models.recurrent import RecurrentBinaryClassifier, synthetic_logits
from .theory import fixed_geometry_susceptibility, projected_statistics


def training_objective(
    model: RecurrentBinaryClassifier,
    batch: SyntheticBatch,
    mitigation: Mapping[str, object],
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = synthetic_logits(model, batch)
    ce = F.binary_cross_entropy_with_logits(logits, batch.y.float())
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
    elif method != "erm":
        raise ValueError(f"Unknown mitigation method: {method}")

    return ce + regularizer, {
        "ce_loss": float(ce.detach()),
        "regularizer": float(regularizer.detach()),
        "penalty_susceptibility": susceptibility,
    }
