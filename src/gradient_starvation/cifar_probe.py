"""E4-lite: matched balanced linear-softmax probes on frozen CIFAR encoders."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .cifar_matched import CIFAREvaluationDataset, weights_sha256
from .cifar_protocol import balanced_brier_skill


@torch.no_grad()
def encoder_features(model: nn.Module, dataset: Dataset, device: torch.device,
                     *, batch_size: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract in source-image order without training or changing the encoder."""
    model.eval()
    features, labels = [], []
    for inputs, target in DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0):
        features.append(model.encoder(inputs.to(device)).cpu())
        labels.append(target.cpu())
    if not features:
        raise ValueError("Probe split must be nonempty.")
    return torch.cat(features), torch.cat(labels)


def _head_metrics(head: nn.Module, features: torch.Tensor,
                  labels: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        logits = head(features)
        result = {
            "accuracy": float((logits.argmax(1) == labels).float().mean()),
            "brier_skill": float(balanced_brier_skill(logits, labels)),
            "cross_entropy": float(F.cross_entropy(logits, labels)),
        }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite linear-probe diagnostic.")
    return result


def fit_balanced_linear_probe(
    fit: tuple[torch.Tensor, torch.Tensor],
    validation: tuple[torch.Tensor, torch.Tensor],
    evaluation: tuple[torch.Tensor, torch.Tensor],
    *, classes: int = 10,
    regularization_grid: tuple[float, ...] = (0.0, 0.001, 0.01),
    max_iter: int = 100,
) -> dict:
    """Zero-initialized softmax heads; choose L2 using validation only.

    Feature mean/std are fitted on the probe-fit split only. Each candidate
    uses the same zero initialization and fixed LBFGS iteration budget.
    """
    x_fit, y_fit = fit
    x_val, y_val = validation
    x_eval, y_eval = evaluation
    if (not all(x.ndim == 2 and y.ndim == 1 and len(x) == len(y)
                for x, y in (fit, validation, evaluation))
            or any(len(x) == 0 for x in (x_fit, x_val, x_eval))
            or len({x.shape[1] for x in (x_fit, x_val, x_eval)}) != 1):
        raise ValueError("Probe splits need nonempty matching feature dimensions.")
    if not regularization_grid or any(not math.isfinite(r) or r < 0
                                      for r in regularization_grid) or max_iter <= 0:
        raise ValueError("Invalid probe regularization grid or iteration budget.")
    if any(not torch.isfinite(x).all() for x in (x_fit, x_val, x_eval)):
        raise ValueError("Nonfinite encoder features.")
    for labels in (y_fit, y_val, y_eval):
        counts = torch.bincount(labels.long(), minlength=classes)
        if len(counts) != classes or (counts == 0).any() or not torch.all(counts == counts[0]):
            raise ValueError("Every probe split must have equal positive class counts.")
    mean = x_fit.mean(0)
    std = x_fit.std(0, unbiased=False).clamp_min(1e-6)
    x_fit = (x_fit - mean) / std
    x_val = (x_val - mean) / std
    x_eval = (x_eval - mean) / std
    best: tuple[float, float, dict] | None = None
    candidates = []
    for reg in sorted(set(regularization_grid)):
        head = nn.Linear(x_fit.shape[1], classes)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        optimizer = torch.optim.LBFGS(head.parameters(), lr=1.0,
                                      max_iter=max_iter, line_search_fn="strong_wolfe")
        closure_calls = 0

        def closure() -> torch.Tensor:
            nonlocal closure_calls
            closure_calls += 1
            optimizer.zero_grad(set_to_none=True)
            logits = head(x_fit)
            objective = F.cross_entropy(logits, y_fit)
            objective = objective + 0.5 * reg * head.weight.square().sum()
            objective.backward()
            return objective

        optimizer.step(closure)
        fit_metrics = _head_metrics(head, x_fit, y_fit)
        val_metrics = _head_metrics(head, x_val, y_val)
        record = {
            "l2": reg,
            "probe_fit": fit_metrics,
            "validation": val_metrics,
            "closure_calls": closure_calls,
        }
        candidates.append(record)
        rank = (val_metrics["cross_entropy"], reg)
        if best is None or rank < best[:2]:
            best = (rank[0], rank[1], copy.deepcopy(head.state_dict()))
    assert best is not None
    selected = nn.Linear(x_fit.shape[1], classes)
    selected.load_state_dict(best[2])
    selected.zero_grad(set_to_none=True)
    selected_objective = (F.cross_entropy(selected(x_fit), y_fit) +
                          0.5 * best[1] * selected.weight.square().sum())
    selected_objective.backward()
    gradient_norm = math.sqrt(sum(
        float(parameter.grad.detach().square().sum())
        for parameter in selected.parameters()
    ))
    if not math.isfinite(gradient_norm):
        raise ValueError("Nonfinite selected linear-head gradient.")
    return {
        "selected_l2": best[1],
        "regularization_grid": list(sorted(set(regularization_grid))),
        "max_iter": max_iter,
        "fit_size": len(y_fit),
        "validation_size": len(y_val),
        "evaluation_size": len(y_eval),
        "feature_dimension": x_fit.shape[1],
        "feature_standardization": "mean/std from probe_fit only; std floor 1e-6",
        "selected_fit_gradient_norm": gradient_norm,
        "candidates": candidates,
        "evaluation": _head_metrics(selected, x_eval, y_eval),
    }


def evaluate_e4_lite(both: nn.Module, weak: nn.Module, base: object,
                     splits: dict[str, np.ndarray], *, block_seed: int,
                     amplitude: float, device: torch.device,
                     regularization_grid: tuple[float, ...] = (0.0, 0.001, 0.01),
                     max_iter: int = 100) -> dict:
    """Compare original and fresh heads on the same neutral evaluation IDs."""
    common = dict(base=base, block_seed=block_seed, amplitude=amplitude,
                  mode="neutral")
    datasets = {
        name: CIFAREvaluationDataset(indices=splits[name], **common)
        for name in ("probe_fit", "validation", "diagnostic_evaluation")
    }
    before = [weights_sha256(model) for model in (both, weak)]
    result = {}
    for condition, model in (("both", both), ("weak_only", weak)):
        features = {name: encoder_features(model, data, device)
                    for name, data in datasets.items()}
        result[condition] = fit_balanced_linear_probe(
            features["probe_fit"], features["validation"],
            features["diagnostic_evaluation"],
            regularization_grid=regularization_grid, max_iter=max_iter,
        )
    if before != [weights_sha256(model) for model in (both, weak)]:
        raise AssertionError("E4-lite modified a supposedly frozen encoder/model.")
    result["fresh_head_neutral_accuracy_gap_B_minus_W"] = (
        result["both"]["evaluation"]["accuracy"] -
        result["weak_only"]["evaluation"]["accuracy"]
    )
    return result
