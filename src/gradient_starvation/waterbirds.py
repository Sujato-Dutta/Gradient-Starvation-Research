from __future__ import annotations

import contextlib
import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .metrics import gsi5
from .utils import create_run_directory, resolve_device, seed_everything


def _setup(config: Mapping[str, Any]):
    try:
        import torchvision.models as models
        import torchvision.transforms as transforms
        from wilds import get_dataset
        from wilds.common.data_loaders import get_eval_loader, get_train_loader
    except ImportError as error:
        raise RuntimeError(
            "Waterbirds dependencies are missing. Install requirements-waterbirds.txt."
        ) from error

    data = config["data"]
    dataset = get_dataset(
        dataset="waterbirds",
        download=bool(data.get("download", False)),
        root_dir=str(data.get("root", "data")),
    )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    batch_size = int(data.get("batch_size", 64))
    workers = int(data.get("num_workers", 4))
    train_loader = get_train_loader(
        "standard", dataset.get_subset("train", transform=train_transform),
        batch_size=batch_size, num_workers=workers,
    )
    validation_loader = get_eval_loader(
        "standard", dataset.get_subset("val", transform=eval_transform),
        batch_size=batch_size * 2, num_workers=workers,
    )
    test_loader = get_eval_loader(
        "standard", dataset.get_subset("test", transform=eval_transform),
        batch_size=batch_size * 2, num_workers=workers,
    )
    architecture = str(config.get("model", {}).get("architecture", "resnet18"))
    pretrained = bool(config.get("model", {}).get("pretrained", True))
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif architecture == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError("Waterbirds supports resnet18 and resnet50.")
    model.fc = torch.nn.Linear(model.fc.in_features, dataset.n_classes)
    background_index = dataset.metadata_fields.index("background")
    return train_loader, validation_loader, test_loader, model, background_index


#: The two CDC-style surrogate rescue arms, whose guarantee requires the raw SGD velocity.
CDC_METHODS = frozenset({"counterfactual_drift_oracle", "counterfactual_drift_modal"})


def _require_raw_velocity_settings(
    methods: list[str], training: Mapping[str, Any]
) -> None:
    """Refuse settings under which the CDC guarantee would not describe the update."""
    optimizer = str(training.get("optimizer", "adamw")).lower()
    if optimizer != "sgd":
        raise ValueError(
            f"Methods {sorted(set(methods) & CDC_METHODS)} require "
            "training.optimizer: sgd, and so does every arm they are compared "
            "against, otherwise the comparison confounds the method with the "
            "optimizer. AdamW's preconditioning, momentum and decoupled weight decay "
            "mean the update taken is not the corrected gradient, so the logged "
            f"strong-drift change would not describe it. Received {optimizer!r}."
        )
    decay = float(training.get("weight_decay", 0.0))
    if decay != 0.0:
        raise ValueError(
            "Counterfactual drift requires training.weight_decay: 0; received "
            f"{decay}. Weight decay perturbs the update after the correction is "
            "computed and so breaks the first-order guarantee."
        )


def _build_optimizer(
    model: torch.nn.Module, training: Mapping[str, Any], *, use_sgd: bool
) -> torch.optim.Optimizer:
    learning_rate = float(training.get("learning_rate", 1e-4))
    if use_sgd or str(training.get("optimizer", "adamw")).lower() == "sgd":
        # Weight decay is already validated to be zero when `use_sgd` is set.
        return torch.optim.SGD(
            model.parameters(), lr=learning_rate,
            weight_decay=0.0 if use_sgd else float(training.get("weight_decay", 0.0)),
        )
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate,
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )


@contextlib.contextmanager
def capture_penultimate_features(model: torch.nn.Module) -> Iterator[list[torch.Tensor]]:
    """Capture the head's input during the ordinary forward pass, via a hook.

    A second forward pass would be wrong here, not merely wasteful.  It runs while
    ``model.train()`` is active, and ``torch.no_grad`` suppresses gradients but **not**
    BatchNorm running-statistic updates, so the modal arm would advance its
    normalization state twice per batch while every baseline advanced it once.  That is
    a different state-update process, and the comparison would no longer be about the
    method.

    Using a forward hook on the head takes the features from the forward pass that
    already happened, so every arm sees exactly one BatchNorm update per batch.
    Captured tensors are detached: the modal coordinates are a *measurement* of internal
    structure, and letting gradients reach the PCA would make the estimated direction
    itself trainable, which is a different method again.
    """
    captured: list[torch.Tensor] = []

    def hook(_module, inputs, _output):
        captured.append(inputs[0].detach())

    handle = model.fc.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def leading_feature_direction(features: torch.Tensor) -> torch.Tensor:
    """Return the leading principal direction of the centred features.

    Deterministic: uses a full SVD rather than ``torch.pca_lowrank``, which draws a
    random projection and so returns a slightly different direction on each call.  The
    correction and its agreement diagnostic must describe the *same* component,
    otherwise the logged agreement does not characterize the direction actually used.
    """
    centred = features - features.mean(dim=0, keepdim=True)
    # `full_matrices=False` keeps this tractable for wide feature blocks.
    _, _, vh = torch.linalg.svd(centred, full_matrices=False)
    return vh[0]


def _margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    correct = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.gather(1, (1 - labels)[:, None]).squeeze(1)
    return correct - other


def _interaction_penalty(
    model: torch.nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    background: torch.Tensor,
    coefficient: float,
    second_order: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    margin = _margin(logits, labels)
    signed_y = labels.mul(2).sub(1).float()
    signed_background = background.mul(2).sub(1).float()
    coordinates = torch.stack((signed_y * signed_background, torch.ones_like(signed_y)), dim=1)
    gram = coordinates.T @ coordinates / len(coordinates)
    mode = torch.linalg.solve(
        gram + 1e-5 * torch.eye(2, device=gram.device), coordinates.T @ margin / len(margin)
    )
    parameters = list(model.fc.parameters())
    mode_gradients = [
        torch.autograd.grad(
            mode[index], parameters, create_graph=second_order, retain_graph=True
        )
        for index in range(2)
    ]
    geometry = torch.stack([
        torch.stack([
            sum((left * right).sum() for left, right in zip(mode_gradients[a], mode_gradients[b]))
            for b in range(2)
        ])
        for a in range(2)
    ])
    if not second_order:
        geometry = geometry.detach()
    q = torch.sigmoid(margin) * torch.sigmoid(-margin)
    sensitivity = torch.einsum("ni,nj,n->ij", coordinates, coordinates, q) / len(margin)
    chi = geometry[1, 0] * sensitivity[0, 0] + geometry[1, 1] * sensitivity[0, 1]
    return coefficient * torch.relu(chi).square(), chi


#: Methods that consume group labels at training time.  They are *oracle*
#: mechanistic probes, not group-agnostic robustness methods, and must never share a
#: comparison column with the latter.
ORACLE_METHODS = frozenset({"counterfactual_drift_oracle", "interaction"})

#: Methods that require no group labels during training.
GROUP_AGNOSTIC_METHODS = frozenset(
    {"erm", "spectral_decoupling", "counterfactual_drift_modal"}
)

WATERBIRDS_METHODS = ORACLE_METHODS | GROUP_AGNOSTIC_METHODS


def information_setting(method: str) -> str:
    """Label a method by the supervision it consumes.

    Emitted as a column on every Waterbirds row so that an oracle result can never
    be tabulated alongside a group-agnostic one without the difference being visible.
    """
    if method in ORACLE_METHODS:
        return "oracle_uses_group_labels"
    if method in GROUP_AGNOSTIC_METHODS:
        return "group_agnostic"
    raise ValueError(f"Unknown Waterbirds method: {method!r}")


def _projected_modes(
    margin: torch.Tensor, coordinates: torch.Tensor, ridge: float = 1e-5
) -> torch.Tensor:
    """Least-squares projection of the margin onto two latent coordinates."""
    gram = coordinates.T @ coordinates / len(coordinates)
    identity = torch.eye(2, device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram + ridge * identity, coordinates.T @ margin / len(margin))


def oracle_feature_coordinates(
    labels: torch.Tensor, background: torch.Tensor
) -> torch.Tensor:
    """Strong/weak latent coordinates built from group labels. **Oracle.**

    The strong coordinate is the label-background agreement ``y_signed * b_signed``,
    which is exactly the spurious cue Waterbirds is built around, and the weak
    coordinate is the constant, capturing the label-only response.  This requires
    the background annotation and is therefore oracle information.
    """
    signed_labels = labels.mul(2).sub(1).float()
    signed_background = background.mul(2).sub(1).float()
    return torch.stack(
        (signed_labels * signed_background, torch.ones_like(signed_labels)), dim=1
    )


def modal_feature_coordinates(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    direction: torch.Tensor | None = None,
) -> torch.Tensor:
    """Strong/weak coordinates estimated from internal structure. **No group labels.**

    The dominant internal mode is taken to be the leading principal direction of the
    centred penultimate features, on the hypothesis that the spurious cue is the
    largest source of representational variance.  The strong coordinate is the
    label-signed projection onto that direction.

    **The weak coordinate is a constant, and that is a surrogate, not an identified
    feature.** Regressing the margin on ``(strong, 1)`` makes the second coefficient an
    *intercept*: the average signed margin not explained by the estimated strong mode.
    It is not a bird-shape response, and nothing here identifies one. Waterbirds can
    therefore only be presented as a surrogate mechanistic probe unless that coordinate
    is independently validated against a known core-feature readout.  The same caveat
    applies to :func:`oracle_feature_coordinates`.

    Pass ``direction`` to reuse a direction already computed by
    :func:`leading_feature_direction`, so that the correction and its diagnostic
    describe the same component.
    """
    if direction is None:
        direction = leading_feature_direction(features)
    centred = features - features.mean(dim=0, keepdim=True)
    projection = centred @ direction
    scale = projection.abs().mean().clamp_min(1e-8)
    signed_labels = labels.mul(2).sub(1).float()
    return torch.stack(
        (signed_labels * projection / scale, torch.ones_like(signed_labels)), dim=1
    )


@torch.no_grad()
def modal_estimator_agreement(
    features: torch.Tensor,
    background: torch.Tensor,
    *,
    direction: torch.Tensor | None = None,
) -> float:
    """Absolute correlation between the modal estimator and the true background.

    Reported as a diagnostic so the group-label-free variant's central assumption is
    measurable rather than assumed.  A low value means the leading component is not
    tracking the spurious cue, and the modal variant's rescue direction is then not
    the causal one.

    Pass the same ``direction`` used by the correction.  Recomputing it here would
    describe a different component whenever the extraction is randomized, so the logged
    agreement would not characterize the direction actually applied.
    """
    if direction is None:
        direction = leading_feature_direction(features)
    centred = features - features.mean(dim=0, keepdim=True)
    projection = centred @ direction
    signed_background = background.mul(2).sub(1).float()
    if projection.std() < 1e-12:
        return float("nan")
    stacked = torch.stack((projection, signed_background))
    return float(torch.corrcoef(stacked)[0, 1].abs())


def constrained_weak_rescue(
    head: torch.nn.Module,
    margin: torch.Tensor,
    coordinates: torch.Tensor,
    target_weak_drift: float,
    loss: torch.Tensor,
    *,
    model: torch.nn.Module | None = None,
    feasibility_epsilon: float = 1e-12,
) -> dict[str, float]:
    """Apply a fixed-head-coordinate Euclidean CDC-style minibatch surrogate.

    The rescue uses the component of ``grad(m_w)`` Euclidean-orthogonal to
    ``grad(m_s)`` in the classifier head's implemented tensor coordinates, so it
    preserves the instantaneous head-coordinate strong-response drift.  This is
    coordinate-dependent and is not invariant under non-isometric
    reparameterization.

    This Waterbirds operation is a minibatch surrogate with an absolute configured
    target.  It is outside the full-batch matched weak-only-shadow theorem used by
    the synthetic CDC trainer and must not be interpreted as that theorem's causal
    target or guarantee.

    The *correction* is confined to the head, matching the existing interaction
    penalty: the protected geometry is defined by head-parameter gradients, and
    extending it to the backbone would change that geometry and need its own
    validation.

    Pass ``model`` so that the **backbone still receives its ordinary ERM gradient**.
    Without it the backbone would be left with ``grad = None`` and silently frozen,
    while every baseline fine-tunes the whole network -- an unfair comparison rather
    than a different method.  ``backbone_updated`` in the returned record states which
    happened.
    """
    if not np.isfinite(feasibility_epsilon) or feasibility_epsilon < 0:
        raise ValueError("feasibility_epsilon must be finite and non-negative.")
    parameters = [p for p in head.parameters() if p.requires_grad]
    head_parameters = {id(p) for p in parameters}
    backbone = (
        [
            p
            for p in model.parameters()
            if p.requires_grad and id(p) not in head_parameters
        ]
        if model is not None
        else []
    )
    if backbone:
        # Ordinary ERM gradient for everything outside the protected head.
        for parameter, gradient in zip(
            backbone,
            torch.autograd.grad(loss, backbone, retain_graph=True, allow_unused=True),
        ):
            parameter.grad = (
                torch.zeros_like(parameter) if gradient is None else gradient.detach().clone()
            )
    mode = _projected_modes(margin, coordinates)
    strong = torch.autograd.grad(mode[0], parameters, retain_graph=True, allow_unused=True)
    weak = torch.autograd.grad(mode[1], parameters, retain_graph=True, allow_unused=True)
    loss_grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    zeros = [torch.zeros_like(p) for p in parameters]
    strong = [z if g is None else g for g, z in zip(strong, zeros)]
    weak = [z if g is None else g for g, z in zip(weak, zeros)]
    loss_grads = [z if g is None else g for g, z in zip(loss_grads, zeros)]

    def inner(left, right):
        return sum((a * b).sum() for a, b in zip(left, right))

    strong_norm_sq = inner(strong, strong)
    # Orthogonality is exact for every mathematically nonzero strong gradient;
    # feasibility_epsilon applies only to whether the rescue can attain its target.
    strong_is_nonzero = strong_norm_sq > 0
    safe_strong_norm_sq = torch.where(
        strong_is_nonzero, strong_norm_sq, torch.ones_like(strong_norm_sq)
    )
    scale = torch.where(
        strong_is_nonzero,
        inner(weak, strong) / safe_strong_norm_sq,
        strong_norm_sq.new_zeros(()),
    )
    protected = [w - scale * s for w, s in zip(weak, strong)]
    velocity = [-g for g in loss_grads]
    weak_before = inner(weak, velocity)
    strong_before = inner(strong, velocity)
    gain = inner(weak, protected)
    deficit = torch.relu(torch.as_tensor(target_weak_drift, device=margin.device) - weak_before)
    alpha = (
        deficit / gain
        if float(gain) > feasibility_epsilon
        else torch.zeros((), device=margin.device)
    )
    corrected = [g - alpha * d for g, d in zip(loss_grads, protected)]
    for parameter, gradient in zip(parameters, corrected):
        parameter.grad = gradient.detach().clone()
    corrected_velocity = [-g for g in corrected]
    return {
        "cdc_alpha": float(alpha),
        "cdc_deficit": float(deficit),
        "cdc_weak_drift_before": float(weak_before),
        "cdc_weak_drift_after": float(inner(weak, corrected_velocity)),
        "cdc_strong_drift_before": float(strong_before),
        "cdc_strong_drift_after": float(inner(strong, corrected_velocity)),
        "cdc_strong_drift_change": float(inner(strong, corrected_velocity) - strong_before),
        "cdc_feasible": bool(float(gain) > feasibility_epsilon or float(deficit) <= 0),
        "backbone_updated": bool(backbone),
    }


@torch.no_grad()
def _evaluate(model, loader, device, background_index: int) -> dict[str, float]:
    model.eval()
    all_correct, all_groups, all_weights = [], [], []
    for images, labels, metadata in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        correct = (logits.argmax(1) == labels).float().cpu()
        background = metadata[:, background_index].long()
        groups = labels.cpu().long() * 2 + background
        all_correct.append(correct)
        all_groups.append(groups)
        all_weights.append(torch.sigmoid(-_margin(logits, labels)).cpu())
    correct, groups = torch.cat(all_correct), torch.cat(all_groups)
    group_accuracies = [float(correct[groups == group].mean()) for group in range(4)]
    return {
        "accuracy": float(correct.mean()),
        "worst_group_accuracy": min(group_accuracies),
        "gsi5": gsi5(torch.cat(all_weights).numpy()),
        **{f"group_{group}_accuracy": value for group, value in enumerate(group_accuracies)},
    }


def run_waterbirds(config: dict[str, Any]) -> Path:
    run_dir = create_run_directory(config)
    device = resolve_device(str(config.get("experiment", {}).get("device", "auto")))
    train_loader, val_loader, test_loader, base_model, background_index = _setup(config)
    initial_state = copy.deepcopy(base_model.state_dict())
    training, mitigation = config["training"], config["mitigation"]
    trajectories: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    methods = list(mitigation.get("methods", ["erm"]))
    unknown = sorted(set(methods) - WATERBIRDS_METHODS)
    if unknown:
        raise ValueError(
            f"Unknown Waterbirds methods {unknown}; expected a subset of "
            f"{sorted(WATERBIRDS_METHODS)}."
        )
    # CDC's Result 1 is stated for the raw corrected velocity. AdamW applies adaptive
    # preconditioning, momentum and decoupled weight decay, so the update actually
    # taken would not be the corrected gradient and the logged orthogonality would
    # not describe it. The constraint has to bind the WHOLE comparison, not just the
    # CDC arms: running CDC on SGD while baselines run on AdamW would trade a broken
    # guarantee for an optimizer confound.
    raw_velocity_required = bool(set(methods) & CDC_METHODS)
    if raw_velocity_required:
        _require_raw_velocity_settings(methods, training)
    for method in methods:
        for seed in training.get("seeds", [0, 1, 2]):
            seed_everything(int(seed))
            model = copy.deepcopy(base_model)
            model.load_state_dict(initial_state)
            # `initial_state` is shared so that a method comparison at a given seed is
            # paired on initialization, which is the point of the design. But it also
            # means the *classifier head* is identical across seeds, so seed variation
            # covers only data order and augmentation. Re-drawing the head per seed
            # restores across-seed variation without breaking the within-seed pairing,
            # since every method at that seed still starts from the same draw.
            if bool(training.get("reinitialize_head_per_seed", True)):
                model.fc.reset_parameters()
            model.to(device)
            optimizer = _build_optimizer(model, training, use_sgd=raw_velocity_required)
            for epoch in range(int(training.get("epochs", 30))):
                model.train()
                loss_sum, count, chi_sum = 0.0, 0, 0.0
                cdc_records: list[dict[str, float]] = []
                needs_features = method == "counterfactual_drift_modal"
                for images, labels, metadata in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    background = metadata[:, background_index].long().to(device)
                    # One forward pass for every arm. The modal method reads the head's
                    # input from a hook on this pass rather than running a second one,
                    # which would advance BatchNorm statistics twice for that arm alone.
                    if needs_features:
                        with capture_penultimate_features(model) as captured:
                            logits = model(images)
                        features = captured[0]
                    else:
                        logits = model(images)
                    loss = F.cross_entropy(logits, labels)
                    chi = logits.new_zeros(())
                    if method == "spectral_decoupling":
                        loss = loss + 0.5 * float(mitigation.get("spectral_coefficient", 0.01)) * logits.square().mean()
                    elif method == "interaction":
                        penalty, chi = _interaction_penalty(
                            model, logits, labels, background,
                            float(mitigation.get("coefficient", 0.01)),
                            str(mitigation.get("mode", "proxy")) == "second_order",
                        )
                        loss = loss + penalty
                    if method in {
                        "counterfactual_drift_oracle", "counterfactual_drift_modal"
                    }:
                        margin = _margin(logits, labels)
                        if method == "counterfactual_drift_oracle":
                            coordinates = oracle_feature_coordinates(labels, background)
                        else:
                            # One direction, shared by the correction and the
                            # diagnostic, so the logged agreement characterizes the
                            # component actually corrected.
                            direction = leading_feature_direction(features)
                            coordinates = modal_feature_coordinates(
                                features, labels, direction=direction
                            )
                            cdc_records.append(
                                {
                                    "modal_background_agreement": modal_estimator_agreement(
                                        features, background, direction=direction
                                    )
                                }
                            )
                        optimizer.zero_grad(set_to_none=True)
                        # Waterbirds has no exact matched weak-only counterfactual.
                        # The configured value is therefore an absolute surrogate
                        # weak-drift target, not a margin added to the current drift,
                        # and must be reported/interpreted as such.
                        record = constrained_weak_rescue(
                            model.fc, margin, coordinates,
                            target_weak_drift=float(
                                mitigation.get("weak_drift_target", 0.0)
                            ),
                            loss=loss,
                            model=model,
                        )
                        cdc_records.append(record)
                        optimizer.step()
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    loss_sum += float(loss.detach()) * len(labels)
                    chi_sum += float(chi.detach()) * len(labels)
                    count += len(labels)
                validation = _evaluate(model, val_loader, device, background_index)
                aggregated = {
                    key: float(np.mean([r[key] for r in cdc_records if key in r]))
                    for key in {k for r in cdc_records for k in r}
                }
                trajectories.append({
                    "method": method,
                    "information_setting": information_setting(method),
                    "seed": int(seed), "epoch": epoch,
                    "train_loss": loss_sum / count, "mean_chi": chi_sum / count,
                    **aggregated,
                    **{f"val_{key}": value for key, value in validation.items()},
                })
                pd.DataFrame(trajectories).to_csv(run_dir / "trajectories.csv", index=False)
            test = _evaluate(model, test_loader, device, background_index)
            summaries.append({
                "method": method,
                "information_setting": information_setting(method),
                "seed": int(seed),
                **test,
            })
            pd.DataFrame(summaries).to_csv(run_dir / "summary.csv", index=False)
    return run_dir

