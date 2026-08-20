from __future__ import annotations

import copy
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
    for method in mitigation.get("methods", ["erm"]):
        for seed in training.get("seeds", [0, 1, 2]):
            seed_everything(int(seed))
            model = copy.deepcopy(base_model)
            model.load_state_dict(initial_state)
            model.to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(training.get("learning_rate", 1e-4)),
                weight_decay=float(training.get("weight_decay", 1e-4)),
            )
            for epoch in range(int(training.get("epochs", 30))):
                model.train()
                loss_sum, count, chi_sum = 0.0, 0, 0.0
                for images, labels, metadata in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    background = metadata[:, background_index].long().to(device)
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
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(labels)
                    chi_sum += float(chi.detach()) * len(labels)
                    count += len(labels)
                validation = _evaluate(model, val_loader, device, background_index)
                trajectories.append({
                    "method": method, "seed": int(seed), "epoch": epoch,
                    "train_loss": loss_sum / count, "mean_chi": chi_sum / count,
                    **{f"val_{key}": value for key, value in validation.items()},
                })
                pd.DataFrame(trajectories).to_csv(run_dir / "trajectories.csv", index=False)
            test = _evaluate(model, test_loader, device, background_index)
            summaries.append({"method": method, "seed": int(seed), **test})
            pd.DataFrame(summaries).to_csv(run_dir / "summary.csv", index=False)
    return run_dir

