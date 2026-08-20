"""
gsi_core.py — Gradient Starvation Index: 5 Variants + Training Callback
=========================================================================

Usage:
    from gsi_core import GSICallback

    callback = GSICallback(val_loader, device="cuda", is_nlp=False)

    for epoch in range(num_epochs):
        train_one_epoch(...)
        result = callback.on_epoch_end(epoch, model)

    callback.save("gsi_log.pt")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────
# Result Container
# ─────────────────────────────────────────────────────────────────

@dataclass
class GSIResult:
    epoch: int = -1
    gsi_1: float = 0.0  # gradient energy ratio
    gsi_2: float = 0.0  # margin dispersion
    gsi_3: float = 0.0  # cosine alignment
    gsi_4: float = 0.0  # effective rank
    gsi_5: float = 0.0  # sigmoid concentration
    avg_accuracy: float = 0.0
    worst_group_accuracy: float = 0.0
    accuracy_gap: float = 0.0
    group_accuracies: Dict[int, float] = field(default_factory=dict)
    per_direction_energy: np.ndarray = field(default_factory=lambda: np.array([]))
    margins: np.ndarray = field(default_factory=lambda: np.array([]))
    sigmoid_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    gradient_singular_values: np.ndarray = field(default_factory=lambda: np.array([]))
    compute_seconds: float = 0.0


# ─────────────────────────────────────────────────────────────────
# Feature Extractor — hooks into the last nn.Linear of any model
# ─────────────────────────────────────────────────────────────────

class _FeatureHook:
    def __init__(self, model: nn.Module):
        self.features = None
        self._handle = None
        # Find the last nn.Linear
        self._target = None
        for _, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._target = module
        if self._target is None:
            raise ValueError("No nn.Linear found in model.")

    def attach(self):
        self._handle = self._target.register_forward_hook(self._hook)

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, inp, out):
        self.features = inp[0].detach()


# ─────────────────────────────────────────────────────────────────
# GSI Computer
# ─────────────────────────────────────────────────────────────────

class GSIComputer:
    def __init__(self, num_directions: int = 20, max_samples: int = 2000,
                 device: str = "cuda", is_nlp: bool = False):
        self.num_directions = num_directions
        self.max_samples = max_samples
        self.device = device
        self.is_nlp = is_nlp

    # ── Data collection ──────────────────────────────────────────

    @torch.no_grad()
    def _collect(self, model: nn.Module, dataloader
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        model.eval()
        hook = _FeatureHook(model)
        hook.attach()

        all_feat, all_logit, all_y, all_g = [], [], [], []
        n = 0

        try:
            for batch in dataloader:
                # ── Unpack batch ──
                if self.is_nlp:
                    x_dict, y, meta = batch
                    x_dev = {k: v.to(self.device) for k, v in x_dict.items()}
                    out = model(**x_dev)
                    logits = out.logits if hasattr(out, "logits") else out
                else:
                    x, y = batch[0].to(self.device), batch[1]
                    meta = batch[2] if len(batch) > 2 else None
                    logits = model(x)

                # ── Groups ──
                if meta is not None:
                    g = meta[:, 0] if meta.dim() > 1 else meta
                    all_g.append(g.cpu() if g.is_cuda else g)
                else:
                    all_g.append(torch.full((len(y),), -1, dtype=torch.long))

                all_feat.append(hook.features.cpu())
                all_logit.append(logits.detach().cpu())
                all_y.append(y.cpu() if y.is_cuda else y)

                n += len(y)
                if n >= self.max_samples:
                    break
        finally:
            hook.detach()

        N = min(n, self.max_samples)
        features = torch.cat(all_feat)[:N]
        logits = torch.cat(all_logit)[:N]
        labels = torch.cat(all_y)[:N].long()
        groups = torch.cat(all_g)[:N]

        return features, logits, labels, groups

    # ── Last-layer gradients (analytic, no autograd) ─────────────

    @staticmethod
    def _last_layer_grads(features: torch.Tensor, logits: torch.Tensor,
                          labels: torch.Tensor) -> torch.Tensor:
        N, d = features.shape
        C = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(labels, num_classes=C).float()
        residuals = probs - one_hot                         # [N, C]
        grads = torch.bmm(
            residuals.unsqueeze(2),   # [N, C, 1]
            features.unsqueeze(1)     # [N, 1, d]
        ).reshape(N, C * d)           # [N, C*d]
        return grads

    # ── GSI-1: Gradient Energy Ratio ─────────────────────────────

    def _gsi_1(self, grads: torch.Tensor, features: torch.Tensor
               ) -> Tuple[float, np.ndarray]:
        fc = features - features.mean(0)
        _, S, Vt = torch.linalg.svd(fc, full_matrices=False)
        k = min(self.num_directions, Vt.shape[0])
        proj = features @ Vt[:k].T                          # [N, k]
        gnorm2 = (grads ** 2).sum(1)                         # [N]
        energy = torch.zeros(k)
        for j in range(k):
            energy[j] = (gnorm2 * proj[:, j].abs()).mean()
        e = energy.numpy()
        S_np = S[:k].numpy()
        mask = S_np > S_np.max() * 0.01
        if mask.sum() < 2:
            return 0.0, e
        es = e[mask]
        gsi = 1.0 - es.min() / (es.max() + 1e-12)
        return float(np.clip(gsi, 0, 1)), e

    # ── GSI-2: Margin Dispersion ─────────────────────────────────

    @staticmethod
    def _gsi_2(logits: torch.Tensor, labels: torch.Tensor
               ) -> Tuple[float, np.ndarray]:
        C = logits.shape[1]
        correct = logits[torch.arange(len(labels)), labels]
        mask = F.one_hot(labels, C).bool()
        wrong_max = logits.masked_fill(mask, float("-inf")).max(1).values
        margins = (correct - wrong_max).numpy()
        mu = abs(margins.mean())
        if mu < 1e-8:
            return 0.0, margins
        cv = margins.std() / mu
        gsi = cv ** 2 / (1.0 + cv ** 2)
        return float(np.clip(gsi, 0, 1)), margins

    # ── GSI-3: Cosine Alignment ──────────────────────────────────

    @staticmethod
    def _gsi_3(grads: torch.Tensor) -> float:
        n = min(500, grads.shape[0])
        g = grads[torch.randperm(grads.shape[0])[:n]]
        norms = g.norm(dim=1)
        keep = norms > 1e-10
        if keep.sum() < 10:
            return 0.0
        g = F.normalize(g[keep], dim=1)
        sim = g @ g.T
        m = ~torch.eye(g.shape[0], dtype=torch.bool)
        mean_sim = sim[m].mean().item()
        return float(np.clip((mean_sim + 1) / 2, 0, 1))

    # ── GSI-4: Effective Rank ────────────────────────────────────

    @staticmethod
    def _gsi_4(grads: torch.Tensor) -> Tuple[float, np.ndarray]:
        n = min(1000, grads.shape[0])
        g = grads[torch.randperm(grads.shape[0])[:n]]
        try:
            S = torch.linalg.svdvals(g).numpy()
        except RuntimeError:
            return 0.0, np.array([])
        S = S[S > 1e-10]
        if len(S) < 2:
            return 0.0, S[:50]
        p = S / S.sum()
        ent = -np.sum(p * np.log(p + 1e-12))
        eff = np.exp(ent)
        mx = min(n, grads.shape[1])
        return float(np.clip(1 - eff / mx, 0, 1)), S[:50]

    # ── GSI-5: Sigmoid Concentration ─────────────────────────────

    @staticmethod
    def _gsi_5(logits: torch.Tensor, labels: torch.Tensor
               ) -> Tuple[float, np.ndarray]:
        C = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        correct_p = probs[torch.arange(len(labels)), labels]
        w = (1.0 - correct_p).numpy()
        s1 = w.sum()
        s2 = (w ** 2).sum()
        if s2 < 1e-12:
            return 1.0, w
        neff = s1 ** 2 / s2
        return float(np.clip(1 - neff / len(w), 0, 1)), w

    # ── Compute all ──────────────────────────────────────────────

    def compute_all(self, model: nn.Module, dataloader) -> GSIResult:
        t0 = time.time()
        features, logits, labels, groups = self._collect(model, dataloader)
        grads = self._last_layer_grads(features, logits, labels)

        g1, energy = self._gsi_1(grads, features)
        g2, margins = self._gsi_2(logits, labels)
        g3 = self._gsi_3(grads)
        g4, sv = self._gsi_4(grads)
        g5, sw = self._gsi_5(logits, labels)

        preds = logits.argmax(1)
        correct = (preds == labels).float()
        avg = correct.mean().item()

        ga = {}
        for gid in groups.unique():
            if gid.item() == -1:
                continue
            m = groups == gid
            if m.sum() > 0:
                ga[gid.item()] = correct[m].mean().item()
        wga = min(ga.values()) if ga else avg

        return GSIResult(
            gsi_1=g1, gsi_2=g2, gsi_3=g3, gsi_4=g4, gsi_5=g5,
            avg_accuracy=avg, worst_group_accuracy=wga,
            accuracy_gap=avg - wga, group_accuracies=ga,
            per_direction_energy=energy, margins=margins,
            sigmoid_weights=sw, gradient_singular_values=sv,
            compute_seconds=time.time() - t0,
        )


# ─────────────────────────────────────────────────────────────────
# Callback — plug into any training loop
# ─────────────────────────────────────────────────────────────────

class GSICallback:
    def __init__(self, eval_dataloader, device: str = "cuda",
                 is_nlp: bool = False, compute_every: int = 1,
                 num_directions: int = 20, max_samples: int = 2000):
        self.loader = eval_dataloader
        self.every = compute_every
        self.computer = GSIComputer(num_directions, max_samples, device, is_nlp)
        self.history: List[GSIResult] = []

    def on_epoch_end(self, epoch: int, model: nn.Module) -> Optional[GSIResult]:
        if epoch % self.every != 0:
            return None
        r = self.computer.compute_all(model, self.loader)
        r.epoch = epoch
        self.history.append(r)
        wg = f"{r.worst_group_accuracy:.4f}" if r.group_accuracies else "n/a"
        print(f"  [GSI e{epoch}] 1={r.gsi_1:.3f} 2={r.gsi_2:.3f} "
              f"3={r.gsi_3:.3f} 4={r.gsi_4:.3f} 5={r.gsi_5:.3f} | "
              f"avg={r.avg_accuracy:.4f} wga={wg} ({r.compute_seconds:.1f}s)")
        return r

    def save(self, path: str):
        d = {
            "epochs":       [r.epoch for r in self.history],
            "gsi_1":        [r.gsi_1 for r in self.history],
            "gsi_2":        [r.gsi_2 for r in self.history],
            "gsi_3":        [r.gsi_3 for r in self.history],
            "gsi_4":        [r.gsi_4 for r in self.history],
            "gsi_5":        [r.gsi_5 for r in self.history],
            "avg_accuracy":          [r.avg_accuracy for r in self.history],
            "worst_group_accuracy":  [r.worst_group_accuracy for r in self.history],
            "accuracy_gap":          [r.accuracy_gap for r in self.history],
            "group_accuracies":      [r.group_accuracies for r in self.history],
            "margins":               [r.margins for r in self.history],
            "sigmoid_weights":       [r.sigmoid_weights for r in self.history],
            "per_direction_energy":  [r.per_direction_energy for r in self.history],
            "gradient_singular_values": [r.gradient_singular_values for r in self.history],
        }
        torch.save(d, path)
        print(f"Saved GSI log: {path}")

    @staticmethod
    def load(path: str) -> dict:
        return torch.load(path, weights_only=False)
