#!/usr/bin/env python3
"""
run_experiment.py — Train ERM + Log GSI
========================================

Usage:
    python run_experiment.py --dataset waterbirds --seed 42
    python run_experiment.py --dataset waterbirds --seed 123
    python run_experiment.py --dataset waterbirds --seed 456

    python run_experiment.py --dataset celeba --seed 42
    python run_experiment.py --dataset civilcomments --seed 42
    python run_experiment.py --dataset multinli --seed 42

After training:
    python plot_gsi.py results/gsi_waterbirds_seed42.pt
"""

import argparse
import os
import time
import torch
import torch.nn.functional as F
import numpy as np

from gsi_core import GSICallback
from datasets import setup_waterbirds, setup_celeba, setup_civilcomments, setup_multinli


# ─────────────────────────────────────────────────────────────────
# Training: one epoch
# ─────────────────────────────────────────────────────────────────

def train_epoch_vision(model, loader, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


def train_epoch_nlp(model, loader, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for batch in loader:
        x_dict, y, _ = batch
        x_dev = {k: v.to(device) for k, v in x_dict.items()}
        y_dev = y.to(device)
        out = model(**x_dev, labels=y_dev)
        loss = out.loss
        logits = out.logits
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y_dev).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, is_nlp):
    model.eval()
    all_correct, all_groups = [], []
    for batch in loader:
        if is_nlp:
            x_dict, y, meta = batch
            x_dev = {k: v.to(device) for k, v in x_dict.items()}
            out = model(**x_dev)
            logits = out.logits if hasattr(out, "logits") else out
        else:
            x, y = batch[0].to(device), batch[1]
            meta = batch[2] if len(batch) > 2 else None
            logits = model(x)

        preds = logits.argmax(1).cpu()
        all_correct.append((preds == y).float())

        if meta is not None:
            g = meta[:, 0] if meta.dim() > 1 else meta
            all_groups.append(g.cpu() if g.is_cuda else g)

    correct = torch.cat(all_correct)
    avg = correct.mean().item()
    ga = {}
    if all_groups:
        groups = torch.cat(all_groups)
        for gid in groups.unique():
            m = groups == gid
            if m.sum() > 0:
                ga[gid.item()] = correct[m].mean().item()
    wga = min(ga.values()) if ga else avg
    return avg, wga, ga


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["waterbirds", "celeba", "civilcomments", "multinli"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--gsi_every", type=int, default=1)
    p.add_argument("--gsi_samples", type=int, default=2000)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Setup ──
    setup = {"waterbirds": setup_waterbirds, "celeba": setup_celeba,
             "civilcomments": setup_civilcomments, "multinli": setup_multinli}
    tr, va, te, model, opt, sch, cfg = setup[args.dataset](args.data_dir)
    model = model.to(device)
    is_nlp = cfg["is_nlp"]
    epochs = cfg["epochs"]
    train_fn = train_epoch_nlp if is_nlp else train_epoch_vision

    print(f"\n{'='*60}")
    print(f"  Dataset: {cfg['dataset']}  |  Epochs: {epochs}  |  "
          f"Seed: {args.seed}  |  Device: {device}")
    if "groups" in cfg:
        for gid, name in cfg["groups"].items():
            print(f"    Group {gid}: {name}")
    print(f"{'='*60}\n")

    # ── GSI callback ──
    cb = GSICallback(va, device=device, is_nlp=is_nlp,
                     compute_every=args.gsi_every, max_samples=args.gsi_samples)

    # ── Train ──
    for epoch in range(epochs):
        t0 = time.time()
        loss, acc = train_fn(model, tr, opt, device)
        sch.step()
        va_avg, va_wga, _ = evaluate(model, va, device, is_nlp)
        dt = time.time() - t0
        print(f"[Epoch {epoch:3d}] loss={loss:.4f}  train_acc={acc:.4f}  "
              f"val_avg={va_avg:.4f}  val_wga={va_wga:.4f}  ({dt:.0f}s)")
        cb.on_epoch_end(epoch, model)

    # ── Save ──
    log_path = os.path.join(args.output_dir,
                            f"gsi_{cfg['dataset']}_seed{args.seed}.pt")
    cb.save(log_path)

    # ── Test ──
    te_avg, te_wga, te_ga = evaluate(model, te, device, is_nlp)
    print(f"\n{'='*60}")
    print(f"  TEST  avg={te_avg:.4f}  wga={te_wga:.4f}")
    for gid in sorted(te_ga):
        name = cfg.get("groups", {}).get(gid, "")
        print(f"    Group {gid}: {te_ga[gid]:.4f}  {name}")
    print(f"{'='*60}")
    print(f"\nNext: python plot_gsi.py {log_path}")


if __name__ == "__main__":
    main()
