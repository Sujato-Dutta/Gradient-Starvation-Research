#!/usr/bin/env python3
"""
plot_gsi.py — Generate 5 diagnostic figures from a GSI log
===========================================================

Usage:
    python plot_gsi.py results/gsi_waterbirds_seed42.pt

Generates:
    fig1_dashboard.pdf       — GSI values + accuracies over epochs
    fig2_margins.pdf         — Margin distribution evolution
    fig3_sigmoid_weights.pdf — Gradient weight concentration
    fig4_energy_spectrum.pdf — Per-direction gradient energy
    fig5_correlations.pdf    — Which GSI predicts worst-group accuracy?
"""

import sys
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.bbox": "tight"})

GSI_KEYS   = ["gsi_1", "gsi_2", "gsi_3", "gsi_4", "gsi_5"]
GSI_NAMES  = ["GSI-1: Grad Energy", "GSI-2: Margin Disp.",
              "GSI-3: Cosine Align", "GSI-4: Eff Rank",
              "GSI-5: Sigmoid Conc."]
GSI_COLORS = ["#e74c3c", "#e67e22", "#f1c40f", "#27ae60", "#2980b9"]


def load(path):
    return torch.load(path, weights_only=False)


# ─────────────────────────────────────────────────────────────────
# FIG 1: Dashboard
# ─────────────────────────────────────────────────────────────────

def fig1_dashboard(log, out):
    ep = log["epochs"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 1]})
    for k, n, c in zip(GSI_KEYS, GSI_NAMES, GSI_COLORS):
        a1.plot(ep, log[k], color=c, lw=2, alpha=0.85, label=n)
    a1.set_ylabel("GSI  (0=balanced, 1=starved)")
    a1.set_ylim(-0.05, 1.05)
    a1.legend(loc="upper left", fontsize=9)
    a1.set_title("Gradient Starvation Index Over Training")
    a1.grid(True, alpha=0.2)

    a2.plot(ep, log["avg_accuracy"], "k-", lw=2, label="Avg Accuracy")
    a2.plot(ep, log["worst_group_accuracy"], "r--", lw=2.5, label="Worst Group")
    a2.fill_between(ep, log["worst_group_accuracy"], log["avg_accuracy"],
                    alpha=0.15, color="red")
    a2.set_xlabel("Epoch")
    a2.set_ylabel("Accuracy")
    a2.set_ylim(-0.05, 1.05)
    a2.legend(loc="lower right")
    a2.set_title("Average vs Worst-Group Accuracy")
    a2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# ─────────────────────────────────────────────────────────────────
# FIG 2: Margin evolution
# ─────────────────────────────────────────────────────────────────

def fig2_margins(log, out):
    ep = log["epochs"]
    ms = log["margins"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Violin
    idx = np.linspace(0, len(ep) - 1, min(12, len(ep)), dtype=int)
    data, pos = [], []
    for i in idx:
        m = ms[i]
        if len(m) > 0:
            data.append(m)
            pos.append(ep[i])
    if data:
        w = max(1, (pos[-1] - pos[0]) / len(pos) * 0.7)
        parts = a1.violinplot(data, positions=pos, widths=w,
                               showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("#3498db")
            pc.set_alpha(0.5)
    a1.axhline(0, color="red", ls="--", alpha=0.5, lw=1)
    a1.set_xlabel("Epoch")
    a1.set_ylabel("Margin")
    a1.set_title("Margin Distribution Over Training")
    a1.grid(True, alpha=0.2)

    # Heatmap
    all_m = np.concatenate([m for m in ms if len(m) > 0])
    lo, hi = np.percentile(all_m, [2, 98])
    nb = 60
    edges = np.linspace(lo, hi, nb + 1)
    hm = np.zeros((nb, len(ep)))
    for t, m in enumerate(ms):
        if len(m) > 0:
            h, _ = np.histogram(m, bins=edges, density=True)
            hm[:, t] = h
    im = a2.imshow(hm, aspect="auto", origin="lower",
                    extent=[ep[0], ep[-1], lo, hi], cmap="inferno",
                    interpolation="bilinear")
    a2.axhline(0, color="white", ls="--", alpha=0.7, lw=1)
    a2.set_xlabel("Epoch")
    a2.set_ylabel("Margin")
    a2.set_title("Margin Density Heatmap")
    plt.colorbar(im, ax=a2, label="Density", shrink=0.8)

    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# ─────────────────────────────────────────────────────────────────
# FIG 3: Sigmoid weight analysis
# ─────────────────────────────────────────────────────────────────

def fig3_sigmoid(log, out):
    ep = log["epochs"]
    sw = log["sigmoid_weights"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel A: fraction active
    ax = axes[0]
    for thr, col in [(0.01, "#bdc3c7"), (0.05, "#95a5a6"),
                     (0.1, "#e67e22"), (0.3, "#e74c3c"), (0.5, "#c0392b")]:
        fr = [(s > thr).mean() if len(s) > 0 else 1.0 for s in sw]
        ax.plot(ep, fr, color=col, lw=1.8, label=f"w > {thr}")
    ax.axhline(0.1, color="red", ls=":", alpha=0.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction of Examples")
    ax.set_title("A) Active Examples")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)

    # Panel B: sorted weights at 3 timepoints
    ax = axes[1]
    tps = [0, len(ep) // 2, -1]
    cols = ["#3498db", "#e67e22", "#e74c3c"]
    labs = ["Early", "Mid", "Late"]
    for tp, c, l in zip(tps, cols, labs):
        s = sw[tp]
        if len(s) > 0:
            sv = np.sort(s)[::-1]
            x = np.arange(len(sv)) / len(sv)
            ax.plot(x, sv + 1e-8, color=c, lw=2, label=f"{l} (ep {ep[tp]})")
    ax.set_xlabel("Example Rank (fraction)")
    ax.set_ylabel("Gradient Weight w_i")
    ax.set_title("B) Sorted Weights")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel C: effective N
    ax = axes[2]
    eff = []
    for s in sw:
        if len(s) > 0:
            s1, s2 = s.sum(), (s ** 2).sum()
            eff.append(s1 ** 2 / (s2 + 1e-12) / len(s))
        else:
            eff.append(1.0)
    ax.plot(ep, eff, "k-", lw=2.5)
    ax.fill_between(ep, 0, eff, alpha=0.2, color="steelblue")
    ax.axhline(0.1, color="red", ls=":", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Effective N / Total N")
    ax.set_title("C) Effective Training Set")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# ─────────────────────────────────────────────────────────────────
# FIG 4: Gradient energy spectrum
# ─────────────────────────────────────────────────────────────────

def fig4_energy(log, out):
    ep = log["epochs"]
    en = log["per_direction_energy"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    tps = [0, len(ep) // 2, -1]
    titles = ["Early", "Mid", "Late"]
    for ax, tp, ti in zip(axes, tps, titles):
        e = en[tp]
        if len(e) > 0:
            k = min(15, len(e))
            bars = ax.bar(range(k), e[:k], color="#3498db", alpha=0.8)
            mx = np.argmax(e[:k])
            bars[mx].set_facecolor("#e74c3c")
            tot = e[:k].sum()
            if tot > 0:
                pct = e[mx] / tot * 100
                bg = "yellow" if pct > 50 else "lightgreen"
                ax.text(0.95, 0.95, f"Top: {pct:.0f}%", transform=ax.transAxes,
                        ha="right", va="top", fontsize=10,
                        bbox=dict(boxstyle="round", facecolor=bg, alpha=0.5))
        ax.set_xlabel("Feature Direction")
        ax.set_ylabel("Gradient Energy")
        ax.set_title(f"{ti} (epoch {ep[tp]})")
        ax.grid(True, alpha=0.2, axis="y")
    plt.suptitle("Gradient Energy Per Feature Direction", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# ─────────────────────────────────────────────────────────────────
# FIG 5: Correlation — which GSI predicts WGA?
# ─────────────────────────────────────────────────────────────────

def fig5_correlations(log, out):
    from scipy import stats
    ep = log["epochs"]
    wga = np.array(log["worst_group_accuracy"])
    if wga.std() < 1e-6:
        print("  No variation in WGA — skipping fig5")
        return

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    results = {}
    for ax, k, n in zip(axes, GSI_KEYS, GSI_NAMES):
        gv = np.array(log[k])
        rho, pv = stats.spearmanr(gv, wga)
        results[k] = (rho, pv)
        ax.scatter(gv, wga, c=ep, cmap="viridis", s=25, alpha=0.8)
        z = np.polyfit(gv, wga, 1)
        xl = np.linspace(gv.min(), gv.max(), 100)
        ax.plot(xl, np.polyval(z, xl), "r--", alpha=0.5, lw=1.5)
        ax.set_xlabel(n, fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Worst Group Accuracy")
        if abs(rho) > 0.5 and pv < 0.05:
            bg, v = "#2ecc71", "PREDICTIVE"
        elif abs(rho) > 0.3:
            bg, v = "#f39c12", "MARGINAL"
        else:
            bg, v = "#e74c3c", "WEAK"
        ax.text(0.05, 0.05, f"\u03C1={rho:.3f}\np={pv:.1e}\n{v}",
                transform=ax.transAxes, fontsize=9, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=bg, alpha=0.3))

    plt.suptitle("Which GSI Predicts Worst-Group Accuracy?", fontsize=13)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

    # Print ranking
    ranked = sorted(results.items(), key=lambda x: abs(x[1][0]), reverse=True)
    print(f"\n  {'GSI Variant Ranking':─^50}")
    for i, (k, (rho, pv)) in enumerate(ranked, 1):
        star = " ★" if i == 1 and abs(rho) > 0.5 else ""
        print(f"  {i}. {k}: \u03C1={rho:+.4f}  p={pv:.1e}{star}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main(log_path):
    log = load(log_path)
    base = log_path.replace(".pt", "")

    print(f"\nGenerating figures from {log_path}")
    for fmt in ["pdf", "png"]:
        fig1_dashboard(log,  f"{base}_fig1_dashboard.{fmt}")
        fig2_margins(log,    f"{base}_fig2_margins.{fmt}")
        fig3_sigmoid(log,    f"{base}_fig3_sigmoid.{fmt}")
        fig4_energy(log,     f"{base}_fig4_energy.{fmt}")
        fig5_correlations(log, f"{base}_fig5_correlations.{fmt}")

    print(f"\nAll figures saved with prefix: {base}_fig*")
    print(f"Start with: {base}_fig5_correlations.pdf")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_gsi.py <log_path.pt>")
        sys.exit(1)
    main(sys.argv[1])
