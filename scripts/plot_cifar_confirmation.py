"""Render the frozen eight-seed CIFAR confirmation figure from compact evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


MODES = ("neutral", "random", "consistent", "conflict")
LABELS = ("Neutral core", "Random cue", "Consistent cue", "Conflict cue")
COLOR_B = "#C2672C"
COLOR_W = "#245A83"
COLOR_NEUTRAL = "#6B7280"


def _interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    half_width = student_t.ppf(0.975, df=len(values) - 1) * (
        values.std(axis=0, ddof=1) / np.sqrt(len(values))
    )
    return mean, half_width


def load_runs(evidence_root: Path) -> tuple[dict, list[dict], list[list[dict]]]:
    config = json.loads(
        (evidence_root / "configs/cifar_matched_preregistration.frozen.json").read_text()
    )
    result_root = evidence_root / "results"
    summaries, traces = [], []
    for seed in config["seed_blocks"]["confirmation"]:
        run = result_root / f"cifar_matched_frozen_v1_seed{seed}"
        summaries.append(json.loads((run / "summary.json").read_text()))
        traces.append(json.loads((run / "trace.json").read_text()))
    return config, summaries, traces


def render(evidence_root: Path, output_prefix: Path) -> None:
    config, summaries, traces = load_runs(evidence_root)
    steps = np.asarray([row["step"] for row in traces[0]], dtype=float)
    both = np.asarray([
        [row["both"]["brier_skill"] for row in trace] for trace in traces
    ])
    weak = np.asarray([
        [row["weak_only"]["brier_skill"] for row in trace] for trace in traces
    ])
    both_mean, both_ci = _interval(both)
    weak_mean, weak_ci = _interval(weak)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), constrained_layout=True)

    ax = axes[0]
    ax.fill_between(steps, both_mean - both_ci, both_mean + both_ci,
                    color=COLOR_B, alpha=0.16, linewidth=0)
    ax.fill_between(steps, weak_mean - weak_ci, weak_mean + weak_ci,
                    color=COLOR_W, alpha=0.14, linewidth=0)
    ax.plot(steps, both_mean, color=COLOR_B, linewidth=2, marker="o",
            markevery=5, markersize=3.5, label="Core + cue (B)")
    ax.plot(steps, weak_mean, color=COLOR_W, linewidth=2, linestyle="--",
            marker="s", markevery=5, markersize=3.2, label="Core only (W)")
    ax.axhline(config["response"]["beta"], color=COLOR_NEUTRAL,
               linewidth=1, linestyle=":", label=r"Gate target $\beta=0.5$")
    ax.set(xlabel="Training update", ylabel="Balanced Brier skill",
           title="A  Core-learning trajectory")
    ax.set_xlim(0, steps[-1])
    lower = min(-0.14, float(np.min(both_mean - both_ci)) - 0.02)
    ax.set_ylim(lower, 0.82)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    x = np.arange(len(MODES))
    offset = 0.10
    for condition, color, marker, shift, label in (
        ("both", COLOR_B, "o", -offset, "Core + cue (B)"),
        ("weak_only", COLOR_W, "s", offset, "Core only (W)"),
    ):
        values = np.asarray([
            [summary[f"behavior_{condition}"][mode]["accuracy"] for mode in MODES]
            for summary in summaries
        ]) * 100
        mean, ci = _interval(values)
        ax.errorbar(x + shift, mean, yerr=ci, fmt=marker, color=color,
                    markersize=5, capsize=3, linewidth=1.5, label=label)
    ax.set_xticks(x, LABELS, rotation=24, ha="right")
    ax.set(ylabel="Accuracy (%)", title="B  Cue dependence at final checkpoint")
    ax.set_ylim(40, 102)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[2]
    original = np.asarray([
        summary["original_head_neutral_accuracy_gap_B_minus_W"]
        for summary in summaries
    ]) * 100
    fresh = np.asarray([
        summary["e4_lite"]["fresh_head_neutral_accuracy_gap_B_minus_W"]
        for summary in summaries
    ]) * 100
    for original_value, fresh_value in zip(original, fresh):
        ax.plot((0, 1), (original_value, fresh_value), color="#C7CBD1",
                linewidth=1, zorder=1)
        ax.scatter((0, 1), (original_value, fresh_value), color="#A8ADB5",
                   s=12, zorder=2)
    values = np.column_stack((original, fresh))
    mean, ci = _interval(values)
    ax.errorbar((0, 1), mean, yerr=ci, fmt="D", color="#252B33",
                markerfacecolor="white", markeredgewidth=1.5,
                markersize=6, capsize=4, linewidth=1.7, zorder=3,
                label="Mean and 95% CI")
    ax.axhline(0, color=COLOR_NEUTRAL, linewidth=1, linestyle=":")
    ax.set_xticks((0, 1), ("Original head", "Fresh balanced\nlinear head"))
    ax.set(ylabel="Neutral accuracy gap B − W (pp)",
           title="C  E4-lite: partial readout recovery")
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-27, 2)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")

    fig.suptitle(
        "A removable cue consistently suppresses core learning across eight matched seeds",
        fontsize=12, fontweight="bold",
    )
    fig.text(
        0.5, -0.035,
        "Lines/points show seed means; intervals are two-sided 95% t intervals over 8 paired seed blocks. "
        "Panel C: negative values favor W.",
        ha="center", va="top", fontsize=8, color="#4B5563",
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    render(args.evidence_root, args.output_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
