#!/usr/bin/env python3
"""
Generate IEEE-style figures from aggregated seed-sweep outputs.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "outputs" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / "outputs" / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "step4": "Step 4 RL",
    "step5": "Step 5 MPC",
    "step6": "Step 6 Forecast RL",
}

METHOD_COLORS = {
    "step4": "#1f77b4",
    "step5": "#111111",
    "step6": "#d55e00",
}

METHOD_MARKERS = {
    "step4": "o",
    "step5": "D",
    "step6": "s",
}


def _read_csv(path: Path) -> list[dict]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _configure_ieee_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.grid": False,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def _plot_validation_convergence(summary_dir: Path, figures_dir: Path) -> None:
    long_rows = _read_csv(summary_dir / "convergence_long.csv")
    summary_rows = _read_csv(summary_dir / "convergence_summary.csv")

    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    grouped_seed = defaultdict(list)
    for row in long_rows:
        if row["method"] not in ("step4", "step6"):
            continue
        grouped_seed[(row["method"], row["seed"])].append(row)

    for (method, _seed), rows in sorted(grouped_seed.items()):
        rows = sorted(rows, key=lambda r: int(r["episode"]))
        x = np.array([int(r["episode"]) for r in rows], dtype=np.int32)
        y = np.array([float(r["profit_mean"]) for r in rows], dtype=np.float64)
        ax.plot(x, y, color=METHOD_COLORS[method], alpha=0.18, linewidth=0.9)

    grouped_summary = defaultdict(list)
    for row in summary_rows:
        if row["method"] in ("step4", "step6"):
            grouped_summary[row["method"]].append(row)

    for method in ("step4", "step6"):
        rows = sorted(grouped_summary[method], key=lambda r: int(r["episode"]))
        x = np.array([int(r["episode"]) for r in rows], dtype=np.int32)
        mean = np.array([float(r["profit_mean_mean"]) for r in rows], dtype=np.float64)
        std = np.array([float(r["profit_mean_std"]) for r in rows], dtype=np.float64)
        ax.fill_between(x, mean - std, mean + std, color=METHOD_COLORS[method], alpha=0.12)
        ax.plot(
            x,
            mean,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markevery=max(len(x) // 8, 1),
            label=METHOD_LABELS[method],
        )

    selection_splits = {
        row["selection_split"]
        for row in long_rows
        if row["method"] in ("step4", "step6") and row["selection_split"]
    }
    if selection_splits == {"validation"}:
        ylabel = "Validation Weekly Profit [$]"
    elif selection_splits == {"train_rolling"}:
        ylabel = "Training Profit [$] (100-ep Rolling Avg)"
    else:
        ylabel = "Selection-Split Weekly Profit [$]"

    ax.set_xlabel("Training Episodes")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    ax.legend(frameon=False, loc="best")
    ax.tick_params(direction="out", length=3, width=0.8)

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "fig_validation_convergence.pdf")
    fig.savefig(figures_dir / "fig_validation_convergence.png", dpi=300)
    plt.close(fig)


def _plot_final_test_profit(summary_dir: Path, figures_dir: Path) -> None:
    final_rows = _read_csv(summary_dir / "final_metrics.csv")

    grouped = defaultdict(list)
    for row in final_rows:
        grouped[row["method"]].append(row)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    x_positions = {"step4": 1.0, "step5": 2.0, "step6": 3.0}

    for method in ("step4", "step6"):
        rows = grouped.get(method, [])
        profits = np.array([float(row["test_profit_mean"]) for row in rows], dtype=np.float64)
        if len(profits) == 0:
            continue
        jitter = np.linspace(-0.08, 0.08, len(profits)) if len(profits) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(profits), x_positions[method]) + jitter,
            profits,
            s=22,
            facecolors="none",
            edgecolors=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidths=1.0,
            alpha=0.9,
        )
        ax.errorbar(
            x_positions[method],
            np.mean(profits),
            yerr=np.std(profits),
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=5,
            capsize=3,
            linewidth=1.4,
        )

    step5_rows = grouped.get("step5", [])
    if step5_rows:
        profit = float(step5_rows[0]["test_profit_mean"])
        ax.scatter(
            [x_positions["step5"]],
            [profit],
            s=28,
            color=METHOD_COLORS["step5"],
            marker=METHOD_MARKERS["step5"],
            zorder=5,
        )

    ax.set_xticks([1.0, 2.0, 3.0], [METHOD_LABELS["step4"], METHOD_LABELS["step5"], METHOD_LABELS["step6"]])
    ax.set_ylabel("Test Weekly Profit [$]")
    ax.tick_params(direction="out", length=3, width=0.8)

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "fig_final_test_profit.pdf")
    fig.savefig(figures_dir / "fig_final_test_profit.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot IEEE-style figures for a seed sweep.")
    parser.add_argument("root", type=str, help="Seed sweep root produced by run_seed_sweep.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    summary_dir = root / "summary"
    figures_dir = root / "figures"

    _configure_ieee_style()
    _plot_validation_convergence(summary_dir, figures_dir)
    _plot_final_test_profit(summary_dir, figures_dir)

    print(f"Figures saved -> {figures_dir}")


if __name__ == "__main__":
    main()
