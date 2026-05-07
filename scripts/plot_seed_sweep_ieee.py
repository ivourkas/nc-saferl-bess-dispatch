#!/usr/bin/env python3
"""
Generate IEEE-style figures from aggregated seed-sweep outputs.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_TMP_ROOT = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "rl_project_plot_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_TMP_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_TMP_ROOT / "xdg-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "step4": "Reactive SAC",
    "step5": "Causal MILP",
}

METHOD_COLORS = {
    "step4": "#1f77b4",
    "step5": "#111111",
}

METHOD_MARKERS = {
    "step4": "o",
    "step5": "D",
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


def _plot_validation_convergence(summary_dir: Path, figures_dir: Path, rl_methods: list[str]) -> None:
    if not rl_methods:
        return

    long_rows = _read_csv(summary_dir / "convergence_long.csv")
    summary_rows = _read_csv(summary_dir / "convergence_summary.csv")

    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    grouped_seed = defaultdict(list)
    for row in long_rows:
        if row["method"] not in rl_methods:
            continue
        grouped_seed[(row["method"], row["seed"])].append(row)

    for (method, _seed), rows in sorted(grouped_seed.items()):
        rows = sorted(rows, key=lambda r: int(r["episode"]))
        x = np.array([int(r["episode"]) for r in rows], dtype=np.int32)
        y = np.array([float(r["profit_mean"]) for r in rows], dtype=np.float64)
        ax.plot(x, y, color=METHOD_COLORS[method], alpha=0.18, linewidth=0.9)

    grouped_summary = defaultdict(list)
    for row in summary_rows:
        if row["method"] in rl_methods:
            grouped_summary[row["method"]].append(row)

    for method in rl_methods:
        rows = sorted(grouped_summary[method], key=lambda r: int(r["episode"]))
        if not rows:
            continue
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

    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Mean Weekly Profit [$]")
    ax.set_xlim(left=0)
    ax.legend(frameon=False, loc="best")
    ax.tick_params(direction="out", length=3, width=0.8)

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "fig_validation_convergence.pdf")
    fig.savefig(figures_dir / "fig_validation_convergence.png", dpi=300)
    plt.close(fig)


def _plot_final_test_profit(summary_dir: Path, figures_dir: Path, methods: list[str]) -> None:
    if not methods:
        return

    final_rows = _read_csv(summary_dir / "final_metrics.csv")

    grouped = defaultdict(list)
    for row in final_rows:
        grouped[row["method"]].append(row)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x_positions = {method: float(i) for i, method in enumerate(methods)}
    plotted_values = []

    for method in methods:
        if method == "step5":
            continue
        rows = grouped.get(method, [])
        profits = np.array([float(row["test_profit_mean"]) for row in rows], dtype=np.float64)
        if len(profits) == 0:
            continue
        mean_profit = float(np.mean(profits))
        std_profit = float(np.std(profits))
        ax.errorbar(
            x_positions[method],
            mean_profit,
            yerr=std_profit,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor=METHOD_COLORS[method],
            markersize=7,
            capsize=4,
            elinewidth=1.8,
            linewidth=1.8,
            zorder=3,
        )
        plotted_values.extend([mean_profit - std_profit, mean_profit, mean_profit + std_profit])
        ax.annotate(
            f"{mean_profit:.0f}",
            (x_positions[method], mean_profit),
            xytext=(8, 4),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=7,
            color=METHOD_COLORS[method],
        )

    step5_rows = grouped.get("step5", [])
    if "step5" in methods and step5_rows:
        profit = float(step5_rows[0]["test_profit_mean"])
        ax.scatter(
            [x_positions["step5"]],
            [profit],
            s=54,
            color=METHOD_COLORS["step5"],
            marker=METHOD_MARKERS["step5"],
            zorder=5,
        )
        plotted_values.append(profit)
        ax.annotate(
            f"{profit:.0f}",
            (x_positions["step5"], profit),
            xytext=(8, 4),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=7,
            color=METHOD_COLORS["step5"],
        )

    display_labels = {
        "step4": "Reactive\nSAC",
        "step5": "Causal\nMILP",
    }
    ax.set_xticks(
        [x_positions[m] for m in methods],
        [display_labels[m] for m in methods],
    )
    ax.set_ylabel("Test Weekly Profit [$]")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.set_xlim(min(x_positions.values()) - 0.2, max(x_positions.values()) + 0.2)
    if plotted_values:
        y_min = min(plotted_values)
        y_max = max(plotted_values)
        y_span = max(y_max - y_min, 1.0)
        ax.set_ylim(y_min - 0.10 * y_span, y_max + 0.18 * y_span)
    ax.tick_params(direction="out", length=3, width=0.8)

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "fig_final_test_profit.pdf")
    fig.savefig(figures_dir / "fig_final_test_profit.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot IEEE-style figures for a seed sweep.")
    parser.add_argument("root", type=str, help="Seed sweep root produced by run_seed_sweep.py")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("step4", "step5"),
        default=("step4", "step5"),
        help="Subset of methods to include in the figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    summary_dir = root / "summary"
    figures_dir = root / "figures"
    methods = list(args.methods)
    rl_methods = [m for m in methods if m == "step4"]

    _configure_ieee_style()
    _plot_validation_convergence(summary_dir, figures_dir, rl_methods)
    _plot_final_test_profit(summary_dir, figures_dir, methods)

    print(f"Figures saved -> {figures_dir}")


if __name__ == "__main__":
    main()
