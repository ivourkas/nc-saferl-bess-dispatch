#!/usr/bin/env python3
"""
Aggregate fixed-budget seed-sweep results into paper-ready CSV/JSON summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP5_METRICS = REPO_ROOT / "outputs" / "step5_final" / "mpc_baseline_metrics.json"


def _load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _extract_step5_eval(step5_payload: dict) -> dict:
    """
    Support both the legacy Step 5 payload format {"final_eval": {...}}
    and the current format {"metrics": {...}}.
    """
    if "final_eval" in step5_payload:
        return step5_payload["final_eval"]
    if "metrics" in step5_payload:
        return step5_payload["metrics"]
    raise KeyError(
        "Step 5 metrics JSON must contain either 'final_eval' or 'metrics'. "
        f"Found keys: {sorted(step5_payload.keys())}"
    )


def _collect_seed_summaries(root: Path) -> list[dict]:
    summaries = []
    for summary_path in sorted(root.glob("step*/seed_*/seed_summary.json")):
        summaries.append(_load_json(summary_path))
    if not summaries:
        raise RuntimeError(f"No seed_summary.json files found under {root}")
    return summaries


def _build_convergence_rows(seed_summaries: list[dict]) -> list[dict]:
    rows = []
    for summary in seed_summaries:
        log = np.load(summary["training_log_path"], allow_pickle=True).item()
        for entry in log["eval_returns"]:
            rows.append({
                "method": summary["method"],
                "seed": int(summary["seed"]),
                "selection_split": summary["selection_split"],
                "episode": int(entry["episode"]),
                "eval_mean": float(entry["eval_mean"]),
                "eval_std": float(entry["eval_std"]),
                "profit_mean": float(entry["profit_mean"]),
                "revenue_mean": float(entry["revenue_mean"]),
                "deg_cost_mean": float(entry["deg_cost_mean"]),
            })
    return rows


def _aggregate_convergence(convergence_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in convergence_rows:
        grouped[(row["method"], row["episode"])].append(row)

    summary_rows = []
    for (method, episode), rows in sorted(grouped.items()):
        eval_mean = np.array([row["eval_mean"] for row in rows], dtype=np.float64)
        profit_mean = np.array([row["profit_mean"] for row in rows], dtype=np.float64)
        summary_rows.append({
            "method": method,
            "episode": int(episode),
            "n_seeds": int(len(rows)),
            "eval_mean_mean": float(np.mean(eval_mean)),
            "eval_mean_std": float(np.std(eval_mean)),
            "profit_mean_mean": float(np.mean(profit_mean)),
            "profit_mean_std": float(np.std(profit_mean)),
        })
    return summary_rows


def _build_final_rows(seed_summaries: list[dict], step5_payload: dict | None) -> list[dict]:
    rows = []
    for summary in seed_summaries:
        final_test = summary["final_test"]
        selected = summary["selected_eval"]
        rows.append({
            "method": summary["method"],
            "seed": int(summary["seed"]),
            "selection_split": summary["selection_split"],
            "selection_episode": int(selected["episode"]),
            "selection_eval_mean": float(selected["eval_mean"]),
            "selection_profit_mean": float(selected["profit_mean"]),
            "test_eval_mean": float(final_test["eval_mean"]),
            "test_eval_std": float(final_test["eval_std"]),
            "test_eval_min": float(final_test["eval_min"]),
            "test_eval_max": float(final_test["eval_max"]),
            "test_profit_mean": float(final_test["profit_mean"]),
            "test_revenue_mean": float(final_test["revenue_mean"]),
            "test_deg_cost_mean": float(final_test["deg_cost_mean"]),
            "test_soc_end_mean": float(final_test["soc_end_mean"]),
            "test_soc_min_mean": float(final_test["soc_min_mean"]),
            "test_frac_discharge": float(final_test["frac_discharge"]),
            "test_frac_charge": float(final_test["frac_charge"]),
            "test_frac_idle": float(final_test["frac_idle"]),
            "train_seconds": float(summary["train_seconds"]),
        })

    if step5_payload is not None:
        final_eval = _extract_step5_eval(step5_payload)
        rows.append({
            "method": "step5",
            "seed": "",
            "selection_split": "",
            "selection_episode": "",
            "selection_eval_mean": "",
            "selection_profit_mean": "",
            "test_eval_mean": float(final_eval["eval_mean"]),
            "test_eval_std": float(final_eval["eval_std"]),
            "test_eval_min": float(final_eval["eval_min"]),
            "test_eval_max": float(final_eval["eval_max"]),
            "test_profit_mean": float(final_eval["profit_mean"]),
            "test_revenue_mean": float(final_eval["revenue_mean"]),
            "test_deg_cost_mean": float(final_eval["deg_cost_mean"]),
            "test_soc_end_mean": float(final_eval["soc_end_mean"]),
            "test_soc_min_mean": float(final_eval["soc_min_mean"]),
            "test_frac_discharge": float(final_eval["frac_discharge"]),
            "test_frac_charge": float(final_eval["frac_charge"]),
            "test_frac_idle": float(final_eval["frac_idle"]),
            "train_seconds": "",
        })

    return rows


def _aggregate_final(final_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in final_rows:
        grouped[row["method"]].append(row)

    summary_rows = []
    for method, rows in sorted(grouped.items()):
        metrics = {
            "test_eval_mean": np.array([float(row["test_eval_mean"]) for row in rows], dtype=np.float64),
            "test_profit_mean": np.array([float(row["test_profit_mean"]) for row in rows], dtype=np.float64),
            "test_eval_std": np.array([float(row["test_eval_std"]) for row in rows], dtype=np.float64),
            "test_eval_min": np.array([float(row["test_eval_min"]) for row in rows], dtype=np.float64),
            "train_seconds": np.array(
                [float(row["train_seconds"]) for row in rows if row["train_seconds"] != ""],
                dtype=np.float64,
            ),
        }
        summary_rows.append({
            "method": method,
            "n_runs": int(len(rows)),
            "test_eval_mean_mean": float(np.mean(metrics["test_eval_mean"])),
            "test_eval_mean_std": float(np.std(metrics["test_eval_mean"])),
            "test_profit_mean_mean": float(np.mean(metrics["test_profit_mean"])),
            "test_profit_mean_std": float(np.std(metrics["test_profit_mean"])),
            "test_eval_std_mean": float(np.mean(metrics["test_eval_std"])),
            "worst_week_mean": float(np.mean(metrics["test_eval_min"])),
            "avg_train_minutes": float(np.mean(metrics["train_seconds"]) / 60.0) if len(metrics["train_seconds"]) else 0.0,
        })
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate seed sweep outputs.")
    parser.add_argument("root", type=str, help="Seed sweep root produced by run_seed_sweep.py")
    parser.add_argument("--step5_metrics", type=str, default=str(DEFAULT_STEP5_METRICS),
                        help="Optional Step 5 metrics JSON to include as deterministic baseline.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    summary_dir = root / "summary"

    seed_summaries = _collect_seed_summaries(root)
    convergence_rows = _build_convergence_rows(seed_summaries)
    convergence_summary = _aggregate_convergence(convergence_rows)

    step5_path = Path(args.step5_metrics).expanduser().resolve()
    step5_payload = _load_json(step5_path) if step5_path.exists() else None
    final_rows = _build_final_rows(seed_summaries, step5_payload)
    final_summary = _aggregate_final(final_rows)

    _write_csv(
        summary_dir / "convergence_long.csv",
        convergence_rows,
        [
            "method", "seed", "selection_split", "episode",
            "eval_mean", "eval_std", "profit_mean", "revenue_mean", "deg_cost_mean",
        ],
    )
    _write_csv(
        summary_dir / "convergence_summary.csv",
        convergence_summary,
        [
            "method", "episode", "n_seeds",
            "eval_mean_mean", "eval_mean_std",
            "profit_mean_mean", "profit_mean_std",
        ],
    )
    _write_csv(
        summary_dir / "final_metrics.csv",
        final_rows,
        [
            "method", "seed", "selection_split", "selection_episode",
            "selection_eval_mean", "selection_profit_mean",
            "test_eval_mean", "test_eval_std", "test_eval_min", "test_eval_max",
            "test_profit_mean", "test_revenue_mean", "test_deg_cost_mean",
            "test_soc_end_mean", "test_soc_min_mean",
            "test_frac_discharge", "test_frac_charge", "test_frac_idle",
            "train_seconds",
        ],
    )
    _write_csv(
        summary_dir / "final_summary.csv",
        final_summary,
        [
            "method", "n_runs",
            "test_eval_mean_mean", "test_eval_mean_std",
            "test_profit_mean_mean", "test_profit_mean_std",
            "test_eval_std_mean", "worst_week_mean",
            "avg_train_minutes",
        ],
    )

    paper_table = {
        "step4": next((row for row in final_summary if row["method"] == "step4"), None),
        "step5": next((row for row in final_summary if row["method"] == "step5"), None),
        "step6": next((row for row in final_summary if row["method"] == "step6"), None),
    }
    _write_json(summary_dir / "paper_table.json", paper_table)
    _write_json(
        summary_dir / "manifest.json",
        {
            "n_seed_runs": int(len(seed_summaries)),
            "step5_metrics_used": str(step5_path) if step5_payload is not None else None,
        },
    )

    print(f"Aggregation complete -> {summary_dir}")


if __name__ == "__main__":
    main()
