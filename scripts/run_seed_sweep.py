#!/usr/bin/env python3
"""
Run fixed-budget multi-seed experiments for Step 4 and Step 6.

Protocol:
  1. Train on the chronological training block only.
  2. Evaluate every N episodes on the validation block.
  3. Select one checkpoint per seed from validation only.
  4. Evaluate that chosen checkpoint once on the untouched test block.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = REPO_ROOT / "outputs"


def _load_module(module_name: str, filename: str):
    path = REPO_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)


def _seed_dir(root: Path, method: str, seed: int) -> Path:
    return root / method / f"seed_{seed:04d}"


def _pick_checkpoint(eval_returns: list[dict], rule: str) -> tuple[dict, dict]:
    if not eval_returns:
        raise RuntimeError("No periodic evaluation checkpoints were logged.")

    means = np.array([entry["eval_mean"] for entry in eval_returns], dtype=np.float64)

    if rule == "ma3" and len(means) >= 3:
        window = np.convolve(means, np.ones(3, dtype=np.float64) / 3.0, mode="valid")
        center_idx = int(np.argmax(window)) + 1
        selected = dict(eval_returns[center_idx])
        meta = {
            "rule": "ma3",
            "window_mean": float(window[center_idx - 1]),
            "window_episodes": [
                int(eval_returns[center_idx - 1]["episode"]),
                int(eval_returns[center_idx]["episode"]),
                int(eval_returns[center_idx + 1]["episode"]),
            ],
        }
        return selected, meta

    best_idx = int(np.argmax(means))
    selected = dict(eval_returns[best_idx])
    meta = {
        "rule": "raw",
        "window_mean": float(means[best_idx]),
        "window_episodes": [int(selected["episode"])],
    }
    return selected, meta


def _configure_step4_hp(step4, seed: int, train_episodes: int, eval_freq: int) -> dict:
    hp = dict(step4.HP)
    hp.update({
        "seed": int(seed),
        "train_episodes": int(train_episodes),
        "eval_freq": int(eval_freq),
        # Fixed-budget final runs: never stop early.
        "early_stop_patience": 10**9,
    })
    return hp


def _configure_step6_hp(step6, seed: int, train_episodes: int, eval_freq: int, env) -> dict:
    hp = dict(step6.HP)
    hp.update({
        "seed": int(seed),
        "train_episodes": int(train_episodes),
        "eval_freq": int(eval_freq),
        "early_stop_patience": 10**9,
        "forecast_horizon": int(env.future_horizon),
        "forecast_feature_channels": int(env.future_channels),
    })
    return hp


def _selected_prefix(run_dir: Path, episode: int) -> str:
    return str(run_dir / "checkpoints" / f"ep{int(episode):04d}" / "agent")


def _run_step4(step4, output_root: Path, seed: int, args: argparse.Namespace,
               oracle_val: dict, oracle_test: dict) -> dict:
    seed_root = _seed_dir(output_root, "step4", seed)
    seed_root.mkdir(parents=True, exist_ok=True)

    step4.OUTPUTS_DIR = str(seed_root)
    env = step4.BESSEnv(seed=seed, validation_days=args.validation_days,
                        ptdf_enabled=args.ptdf_enabled)
    hp = _configure_step4_hp(step4, seed, args.train_episodes, args.eval_freq)

    t0 = time.time()
    agent, log = step4.train_sac(env, hp)
    train_seconds = time.time() - t0

    run_dir = Path(step4.OUTPUTS_DIR) / "runs" / log["run_id"]
    selected_eval, selection_meta = _pick_checkpoint(log["eval_returns"], args.selection_rule)
    selected_prefix = _selected_prefix(run_dir, selected_eval["episode"])
    agent.load(selected_prefix)

    selected_agent_prefix = seed_root / "selected_agent" / "agent"
    agent.save(str(selected_agent_prefix))
    final_test = step4._run_evaluation(agent, env, starts=env.test_starts)

    summary = {
        "method": "step4",
        "seed": int(seed),
        "selection_split": log.get("selection_split", "validation"),
        "train_seconds": float(train_seconds),
        "train_episodes_budget": int(args.train_episodes),
        "eval_freq": int(args.eval_freq),
        "train_starts": int(len(env.train_starts)),
        "val_starts": int(len(env.val_starts)),
        "test_starts": int(len(env.test_starts)),
        "validation_days": int(args.validation_days),
        "run_id": log["run_id"],
        "run_dir": str(run_dir),
        "training_log_path": str(run_dir / "training_log.npy"),
        "selected_checkpoint_prefix": selected_prefix,
        "selected_eval": selected_eval,
        "selection_meta": selection_meta,
        "final_test": final_test,
        "oracle_val": oracle_val,
        "oracle_test": oracle_test,
        "hp": hp,
    }
    _save_json(seed_root / "seed_summary.json", summary)
    return summary


def _run_step6(step6, output_root: Path, seed: int, args: argparse.Namespace,
               oracle_val: dict, oracle_test: dict) -> dict:
    seed_root = _seed_dir(output_root, "step6", seed)
    workspace_dir = seed_root / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    step6.STEP6_WORKSPACE_DIR = str(workspace_dir)
    step6.STEP6_FINAL_DIR = str(workspace_dir)  # redirect; save_step6_outputs() not called in sweep
    step6.step4.OUTPUTS_DIR = str(workspace_dir)

    base_env = step6.BESSEnv(seed=seed, validation_days=args.validation_days,
                             ptdf_enabled=args.ptdf_enabled)
    forecaster = step6.AuditedAnalogForecaster(
        env=base_env,
        horizon=step6.FORECAST_HORIZON_HOURS,
        k_neighbors=step6.HP["forecast_k_neighbors"],
        eps=step6.HP["forecast_eps"],
    )
    env = step6.ForecastAugmentedEnv(
        base_env=base_env,
        forecaster=forecaster,
        include_horizon_mask=step6.HP["forecast_use_mask"],
        exclude_current_train_episode=step6.HP["forecast_exclude_current_train_episode"],
        future_horizon=step6.FORECAST_HORIZON_HOURS,
    )
    hp = _configure_step6_hp(step6, seed, args.train_episodes, args.eval_freq, env)

    original_agent_cls = step6.step4.SACAgent
    step6.step4.SACAgent = step6.ForecastAugmentedSACAgent
    try:
        t0 = time.time()
        agent, log = step6.step4.train_sac(env, hp)
        train_seconds = time.time() - t0
    finally:
        step6.step4.SACAgent = original_agent_cls

    run_dir = Path(step6.STEP6_WORKSPACE_DIR) / "runs" / log["run_id"]
    selected_eval, selection_meta = _pick_checkpoint(log["eval_returns"], args.selection_rule)
    selected_prefix = _selected_prefix(run_dir, selected_eval["episode"])
    agent.load(selected_prefix)

    selected_agent_prefix = seed_root / "selected_agent" / "agent"
    agent.save(str(selected_agent_prefix))

    # Rebuild forecaster with full pre-test library (train + val) for test evaluation.
    # Training used a train-only forecaster to prevent val leakage into observations.
    # At evaluation time the agent weights are frozen, so enriching the KNN library
    # is safe and matches real deployment conditions (all history before test is available).
    _saved_fte = base_env.forecast_train_end
    base_env.forecast_train_end = base_env.test_start_hour
    eval_forecaster = step6.AuditedAnalogForecaster(
        env=base_env,
        horizon=step6.FORECAST_HORIZON_HOURS,
        k_neighbors=step6.HP["forecast_k_neighbors"],
        eps=step6.HP["forecast_eps"],
    )
    base_env.forecast_train_end = _saved_fte  # restore for any subsequent use
    eval_env = step6.ForecastAugmentedEnv(
        base_env=base_env,
        forecaster=eval_forecaster,
        include_horizon_mask=step6.HP["forecast_use_mask"],
        exclude_current_train_episode=False,  # no training loop; exclusion not needed
        future_horizon=step6.FORECAST_HORIZON_HOURS,
    )
    final_test, _ = step6.run_audited_evaluation(
        agent,
        eval_env,
        starts=base_env.test_starts,
        audit_max_episodes=0,
        audit_max_steps=0,
    )

    summary = {
        "method": "step6",
        "seed": int(seed),
        "selection_split": log.get("selection_split", "validation"),
        "train_seconds": float(train_seconds),
        "train_episodes_budget": int(args.train_episodes),
        "eval_freq": int(args.eval_freq),
        "train_starts": int(len(base_env.train_starts)),
        "val_starts": int(len(base_env.val_starts)),
        "test_starts": int(len(base_env.test_starts)),
        "validation_days": int(args.validation_days),
        "run_id": log["run_id"],
        "run_dir": str(run_dir),
        "training_log_path": str(run_dir / "training_log.npy"),
        "selected_checkpoint_prefix": selected_prefix,
        "selected_eval": selected_eval,
        "selection_meta": selection_meta,
        "final_test": final_test,
        "oracle_val": oracle_val,
        "oracle_test": oracle_test,
        "hp": hp,
    }
    _save_json(seed_root / "seed_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-budget Step 4/6 seed sweeps.")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root directory for the sweep. Defaults to outputs/paper_seed_sweep_<timestamp>.")
    parser.add_argument("--train_episodes", type=int, default=3000)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--validation_days", type=int, default=30)
    parser.add_argument("--selection_rule", choices=("raw", "ma3"), default="ma3")
    parser.add_argument("--ptdf_enabled", type=lambda x: x.lower() != "false", default=True,
                        help="Enable PTDF safety layer (default True). Pass False for unsafe ablation.")
    parser.add_argument("--methods", nargs="+", choices=("step4", "step6"), default=("step4", "step6"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(7, 13, 29, 42, 101))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_root = OUTPUTS_ROOT / f"paper_seed_sweep_{stamp}"
    else:
        output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_episodes": int(args.train_episodes),
        "eval_freq": int(args.eval_freq),
        "validation_days": int(args.validation_days),
        "selection_rule": args.selection_rule,
        "ptdf_enabled": bool(args.ptdf_enabled),
        "methods": list(args.methods),
        "seeds": [int(seed) for seed in args.seeds],
    }
    _save_json(output_root / "manifest.json", manifest)

    step4 = _load_module("paper_step4", "4.SACAgent.py")
    step6 = _load_module("paper_step6", "6.ForecastAugmentedSAC.py")

    # LP oracle is deterministic — compute once for all seeds.
    print("\nComputing LP oracle (perfect foresight, deterministic — runs once)...")
    _oracle_env = step4.BESSEnv(seed=42, validation_days=args.validation_days)
    if len(_oracle_env.val_starts) > 0:
        oracle_val = step4._run_oracle_evaluation_on_starts(_oracle_env, _oracle_env.val_starts)
        print(f"  Oracle val  (n={oracle_val['n_eval']:3d}): "
              f"profit=${oracle_val['oracle_profit_mean']:.0f}  "
              f"eval={oracle_val['oracle_eval_mean']:+.4f}")
    else:
        oracle_val = {}
        print("  Oracle val  (n=  0): no val split (validation_days=0) — skipped.")
    oracle_test = step4._run_oracle_evaluation_on_starts(_oracle_env, _oracle_env.test_starts)
    del _oracle_env
    print(f"  Oracle test (n={oracle_test['n_eval']:3d}): "
          f"profit=${oracle_test['oracle_profit_mean']:.0f}  "
          f"eval={oracle_test['oracle_eval_mean']:+.4f}")
    manifest["oracle_val"]  = _to_jsonable(oracle_val)
    manifest["oracle_test"] = _to_jsonable(oracle_test)
    _save_json(output_root / "manifest.json", manifest)

    summaries = []
    for method in args.methods:
        for seed in args.seeds:
            print(f"\n=== {method.upper()} | seed={seed} ===")
            if method == "step4":
                summary = _run_step4(step4, output_root, int(seed), args, oracle_val, oracle_test)
            else:
                summary = _run_step6(step6, output_root, int(seed), args, oracle_val, oracle_test)
            summaries.append(summary)
            _oracle_cap = (
                summary["final_test"]["eval_mean"] / oracle_test["oracle_eval_mean"] * 100
                if oracle_test["oracle_eval_mean"] > 1e-8 else float("nan")
            )
            print(
                f"Selected ep {summary['selected_eval']['episode']} on {summary['selection_split']} "
                f"-> test profit ${summary['final_test']['profit_mean']:.0f}  "
                f"oracle capture {_oracle_cap:.1f}%"
            )

    _save_json(output_root / "all_seed_summaries.json", {"runs": summaries})
    print(f"\nSeed sweep complete -> {output_root}")


if __name__ == "__main__":
    main()
