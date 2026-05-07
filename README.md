# NC-SafeRL for Grid-Constrained BESS Dispatch

Research code for network-constrained safe reinforcement learning (NC-SafeRL) for price-taking battery energy storage dispatch on the RTS-GMLC 73-bus system.

## Active pipeline

- `1.Network.py`: build the RTS-GMLC network and export Step 1 artifacts.
- `2.PrecomputeLMPs.py`: precompute yearly LMP, PTDF, and flow arrays.
- `3.BESSEnvironment.py`: Gymnasium environment with SoC, degradation, and PTDF-aware safety.
- `4.SACAgent.py`: reactive safe SAC training and evaluation.
- `5.OptimizationBaseline.py`: causal 24-hour MPC/MILP baseline.
- `scripts/run_seed_sweep.py`: fixed-budget Step 4 seed sweeps.
- `scripts/aggregate_seed_sweep.py`: aggregate Step 4 / Step 5 outputs.
- `scripts/plot_seed_sweep_ieee.py`: generate paper figures from aggregated summaries.

## Local-only assets

The following stay local and are intentionally not versioned:
- `Reports/`
- top-level `figures/`
- `RTS-GMLC/`
- `Literature/`

## Reproducible run order

```bash
python 1.Network.py
python 2.PrecomputeLMPs.py
python 5.OptimizationBaseline.py --validation_days 0
python scripts/run_seed_sweep.py --methods step4 --validation_days 0 --seeds 7 13 29 41 97 --train_episodes 3000 --eval_freq 200 --output_root outputs/paper_seed_sweep_20260506_5seeds
python scripts/aggregate_seed_sweep.py outputs/paper_seed_sweep_20260506_5seeds --methods step4 step5
python scripts/plot_seed_sweep_ieee.py outputs/paper_seed_sweep_20260506_5seeds --methods step4 step5
```

## Tracked outputs

This repo keeps only the current reproducibility artifacts under `outputs/`:
- `step1_*`
- `step2_*`
- `step5_final/*`
- `paper_seed_sweep_20260506_5seeds/*`

## Dependencies

- `requirements.txt`: top-level, human-maintained dependency spec.
- `requirements-lock.txt`: exact pinned environment snapshot used for reproducibility.

## Data source

RTS-GMLC data should be obtained from the official source:
- https://github.com/GridMod/RTS-GMLC
