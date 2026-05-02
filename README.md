# NC-SafeRL for Grid-Constrained BESS Dispatch

Research code for **network-constrained safe reinforcement learning (NC-SafeRL)** applied to battery energy storage system (BESS) dispatch on the **RTS-GMLC 73-bus** test system.

The project couples:
- DC-OPF-based locational marginal prices (LMPs),
- PTDF-aware transmission constraints,
- a Gymnasium environment with an exact 1D safety projection layer,
- a Soft Actor-Critic (SAC) agent,
- a causal rolling-horizon MPC/MILP baseline, and
- a forecast-augmented SAC variant.

## What This Repository Contains

- `1.Network.py`: builds the RTS-GMLC network in pandapower and exports Step-1 artifacts.
- `2.PrecomputeLMPs.py`: computes yearly LMP/flow/PTDF arrays used during RL.
- `3.BESSEnvironment.py`: Gymnasium environment with SoC, degradation, and safety-layer projection.
- `4.SACAgent.py`: SAC training loop, replay buffer, evaluation, and checkpoints.
- `5.OptimizationBaseline.py`: deterministic rolling-horizon MPC/MILP baseline with PTDF-aware bounds and Xu-style degradation.
- `6.ForecastAugmentedSAC.py`: forecast-augmented SAC — same bounded architecture as Step 4 with causal price/SoC forecasts injected via an observation wrapper.
- `scripts/`: utility scripts for seed sweeps and result aggregation/plotting.

## Data Policy

This repository **does not include dataset files**.  
RTS-GMLC data should be obtained directly from the official source and cited accordingly.

Dataset source:
- https://github.com/GridMod/RTS-GMLC

## Quick Start

1. Clone this repository.
2. Install dependencies.
3. Place RTS-GMLC data in `RTS-GMLC/` (matching expected structure in the scripts).
4. Run the pipeline in order:

```bash
python 1.Network.py
python 2.PrecomputeLMPs.py
python 4.SACAgent.py          # reactive SAC
python 5.OptimizationBaseline.py  # MPC/MILP baseline
python 6.ForecastAugmentedSAC.py  # forecast-augmented SAC
```

`3.BESSEnvironment.py` is imported by Steps 4 and 6.

## Typical Workflow

1. **Network build (Step 1)**  
   Creates the pandapower model and saves `outputs/step1_*`.

2. **Offline OPF precomputation (Step 2)**  
   Generates hourly LMPs, line flows, PTDF matrix, and metadata in `outputs/`.

3. **Safe RL training (Step 4 with Step 3 env)**  
   Trains SAC on precomputed arrays with hard safety enforcement from PTDF + SoC bounds.

4. **Deterministic baseline (Step 5)**  
   Runs causal rolling-horizon MPC/MILP over the same episode, producing a comparable revenue/constraint-violation benchmark.

5. **Forecast-augmented SAC (Step 6)**  
   Extends Step 4 with the same causal forecasts used by Step 5, enabling a direct three-way comparison: reactive SAC vs. MPC vs. forecast-SAC.

## Comparison Ladder

| Step | Method | Forecasts |
|------|--------|-----------|
| 4 | Reactive SAC | None |
| 5 | MPC / MILP baseline | Causal (historical analog) |
| 6 | Forecast-augmented SAC | Causal (historical analog) |

## Outputs

Generated outputs are written to `outputs/`, including:
- precomputed OPF arrays (`step2_*`),
- training logs,
- model checkpoints (`outputs/step4_checkpoints/`, `outputs/step6_checkpoints/`).

These artifacts are excluded from version control by design.

## Citation

If you use this code, please cite:
- this repository, and
- the RTS-GMLC dataset source: https://github.com/GridMod/RTS-GMLC
