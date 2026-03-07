# NC-SafeRL for Grid-Constrained BESS Dispatch

Research code for **network-constrained safe reinforcement learning (NC-SafeRL)** applied to battery energy storage system (BESS) dispatch on the **RTS-GMLC 73-bus** test system.

The project couples:
- DC-OPF-based locational marginal prices (LMPs),
- PTDF-aware transmission constraints,
- a Gymnasium environment with an exact 1D safety projection layer,
- and a Soft Actor-Critic (SAC) agent.

## What This Repository Contains

- `1.Network.py`: builds the RTS-GMLC network in pandapower and exports Step-1 artifacts.
- `2.PrecomputeLMPs.py`: computes yearly LMP/flow/PTDF arrays used during RL.
- `3.BESSEnvironment.py`: Gymnasium environment with SoC, degradation, and safety-layer projection.
- `4.SACAgent.py`: SAC training loop, replay buffer, evaluation, and checkpoints.

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
python 4.SACAgent.py
```

`3.BESSEnvironment.py` is imported by the SAC script.

## Typical Workflow

1. **Network build (Step 1)**  
Creates the pandapower model and saves `outputs/step1_*`.

2. **Offline OPF precomputation (Step 2)**  
Generates hourly LMPs, line flows, PTDF matrix, and metadata in `outputs/`.

3. **Safe RL training (Step 4 with Step 3 env)**  
Trains SAC on precomputed arrays with hard safety enforcement from PTDF + SoC bounds.

## Outputs

Generated outputs are written to `outputs/`, including:
- precomputed OPF arrays (`step2_*`),
- training logs,
- model checkpoints (`outputs/step4_checkpoints/`).

These artifacts are excluded from version control by design.

## Citation

If you use this code, please cite:
- this repository, and
- the RTS-GMLC dataset source: https://github.com/GridMod/RTS-GMLC

