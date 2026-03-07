"""
Step 2: Pre-compute LMPs, Line Flows, and PTDF for all 8,784 Hours
====================================================================
Author: Yiannis Vourkas

WHY THIS STEP EXISTS
--------------------
The RL agent needs LMPs and line flows at every training step (millions of
calls). Running a full DC-OPF inside the training loop would make training
10-100x slower. Instead, we run all 8,784 OPFs ONCE here and store the
results as fast numpy lookup arrays.

During RL training, the environment just does:
    lmp_now    = lmps[hour]          # 73 values, 0.0001 ms
    flows_now  = line_flows[hour]    # 120 values, 0.0001 ms
instead of:
    pp.rundcopp(net)                 # ~500-1000 ms

WHAT THIS SCRIPT PRODUCES (in outputs/)
----------------------------------------
  step2_lmps.npy          (8784, 73)  LMP at each bus each hour [$/MWh]
  step2_line_flows.npy    (8784, 120) Baseline MW flow on each AC branch
  step2_ptdf.npy          (120,  73)  PTDF matrix (constant, topology-only)
  step2_converged.npy     (8784,)     Bool: did the OPF converge this hour?
  step2_line_limits.npy   (120,)      Thermal limit per branch [MW], in
                                      pandapower [lines, trafos] order.
                                      Matches the constraint the OPF enforces.
                                      Used by Step 3 safety-layer QP.
  step2_pv_total.npy      (8784,)     System-wide PV+RTPV generation [MW].
                                      Used as a state feature in Step 3.
  step2_metadata.json                 Shapes, bus order, branch names

  DROPPED (no longer saved):
  step2_line_loadings.npy -- Redundant: derivable as line_flows / line_limits * 100.
                             Not loaded by Step 3 (safety layer uses flows + limits directly).
  step2_pv_bus313.npy     -- Not used anywhere; saved for reference only. Eliminated.

THE PTDF MATRIX
---------------
PTDF[l, k] = change in flow on branch l (MW) for 1 MW injection at bus k,
             with the slack bus (bus 113) as the reference withdrawal point.

Used in the safety layer QP (Step 4):
    F_l,t = F_l,t_base + PTDF[l, battery_bus] * P_battery
    |F_l,t| <= F_l_max   for all lines l

PTDF is constant for a fixed topology. We compute it ONCE numerically:
for each non-slack bus k:
    1. Add +1 MW injection at bus k (via -1 MW temporary load)
    2. Run DC power flow (slack absorbs the +1 MW withdrawal automatically)
    3. PTDF[:, k] = new_flows - base_flows   (sensitivity per unit injection)
"""

import pandas as pd
import numpy as np
import pandapower as pp
import os
import sys
import json
import time

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════
OUT_DIR          = os.path.join(os.path.dirname(__file__), "outputs")
RTS_SRC          = os.path.join(os.path.dirname(__file__),
                                "RTS-GMLC", "RTS_Data", "SourceData")
RTS_TS           = os.path.join(os.path.dirname(__file__),
                                "RTS-GMLC", "RTS_Data", "timeseries_data_files")
CHECKPOINT_EVERY = 500    # Save partial arrays every N hours (crash recovery)


# ════════════════════════════════════════════════════════════════
# PART A: LOAD STEP 1 OUTPUTS
# ════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART A: Loading Step 1 Network and Mapping")
print("=" * 70)

for fname in ["step1_network.json", "step1_mapping.json"]:
    if not os.path.exists(os.path.join(OUT_DIR, fname)):
        sys.exit(f"ERROR: {fname} not found. Run 1.Network.py first.")

# Load the full pandapower network (buses, lines, generators, costs, ...)
net = pp.from_json(os.path.join(OUT_DIR, "step1_network.json"))
print(f"  Network: {len(net.bus)} buses | {len(net.line)} lines | "
      f"{len(net.trafo)} trafos | {len(net.gen)} gens | "
      f"{len(net.dcline)} HVDC")

# Load the metadata saved in Step 1
with open(os.path.join(OUT_DIR, "step1_mapping.json")) as f:
    mapping = json.load(f)

bus_id_to_pp   = {int(k): int(v) for k, v in mapping["bus_id_to_pp"].items()}
bus_area       = {int(k): int(v) for k, v in mapping["bus_area"].items()}
area_base_load = {int(k): float(v) for k, v in mapping["area_base_load"].items()}
LINE_LIMIT_SCALE = float(mapping["line_limit_scale"])

# Derived sizes  (used throughout as axis lengths)
N_BUS   = len(net.bus)          # 73 buses
N_LINE  = len(net.line)         # 104 AC transmission lines
N_TRAFO = len(net.trafo)        # 16 transformers
N_BR    = N_LINE + N_TRAFO      # 120 total AC branches (lines + trafos)

# pandapower index -> RTS bus ID (for metadata and readability)
pp_to_bus_id = {int(v): int(k) for k, v in bus_id_to_pp.items()}

print(f"  Sizes: N_BUS={N_BUS}, N_LINE={N_LINE}, N_TRAFO={N_TRAFO}, N_BR={N_BR}")


# ════════════════════════════════════════════════════════════════
# PART B: COMPUTE PTDF MATRIX  (one-time, topology is fixed)
# ════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PART B: Computing PTDF Matrix (numerical sensitivity)")
print("=" * 70)

def compute_ptdf_numerical(net):
    """
    Compute PTDF by perturbing each non-slack bus one at a time.

    For each bus k:
      1. Add a temporary -1 MW load at bus k  (+1 MW net injection)
      2. Run DC power flow  (slack gen absorbs the extra MW automatically,
         so the net effect is: +1 MW at bus k, -1 MW at slack bus)
      3. PTDF[:, k] = (flows_with_injection) - (base_flows)

    This is the standard "shift factor" method. It gives PTDF in the
    exact pandapower line/trafo ordering, no internal API needed.

    Returns: ndarray of shape (N_BR, N_BUS) = (120, 73)
    """
    slack_pp_idx = int(net.ext_grid["bus"].values[0])   # pandapower bus index of slack

    # --- baseline DC power flow (loads in current state, no extra injection) ---
    pp.rundcpp(net, check_connectivity=False, verbose=False)
    base_flows_line  = net.res_line["p_from_mw"].values.copy()    # (104,) MW
    base_flows_trafo = net.res_trafo["p_hv_mw"].values.copy()     # (16,)  MW
    base_flows = np.concatenate([base_flows_line, base_flows_trafo])  # (120,)

    # --- allocate output ---
    ptdf = np.zeros((N_BR, N_BUS))

    # --- one DC power flow per non-slack bus ---
    n_buses = len(net.bus)
    print(f"  Running {n_buses - 1} DC power flows for PTDF ...", end="", flush=True)

    for col, pp_bus_idx in enumerate(range(n_buses)):
        if pp_bus_idx == slack_pp_idx:
            # By definition, PTDF[:, slack_col] = 0
            # (injecting at slack and withdrawing at slack = no flow change)
            continue

        # Temporarily inject +1 MW at this bus
        tmp_idx = pp.create_load(net, bus=pp_bus_idx, p_mw=-1.0, q_mvar=0.0)

        try:
            pp.rundcpp(net, check_connectivity=False, verbose=False)
            flows_line  = net.res_line["p_from_mw"].values
            flows_trafo = net.res_trafo["p_hv_mw"].values
            flows = np.concatenate([flows_line, flows_trafo])
            ptdf[:, col] = flows - base_flows    # sensitivity: ΔMW per 1 MW injection
        except Exception:
            pass    # leave column as zeros if DC power flow fails

        # Remove the temporary load (keep net.load clean for the OPF loop)
        net.load.drop(index=tmp_idx, inplace=True)

    print(f" done.")
    return ptdf


ptdf = compute_ptdf_numerical(net)
np.save(os.path.join(OUT_DIR, "step2_ptdf.npy"), ptdf)

print(f"  PTDF shape: {ptdf.shape}  (branches x buses)")
print(f"  Max |PTDF|: {np.abs(ptdf).max():.4f}  (must be <= 1.0 for DC networks)")
print(f"  Saved: step2_ptdf.npy")


# ════════════════════════════════════════════════════════════════
# PART C: LOAD ALL TIME-SERIES FILES INTO MEMORY
# ════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PART C: Loading Time-Series Files")
print("=" * 70)

# Each file is a (8784 rows × N_plants cols) CSV.
# Column names are the GEN UID strings (e.g. "303_WIND_1") that match gen.csv.
load_ts  = pd.read_csv(os.path.join(RTS_TS, "Load/DAY_AHEAD_regional_Load.csv"))
wind_ts  = pd.read_csv(os.path.join(RTS_TS, "WIND/DAY_AHEAD_wind.csv"))
pv_ts    = pd.read_csv(os.path.join(RTS_TS, "PV/DAY_AHEAD_pv.csv"))
rtpv_ts  = pd.read_csv(os.path.join(RTS_TS, "RTPV/DAY_AHEAD_rtpv.csv"))
hydro_ts = pd.read_csv(os.path.join(RTS_TS, "Hydro/DAY_AHEAD_hydro.csv"))
csp_ts   = pd.read_csv(os.path.join(RTS_TS, "CSP/DAY_AHEAD_Natural_Inflow.csv"))

N_HOURS = len(load_ts)   # 8784 (366-day leap year)
print(f"  {N_HOURS} hours ({N_HOURS // 24} days) loaded into memory")

# ── Build generator UID → pandapower table lookup ─────────────────────────
# This maps a GEN UID string (e.g. "303_WIND_1") to (table, row_index)
# where table = "gen" or "ext" (for the slack ext_grid).
gen_uid_to_pp = {}
for i, row in net.gen.iterrows():
    gen_uid_to_pp[row["name"]] = ("gen", i)
for i, row in net.ext_grid.iterrows():
    gen_uid_to_pp[row["name"].replace("Slack_", "")] = ("ext", i)

# ── Pre-load base bus loads from bus.csv (used in proportional scaling) ──
bus_df = pd.read_csv(os.path.join(RTS_SRC, "bus.csv"))
base_load_by_bus = {
    int(r["Bus ID"]): (float(r["MW Load"]), float(r["MVAR Load"]))
    for _, r in bus_df.iterrows()
}


# ════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS  (same physics as Step 1, factored out cleanly)
# ════════════════════════════════════════════════════════════════

def apply_loads_for_hour(hour):
    """
    Scale every bus load proportionally to the area's actual hourly MW.

    Formula:  bus_load_h = base_bus_load × (area_total_h / area_base_total)

    base_bus_load  = peak value from bus.csv  (constant)
    area_total_h   = actual area MW from time-series for hour h
    area_base_total = sum of base loads in that area (~2850 MW each)
    """
    for bus_id, (base_p, base_q) in base_load_by_bus.items():
        if base_p <= 0 and base_q <= 0:
            continue                              # no load at this bus
        area  = bus_area[bus_id]
        scale = float(load_ts.iloc[hour][str(area)]) / area_base_load[area]
        mask  = net.load["name"] == f"Load_{bus_id}"
        if mask.any():
            net.load.loc[mask, "p_mw"]   = base_p * scale
            net.load.loc[mask, "q_mvar"] = base_q * scale


def apply_renewables_for_hour(hour):
    """
    Cap each renewable generator's max output to what the time-series says.

    The OPF will dispatch renewables at their time-series value (zero cost →
    always fully dispatched up to the cap). Thermal units fill the rest.

    We also guard against max_p_mw < min_p_mw which would make the OPF
    infeasible (e.g. a hydro plant with PMin=5 MW but time-series gives 0).
    """
    for ts_df, keyword in [
        (wind_ts,  "WIND"),
        (pv_ts,    "PV"),
        (rtpv_ts,  "RTPV"),
        (hydro_ts, "HYDRO"),
        (csp_ts,   "CSP"),
    ]:
        for col in (c for c in ts_df.columns if keyword in c):
            if col not in gen_uid_to_pp:
                continue
            kind, idx = gen_uid_to_pp[col]
            val = max(float(ts_df.iloc[hour][col]), 0.0)

            if kind == "ext":
                # Slack ext_grid acting as a renewable: cap its max output
                net.ext_grid.at[idx, "max_p_mw"] = max(val, 0.1)
            else:
                pmin = float(net.gen.at[idx, "min_p_mw"])
                # Ensure max ≥ min (otherwise OPF is infeasible for this unit)
                net.gen.at[idx, "max_p_mw"] = max(val, pmin)
                # Keep starting dispatch ≤ new max  (feasibility of warm-start)
                net.gen.at[idx, "p_mw"]     = min(net.gen.at[idx, "p_mw"], val)


# ════════════════════════════════════════════════════════════════
# PART D: MAIN OPF LOOP  —  8,784 hours
# ════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"PART D: DC-OPF Loop — {N_HOURS} hours")
print("=" * 70)
print(f"  Saving checkpoints every {CHECKPOINT_EVERY} hours.")
est_lo = N_HOURS * 0.5 / 3600
est_hi = N_HOURS * 1.5 / 3600
print(f"  Estimated time: {est_lo:.1f} – {est_hi:.1f} hours (hardware dependent)\n")

# ── Pre-allocate result arrays (NaN = not yet filled / failed) ────────────
# lmps[h, k]          = LMP at pandapower bus index k at hour h  [$/MWh]
# line_flows[h, l]    = MW flow on branch l at hour h            [MW]
#                       columns 0..N_LINE-1   → AC lines  (net.res_line)
#                       columns N_LINE..N_BR-1 → trafos   (net.res_trafo)
# line_loadings[h, l] = loading % on branch l at hour h          [%]
# converged[h]        = True if the OPF solved successfully
lmps          = np.full((N_HOURS, N_BUS), np.nan, dtype=np.float32)
line_flows    = np.full((N_HOURS, N_BR),  np.nan, dtype=np.float32)
converged     = np.zeros(N_HOURS, dtype=bool)

# Fallback LMPs: if OPF fails at hour h, use the last successful LMPs.
# This prevents NaN from propagating into the RL state space.
prev_lmps = np.zeros(N_BUS, dtype=np.float32)
# Fallback line flows: carry forward the last successful branch-flow vector.
# Using zeros here would create physically impossible "empty network" snapshots
# that can poison downstream PTDF safety bounds.
prev_flows = np.zeros(N_BR, dtype=np.float32)

t_start  = time.time()
n_failed = 0

for hour in range(N_HOURS):

    # ── D1: Update network state for this hour ─────────────────────────────
    apply_loads_for_hour(hour)
    apply_renewables_for_hour(hour)

    # ── D2: Solve DC Optimal Power Flow ───────────────────────────────────
    # rundcopp minimizes total generation cost subject to:
    #   - generator PMin/PMax bounds
    #   - line thermal limits (set at 75% in Step 1 via max_i_ka)
    #   - DC power flow equations (Kirchhoff's laws)
    # The shadow prices on the nodal balance constraints = LMPs (lam_p)
    try:
        pp.rundcopp(net, verbose=False)

        # ── D3: Extract results ────────────────────────────────────────────
        # LMPs at all buses  (shape: N_BUS)
        lmps[hour] = net.res_bus["lam_p"].values.astype(np.float32)

        # Branch flows: AC lines first, then transformers
        # p_from_mw = MW injected at the "from" bus end of the branch
        line_flows[hour, :N_LINE]    = net.res_line["p_from_mw"].values
        line_flows[hour, N_LINE:]    = net.res_trafo["p_hv_mw"].values

        converged[hour] = True
        prev_lmps = lmps[hour].copy()
        prev_flows = line_flows[hour].copy()

    except Exception as e:
        # ── D4: Failed OPF ─────────────────────────────────────────────────
        # Mark as failed; use previous hour's LMPs as a fallback.
        # Failed hours are flagged in converged[] so Step 3 can skip them or
        # handle them specially during RL episode construction.
        lmps[hour]       = prev_lmps
        line_flows[hour] = prev_flows
        converged[hour]  = False
        n_failed += 1

    # ── D5: Progress report every 100 hours ───────────────────────────────
    if (hour + 1) % 100 == 0 or hour == N_HOURS - 1:
        elapsed   = time.time() - t_start
        rate      = (hour + 1) / elapsed          # solved hours per second
        eta_s     = (N_HOURS - hour - 1) / rate   # estimated seconds remaining
        pct       = (hour + 1) / N_HOURS * 100
        lmp_mean  = float(lmps[hour].mean()) if converged[hour] else float("nan")
        print(f"  [{pct:5.1f}%] h={hour+1:4d}/{N_HOURS} | "
              f"failed={n_failed:3d} | "
              f"elapsed={elapsed/60:5.1f}m | "
              f"ETA={eta_s/60:5.1f}m | "
              f"LMP_mean=${lmp_mean:.2f}")

    # ── D6: Checkpoint save every CHECKPOINT_EVERY hours ──────────────────
    # Protects against crashes on long runs.
    # Partial files are deleted once the final save succeeds.
    if (hour + 1) % CHECKPOINT_EVERY == 0:
        np.save(os.path.join(OUT_DIR, "_ckpt_lmps.npy"),       lmps)
        np.save(os.path.join(OUT_DIR, "_ckpt_line_flows.npy"), line_flows)
        print(f"  [CHECKPOINT] Partial save at hour {hour + 1}")


# ════════════════════════════════════════════════════════════════
# PART E: SAVE FINAL OUTPUTS
# ════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PART E: Saving Final Outputs")
print("=" * 70)

os.makedirs(OUT_DIR, exist_ok=True)

np.save(os.path.join(OUT_DIR, "step2_lmps.npy"),       lmps)
np.save(os.path.join(OUT_DIR, "step2_line_flows.npy"), line_flows)
np.save(os.path.join(OUT_DIR, "step2_converged.npy"),  converged)

# ── step2_line_limits.npy ──────────────────────────────────────────────────
# Thermal limit in MW for each of the 120 branches, in pandapower ordering
# (first N_LINE AC lines, then N_TRAFO transformers).  This is exactly what
# the DC-OPF enforced: max_i_ka * sqrt(3) * vn_kv for lines, and
# sn_mva * max_loading_percent/100 for transformers.
# Step 3 safety-layer QP uses this array to bound BESS dispatch.
_from_bus_idx  = net.line["from_bus"].values
_vn_kv_arr     = net.bus.loc[_from_bus_idx, "vn_kv"].values
_line_lim_mw   = (net.line["max_i_ka"].values * np.sqrt(3) * _vn_kv_arr).astype(np.float32)
_trafo_lim_mw  = (net.trafo["sn_mva"].values *
                  net.trafo["max_loading_percent"].values / 100.0).astype(np.float32)
line_limits    = np.concatenate([_line_lim_mw, _trafo_lim_mw])
np.save(os.path.join(OUT_DIR, "step2_line_limits.npy"), line_limits)

# ── step2_pv_total.npy ────────────────────────────────────────────────────
# System-wide utility PV + rooftop PV (RTPV) generation per hour [MW].
# Excludes Year/Month/Day/Period timestamp columns.
# Used as a state feature in Step 3 (proxy for renewable generation level).
_pv_data_cols   = [c for c in pv_ts.columns   if c not in ["Year", "Month", "Day", "Period"]]
_rtpv_data_cols = [c for c in rtpv_ts.columns if c not in ["Year", "Month", "Day", "Period"]]
pv_total = (pv_ts[_pv_data_cols].sum(axis=1).values +
            rtpv_ts[_rtpv_data_cols].sum(axis=1).values).astype(np.float32)
np.save(os.path.join(OUT_DIR, "step2_pv_total.npy"), pv_total)

# Clean up checkpoint files now that the final save succeeded
for ckpt in ["_ckpt_lmps.npy", "_ckpt_line_flows.npy"]:
    p = os.path.join(OUT_DIR, ckpt)
    if os.path.exists(p):
        os.remove(p)

# ── Metadata: documents what each array axis means ────────────────────────
# Step 3 reads this to know: "column 5 of lmps = which RTS bus?"
branch_names = net.line["name"].tolist() + net.trafo["name"].tolist()
bus_order    = [pp_to_bus_id[i] for i in range(N_BUS)]   # pp_idx -> RTS ID

metadata = {
    # Array shapes
    "n_hours":    N_HOURS,
    "n_buses":    N_BUS,
    "n_branches": N_BR,
    "n_lines":    N_LINE,
    "n_trafos":   N_TRAFO,
    # Quality
    "n_converged":       int(converged.sum()),
    "n_failed":          int(n_failed),
    "convergence_rate":  float(converged.mean()),
    # Axis labels
    # lmps[:, k]          → LMP at RTS bus ID bus_order[k]
    # line_flows[:, l]    → flow on branch branch_names[l]
    "bus_order_rts_ids": bus_order,
    "branch_names":      branch_names,
    # Config
    "line_limit_scale": LINE_LIMIT_SCALE,
    "total_time_minutes": round((time.time() - t_start) / 60, 2),
}

with open(os.path.join(OUT_DIR, "step2_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  step2_lmps.npy         {lmps.shape}   {lmps.nbytes/1e6:.1f} MB")
print(f"  step2_line_flows.npy   {line_flows.shape}  {line_flows.nbytes/1e6:.1f} MB")
print(f"  step2_ptdf.npy         {ptdf.shape}     {ptdf.nbytes/1e3:.0f} KB")
print(f"  step2_converged.npy    {converged.shape}")
print(f"  step2_line_limits.npy  {line_limits.shape}   "
      f"range=[{line_limits.min():.1f}, {line_limits.max():.1f}] MW")
print(f"  step2_pv_total.npy     {pv_total.shape}  "
      f"range=[{pv_total.min():.1f}, {pv_total.max():.1f}] MW")
print(f"  step2_metadata.json")


# ════════════════════════════════════════════════════════════════
# PART F: SUMMARY STATISTICS  (sanity checks)
# ════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PART F: Summary Statistics")
print("=" * 70)

valid = lmps[converged]          # only use converged hours for stats

print(f"\n  LMP STATISTICS  ({converged.sum()}/{N_HOURS} hours converged)")
print(f"  --------------------------------------------------------")
print(f"  Mean:       ${valid.mean():>8.2f} /MWh")
print(f"  Std:        ${valid.std():>8.2f} /MWh")
print(f"  Min:        ${valid.min():>8.2f} /MWh")
print(f"  Max:        ${valid.max():>8.2f} /MWh")
print(f"  Negative:   {(valid < 0).sum():6d} bus-hours ({(valid<0).mean()*100:.1f}%)")
print(f"  Zero:       {(valid == 0).sum():6d} bus-hours ({(valid==0).mean()*100:.1f}%)")
print(f"  --------------------------------------------------------")

# Per-area mean LMP (key for locational value analysis)
print(f"\n  Per-area mean LMP ($/MWh):")
for area in sorted(set(bus_area.values())):
    area_cols = [bus_id_to_pp[bid] for bid, a in bus_area.items() if a == area]
    area_lmps = valid[:, area_cols]
    print(f"    Area {area}: mean=${area_lmps.mean():.2f}  "
          f"std=${area_lmps.std():.2f}  "
          f"min=${area_lmps.min():.2f}  "
          f"max=${area_lmps.max():.2f}")

# Congestion summary (derive loading% from flows / limits on the fly)
# line_loadings no longer saved; compute here for stats only.
_valid_flows = line_flows[converged]                                 # (N_conv, N_BR)
_loading_pct = np.abs(_valid_flows) / np.maximum(line_limits, 1e-3) * 100.0
n_cong_hours  = (_loading_pct.max(axis=1) > 90).sum()
most_cong_idx = _loading_pct.mean(axis=0).argmax()
print(f"\n  Congestion (loading > 90%):")
print(f"    Hours with any congested branch: {n_cong_hours} "
      f"({n_cong_hours/converged.sum()*100:.1f}% of converged hours)")
print(f"    Most congested branch on average: "
      f"'{branch_names[most_cong_idx]}'  "
      f"(mean loading {_loading_pct.mean(axis=0).max():.1f}%)")

# LMP spread distribution (key metric for RL locational value)
spreads = valid.max(axis=1) - valid.min(axis=1)
print(f"\n  LMP spread (max - min across 73 buses per hour):")
print(f"    Mean spread:   ${spreads.mean():.2f}")
print(f"    Median spread: ${np.median(spreads):.2f}")
print(f"    Max spread:    ${spreads.max():.2f}")
print(f"    Hours > $20 spread: {(spreads > 20).sum()} "
      f"({(spreads > 20).mean()*100:.1f}%)")
print(f"    Hours > $50 spread: {(spreads > 50).sum()} "
      f"({(spreads > 50).mean()*100:.1f}%)")

total_mins = (time.time() - t_start) / 60
print(f"\n  Total runtime: {total_mins:.1f} min  ({total_mins/60:.2f} hrs)")
print(f"\n{'=' * 70}")
print("STEP 2 COMPLETE")
print("Outputs ready for Step 3: Build Gymnasium BESS Environment")
print("=" * 70)
