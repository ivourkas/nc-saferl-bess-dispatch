"""
Step 2: pre-compute hourly LMPs, baseline line flows, and PTDF data.

The main output of this step is a set of numpy arrays used directly by the
environment so RL does not need to solve DC-OPFs online.
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
RELAXED_OPF_KW   = {
    # Last-chance numerical rescue before imputation.
    # This does not change the modeled network or costs; it only relaxes the
    # interior-point convergence tolerances and iteration cap.
    "check_connectivity": False,
    "PDIPM_MAX_IT": 300,
    "OPF_VIOLATION": 1e-4,
    "PDIPM_GRADTOL": 1e-4,
    "PDIPM_COMPTOL": 1e-4,
    "PDIPM_COSTTOL": 1e-4,
}


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
    Compute PTDF columns by injecting 1 MW at each non-slack bus in turn.
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

def apply_loads_for_hour(hour, working_net=None):
    """
    Scale bus loads to the area's hourly total for the requested hour.
    """
    global net
    _net = working_net if working_net is not None else net
    for bus_id, (base_p, base_q) in base_load_by_bus.items():
        if base_p <= 0 and base_q <= 0:
            continue                              # no load at this bus
        area  = bus_area[bus_id]
        scale = float(load_ts.iloc[hour][str(area)]) / area_base_load[area]
        mask  = _net.load["name"] == f"Load_{bus_id}"
        if mask.any():
            _net.load.loc[mask, "p_mw"]   = base_p * scale
            _net.load.loc[mask, "q_mvar"] = base_q * scale


def apply_renewables_for_hour(hour, working_net=None):
    """
    Update renewable output limits from the hourly time-series data.
    """
    global net
    _net = working_net if working_net is not None else net
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
                _net.ext_grid.at[idx, "max_p_mw"] = max(val, 0.1)
            else:
                pmin = float(_net.gen.at[idx, "min_p_mw"])
                # Ensure max ≥ min (otherwise OPF is infeasible for this unit)
                _net.gen.at[idx, "max_p_mw"] = max(val, pmin)
                # Keep starting dispatch ≤ new max  (feasibility of warm-start)
                _net.gen.at[idx, "p_mw"]     = min(_net.gen.at[idx, "p_mw"], val)


def build_fresh_hour_net(hour):
    """
    Reload the clean Step 1 network and apply the requested hour's exogenous data.
    """
    fresh_net = pp.from_json(os.path.join(OUT_DIR, "step1_network.json"))
    apply_loads_for_hour(hour, working_net=fresh_net)
    apply_renewables_for_hour(hour, working_net=fresh_net)
    return fresh_net


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
# lmps[h, k]       = LMP at pandapower bus index k at hour h  [$/MWh]
# line_flows[h, l]  = MW flow on branch l at hour h            [MW]
#                     columns 0..N_LINE-1    → AC lines  (net.res_line)
#                     columns N_LINE..N_BR-1 → trafos   (net.res_trafo)
# converged[h]      = True if the OPF solved successfully
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
n_recovered_fresh   = 0
n_recovered_relaxed = 0

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
    #
    # Retry logic:
    # 1) try the rolling sequential net,
    # 2) if that fails, reload a fresh copy of the Step 1 network and retry,
    # 3) if that still fails, try one last fresh-net solve with mildly relaxed
    #    interior-point tolerances before falling back to imputation.
    #
    # This keeps the physical model unchanged while giving the OPF solver a more
    # robust numerical path on the handful of hard hours.
    def _try_solve(working_net, **opf_kwargs):
        """Run rundcopp on working_net and return extracted results, or raise."""
        pp.rundcopp(working_net, verbose=False, **opf_kwargs)
        lmp_h   = working_net.res_bus["lam_p"].values.astype(np.float32)
        flow_h  = np.empty(N_BR, dtype=np.float32)
        flow_h[:N_LINE] = working_net.res_line["p_from_mw"].values
        flow_h[N_LINE:] = working_net.res_trafo["p_hv_mw"].values
        return lmp_h, flow_h

    _solved = False
    _solve_mode = "sequential"
    try:
        _lmp_h, _flow_h = _try_solve(net)
        _solved = True
    except Exception:
        # First failure: may be sequential-state path dependence.
        # Retry on a clean copy of the network with the same hour's exogenous data.
        _net_fresh = build_fresh_hour_net(hour)
        try:
            _lmp_h, _flow_h = _try_solve(_net_fresh)
            net = _net_fresh   # promote fresh net: subsequent hours use clean state
            _solve_mode = "fresh"
            _solved = True
        except Exception:
            # Last-chance rescue: same physical hour, but with more permissive
            # numerical tolerances for the interior-point OPF solver.
            _net_relaxed = build_fresh_hour_net(hour)
            try:
                _lmp_h, _flow_h = _try_solve(_net_relaxed, **RELAXED_OPF_KW)
                net = _net_relaxed
                _solve_mode = "relaxed"
                _solved = True
            except Exception:
                pass   # genuine failed hour after all rescue attempts

    if _solved:
        # ── D3: Extract results ────────────────────────────────────────────
        lmps[hour]       = _lmp_h
        line_flows[hour] = _flow_h
        converged[hour]  = True
        prev_lmps  = _lmp_h.copy()
        prev_flows = _flow_h.copy()
        if _solve_mode == "fresh":
            n_recovered_fresh += 1
        elif _solve_mode == "relaxed":
            n_recovered_relaxed += 1
    else:
        # ── D4: Genuine failed OPF ─────────────────────────────────────────
        # Only reached after sequential solve, fresh exact retry, and relaxed
        # fresh retry all fail.  Carry forward the previous converged snapshot
        # as a best-effort fallback.
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
              f"rescued={n_recovered_fresh + n_recovered_relaxed:3d} "
              f"(fresh={n_recovered_fresh:3d}, relaxed={n_recovered_relaxed:3d}) | "
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
    "n_recovered_fresh": int(n_recovered_fresh),
    "n_recovered_relaxed": int(n_recovered_relaxed),
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
