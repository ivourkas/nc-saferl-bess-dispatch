"""
Step 1: Build RTS-GMLC 73-bus Network in Pandapower 
====================================================================
Author: Yiannis Vourkas 

Data source: https://github.com/GridMod/RTS-GMLC
  git clone https://github.com/GridMod/RTS-GMLC.git

Network: 73 buses, 120 AC branches, 1 HVDC line, ~155 generators (3 areas).
Bus numbering: Area 1 = 101-124, Area 2 = 201-224, Area 3 = 301-325.

DATA FILES USED (from SourceData/):
  bus.csv     - Bus ID, BaseKV, MW Load, MVAR Load, Area     (all relevant cols)
  branch.csv  - topology, R/X/B p.u., Cont Rating, Tr Ratio  (all relevant cols)
  gen.csv     - PMax/PMin, 4-point piecewise heat rate curve,  (full cost curve)
                Fuel Price, VOM, Category
  dc_branch.csv - HVDC line: bus 113 <-> bus 316, 100 MW      (modeled as dcline)
  storage.csv   - 313_STORAGE_1 at bus 313: 50 MW / 150 MWh   (EXCLUDED - see note)

DATA FILES NOT USED (with justification):
  reserves.csv           - For publication extension (energy+reserves), not class project.
  simulation_objects.csv - Simulation config, not needed for network model.
  timeseries_pointers.csv - We match generators to time-series by UID directly.

DESIGN DECISIONS:
  1. Generator costs use the PIECEWISE LINEAR heat rate curve from gen.csv
     (Output_pct_0..3, HR_avg_0, HR_Incr_1..3), not a flat average.
  2. The existing 50 MW battery (313_STORAGE_1) is EXCLUDED from the model.
     Our project studies where to place a NEW battery. Having background storage
     at bus 313 (the largest PV bus) would absorb excess PV and reduce the
     interesting locational price dynamics we want to study.
  3. The HVDC line (bus 113 <-> bus 316, 100 MW) is modeled as a pandapower dcline.
     Bidirectional: min_p_mw=-75, max_p_mw=+75 MW. OPF determines flow direction.
  4. Line thermal limits are scaled by LINE_LIMIT_SCALE to approximate N-1
     security margins, consistent with real transmission operating practices.
  5. Slack bus is bus 113 (Ref type in bus.csv), with 113_CT_1 as ext_grid.
  6. PMin = 0 for all generators (economic dispatch without unit commitment).
     Equivalent to assuming optimal commitment. Standard in DC-OPF studies
     that do not co-optimize binary on/off decisions (Barrows et al. 2019).

TIME-SERIES LOADING (Part B):
  Load/DAY_AHEAD_regional_Load.csv - Area-level load scaling factors
  WIND/DAY_AHEAD_wind.csv          - Per-plant wind output
  PV/DAY_AHEAD_pv.csv              - Per-plant utility PV output
  RTPV/DAY_AHEAD_rtpv.csv          - Per-plant rooftop PV output
  Hydro/DAY_AHEAD_hydro.csv        - Per-plant hydro output
  CSP/DAY_AHEAD_Natural_Inflow.csv - CSP thermal inflow
"""

import pandas as pd
import numpy as np
import pandapower as pp
import os
import sys
import json


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
RTS_PATH = os.path.join(os.path.dirname(__file__), "RTS-GMLC", "RTS_Data")
LINE_LIMIT_SCALE  = 0.75     # Scale line ratings (0.7-0.8 approximates N-1 margins)
TEST_HOUR         = 5726     # Peak load hour for validation
SN_MVA            = 100.0    # System base MVA

# Fuel price scaling to match US wholesale electricity market stress conditions.
# RTS-GMLC baseline uses 2020-era WECC-calibrated gas prices: $3.89/MMBTU.
# Without scaling, bus 111 mean LMP ≈ $11–13/MWh — below the Xu segment-1
# degradation breakeven ($13.93/MWh), making BESS arbitrage economically trivial.
#
# Scale factor 2.0× → $7.78/MMBTU for gas CTs.
# Rationale: natural gas delivery prices at NYISO-relevant hubs in 2022:
#   Henry Hub spot (national):      $6.45/MMBTU annual avg (EIA, Natural Gas Annual 2023)
#   Transco Zone 6 NY (NYISO):      ~$9–12/MMBTU annual avg (2022, EIA Natural Gas Monthly)
#   Algonquin Citygate (NE border): ~$10–15/MMBTU annual avg (2022)
# $7.78/MMBTU is a conservative midpoint between Henry Hub and NYISO delivery
# prices, representing realistic US Northeast wholesale market conditions in 2022.
#
# Reference: EIA, "Natural Gas Annual 2023," DOE/EIA-0131(2023), Nov 2024.
#
# Expected LMPs after scaling (re-run Step 2 for exact values):
#   Gas CT marginal cost:  ~$54/MWh  (vs $49/MWh at 1.8×)
#   Bus 111 annual mean:   ~$21–23/MWh
#   Bus 111 peak LMP:      ~$82–90/MWh  → segments 1–3 profitable at peaks
#   System-wide max LMP:   ~$185–200/MWh (congested Area 3 buses)
FUEL_PRICE_SCALE  = 2.0   # 2.0× → $7.78/MMBTU (US Northeast high-gas 2022 scenario)

# Generators to EXCLUDE from the model (and why)
EXCLUDED_GENS = {
    "313_STORAGE_1": "Existing battery at bus 313 - excluded so we can study "
                     "placement of our own battery without background storage.",
}


# ================================================================
# PART A: BUILD THE STATIC NETWORK
# ================================================================
print("=" * 70)
print("PART A: Building RTS-GMLC 73-bus Network")
print("=" * 70)

# -- A1: Load CSVs --
src = os.path.join(RTS_PATH, "SourceData")
bus_df    = pd.read_csv(os.path.join(src, "bus.csv"))
branch_df = pd.read_csv(os.path.join(src, "branch.csv"))
gen_df    = pd.read_csv(os.path.join(src, "gen.csv"))
dc_df     = pd.read_csv(os.path.join(src, "dc_branch.csv"))

print(f"\n  Raw data: {len(bus_df)} buses, {len(branch_df)} AC branches, "
      f"{len(dc_df)} DC branches, {len(gen_df)} generators")

# -- A2: Create network and bus mapping --
net = pp.create_empty_network(name="RTS-GMLC", sn_mva=SN_MVA)
bus_id_to_pp = {}   # RTS bus ID (e.g. 313) -> pandapower index (e.g. 45)
bus_area = {}       # RTS bus ID -> area number (from bus.csv Area column)

# -- A3: Add buses (using Area column from bus.csv) --
for _, row in bus_df.iterrows():
    bus_id = int(row["Bus ID"])
    pp_idx = pp.create_bus(net, vn_kv=float(row["BaseKV"]),
                           name=f"{bus_id}_{row['Bus Name']}")
    bus_id_to_pp[bus_id] = pp_idx
    bus_area[bus_id] = int(row["Area"])

print(f"  Added {len(net.bus)} buses (areas: {sorted(set(bus_area.values()))})")

# -- A4: Add loads --
for _, row in bus_df.iterrows():
    bus_id = int(row["Bus ID"])
    p_mw   = float(row["MW Load"])
    q_mvar = float(row["MVAR Load"])
    if p_mw > 0 or q_mvar > 0:
        pp.create_load(net, bus=bus_id_to_pp[bus_id],
                       p_mw=p_mw, q_mvar=q_mvar, name=f"Load_{bus_id}")

net.load["controllable"] = False
print(f"  Added {len(net.load)} loads (base total: {net.load.p_mw.sum():.0f} MW)")

# -- A5: Add AC branches (lines and transformers) --
line_count = 0
trafo_count = 0

for _, row in branch_df.iterrows():
    from_bus = int(row["From Bus"])
    to_bus   = int(row["To Bus"])
    r_pu     = float(row["R"])
    x_pu     = float(row["X"])
    b_pu     = float(row["B"])
    rate_mva = float(row["Cont Rating"])
    tr_ratio = float(row["Tr Ratio"])
    uid      = str(row["UID"])

    pp_from = bus_id_to_pp[from_bus]
    pp_to   = bus_id_to_pp[to_bus]

    if tr_ratio > 0:
        # -- TRANSFORMER --
        vn_from = float(bus_df.loc[bus_df["Bus ID"] == from_bus, "BaseKV"].values[0])
        vn_to   = float(bus_df.loc[bus_df["Bus ID"] == to_bus,   "BaseKV"].values[0])

        if vn_from >= vn_to:
            vn_hv, vn_lv, hv_bus, lv_bus = vn_from, vn_to, pp_from, pp_to
        else:
            vn_hv, vn_lv, hv_bus, lv_bus = vn_to, vn_from, pp_to, pp_from

        z_pu = np.sqrt(r_pu**2 + x_pu**2)
        pp.create_transformer_from_parameters(
            net, hv_bus=hv_bus, lv_bus=lv_bus,
            sn_mva=rate_mva, vn_hv_kv=vn_hv, vn_lv_kv=vn_lv,
            vk_percent=z_pu * 100.0,
            vkr_percent=r_pu * 100.0,
            pfe_kw=0, i0_percent=0, name=uid,
            max_loading_percent=LINE_LIMIT_SCALE * 100,
        )
        trafo_count += 1
    else:
        # -- TRANSMISSION LINE --
        vn_kv = float(bus_df.loc[bus_df["Bus ID"] == from_bus, "BaseKV"].values[0])
        z_base = (vn_kv ** 2) / SN_MVA

        r_ohm = r_pu * z_base
        x_ohm = x_pu * z_base
        c_nf  = max(b_pu / (2 * np.pi * 60) / z_base * 1e9, 0)
        max_i_ka = rate_mva / (np.sqrt(3) * vn_kv) * LINE_LIMIT_SCALE

        pp.create_line_from_parameters(
            net, from_bus=pp_from, to_bus=pp_to, length_km=1.0,
            r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm, c_nf_per_km=c_nf,
            max_i_ka=max_i_ka, name=uid,
        )
        line_count += 1

net.line["max_loading_percent"] = 100.0
print(f"  Added {line_count} lines + {trafo_count} trafos "
      f"(limits at {LINE_LIMIT_SCALE*100:.0f}% of rated)")

# -- A6: Add HVDC line --
for _, row in dc_df.iterrows():
    from_bus = int(row["From Bus"])
    to_bus   = int(row["To Bus"])
    p_mw_scheduled = float(row["MW Load"])
    uid = str(row["UID"])

    dc_idx = pp.create_dcline(
        net,
        from_bus=bus_id_to_pp[from_bus],
        to_bus=bus_id_to_pp[to_bus],
        p_mw=0.0,                                      # OPF determines flow
        loss_percent=2.0,
        loss_mw=0,
        vm_from_pu=1.0,
        vm_to_pu=1.0,
        name=uid,
        max_p_mw= p_mw_scheduled * LINE_LIMIT_SCALE,  # +75 MW: 113 -> 316
        min_p_mw=-p_mw_scheduled * LINE_LIMIT_SCALE,  # -75 MW: 316 -> 113
    )
    pp.create_poly_cost(net, element=dc_idx, et="dcline",
                        cp1_eur_per_mw=0.0, cp0_eur=0.0)

print(f"  Added {len(net.dcline)} HVDC line(s): "
      f"bus {from_bus} -> bus {to_bus}, {p_mw_scheduled:.0f} MW rated")


# -- A7: Add generators with PIECEWISE LINEAR cost curves --
#
# Heat rate curve from gen.csv (4 output points, 3 incremental segments):
#
#   Point 0: P0 = PMax * Output_pct_0  (= PMin, minimum stable level)
#   Point 1: P1 = PMax * Output_pct_1
#   Point 2: P2 = PMax * Output_pct_2
#   Point 3: P3 = PMax * Output_pct_3 = PMax
#
#   HR_avg_0:  Average heat rate from 0 to P0 (BTU/kWh) - used for no-load cost
#   HR_Incr_1: Incremental HR from P0 to P1 (BTU/kWh)
#   HR_Incr_2: Incremental HR from P1 to P2 (BTU/kWh)
#   HR_Incr_3: Incremental HR from P2 to P3 (BTU/kWh)
#
# Marginal cost in segment k ($/MWh):
#   mc_k = HR_Incr_k (BTU/kWh) * Fuel_Price ($/MMBTU) / 1000 + VOM ($/MWh)
#
# pandapower PWL format: [[p_from, p_to, cost_per_mw], ...]

print("\n  Adding generators...")

def is_renewable(row):
    """Check if generator is a zero-cost renewable or storage."""
    cat = str(row.get("Category", ""))
    return any(r in cat for r in ["Solar", "Wind", "Hydro", "Storage"])


def build_pwl_points(row):
    """Build pandapower PWL cost points from the 4-point heat rate curve.
    Returns list of [p_from_mw, p_to_mw, cost_per_mw] segments,
    or None for zero-cost generators.

    NOTE on PMin=0 design decision:
    We set min_p_mw=0 for all generators (DC-OPF without unit commitment).
    In a real system you would turn generators OFF at low load rather than
    run them at PMin. Since we do not model commitment (binary on/off), we
    allow every generator to dispatch down to 0 MW. The cost curve is
    therefore extended to start at 0, using the first segment's marginal cost
    for the [0, PMin] range.  This gives proper merit-order dispatch and
    realistic LMPs (vs. forcing all units ON at PMin, which makes renewables
    the marginal resource and produces LMP ≈ $0 for 96% of hours)."""

    if is_renewable(row):
        return None

    pmax = float(row["PMax MW"])
    fuel_price = float(row.get("Fuel Price $/MMBTU", 0)) * FUEL_PRICE_SCALE
    vom = float(row.get("VOM", 0))

    if pmax <= 0 or fuel_price <= 0:
        return None

    # Output points (MW): Output_pct_0 = PMin/PMax, ..., Output_pct_3 = 1.0
    p_points = []
    for i in range(4):
        col = f"Output_pct_{i}"
        p_points.append(float(row.get(col, 0)) * pmax)

    # Incremental heat rates (BTU/kWh) -> marginal cost ($/MWh)
    # Segment 1 starts at 0 (not PMin) so the cost curve covers [0, PMax]
    segments = []
    for i in range(1, 4):
        col = f"HR_incr_{i}"
        hr = 0.0
        try:
            hr = float(row.get(col, 0))
        except (ValueError, TypeError):
            pass
        mc = hr * fuel_price / 1000.0 + vom
        p_from = 0.0 if i == 1 else p_points[i - 1]   # first seg: 0 -> P1
        p_to   = p_points[i]
        if p_to > p_from:
            segments.append([p_from, p_to, mc])

    return segments if segments else None


# Identify the Ref bus from bus.csv (Bus Type column)
gen_count = 0
excluded_count = 0
pwl_count = 0
poly_count = 0

ref_buses = bus_df[bus_df["Bus Type"] == "Ref"]
if len(ref_buses) == 0:
    sys.exit("ERROR: No Ref bus found in bus.csv")
ref_bus_id = int(ref_buses.iloc[0]["Bus ID"])
print(f"    Ref bus from bus.csv: {ref_bus_id} ({ref_buses.iloc[0]['Bus Name']})")

# Find a thermal generator at the Ref bus for ext_grid
ref_gens = gen_df[(gen_df["Bus ID"] == ref_bus_id)]
ref_thermal = ref_gens[ref_gens["Category"].apply(
    lambda c: not any(r in str(c) for r in ["Solar", "Wind", "Hydro", "Storage"])
)]
ref_thermal = ref_thermal.sort_values("PMax MW", ascending=False)
if len(ref_thermal) == 0:
    sys.exit(f"ERROR: No thermal generator at Ref bus {ref_bus_id}")

# First thermal gen at Ref bus becomes ext_grid
slack_row = ref_thermal.iloc[0]
slack_uid = str(slack_row["GEN UID"])
slack_pmax = float(slack_row["PMax MW"])
slack_pmin = float(slack_row["PMin MW"])
slack_pwl = build_pwl_points(slack_row)

pp.create_ext_grid(
    net, bus=bus_id_to_pp[ref_bus_id], vm_pu=1.0,
    name=f"Slack_{slack_uid}",
    max_p_mw=slack_pmax, min_p_mw=0.0,   # 0 = no unit-commitment floor (DC-OPF)
)
if slack_pwl:
    pp.create_pwl_cost(net, element=0, et="ext_grid", points=slack_pwl)
    mc_str = f"${slack_pwl[0][2]:.1f}-${slack_pwl[-1][2]:.1f}"
    pwl_count += 1
else:
    fuel_price = float(slack_row.get("Fuel Price $/MMBTU", 0))
    hr_avg = float(slack_row.get("HR_avg_0", 10000)) / 1000.0
    mc = fuel_price * hr_avg + float(slack_row.get("VOM", 0))
    pp.create_poly_cost(net, element=0, et="ext_grid",
                        cp1_eur_per_mw=mc, cp0_eur=0)
    mc_str = f"${mc:.1f} (flat)"
    poly_count += 1

print(f"    Slack: {slack_uid} at bus {ref_bus_id}, "
      f"Pmax={slack_pmax:.0f} MW, cost={mc_str}/MWh")

# Now add all OTHER generators (skip the one we just used as slack)
for _, row in gen_df.iterrows():
    gen_uid = str(row["GEN UID"])
    bus_id  = int(row["Bus ID"])
    p_max   = float(row["PMax MW"])
    p_min   = float(row["PMin MW"])

    # Skip the generator we already used as slack
    if gen_uid == slack_uid:
        continue

    # Skip excluded generators
    if gen_uid in EXCLUDED_GENS:
        excluded_count += 1
        continue

    # Skip zero-capacity (synchronous condensers)
    if p_max <= 0:
        continue

    pp_bus = bus_id_to_pp[bus_id]
    pwl_points = build_pwl_points(row)

    # -- Add as controllable generator --
    # min_p_mw=0: DC-OPF without unit commitment — generators can dispatch
    # down to 0 MW instead of being forced to run at their physical PMin.
    # This gives proper merit-order dispatch and realistic LMPs.
    gen_idx = pp.create_gen(
        net, bus=pp_bus, p_mw=0.0,
        max_p_mw=p_max, min_p_mw=0.0,
        controllable=True, name=gen_uid,
    )

    if pwl_points:
        pp.create_pwl_cost(net, element=gen_idx, et="gen", points=pwl_points)
        pwl_count += 1
    else:
        pp.create_poly_cost(net, element=gen_idx, et="gen",
                            cp1_eur_per_mw=0.0, cp0_eur=0.0)
        poly_count += 1

    gen_count += 1

print(f"    Added 1 ext_grid + {gen_count} generators")
print(f"    Cost curves: {pwl_count} piecewise linear, {poly_count} polynomial (zero-cost)")
print(f"    Excluded: {excluded_count} ({', '.join(EXCLUDED_GENS.keys())})")

# Capacity summary
thermal_cap = sum(float(r["PMax MW"]) for _, r in gen_df.iterrows()
                  if not is_renewable(r) and r["GEN UID"] not in EXCLUDED_GENS)
renew_cap = sum(float(r["PMax MW"]) for _, r in gen_df.iterrows()
                if is_renewable(r) and r["GEN UID"] not in EXCLUDED_GENS)
print(f"    Thermal: {thermal_cap:.0f} MW | Renewable: {renew_cap:.0f} MW")


# ================================================================
# PART B: APPLY TIME-SERIES AND RUN DC-OPF (validation)
# ================================================================
print(f"\n{'=' * 70}")
print(f"PART B: DC-OPF Validation at Hour {TEST_HOUR} (peak load)")
print("=" * 70)

# -- B1: Load time-series files --
ts_path = os.path.join(RTS_PATH, "timeseries_data_files")
load_ts  = pd.read_csv(os.path.join(ts_path, "Load/DAY_AHEAD_regional_Load.csv"))
wind_ts  = pd.read_csv(os.path.join(ts_path, "WIND/DAY_AHEAD_wind.csv"))
pv_ts    = pd.read_csv(os.path.join(ts_path, "PV/DAY_AHEAD_pv.csv"))
rtpv_ts  = pd.read_csv(os.path.join(ts_path, "RTPV/DAY_AHEAD_rtpv.csv"))
hydro_ts = pd.read_csv(os.path.join(ts_path, "Hydro/DAY_AHEAD_hydro.csv"))
csp_ts   = pd.read_csv(os.path.join(ts_path, "CSP/DAY_AHEAD_Natural_Inflow.csv"))

print(f"  Time-series: {len(load_ts)} hours ({len(load_ts)/24:.0f} days)")

# -- B2: Compute base loads per area (for scaling) --
area_base_load = {}
for area in sorted(set(bus_area.values())):
    area_buses = [bid for bid, a in bus_area.items() if a == area]
    area_base_load[area] = bus_df[bus_df["Bus ID"].isin(area_buses)]["MW Load"].sum()
    print(f"  Area {area} base load: {area_base_load[area]:.0f} MW")

# -- B3: Scale loads for this hour --
hour = TEST_HOUR
for _, row in bus_df.iterrows():
    bus_id = int(row["Bus ID"])
    area   = bus_area[bus_id]
    base_p = float(row["MW Load"])
    if base_p > 0:
        scale = float(load_ts.iloc[hour][str(area)]) / area_base_load[area]
        mask = net.load.name == f"Load_{bus_id}"
        if mask.any():
            net.load.loc[mask, "p_mw"]   = base_p * scale
            net.load.loc[mask, "q_mvar"] = float(row["MVAR Load"]) * scale

print(f"\n  Hour {hour}: Total load = {net.load.p_mw.sum():.0f} MW")

# -- B4: Set renewable max output from time-series --
gen_uid_to_pp = {}
for i, row in net.gen.iterrows():
    gen_uid_to_pp[row["name"]] = ("gen", i)
for i, row in net.ext_grid.iterrows():
    gen_uid_to_pp[row["name"].replace("Slack_", "")] = ("ext", i)

def apply_renewable_timeseries(ts_df, keyword):
    """Set max_p_mw for renewable generators from their time-series."""
    matched = 0
    for col in [c for c in ts_df.columns if keyword in c]:
        if col in gen_uid_to_pp:
            kind, idx = gen_uid_to_pp[col]
            val = max(float(ts_df.iloc[hour][col]), 0)
            if kind == "ext":
                net.ext_grid.at[idx, "max_p_mw"] = max(val, 0.1)
            else:
                net.gen.at[idx, "max_p_mw"] = val
                net.gen.at[idx, "p_mw"] = min(net.gen.at[idx, "p_mw"], val)
            matched += 1
    return matched

n_wind  = apply_renewable_timeseries(wind_ts,  "WIND")
n_pv    = apply_renewable_timeseries(pv_ts,    "PV")
n_rtpv  = apply_renewable_timeseries(rtpv_ts,  "RTPV")
n_hydro = apply_renewable_timeseries(hydro_ts, "HYDRO")
n_csp   = apply_renewable_timeseries(csp_ts,   "CSP")

def ts_total(ts_df, keyword, h):
    return sum(float(ts_df.iloc[h][c]) for c in ts_df.columns if keyword in c)

wind_mw  = ts_total(wind_ts,  "WIND",  hour)
pv_mw    = ts_total(pv_ts,    "PV",    hour)
rtpv_mw  = ts_total(rtpv_ts,  "RTPV",  hour)
hydro_mw = ts_total(hydro_ts, "HYDRO", hour)
csp_mw   = ts_total(csp_ts,   "CSP",   hour)
renew_total = wind_mw + pv_mw + rtpv_mw + hydro_mw + csp_mw

print(f"  Renewables: Wind={wind_mw:.0f}, PV={pv_mw:.0f}, RTPV={rtpv_mw:.0f}, "
      f"Hydro={hydro_mw:.0f}, CSP={csp_mw:.0f} -> Total={renew_total:.0f} MW")
print(f"  Thermal needed: ~{net.load.p_mw.sum() - renew_total:.0f} MW")

# -- B5: Run DC-OPF --
print(f"\n  Running DC-OPF ({LINE_LIMIT_SCALE*100:.0f}% line limits)...")
try:
    pp.rundcopp(net)
    print("  CONVERGED!")
except Exception as e:
    print(f"  FAILED: {e}")
    total_cap = net.gen.max_p_mw.sum() + net.ext_grid.max_p_mw.sum()
    print(f"  Gen capacity: {total_cap:.0f} MW vs load: {net.load.p_mw.sum():.0f} MW")
    sys.exit(1)

# -- B6: Display results --
lmps = net.res_bus["lam_p"].values

print(f"\n  +---------------------------------------+")
print(f"  |  LMP SUMMARY  (Hour {hour})             |")
print(f"  +---------------------------------------+")
print(f"  |  Min:    ${lmps.min():>8.2f} /MWh             |")
print(f"  |  Max:    ${lmps.max():>8.2f} /MWh             |")
print(f"  |  Mean:   ${lmps.mean():>8.2f} /MWh             |")
print(f"  |  Spread: ${lmps.max()-lmps.min():>8.2f} /MWh             |")
print(f"  +---------------------------------------+")

for area in sorted(set(bus_area.values())):
    area_pp_idx = [bus_id_to_pp[bid] for bid, a in bus_area.items() if a == area]
    a_lmps = lmps[area_pp_idx]
    print(f"  Area {area}: ${a_lmps.min():.2f} - ${a_lmps.max():.2f} "
          f"(mean ${a_lmps.mean():.2f})")

# Congestion
loadings = net.res_line["loading_percent"].values
cong_lines = np.where(loadings > 95)[0]
print(f"\n  Congested lines (>95%): {len(cong_lines)}")
for idx in cong_lines:
    fb = net.line.at[idx, "from_bus"]
    tb = net.line.at[idx, "to_bus"]
    print(f"    {net.line.at[idx, 'name']:6s}: "
          f"{net.bus.at[fb, 'name']} -> {net.bus.at[tb, 'name']}, "
          f"{loadings[idx]:.1f}%")

print(f"\n  Top 5 loaded lines:")
for idx in np.argsort(loadings)[::-1][:5]:
    print(f"    {net.line.at[idx, 'name']:6s}: {loadings[idx]:.1f}%")

# HVDC flow
if len(net.res_dcline) > 0:
    for i, row in net.res_dcline.iterrows():
        print(f"\n  HVDC '{net.dcline.at[i, 'name']}': "
              f"P = {row['p_from_mw']:.1f} MW -> {row['p_to_mw']:.1f} MW "
              f"(loss = {row['pl_mw']:.1f} MW)")

# Full LMP table
print(f"\n  {'Bus':20s} {'Area':>4s} {'LMP ($/MWh)':>12s}")
print(f"  {'-'*38}")
for bus_id in sorted(bus_id_to_pp.keys()):
    pp_idx = bus_id_to_pp[bus_id]
    name = net.bus.at[pp_idx, "name"]
    area = bus_area[bus_id]
    print(f"  {name:20s} {area:>4d} {lmps[pp_idx]:>12.2f}")

# -- B7: Save outputs --
out_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(out_dir, exist_ok=True)

net_path     = os.path.join(out_dir, "step1_network.json")
mapping_path = os.path.join(out_dir, "step1_mapping.json")

pp.to_json(net, net_path)
print(f"\n  Network saved to: {net_path}")

mapping = {
    "bus_id_to_pp": {str(k): int(v) for k, v in bus_id_to_pp.items()},
    "bus_area": {str(k): int(v) for k, v in bus_area.items()},
    "area_base_load": {str(k): float(v) for k, v in area_base_load.items()},
    "line_limit_scale": LINE_LIMIT_SCALE,
    "excluded_gens": list(EXCLUDED_GENS.keys()),
}
with open(mapping_path, "w") as f:
    json.dump(mapping, f, indent=2)
print(f"  Bus mapping saved to: {mapping_path}")

print(f"\n{'=' * 70}")
print("STEP 1 COMPLETE")
print(f"  Network: {len(net.bus)} buses | {len(net.line)} lines | "
      f"{len(net.trafo)} trafos | {len(net.gen)+1} generators")
print("Next: Step 2 - Pre-compute LMPs for all 8,784 hours")
print("=" * 70)










