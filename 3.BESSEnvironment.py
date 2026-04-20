"""
Step 3: Gymnasium environment for degradation-aware BESS dispatch.

The environment exposes a 38-dimensional state with battery inventory,
local price history, system context, degradation state, and instantaneous
feasibility bounds from the PTDF safety layer.
"""

import math
import os
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

# ================================================================
# BESS CONFIGURATION (all parameters in one place)
# ================================================================

# --- Battery parameters ---
BESS_BUS_RTS     = 117       # Default deployment bus: better causal learnability than bus 303
BESS_P_MAX_MW    = 25.0      # Power rating [MW]
BESS_E_CAP_MWH   = 100.0     # Energy capacity [MWh]
BESS_ETA_CH      = 0.95      # Charging efficiency (one-way)
BESS_ETA_DIS     = 0.95      # Discharging efficiency (one-way)
BESS_SOC_MIN     = 0.10      # Min SoC as fraction of E_cap
BESS_SOC_MAX     = 0.90      # Max SoC as fraction of E_cap
BESS_SOC_INIT    = 0.875     # Evaluation reset SoC

# --- Degradation model — Xu et al. (2017), IEEE Trans. Power Systems ---
# Phi(d) = BESS_PHI_XI * d^BESS_PHI_ALPHA   [Xu Eq. 24, NMC 18650 empirical fit]
BESS_PHI_XI      = 5.24e-4   # Life-loss coefficient  (Laresgoiti et al. 2015)
BESS_PHI_ALPHA   = 2.03      # Life-loss exponent     (convex: deeper = costlier)
BESS_R_PER_KWH   = 100.0     # Battery replacement cost [$/kWh]
BESS_J_DEG       = 4         # Number of piecewise-linear degradation segments

# --- Episode ---
EPISODE_LEN      = 168       # Timesteps per episode (1 week)
DT               = 1.0       # Timestep duration [hours]

# --- State normalization constants ---
LMP_NORM         = 100.0     # LMP normalization [$/MWh] (max observed ~$94)
PV_MAX_MW        = 2500.0    # System PV normalization [MW]  (max observed ~2421)
WIND_MAX_MW      = 2600.0    # Reference wind normalization [MW]
REWARD_SCALE     = 0.01      # Reward scale used by the RL agent

# --- PTDF sensitivity threshold ---
PTDF_THRESH      = 1e-5      # Lines with |PTDF| < thresh are treated as insensitive

# --- System load normalization ---
# RTS-GMLC 2020: peak 3-area combined load ~8900 MW. Use 9500 as conservative ceiling.
SYSTEM_LOAD_MAX  = 9500.0    # [MW] normalization for system total load

# --- Paths ---
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(_HERE, "outputs")


# ================================================================
# HELPER: Compute degradation segment costs
# ================================================================

def _compute_segment_costs(
    r_per_kwh: float = BESS_R_PER_KWH,
    eta_dis:   float = BESS_ETA_DIS,
    j_deg:     int   = BESS_J_DEG,
) -> np.ndarray:
    """
    Compute Xu-style marginal degradation costs for each segment.
    """
    Phi = lambda d: BESS_PHI_XI * d ** BESS_PHI_ALPHA
    costs = np.zeros(j_deg)
    for j in range(1, j_deg + 1):
        delta_phi = Phi(j / j_deg) - Phi((j - 1) / j_deg)
        costs[j - 1] = (r_per_kwh * 1000.0 / eta_dis) * j_deg * delta_phi
    return costs


# ================================================================
# MAIN ENVIRONMENT CLASS
# ================================================================

class BESSEnv(gym.Env):
    """
    Gymnasium environment for degradation-aware BESS dispatch on the RTS study case.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bess_bus_rts: int   = BESS_BUS_RTS,
        p_max_mw:     float = BESS_P_MAX_MW,
        e_cap_mwh:    float = BESS_E_CAP_MWH,
        eta_ch:       float = BESS_ETA_CH,
        eta_dis:      float = BESS_ETA_DIS,
        soc_min:      float = BESS_SOC_MIN,
        soc_max:      float = BESS_SOC_MAX,
        soc_init:     float = BESS_SOC_INIT,
        r_per_kwh:    float = BESS_R_PER_KWH,
        j_deg:        int   = BESS_J_DEG,
        episode_len:  int   = EPISODE_LEN,
        lmp_norm:     float = LMP_NORM,
        reward_scale: float = REWARD_SCALE,
        seed:         Optional[int] = None,
    ):
        super().__init__()

        # --- Store BESS parameters ---
        self.P_MAX     = p_max_mw
        self.E_CAP     = e_cap_mwh
        self.ETA_CH    = eta_ch
        self.ETA_DIS   = eta_dis
        self.SOC_MIN   = soc_min * e_cap_mwh    # [MWh] absolute lower bound
        self.SOC_MAX   = soc_max * e_cap_mwh    # [MWh] absolute upper bound
        self.SOC_INIT  = soc_init * e_cap_mwh   # [MWh] reset SoC
        self.J_DEG     = j_deg
        self.SEG_SIZE  = e_cap_mwh / j_deg      # [MWh] energy per segment
        self.EPS_LEN   = episode_len
        self.LMP_NORM  = lmp_norm
        self.RWD_SCALE = reward_scale

        # No terminal SoC salvage term is applied.

        # --- Compute degradation costs ---
        self.c_deg = _compute_segment_costs(
            r_per_kwh=r_per_kwh,
            eta_dis=eta_dis,
            j_deg=j_deg,
        )  # shape (J,) in $/MWh extracted from battery (Xu et al. 2017)

        # --- Load precomputed data from Step 2 ---
        self._load_data(bess_bus_rts)

        # --- Gymnasium spaces ---
        # State: 38-dimensional continuous.
        #   SoC (1) + LMP-hist-24 (24) + LMP-now (1) + PV (1) + step (1)
        #   + deg-segs (j_deg) + safety-bounds (2) + sin/cos-dow (2)
        #   + system-load (1) + wind-total (1)  =  34 + j_deg
        n_obs = (
            1          # SoC
            + 24       # LMP history  (lag-24 .. lag-1)
            + 1        # LMP current
            + 1        # system PV
            + 1        # episode progress
            + j_deg    # degradation segment occupancies
            + 2        # safety bounds p_lo, p_hi
            + 2        # day-of-week  sin + cos  (cyclic encoding)
            + 1        # system load
            + 1        # system wind
        )   # = 38 for j_deg=4
        self.observation_space = spaces.Box(
            low=-np.ones(n_obs, dtype=np.float32),
            high=np.ones(n_obs, dtype=np.float32),
            dtype=np.float32,
        )
        # Action: scalar in [-1, 1] -> rescaled to [-P_max, +P_max]
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # --- State variables (initialized in reset) ---
        self.soc:         float = self.SOC_INIT
        self.step_in_ep:  int   = 0
        self.global_t:    int   = 0          # hour index in the 8784-hour year
        self.lmp_history: np.ndarray = np.zeros(24, dtype=np.float32)

        # --- Optional RNG ---
        self._np_rng = np.random.default_rng(seed)

        # Print summary
        print(self._summary())

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------

    def _load_data(self, bess_bus_rts: int) -> None:
        """Load Step 2 outputs and identify BESS bus indices."""

        def load(name):
            path = os.path.join(OUTPUTS_DIR, name)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required Step 2 output missing: {path}\n"
                    f"Run 2.PrecomputeLMPs.py first."
                )
            return np.load(path)

        self.lmps       = load("step2_lmps.npy")           # (8784, 73) float32 $/MWh
        self.line_flows = load("step2_line_flows.npy")     # (8784, 120) float32 MW
        self.ptdf       = load("step2_ptdf.npy").astype(np.float32)  # (120, 73)
        self.converged     = load("step2_converged.npy")       # (8784,) bool
        self.f_lmax        = load("step2_line_limits.npy")     # (120,) float32 MW
        self.pv_total      = load("step2_pv_total.npy")        # (8784,) float32 MW

        with open(os.path.join(OUTPUTS_DIR, "step2_metadata.json")) as f:
            meta = json.load(f)

        # Map RTS bus ID -> column index in the (8784, 73) and (120, 73) arrays
        bus_order = meta["bus_order_rts_ids"]
        bus_id_to_col = {bus_id: i for i, bus_id in enumerate(bus_order)}

        if bess_bus_rts not in bus_id_to_col:
            raise ValueError(
                f"BESS bus {bess_bus_rts} not found in network. "
                f"Available buses: {bus_order}"
            )
        self.bess_col = bus_id_to_col[bess_bus_rts]   # column index in LMP/PTDF arrays
        self.bess_bus_rts = bess_bus_rts

        # PTDF column for BESS bus: sensitivity of each line to BESS injection
        self.ptdf_k = self.ptdf[:, self.bess_col]     # shape (120,)

        # Converged-hour index
        self.converged_hours = np.where(self.converged)[0]   # indices of valid hours

        # Day-aligned starts with a converged reset hour.
        N_HOURS = self.lmps.shape[0]
        day_starts = np.arange(0, N_HOURS - self.EPS_LEN, 24, dtype=np.int32)
        self.valid_starts = day_starts[self.converged[day_starts]]
        conv_i = self.converged.astype(np.int32)
        fail_per_start = self.EPS_LEN - np.convolve(
            conv_i,
            np.ones(self.EPS_LEN, dtype=np.int32),
            mode="valid",
        )
        self.fail_hours_per_episode = fail_per_start[self.valid_starts]

        # Forward temporal holdout with zero episode overlap.
        N_HOURS = self.lmps.shape[0]
        _h_split_raw = int(N_HOURS * 0.75)
        # Snap to nearest day boundary (multiple of 24)
        _h_split = (_h_split_raw // 24) * 24
        _train = self.valid_starts[self.valid_starts + self.EPS_LEN <= _h_split]
        _test  = self.valid_starts[self.valid_starts >= _h_split]
        self.train_starts = _train
        self.test_starts  = _test
        self._mode        = "train"    # default: sample from training pool

        # Normalization constants from data
        self.pv_max = float(np.nanmax(self.pv_total))  # for state normalization

        # System total load across the three RTS areas.
        _ts_base = os.path.join(_HERE, "RTS-GMLC", "RTS_Data", "timeseries_data_files")
        import pandas as _pd
        _load_df = _pd.read_csv(os.path.join(_ts_base, "Load", "DAY_AHEAD_regional_Load.csv"))
        self.system_load = (
            _load_df["1"].values + _load_df["2"].values + _load_df["3"].values
        ).astype(np.float32)[:self.lmps.shape[0]]   # trim to 8784 hours

        # --- System-wide wind output ---
        # Wind is a useful proxy for cheap charging hours.
        _wind_df  = _pd.read_csv(os.path.join(_ts_base, "WIND", "DAY_AHEAD_wind.csv"))
        _wind_cols = [c for c in _wind_df.columns
                      if c not in ("Year", "Month", "Day", "Period")]
        self.wind_total = (
            _wind_df[_wind_cols].sum(axis=1).values
        ).astype(np.float32)[:self.lmps.shape[0]]
        self.wind_max = float(np.nanmax(self.wind_total))   # for normalization

        print(f"  Loaded Step 2 data: {self.lmps.shape[0]} hours, {self.lmps.shape[1]} buses")
        print(f"  BESS bus {bess_bus_rts} -> column {self.bess_col}")
        print(f"  PTDF[l,{bess_bus_rts}] nonzero: {(np.abs(self.ptdf_k) > PTDF_THRESH).sum()} lines")
        print(f"  Valid episode starts: {len(self.valid_starts)} "
              f"(train={len(self.train_starts)}, test={len(self.test_starts)})")
        if len(self.fail_hours_per_episode) > 0:
            print(
                "  Failed OPF hrs / episode (imputed): "
                f"mean={self.fail_hours_per_episode.mean():.2f}, "
                f"p95={np.percentile(self.fail_hours_per_episode, 95):.1f}, "
                f"max={int(self.fail_hours_per_episode.max())}"
            )

    # ------------------------------------------------------------------
    # GYMNASIUM API
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Switch between the train and test episode-start pools."""
        if mode not in ("train", "test"):
            raise ValueError(f"mode must be 'train' or 'test', got {mode!r}")
        self._mode = mode

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment and return the initial observation.
        """
        super().reset(seed=seed)

        # Pick start hour: exact override takes priority over random pool sample
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        if options is not None and "start_t" in options:
            # Fixed start and SoC for deterministic evaluation.
            self.global_t = int(options["start_t"])
            self.soc      = self.SOC_INIT
        else:
            # Randomize both start hour and initial SoC during training.
            pool = self.test_starts if self._mode == "test" else self.train_starts
            idx  = self._np_rng.integers(0, len(pool))
            self.global_t = int(pool[idx])
            soc_lo = self.SOC_MIN + 5.0   # 15 MWh  — small buffer above SOC_MIN
            soc_hi = self.SOC_INIT        # 87.5 MWh — matches eval starting SoC
            self.soc = float(self._np_rng.uniform(soc_lo, soc_hi))

        self.step_in_ep  = 0
        self.lmp_history = np.zeros(24, dtype=np.float32)

        # Back-fill the pre-episode LMP history from the stored Step 2 series.
        for lag in range(1, 25):
            h = self.global_t - lag
            if h >= 0:
                self.lmp_history[lag - 1] = float(self.lmps[h, self.bess_col])

        obs = self._get_obs()
        info = {
            "start_hour": self.global_t,
            "soc_init_mwh": self.soc,
            "episode_len": self.EPS_LEN,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Apply one timestep of BESS dispatch.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # 1. Rescale action from [-1,1] to [-P_max, +P_max] MW
        p_hat = float(action[0]) * self.P_MAX

        # 2. Safety QP: project onto feasible set (analytical 1D clipping)
        p_safe, p_lo, p_hi = self._safety_project(p_hat)

        # 3. Update SoC
        soc_prev = self.soc
        if p_safe >= 0:   # discharging
            e_from_battery = p_safe * DT / self.ETA_DIS  # MWh taken from battery
            self.soc -= e_from_battery
        else:             # charging
            e_to_battery = abs(p_safe) * self.ETA_CH * DT  # MWh into battery
            self.soc += e_to_battery

        # Clip SoC to hard bounds (safety net for floating point drift)
        self.soc = float(np.clip(self.soc, self.SOC_MIN - 1e-6, self.SOC_MAX + 1e-6))

        # 4. Get LMP at current timestep
        lmp_t = float(self.lmps[self.global_t, self.bess_col])

        # 5. Compute degradation cost (only when discharging)
        deg_cost = 0.0
        if p_safe > 0:
            e_from_battery = p_safe * DT / self.ETA_DIS
            deg_cost = self._compute_deg_cost(soc_prev, e_from_battery)

        # 6. Compute reward: revenue minus degradation cost
        revenue  = lmp_t * p_safe * DT    # $ (positive = earning, negative = paying)
        reward   = (revenue - deg_cost) * self.RWD_SCALE

        # 7. Update LMP history (shift left, add current LMP)
        self.lmp_history = np.roll(self.lmp_history, shift=1)
        self.lmp_history[0] = lmp_t

        # 8. Advance time
        self.step_in_ep += 1
        self.global_t   += 1

        # 9. Check episode termination
        terminated = self.step_in_ep >= self.EPS_LEN
        truncated  = False


        # 10. Get next observation.
        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_obs()

        info = {
            "p_hat_mw":     p_hat,
            "p_safe_mw":    p_safe,
            "p_lo_mw":      p_lo,
            "p_hi_mw":      p_hi,
            "soc_mwh":      self.soc,
            "lmp_per_mwh":  lmp_t,
            "revenue_$":    revenue,
            "deg_cost_$":   deg_cost,
            "reward_raw_$": revenue - deg_cost,
            "clipped":      abs(p_safe - p_hat) > 1e-4,
            "global_hour":  self.global_t - 1,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # SAFETY LAYER: 1D Analytical QP Projection
    # ------------------------------------------------------------------

    def _safety_project(self, p_hat: float) -> Tuple[float, float, float]:
        """
        Project the proposed battery power onto the current feasible interval.
        """
        # -- (a) Power rating bounds --
        p_lo = -self.P_MAX
        p_hi =  self.P_MAX

        # -- (b) SoC bounds --
        # Discharge (P > 0): SoC_next = SoC - P*dt/eta_dis >= SOC_MIN
        #   => P <= (SoC - SOC_MIN) * eta_dis / dt
        p_hi = min(p_hi, (self.soc - self.SOC_MIN) * self.ETA_DIS / DT)

        # Charge (P < 0): SoC_next = SoC + |P|*eta_ch*dt <= SOC_MAX
        #   => -P <= (SOC_MAX - SoC) * (1 / (eta_ch * dt))
        #   => P >= -(SOC_MAX - SoC) / (eta_ch * dt)
        p_lo = max(p_lo, -(self.SOC_MAX - self.soc) / (self.ETA_CH * DT))

        # -- (c) Line flow bounds (PTDF-based) --
        # Current baseline line flows at this timestep
        f_base = self.line_flows[self.global_t]   # (120,) MW, baseline without BESS

        # Active lines: |PTDF[l, k]| > threshold
        active = np.abs(self.ptdf_k) > PTDF_THRESH

        if active.any():
            ptdf_act = self.ptdf_k[active]   # (n_act,)
            f_act    = f_base[active]         # (n_act,) MW
            f_lmax_act = self.f_lmax[active]  # (n_act,) MW

            # Upper constraint: f_base + ptdf * P <= +F_max  =>  P <= (F_max - f_base) / ptdf
            # Lower constraint: f_base + ptdf * P >= -F_max  =>  P >= (-F_max - f_base) / ptdf
            # (Direction flips when ptdf < 0)

            upper_from_lines = np.where(
                ptdf_act > 0,
                (f_lmax_act - f_act) / ptdf_act,    # ptdf > 0 -> upper bound on P
                (-f_lmax_act - f_act) / ptdf_act,   # ptdf < 0 -> also upper bound (division flips)
            )
            lower_from_lines = np.where(
                ptdf_act > 0,
                (-f_lmax_act - f_act) / ptdf_act,   # ptdf > 0 -> lower bound on P
                (f_lmax_act - f_act) / ptdf_act,    # ptdf < 0 -> also lower bound (division flips)
            )

            p_hi = min(p_hi, float(np.min(upper_from_lines)))
            p_lo = max(p_lo, float(np.max(lower_from_lines)))

        # Feasibility guard for numerical edge cases.
        if p_lo > p_hi:
            p_lo = 0.0
            p_hi = 0.0

        # Exact 1D projection.
        p_safe = float(np.clip(p_hat, p_lo, p_hi))

        return p_safe, p_lo, p_hi

    # ------------------------------------------------------------------
    # DEGRADATION COST
    # ------------------------------------------------------------------

    def _compute_deg_cost(self, soc: float, e_dis: float) -> float:
        """Compute discharge degradation cost by draining shallow segments first."""
        deg_cost   = 0.0
        remaining  = min(e_dis, soc - self.SOC_MIN)  # can't go below SOC_MIN
        remaining  = max(remaining, 0.0)

        soc_cursor = soc  # current level we're drawing from

        for j in range(1, self.J_DEG + 1):
            if remaining <= 1e-9:
                break

            # Segment j covers SoC in [(1 - j/J)*E, (1 - (j-1)/J)*E]
            seg_upper = self.E_CAP - (j - 1) * self.SEG_SIZE   # e.g., j=1: E_CAP
            seg_lower = self.E_CAP - j * self.SEG_SIZE          # e.g., j=1: 0.75*E_CAP

            # How much of this segment is currently "filled" (available to discharge)
            available = max(0.0, min(soc_cursor, seg_upper) - seg_lower)

            # Draw from this segment
            draw = min(remaining, available)

            deg_cost  += self.c_deg[j - 1] * draw   # c_deg in $/MWh, draw in MWh -> $
            remaining -= draw
            soc_cursor -= draw

        return deg_cost

    # ------------------------------------------------------------------
    # OBSERVATION BUILDER
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        Build the 38-dimensional observation vector for the current timestep.
        """
        N_HOURS = self.lmps.shape[0]
        t = self.global_t

        # -- SoC (normalized to [0, 1] over full E_cap) --
        soc_norm = self.soc / self.E_CAP

        # -- Current LMP --
        lmp_t    = float(self.lmps[t, self.bess_col]) if t < N_HOURS else 0.0
        lmp_norm = lmp_t / self.LMP_NORM

        # LMP history oldest-first for the Conv1D encoder.
        lmp_hist_norm = self.lmp_history[::-1] / self.LMP_NORM   # oldest → newest

        # -- System PV (proxy for time-of-day solar output) --
        pv_norm = float(self.pv_total[t]) / max(self.pv_max, 1.0) if t < N_HOURS else 0.0

        # -- Episode progress (helps finite-horizon policy manage terminal SoC) --
        step_norm = float(self.step_in_ep) / max(float(self.EPS_LEN), 1.0)

        # -- Degradation segment occupancy (tells agent which cost tier is active) --
        seg_occ = self._get_seg_occupancy()   # (J,) in [0, 1]

        # Current feasible dispatch interval from the safety layer.
        _, p_lo, p_hi = self._safety_project(0.0)
        p_lo_norm = p_lo / self.P_MAX   # in [-1, 0]
        p_hi_norm = p_hi / self.P_MAX   # in [0, 1]

        # Day-of-week with a cyclic sin/cos encoding.
        dow     = float(((t // 24) + 2) % 7)   # 0=Mon … 6=Sun
        sin_dow = math.sin(2.0 * math.pi * dow / 7.0)
        cos_dow = math.cos(2.0 * math.pi * dow / 7.0)

        # -- System total load (3-area combined MW demand) --
        load_norm = (
            float(self.system_load[t]) / SYSTEM_LOAD_MAX
            if t < len(self.system_load) else 0.0
        )

        # System-wide wind generation.
        wind_norm = (
            float(self.wind_total[t]) / max(self.wind_max, 1.0)
            if t < len(self.wind_total) else 0.0
        )

        obs = np.concatenate([
            [soc_norm],    # 1    → obs[0]
            lmp_hist_norm, # 24   → obs[1..24]  (lag-24 .. lag-1, oldest-first)
            [lmp_norm],    # 1    → obs[25]     (current LMP, rightmost for causal conv)
            [pv_norm],     # 1    → obs[26]
            [step_norm],   # 1    → obs[27]
            seg_occ,       # J=4  → obs[28..31]
            [p_lo_norm],   # 1    → obs[32]
            [p_hi_norm],   # 1    → obs[33]
            [sin_dow],     # 1    → obs[34]
            [cos_dow],     # 1    → obs[35]
            [load_norm],   # 1    → obs[36]
            [wind_norm],   # 1    → obs[37]
        ]).astype(np.float32)   # total: 38

        # Clip to valid observation-space range (guards against minor float drift)
        return np.clip(obs, -1.0, 1.0)

    def _get_seg_occupancy(self) -> np.ndarray:
        """Return the normalized occupancy of each degradation segment."""
        occ = np.zeros(self.J_DEG, dtype=np.float32)
        for j in range(1, self.J_DEG + 1):
            seg_upper = self.E_CAP - (j - 1) * self.SEG_SIZE
            seg_lower = self.E_CAP - j * self.SEG_SIZE
            energy_in_seg = max(0.0, min(self.soc, seg_upper) - seg_lower)
            occ[j - 1] = energy_in_seg / self.SEG_SIZE   # normalized to [0, 1]
        return occ

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    def _summary(self) -> str:
        """Return a human-readable configuration summary."""
        lines = [
            "\n" + "=" * 65,
            "NC-SafeRL BESS Environment Configuration",
            "=" * 65,
            f"  BESS bus:          {self.bess_bus_rts} (column {self.bess_col})",
            f"  Power rating:      {self.P_MAX:.0f} MW",
            f"  Energy capacity:   {self.E_CAP:.0f} MWh (4-hour battery)",
            f"  Efficiency:        {self.ETA_CH:.0%} ch / {self.ETA_DIS:.0%} dis "
            f"(round-trip {self.ETA_CH * self.ETA_DIS:.1%})",
            f"  SoC bounds:        [{self.SOC_MIN:.1f}, {self.SOC_MAX:.1f}] MWh "
            f"([{self.SOC_MIN/self.E_CAP:.0%}, {self.SOC_MAX/self.E_CAP:.0%}])",
            f"  Initial SoC:       {self.SOC_INIT:.1f} MWh ({self.SOC_INIT/self.E_CAP:.0%})",
            "",
            f"  Degradation (Xu et al. 2017 IEEE TPS, J={self.J_DEG} segments):",
            f"    Segment costs [$/MWh extracted]:",
        ]
        for j in range(self.J_DEG):
            soc_hi = (1 - j / self.J_DEG) * self.E_CAP
            soc_lo = (1 - (j + 1) / self.J_DEG) * self.E_CAP
            lines.append(
                f"      j={j+1}  SoC=[{soc_lo:.0f},{soc_hi:.0f}] MWh  "
                f"c_j = ${self.c_deg[j]:.2f}/MWh"
            )
        lines += [
            "",
            f"  State dim:         {self.observation_space.shape[0]}  (SoC+hist24+LMP+PV+step+segs4+p_lo+p_hi+sin/cos_dow+load+wind)",
            f"  Action dim:        {self.action_space.shape[0]}",
            f"  Episode length:    {self.EPS_LEN} steps (1 week)",
            f"  Reward scale:      {self.RWD_SCALE} (raw $/h -> scaled units)",
            "=" * 65,
        ]
        return "\n".join(lines)



# ================================================================
# FACTORY / REGISTRATION
# ================================================================

def make_bess_env(seed: Optional[int] = None, **kwargs) -> BESSEnv:
    """Convenience wrapper that returns a configured BESSEnv instance."""
    return BESSEnv(seed=seed, **kwargs)





# ================================================================
# PTDF BINDING ANALYSIS
# ================================================================

def ptdf_binding_analysis(env: BESSEnv, save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Summarize how often PTDF limits bind at full BESS charge or discharge.
    """
    # ------------------------------------------------------------------
    # 1. Active PTDF lines (same threshold as _safety_project)
    # ------------------------------------------------------------------
    active = np.abs(env.ptdf_k) > PTDF_THRESH   # (120,) bool
    ptdf_act  = env.ptdf_k[active]               # (n_act,)
    f_lmax_act = env.f_lmax[active]              # (n_act,) MW

    n_act = int(active.sum())

    # ------------------------------------------------------------------
    # 2. Base flows for all converged hours: shape (N_conv, n_act)
    # ------------------------------------------------------------------
    conv_idx = env.converged_hours               # integer indices into 8784-hr array
    F_base_act = env.line_flows[conv_idx][:, active]   # (N_conv, n_act) MW
    N = len(conv_idx)

    # ------------------------------------------------------------------
    # 3. PTDF-derived bounds at every converged hour (vectorized)
    #    Mirrors _safety_project logic exactly
    # ------------------------------------------------------------------
    # upper_from_lines[t, l]: max discharge power before line l overflows
    upper_from_lines = np.where(
        ptdf_act[np.newaxis, :] > 0,
        ( f_lmax_act - F_base_act) / ptdf_act,   # ptdf > 0
        (-f_lmax_act - F_base_act) / ptdf_act,   # ptdf < 0
    )   # (N_conv, n_act)

    # lower_from_lines[t, l]: min (most negative) charge power
    lower_from_lines = np.where(
        ptdf_act[np.newaxis, :] > 0,
        (-f_lmax_act - F_base_act) / ptdf_act,   # ptdf > 0
        ( f_lmax_act - F_base_act) / ptdf_act,   # ptdf < 0
    )   # (N_conv, n_act)

    # Tightest bound per hour
    ptdf_p_hi = np.min(upper_from_lines, axis=1)   # (N_conv,) tightest discharge cap
    ptdf_p_lo = np.max(lower_from_lines, axis=1)   # (N_conv,) tightest charge floor

    # ------------------------------------------------------------------
    # 4. Does PTDF bind vs. power rating?
    # ------------------------------------------------------------------
    # Discharge: PTDF binds when its cap is tighter than P_MAX
    dis_ptdf_binds = ptdf_p_hi < env.P_MAX   # (N_conv,) bool

    # Charge: PTDF binds when its floor is less negative than -P_MAX
    ch_ptdf_binds  = ptdf_p_lo > -env.P_MAX  # (N_conv,) bool

    n_dis  = int(dis_ptdf_binds.sum())
    n_ch   = int(ch_ptdf_binds.sum())
    n_any  = int((dis_ptdf_binds | ch_ptdf_binds).sum())

    # ------------------------------------------------------------------
    # 5. Per-line binding frequencies
    # ------------------------------------------------------------------
    # Which lines cause the tightest bound each hour?
    # Discharge: line l binds if its upper_from_lines[t,l] == ptdf_p_hi[t]
    # (within floating-point tolerance)
    dis_binding_mask = (
        upper_from_lines <= ptdf_p_hi[:, np.newaxis] + 1e-3
    ) & dis_ptdf_binds[:, np.newaxis]   # (N_conv, n_act)

    ch_binding_mask  = (
        lower_from_lines >= ptdf_p_lo[:, np.newaxis] - 1e-3
    ) & ch_ptdf_binds[:, np.newaxis]    # (N_conv, n_act)

    # Expand back to full 120-line arrays
    dis_bind_freq = np.zeros(120)
    ch_bind_freq  = np.zeros(120)
    dis_bind_freq[active] = dis_binding_mask.mean(axis=0)
    ch_bind_freq[active]  = ch_binding_mask.mean(axis=0)

    # Combined frequency for ranking
    combined_freq = dis_bind_freq + ch_bind_freq
    top_lines = np.argsort(combined_freq)[::-1][:10]

    # ------------------------------------------------------------------
    # 6. Base loading statistics (all 120 lines, converged hours)
    # ------------------------------------------------------------------
    F_base_all  = env.line_flows[conv_idx]                   # (N_conv, 120)
    loading_pct = np.abs(F_base_all) / np.maximum(env.f_lmax, 1e-3) * 100.0
    mean_base_loading = loading_pct.mean(axis=0)             # (120,)

    pct_gt80 = float((loading_pct.max(axis=1) > 80.0).mean() * 100.0)
    pct_gt90 = float((loading_pct.max(axis=1) > 90.0).mean() * 100.0)

    # ------------------------------------------------------------------
    # 7. Print report
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PTDF Constraint Binding Analysis")
    print(f"  BESS bus {env.bess_bus_rts} | P_MAX = {env.P_MAX:.0f} MW | "
          f"{n_act} active PTDF lines")
    print("=" * 65)

    print(f"\n  Converged hours analyzed : {N:,}")
    print(f"  PTDF binds at +P_MAX (discharge): {n_dis:,} / {N:,}  "
          f"({100*n_dis/N:.1f}%)")
    print(f"  PTDF binds at -P_MAX (charge)   : {n_ch:,} / {N:,}  "
          f"({100*n_ch/N:.1f}%)")
    print(f"  PTDF binds in either direction  : {n_any:,} / {N:,}  "
          f"({100*n_any/N:.1f}%)")

    print(f"\n  Base network loading (without BESS):")
    print(f"    Hours with any line > 80% loading : {pct_gt80:.1f}%")
    print(f"    Hours with any line > 90% loading : {pct_gt90:.1f}%")

    print(f"\n  Top 10 binding lines (by combined binding frequency):")
    print(f"    {'Line':>4}  {'PTDF_k':>9}  {'Dis%':>6}  {'Ch%':>6}  "
          f"{'F_max[MW]':>9}  {'MeanLoad%':>9}")
    print(f"    {'-'*4}  {'-'*9}  {'-'*6}  {'-'*6}  {'-'*9}  {'-'*9}")
    for l in top_lines:
        if combined_freq[l] < 1e-6:
            continue
        print(f"    {l:>4}  {env.ptdf_k[l]:>+9.4f}  "
              f"{100*dis_bind_freq[l]:>5.1f}%  "
              f"{100*ch_bind_freq[l]:>5.1f}%  "
              f"{env.f_lmax[l]:>9.1f}  "
              f"{mean_base_loading[l]:>8.1f}%")

    print()
    if n_any / N < 0.05:
        print("  *** WARNING: PTDF constraints bind < 5% of hours. ***")
        print("      The 25 MW battery may be too small vs line capacities.")
        print("      NC-SafeSAC vs SafeSAC ablation gap will be negligible.")
        print("      Consider: reduce LINE_LIMIT_SCALE or increase P_MAX.")
    elif n_any / N < 0.20:
        print(f"  NOTE: PTDF constraints bind {100*n_any/N:.1f}% of hours.")
        print("        Moderate constraint activity — ablation gap will be visible.")
    else:
        print(f"  OK: PTDF constraints bind {100*n_any/N:.1f}% of hours.")
        print("      Network constraints are actively limiting dispatch.")
        print("      NC-SafeSAC vs SafeSAC ablation gap should be significant.")

    print("=" * 65)

    # ------------------------------------------------------------------
    # 8. Optionally save
    # ------------------------------------------------------------------
    stats = {
        "n_hours":           N,
        "pct_ptdf_dis":      100.0 * n_dis  / N,
        "pct_ptdf_ch":       100.0 * n_ch   / N,
        "pct_ptdf_either":   100.0 * n_any  / N,
        "pct_base_gt80":     pct_gt80,
        "pct_base_gt90":     pct_gt90,
        "top_lines":         top_lines,
        "dis_bind_freq":     dis_bind_freq,
        "ch_bind_freq":      ch_bind_freq,
        "mean_base_loading": mean_base_loading,
    }

    if save_dir is not None:
        out = os.path.join(save_dir, "ptdf_binding_stats.npy")
        np.save(out, stats, allow_pickle=True)
        print(f"  Stats saved -> {out}")

    return stats

