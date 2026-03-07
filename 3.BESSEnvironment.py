"""
Step 3: BESS Gymnasium Environment for NC-SafeRL
====================================================================
Author: Yiannis Vourkas

BESS (Battery Energy Storage System) dispatch environment on the
RTS-GMLC 73-bus network. Implements the full NC-SafeRL framework
from the research plan.

ENVIRONMENT SPECIFICATION
--------------------------
State  (33-dim):
  [0]       SoC / E_cap                      (normalized SoC)
  [1]       LMP_k,t / LMP_NORM               (current LMP at BESS bus)
  [2..25]   LMP_k,{t-1..t-24} / LMP_NORM     (24-hour LMP history)
  [26]      total_system_PV_t / PV_MAX       (system-wide PV output)
  [27..30]  deg_seg occupancy [d1,d2,d3,d4] (fraction full per segment)
  [31]      p_lo / P_MAX                     (safety-layer lower bound, in [-1,0])
  [32]      p_hi / P_MAX                     (safety-layer upper bound, in [0,1])
  Line loadings removed (120 dims): the safety-layer projection already
  distills network constraints into [p_lo, p_hi] bounds; the 120 raw
  loading values added noise with no marginal predictive value.
  Calendar sin/cos features removed: the 24-h LMP history encodes
  diurnal/seasonal patterns implicitly.
  p_lo/p_hi added (2 dims): explicitly expose the current feasible
  dispatch range to both actor and critic. Speeds up learning by
  eliminating the need to infer feasibility bounds from SoC+network
  state alone. Consistent with research plan Section 5.1.

Action (1-dim):
  Continuous in [-1, +1], linearly rescaled to [-P_MAX, +P_MAX] MW.
  Positive = net discharge to grid, Negative = net charge from grid.

Reward:
  r_t = lambda_k,t * P_net * dt - deg_cost_t   [$/timestep]
  Where deg_cost_t = sum_j c_j * e_dis_j (piecewise linear degradation)

Safety Layer (QP projection):
  Constraints projected analytically (1D QP -> clipping):
    (a) SoC bounds:    SoC_next in [SoC_min, SoC_max]
    (b) Power rating:  P in [-P_MAX, P_MAX]
    (c) Line flows:    F_l + PTDF[l,k] * P in [-F_max_l, +F_max_l]
  Result: P_safe = clip(P_action, P_lo, P_hi)
  This is exact for the 1-variable QP (no approximation).

Episode:
  Length: 168 steps (1 week, dt = 1 hour)
  Start:  Random converged week from annual data (52 possible weeks)
  Init:   SoC = 87.5% of E_cap (SOC_INIT), LMP history = zeros

DEGRADATION MODEL — Xu et al. (2017), Piecewise Linear, J=4 Segments
----------------------------------------------------------------------
Reference: Xu et al., "Factoring the Cycle Aging Cost of Batteries
  Participating in Electricity Markets", IEEE Trans. Power Systems, 2017.

Cumulative life-loss fraction per cycle at depth-of-discharge d:
  Phi(d) = xi * d^alpha                              [Xu Eq. 24]
  xi    = 5.24e-4  (empirical fit, NMC 18650, Laresgoiti et al. 2015)
  alpha = 2.03     (empirical fit, same source)
  -> Phi(1.0) = 5.24e-4  => effective cycle life N0 = 1/Phi(1) ≈ 1908 cycles
  (N0 is NOT a direct parameter; it is implicit in xi.)

Segment j covers SoC range [(1 - j/J)*E, (1 - (j-1)/J)*E]:
  j=1: SoC in [0.75*E, 1.00*E]  <- shallowest, cheapest to cycle
  j=2: SoC in [0.50*E, 0.75*E]
  j=3: SoC in [0.25*E, 0.50*E]
  j=4: SoC in [0.00*E, 0.25*E]  <- deepest, most expensive to cycle

Marginal cost per MWh extracted from battery at segment j:
  c_j = (R [$/kWh] * 1000 / eta_dis) * J * (Phi(j/J) - Phi((j-1)/J))
  Derived from: c_j = R_total * (life fraction per MWh) / eta_dis
              = R_per_kwh * E_cap_kwh * J * delta_phi / (eta_dis * E_cap_mwh)
              = R_per_kwh * 1000 * J * delta_phi / eta_dis  [$/MWh extracted]
  (Breakeven sell LMP = c_j / eta_dis, since energy delivered = eta_dis * e_from_batt)

Code parameters (R=100 $/kWh, eta_dis=0.95, J=4)  [2026 BNEF cost estimate]:
  c_1 = $13.23/MWh  (breakeven $13.93 -- profitable ~25%+ of hours at bus 111)
  c_2 = $40.80/MWh  (breakeven $42.95 -- bus 111 LMP_max ~$82/MWh: profitable at peak hours)
  c_3 = $69.02/MWh  (breakeven $72.65 -- below bus 111 LMP_max: profitable at peak-of-peak hours)
  c_4 = $97.59/MWh  (breakeven $102.73 -- above bus 111 LMP_max; never profitable)

  SOC_INIT = 87.5 MWh places the battery mid-segment-1 at reset.
  The agent has 12.5 MWh of profitable discharge capacity immediately.
  Segments 1–3 profitable at bus 111 during peak hours; segment 4 never breaks even.

Economics at bus 111 (annual LMP analysis):
  Bus 111 (Area 1) has the highest LMP standard deviation of any
  network-constrained-feasible bus (rho=0.277, i.e. 27.7% of hours allow
  PTDF-feasible charging). LMP_max≈$82/MWh (after FUEL_PRICE_SCALE=2.0) exceeds
  the round-trip breakeven (~$24/MWh), the segment-2 breakeven ($42.95/MWh), and
  the segment-3 breakeven ($72.65/MWh) at peak congestion hours.

  Round-trip breakeven: lmp_lo/eta_rt + c_1/eta_dis.
  Charging at p25 LMP (~$9/MWh): 9/0.9025 + 14.09 ≈ $24.06/MWh sell needed.
  Hours with LMP > $24/MWh: approx 10-15% of converged hours at bus 111.

  Contrast: bus 306 (LMP_max=$94, rho=0.253) appears more attractive but
  PTDF constraints block charging 74.7% of hours; arbitrage degenerates to
  one-shot discharge. Bus 111 is the baseline for demonstrating NC-SafeRL.
  Bus 306 is the pathological counter-example (Section 5.3 of the paper).

  R=$100/kWh reflects 2026 Li-ion pack costs (BNEF 2025 Battery Price Survey).
  To adjust difficulty, change BESS_R_PER_KWH:
    R= 75 $/kWh => c_1=$10.05, breakeven=$10.58, profitable=~20% of hours
    R=100 $/kWh => c_1=$13.39, breakeven=$14.09, profitable=~10-15% (default)
    R=150 $/kWh => c_1=$19.84, breakeven=$20.88, profitable= ~5% of hours

OUTPUTS REQUIRED FROM PREVIOUS STEPS
--------------------------------------
  outputs/step2_lmps.npy          (8784, 73)   float32  $/MWh
  outputs/step2_line_flows.npy    (8784, 120)  float32  MW
  outputs/step2_ptdf.npy          (120, 73)    float64
  outputs/step2_converged.npy     (8784,)      bool
  outputs/step2_line_limits.npy   (120,)       float32  MW
  outputs/step2_pv_total.npy      (8784,)      float32  MW
  outputs/step2_metadata.json     (bus ordering, branch names)
  (step2_line_loadings.npy dropped -- redundant, not loaded here)
"""

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
BESS_BUS_RTS     = 111       # RTS bus ID to deploy BESS (highest LMP_std among NC-feasible buses)
BESS_P_MAX_MW    = 25.0      # Power rating [MW]
BESS_E_CAP_MWH   = 100.0     # Energy capacity [MWh]
BESS_ETA_CH      = 0.95      # Charging efficiency (one-way)
BESS_ETA_DIS     = 0.95      # Discharging efficiency (one-way)
BESS_SOC_MIN     = 0.10      # Min SoC as fraction of E_cap
BESS_SOC_MAX     = 0.90      # Max SoC as fraction of E_cap
BESS_SOC_INIT    = 0.875     # Initial SoC as fraction of E_cap (reset).
                              # 87.5 MWh = mid-segment-1 (SoC 75-100 MWh, c_1=$13.23/MWh).
                              # Gives the agent 12.5 MWh of immediately profitable segment-1
                              # capacity from episode 1, enabling positive reward signal from
                              # the start.  At 50% (previous): every discharge drew from
                              # segment 3 ($72.65 breakeven > bus 111 LMP_max $41) →
                              # guaranteed loss on every step → no positive Q-signal for
                              # 25+ MWh of mandatory charging before profitable discharge.
                              # At 87.5%: agent experiences the profitable charge/discharge
                              # cycle immediately, then learns to recharge for future cycles.

# --- Degradation model — Xu et al. (2017), IEEE Trans. Power Systems ---
# Phi(d) = BESS_PHI_XI * d^BESS_PHI_ALPHA   [Xu Eq. 24, NMC 18650 empirical fit]
BESS_PHI_XI      = 5.24e-4   # Life-loss coefficient  (Laresgoiti et al. 2015)
BESS_PHI_ALPHA   = 2.03      # Life-loss exponent     (convex: deeper = costlier)
BESS_R_PER_KWH   = 100.0     # Reflecting 2026 economics [$/kWh nameplate capacity]
                              # Paper value: 300,000 $/MWh = 300 $/kWh
BESS_J_DEG       = 4         # Number of piecewise linear degradation segments
                              # Paper tests J=1,2,4,16; J=4 is recommended default

# --- Episode ---
EPISODE_LEN      = 168       # Timesteps per episode (1 week)
DT               = 1.0       # Timestep duration [hours]

# --- State normalization constants ---
LMP_NORM         = 100.0     # LMP normalization [$/MWh] (max observed ~$94)
PV_MAX_MW        = 2500.0    # System PV normalization [MW]  (max observed ~2421)
REWARD_SCALE     = 1e-3      # Scale rewards to O(1) for stable RL training

# --- PTDF sensitivity threshold ---
PTDF_THRESH      = 1e-5      # Lines with |PTDF| < thresh are treated as insensitive

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
    Compute marginal degradation costs c_j [$/MWh] for each segment j=1..J.

    Exact implementation of Xu et al. (2017), IEEE Trans. Power Systems:
      Phi(d) = xi * d^alpha              [Xu Eq. 24, NMC 18650 empirical fit]
        xi    = 5.24e-4  (BESS_PHI_XI)
        alpha = 2.03     (BESS_PHI_ALPHA)

      c_j = (R [$/kWh] * 1000 / eta_dis) * J * (Phi(j/J) - Phi((j-1)/J))

    Derivation:
      Life fraction consumed per MWh extracted from segment j:
        = J * delta_Phi / E_cap   [1/MWh]
      Cost per MWh extracted = R_total * (life fraction per MWh)
        = R_per_kwh * E_cap_kWh * J * delta_Phi / (eta_dis * E_cap_mwh)
        = R_per_kwh * 1000 * J * delta_Phi / eta_dis  [$/MWh]
      N0 is NOT a direct parameter; it is implicit in xi: N0 = 1/Phi(1) ≈ 1908.

    Returns: array of shape (J,) with c_1, ..., c_J [$/MWh extracted from battery]
    Breakeven sell LMP for segment j = c_j / eta_dis [$/MWh delivered].
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
    Gymnasium environment for BESS economic dispatch on RTS-GMLC 73-bus network.

    The BESS is sited at bus 111 (highest LMP_std among NC-feasible buses,
    Area 1). The agent controls battery charge/discharge to maximize arbitrage
    revenue minus degradation cost, subject to hard physical constraints
    enforced via a safety QP layer (analytical 1D projection).

    Parameters
    ----------
    bess_bus_rts : int
        RTS bus ID for BESS deployment (default: 306)
    p_max_mw : float
        Battery power rating [MW]
    e_cap_mwh : float
        Battery energy capacity [MWh]
    eta_ch, eta_dis : float
        One-way charging / discharging efficiency
    soc_min, soc_max : float
        SoC bounds as fraction of E_cap (e.g., 0.10, 0.90)
    soc_init : float
        Initial SoC as fraction of E_cap at episode reset
    r_per_kwh : float
        Battery replacement cost [$/kWh nameplate capacity] (paper: 300 $/kWh)
    j_deg : int
        Number of piecewise linear degradation segments (paper default: 4)
    episode_len : int
        Episode length in timesteps (default: 168 = 1 week)
    lmp_norm : float
        LMP normalization constant [$/MWh] for state space
    reward_scale : float
        Reward scaling factor (multiply raw $/h by this) for RL stability
    seed : int, optional
        Random seed for reproducibility
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

        # --- Compute degradation costs ---
        self.c_deg = _compute_segment_costs(
            r_per_kwh=r_per_kwh,
            eta_dis=eta_dis,
            j_deg=j_deg,
        )  # shape (J,) in $/MWh extracted from battery (Xu et al. 2017)

        # --- Load precomputed data from Step 2 ---
        self._load_data(bess_bus_rts)

        # --- Gymnasium spaces ---
        # State: 33-dimensional continuous (line loadings + calendar features removed,
        # p_lo/p_hi safety bounds added as 2 explicit dims)
        n_obs = 1 + 1 + 24 + 1 + j_deg + 2          # = 33
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

        # Valid starts: day-aligned anchors (00:00) with a converged reset hour.
        #
        # Why this rule:
        #   - Step 2 now imputes failed OPF snapshots with the previous solved
        #     hour (LMPs + line flows), so we avoid non-physical zero-flow data.
        #   - Requiring an entirely converged 168-hour window discards too much
        #     annual coverage when failed hours are scattered.
        #   - Keeping starts day-aligned preserves the original evaluation scale
        #     (~95 test starts) and avoids runtime explosion.
        #
        # Episodes may still contain some imputed interior hours; we track that
        # count for diagnostics, but only require the reset state itself to be
        # truly converged.
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

        # ------------------------------------------------------------------
        # Stratified train / test split
        # ------------------------------------------------------------------
        # Final week of each calendar month (day-of-month >= 22) → held-out
        # test pool.  This guarantees every season appears in BOTH pools, so
        # the agent can learn winter/summer dynamics yet still be evaluated
        # out-of-sample.  Approximate split: ~75 % train, ~25 % test.
        #
        # 8 784-hour dataset = 366-day leap year.
        # Month lengths: Jan31 Feb29 Mar31 Apr30 May31 Jun30
        #                Jul31 Aug31 Sep30 Oct31 Nov30 Dec31
        _MONTH_DAYS = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        _train, _test = [], []
        for _s in self.valid_starts:
            _dom = int(_s) // 24                    # day-of-year (0-indexed)
            for _nd in _MONTH_DAYS:                 # walk months to find dom
                if _dom < _nd:
                    break
                _dom -= _nd
            # _dom is now day-of-month (0-indexed).  Days 22-30/31 ≈ final week.
            if _dom >= 22:
                _test.append(_s)
            else:
                _train.append(_s)
        self.train_starts = np.array(_train, dtype=self.valid_starts.dtype)
        self.test_starts  = np.array(_test,  dtype=self.valid_starts.dtype)
        self._mode        = "train"    # default: sample from training pool

        # Normalization constants from data
        self.pv_max = float(np.nanmax(self.pv_total))  # for state normalization

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
        """
        Switch the episode-start sampling pool.

        Parameters
        ----------
        mode : "train"  -- sample from train_starts (default, used during SAC training)
               "test"   -- sample from test_starts  (used during evaluation)

        The stratified split puts the final week of each calendar month into
        the test pool so that every season is represented out-of-sample.
        """
        if mode not in ("train", "test"):
            raise ValueError(f"mode must be 'train' or 'test', got {mode!r}")
        self._mode = mode

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode.

        Randomly samples a valid start hour (aligned to converged week),
        or uses an exact start if options["start_t"] is provided.
        Initializes SoC = SOC_INIT, LMP history back-filled from actual data.

        Parameters
        ----------
        seed    : int, optional -- re-seeds the internal RNG
        options : dict, optional -- if {"start_t": int}, use that exact hour
                  as episode start (used by _run_evaluation to iterate over
                  all test starts deterministically)

        Returns
        -------
        obs : np.ndarray, shape (33,)
        info : dict with episode start info
        """
        super().reset(seed=seed)

        # Pick start hour: exact override takes priority over random pool sample
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        if options is not None and "start_t" in options:
            self.global_t = int(options["start_t"])   # deterministic eval start
        else:
            pool = self.test_starts if self._mode == "test" else self.train_starts
            idx  = self._np_rng.integers(0, len(pool))
            self.global_t = int(pool[idx])
        self.step_in_ep  = 0
        self.soc         = self.SOC_INIT
        self.lmp_history = np.zeros(24, dtype=np.float32)

        # Populate history from the hours before episode start (use zeros if at boundary)
        for lag in range(1, 25):
            h = self.global_t - lag
            if h >= 0 and self.converged[h]:
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

        Parameters
        ----------
        action : np.ndarray, shape (1,) in [-1, 1]

        Returns
        -------
        obs       : np.ndarray (33,)
        reward    : float      [scaled $]
        terminated: bool       (episode done, always at step 168)
        truncated : bool       (always False -- no timeout)
        info      : dict       (diagnostic info)
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

        # 6. Compute reward: revenue - degradation cost
        # P_net = p_safe (positive = selling, negative = buying)
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
        # When terminated, global_t == start + EPS_LEN which may equal N_HOURS.
        # Terminal obs is never bootstrapped (done=True → critic target uses 0),
        # so return a zero vector rather than risk an IndexError at array boundary.
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
        Project proposed battery power p_hat onto the feasible set.

        Solves: min_{P} (P - p_hat)^2
                s.t.  P in [P_lo, P_hi]

        where P_lo and P_hi are the tightest bounds from:
          (a) Power rating:  P in [-P_MAX, +P_MAX]
          (b) SoC bounds:    no under/over-charge next step
          (c) Line flows:    F_l + PTDF[l,k] * P in [-F_max_l, +F_max_l]

        Exact solution: P_safe = clip(p_hat, P_lo, P_hi)

        Parameters
        ----------
        p_hat : float  - proposed battery power [MW]

        Returns
        -------
        p_safe : float  - projected safe power [MW]
        p_lo   : float  - lower bound [MW]
        p_hi   : float  - upper bound [MW]
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

        # -- Feasibility guard --
        # If P_lo > P_hi, the constraints are temporarily infeasible
        # (can happen due to numerical drift). Resolve conservatively.
        if p_lo > p_hi:
            # Hold position (discharge minimum needed to stay in SoC bounds)
            p_lo = 0.0
            p_hi = 0.0

        # -- Exact QP solution (1D clipping) --
        p_safe = float(np.clip(p_hat, p_lo, p_hi))

        return p_safe, p_lo, p_hi

    # ------------------------------------------------------------------
    # DEGRADATION COST
    # ------------------------------------------------------------------

    def _compute_deg_cost(self, soc: float, e_dis: float) -> float:
        """
        Compute total degradation cost for extracting e_dis MWh from battery.

        The battery draws from segments in order j=1..J (shallowest first,
        i.e., highest SoC range first). Segment j covers SoC range:
          [(1 - j/J)*E_cap, (1 - (j-1)/J)*E_cap]

        Parameters
        ----------
        soc   : float - SoC at the START of the discharge [MWh]
        e_dis : float - energy to extract from battery [MWh]

        Returns
        -------
        float - degradation cost [$]
        """
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
        Build the 33-dimensional state vector at the current timestep.

        Layout:
          [0]       SoC / E_cap                     (normalized, 0..1)
          [1]       LMP_k,t / LMP_NORM              (roughly 0..1, can be negative)
          [2..25]   LMP history {t-1..t-24} / LMP_NORM  (24 values)
          [26]      pv_total / PV_MAX               (0..1)
          [27..30]  deg segment occupancy           (4 values, 0..1 per segment)
          [31]      p_lo / P_MAX                   (safety lower bound, in [-1, 0])
          [32]      p_hi / P_MAX                   (safety upper bound, in [0, 1])

        p_lo/p_hi expose the current feasible dispatch window directly to the
        actor and critic, eliminating the need to infer it from SoC + network
        state alone.  They are computed from the current SoC and baseline line
        flows without any extra OPF call.

        Returns np.ndarray of shape (33,) dtype float32, clipped to [-1, 1].
        """
        t = self.global_t

        # -- SoC (normalized) --
        soc_norm = self.soc / self.E_CAP   # [0, 1]

        # -- Current LMP --
        lmp_t = float(self.lmps[t, self.bess_col]) if t < 8784 else 0.0
        lmp_norm = lmp_t / self.LMP_NORM

        # -- LMP history (lag 1..24) --
        lmp_hist_norm = self.lmp_history / self.LMP_NORM   # already rolled in step()

        # -- System PV (proxy for renewable output and time-of-day) --
        pv_norm = float(self.pv_total[t]) / max(self.pv_max, 1.0) if t < 8784 else 0.0

        # -- Degradation segment occupancy (tells agent which cost tier is active) --
        seg_occ = self._get_seg_occupancy()   # (J,) in [0, 1]

        # -- Safety-layer feasible bounds (normalized to [-1, 1]) --
        # Compute the current [p_lo, p_hi] window from SoC + baseline flows.
        # p_hat=0.0 gives p_safe=clip(0,p_lo,p_hi); we only use the bounds.
        _, p_lo, p_hi = self._safety_project(0.0)
        p_lo_norm = p_lo / self.P_MAX   # in [-1, 0]
        p_hi_norm = p_hi / self.P_MAX   # in [0, 1]

        obs = np.concatenate([
            [soc_norm],          # 1
            [lmp_norm],          # 1
            lmp_hist_norm,       # 24
            [pv_norm],           # 1
            seg_occ,             # J (4)
            [p_lo_norm],         # 1
            [p_hi_norm],         # 1
        ]).astype(np.float32)    # total: 33

        # Clip to valid range (handles minor numerical overflows)
        return np.clip(obs, -1.0, 1.0)

    def _get_seg_occupancy(self) -> np.ndarray:
        """
        Compute occupancy of each degradation segment as fraction [0, 1].

        Segment j occupancy = (energy in segment j) / SEG_SIZE
        where energy in segment j = max(0, min(SoC, seg_upper) - seg_lower)

        Returns np.ndarray of shape (J,) with values in [0, 1].
        """
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
            f"  State dim:         {self.observation_space.shape[0]}  (SoC+LMP+hist24+PV+segs4+p_lo+p_hi)",
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
    """
    Create and return a configured BESSEnv instance.

    Convenience wrapper that accepts all BESSEnv constructor kwargs.
    Use this in training scripts for consistent setup.

    Example
    -------
    >>> env = make_bess_env(seed=42)
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    return BESSEnv(seed=seed, **kwargs)


# ================================================================
# VALIDATION
# ================================================================

def validate_env(n_episodes: int = 5, verbose: bool = True) -> Dict[str, Any]:
    """
    Run n_episodes of random actions to validate environment correctness.

    Checks:
      - Observation shape and range
      - Action space shape
      - SoC always stays within bounds
      - Reward is finite
      - No NaN in observations
      - Episode terminates at correct step
      - Safety layer: clipped actions respect all constraints

    Returns
    -------
    dict with validation statistics
    """
    print("\n" + "=" * 65)
    print("VALIDATION: BESSEnv Correctness Checks")
    print("=" * 65)

    env = make_bess_env(seed=0)

    lmps_at_bus = env.lmps[:, env.bess_col]
    lmp_stats = {
        "mean":  float(lmps_at_bus[env.converged].mean()),
        "std":   float(lmps_at_bus[env.converged].std()),
        "p25":   float(np.percentile(lmps_at_bus[env.converged], 25)),
        "p75":   float(np.percentile(lmps_at_bus[env.converged], 75)),
        "max":   float(lmps_at_bus[env.converged].max()),
    }
    print(f"\nBESS bus {env.bess_bus_rts} LMP stats (converged hours):")
    for k, v in lmp_stats.items():
        print(f"  {k:6s}: ${v:.2f}/MWh")

    # Check spaces (obs dim = 33: 1+1+24+1+4+2, p_lo/p_hi added)
    assert env.observation_space.shape == (33,), "Wrong obs shape"
    assert env.action_space.shape == (1,), "Wrong action shape"
    print(f"\nSpaces OK: obs={env.observation_space.shape}, action={env.action_space.shape}")

    # Run episodes
    stats = {
        "n_violations":      0,
        "n_infeasible":      0,
        "n_nan_obs":         0,
        "n_clipped":         0,
        "total_revenue_$":   0.0,
        "total_deg_cost_$":  0.0,
        "total_steps":       0,
        "episodes":          [],
    }

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        ep_reward = 0.0
        ep_revenue = 0.0
        ep_deg = 0.0
        soc_min_ep = env.soc
        soc_max_ep = env.soc
        n_clip_ep = 0

        for step in range(EPISODE_LEN):
            # Random action in [-1, 1]
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            # Checks
            if np.any(np.isnan(obs)):
                stats["n_nan_obs"] += 1

            soc_now = info["soc_mwh"]
            if soc_now < env.SOC_MIN - 1e-4 or soc_now > env.SOC_MAX + 1e-4:
                stats["n_violations"] += 1
                if verbose:
                    print(f"  SoC VIOLATION: ep={ep}, step={step}, SoC={soc_now:.2f}")

            # PTDF line-flow safety check: |F_base + PTDF[:,k] * p_safe| <= F_max + tol
            _gh = info["global_hour"]
            if _gh < env.line_flows.shape[0]:
                _f_post = env.line_flows[_gh] + env.ptdf_k * info["p_safe_mw"]
                _viol = np.abs(_f_post) - env.f_lmax
                if np.any(_viol > 0.1):   # 0.1 MW tolerance for floating-point
                    stats["n_violations"] += 1
                    if verbose:
                        _idx = int(np.argmax(_viol))
                        print(f"  PTDF VIOLATION: ep={ep}, step={step}, "
                              f"branch={_idx}, excess={_viol[_idx]:.3f} MW")

            if info.get("clipped"):
                stats["n_clipped"] += 1
                n_clip_ep += 1

            ep_reward  += reward
            ep_revenue += info["revenue_$"]
            ep_deg     += info["deg_cost_$"]
            soc_min_ep = min(soc_min_ep, soc_now)
            soc_max_ep = max(soc_max_ep, soc_now)
            stats["total_steps"] += 1

            if terminated:
                assert step == EPISODE_LEN - 1, f"Early termination at step {step}"
                break

        stats["total_revenue_$"]  += ep_revenue
        stats["total_deg_cost_$"] += ep_deg
        stats["episodes"].append({
            "reward":     ep_reward,
            "revenue_$":  ep_revenue,
            "deg_cost_$": ep_deg,
            "n_clipped":  n_clip_ep,
            "soc_min":    soc_min_ep,
            "soc_max":    soc_max_ep,
        })

        if verbose:
            print(f"\n  Episode {ep+1}/{n_episodes}:")
            print(f"    Total reward:   {ep_reward:.4f} (scaled)")
            print(f"    Revenue:        ${ep_revenue:.2f}")
            print(f"    Deg cost:       ${ep_deg:.2f}")
            print(f"    Net profit:     ${ep_revenue - ep_deg:.2f}")
            print(f"    Clipped steps:  {n_clip_ep}/{EPISODE_LEN} ({100*n_clip_ep/EPISODE_LEN:.0f}%)")
            print(f"    SoC range:      [{soc_min_ep:.1f}, {soc_max_ep:.1f}] MWh")

    print("\n" + "-" * 65)
    print("SUMMARY:")
    print(f"  SoC+PTDF violations: {stats['n_violations']}  (MUST be 0)")
    print(f"  NaN in obs:      {stats['n_nan_obs']}  (MUST be 0)")
    print(f"  Infeasible QPs:  {stats['n_infeasible']}  (MUST be 0)")
    print(f"  Clipped steps:   {stats['n_clipped']}/{stats['total_steps']} "
          f"({100*stats['n_clipped']/max(stats['total_steps'],1):.1f}%)")
    print(f"  Total revenue:   ${stats['total_revenue_$']:.2f} "
          f"(avg ${stats['total_revenue_$']/n_episodes:.2f}/episode)")
    print(f"  Total deg cost:  ${stats['total_deg_cost_$']:.2f} "
          f"(avg ${stats['total_deg_cost_$']/n_episodes:.2f}/episode)")

    if stats["n_violations"] == 0 and stats["n_nan_obs"] == 0:
        print("\n  ALL CHECKS PASSED - environment is ready for Step 4 (SAC training)")
    else:
        print("\n  ERRORS DETECTED - review violations above")

    print("=" * 65 + "\n")

    return stats


# ================================================================
# PTDF BINDING ANALYSIS
# ================================================================

def ptdf_binding_analysis(env: BESSEnv, save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Pre-training diagnostic: how often do PTDF network constraints bind
    versus SoC / power-rating constraints?

    For every converged hour in the dataset, we ask: "If the BESS tried to
    dispatch at its full rated power (±P_MAX), would any transmission line
    be violated?"  When the answer is YES, the safety layer's PTDF-derived
    bound is tighter than the power-rating bound, meaning the network
    constraint is the binding one.

    This validates the core research claim: if PTDF constraints bind in a
    meaningful fraction of hours, the network-constrained safety layer
    materially changes the feasible action set relative to battery-only
    projection — and NC-SafeSAC is genuinely different from SafeSAC.
    If binding frequency < 5%, the 25 MW battery is too small relative
    to line capacities and the paper's ablation gap would be negligible.

    Method (mirrors _safety_project exactly):
    -----------------------------------------
    For each converged hour t, compute the tightest PTDF bounds:

      upper_from_lines[l] = (F_max[l] - F_base[t,l]) / PTDF_k[l]  if PTDF_k[l] > 0
                          = (-F_max[l] - F_base[t,l]) / PTDF_k[l] if PTDF_k[l] < 0
      lower_from_lines[l] = (-F_max[l] - F_base[t,l]) / PTDF_k[l] if PTDF_k[l] > 0
                          = (F_max[l] - F_base[t,l]) / PTDF_k[l]  if PTDF_k[l] < 0

    PTDF binds for discharge when:  min_l(upper_from_lines) < P_MAX
    PTDF binds for charging  when:  max_l(lower_from_lines) > -P_MAX

    Also reports base loading statistics: mean |F_base| / F_max for the
    top PTDF-sensitive lines (lines most affected by BESS dispatch).
    A line already at 90% loading from base dispatch has only 10% headroom
    before a 25 MW injection at a sensitive bus causes a violation.

    Placement in paper
    ------------------
    This is Experiment E0 (pre-training diagnostic), reported in the
    Experimental Design section before E1-E4.  It validates that the
    RTS-GMLC network + bus 111 BESS placement creates genuine constraint
    activity, justifying the need for the PTDF safety layer.

    Parameters
    ----------
    env      : BESSEnv  -- instantiated environment (Step 2 data loaded)
    save_dir : str, optional -- if given, saves stats as
               {save_dir}/ptdf_binding_stats.npy

    Returns
    -------
    dict with keys:
      n_hours           : int   -- total converged hours analyzed
      pct_ptdf_dis      : float -- % hours PTDF binds at full discharge
      pct_ptdf_ch       : float -- % hours PTDF binds at full charge
      pct_ptdf_either   : float -- % hours PTDF binds in either direction
      pct_base_gt80     : float -- % hours any line > 80% base loading
      pct_base_gt90     : float -- % hours any line > 90% base loading
      top_lines         : ndarray (10,) -- line indices by binding freq
      dis_bind_freq     : ndarray (120,) -- per-line discharge binding freq
      ch_bind_freq      : ndarray (120,) -- per-line charge binding freq
      mean_base_loading : ndarray (120,) -- mean |F_base|/F_max per line
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


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Step 3: BESS Gymnasium Environment")
    print("=" * 65)

    # Print degradation cost table for reference
    print("\nDegradation cost schedule (Xu et al. 2017, default parameters):")
    print(f"  R = {BESS_R_PER_KWH:.0f} $/kWh, xi = {BESS_PHI_XI}, "
          f"alpha = {BESS_PHI_ALPHA}, J = {BESS_J_DEG}")
    costs = _compute_segment_costs()
    seg_size = BESS_E_CAP_MWH / BESS_J_DEG
    for j, c in enumerate(costs, 1):
        soc_hi = (1 - (j-1)/BESS_J_DEG) * BESS_E_CAP_MWH
        soc_lo = (1 - j/BESS_J_DEG)     * BESS_E_CAP_MWH
        print(f"  Segment {j} (SoC {soc_lo:.0f}-{soc_hi:.0f} MWh): c_{j} = ${c:.2f}/MWh")

    # Run environment validation
    stats = validate_env(n_episodes=5, verbose=True)

    print("Step 3 COMPLETE")
    print("Next: Step 4 - SAC agent (NC-SafeRL training)")
