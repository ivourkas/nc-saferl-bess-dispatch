"""
Step 5: Deterministic optimization baseline for NC-SafeRL.

This script implements a causal rolling-horizon MPC baseline with:
  - a simple historical-analog forecaster,
  - the same Xu-style segment degradation model as the environment,
  - PTDF-aware network bounds,
  - and a binary operating-mode variable to prevent simultaneous
    charging and discharging.

The controller is deliberately simple and transparent so it can serve as
the "obvious" classical baseline for both class comparison and later paper
discussion.
"""

# ================================================================
# IMPORTS
# ================================================================

import os
import json
import time
import argparse
import importlib.util
from dataclasses import dataclass

import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix, csc_matrix


# ================================================================
# IMPORT 3.BESSEnvironment  (filename starts with a digit)
# ================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(_HERE, "3.BESSEnvironment.py")

_spec = importlib.util.spec_from_file_location("bess_env", _ENV_PATH)
_bess_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bess_mod)
BESSEnv = _bess_mod.BESSEnv

OUTPUTS_DIR = os.path.join(_HERE, "outputs")


# ================================================================
# CONFIG
# ================================================================

CFG = {
    "seed": 42,
    "k_neighbors": 20,
    "forecast_eps": 1e-6,
    "idle_threshold_mw": 0.5,
    "time_limit_sec": 5.0,
    "use_mode_binary": True,
}

FORECAST_HORIZON_HOURS = 24


# ================================================================
# HELPERS
# ================================================================

def _day_of_week_features(hour: int) -> tuple[float, float]:
    """Match the day-of-week encoding used in the environment."""
    dow = float(((hour // 24) + 2) % 7)   # 0=Mon … 6=Sun
    ang = 2.0 * np.pi * dow / 7.0
    return float(np.sin(ang)), float(np.cos(ang))


def _compute_network_only_bounds(env: BESSEnv) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute PTDF/network-only bounds p_lo_net[t], p_hi_net[t] for all hours.

    These bounds ignore SoC and only capture transmission headroom plus the
    BESS power rating.
    """
    active = np.abs(env.ptdf_k) > _bess_mod.PTDF_THRESH
    n_hours = env.line_flows.shape[0]

    p_hi = np.full(n_hours, env.P_MAX, dtype=np.float64)
    p_lo = np.full(n_hours, -env.P_MAX, dtype=np.float64)

    if not np.any(active):
        return p_lo, p_hi

    ptdf_act = env.ptdf_k[active].astype(np.float64)      # (n_act,)
    f_lim = env.f_lmax[active].astype(np.float64)         # (n_act,)
    f_base = env.line_flows[:, active].astype(np.float64) # (T, n_act)

    upper_num = f_lim[np.newaxis, :] - f_base
    lower_num = f_lim[np.newaxis, :] + f_base

    pos = ptdf_act > 0.0
    neg = ~pos

    if np.any(pos):
        p_hi = np.minimum(p_hi, np.min(upper_num[:, pos] / ptdf_act[pos], axis=1))
        p_lo = np.maximum(p_lo, np.max(-lower_num[:, pos] / ptdf_act[pos], axis=1))

    if np.any(neg):
        p_hi = np.minimum(p_hi, np.min(lower_num[:, neg] / (-ptdf_act[neg]), axis=1))
        p_lo = np.maximum(p_lo, np.max(-upper_num[:, neg] / (-ptdf_act[neg]), axis=1))

    p_hi = np.clip(p_hi, -env.P_MAX, env.P_MAX)
    p_lo = np.clip(p_lo, -env.P_MAX, env.P_MAX)

    bad = p_lo > p_hi
    p_lo[bad] = 0.0
    p_hi[bad] = 0.0
    return p_lo, p_hi


# ================================================================
# ANALOG FORECASTER
# ================================================================

class AnalogForecaster:
    """
    Causal nearest-neighbor forecaster over historical training anchors.

    Each anchor stores:
      x_t   = current observable exogenous context
      y_t   = actual future trajectory over the next H_max hours

    At runtime, the future is predicted as an inverse-distance-weighted average
    of the K nearest historical anchors.
    """

    def __init__(
        self,
        env: BESSEnv,
        horizon: int,
        k_neighbors: int = 20,
        eps: float = 1e-6,
    ):
        self.env = env
        self.H_max = int(horizon)
        self.k = int(k_neighbors)
        self.eps = float(eps)

        self.p_lo_net, self.p_hi_net = _compute_network_only_bounds(env)
        self.train_end = int(getattr(env, "forecast_train_end", np.min(env.test_starts) if len(env.test_starts) else (
            (int(env.lmps.shape[0] * 0.75) // 24) * 24
        )))

        self.anchor_hours = self._build_anchor_hours()
        self.X = np.stack([self._feature_from_hour(t) for t in self.anchor_hours], axis=0)
        self.Y_lmp = np.stack(
            [env.lmps[t:t + self.H_max, env.bess_col] for t in self.anchor_hours],
            axis=0,
        ).astype(np.float64)
        self.Y_lo = np.stack(
            [self.p_lo_net[t:t + self.H_max] for t in self.anchor_hours],
            axis=0,
        ).astype(np.float64)
        self.Y_hi = np.stack(
            [self.p_hi_net[t:t + self.H_max] for t in self.anchor_hours],
            axis=0,
        ).astype(np.float64)

        self.mean = self.X.mean(axis=0)
        self.std = self.X.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        self.X_norm = ((self.X - self.mean) / self.std).astype(np.float64)
        self.X_norm_sq = np.einsum("ij,ij->i", self.X_norm, self.X_norm)

        print(f"\n{'=' * 65}")
        print("STEP 5  ANALOG FORECASTER")
        print("=" * 65)
        print(f"  Horizon:           {self.H_max} hours")
        print(f"  K neighbors:       {self.k}")
        print(f"  Train cut-off:     hour {self.train_end}")
        print(f"  Anchor count:      {len(self.anchor_hours)}")
        print(f"  Feature dim:       {self.X.shape[1]}")
        print(f"  Network-only PTDF bounds available for all {len(self.p_lo_net)} hours")

    def _build_anchor_hours(self) -> np.ndarray:
        """
        Build training anchors from all train-region hours that have:
          - 24h lookback,
          - H_max lookahead fully contained in the training block,
          - and a converged current hour.
        """
        start = 24
        stop = self.train_end - self.H_max
        if stop < start:
            raise RuntimeError("Not enough train-region hours to build the forecast library.")

        hours = np.arange(start, stop + 1, dtype=np.int32)
        hours = hours[self.env.converged[hours]]
        if len(hours) == 0:
            raise RuntimeError("Forecast library is empty after filtering converged anchors.")
        return hours

    def _feature_from_hour(self, t: int) -> np.ndarray:
        """Build the exogenous feature vector for historical hour t."""
        lmp_hist = self.env.lmps[t - 24:t, self.env.bess_col].astype(np.float64)  # oldest -> newest
        lmp_now = float(self.env.lmps[t, self.env.bess_col])
        pv_now = float(self.env.pv_total[t])
        load_now = float(self.env.system_load[t])
        wind_now = float(self.env.wind_total[t])
        sin_dow, cos_dow = _day_of_week_features(t)
        return np.concatenate([
            lmp_hist,
            np.array([lmp_now, pv_now, load_now, wind_now, sin_dow, cos_dow], dtype=np.float64),
        ])

    def feature_from_env(self, env: BESSEnv) -> np.ndarray:
        """Build the current exogenous feature vector from the live environment state."""
        t = int(env.global_t)
        lmp_hist = env.lmp_history[::-1].astype(np.float64)  # oldest -> newest
        lmp_now = float(env.lmps[t, env.bess_col])
        pv_now = float(env.pv_total[t])
        load_now = float(env.system_load[t])
        wind_now = float(env.wind_total[t])
        sin_dow, cos_dow = _day_of_week_features(t)
        return np.concatenate([
            lmp_hist,
            np.array([lmp_now, pv_now, load_now, wind_now, sin_dow, cos_dow], dtype=np.float64),
        ])

    def forecast(self, x_now: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Return deterministic forecasts for LMP and future network-only bounds."""
        horizon = int(horizon)
        x_norm = (x_now - self.mean) / self.std
        x_norm_sq = float(np.dot(x_norm, x_norm))

        d2 = self.X_norm_sq + x_norm_sq - 2.0 * (self.X_norm @ x_norm)
        d2 = np.maximum(d2, 0.0)

        k = min(self.k, len(self.anchor_hours))
        idx = np.argpartition(d2, k - 1)[:k]
        idx = idx[np.argsort(d2[idx])]

        d = np.sqrt(d2[idx])
        w = 1.0 / (d + self.eps)
        w /= np.sum(w)

        lmp_hat = w @ self.Y_lmp[idx, :horizon]
        lo_hat = w @ self.Y_lo[idx, :horizon]
        hi_hat = w @ self.Y_hi[idx, :horizon]

        lo_hat = np.clip(lo_hat, -self.env.P_MAX, self.env.P_MAX)
        hi_hat = np.clip(hi_hat, -self.env.P_MAX, self.env.P_MAX)
        bad = lo_hat > hi_hat
        lo_hat[bad] = 0.0
        hi_hat[bad] = 0.0

        meta = {
            "neighbor_hours": self.anchor_hours[idx].tolist(),
            "neighbor_distances": d.tolist(),
            "neighbor_weights": w.tolist(),
        }
        return lmp_hat, lo_hat, hi_hat, meta


# ================================================================
# MPC / MILP CONTROLLER
# ================================================================

@dataclass
class _MilpTemplate:
    horizon: int
    n_vars: int
    n_dis: int
    n_ch: int
    n_e: int
    n_z: int
    A_eq: csc_matrix
    A_mode: csc_matrix
    A_soc: csc_matrix
    A_pnet: csc_matrix
    A_terminal: csc_matrix
    bounds: Bounds
    integrality: np.ndarray


class DeterministicMPCBaseline:
    """Rolling-horizon deterministic MPC with a mode binary and Xu degradation."""

    def __init__(
        self,
        env: BESSEnv,
        forecaster: AnalogForecaster,
        time_limit_sec: float = 5.0,
        use_mode_binary: bool = True,
        terminal_soc_target: float | None = None,
        terminal_soc_penalty: float | None = None,
    ):
        self.env = env
        self.forecaster = forecaster
        self.time_limit_sec = float(time_limit_sec)
        self.use_mode_binary = bool(use_mode_binary)

        self.T_max = env.EPS_LEN
        self.J = env.J_DEG
        self.P_max = env.P_MAX
        self.SoC_min = env.SOC_MIN
        self.SoC_max = env.SOC_MAX
        self.seg_size = env.SEG_SIZE
        self.eta_ch = env.ETA_CH
        self.eta_dis = env.ETA_DIS
        self.c_j = env.c_deg.astype(np.float64)
        self.dt = 1.0
        neutral_soc = 0.5 * (env.SOC_MIN + env.SOC_MAX)
        self.terminal_soc_target = float(neutral_soc if terminal_soc_target is None else terminal_soc_target)
        self.terminal_soc_penalty = float(
            self.c_j[0] if terminal_soc_penalty is None else terminal_soc_penalty
        )

        self._template_cache: dict[int, _MilpTemplate] = {}

    def _idx_dis(self, h: int, j: int, H: int) -> int:
        return h * self.J + j

    def _idx_ch(self, h: int, j: int, H: int) -> int:
        return H * self.J + h * self.J + j

    def _idx_e(self, h: int, j: int, H: int) -> int:
        return 2 * H * self.J + h * self.J + j

    def _idx_z(self, h: int, H: int) -> int:
        return 3 * H * self.J + h

    def _idx_term_pos(self, H: int) -> int:
        return 3 * H * self.J + H

    def _idx_term_neg(self, H: int) -> int:
        return 3 * H * self.J + H + 1

    def _initial_seg_energies(self, soc: float) -> np.ndarray:
        """Decompose scalar SoC into shallow-to-deep segment energies."""
        e0 = np.zeros(self.J, dtype=np.float64)
        e_cap = self.J * self.seg_size
        for j in range(self.J):
            seg_upper = e_cap - j * self.seg_size
            seg_lower = e_cap - (j + 1) * self.seg_size
            e0[j] = max(0.0, min(soc, seg_upper) - max(seg_lower, 0.0))
        return e0

    def _build_template(self, H: int) -> _MilpTemplate:
        """Create and cache the MILP matrix structure for a given horizon."""
        if H in self._template_cache:
            return self._template_cache[H]

        n_dis = H * self.J
        n_ch = H * self.J
        n_e = H * self.J
        n_z = H
        n_term = 2
        n_vars = n_dis + n_ch + n_e + n_z + n_term

        # Equality constraints: segment dynamics
        eq_rows, eq_cols, eq_data = [], [], []
        for h in range(H):
            for j in range(self.J):
                row = h * self.J + j
                eq_rows.append(row)
                eq_cols.append(self._idx_e(h, j, H))
                eq_data.append(1.0)

                eq_rows.append(row)
                eq_cols.append(self._idx_dis(h, j, H))
                eq_data.append(self.dt / self.eta_dis)

                eq_rows.append(row)
                eq_cols.append(self._idx_ch(h, j, H))
                eq_data.append(-self.dt * self.eta_ch)

                if h > 0:
                    eq_rows.append(row)
                    eq_cols.append(self._idx_e(h - 1, j, H))
                    eq_data.append(-1.0)

        A_eq = coo_matrix(
            (eq_data, (eq_rows, eq_cols)),
            shape=(H * self.J, n_vars),
            dtype=np.float64,
        ).tocsc()

        # Inequality constraints: mode coupling
        mode_rows, mode_cols, mode_data = [], [], []
        for h in range(H):
            row_dis = h
            row_ch = H + h
            for j in range(self.J):
                mode_rows.append(row_dis)
                mode_cols.append(self._idx_dis(h, j, H))
                mode_data.append(1.0)

                mode_rows.append(row_ch)
                mode_cols.append(self._idx_ch(h, j, H))
                mode_data.append(1.0)

            mode_rows.append(row_dis)
            mode_cols.append(self._idx_z(h, H))
            mode_data.append(-self.P_max)

            mode_rows.append(row_ch)
            mode_cols.append(self._idx_z(h, H))
            mode_data.append(self.P_max)

        A_mode = coo_matrix(
            (mode_data, (mode_rows, mode_cols)),
            shape=(2 * H, n_vars),
            dtype=np.float64,
        ).tocsc()

        # Aggregate SoC constraints
        soc_rows, soc_cols, soc_data = [], [], []
        for h in range(H):
            for j in range(self.J):
                soc_rows.append(h)
                soc_cols.append(self._idx_e(h, j, H))
                soc_data.append(1.0)

        A_soc = coo_matrix(
            (soc_data, (soc_rows, soc_cols)),
            shape=(H, n_vars),
            dtype=np.float64,
        ).tocsc()

        # Net power constraints
        pnet_rows, pnet_cols, pnet_data = [], [], []
        for h in range(H):
            for j in range(self.J):
                pnet_rows.append(h)
                pnet_cols.append(self._idx_dis(h, j, H))
                pnet_data.append(1.0)

                pnet_rows.append(h)
                pnet_cols.append(self._idx_ch(h, j, H))
                pnet_data.append(-1.0)

        A_pnet = coo_matrix(
            (pnet_data, (pnet_rows, pnet_cols)),
            shape=(H, n_vars),
            dtype=np.float64,
        ).tocsc()

        # Terminal inventory soft-landing:
        #   sum_j e[H-1, j] - u_pos + u_neg = target_soc
        # with u_pos, u_neg >= 0 and linear penalty on u_pos + u_neg.
        term_rows, term_cols, term_data = [], [], []
        for j in range(self.J):
            term_rows.append(0)
            term_cols.append(self._idx_e(H - 1, j, H))
            term_data.append(1.0)
        term_rows.extend([0, 0])
        term_cols.extend([self._idx_term_pos(H), self._idx_term_neg(H)])
        term_data.extend([-1.0, 1.0])
        A_terminal = coo_matrix(
            (term_data, (term_rows, term_cols)),
            shape=(1, n_vars),
            dtype=np.float64,
        ).tocsc()

        lb = np.zeros(n_vars, dtype=np.float64)
        ub = np.full(n_vars, np.inf, dtype=np.float64)
        ub[:n_dis] = self.P_max
        ub[n_dis:n_dis + n_ch] = self.P_max
        ub[n_dis + n_ch:n_dis + n_ch + n_e] = self.seg_size
        z_start = self._idx_z(0, H)
        ub[z_start:z_start + n_z] = 1.0

        bounds = Bounds(lb=lb, ub=ub)
        integrality = np.zeros(n_vars, dtype=np.int8)
        if self.use_mode_binary:
            integrality[z_start:z_start + n_z] = 1

        tmpl = _MilpTemplate(
            horizon=H,
            n_vars=n_vars,
            n_dis=n_dis,
            n_ch=n_ch,
            n_e=n_e,
            n_z=n_z,
            A_eq=A_eq,
            A_mode=A_mode,
            A_soc=A_soc,
            A_pnet=A_pnet,
            A_terminal=A_terminal,
            bounds=bounds,
            integrality=integrality,
        )
        self._template_cache[H] = tmpl
        return tmpl

    def solve(
        self,
        lmp_forecast: np.ndarray,
        p_lo: np.ndarray,
        p_hi: np.ndarray,
        soc_init: float,
    ) -> tuple[float, dict]:
        """Solve the rolling-horizon MILP and return the first net-power action."""
        H = int(len(lmp_forecast))
        tmpl = self._build_template(H)

        p_lo = np.asarray(p_lo, dtype=np.float64).copy()
        p_hi = np.asarray(p_hi, dtype=np.float64).copy()
        p_lo = np.clip(p_lo, -self.P_max, self.P_max)
        p_hi = np.clip(p_hi, -self.P_max, self.P_max)
        bad = p_lo > p_hi
        p_lo[bad] = 0.0
        p_hi[bad] = 0.0

        c = np.zeros(tmpl.n_vars, dtype=np.float64)
        for h in range(H):
            for j in range(self.J):
                c[self._idx_dis(h, j, H)] = (self.c_j[j] / self.eta_dis) - lmp_forecast[h]
                c[self._idx_ch(h, j, H)] = lmp_forecast[h]

        # Terminal inventory regularization:
        # keep the end-of-horizon SoC near a neutral reference level so the 24h MPC
        # does not myopically empty or overfill the battery in the last few hours.
        # This is an internal continuation-value approximation only; reported reward
        # still comes exclusively from the environment and remains directly
        # comparable to Steps 4 and 6.
        c[self._idx_term_pos(H)] = self.terminal_soc_penalty
        c[self._idx_term_neg(H)] = self.terminal_soc_penalty

        b_eq = np.zeros(H * self.J, dtype=np.float64)
        b_eq[:self.J] = self._initial_seg_energies(float(soc_init))

        lb_mode = np.full(2 * H, -np.inf, dtype=np.float64)
        ub_mode = np.concatenate([
            np.zeros(H, dtype=np.float64),
            np.full(H, self.P_max, dtype=np.float64),
        ])

        lb_soc = np.full(H, self.SoC_min, dtype=np.float64)
        ub_soc = np.full(H, self.SoC_max, dtype=np.float64)

        constraints = (
            LinearConstraint(tmpl.A_eq, b_eq, b_eq),
            LinearConstraint(tmpl.A_mode, lb_mode, ub_mode),
            LinearConstraint(tmpl.A_soc, lb_soc, ub_soc),
            LinearConstraint(tmpl.A_pnet, p_lo, p_hi),
            LinearConstraint(
                tmpl.A_terminal,
                np.array([self.terminal_soc_target], dtype=np.float64),
                np.array([self.terminal_soc_target], dtype=np.float64),
            ),
        )

        opts = {"disp": False}
        if self.time_limit_sec > 0:
            opts["time_limit"] = self.time_limit_sec

        res = milp(
            c=c,
            integrality=tmpl.integrality,
            bounds=tmpl.bounds,
            constraints=constraints,
            options=opts,
        )

        x = getattr(res, "x", None)
        if x is None or not np.all(np.isfinite(x)):
            return 0.0, {
                "success": False,
                "status": int(getattr(res, "status", -1)),
                "message": str(getattr(res, "message", "No solution returned")),
                "objective": np.nan,
            }

        p_dis = x[:tmpl.n_dis].reshape(H, self.J).sum(axis=1)
        p_ch = x[tmpl.n_dis:tmpl.n_dis + tmpl.n_ch].reshape(H, self.J).sum(axis=1)
        p_net = p_dis - p_ch

        return float(p_net[0]), {
            "success": bool(getattr(res, "success", False)),
            "status": int(getattr(res, "status", -1)),
            "message": str(getattr(res, "message", "")),
            "objective": float(getattr(res, "fun", np.nan)),
            "p_net_first": float(p_net[0]),
        }


# ================================================================
# EVALUATION
# ================================================================

def run_mpc_evaluation_on_starts(
    env: BESSEnv,
    forecaster: AnalogForecaster,
    controller: DeterministicMPCBaseline,
    starts: np.ndarray,
    progress_label: str = "Eval",
) -> tuple[dict, list[dict]]:
    """Evaluate the optimization baseline on a fixed set of episode starts."""
    returns = []
    soc_ends = []
    soc_mins = []
    revenues = []
    deg_costs = []
    n_discharge = []
    n_charge = []
    n_idle = []
    clip_rates = []
    solve_failures = []

    episode_records = []
    starts = np.asarray(starts, dtype=np.int32)
    if len(starts) == 0:
        raise RuntimeError("No episode starts were provided for MPC evaluation.")

    t_eval = time.time()
    for ep_idx, start_t in enumerate(starts, start=1):
        _, _ = env.reset(options={"start_t": int(start_t)})
        ep_return = 0.0
        ep_rev = 0.0
        ep_deg = 0.0
        ep_n_dis = 0
        ep_n_ch = 0
        ep_n_idle = 0
        ep_soc_min = env.soc
        ep_clipped = 0
        ep_fail = 0

        for _ in range(env.EPS_LEN):
            H = min(FORECAST_HORIZON_HOURS, env.EPS_LEN - env.step_in_ep)

            x_now = forecaster.feature_from_env(env)
            lmp_hat, lo_hat, hi_hat, _ = forecaster.forecast(x_now, H)

            lmp_hat[0] = float(env.lmps[env.global_t, env.bess_col])
            _, p_lo_now, p_hi_now = env._safety_project(0.0)
            lo_hat[0] = p_lo_now
            hi_hat[0] = p_hi_now

            p_cmd, solve_meta = controller.solve(
                lmp_forecast=lmp_hat,
                p_lo=lo_hat,
                p_hi=hi_hat,
                soc_init=env.soc,
            )
            if not solve_meta["success"]:
                ep_fail += 1

            act = np.array([np.clip(p_cmd / env.P_MAX, -1.0, 1.0)], dtype=np.float32)
            _, rew, terminated, truncated, info = env.step(act)

            ep_return += rew
            ep_rev += info["revenue_$"]
            ep_deg += info["deg_cost_$"]
            ep_soc_min = min(ep_soc_min, info["soc_mwh"])
            ep_clipped += int(info["clipped"])

            p_safe = info["p_safe_mw"]
            if p_safe > CFG["idle_threshold_mw"]:
                ep_n_dis += 1
            elif p_safe < -CFG["idle_threshold_mw"]:
                ep_n_ch += 1
            else:
                ep_n_idle += 1

            if terminated or truncated:
                break

        returns.append(ep_return)
        soc_ends.append(info["soc_mwh"])
        soc_mins.append(ep_soc_min)
        revenues.append(ep_rev)
        deg_costs.append(ep_deg)
        n_discharge.append(ep_n_dis)
        n_charge.append(ep_n_ch)
        n_idle.append(ep_n_idle)
        clip_rates.append(ep_clipped / max(env.EPS_LEN, 1))
        solve_failures.append(ep_fail / max(env.EPS_LEN, 1))

        episode_records.append({
            "start_t": int(start_t),
            "return": float(ep_return),
            "profit_$": float(ep_rev - ep_deg),
            "revenue_$": float(ep_rev),
            "deg_cost_$": float(ep_deg),
            "soc_end_mwh": float(info["soc_mwh"]),
            "soc_min_mwh": float(ep_soc_min),
            "clip_rate": float(clip_rates[-1]),
            "solve_failure_rate": float(solve_failures[-1]),
            "n_discharge": int(ep_n_dis),
            "n_charge": int(ep_n_ch),
            "n_idle": int(ep_n_idle),
        })

        if ep_idx % 10 == 0 or ep_idx == len(starts):
            elapsed = time.time() - t_eval
            print(
                f"  {progress_label} {ep_idx:3d}/{len(starts)} | "
                f"mean_return={float(np.mean(returns)):+.4f} | "
                f"mean_profit=${float(np.mean(np.array(revenues) - np.array(deg_costs))):.0f} | "
                f"{elapsed:.0f}s"
            )

    returns = np.array(returns, dtype=np.float64)
    total_steps = np.array(n_discharge) + np.array(n_charge) + np.array(n_idle)
    metrics = {
        "eval_mean": float(np.mean(returns)),
        "eval_std": float(np.std(returns)),
        "eval_min": float(np.min(returns)),
        "eval_max": float(np.max(returns)),
        "n_eval": int(len(returns)),
        "soc_end_mean": float(np.mean(soc_ends)),
        "soc_min_mean": float(np.mean(soc_mins)),
        "revenue_mean": float(np.mean(revenues)),
        "deg_cost_mean": float(np.mean(deg_costs)),
        "profit_mean": float(np.mean(np.array(revenues) - np.array(deg_costs))),
        "frac_discharge": float(np.mean(np.array(n_discharge) / np.maximum(total_steps, 1))),
        "frac_charge": float(np.mean(np.array(n_charge) / np.maximum(total_steps, 1))),
        "frac_idle": float(np.mean(np.array(n_idle) / np.maximum(total_steps, 1))),
        "clip_rate_mean": float(np.mean(clip_rates)),
        "solve_failure_rate_mean": float(np.mean(solve_failures)),
    }
    return metrics, episode_records


def run_mpc_evaluation(
    env: BESSEnv,
    forecaster: AnalogForecaster,
    controller: DeterministicMPCBaseline,
    max_test_episodes: int | None = None,
) -> tuple[dict, list[dict]]:
    """Evaluate the optimization baseline on held-out test starts."""
    test_starts = env.test_starts[:max_test_episodes] if max_test_episodes else env.test_starts
    return run_mpc_evaluation_on_starts(
        env=env,
        forecaster=forecaster,
        controller=controller,
        starts=test_starts,
        progress_label="Test",
    )


def maybe_save_results(
    config: dict,
    metrics: dict,
    episode_records: list[dict],
    validation_eval: dict | None = None,
) -> None:
    """Persist summary artifacts under outputs/step5_final."""
    out_dir = os.path.join(OUTPUTS_DIR, "step5_final")
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "config": config,
        "metrics": metrics,
        "n_episodes": len(episode_records),
    }
    if validation_eval is not None:
        payload["validation_eval"] = validation_eval

    json_path = os.path.join(out_dir, "mpc_baseline_metrics.json")
    npy_path = os.path.join(out_dir, "mpc_baseline_episodes.npy")

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    np.save(npy_path, episode_records, allow_pickle=True)

    print(f"\nSaved Step 5 metrics -> {json_path}")
    print(f"Saved Step 5 episode records -> {npy_path}")


def print_results_summary(metrics: dict, config: dict) -> None:
    """Print a compact summary mirroring the style of Step 4."""
    print("\n" + "=" * 65)
    print("NC-SafeRL  STEP 5  OPTIMIZATION BASELINE SUMMARY")
    print("=" * 65)
    print(f"  Controller          : Deterministic analog-forecast MPC")
    print(f"  Horizon             : {config['horizon']} hours")
    print(f"  K neighbors         : {config['k_neighbors']}")
    print(f"  Mode binary         : {config['use_mode_binary']}")
    print(f"  MILP time limit [s] : {config['time_limit_sec']}")
    print(f"  Terminal SoC target : {config['terminal_soc_target_mwh']:.1f} MWh")
    print(f"  Terminal penalty    : ${config['terminal_soc_penalty']:.2f}/MWh")
    print(f"  Test episodes       : {metrics['n_eval']}")
    print("")
    print(f"  Eval mean return    : {metrics['eval_mean']:+.4f}")
    print(f"  Eval std            : {metrics['eval_std']:.4f}")
    print(f"  Eval min / max      : {metrics['eval_min']:+.4f} / {metrics['eval_max']:+.4f}")
    print(f"  Profit mean         : ${metrics['profit_mean']:.0f}")
    print(f"  Revenue mean        : ${metrics['revenue_mean']:.0f}")
    print(f"  Deg cost mean       : ${metrics['deg_cost_mean']:.0f}")
    print(f"  SoC end / min mean  : {metrics['soc_end_mean']:.1f} / {metrics['soc_min_mean']:.1f} MWh")
    print(f"  Action mix          : dis={metrics['frac_discharge']:.0%} "
          f"ch={metrics['frac_charge']:.0%} idle={metrics['frac_idle']:.0%}")
    print(f"  Clip rate mean      : {metrics['clip_rate_mean']:.4%}")
    print(f"  Solve fail rate     : {metrics['solve_failure_rate_mean']:.4%}")
    print("=" * 65)


# ================================================================
# MAIN
# ================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 5 deterministic optimization baseline.")
    parser.add_argument("--k", type=int, default=CFG["k_neighbors"],
                        help="Number of nearest historical anchors.")
    parser.add_argument("--time_limit_sec", type=float, default=CFG["time_limit_sec"],
                        help="Per-step MILP time limit in seconds.")
    parser.add_argument("--max_test_episodes", type=int, default=None,
                        help="Evaluate only the first N test episodes (for smoke tests).")
    parser.add_argument("--relax_mode_binary", action="store_true",
                        help="Relax the operating-mode binary to [0,1] continuous.")
    parser.add_argument("--terminal_soc_target_mwh", type=float, default=None,
                        help="Soft terminal SoC target in MWh. Default = midpoint of SoC bounds.")
    parser.add_argument("--terminal_soc_penalty", type=float, default=None,
                        help="Penalty on |terminal SoC - target| in $/MWh. Default = shallow-segment degradation cost.")
    parser.add_argument("--validation_days", type=int, default=30,
                        help="Days reserved for RL validation. "
                             "Validation-style checks use a train-only forecast library; "
                             "the final held-out test uses the full causal pre-test library.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("\n" + "=" * 65)
    print("NC-SafeRL  STEP 5: Optimization Baseline")
    print("=" * 65)

    env = BESSEnv(seed=CFG["seed"], validation_days=args.validation_days)
    print(f"RTS-GMLC 73-bus | BESS bus {env.bess_bus_rts} | "
          f"{len(env.test_starts)} held-out test starts")

    selection_forecaster = AnalogForecaster(
        env=env,
        horizon=FORECAST_HORIZON_HOURS,
        k_neighbors=args.k,
        eps=CFG["forecast_eps"],
    )

    controller = DeterministicMPCBaseline(
        env=env,
        forecaster=selection_forecaster,
        time_limit_sec=args.time_limit_sec,
        use_mode_binary=not args.relax_mode_binary,
        terminal_soc_target=args.terminal_soc_target_mwh,
        terminal_soc_penalty=args.terminal_soc_penalty,
    )

    validation_eval = None
    if len(env.val_starts) > 0:
        print("\nValidation-style evaluation (train-only forecast library):")
        validation_eval, _ = run_mpc_evaluation_on_starts(
            env=env,
            forecaster=selection_forecaster,
            controller=controller,
            starts=env.val_starts,
            progress_label="Val",
        )
        print(f"  Val mean return     : {validation_eval['eval_mean']:+.4f}")
        print(f"  Val profit mean     : ${validation_eval['profit_mean']:.0f}")

    # Rebuild the forecaster with all causal history available before the test block.
    # Train-only history governs any pre-test checks, while the frozen final
    # controller may use train+val history once the held-out test begins.
    saved_fte = env.forecast_train_end
    env.forecast_train_end = env.test_start_hour
    test_forecaster = AnalogForecaster(
        env=env,
        horizon=FORECAST_HORIZON_HOURS,
        k_neighbors=args.k,
        eps=CFG["forecast_eps"],
    )
    env.forecast_train_end = saved_fte

    test_controller = DeterministicMPCBaseline(
        env=env,
        forecaster=test_forecaster,
        time_limit_sec=args.time_limit_sec,
        use_mode_binary=not args.relax_mode_binary,
        terminal_soc_target=args.terminal_soc_target_mwh,
        terminal_soc_penalty=args.terminal_soc_penalty,
    )

    t0 = time.time()
    metrics, episode_records = run_mpc_evaluation(
        env=env,
        forecaster=test_forecaster,
        controller=test_controller,
        max_test_episodes=args.max_test_episodes,
    )
    total_time = time.time() - t0

    config = {
        "seed": CFG["seed"],
        "horizon": FORECAST_HORIZON_HOURS,
        "k_neighbors": args.k,
        "time_limit_sec": args.time_limit_sec,
        "use_mode_binary": not args.relax_mode_binary,
        "terminal_soc_target_mwh": controller.terminal_soc_target,
        "terminal_soc_penalty": controller.terminal_soc_penalty,
        "max_test_episodes": args.max_test_episodes,
        "validation_days": int(args.validation_days),
        "selection_forecast_train_end_hour": int(selection_forecaster.train_end),
        "test_forecast_train_end_hour": int(test_forecaster.train_end),
        "forecast_policy": {
            "pre_test_checks": "train_only_history",
            "final_test": "full_causal_pre_test_history",
        },
    }

    print(f"\nTotal wall-clock time: {total_time/60:.1f} min")
    print_results_summary(metrics, config)
    maybe_save_results(config, metrics, episode_records, validation_eval=validation_eval)
    print("\nStep 5 complete.")


if __name__ == "__main__":
    main()
