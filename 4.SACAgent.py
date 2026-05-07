"""
Step 4: Soft Actor-Critic agent for the NC-SafeRL BESS environment.

This file contains the replay buffer, actor/critic models, LP warm-start,
training loop, evaluation utilities, and checkpoint/log handling.
"""

# ================================================================
# IMPORTS
# ================================================================

import os
import sys
import math
import time
import datetime
import importlib.util
from collections import deque   # used by run_episode_collect for N-step window
import numpy as np

# -- TensorFlow device selection --
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"          # suppress TF INFO logs
import tensorflow as tf

# Force CPU execution for reproducibility while debugging the Metal path.
FORCE_CPU = True
if FORCE_CPU:
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

_visible_gpus = tf.config.get_visible_devices("GPU")
if _visible_gpus:
    print(f"[TF] Metal GPU detected: {[g.name for g in _visible_gpus]}")
else:
    print("[TF] CPU-only mode enabled")

from scipy.optimize import linprog

# ================================================================
# IMPORT 3.BESSEnvironment  (filename starts with a digit)
# ================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(_HERE, "3.BESSEnvironment.py")

_spec = importlib.util.spec_from_file_location("bess_env", _ENV_PATH)
_bess_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bess_mod)
BESSEnv               = _bess_mod.BESSEnv
ptdf_binding_analysis = _bess_mod.ptdf_binding_analysis

OUTPUTS_DIR = os.path.join(_HERE, "outputs")

# ================================================================
# HYPERPARAMETERS
# ================================================================

HP = {
    # SAC
    "gamma":          0.99,       # discount factor
    "tau":            0.005,      # target-network soft-update rate
    "lr":             1e-4,       # actor/critic learning rate
    "alpha_lr":       3e-5,       # entropy learning rate; faster recovery from alpha floor
    "hidden_dim":     256,        # hidden layer width
    "hidden_layers":  2,
    "target_entropy_scale": -1,  # target_entropy = -1 × action_dim (automatic entropy tuning disabled by default for stability; set to -action_dim to enable)
    # Buffer & batch
    "buffer_size":    300_000,    # replay capacity
    "batch_size":     256,
    "min_buffer_fill": 12_000,    # dilute LP warm-start before online SAC updates
    "lp_warmstart_steps": 15_000,  # PTDF-feasible LP transitions
    "lp_warmstart_priority_scale": 0.3,  # retroactively lower LP priorities so online transitions dominate PER sampling
    "lp_bc_steps":        3000,
    "lp_bc_active_threshold": 0.02,
    "lp_bc_active_frac":      0.05,  
    "action_scale_floor": 0.0,      # Jacobian floor for bounded actor; 0.04 = 1 MW on a 25 MW battery
    "randomize_train_soc": True,     # Train-only SoC randomization broadens charge/discharge exposure; eval stays at SOC_INIT

    "proj_lambda":    0.0,        # projection penalty disabled
    "n_step":         4,          # 4-step TD: propagates SoC opportunity-cost 4 steps back
    "init_alpha":     0.005,      # initial entropy coefficient
    "alpha_min":      0.005,      # higher floor prevents absorbing-state collapse; alpha_lr=3e-5 now fast enough to recover
    "alpha_max":      0.10,       # upper alpha bound
    "utd_ratio_denominator": 4,   # UTD=0.25: one gradient step per 4 env steps (RLBattEM4 literature)
    "train_episodes": 2_000,      # dev default; paper seed sweeps use 3000
    "eval_freq":      100,        # dev default; sweep uses 200 (full train-set eval is ~11× more expensive)
    "early_stop_patience":  10,   # plateau patience (consecutive evals without min_delta improvement)
    "early_stop_min_delta": 0.001,
    "early_stop_max_degradation": 0.5,   # stop if eval drops >0.5 below best (Q-overfit guard)
    "early_stop_degrade_patience": 4,    # consecutive degraded evals before stopping
    "per_alpha":      0.6,
    "terminal_soc_weight": 1.0,   # end-of-episode SoC L1 penalty toward episode-start SoC; 0.5 ≈ half c_j[0] signal per MWh
    "validation_days": 30,   # dev default; sweep uses 0 → selection on full train set (SEM≈0.62 vs 2.0)
    "seed":           42,
}

# Actor log-std clipping.
# These bounds keep exploration finite and the squashed-Gaussian numerically stable.
LOG_STD_MIN = -5.0
LOG_STD_MAX =  1.0


# ================================================================
# SUM TREE (for Prioritized Experience Replay)
# ================================================================

class SumTree:
    """
    Flat binary tree for O(log N) PER sampling and updates.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        # tree[0] unused; tree[1] = root; leaves at [capacity, 2*capacity)
        self.tree = np.zeros(2 * capacity, dtype=np.float64)

    def update(self, idx: int, priority: float) -> None:
        """Set leaf idx to priority and propagate the sum change up."""
        pos = idx + self.capacity
        self.tree[pos] = priority
        pos >>= 1                       # move to parent
        while pos >= 1:
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]
            pos >>= 1

    def retrieve(self, value: float) -> int:
        """Return the data index whose prefix sum spans the sampled value."""
        pos = 1
        while pos < self.capacity:
            left = 2 * pos
            if value <= self.tree[left]:
                pos = left
            else:
                value -= self.tree[left]
                pos = left + 1
        return pos - self.capacity

    @property
    def total(self) -> float:
        return float(self.tree[1])


# ================================================================
# REPLAY BUFFER
# ================================================================

class ReplayBuffer:
    """
    Circular replay buffer with optional PER.

    Stores observations, actions, rewards, dones, and current/next feasibility
    bounds used by the safety-aware critic target.
    """

    def __init__(
        self,
        obs_dim:       int,
        act_dim:       int,
        max_size:      int   = 100_000,
        per_alpha:     float = 0.6,
        per_beta_init: float = 0.4,
        per_eps:       float = 1e-6,
    ):
        self.max_size = max_size
        self.ptr  = 0
        self.size = 0

        self.obs         = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act         = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew         = np.zeros((max_size, 1),       dtype=np.float32)
        self.next_obs    = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.done        = np.zeros((max_size, 1),       dtype=np.float32)
        self.bounds      = np.zeros((max_size, 2),       dtype=np.float32)
        self.next_bounds = np.zeros((max_size, 2),       dtype=np.float32)

        # PER bookkeeping
        self._sum_tree     = SumTree(max_size)
        self._max_priority = 1.0        # new transitions get max priority (see Schaul §3.3)
        self.per_alpha     = per_alpha
        self.per_eps       = per_eps
        self._beta         = per_beta_init
        self._beta_init    = per_beta_init
        self._beta_end     = 1.0

    def add(
        self,
        obs:            np.ndarray,
        act:            np.ndarray,
        rew:            float,
        next_obs:       np.ndarray,
        done:           bool,
        p_lo_norm:      float = -1.0,
        p_hi_norm:      float =  1.0,
        next_p_lo_norm: float = -1.0,
        next_p_hi_norm: float =  1.0,
    ) -> None:
        """Store one transition and its current/next normalized safety bounds."""
        idx = self.ptr                  # data index BEFORE advance (for SumTree)
        self.obs[idx]              = obs
        self.act[idx]              = act
        self.rew[idx, 0]           = float(rew)
        self.next_obs[idx]         = next_obs
        self.done[idx, 0]          = float(done)
        self.bounds[idx, 0]        = float(p_lo_norm)
        self.bounds[idx, 1]        = float(p_hi_norm)
        self.next_bounds[idx, 0]   = float(next_p_lo_norm)
        self.next_bounds[idx, 1]   = float(next_p_hi_norm)
        # Give new transitions the current max priority until they are updated.
        self._sum_tree.update(idx, self._max_priority)
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_episode(self, transitions: list) -> None:
        """Bulk-add transitions with optional current/next safety bounds."""
        for t in transitions:
            if len(t) == 9:
                obs, act, rew, next_obs, done, p_lo, p_hi, np_lo, np_hi = t
                self.add(obs, act, rew, next_obs, done, p_lo, p_hi, np_lo, np_hi)
            elif len(t) == 7:
                obs, act, rew, next_obs, done, p_lo, p_hi = t
                self.add(obs, act, rew, next_obs, done, p_lo, p_hi)
            else:
                obs, act, rew, next_obs, done = t
                self.add(obs, act, rew, next_obs, done)

    def sample(self, batch_size: int):
        """
        Sample a mini-batch with PER stratification and IS weights.
        """
        total   = self._sum_tree.total
        segment = total / batch_size

        indices = np.empty(batch_size, dtype=np.int64)
        for i in range(batch_size):
            mass       = np.random.uniform(segment * i, segment * (i + 1))
            # Guard against floating-point overshoot at the right boundary
            mass       = min(mass, total - 1e-9)
            indices[i] = self._sum_tree.retrieve(mass)

        # Sampling probabilities.
        priorities = self._sum_tree.tree[indices + self._sum_tree.capacity]
        probs      = priorities / total

        # IS weights w_i = (N·P(i))^{-β} (eq 2), normalised by max_j(w_j)
        # so the maximum weight is always 1.0 (clamps gradient scale).
        weights = (self.size * probs) ** (-self._beta)
        weights /= weights.max()
        weights  = weights.astype(np.float32)[:, np.newaxis]  # (batch, 1)

        return (
            self.obs[indices],
            self.act[indices],
            self.rew[indices],
            self.next_obs[indices],
            self.done[indices],
            self.bounds[indices],
            self.next_bounds[indices],
            indices,    # for update_priorities()
            weights,    # IS correction, shape (batch, 1)
        )

    def update_priorities(
        self,
        indices:   np.ndarray,
        td_errors: np.ndarray,
    ) -> None:
        """Update PER priorities from per-sample TD errors."""
        priorities = (np.abs(td_errors) + self.per_eps) ** self.per_alpha
        for idx, p in zip(indices, priorities):
            self._sum_tree.update(int(idx), float(p))
        self._max_priority = max(self._max_priority, float(priorities.max()))

    def anneal_beta(self, fraction: float) -> None:
        """Linearly anneal the PER importance-sampling exponent."""
        self._beta = min(
            self._beta_end,
            self._beta_init + fraction * (self._beta_end - self._beta_init),
        )

    def __len__(self) -> int:
        return self.size


# ================================================================
# NEURAL NETWORKS
# ================================================================

class SquashedGaussianActor(tf.keras.Model):
    """
    Dual-stream actor: Conv1D over the 25-point LMP window plus an MLP over
    scalar state features, followed by a squashed Gaussian policy.
    """

    # State index constants (must match BESSEnvironment._get_obs layout)
    # obs[0]     : SoC_norm
    # obs[1:26]  : LMP window — [1..24] = lag-24..lag-1 (oldest first), [25] = current LMP
    # obs[26]    : PV_norm
    # obs[27]    : step_norm
    # obs[28:32] : seg_occ (4 segments)
    # obs[32]    : p_lo_norm
    # obs[33]    : p_hi_norm
    # obs[34]    : sin_dow
    # obs[35]    : cos_dow
    # obs[36]    : load_norm
    # obs[37]    : wind_norm
    _LMP_START  = 1
    _LMP_END    = 26
    _SCALAR_IDX = [0] + list(range(26, 38))   # SoC(1) + scalars[26..37](12) = 13 total

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        action_scale_floor: float = 0.04,
    ):
        super().__init__()
        self.action_scale_floor = float(action_scale_floor)

        # --- LMP multi-scale temporal stream ---
        # Short-range: kernel=4 captures 4-hour local price spikes.
        self.lmp_conv_short = tf.keras.layers.Conv1D(32, kernel_size=4, strides=1,
                                                      padding="causal", activation="relu")
        # Medium-range: kernel=8 captures 8-hour intra-day price trends.
        self.lmp_conv_long  = tf.keras.layers.Conv1D(32, kernel_size=8, strides=1,
                                                      padding="causal", activation="relu")
        # Full-day: kernel=24 closes the receptive-field gap to the daily LMP cycle.
        self.lmp_conv_daily = tf.keras.layers.Conv1D(32, kernel_size=24, strides=1,
                                                      padding="causal", activation="relu")
        # Each branch: concat(last_position, avg) → 64-dim; fuse 3×64=192 → 32
        self.lmp_dense      = tf.keras.layers.Dense(32, activation="relu")

        # --- Scalar physical stream ---
        self.scalar_dense = tf.keras.layers.Dense(64, activation="relu")

        # Normalize the fused embedding before the shared trunk.
        self.joint_ln = tf.keras.layers.LayerNormalization()

        # --- Joint trunk ---
        self.trunk1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.trunk2 = tf.keras.layers.Dense(hidden_dim, activation="relu")

        # --- Output heads ---
        self.mu_layer = tf.keras.layers.Dense(act_dim)
        # Start near-deterministic and let SAC increase variance as needed.
        self.log_std_layer = tf.keras.layers.Dense(
            act_dim,
            kernel_initializer="zeros",
            bias_initializer=tf.keras.initializers.Constant(LOG_STD_MIN + 2.0),
        )

        # Pre-build the gather index used by the scalar stream.
        self._scalar_idx_tf = tf.constant(self._SCALAR_IDX, dtype=tf.int32)

    def _encode(self, obs: tf.Tensor) -> tf.Tensor:
        """Encode the observation into the shared trunk representation."""
        # LMP stream: (batch, 25) → (batch, 25, 1) for Conv1D
        lmp = tf.expand_dims(obs[:, self._LMP_START:self._LMP_END], axis=-1)

        # Triple-scale convolution; last position retains full causal context, avg adds stability
        short_c = self.lmp_conv_short(lmp)                                              # (batch, 25, 32)
        long_c  = self.lmp_conv_long(lmp)                                               # (batch, 25, 32)
        daily_c = self.lmp_conv_daily(lmp)                                              # (batch, 25, 32)
        short_e = tf.concat([short_c[:, -1, :], tf.reduce_mean(short_c, axis=1)], axis=-1)  # (batch, 64)
        long_e  = tf.concat([long_c[:,  -1, :], tf.reduce_mean(long_c,  axis=1)], axis=-1)  # (batch, 64)
        daily_e = tf.concat([daily_c[:, -1, :], tf.reduce_mean(daily_c, axis=1)], axis=-1)  # (batch, 64)
        lmp_e   = self.lmp_dense(tf.concat([short_e, long_e, daily_e], axis=-1))       # (batch, 32)

        # Scalar stream (use pre-built index to avoid per-call tf.constant allocation)
        scalar     = tf.gather(obs, self._scalar_idx_tf, axis=1)   # (batch, 13)
        scalar_e   = self.scalar_dense(scalar)                     # (batch, 64)

        # LayerNorm on concatenated embedding, then joint trunk
        joint = self.joint_ln(tf.concat([lmp_e, scalar_e], axis=-1))  # (batch, 96) normalised
        return self.trunk2(self.trunk1(joint))                         # (batch, 256)

    def _distribution_params(self, obs: tf.Tensor):
        """Return Gaussian parameters for the latent normalized action."""
        _LOGIT_CLIP = 3.0   # clip (not tanh): straight-through gradient for |mu_raw|<3; tanh(3)=0.995≈full power
        h = self._encode(obs)
        mu_raw  = self.mu_layer(h)
        mu      = tf.clip_by_value(mu_raw, -_LOGIT_CLIP, _LOGIT_CLIP)
        log_std = tf.clip_by_value(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    # Obs indices for PTDF safety bounds (must match BESSEnvironment._get_obs layout)
    _P_LO_IDX = 32   # obs[32] = p_lo / P_MAX  ∈ [-1, 0]
    _P_HI_IDX = 33   # obs[33] = p_hi / P_MAX  ∈ [0, 1]

    def _map_to_feasible_interval(
        self,
        u: tf.Tensor,
        p_lo_norm: tf.Tensor,
        p_hi_norm: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Map squashed latent action u=tanh(z) into [p_lo_norm, p_hi_norm] while
        preserving 0 as the neutral point.

        Positive u spans [0, p_hi], negative u spans [p_lo, 0].  This removes
        the midpoint bias of the old affine map, where u=0 implied the interval
        center rather than zero dispatch.
        """
        pos_scale = tf.maximum(p_hi_norm, 0.0)
        neg_scale = tf.maximum(-p_lo_norm, 0.0)
        scale = tf.where(u >= 0.0, pos_scale, neg_scale)
        action = scale * u
        return action, scale

    def call(self, obs):
        """
        Sample a bounded action directly inside the current feasible interval.
        """
        mu, log_std = self._distribution_params(obs)
        std = tf.exp(log_std)

        # Extract PTDF safety bounds from observation
        p_lo_norm = obs[:, self._P_LO_IDX : self._P_LO_IDX + 1]   # (batch, 1) ∈ [-1, 0]
        p_hi_norm = obs[:, self._P_HI_IDX : self._P_HI_IDX + 1]   # (batch, 1) ∈ [0, 1]

        # Reparameterization trick
        eps = tf.random.normal(tf.shape(mu))
        z = mu + std * eps
        u = tf.tanh(z)

        # Log probability under Gaussian (pre-squashing)
        log_pi_normal = (
            -0.5 * (eps ** 2 + tf.math.log(2.0 * math.pi))
            - log_std
        )
        log_pi_normal = tf.reduce_sum(log_pi_normal, axis=-1, keepdims=True)

        # Tanh squashing correction (same formula as standard SAC)
        squash_correction = tf.reduce_sum(
            tf.math.log(1.0 - u ** 2 + 1e-6),
            axis=-1, keepdims=True,
        )

        action, scale = self._map_to_feasible_interval(u, p_lo_norm, p_hi_norm)
        scale_safe = tf.maximum(scale, self.action_scale_floor)
        log_pi = log_pi_normal - tf.math.log(scale_safe) - squash_correction

        return action, log_pi

    def get_deterministic_action(self, obs):
        """Return the deterministic bounded action used at evaluation time."""
        mu, _ = self._distribution_params(obs)
        p_lo_norm = obs[:, self._P_LO_IDX : self._P_LO_IDX + 1]
        p_hi_norm = obs[:, self._P_HI_IDX : self._P_HI_IDX + 1]
        action, _ = self._map_to_feasible_interval(tf.tanh(mu), p_lo_norm, p_hi_norm)
        return action


class TwinQCritic(tf.keras.Model):
    """
    Twin Q-networks with the same dual-stream encoder structure as the actor.
    """

    # State index constants — must match BESSEnvironment._get_obs and Actor._SCALAR_IDX.
    # obs[0]:    SoC_norm
    # obs[1:26]: LMP window (25 dims)
    # obs[26:38]: 12 scalar features (PV, step, seg_occ×4, p_lo, p_hi, sin_dow, cos_dow, load, wind)
    _LMP_START  = 1
    _LMP_END    = 26
    _SCALAR_IDX = [0] + list(range(26, 38))   # 13 scalars total (matches Actor)

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Q1 multi-scale dual-stream encoder
        self.q1_lmp_conv_short = tf.keras.layers.Conv1D(32, kernel_size=4, strides=1,
                                                         padding="causal", activation="relu")
        self.q1_lmp_conv_long  = tf.keras.layers.Conv1D(32, kernel_size=8, strides=1,
                                                         padding="causal", activation="relu")
        self.q1_lmp_conv_daily = tf.keras.layers.Conv1D(32, kernel_size=24, strides=1,
                                                         padding="causal", activation="relu")
        self.q1_lmp_dense      = tf.keras.layers.Dense(32, activation="relu")
        self.q1_scalar         = tf.keras.layers.Dense(64, activation="relu")
        self.q1_joint_ln       = tf.keras.layers.LayerNormalization()
        self.q1_fc1            = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.q1_fc2            = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.q1_out            = tf.keras.layers.Dense(1)

        # Q2 multi-scale dual-stream encoder (identical architecture, separate weights)
        self.q2_lmp_conv_short = tf.keras.layers.Conv1D(32, kernel_size=4, strides=1,
                                                         padding="causal", activation="relu")
        self.q2_lmp_conv_long  = tf.keras.layers.Conv1D(32, kernel_size=8, strides=1,
                                                         padding="causal", activation="relu")
        self.q2_lmp_conv_daily = tf.keras.layers.Conv1D(32, kernel_size=24, strides=1,
                                                         padding="causal", activation="relu")
        self.q2_lmp_dense      = tf.keras.layers.Dense(32, activation="relu")
        self.q2_scalar         = tf.keras.layers.Dense(64, activation="relu")
        self.q2_joint_ln       = tf.keras.layers.LayerNormalization()
        self.q2_fc1            = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.q2_fc2            = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.q2_out            = tf.keras.layers.Dense(1)

        # Pre-built index tensor shared by Q1 and Q2 scalar streams.
        self._scalar_idx_tf = tf.constant(self._SCALAR_IDX, dtype=tf.int32)

    def _encode_one(self, obs, act,
                    lmp_cs, lmp_cl, lmp_cd, lmp_d,
                    scalar_d, joint_ln, fc1, fc2, out):
        """Shared encoder path for one critic head."""
        lmp = tf.expand_dims(obs[:, self._LMP_START:self._LMP_END], axis=-1)

        short_c = lmp_cs(lmp)                                                           # (batch, 25, 32)
        long_c  = lmp_cl(lmp)                                                           # (batch, 25, 32)
        daily_c = lmp_cd(lmp)                                                           # (batch, 25, 32)
        short_e = tf.concat([short_c[:, -1, :], tf.reduce_mean(short_c, axis=1)], axis=-1)  # (batch, 64)
        long_e  = tf.concat([long_c[:,  -1, :], tf.reduce_mean(long_c,  axis=1)], axis=-1)  # (batch, 64)
        daily_e = tf.concat([daily_c[:, -1, :], tf.reduce_mean(daily_c, axis=1)], axis=-1)  # (batch, 64)
        lmp_e   = lmp_d(tf.concat([short_e, long_e, daily_e], axis=-1))                # (batch, 32)

        scalar   = tf.gather(obs, self._scalar_idx_tf, axis=1)
        scalar_e = scalar_d(tf.concat([scalar, act], axis=-1))                          # action in scalar stream

        joint = joint_ln(tf.concat([lmp_e, scalar_e], axis=-1))
        return out(fc2(fc1(joint)))

    def call(self, inputs):
        """Return Q1 and Q2 for a batch of observations and actions."""
        obs, act = inputs
        q1 = self._encode_one(obs, act,
                               self.q1_lmp_conv_short, self.q1_lmp_conv_long,
                               self.q1_lmp_conv_daily,
                               self.q1_lmp_dense, self.q1_scalar, self.q1_joint_ln,
                               self.q1_fc1, self.q1_fc2, self.q1_out)
        q2 = self._encode_one(obs, act,
                               self.q2_lmp_conv_short, self.q2_lmp_conv_long,
                               self.q2_lmp_conv_daily,
                               self.q2_lmp_dense, self.q2_scalar, self.q2_joint_ln,
                               self.q2_fc1, self.q2_fc2, self.q2_out)
        return q1, q2


# ================================================================
# SAC AGENT
# ================================================================

class SACAgent:
    """
    Soft Actor-Critic agent with twin critics and automatic entropy tuning.
    """

    def __init__(self, obs_dim: int, act_dim: int, hp: dict):
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim
        self.gamma      = hp["gamma"]
        self.tau        = hp["tau"]
        self.batch_size = hp["batch_size"]

        hidden_dim = hp["hidden_dim"]

        # --- Networks ---
        self.actor         = SquashedGaussianActor(
            obs_dim, act_dim, hidden_dim,
            action_scale_floor=float(hp.get("action_scale_floor", 0.04)),
        )
        self.critic        = TwinQCritic(obs_dim, act_dim, hidden_dim)
        self.critic_target = TwinQCritic(obs_dim, act_dim, hidden_dim)

        # Build weights by running a dummy forward pass
        _dummy_obs = tf.zeros((1, obs_dim), dtype=tf.float32)
        _dummy_act = tf.zeros((1, act_dim), dtype=tf.float32)
        self.actor(_dummy_obs)
        self.critic([_dummy_obs, _dummy_act])
        self.critic_target([_dummy_obs, _dummy_act])

        # Pre-trace the per-step actor calls used during rollout and evaluation.
        self._tf_act_stoch = tf.function(
            lambda obs: self.actor(obs),
            reduce_retracing=True,
        )
        self._tf_act_det = tf.function(
            lambda obs: self.actor.get_deterministic_action(obs),
            reduce_retracing=True,
        )

        # Initialise target == online critic
        self.critic_target.set_weights(self.critic.get_weights())

        # --- Optimizers ---
        # Use gradient clipping for stability.
        lr = hp["lr"]
        alpha_lr = float(hp.get("alpha_lr", lr))
        self.actor_opt  = tf.keras.optimizers.Adam(lr, global_clipnorm=2.0)
        self.critic_opt = tf.keras.optimizers.Adam(lr, global_clipnorm=2.0)
        self.alpha_opt  = tf.keras.optimizers.Adam(alpha_lr, global_clipnorm=0.5)

        # --- Entropy coefficient (automatic tuning) ---
        # log_alpha is trainable and alpha = exp(log_alpha).
        _init_alpha = float(hp.get("init_alpha", 0.01))
        self.log_alpha      = tf.Variable(
            math.log(_init_alpha), trainable=True, dtype=tf.float32
        )
        self.target_entropy = tf.constant(
            hp["target_entropy_scale"] * float(act_dim), dtype=tf.float32
        )
        # Hard lower bound: alpha never falls below alpha_min.
        _alpha_min = float(hp.get("alpha_min", 0.0005))
        self._log_alpha_min = tf.constant(math.log(_alpha_min), dtype=tf.float32)
        # Hard upper bound on alpha.
        _alpha_max = float(hp.get("alpha_max", 1.0))
        self._log_alpha_max = tf.constant(math.log(_alpha_max), dtype=tf.float32)

        # Pre-build optimizer state for the full variable sets.
        for _opt, _vars in (
            (self.actor_opt, self.actor.trainable_variables),
            (self.critic_opt, self.critic.trainable_variables),
            (self.alpha_opt, [self.log_alpha]),
        ):
            try:
                _opt.build(_vars)
            except Exception:
                # Older TF/Keras paths may lazily build on first gradient step.
                # In that case this no-op fallback preserves compatibility.
                pass

        # --- Anticipatory projection penalty weight ---
        self.proj_lambda = tf.constant(hp.get("proj_lambda", 0.1), dtype=tf.float32)

        # Discount factor applied to the stored N-step return target.
        self.n_step  = int(hp.get("n_step", 1))
        self.gamma_n = tf.constant(
            self.gamma ** self.n_step, dtype=tf.float32
        )

    @property
    def alpha(self) -> tf.Tensor:
        return tf.exp(self.log_alpha)

    # ------------------------------------------------------------------
    # Gradient update methods (compiled for speed)
    # ------------------------------------------------------------------

    @tf.function
    def _update_critic(
        self,
        obs:         tf.Tensor,
        act:         tf.Tensor,
        rew:         tf.Tensor,
        next_obs:    tf.Tensor,
        done:        tf.Tensor,
        is_weights:  tf.Tensor,   # (batch, 1)  PER importance-sampling weights
    ):
        """
        Apply one IS-weighted critic update and return TD errors for PER.
        """
        # Actor already outputs actions inside the feasible interval.
        next_act, next_log_pi = self.actor(next_obs)

        # Clamp log_pi to keep targets numerically stable.
        # Upper bound: at half_width=0.1 (PTDF-constrained) and log_std=-2,
        # log_pi ≈ 3.4; with log_std=-5 (floor) it can reach ~6. Cap at 4.0.
        next_log_pi = tf.clip_by_value(next_log_pi, -20.0, 4.0)

        # N-step target Q values.
        q1_next, q2_next = self.critic_target([next_obs, next_act])
        min_q_next = tf.minimum(q1_next, q2_next)
        # Clamp the target to keep early critic updates bounded.
        q_target = tf.stop_gradient(
            tf.clip_by_value(
                rew + self.gamma_n * (1.0 - done) * (min_q_next - self.alpha * next_log_pi),
                -100.0, 100.0,
            )
        )

        # IS-weighted critic gradient step.
        # Per-sample loss: l_i = 0.5 * w_i * [(Q1_i - y)^2 + (Q2_i - y)^2]
        with tf.GradientTape() as tape:
            q1, q2 = self.critic([obs, act])
            per_sample_loss = 0.5 * is_weights * (
                tf.square(q1 - q_target) + tf.square(q2 - q_target)
            )
            critic_loss = tf.reduce_mean(per_sample_loss)

        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(
            zip(grads, self.critic.trainable_variables)
        )

        # Per-sample TD errors for the next PER priority update.
        td_errors = tf.squeeze(
            0.5 * (tf.abs(q1 - q_target) + tf.abs(q2 - q_target)),
            axis=1,
        )   # shape (batch,)
        return critic_loss, td_errors

    @tf.function
    def _update_actor(self, obs: tf.Tensor) -> tf.Tensor:
        """
        Apply one actor update using bounded actions from the policy.
        """
        with tf.GradientTape() as tape:
            act, log_pi = self.actor(obs)    # a_safe in [p_lo_norm, p_hi_norm]
            log_pi = tf.clip_by_value(log_pi, -20.0, 4.0)
            q1, q2 = self.critic([obs, act])          # Q at safe action
            min_q   = tf.minimum(q1, q2)
            proj_pen = tf.constant(0.0)               # no projection needed (kept for log API)
            actor_loss = tf.reduce_mean(self.alpha * log_pi - min_q)

        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(
            zip(grads, self.actor.trainable_variables)
        )
        return actor_loss, proj_pen

    @tf.function
    def _update_alpha(self, obs: tf.Tensor) -> tf.Tensor:
        """
        Update the entropy coefficient toward the target entropy.
        """
        with tf.GradientTape() as tape:
            _, log_pi = self.actor(obs)
            # Upper clip at 0.0 (not 2.0) for alpha only.
            # The actor adds -log(half_width_safe) to log_pi; at PTDF-blocked steps
            # where half_width_safe=0.1 this term is +2.3, making log_pi artificially
            # positive. That inflated signal would push alpha toward zero on 61% of
            # steps regardless of true policy entropy. Clipping at 0 treats any
            # positive log_pi as a neutral signal, letting unconstrained active steps
            # (which have genuine negative log_pi) determine the alpha direction.
            log_pi = tf.clip_by_value(log_pi, -20.0, 0.0)
            # Stop gradient: log_pi is a constant label for alpha update
            alpha_loss = -tf.reduce_mean(
                self.log_alpha * tf.stop_gradient(log_pi + self.target_entropy)
            )

        grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_opt.apply_gradients(zip(grads, [self.log_alpha]))
        # Hard floor: clamp log_alpha from below so alpha >= alpha_min
        self.log_alpha.assign(
            tf.clip_by_value(self.log_alpha, self._log_alpha_min, self._log_alpha_max)
        )
        return alpha_loss

    def _soft_update_target(self) -> None:
        """Polyak-average critic weights into target critic."""
        tau = self.tau
        for target_var, online_var in zip(
            self.critic_target.variables, self.critic.variables
        ):
            target_var.assign(tau * online_var + (1.0 - tau) * target_var)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_step(self, buffer: ReplayBuffer) -> dict:
        """Run one SAC update from a replay-buffer mini-batch."""
        (obs, act, rew, next_obs, done,
         _bounds, _next_bounds,
         sample_indices, is_weights) = buffer.sample(self.batch_size)

        # Pass numpy arrays directly to avoid building extra TensorFlow constants
        # in the Python training loop.
        obs_tf          = obs.astype(np.float32)
        act_tf          = act.astype(np.float32)
        rew_tf          = rew.astype(np.float32)
        next_obs_tf     = next_obs.astype(np.float32)
        done_tf         = done.astype(np.float32)
        is_weights_tf   = is_weights.astype(np.float32)   # (batch, 1)

        # IS-weighted critic update; returns per-sample |δ_i| for PER
        critic_loss, td_errors = self._update_critic(
            obs_tf, act_tf, rew_tf, next_obs_tf, done_tf,
            is_weights_tf,
        )
        actor_loss, proj_pen   = self._update_actor(obs_tf)
        alpha_loss             = self._update_alpha(obs_tf)
        self._soft_update_target()

        # Refresh SumTree priorities with the freshly computed TD errors
        buffer.update_priorities(sample_indices, td_errors.numpy())

        return {
            "critic_loss": float(critic_loss),
            "actor_loss":  float(actor_loss),
            "alpha_loss":  float(alpha_loss),
            "alpha":       float(self.alpha),
            "proj_pen":    float(proj_pen),   # trends → 0 as actor learns feasibility
        }

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select one action for a single observation."""
        # Use the pre-traced wrappers created in __init__.
        obs_tf = obs[np.newaxis].astype(np.float32)  # (1, obs_dim) — fixed shape → single trace
        if deterministic:
            act_tf = self._tf_act_det(obs_tf)
        else:
            act_tf, _ = self._tf_act_stoch(obs_tf)
        return act_tf.numpy()[0]   # shape (act_dim,) = (1,)

    def save(self, path: str) -> None:
        """
        Save actor, critic, and log_alpha to disk.
        path is a prefix; extensions are appended automatically.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.actor.save_weights(path + "_actor.weights.h5")
        self.critic.save_weights(path + "_critic.weights.h5")
        np.save(path + "_log_alpha.npy", self.log_alpha.numpy())

    def load(self, path: str) -> None:
        """Load weights from a previously saved checkpoint."""
        self.actor.load_weights(path + "_actor.weights.h5")
        self.critic.load_weights(path + "_critic.weights.h5")
        self.critic_target.set_weights(self.critic.get_weights())
        self.log_alpha.assign(float(np.load(path + "_log_alpha.npy")))


# ================================================================
# PERFECT FORESIGHT LP ORACLE  (multi-segment Xu et al. 2017)
# ================================================================

class PerfectForesightLP:
    """
    Multi-segment perfect-foresight LP used for warm-start trajectories.

    The formulation follows the Xu-style segment model for degradation and adds
    PTDF-based network limits so expert actions respect the same feasibility
    structure as the environment.
    """

    def __init__(self, env: BESSEnv):
        from scipy.sparse import lil_matrix as _lil

        self.T        = env.EPS_LEN
        self.J        = env.J_DEG
        self.P_max    = env.P_MAX
        self.SoC_max  = env.SOC_MAX
        self.eta_ch   = env.ETA_CH
        self.eta_dis  = env.ETA_DIS
        self.seg_size = env.SEG_SIZE
        self.c_j      = env.c_deg.copy()    # shape (J,) [$/MWh extracted from battery]

        # PTDF data for episode-level network bounds.
        _PTDF_THRESH  = 1e-5
        self.ptdf_k   = env.ptdf_k.astype(np.float64)     # (120,) PTDF of BESS bus
        self.f_lmax   = env.f_lmax.astype(np.float64)     # (120,) line ratings [MW]
        self.active   = np.abs(self.ptdf_k) > _PTDF_THRESH  # (120,) bool

        # Physical SoC minimum only; no artificial depth gate.
        self.SoC_min = env.SOC_MIN

        # Initial segment energies implied by SOC_INIT.
        self.e0 = self._initial_seg_energies(env.SOC_INIT)

        # Pre-build the static LP structure.
        T, J = self.T, self.J
        n_vars = 3 * T * J

        # Variable block offsets
        i_dis = 0           # p_dis[t,j] at x[i_dis + t*J + j]
        i_ch  = T * J       # p_ch[t,j]  at x[i_ch  + t*J + j]
        i_e   = 2 * T * J   # e[t,j]     at x[i_e   + t*J + j]

        # ----------------------------------------------------------------
        # Equality constraints: energy balance (T*J rows)
        # ----------------------------------------------------------------
        A_eq = _lil((T * J, n_vars), dtype=np.float64)
        b_eq = np.zeros(T * J, dtype=np.float64)

        for t in range(T):
            for j in range(J):
                row = t * J + j
                A_eq[row, i_e   + t * J + j] =  1.0
                A_eq[row, i_dis + t * J + j] =  1.0 / self.eta_dis
                A_eq[row, i_ch  + t * J + j] = -self.eta_ch
                if t == 0:
                    b_eq[row] = self.e0[j]
                else:
                    A_eq[row, i_e + (t - 1) * J + j] = -1.0
                    # b_eq[row] = 0.0  (already initialised)

        # ----------------------------------------------------------------
        # Inequality constraints: power rating + SoC bounds (4T rows)
        # ----------------------------------------------------------------
        A_ub = _lil((4 * T, n_vars), dtype=np.float64)
        b_ub = np.zeros(4 * T, dtype=np.float64)

        for t in range(T):
            for j in range(J):
                A_ub[t,           i_dis + t * J + j] =  1.0   # Σ p_dis ≤ P_max
                A_ub[T + t,       i_ch  + t * J + j] =  1.0   # Σ p_ch  ≤ P_max
                A_ub[2 * T + t,   i_e   + t * J + j] = -1.0   # −Σ e    ≤ −SoC_min
                A_ub[3 * T + t,   i_e   + t * J + j] =  1.0   # Σ e     ≤  SoC_max
            b_ub[t]           = self.P_max
            b_ub[T + t]       = self.P_max
            b_ub[2 * T + t]   = -self.SoC_min
            b_ub[3 * T + t]   = self.SoC_max

        # ----------------------------------------------------------------
        # PTDF inequality constraints: net-injection bounds (2T rows, STATIC structure)
        # ----------------------------------------------------------------
        # Row t      (0..T-1): Σ_j p_dis[t,j] − Σ_j p_ch[t,j] ≤ p_hi_ptdf[t]
        # Row T+t (T..2T-1): −Σ_j p_dis[t,j] + Σ_j p_ch[t,j] ≤ −p_lo_ptdf[t]
        #
        # The coefficient matrix is STATIC (same for every episode); only the RHS
        # b_ptdf = [p_hi_ptdf[0..T-1], −p_lo_ptdf[0..T-1]] is episode-dependent and
        # recomputed in solve() from the episode's baseline line flows f_base_episode.
        A_ptdf = _lil((2 * T, n_vars), dtype=np.float64)
        for t in range(T):
            for j in range(J):
                A_ptdf[t,       i_dis + t * J + j] =  1.0   # +p_dis → upper net bound
                A_ptdf[t,       i_ch  + t * J + j] = -1.0   # −p_ch  → upper net bound
                A_ptdf[T + t,   i_dis + t * J + j] = -1.0   # −p_dis → lower net bound
                A_ptdf[T + t,   i_ch  + t * J + j] =  1.0   # +p_ch  → lower net bound

        # ----------------------------------------------------------------
        # Box bounds
        # ----------------------------------------------------------------
        self._lp_bounds = (
            [(0.0, self.P_max)]      * (T * J)   # p_dis
            + [(0.0, self.P_max)]    * (T * J)   # p_ch
            + [(0.0, self.seg_size)] * (T * J)   # e (per-segment capacity)
        )

        self._A_eq   = A_eq.tocsr()
        self._b_eq   = b_eq
        self._A_ub   = A_ub.tocsr()
        self._b_ub   = b_ub
        self._A_ptdf = A_ptdf.tocsr()   # static PTDF coefficient matrix

    def _initial_seg_energies(self, soc: float) -> np.ndarray:
        """Decompose scalar SoC into shallow-to-deep segment energies."""
        E_cap = self.J * self.seg_size
        e0 = np.zeros(self.J, dtype=np.float64)
        for j in range(self.J):
            seg_upper = E_cap - j * self.seg_size
            seg_lower = E_cap - (j + 1) * self.seg_size
            e0[j] = max(0.0, min(soc, seg_upper) - max(seg_lower, 0.0))
        return e0

    def _ptdf_bounds_episode(
        self, f_base_episode: np.ndarray
    ) -> tuple:
        """Vectorized PTDF net-injection bounds for one episode."""
        T   = min(self.T, len(f_base_episode))
        fb  = np.asarray(f_base_episode[:T], dtype=np.float64)   # (T, 120)

        ptdf_act = self.ptdf_k[self.active]   # (n_act,)
        f_lim    = self.f_lmax[self.active]   # (n_act,)
        fb_act   = fb[:, self.active]         # (T, n_act)

        # Headroom numerators for each line l and each direction:
        #   upper_num[t,l] = f_lim[l] - fb_act[t,l]   (slack before upper limit)
        #   lower_num[t,l] = f_lim[l] + fb_act[t,l]   (slack before lower limit)
        upper_num = f_lim[np.newaxis, :] - fb_act   # (T, n_act)
        lower_num = f_lim[np.newaxis, :] + fb_act   # (T, n_act)

        # Per-line per-timestep bounds on p_net:
        #   ptdf > 0: upper p_net = upper_num / ptdf,  lower p_net = -lower_num / ptdf
        #   ptdf < 0: upper p_net = lower_num / (-ptdf), lower p_net = -upper_num / (-ptdf)
        p_hi = np.full(T, self.P_max)
        p_lo = np.full(T, -self.P_max)

        pos = ptdf_act > 0
        neg = ~pos

        if pos.any():
            # upper bound from pos lines (smallest positive slack / ptdf)
            p_hi = np.minimum(p_hi, (upper_num[:, pos] / ptdf_act[pos]).min(axis=1))
            # lower bound from pos lines (largest of -lower_num / ptdf)
            p_lo = np.maximum(p_lo, (-lower_num[:, pos] / ptdf_act[pos]).max(axis=1))

        if neg.any():
            # For negative ptdf: dividing by a negative flips the bound direction.
            # Upper bound on p_net: -lower_num / ptdf = lower_num / |ptdf|
            p_hi = np.minimum(p_hi, (lower_num[:, neg] / (-ptdf_act[neg])).min(axis=1))
            # Lower bound on p_net: -upper_num / ptdf = upper_num / |ptdf|
            p_lo = np.maximum(p_lo, (-upper_num[:, neg] / (-ptdf_act[neg])).max(axis=1))

        # Clip to physical power rating
        p_hi = np.clip(p_hi, -self.P_max, self.P_max)
        p_lo = np.clip(p_lo, -self.P_max, self.P_max)

        # Pad to T if f_base_episode was shorter
        if T < self.T:
            p_hi = np.concatenate([p_hi, np.full(self.T - T, self.P_max)])
            p_lo = np.concatenate([p_lo, np.full(self.T - T, -self.P_max)])

        return p_hi, p_lo

    def solve(
        self,
        lmps_episode: np.ndarray,
        f_base_episode: np.ndarray | None = None,
        soc_init: float | None = None,
    ):
        """Solve the episode LP and return profit plus aggregate charge/discharge."""
        from scipy.sparse import vstack as _sp_vstack

        T, J = self.T, self.J
        lmps = np.asarray(lmps_episode[:T], dtype=np.float64)

        # ------------------------------------------------------------------
        # Episode-specific equality RHS: update t=0 energy-balance rows
        # if the episode starts from a different SoC than SOC_INIT.
        # ------------------------------------------------------------------
        if soc_init is not None:
            e0_ep = self._initial_seg_energies(float(soc_init))
            b_eq = self._b_eq.copy()
            b_eq[:J] = e0_ep   # only t=0 rows (indices 0..J-1) differ
        else:
            b_eq = self._b_eq

        # ------------------------------------------------------------------
        # Episode-dependent objective.
        # c_j is in $/MWh extracted from the battery, while p_dis is grid-side
        # discharge power. Divide by η_dis to compare both on the same basis.
        # p_dis[t,j]: (c_j/η_dis − λ_t) — profitable when λ_t > c_j/η_dis.
        # p_ch[t,j]:  λ_t               — charging costs the spot price.
        # e[t,j]:     0                  — no direct cost on stored energy.
        # ------------------------------------------------------------------
        c_dis = np.array(
            [(self.c_j[j] / self.eta_dis) - lmps[t]
             for t in range(T) for j in range(J)],
            dtype=np.float64,
        )
        c_ch = np.array(
            [lmps[t] for t in range(T) for j in range(J)],
            dtype=np.float64,
        )
        c_obj = np.concatenate([c_dis, c_ch, np.zeros(T * J)])

        # ------------------------------------------------------------------
        # PTDF network constraints (episode-specific RHS only).
        # A_ptdf is pre-built in __init__; only b_ptdf changes per episode.
        # ------------------------------------------------------------------
        if f_base_episode is not None:
            p_hi_ptdf, p_lo_ptdf = self._ptdf_bounds_episode(f_base_episode)
            # Inequality rows:
            #   rows 0..T-1:   p_net ≤ p_hi_ptdf  →  b = p_hi_ptdf
            #   rows T..2T-1: −p_net ≤ −p_lo_ptdf →  b = −p_lo_ptdf
            b_ptdf = np.concatenate([p_hi_ptdf, -p_lo_ptdf])
            A_ub_full = _sp_vstack([self._A_ub, self._A_ptdf], format="csr")
            b_ub_full = np.concatenate([self._b_ub, b_ptdf])
        else:
            A_ub_full = self._A_ub
            b_ub_full = self._b_ub

        result = linprog(
            c=c_obj,
            A_ub=A_ub_full,
            b_ub=b_ub_full,
            A_eq=self._A_eq,
            b_eq=b_eq,
            bounds=self._lp_bounds,
            method="highs",
        )

        if result.status == 0:
            p_dis_seg = result.x[:T * J].reshape(T, J)
            p_ch_seg  = result.x[T * J: 2 * T * J].reshape(T, J)
            p_dis  = p_dis_seg.sum(axis=1)   # aggregate [MW, delivered to grid]
            p_ch   = p_ch_seg.sum(axis=1)    # aggregate [MW, drawn from grid]
            profit = -float(result.fun)       # linprog minimised −profit
            return profit, p_dis, p_ch
        else:
            return 0.0, np.zeros(T), np.zeros(T)


# ================================================================
# TRAINING FUNCTIONS
# ================================================================


def _commit_nstep_transition_from_window(
    window: deque,
    buffer: ReplayBuffer,
    n_step: int,
    gamma: float,
) -> bool:
    """Commit the oldest transition from the rolling N-step window."""
    if len(window) == 0:
        return False

    horizon = n_step if len(window) >= (n_step + 1) else min(n_step, len(window))

    s0, a0, _, _, _, lo0, hi0 = window[0]

    g_n = 0.0
    done_n = False
    steps_used = 0
    for k in range(horizon):
        _, _, r_k, _, d_k, _, _ = window[k]
        g_n += (gamma ** k) * r_k
        steps_used = k + 1
        if d_k:
            done_n = True
            break

    # next observation after the final reward included in G_t^N
    s_n = window[steps_used - 1][3]

    # next-state bounds for Convention-B target projection.
    # If no look-ahead bounds exist (tail flush or terminal), bounds are unused
    # because done_n=True suppresses bootstrap, so we pass the last available pair.
    if (not done_n) and (len(window) > steps_used):
        lo_n = window[steps_used][5]
        hi_n = window[steps_used][6]
    else:
        lo_n = window[steps_used - 1][5]
        hi_n = window[steps_used - 1][6]

    buffer.add(s0, a0, g_n, s_n, done_n, lo0, hi0, lo_n, hi_n)
    window.popleft()
    return True



def run_episode_collect(
    agent:  SACAgent,
    env:    BESSEnv,
    buffer: ReplayBuffer,
    n_step: int   = 1,
    gamma:  float = 0.99,
    terminal_soc_weight: float = 0.0,
) -> tuple:
    """Collect one rollout and store N-step transitions in replay."""
    obs, reset_info = env.reset()
    soc_episode_init = float(reset_info["soc_init_mwh"])   # actual starting SoC (random in train, SOC_INIT in eval)
    ep_return = 0.0
    ep_infos  = []
    n_added   = 0
    # Rolling window: each entry is a raw 1-step transition
    # (obs, act_proj, reward, next_obs, done, p_lo_norm, p_hi_norm)
    window = deque()

    for _ in range(env.EPS_LEN):
        act = agent.act(obs, deterministic=False)   # shape (1,) in [p_lo_norm, p_hi_norm]
        # No external clip needed: bounded-tanh actor already outputs in [p_lo, p_hi].
        # The env's safety layer will further project (redundantly but harmlessly).
        act = np.clip(act, -1.0, 1.0)              # float32 overflow guard only
        next_obs, rew, terminated, truncated, info = env.step(act)
        done = terminated or truncated

        # Terminal SoC shaping: L1 penalty on net energy extracted this episode.
        # Target is soc_episode_init (the actual reset SoC, random in train) so the
        # signal is always "return the battery to where it started", not a fixed constant.
        # Applied here (training only); _run_evaluation() is untouched → clean metrics.
        if terminated and terminal_soc_weight > 0.0:
            rew = float(rew) - terminal_soc_weight * env.c_deg[0] * abs(env.soc - soc_episode_init) * env.RWD_SCALE

        # Projected action and safety bounds for Convention B
        act_proj  = np.array([info["p_safe_mw"] / env.P_MAX], dtype=np.float32)
        p_lo_norm = float(info["p_lo_mw"] / env.P_MAX)
        p_hi_norm = float(info["p_hi_mw"] / env.P_MAX)

        window.append((obs, act_proj, float(rew), next_obs, done,
                       p_lo_norm, p_hi_norm))

        # Full-horizon commits (len >= n_step+1) use look-ahead bounds at s_{t+N}.
        while len(window) >= n_step + 1:
            _commit_nstep_transition_from_window(window, buffer, n_step, gamma)
            n_added += 1

        ep_return += rew
        ep_infos.append(info)
        obs = next_obs

        if done:
            break

    # Tail flush: commit remaining partial windows so episode-end transitions are kept.
    while len(window) > 0:
        _commit_nstep_transition_from_window(window, buffer, n_step, gamma)
        n_added += 1

    return float(ep_return), ep_infos, n_added


def lp_warmstart(
    env:       BESSEnv,
    buffer:    ReplayBuffer,
    lp_oracle: "PerfectForesightLP",
    n_target:  int,
    gamma:     float = 0.99,
) -> int:
    """
    Pre-fill the replay buffer with PTDF-feasible LP trajectories.

    Always stores 1-step transitions regardless of the online SAC n_step.
    """
    _lp_nstep = 1
    n_added = 0

    while n_added < n_target:
        obs, _ = env.reset()   # training reset; start SoC follows the env protocol

        start_t    = env.global_t
        soc_start  = env.soc                                          # episode-start SoC [MWh]
        lmps_ep    = env.lmps[start_t : start_t + env.EPS_LEN, env.bess_col]
        f_base_ep  = env.line_flows[start_t : start_t + env.EPS_LEN] # (T, 120)
        _, p_dis, p_ch = lp_oracle.solve(lmps_ep, f_base_ep, soc_init=soc_start)
        actions_lp = np.clip((p_dis - p_ch) / env.P_MAX, -1.0, 1.0).astype(np.float32)

        window = deque()

        for t in range(env.EPS_LEN):
            act = np.array([actions_lp[t]], dtype=np.float32)
            next_obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            act_proj  = np.array([info["p_safe_mw"] / env.P_MAX], dtype=np.float32)
            p_lo_norm = float(info["p_lo_mw"] / env.P_MAX)
            p_hi_norm = float(info["p_hi_mw"] / env.P_MAX)

            window.append((obs, act_proj, float(rew), next_obs, done,
                           p_lo_norm, p_hi_norm))

            while len(window) >= _lp_nstep + 1 and n_added < n_target:
                _commit_nstep_transition_from_window(window, buffer, _lp_nstep, gamma)
                n_added += 1

            obs = next_obs
            if done or n_added >= n_target:
                break

        while len(window) > 0 and n_added < n_target:
            _commit_nstep_transition_from_window(window, buffer, _lp_nstep, gamma)
            n_added += 1

    return n_added


def _sample_bc_batch(
    buffer: ReplayBuffer,
    batch_size: int,
    active_threshold: float,
    active_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a behavior-cloning batch with explicit emphasis on active LP actions.

    The replay is dominated by idle LP actions, so uniform BC mostly teaches
    "do nothing". This sampler forces a configurable share of the batch to come
    from charge/discharge steps while still keeping some idle context.
    """
    size = len(buffer)
    if size == 0:
        raise ValueError("Cannot sample BC batch from an empty replay buffer.")

    idx_all = np.arange(size, dtype=np.int64)
    act_abs = np.abs(buffer.act[:size, 0])
    active_idx = idx_all[act_abs > active_threshold]
    idle_idx   = idx_all[act_abs <= active_threshold]

    if len(active_idx) == 0 or len(idle_idx) == 0:
        chosen = np.random.choice(idx_all, size=batch_size, replace=size < batch_size)
        return buffer.obs[chosen], buffer.act[chosen]

    n_active = int(round(batch_size * active_frac))
    n_active = max(1, min(batch_size - 1, n_active))
    n_idle   = batch_size - n_active

    chosen_active = np.random.choice(
        active_idx, size=n_active, replace=len(active_idx) < n_active
    )
    chosen_idle = np.random.choice(
        idle_idx, size=n_idle, replace=len(idle_idx) < n_idle
    )
    chosen = np.concatenate([chosen_active, chosen_idle])
    np.random.shuffle(chosen)
    return buffer.obs[chosen], buffer.act[chosen]


def _run_evaluation(
    agent: SACAgent,
    env:   BESSEnv,
    starts: np.ndarray | None = None,
) -> dict:
    """Run deterministic evaluation on the provided episode starts."""
    if starts is None:
        starts = env.test_starts

    returns      = []
    soc_ends     = []
    soc_mins     = []
    revenues     = []
    deg_costs    = []
    n_discharge  = []
    n_charge     = []
    n_idle       = []
    n_ptdf_viol  = []   # steps with PTDF violation (non-zero only when ptdf_enabled=False)
    n_clipped    = []   # steps where safety projection changed the action

    for start_t in starts:
        obs, _    = env.reset(options={"start_t": int(start_t)})
        ep_return = 0.0
        ep_rev    = 0.0
        ep_deg    = 0.0
        ep_n_dis  = 0
        ep_n_ch   = 0
        ep_n_idle = 0
        ep_n_viol = 0
        ep_n_clip = 0
        ep_soc_min = env.soc

        for _ in range(env.EPS_LEN):
            act = np.clip(agent.act(obs, deterministic=True), -1.0, 1.0)
            obs, rew, terminated, truncated, info = env.step(act)
            ep_return  += rew
            ep_rev     += info["revenue_$"]
            ep_deg     += info["deg_cost_$"]
            p_safe      = info["p_safe_mw"]
            ep_soc_min  = min(ep_soc_min, info["soc_mwh"])
            if info.get("ptdf_violated", False):
                ep_n_viol += 1
            if info.get("clipped", False):
                ep_n_clip += 1
            if p_safe > 0.5:
                ep_n_dis  += 1
            elif p_safe < -0.5:
                ep_n_ch   += 1
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
        n_ptdf_viol.append(ep_n_viol)
        n_clipped.append(ep_n_clip)

    returns     = np.array(returns)
    total_steps = np.array(n_discharge) + np.array(n_charge) + np.array(n_idle)
    return {
        "eval_mean":           float(np.mean(returns)),
        "eval_std":            float(np.std(returns)),
        "eval_min":            float(np.min(returns)),
        "eval_max":            float(np.max(returns)),
        "n_eval":              len(returns),
        "soc_end_mean":        float(np.mean(soc_ends)),
        "soc_min_mean":        float(np.mean(soc_mins)),
        "revenue_mean":        float(np.mean(revenues)),
        "deg_cost_mean":       float(np.mean(deg_costs)),
        "profit_mean":         float(np.mean(np.array(revenues) - np.array(deg_costs))),
        "frac_discharge":      float(np.mean(np.array(n_discharge) / np.maximum(total_steps, 1))),
        "frac_charge":         float(np.mean(np.array(n_charge)    / np.maximum(total_steps, 1))),
        "frac_idle":           float(np.mean(np.array(n_idle)      / np.maximum(total_steps, 1))),
        # ptdf_violation_rate > 0 only when env.ptdf_enabled=False (unsafe ablation)
        "ptdf_violation_rate": float(np.mean(np.array(n_ptdf_viol) / np.maximum(total_steps, 1))),
        "clip_rate_mean":      float(np.mean(np.array(n_clipped)   / np.maximum(total_steps, 1))),
    }


def _run_oracle_evaluation(env: BESSEnv) -> dict:
    """Evaluate the perfect-foresight LP on the provided episode starts."""
    starts = env.test_starts
    return _run_oracle_evaluation_on_starts(env, starts)


def _run_oracle_evaluation_on_starts(env: BESSEnv, starts: np.ndarray) -> dict:
    """Evaluate the perfect-foresight LP on the provided episode starts."""
    oracle = PerfectForesightLP(env)
    returns = []
    profits = []

    for start_t in starts:
        lmps_ep    = env.lmps[start_t:start_t + env.EPS_LEN, env.bess_col]
        flows_ep   = env.line_flows[start_t:start_t + env.EPS_LEN]
        _, p_dis, p_ch = oracle.solve(lmps_ep, flows_ep, env.SOC_INIT)
        actions_lp = np.clip((p_dis - p_ch) / env.P_MAX, -1.0, 1.0).astype(np.float32)

        obs, _ = env.reset(options={"start_t": int(start_t)})
        ep_return = 0.0
        ep_profit = 0.0

        for a in actions_lp:
            obs, rew, terminated, truncated, info = env.step(
                np.array([a], dtype=np.float32)
            )
            ep_return += rew
            ep_profit += info["reward_raw_$"]
            if terminated or truncated:
                break

        returns.append(ep_return)
        profits.append(ep_profit)

    return {
        "oracle_eval_mean":   float(np.mean(returns)),
        "oracle_eval_std":    float(np.std(returns)),
        "oracle_profit_mean": float(np.mean(profits)),
        "n_eval":             len(returns),
    }


def train_sac(env: BESSEnv, hp: dict) -> tuple:
    """Train SAC with LP warm-start, periodic evaluation, and checkpointing."""
    seed = hp["seed"]
    np.random.seed(seed)
    tf.random.set_seed(seed)
    selection_starts = getattr(env, "selection_starts", env.test_starts)
    selection_label = getattr(env, "selection_split_name", "test")
    selection_tag = "val" if selection_label == "validation" else selection_label

    obs_dim = env.observation_space.shape[0]   # 38 (SoC+LMP+hist24+PV+step+segs4+p_lo+p_hi+sin_dow+cos_dow+load+wind)
    act_dim = env.action_space.shape[0]        # 1

    buffer = ReplayBuffer(obs_dim, act_dim, hp["buffer_size"],
                          per_alpha=hp.get("per_alpha", 0.0))
    agent  = SACAgent(obs_dim, act_dim, hp)

    print(f"\nSAC Agent created:")
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"  Actor params:  {agent.actor.count_params():,}")
    print(f"  Critic params: {agent.critic.count_params():,}")
    print(f"  Buffer size:   {hp['buffer_size']:,} transitions")
    print(f"  Batch size:    {hp['batch_size']}")
    print(f"  N-step:        {hp.get('n_step', 1)}")
    print(f"  init_alpha:    {hp.get('init_alpha', 0.01):.4f}")
    print(f"  alpha_lr:      {hp.get('alpha_lr', hp['lr']):.1e}")
    print(f"  target_entropy = {float(agent.target_entropy):.2f}")
    print(f"  proj_lambda:   {hp.get('proj_lambda', 0.0):.3f}")
    print(f"  lp_warmstart:  {hp.get('lp_warmstart_steps', 0)} transitions "
          f"({hp.get('lp_warmstart_steps', 0) / hp['buffer_size'] * 100:.1f}% "
          f"of total buffer capacity, n_step=1 forced)")
    print(f"  lp_bc_steps:   {hp.get('lp_bc_steps', 0)}")

    # ---- LP Expert Warm-Start ----
    _ws_steps = hp.get("lp_warmstart_steps", 0)
    _bc_threshold = float(hp.get("lp_bc_active_threshold", 0.02))
    _bc_active_frac = float(hp.get("lp_bc_active_frac", 0.10))

    if _ws_steps > 0:
        print(f"\n{'='*65}")
        print(f"LP Expert Warm-Start (target: {_ws_steps} transitions)")
        print(f"{'='*65}")
        _lp_oracle = PerfectForesightLP(env)
        _lmp_max = float(np.max(env.lmps[env.converged, env.bess_col]))
        _breakevens = _lp_oracle.c_j / _lp_oracle.eta_dis
        _n_profitable = int(np.sum(_lmp_max >= _breakevens))
        print(f"  Oracle economics check: LMP_max={_lmp_max:.2f} $/MWh, "
              f"profitable segments={_n_profitable}/{_lp_oracle.J} "
              f"(breakevens: {', '.join(f'${b:.2f}' for b in _breakevens)})")
        print(f"  Oracle SoC band: [{_lp_oracle.SoC_min:.1f}, {_lp_oracle.SoC_max:.1f}] MWh, "
              f"e0={[f'{e:.1f}' for e in _lp_oracle.e0]} MWh per segment")
        _t_ws = time.time()
        n_ws = lp_warmstart(
            env, buffer, _lp_oracle,
            n_target=_ws_steps,
            gamma=hp["gamma"],
        )
        print(f"  Added {n_ws} LP transitions in {time.time()-_t_ws:.1f}s  "
              f"(buffer fill: {len(buffer)}/{hp['buffer_size']})")
        _lp_actions = buffer.act[:n_ws, 0]
        _lp_dis = float(np.mean(_lp_actions >  _bc_threshold))
        _lp_ch  = float(np.mean(_lp_actions < -_bc_threshold))
        _lp_idle = float(np.mean(np.abs(_lp_actions) <= _bc_threshold))
        print(f"  LP replay mix: dis={_lp_dis*100:.1f}%  "
              f"ch={_lp_ch*100:.1f}%  idle={_lp_idle*100:.1f}%")

        # Retroactively lower LP warmstart priorities so online TD transitions
        # dominate PER sampling once SAC starts.  New online transitions will be
        # inserted at buffer._max_priority (still 1.0), so they are sampled far
        # more often than these deflated LP entries (scale×1.0).
        _lp_pscale = float(hp.get("lp_warmstart_priority_scale", 1.0))
        if _lp_pscale < 1.0 and n_ws > 0:
            _lp_p = max(buffer.per_eps ** buffer.per_alpha, _lp_pscale * buffer._max_priority)
            for _idx in range(n_ws):
                buffer._sum_tree.update(_idx, _lp_p)
            print(f"  LP priority scaled ×{_lp_pscale:.2f} → {_lp_p:.4f} each "
                  f"(online transitions will be inserted at {buffer._max_priority:.4f})")

    # ---- LP Actor BC Pre-Training ----
    # Pretrain the actor on stored LP actions before SAC updates begin.
    _bc_steps = hp.get("lp_bc_steps", 0)
    if _bc_steps > 0 and len(buffer) >= hp["batch_size"]:
        print(f"\n{'='*65}")
        print(f"LP Actor BC Pre-Training ({_bc_steps} steps, batch={hp['batch_size']})")
        print(f"{'='*65}")
        _bc_t = time.time()
        _bc_losses = []
        for _s in range(_bc_steps):
            _obs_np, _act_np = _sample_bc_batch(
                buffer,
                hp["batch_size"],
                active_threshold=_bc_threshold,
                active_frac=_bc_active_frac,
            )
            _obs_b   = tf.constant(_obs_np, dtype=tf.float32)  # (batch, obs_dim)
            _act_lp  = tf.constant(_act_np, dtype=tf.float32)  # (batch, 1) LP action
            with tf.GradientTape() as _tape:
                # Clone the deterministic policy mean.
                _act_pred = agent.actor.get_deterministic_action(_obs_b)  # (batch, 1)
                # Slightly upweight active LP transitions on top of stratified sampling.
                _active_w = 1.0 + 0.0 * tf.cast(tf.abs(_act_lp) > _bc_threshold, tf.float32)
                _bc_loss  = tf.reduce_mean(_active_w * (_act_pred - _act_lp) ** 2)
            _grads = _tape.gradient(_bc_loss, agent.actor.trainable_variables)
            _grads_and_vars = [
                (_g, _v)
                for _g, _v in zip(_grads, agent.actor.trainable_variables)
                if _g is not None
            ]
            agent.actor_opt.apply_gradients(_grads_and_vars)
            _bc_losses.append(float(_bc_loss))
        # Extend BC if not yet converged: guards against unlucky TF initialization.
        _bc_loss_floor = 0.060
        _extra = 0
        while float(np.mean(_bc_losses[-100:])) > _bc_loss_floor and _extra < 2000:
            _obs_np, _act_np = _sample_bc_batch(buffer, hp["batch_size"],
                                                active_threshold=_bc_threshold,
                                                active_frac=_bc_active_frac)
            _obs_b  = tf.constant(_obs_np, dtype=tf.float32)
            _act_lp = tf.constant(_act_np, dtype=tf.float32)
            with tf.GradientTape() as _tape:
                _act_pred = agent.actor.get_deterministic_action(_obs_b)
                _active_w = 1.0 + 1.0 * tf.cast(tf.abs(_act_lp) > _bc_threshold, tf.float32)
                _bc_loss  = tf.reduce_mean(_active_w * (_act_pred - _act_lp) ** 2)
            _grads = _tape.gradient(_bc_loss, agent.actor.trainable_variables)
            agent.actor_opt.apply_gradients(
                [(_g, _v) for _g, _v in zip(_grads, agent.actor.trainable_variables) if _g is not None]
            )
            _bc_losses.append(float(_bc_loss))
            _extra += 1
        print(f"  BC done in {time.time()-_bc_t:.1f}s: "
              f"loss {_bc_losses[0]:.4f} → {_bc_losses[-1]:.4f}  "
              f"(mean last 100: {float(np.mean(_bc_losses[-100:])):.4f}"
              f"{f', +{_extra} ext steps' if _extra else ''})")
        print(f"  BC sampler mix: target active share={_bc_active_frac*100:.0f}% "
              f"above |a|>{_bc_threshold:.2f}")

    # ---- SAC training ----
    print(f"\n{'='*65}")
    print(f"SAC Training ({hp['train_episodes']} episodes)")
    print(f"{'='*65}")

    oracle_eval = _run_oracle_evaluation_on_starts(env, selection_starts)
    print(f"  {selection_label.title()} LP oracle: eval={oracle_eval['oracle_eval_mean']:+.4f}±"
          f"{oracle_eval['oracle_eval_std']:.4f} | "
          f"profit=${oracle_eval['oracle_profit_mean']:.0f}")

    # Give each run its own output directory.
    _run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir        = os.path.join(OUTPUTS_DIR, "runs", _run_id)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"  Run directory: {run_dir}")

    log = {
        "train_returns":   [],
        "critic_losses":   [],
        "actor_losses":    [],
        "alpha_values":    [],
        "proj_penalties":  [],   # ||a_raw - a_proj||^2; trends → 0 as actor learns feasibility
        "eval_returns":    [],   # list of dicts: {episode, eval_mean, eval_std, diagnostics…}
        "episodes":        [],
        "run_id":          _run_id,
        "hp":              hp,
        "selection_split": selection_label,
        "oracle_eval":     oracle_eval,
    }

    # Early-stopping state
    patience          = hp.get("early_stop_patience",  4)
    min_delta         = hp.get("early_stop_min_delta",  0.001)
    best_eval         = -np.inf   # all-time best eval_mean (gates checkpoint save)
    best_eval_patience = -np.inf  # best eval used for patience counting (requires min_delta)
    no_improve        = 0         # consecutive evals without min_delta improvement
    best_ep           = 0         # episode at which best_eval was achieved
    # Degradation guard: stop when Q-overfit drives eval below peak
    degrade_patience  = int(hp.get("early_stop_degrade_patience", 0))
    max_degradation   = float(hp.get("early_stop_max_degradation", np.inf))
    no_degrade        = 0         # consecutive evals more than max_degradation below best

    t_train_start = time.time()

    for ep in range(1, hp["train_episodes"] + 1):

        # 1. Collect one episode (N-step transitions stored in buffer)
        ep_return, _, n_collected = run_episode_collect(
            agent, env, buffer,
            n_step=hp.get("n_step", 1),
            gamma=hp["gamma"],
            terminal_soc_weight=float(hp.get("terminal_soc_weight", 0.0)),
        )

        # Anneal the PER importance-sampling exponent.
        buffer.anneal_beta(ep / hp["train_episodes"])

        # UTD=0.25: one gradient step every 4 env steps; matches 
        _min_fill = hp.get("min_buffer_fill", hp["batch_size"])
        _utd_denom = int(hp.get("utd_ratio_denominator", 1))
        c_losses, a_losses, alphas, proj_pens = [], [], [], []
        for _i in range(n_collected):
            if len(buffer) >= _min_fill and _i % _utd_denom == 0:
                m = agent.train_step(buffer)
                c_losses.append(m["critic_loss"])
                a_losses.append(m["actor_loss"])
                alphas.append(m["alpha"])
                proj_pens.append(m["proj_pen"])

        # 3. Log
        log["train_returns"].append(ep_return)
        log["critic_losses"].append(float(np.mean(c_losses))   if c_losses  else 0.0)
        log["actor_losses"].append(float(np.mean(a_losses))    if a_losses  else 0.0)
        log["alpha_values"].append(float(np.mean(alphas))      if alphas    else 1.0)
        log["proj_penalties"].append(float(np.mean(proj_pens)) if proj_pens else 0.0)
        log["episodes"].append(ep)

        # 4. Periodic evaluation + checkpoint
        if ep % hp["eval_freq"] == 0:
            eval_m = _run_evaluation(agent, env, starts=selection_starts)
            log["eval_returns"].append({"episode": ep, **eval_m})

            # Save the all-time best checkpoint separately from patience logic.
            if eval_m["eval_mean"] > best_eval:
                best_eval = eval_m["eval_mean"]
                best_ep   = ep
                best_path = os.path.join(checkpoint_dir, "best", "agent")
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                agent.save(best_path)
                tag = " (BEST)"
            else:
                tag = ""

            # Reset patience only on a material improvement.
            if eval_m["eval_mean"] > best_eval_patience + min_delta:
                best_eval_patience = eval_m["eval_mean"]
                no_improve = 0
            else:
                no_improve += 1
                tag += f" (no improve {no_improve}/{patience})"

            # Degradation guard: track consecutive evals below (best - max_degradation).
            if degrade_patience > 0:
                if eval_m["eval_mean"] < best_eval - max_degradation:
                    no_degrade += 1
                    tag += f" (degrade {no_degrade}/{degrade_patience})"
                else:
                    no_degrade = 0

            # Save the periodic checkpoint for later inspection.
            ckpt_path = os.path.join(checkpoint_dir, f"ep{ep:04d}", "agent")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            agent.save(ckpt_path)

            # Incremental log save.
            _log_path = os.path.join(run_dir, "training_log.npy")
            np.save(_log_path, log, allow_pickle=True)

            elapsed = time.time() - t_train_start
            _oracle_mean = oracle_eval["oracle_eval_mean"]
            _oracle_capture = (
                eval_m["eval_mean"] / _oracle_mean if _oracle_mean > 1e-8 else np.nan
            )
            print(
                f"  Ep {ep:4d}/{hp['train_episodes']} | "
                f"train_ret={ep_return:+.4f} | "
                f"{selection_tag}={eval_m['eval_mean']:+.4f}±{eval_m['eval_std']:.4f} | "
                f"oracle={_oracle_mean:+.4f} cap={_oracle_capture*100:.1f}% | "
                f"alpha={log['alpha_values'][-1]:.4f} | "
                f"critic={log['critic_losses'][-1]:.4f} actor={log['actor_losses'][-1]:.4f} | "
                f"SoC={eval_m['soc_end_mean']:.1f}/{eval_m['soc_min_mean']:.1f}MWh | "
                f"profit=${eval_m['profit_mean']:.0f} "
                f"(rev=${eval_m['revenue_mean']:.0f} deg=${eval_m['deg_cost_mean']:.0f}) | "
                f"dis={eval_m['frac_discharge']:.0%} ch={eval_m['frac_charge']:.0%} "
                f"idle={eval_m['frac_idle']:.0%} | "
                f"{elapsed:.0f}s{tag}"
            )

            # --- Stop if patience exceeded ---
            if no_improve >= patience:
                print(f"\n  Early stop at ep {ep}: no min_delta improvement on {selection_label} for "
                      f"{patience} evals. "
                      f"True best eval={best_eval:+.4f} at ep {best_ep}.")
                break

            # --- Stop if degradation threshold exceeded ---
            if degrade_patience > 0 and no_degrade >= degrade_patience:
                print(f"\n  Early stop at ep {ep}: eval degraded "
                      f"{best_eval - eval_m['eval_mean']:.4f} > {max_degradation:.4f} below best "
                      f"for {degrade_patience} consecutive evals. "
                      f"True best eval={best_eval:+.4f} at ep {best_ep}.")
                break

        elif ep % 100 == 0:
            elapsed = time.time() - t_train_start
            print(
                f"  Ep {ep:4d}/{hp['train_episodes']} | "
                f"train_ret={ep_return:+.4f} | "
                f"alpha={log['alpha_values'][-1]:.4f} | "
                f"{elapsed:.0f}s"
            )

    # ---- Save final training log ----
    log_path = os.path.join(run_dir, "training_log.npy")
    np.save(log_path, log, allow_pickle=True)
    print(f"\nTraining log saved -> {log_path}")

    return agent, log


# ================================================================
# RESULTS SUMMARY
# ================================================================

def print_results_summary(
    log:   dict,
    agent: SACAgent,
    env:   BESSEnv,
    hp:    dict,
) -> None:
    """Print a compact training and evaluation summary."""
    print("\n" + "=" * 65)
    print("NC-SafeRL  STEP 4  RESULTS SUMMARY")
    print("=" * 65)

    train_returns = np.array(log["train_returns"])
    n_ep = len(train_returns)
    print(f"\nTraining ({n_ep} episodes run):")
    print(f"  All-episode mean return : {np.mean(train_returns):+.4f}")
    print(f"  Final 100 ep mean       : {np.mean(train_returns[-100:]):+.4f}")
    print(f"  Best ep return          : {np.max(train_returns):+.4f}")
    print(f"  Final alpha             : {log['alpha_values'][-1]:.4f}")
    print(f"  Final critic loss       : {log['critic_losses'][-1]:.5f}")

    if log["eval_returns"]:
        selection_label = log.get("selection_split", "test")
        eval_means = [e["eval_mean"] for e in log["eval_returns"]]
        best_idx   = int(np.argmax(eval_means))
        n_eval     = log["eval_returns"][0].get("n_eval", "?")
        print(f"\nSelection evaluation (every {hp['eval_freq']} eps, all {n_eval} {selection_label} starts):")
        print(f"  Best {selection_label} mean  : {eval_means[best_idx]:+.4f}  "
              f"(ep {log['eval_returns'][best_idx]['episode']})")
        print(f"  Final {selection_label} mean : {eval_means[-1]:+.4f}")
        print(f"  Final {selection_label} std  : {log['eval_returns'][-1]['eval_std']:.4f}")
        print(f"  Final {selection_label} min/max : "
              f"{log['eval_returns'][-1]['eval_min']:+.4f} / "
              f"{log['eval_returns'][-1]['eval_max']:+.4f}")
        oracle_eval = log.get("oracle_eval") or {}
        oracle_mean = oracle_eval.get("oracle_eval_mean")
        if oracle_mean is not None:
            best_capture = eval_means[best_idx] / oracle_mean if oracle_mean > 1e-8 else np.nan
            print(f"  Oracle {selection_label} mean: {oracle_mean:+.4f}")
            print(f"  Best oracle capture     : {best_capture*100:.1f}%")

    print("=" * 65)


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    """Instantiate the environment, train SAC, and export the best checkpoint."""
    # Ensure outputs directory exists
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("\n" + "=" * 65)
    print("NC-SafeRL  STEP 4: SAC Agent Training")
    print("=" * 65)

    # ---- Create environment ----
    env = BESSEnv(
        seed=HP["seed"],
        validation_days=HP.get("validation_days", 30),
        randomize_train_soc=bool(HP.get("randomize_train_soc", False)),
    )
    _device_tag = "Metal GPU" if tf.config.get_visible_devices("GPU") else "CPU"
    print(f"RTS-GMLC 73-bus | BESS bus {env.bess_bus_rts} | TF 2.x {_device_tag}")

    # ---- E0: PTDF binding diagnostic ----
    ptdf_binding_analysis(env, save_dir=OUTPUTS_DIR)

    # ---- Train ----
    # Use deterministic validation (or test fallback) evaluation, not noisy train returns, for model selection.
    t0 = time.time()
    agent, log = train_sac(env, HP)
    total_time = time.time() - t0

    print(f"\nTotal wall-clock time: {total_time/60:.1f} min")

    # ---- Results summary ----
    print_results_summary(log, agent, env, HP)

    # Reload the best checkpoint before exporting the final weights.
    _run_dir    = os.path.join(OUTPUTS_DIR, "runs", log["run_id"])
    best_prefix = os.path.join(_run_dir, "checkpoints", "best", "agent")
    best_actor_w = best_prefix + "_actor.weights.h5"
    best_critic_w = best_prefix + "_critic.weights.h5"
    best_alpha_w = best_prefix + "_log_alpha.npy"
    if (os.path.exists(best_actor_w)
            and os.path.exists(best_critic_w)
            and os.path.exists(best_alpha_w)):
        agent.load(best_prefix)
        print(f"\nLoaded best checkpoint -> {best_prefix}")
    else:
        print("\nBest checkpoint not found; exporting current agent state.")

    # ---- Save final agent ----
    final_path = os.path.join(OUTPUTS_DIR, "step4_final", "agent")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    agent.save(final_path)
    print(f"\nFinal agent saved -> {final_path}")

    final_test = _run_evaluation(agent, env, starts=env.test_starts)
    print("\nFinal untouched test evaluation:")
    print(f"  Test mean return       : {final_test['eval_mean']:+.4f}")
    print(f"  Test std               : {final_test['eval_std']:.4f}")
    print(f"  Test min/max           : {final_test['eval_min']:+.4f} / {final_test['eval_max']:+.4f}")
    print(f"  Profit mean            : ${final_test['profit_mean']:.0f}")
    print(f"  Revenue / deg cost     : ${final_test['revenue_mean']:.0f} / ${final_test['deg_cost_mean']:.0f}")
    print(f"  SoC end / min mean     : {final_test['soc_end_mean']:.1f} / {final_test['soc_min_mean']:.1f} MWh")
    print(f"  Action mix             : dis={final_test['frac_discharge']:.0%} "
          f"ch={final_test['frac_charge']:.0%} idle={final_test['frac_idle']:.0%}")
    print("\nStep 4 complete.  Ready for Step 5 (analysis / paper figures).")


if __name__ == "__main__":
    main()
