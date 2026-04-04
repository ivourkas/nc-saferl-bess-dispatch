"""
Step 4: SAC Agent with Expert Warm-Start for NC-SafeRL BESS Dispatch
=====================================================================
Author: Yiannis Vourkas

Implements Soft Actor-Critic (SAC-v2, Haarnoja et al. 2019,
arXiv:1812.05905v2) for the BESS economic dispatch environment built
in Step 3.

NC-SafeRL NOVELTY
-----------------
This is the only BESS RL paper that embeds a PTDF-based network safety layer
inside the Gymnasium environment (Step 3). The safety layer performs an exact
1-D QP projection at every step, enforcing:
  (a) SoC feasibility bounds
  (b) DC line-flow limits via PTDF sensitivity for all 120 branches
Safety is guaranteed by construction -- the agent never sees a constraint
violation, yet the RL training operates on the *projected* feasible action.

ARCHITECTURE
------------
  ReplayBuffer          Circular numpy buffer (200 k transitions, ~25 MB RAM)
  SquashedGaussianActor MLP 36 -> [256, 256] -> (mu, log_std), tanh squash
  TwinQCritic           Two Q-networks in one model, min(Q1, Q2) for targets
  SACAgent              Wires all pieces; @tf.function for all gradient steps
  PerfectForesightLP    scipy linprog oracle (no network constraints) for benchmark

TRAINING PROTOCOL
-----------------
  SAC training  : up to 5 000 episodes × 168 gradient updates/episode
  Eval          : every 100 episodes on ALL ~95 test starts (deterministic)
  Checkpoint    : every 100 episodes → outputs/step4_checkpoints/
  Early stop    : patience=12 evals, min_delta=0.001

HYPERPARAMETERS (HP dict)
-------------------------
  gamma = 0.99   tau = 0.005   lr = 3e-4   alpha_lr = 1e-4
  batch_size = 256   buffer_size = 200 000   n_step = 8
  target_entropy_scale = -0.75   init_alpha = 0.03   alpha_min = 0.0005
  proj_lambda = 0.01   lp_warmstart_steps = 10 000   warmstart_bc_steps = 750

OUTPUTS
-------
  outputs/step4_training_log.npy      dict with training metrics
  outputs/step4_checkpoints/ep<N>/    Keras weight files + log_alpha

FRAMEWORK
---------
  TensorFlow 2.16.2 + tensorflow-metal 1.2.0 (Apple Silicon GPU via Metal API).
  Falls back to CPU automatically if Metal plugin is not installed.
  3.BESSEnvironment imported via importlib (filename starts with digit).
"""

# ================================================================
# IMPORTS
# ================================================================

import os
import sys
import math
import time
import importlib.util
from collections import deque   # used by run_episode_collect for N-step window
import numpy as np

# -- TensorFlow (Apple Silicon / Metal GPU) --
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"          # suppress TF INFO logs
import tensorflow as tf

# Verify Metal GPU is available; fall back to CPU gracefully
_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    print(f"[TF] Metal GPU detected: {[g.name for g in _gpus]}")
else:
    print("[TF] No GPU found — running on CPU")

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
    "tau":            0.005,      # target network soft-update rate
    "lr":             3e-4,       # learning rate (actor, critic)
    "alpha_lr":       1e-4,       # slower alpha updates: avoids early collapse
    "hidden_dim":     256,        # hidden layer width
    "hidden_layers":  2,          # number of hidden layers    
    "target_entropy_scale": -8.0,    # Calibrated to actual achievable entropy.
                                     # After BC warmstart saturates tanh at ±1, log_pi ≈ +4 to +8.
                                     # Target must be BELOW achievable entropy so alpha decays.
                                     # With PTDF constraints binding 86% of hours (bang-bang policy),
                                     # steady-state policy entropy ≈ -5 to -10.
                                     # -8.0 allows alpha to decay from init while preserving
                                     # enough entropy incentive to prevent premature convergence.
    # Buffer & batch
    "buffer_size":    200_000,    
    "batch_size":     256,
    # Minimum buffer fill before ANY gradient update begins.
    # Starting at batch_size=256 (≈1 episode) caused the critic to overfit
    # catastrophically to near-identical early transitions.
    # 2000 transitions ≈ 12 random episodes,
    # providing enough diversity for stable initial TD targets.
    "min_buffer_fill": 2_000,
    # LP Expert Warm-Start (fills buffer before SAC training begins).
    # 10 000 transitions = 5% of total buffer capacity (200 k).
    
    "lp_warmstart_steps": 10_000,
    # Optional actor behavior-cloning pretrain on LP warm-start transitions.
    # Warm-start transitions directly supervise the policy before on-policy SAC
    # updates begin, making the expert data useful for BOTH critic and actor.
    "warmstart_bc_steps": 750,
    "warmstart_bc_batch_size": 256,
    
    # Anticipatory projection penalty: pushes the actor toward naturally feasible
    # outputs so the Q-gradient (zero when clipped) is supplemented by a direct
    # feasibility gradient at the ~86% of steps where PTDF constraints bind.
    "proj_lambda":    0.01,       # non-zero restores gradient under tight PTDF constraints
    # N-step return horizon.
    # Buffer stores G_t^N = sum_{k=0}^{N-1} gamma^k * r_{t+k}; critic
    # bootstraps with gamma^N. N=8 spans one-third of a daily price cycle,
    # directly bridging the credit-assignment gap between charging (negative
    # immediate reward) and profitable discharge 4-12 h later.
    "n_step":         8,
    # Initial entropy coefficient alpha.
    # SAC default log_alpha=0.0 → alpha=1.0. With REWARD_SCALE=1e-3 our
    # per-step rewards are O(0.001), so alpha=1.0 overwhelms the Q-signal
    # by ~500× in early training, forcing pure-entropy optimisation for
    # 100+ episodes before alpha decays. alpha=0.01 starts the entropy
    # term at the same order of magnitude as per-step rewards from ep 1.
    # Slightly higher start value to preserve exploration while the critic is
    # still forming its first profitable arbitrage estimates.
    "init_alpha":     0.03,
    # Hard lower bound on alpha (prevents critic divergence from alpha → 0).
    # Without this, auto-alpha can drive log_alpha → -∞ when the policy
    # over-concentrates (e.g. after policy collapses onto a local optimum).
   
    "alpha_min":      0.0005,
    # Hard upper bound on alpha (prevents runaway when log_pi > -target_entropy,
    # which occurs after BC warmstart pushes policy to saturate tanh at ±1).
    "alpha_max":      0.05,   # was 1.0: with reward_scale=0.001, alpha=1.0 is 20× too large.
                              # Per-step rewards O(0.001); alpha × log_pi must stay same order.
                              # 0.05 × log_pi(~5) = 0.25 >> per-step reward O(0.001-0.1) ✓
    # Training schedule
    "train_episodes": 5_000,      # SAC episodes (upper bound; early stopping
                                  # will halt before this if policy converges)
    "eval_freq":      100,       
    

    "early_stop_patience":  12,   
    
    "early_stop_min_delta": 0.001,# min improvement to reset patience 
    # Reproducibility
    "seed":           42,
}

# Actor log-std clipping
LOG_STD_MIN = -20.0
LOG_STD_MAX =  2.0


# ================================================================
# SUM TREE (for Prioritized Experience Replay)
# ================================================================

class SumTree:
    """
    Binary sum-tree for O(log N) priority-based sampling.

    Internal layout
    ---------------
    The tree has 2*capacity nodes stored in a flat array.  Leaves occupy
    indices [capacity, 2*capacity).  Internal node k stores the sum of its
    two children: tree[k] = tree[2k] + tree[2k+1].  The root tree[1] holds
    the total priority sum used to draw stratified samples.

    Operations
    ----------
    update(idx, p)   O(log N) — set leaf idx = p and propagate sums upward.
    retrieve(value)  O(log N) — return leaf whose cumulative sum spans value.
    total            O(1)     — root value = sum of all priorities.

    Reference
    ---------
    Schaul et al. (2016) "Prioritized Experience Replay," ICLR 2016,
    arXiv:1511.05952, §3.3 (sum-tree implementation).
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
        """
        Walk the tree top-down to find the leaf whose prefix sum spans value.

        Returns the data index in [0, capacity).
        value must satisfy 0 <= value < self.total.
        """
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
    Circular numpy replay buffer for off-policy experience with Prioritized
    Experience Replay (PER).

    Memory layout (200 k transitions, obs_dim=36, act_dim=1):
      obs        : (200k, 36) float32  ~28.8 MB
      next_obs   : (200k, 36) float32  ~28.8 MB
      act        : (200k,  1) float32   ~0.8 MB   <- projected action (a_safe)
      rew        : (200k,  1) float32   ~0.8 MB
      done       : (200k,  1) float32   ~0.8 MB
      bounds     : (200k,  2) float32   ~1.6 MB   <- [p_lo, p_hi] at s_t
      next_bounds: (200k,  2) float32   ~1.6 MB   <- [p_lo, p_hi] at s_{t+N}
      SumTree    : (400k,)   float64   ~3.2 MB
    Total: ~63 MB

    bounds / next_bounds semantics (Convention B)
    ---------------------------------------------
    bounds stores the safety-layer feasibility interval [p_lo, p_hi]
    (normalised to [-1,1]) for the CURRENT state s_t.  The actor update
    projects a_raw → a_proj before Q-evaluation.

    next_bounds stores the same interval for the NEXT (bootstrap) state s_{t+N}.
    Used in the critic target to project the sampled next action, ensuring
    Convention B consistency:
      Q̂(s_t,a_t) = G_t^N + γ^N*(1-done)*[min_Q(s_{t+N}, clip(a',next_bounds))
                                            - α*log_π(a'|s_{t+N})]
    Without this, the critic target would be evaluated at a raw (infeasible)
    action ~83% of the time, causing off-distribution Q-overestimation.

    Prioritized Experience Replay (PER)
    ------------------------------------
    New transitions are assigned max_priority (initially 1.0), ensuring they
    are sampled at least once before being deprioritised.  After each gradient
    step, the caller supplies per-sample TD errors; priorities are updated as:

        p_i = (|δ_i| + ε)^α_per

    Sampling probability: P(i) = p_i / Σ_j p_j   (via SumTree)

    Importance-sampling correction weight (bias correction for non-uniform
    sampling; see Schaul et al. 2016 eq 1–2):

        w_i = (1/N · 1/P(i))^β,   normalised by max_j(w_j)

    β is annealed from β_init=0.4 → 1.0 over training, removing IS bias
    asymptotically (full correction at convergence).

    Hyper-parameters (Schaul et al. 2016, Table 1):
      per_alpha    = 0.6   prioritisation strength (0 = uniform)
      per_beta_init= 0.4   IS correction exponent start value
      per_eps      = 1e-6  priority floor (prevents zero sampling probability)

    Reference: Schaul et al. (2016) "Prioritized Experience Replay,"
    ICLR 2016, arXiv:1511.05952.
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
        """Store one transition.

        p_lo_norm / p_hi_norm      : safety bounds at s_t (current state).
        next_p_lo_norm / next_p_hi_norm : safety bounds at s_{t+N} (bootstrap
            state).  Used in the critic target to project the sampled next
            action before querying the target Q-network (Convention B).
        All bounds normalised to [-1, 1] via p_mw / P_MAX."""
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
        # Assign max priority to new transition so it is sampled at least once
        # before its TD error is known (Schaul et al. 2016 §3.3).
        self._sum_tree.update(idx, self._max_priority)
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_episode(self, transitions: list) -> None:
        """Bulk-add a list of transitions.  Accepted tuple lengths:
        9: (obs, act, rew, next_obs, done, p_lo, p_hi, next_p_lo, next_p_hi)
        7: (obs, act, rew, next_obs, done, p_lo, p_hi)
        5: (obs, act, rew, next_obs, done)"""
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
        Priority-weighted stratified sampling with IS correction.

        Stratified sampling (Schaul et al. 2016 §3.3):
          Divide [0, total) into batch_size equal segments; draw one sample
          uniformly from each segment.  This reduces variance vs i.i.d. sampling
          while preserving the priority-proportional distribution.

        Returns 9-tuple:
          (obs, act, rew, next_obs, done, bounds, next_bounds,
           sample_indices, is_weights)

          sample_indices : np.ndarray int64, shape (batch,)
              Buffer indices of sampled transitions.  Pass to update_priorities()
              after computing per-sample TD errors in train_step().
          is_weights     : np.ndarray float32, shape (batch, 1)
              Importance-sampling correction weights w_i = (N·P(i))^{-β},
              normalised by max_i(w_i).  Multiply elementwise into critic loss
              to correct the sampling bias introduced by prioritisation.

        Reference: Schaul et al. (2016) arXiv:1511.05952, eqs 1–2.
        """
        total   = self._sum_tree.total
        segment = total / batch_size

        indices = np.empty(batch_size, dtype=np.int64)
        for i in range(batch_size):
            mass       = np.random.uniform(segment * i, segment * (i + 1))
            # Guard against floating-point overshoot at the right boundary
            mass       = min(mass, total - 1e-9)
            indices[i] = self._sum_tree.retrieve(mass)

        # Sampling probabilities P(i) = p_i / total  (Schaul et al. eq 1)
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
        """
        Update transition priorities from per-sample TD errors.

        Called after each train_step() with the critic TD residuals
        returned by _update_critic().

        Priority formula (Schaul et al. 2016 eq 1):
          p_i = (|δ_i| + ε)^α

        max_priority is tracked so that newly added transitions (which
        start at max_priority) are always at least as likely to be sampled
        as the most recently updated transition.

        Parameters
        ----------
        indices   : int64 array, shape (batch,) — from sample() return
        td_errors : float32 array, shape (batch,) — |Q - y| per sample
        """
        priorities = (np.abs(td_errors) + self.per_eps) ** self.per_alpha
        for idx, p in zip(indices, priorities):
            self._sum_tree.update(int(idx), float(p))
        self._max_priority = max(self._max_priority, float(priorities.max()))

    def anneal_beta(self, fraction: float) -> None:
        """
        Linearly anneal the IS exponent β from β_init → 1.0.

        At fraction=0 (start of training), β = β_init = 0.4 → partial IS
        correction (biased but lower variance, aids early learning).
        At fraction=1 (end of training), β = 1.0 → full unbiased IS
        correction (Schaul et al. 2016 §3.4).

        Call once per episode: buffer.anneal_beta(ep / total_episodes).
        """
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
    Stochastic actor with structured state encoder + tanh-squashed Gaussian policy.

    State is split into two streams with different inductive biases:

      LMP stream  obs[1:26]  (25-dim: current LMP + 24h history)
        → Conv1D(32 filters, kernel=4, stride=1, relu)
        → GlobalAveragePooling1D
        → Dense(32, relu)
        → 32-dim temporal embedding

      Scalar stream  obs[0] + obs[26:36]  (11-dim: SoC, PV, step, segs×4, p_lo, p_hi, DoW, load)
        → Dense(32, relu)
        → 32-dim scalar embedding

      Concat (64-dim) → Dense(256, relu) → Dense(256, relu) → (mu, log_std)

    Rationale:
      - Conv1D on the LMP stream learns local temporal patterns (e.g., "prices
        rose for 4 consecutive hours → discharge now") without storing an LSTM
        hidden state, keeping training stable and avoiding truncated-BPTT issues.
      - GlobalAveragePooling pools over the 22-step convolution output, giving
        translation-invariant pattern detection across the 24h window.
      - Scalar stream encodes physical state and safety bounds via a small MLP;
        no temporal structure is needed here.
      - Joint trunk (256 × 2) fuses both embeddings before computing action.

    Safety projection is the caller's responsibility (Convention B).
    """

    # State index constants (must match BESSEnvironment._get_obs layout)
    _LMP_START  = 1    # obs[1..25]: current LMP + 24h history  (25 values)
    _LMP_END    = 26
    _SCALAR_IDX = [0] + list(range(26, 36))   # SoC + PV + step + segs4 + p_lo/hi + DoW + load

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()

        # --- LMP temporal stream ---
        self.lmp_conv   = tf.keras.layers.Conv1D(32, kernel_size=4, strides=1,
                                                  padding="causal", activation="relu")
        self.lmp_pool   = tf.keras.layers.GlobalAveragePooling1D()
        self.lmp_dense  = tf.keras.layers.Dense(32, activation="relu")

        # --- Scalar physical stream ---
        self.scalar_dense = tf.keras.layers.Dense(32, activation="relu")

        # --- Joint trunk ---
        self.trunk1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.trunk2 = tf.keras.layers.Dense(hidden_dim, activation="relu")

        # --- Output heads ---
        self.mu_layer      = tf.keras.layers.Dense(act_dim)
        self.log_std_layer = tf.keras.layers.Dense(act_dim)

    def _encode(self, obs: tf.Tensor) -> tf.Tensor:
        """Encode obs into a joint embedding via dual-stream architecture."""
        # LMP stream: shape (batch, 25) → (batch, 25, 1) for Conv1D
        lmp   = tf.expand_dims(obs[:, self._LMP_START:self._LMP_END], axis=-1)
        lmp_e = self.lmp_dense(self.lmp_pool(self.lmp_conv(lmp)))   # (batch, 32)

        # Scalar stream
        scalar_idx = tf.constant(self._SCALAR_IDX, dtype=tf.int32)
        scalar     = tf.gather(obs, scalar_idx, axis=1)              # (batch, 11)
        scalar_e   = self.scalar_dense(scalar)                       # (batch, 32)

        # Joint trunk
        joint = tf.concat([lmp_e, scalar_e], axis=-1)                # (batch, 64)
        return self.trunk2(self.trunk1(joint))                        # (batch, 256)

    def _distribution_params(self, obs: tf.Tensor):
        """Return Gaussian parameters for the latent normalized action."""
        h = self._encode(obs)
        mu      = self.mu_layer(h)
        log_std = tf.clip_by_value(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def call(self, obs):
        """
        Stochastic forward pass (used during training).

        Parameters
        ----------
        obs : tf.Tensor, shape (batch, obs_dim)

        Returns
        -------
        action : tf.Tensor, shape (batch, act_dim)  in (-1, 1)  — raw tanh output
        log_pi : tf.Tensor, shape (batch, 1)  — log-prob under the squashed Gaussian
        """
        mu, log_std = self._distribution_params(obs)
        std = tf.exp(log_std)

        # Reparameterization trick
        eps = tf.random.normal(tf.shape(mu))
        z = mu + std * eps

        # Log probability under Gaussian (pre-squashing)
        log_pi_normal = (
            -0.5 * (eps ** 2 + tf.math.log(2.0 * math.pi))
            - log_std
        )
        log_pi_normal = tf.reduce_sum(log_pi_normal, axis=-1, keepdims=True)

        # Tanh squashing correction: sum_i log(1 - tanh(z_i)^2)
        squash_correction = tf.reduce_sum(
            tf.math.log(1.0 - tf.tanh(z) ** 2 + 1e-6),
            axis=-1, keepdims=True,
        )

        log_pi = log_pi_normal - squash_correction
        action = tf.tanh(z)   # raw action in (-1, 1); projection applied in actor-update step
        return action, log_pi

    def get_deterministic_action(self, obs):
        """
        Deterministic forward pass. Returns tanh(mu) in (-1, 1).
        Used during evaluation (no exploration noise).
        Projection to the feasible interval is the caller's responsibility.
        """
        mu, _ = self._distribution_params(obs)
        return tf.tanh(mu)


class TwinQCritic(tf.keras.Model):
    """
    Twin Q-networks in a single keras.Model (Q1 and Q2 share no weights).

    Architecture for each Qi uses the same dual-stream encoder as the actor,
    with action appended to the scalar stream (not to raw obs) so the critic
    sees the action in the same semantic space as the physical state variables:

      LMP stream  obs[1:26]  → Conv1D(32) → GAP → Dense(32) → 32-dim
      Scalar+act  [obs_scalars, act]  → Dense(32) → 32-dim
      Concat (64) → Dense(256, relu) → Dense(256, relu) → Dense(1)

    Using min(Q1, Q2) for target value reduces overestimation bias (Fujimoto 2018).
    """

    _LMP_START  = 1
    _LMP_END    = 26
    _SCALAR_IDX = [0] + list(range(26, 36))

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Q1 dual-stream encoder
        self.q1_lmp_conv   = tf.keras.layers.Conv1D(32, kernel_size=4, strides=1,
                                                     padding="causal", activation="relu")
        self.q1_lmp_pool   = tf.keras.layers.GlobalAveragePooling1D()
        self.q1_lmp_dense  = tf.keras.layers.Dense(32, activation="relu")
        self.q1_scalar     = tf.keras.layers.Dense(32, activation="relu")
        self.q1_fc1        = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.q1_fc2        = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.q1_out        = tf.keras.layers.Dense(1)

        # Q2 dual-stream encoder (identical architecture, separate weights)
        self.q2_lmp_conv   = tf.keras.layers.Conv1D(32, kernel_size=4, strides=1,
                                                     padding="causal", activation="relu")
        self.q2_lmp_pool   = tf.keras.layers.GlobalAveragePooling1D()
        self.q2_lmp_dense  = tf.keras.layers.Dense(32, activation="relu")
        self.q2_scalar     = tf.keras.layers.Dense(32, activation="relu")
        self.q2_fc1        = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.q2_fc2        = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.q2_out        = tf.keras.layers.Dense(1)

    def _encode_one(self, obs, act, lmp_conv, lmp_pool, lmp_dense, scalar_dense, fc1, fc2, out):
        """Shared encoding logic for one Q-network."""
        lmp    = tf.expand_dims(obs[:, self._LMP_START:self._LMP_END], axis=-1)
        lmp_e  = lmp_dense(lmp_pool(lmp_conv(lmp)))

        scalar_idx = tf.constant(self._SCALAR_IDX, dtype=tf.int32)
        scalar     = tf.gather(obs, scalar_idx, axis=1)
        scalar_e   = scalar_dense(tf.concat([scalar, act], axis=-1))   # action in scalar stream

        joint = tf.concat([lmp_e, scalar_e], axis=-1)
        return out(fc2(fc1(joint)))

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : list [obs, act]
            obs : tf.Tensor, shape (batch, obs_dim)
            act : tf.Tensor, shape (batch, act_dim)

        Returns
        -------
        q1, q2 : tf.Tensor, shape (batch, 1) each
        """
        obs, act = inputs
        q1 = self._encode_one(obs, act,
                               self.q1_lmp_conv, self.q1_lmp_pool, self.q1_lmp_dense,
                               self.q1_scalar, self.q1_fc1, self.q1_fc2, self.q1_out)
        q2 = self._encode_one(obs, act,
                               self.q2_lmp_conv, self.q2_lmp_pool, self.q2_lmp_dense,
                               self.q2_scalar, self.q2_fc1, self.q2_fc2, self.q2_out)
        return q1, q2


# ================================================================
# SAC AGENT
# ================================================================

class SACAgent:
    """
    Soft Actor-Critic agent (Haarnoja et al., SAC-v2, 2018).

    Key features:
      - Twin Q-critics with soft target updates (Polyak averaging)
      - Automatic entropy tuning via trainable log_alpha
      - All three gradient steps decorated with @tf.function for speed

    Parameters
    ----------
    obs_dim : int   -- observation space dimensionality (36)
    act_dim : int   -- action space dimensionality (1)
    hp      : dict  -- hyperparameter dict (see HP at top of file)
    """

    def __init__(self, obs_dim: int, act_dim: int, hp: dict):
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim
        self.gamma      = hp["gamma"]
        self.tau        = hp["tau"]
        self.batch_size = hp["batch_size"]

        hidden_dim = hp["hidden_dim"]

        # --- Networks ---
        self.actor         = SquashedGaussianActor(obs_dim, act_dim, hidden_dim)
        self.critic        = TwinQCritic(obs_dim, act_dim, hidden_dim)
        self.critic_target = TwinQCritic(obs_dim, act_dim, hidden_dim)

        # Build weights by running a dummy forward pass
        _dummy_obs = tf.zeros((1, obs_dim), dtype=tf.float32)
        _dummy_act = tf.zeros((1, act_dim), dtype=tf.float32)
        self.actor(_dummy_obs)
        self.critic([_dummy_obs, _dummy_act])
        self.critic_target([_dummy_obs, _dummy_act])

        # Initialise target == online critic
        self.critic_target.set_weights(self.critic.get_weights())

        # --- Optimizers (same lr for all three, gradient clipping) ---
        # global_clipnorm=1.0 clips ALL parameter gradients jointly so their
        # combined L2-norm is ≤1.0. 
        lr = hp["lr"]
        alpha_lr = float(hp.get("alpha_lr", lr))
        self.actor_opt  = tf.keras.optimizers.Adam(lr, global_clipnorm=1.0)
        self.critic_opt = tf.keras.optimizers.Adam(lr, global_clipnorm=1.0)
        self.alpha_opt  = tf.keras.optimizers.Adam(alpha_lr, global_clipnorm=1.0)

        # --- Entropy coefficient (automatic tuning) ---
        # log_alpha is the trainable scalar; alpha = exp(log_alpha).
        # We initialise to hp['init_alpha'] (default 0.01) instead of the
        # SAC-paper default of 1.0.  With REWARD_SCALE=1e-3, per-step rewards
        # are O(0.001); alpha=1.0 overwhelms the Q-signal by ~500× for the
        # first 100 episodes, causing the actor to optimise pure entropy and
        # the alpha decay mechanism to crash alpha to ~0.0003 before the critic
        # converges.  
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
        # Hard upper bound: alpha never exceeds alpha_max.
        # Without this, BC warmstart (bang-bang LP oracle saturates tanh → log_pi > 0)
        # causes alpha to explode exponentially, drowning the Q-signal.
        _alpha_max = float(hp.get("alpha_max", 1.0))
        self._log_alpha_max = tf.constant(math.log(_alpha_max), dtype=tf.float32)

        # Keras 3 tracks variables per optimizer on first apply_gradients().
        # BC pretrain may touch only a subset of actor params (mu head), so
        # pre-build optimizers with the FULL variable sets to avoid
        # "Unknown variable" errors later in SAC updates.
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

        # --- N-step return parameters ---
        # n_step: how many steps of actual rewards are summed before
        # bootstrapping. The buffer stores the pre-computed N-step sum
        # G_t^N, so the critic target uses gamma^N (not gamma) as the
        # discount on the bootstrap value V(s_{t+N}).
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
        next_bounds: tf.Tensor,   # (batch, 2)  safety bounds at s_{t+N}
        is_weights:  tf.Tensor,   # (batch, 1)  PER importance-sampling weights
    ):
        """
        Compute and apply IS-weighted critic gradients (PER-aware).

        N-step target (n_step=1 reduces to standard 1-step SAC):
          y = G_t^N + gamma^N*(1-done_N)*[min(Q1',Q2')(s_{t+N}, a'_proj)
                                           - alpha*log_pi(a'|s_{t+N})]
        where a'_proj = clip(a', next_p_lo, next_p_hi) enforces Convention B:
        the critic is trained on projected actions and must also be evaluated
        at projected actions in the target to avoid off-distribution Q-bias.
        next_bounds (stored in the replay buffer via the N+1 look-ahead window)
        provides the exact safety bounds at s_{t+N}.

        IS-weighted loss (Schaul et al. 2016, eq 9):
          JQ = 0.5 * mean[ w_i * ((Q1_i - y_i)^2 + (Q2_i - y_i)^2) ]
        When all w_i = 1 (uniform sampling) this reduces exactly to the
        standard SAC critic loss (SAC-v2 paper eq 5): 0.5*mean[(Q1-y)^2+(Q2-y)^2].

        Returns
        -------
        critic_loss : scalar Tensor — IS-weighted mean squared TD error
        td_errors   : Tensor shape (batch,) — per-sample |δ_i| for priority update
                      δ_i = 0.5*(|Q1_i - y_i| + |Q2_i - y_i|)
                      Computed from Q values BEFORE the weight update (standard PER).
        """
        # Sample next action from the stochastic policy.
        next_act, next_log_pi = self.actor(next_obs)

        # Project next action to next-state safety bounds (Convention B fix)
        next_p_lo = next_bounds[:, :1]   # (batch, 1)
        next_p_hi = next_bounds[:, 1:]   # (batch, 1)
        next_act_proj = tf.clip_by_value(next_act, next_p_lo, next_p_hi)

        # Clamp log_pi to prevent critic target corruption when tanh saturates.
        # After BC warmstart, log_pi can be +5 to +8 (squash correction artifact).
        # alpha * log_pi in the target would then be O(1000s) vs Q O(0.001).
        next_log_pi = tf.clip_by_value(next_log_pi, -20.0, 2.0)

        # N-step target Q values (stop_gradient: treated as a constant label)
        q1_next, q2_next = self.critic_target([next_obs, next_act_proj])
        min_q_next = tf.minimum(q1_next, q2_next)
        q_target = tf.stop_gradient(
            rew + self.gamma_n * (1.0 - done) * (min_q_next - self.alpha * next_log_pi)
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

        # Per-sample TD errors for priority update (Schaul et al. 2016 eq 1).
        # Averaged over both critics; computed from Q values BEFORE the step.
        td_errors = tf.squeeze(
            0.5 * (tf.abs(q1 - q_target) + tf.abs(q2 - q_target)),
            axis=1,
        )   # shape (batch,)
        return critic_loss, td_errors

    @tf.function
    def _update_actor(self, obs: tf.Tensor, bounds: tf.Tensor) -> tf.Tensor:
        """
        Compute and apply actor gradients.

        Convention B (differentiable projection, no CVXPYLayers needed):
          1. Actor proposes a_raw = tanh(z) in (-1, 1).
          2. Clip to stored safety bounds [p_lo, p_hi] -> a_proj.
             tf.clip_by_value: grad = 1 when in bounds, 0 when clipped.
             For this 1-D linear QP this IS the exact KKT gradient.
          3. Q is evaluated at a_proj (consistent with critic training).
          4. Anticipatory penalty ||a_raw - a_proj||^2 restores non-zero
             gradient at the ~86% of steps where PTDF binding makes the
             clip active (where Q-gradient alone would be zero).

        Loss = E[alpha * log_pi(a_raw|s)           (entropy on raw dist.)
                - min(Q1,Q2)(s, a_proj)             (value at projected act.)
                + proj_lambda * ||a_raw-a_proj||^2] (anticipatory penalty)
        """
        p_lo = bounds[:, :1]   # (batch, 1)  normalised lower bound
        p_hi = bounds[:, 1:]   # (batch, 1)  normalised upper bound

        with tf.GradientTape() as tape:
            act, log_pi = self.actor(obs)                              # a_raw in (-1, 1)
            act_proj = tf.clip_by_value(act, p_lo, p_hi)              # project to feasible interval
            q1, q2 = self.critic([obs, act_proj])                     # Convention B: Q at projected action
            min_q   = tf.minimum(q1, q2)
            proj_pen = tf.reduce_mean(tf.square(act - act_proj))      # should stay ~0
            actor_loss = (tf.reduce_mean(self.alpha * log_pi - min_q)
                          + self.proj_lambda * proj_pen)

        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(
            zip(grads, self.actor.trainable_variables)
        )
        return actor_loss, proj_pen

    @tf.function
    def _bc_actor_step(self, obs: tf.Tensor, act_expert: tf.Tensor) -> tf.Tensor:
        """
        One behavior-cloning gradient step on expert actions.

        Uses deterministic feasible policy output and MSE loss:
          L_BC = E[ ||a_det(s) - a_expert||^2 ].
        """
        with tf.GradientTape() as tape:
            act_det = self.actor.get_deterministic_action(obs)
            bc_loss = tf.reduce_mean(tf.square(act_det - act_expert))

        grads = tape.gradient(bc_loss, self.actor.trainable_variables)
        grads_and_vars = [
            (g, v)
            for g, v in zip(grads, self.actor.trainable_variables)
            if g is not None
        ]
        self.actor_opt.apply_gradients(grads_and_vars)
        return bc_loss

    @tf.function
    def _update_alpha(self, obs: tf.Tensor) -> tf.Tensor:
        """
        Update log_alpha to match target entropy.

        Practical log_alpha surrogate for the SAC temperature update.

        The sign matches the Eq. 18 dual descent direction, while avoiding
        the vanishing-gradient effect of alpha * (...) once alpha becomes
        very small.

        After the gradient step a hard lower bound is enforced:
          log_alpha >= log(alpha_min)
        This prevents alpha from collapsing to zero when the policy is
        concentrated near tanh boundaries (e.g. post LP warm-start), which
        would eliminate entropy regularisation and destabilise the critic.
        """
        with tf.GradientTape() as tape:
            _, log_pi = self.actor(obs)
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
        """
        Sample a PER mini-batch and perform one SAC gradient step.

        Steps:
          1. Stratified-priority sample from buffer (returns IS weights +
             buffer indices for later priority update).
          2. IS-weighted critic update; critic returns per-sample TD errors.
          3. Actor + alpha updates (uniform weights — standard SAC-v2).
          4. Polyak target update.
          5. Push new priorities back to the SumTree.

        Returns dict of scalar metrics.
        """
        (obs, act, rew, next_obs, done,
         bounds, next_bounds,
         sample_indices, is_weights) = buffer.sample(self.batch_size)

        # Pass numpy arrays directly — @tf.function on the inner methods handles
        # conversion without creating persistent constant graph nodes.
        # tf.constant() inside a Python loop accumulates graph nodes on Metal backend,
        # causing episode time to grow 4× over 1000 episodes → OOM kill.
        obs_tf          = obs.astype(np.float32)
        act_tf          = act.astype(np.float32)
        rew_tf          = rew.astype(np.float32)
        next_obs_tf     = next_obs.astype(np.float32)
        done_tf         = done.astype(np.float32)
        bounds_tf       = bounds.astype(np.float32)       # (batch, 2) at s_t
        next_bounds_tf  = next_bounds.astype(np.float32)  # (batch, 2) at s_{t+N}
        is_weights_tf   = is_weights.astype(np.float32)   # (batch, 1)

        # IS-weighted critic update; returns per-sample |δ_i| for PER
        critic_loss, td_errors = self._update_critic(
            obs_tf, act_tf, rew_tf, next_obs_tf, done_tf, next_bounds_tf,
            is_weights_tf,
        )
        actor_loss, proj_pen   = self._update_actor(obs_tf, bounds_tf)
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
        """
        Select an action for a single observation.

        Parameters
        ----------
        obs          : np.ndarray, shape (obs_dim,)
        deterministic: bool -- use tanh(mu) instead of sampling

        Returns
        -------
        np.ndarray, shape (act_dim,) in (-1, 1)  — ready for env.step()
        """
        obs_tf = tf.constant(obs[np.newaxis], dtype=tf.float32)  # (1, obs_dim)
        if deterministic:
            act_tf = self.actor.get_deterministic_action(obs_tf)
        else:
            act_tf, _ = self.actor(obs_tf)
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
# PERFECT FORESIGHT LP ORACLE  (single-bus, simplified degradation)
# ================================================================

class PerfectForesightLP:
    """
    Compute the maximum profit achievable with perfect foresight of all
    future LMPs over one episode.  Serves as a warm-start expert.

    Formulation (LP, scipy linprog, HiGHS solver):
    ------------------------------------------------
    Variables: x = [p_dis(0..T-1), p_ch(0..T-1)]  in R^{2T},  x >= 0

    Objective (minimise = maximise profit):
      c_obj[t]   = c_dis - lmp[t]   for t in 0..T-1   (discharge term)
      c_obj[T+t] = lmp[t]           for t in 0..T-1   (charging term)

    SoC dynamics (dt = 1 h):
      SoC[t] = SoC_0 + sum_{s=0}^{t-1} [ p_ch[s]*eta_ch - p_dis[s]/eta_dis ]

    Inequality constraints A_ub @ x <= b_ub  (shape 2T x 2T):
      SoC >= SoC_min:  sum_{s<t} [ p_dis[s]/eta_dis - p_ch[s]*eta_ch ] <= SoC_0 - SoC_min
      SoC <= SoC_max: -sum_{s<t} [ p_dis[s]/eta_dis - p_ch[s]*eta_ch ] <= SoC_max - SoC_0

    Box bounds: 0 <= p_dis[t] <= P_max,  0 <= p_ch[t] <= P_max

    Notes:
      - No network constraints (LP is a single-bus oracle).
      - c_dis is set with an economics-aware conservative approximation:
          • determine deepest potentially profitable degradation segment from
            observed LMP_max at the BESS bus;
          • restrict SoC_min to exclude clearly unprofitable deeper segments;
          • set c_dis to the overlap-weighted mean segment cost over
            [SoC_min, SoC_max], converted to $/MWh delivered (divide by eta_dis).
        This keeps warm-start transitions aligned with current market economics
        without assuming the old "segment-1 only" regime.
      - Simultaneous charge/discharge will not occur at optimum when prices > 0.
    """

    def __init__(self, env: BESSEnv):
        self.T       = env.EPS_LEN
        self.P_max   = env.P_MAX
        self.SoC_0   = env.SOC_INIT
        self.SoC_max = env.SOC_MAX
        self.eta_ch  = env.ETA_CH
        self.eta_dis = env.ETA_DIS

        # Economics-aware depth gating:
        # segments with breakeven above observed LMP_max are excluded from the LP.
        lmp_bus = env.lmps[env.converged, env.bess_col]
        lmp_max = float(np.max(lmp_bus)) if len(lmp_bus) else 0.0
        self.lmp_max = lmp_max
        breakeven = env.c_deg / env.ETA_DIS
        profitable = np.where(lmp_max >= breakeven)[0]
        deepest_profitable = int(profitable[-1] + 1) if len(profitable) else 1
        self.deepest_profitable_seg = deepest_profitable
        self.SoC_min = max(
            env.SOC_MIN,
            env.E_CAP - deepest_profitable * env.SEG_SIZE,
        )

        # Conservative single-coefficient approximation over the allowed SoC band.
        # c_j is in $/MWh extracted; convert to $/MWh delivered via /eta_dis.
        weights = []
        costs = []
        for j in range(1, env.J_DEG + 1):
            seg_upper = env.E_CAP - (j - 1) * env.SEG_SIZE
            seg_lower = env.E_CAP - j * env.SEG_SIZE
            overlap = max(0.0, min(self.SoC_max, seg_upper) - max(self.SoC_min, seg_lower))
            if overlap > 1e-9:
                weights.append(overlap)
                costs.append(env.c_deg[j - 1] / env.ETA_DIS)
        if weights:
            self.c_dis = float(np.dot(weights, costs) / np.sum(weights))
        else:
            self.c_dis = float(env.c_deg[0] / env.ETA_DIS)

        self._build_constraint_matrix()

    def _build_constraint_matrix(self) -> None:
        """Pre-build the (2T x 2T) inequality constraint matrix."""
        T = self.T

        # Lower-triangular cumulative matrix:
        #   L[i, j] = 1 if j <= i (sum_{s=0}^{t-1} over rows t=1..T)
        L = np.tril(np.ones((T, T), dtype=np.float64))

        # SoC_min rows (0..T-1):
        #   sum_{s<t} p_dis[s]/eta_dis - p_ch[s]*eta_ch <= SoC_0 - SoC_min
        A_min = np.hstack([L / self.eta_dis, -L * self.eta_ch])   # (T, 2T)

        # SoC_max rows (T..2T-1):
        #  -sum_{s<t} p_dis[s]/eta_dis + p_ch[s]*eta_ch <= SoC_max - SoC_0
        A_max = np.hstack([-L / self.eta_dis, L * self.eta_ch])   # (T, 2T)

        self.A_ub = np.vstack([A_min, A_max])                      # (2T, 2T)
        self.b_ub = np.concatenate([
            np.full(T, self.SoC_0 - self.SoC_min),
            np.full(T, self.SoC_max - self.SoC_0),
        ])

    def solve(self, lmps_episode: np.ndarray):
        """
        Solve the LP for a given episode LMP sequence.

        Parameters
        ----------
        lmps_episode : np.ndarray, shape (T,)  [$/MWh]

        Returns
        -------
        profit : float  -- maximum episode profit [$]
        p_dis  : np.ndarray (T,)  -- optimal discharge schedule [MW]
        p_ch   : np.ndarray (T,)  -- optimal charge schedule [MW]
        Returns (0, zeros, zeros) if LP is infeasible.
        """
        T    = self.T
        lmps = np.asarray(lmps_episode[:T], dtype=np.float64)

        # Objective: minimise -profit
        #   c_dis[t] - lmp[t]  for discharge variables
        #   lmp[t]              for charge variables
        c_obj = np.concatenate([self.c_dis - lmps, lmps])

        bounds = [(0.0, self.P_max)] * (2 * T)

        result = linprog(
            c=c_obj,
            A_ub=self.A_ub,
            b_ub=self.b_ub,
            bounds=bounds,
            method="highs",
        )

        if result.status == 0:
            p_dis  = result.x[:T]
            p_ch   = result.x[T:]
            profit = -float(result.fun)   # negate: linprog minimised -profit
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
    """
    Commit the oldest transition from the rolling 1-step window.

    Uses full N-step horizon when available (len >= n_step+1).  During episode-tail
    flush (len < n_step+1), commits the remaining partial horizon with done_n=True
    whenever terminal is observed in the used reward slice.
    """
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
) -> tuple:
    """
    Run one episode with the stochastic SAC policy, storing N-step transitions.

    For n_step > 1 the buffer receives aggregated transitions:
      (s_t, a_t, G_t^N, s_{t+N}, done_N, bounds_t)
    where:
      G_t^N    = sum_{k=0}^{N-1} gamma^k * r_{t+k}   (N-step discounted return)
      s_{t+N}  = observation N steps after s_t
      done_N   = True if the episode ended within the N-step window
                 (suppresses bootstrapping from s_{t+N})
      bounds_t = safety-layer feasibility interval [p_lo, p_hi] at state s_t,
                 used in the Convention-B actor update at s_t (unchanged).

    Episode tails are flushed after termination, so no transitions are dropped.

    Parameters
    ----------
    agent  : SACAgent
    env    : BESSEnv
    buffer : ReplayBuffer  (modified in-place)
    n_step : int    -- N-step horizon (from HP["n_step"])
    gamma  : float  -- discount factor (from HP["gamma"])

    Returns
    -------
    ep_return : float  -- total (scaled) episode return (sum of raw 1-step rewards)
    ep_infos  : list   -- list of info dicts from env.step()
    n_added   : int    -- number of transitions pushed to replay buffer
    """
    obs, _    = env.reset()
    ep_return = 0.0
    ep_infos  = []
    n_added   = 0
    # Rolling window: each entry is a raw 1-step transition
    # (obs, act_proj, reward, next_obs, done, p_lo_norm, p_hi_norm)
    window = deque()

    for _ in range(env.EPS_LEN):
        act = agent.act(obs, deterministic=False)   # shape (1,) in [-1, 1]
        act = np.clip(act, -1.0, 1.0)              # guard against tanh float overflow
        next_obs, rew, terminated, truncated, info = env.step(act)
        done = terminated or truncated

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
    n_step:    int   = 1,
    gamma:     float = 0.99,
) -> int:
    """
    Pre-fill the replay buffer with LP-optimal expert trajectories.

    Motivation (Rasic et al. 2025, RLBattEM4):
      Without expert guidance the agent often converges to a trivial
      discharge-only policy and fails to discover long-term arbitrage
      (observed in run-7 cold-start failure). Filling 62.5% of the
      initial batch with LP-optimal trajectories exposes the critic to
      high-quality (s,a,r,s') pairs from episode 1, anchoring the
      Q-estimates in profitable operating regions before SAC exploration
      begins. The expert data is gradually diluted as the 200k buffer
      fills, preventing it from dominating later training.

    Implementation:
      1. Reset env to a random training start.
      2. Read episode LMPs directly from env.lmps (no env modification needed).
      3. Solve PerfectForesightLP for the optimal p_dis, p_ch schedule.
      4. Replay the LP schedule step-by-step through the REAL env.step():
           - Safety layer projects actions to PTDF-feasible set (LP is single-bus).
           - Actual rewards, SoC dynamics, and bounds are recorded.
      5. Use the same N+1 sliding window as run_episode_collect to build
         N-step transitions, ensuring expert data is in the exact same format
         as SAC-collected data (gamma^N bootstrap, next_bounds stored).

    Parameters
    ----------
    env       : BESSEnv               -- environment (training mode expected)
    buffer    : ReplayBuffer           -- buffer to fill in-place
    lp_oracle : PerfectForesightLP     -- pre-built LP oracle
    n_target  : int                    -- number of transitions to add
    n_step    : int                    -- N-step horizon (match HP["n_step"])
    gamma     : float                  -- discount factor (match HP["gamma"])

    Returns
    -------
    int  -- actual number of transitions added (may be < n_target if
            episode boundaries are reached before n_target)
    """
    n_added = 0

    while n_added < n_target:
        # Reset to a random training start
        obs, _ = env.reset()

        # Extract episode LMPs from the data array (no env modification needed)
        start_t    = env.global_t
        lmps_ep    = env.lmps[start_t : start_t + env.EPS_LEN, env.bess_col]

        # Solve LP for the optimal schedule
        _, p_dis, p_ch = lp_oracle.solve(lmps_ep)

        # Convert to normalised [-1, 1] action: positive = discharge
        actions_lp = np.clip((p_dis - p_ch) / env.P_MAX, -1.0, 1.0).astype(np.float32)

        # Replay through real env with N+1 look-ahead window
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

            while len(window) >= n_step + 1 and n_added < n_target:
                _commit_nstep_transition_from_window(window, buffer, n_step, gamma)
                n_added += 1

            obs = next_obs
            if done:
                break

            if n_added >= n_target:
                break

        # Tail flush for this episode
        while len(window) > 0 and n_added < n_target:
            _commit_nstep_transition_from_window(window, buffer, n_step, gamma)
            n_added += 1

    return n_added


def warmstart_actor_bc(
    agent: SACAgent,
    buffer: ReplayBuffer,
    n_expert: int,
    n_steps: int,
    batch_size: int,
) -> dict:
    """
    Behavior-clone actor on the expert prefix already stored in replay buffer.

    Assumes warm-start ran on an empty buffer, so indices [0, n_expert) are
    LP expert transitions.
    """
    n_expert = int(min(n_expert, len(buffer)))
    n_steps = int(max(0, n_steps))
    if n_expert <= 0 or n_steps == 0:
        return {"n_steps": 0, "loss_init": np.nan, "loss_final": np.nan, "loss_mean": np.nan}

    bsz = int(max(1, min(batch_size, n_expert)))
    losses = []
    for _ in range(n_steps):
        idx = np.random.randint(0, n_expert, size=bsz)
        obs_tf = tf.convert_to_tensor(buffer.obs[idx], dtype=tf.float32)
        act_tf = tf.convert_to_tensor(buffer.act[idx], dtype=tf.float32)
        loss = agent._bc_actor_step(obs_tf, act_tf)
        losses.append(float(loss))

    return {
        "n_steps": n_steps,
        "loss_init": float(losses[0]),
        "loss_final": float(losses[-1]),
        "loss_mean": float(np.mean(losses)),
    }


def _run_evaluation(
    agent: SACAgent,
    env:   BESSEnv,
) -> dict:
    """
    Evaluate the agent deterministically (no exploration noise) on every
    held-out test start exactly once.

    Iterates over all env.test_starts (final week of each calendar month,
    ~95 episodes) using reset(options={"start_t": t}) to set the exact
    episode start.  This gives a reproducible, full-coverage test evaluation
    without random sampling variance.

    Parameters
    ----------
    agent : SACAgent
    env   : BESSEnv

    Returns
    -------
    dict with keys: eval_mean, eval_std, eval_min, eval_max, n_eval
    """
    returns = []
    for start_t in env.test_starts:
        obs, _    = env.reset(options={"start_t": int(start_t)})
        ep_return = 0.0

        for _ in range(env.EPS_LEN):
            act = np.clip(agent.act(obs, deterministic=True), -1.0, 1.0)
            obs, rew, terminated, truncated, _ = env.step(act)
            ep_return += rew
            if terminated or truncated:
                break

        returns.append(ep_return)

    returns = np.array(returns)
    return {
        "eval_mean": float(np.mean(returns)),
        "eval_std":  float(np.std(returns)),
        "eval_min":  float(np.min(returns)),
        "eval_max":  float(np.max(returns)),
        "n_eval":    len(returns),
    }


def train_sac(env: BESSEnv, hp: dict) -> tuple:
    """
    Full SAC training pipeline.

    SAC Training
    ~~~~~~~~~~~~
    For each episode:
      1. Collect one episode with the stochastic policy.
      2. Perform env.EPS_LEN (=168) gradient updates from the buffer.
      3. Every hp['eval_freq'] episodes: evaluate on ALL test starts + checkpoint.

    Parameters
    ----------
    env : BESSEnv    -- instantiated environment
    hp  : dict       -- hyperparameter dict

    Returns
    -------
    agent : SACAgent  -- trained agent
    log   : dict      -- training metrics
    """
    seed = hp["seed"]
    np.random.seed(seed)
    tf.random.set_seed(seed)

    obs_dim = env.observation_space.shape[0]   # 34 (SoC+LMP+hist24+PV+step+segs4+p_lo+p_hi)
    act_dim = env.action_space.shape[0]        # 1

    buffer = ReplayBuffer(obs_dim, act_dim, hp["buffer_size"])
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
    print(f"  lp_warmstart:  {hp.get('lp_warmstart_steps', 0)} transitions "
          f"({hp.get('lp_warmstart_steps', 0) / hp['buffer_size'] * 100:.1f}% "
          f"of total buffer capacity)")
    print(f"  warmstart BC:  {hp.get('warmstart_bc_steps', 0)} steps "
          f"(batch={hp.get('warmstart_bc_batch_size', hp['batch_size'])})")

    # ---- LP Expert Warm-Start ----
    _ws_steps = hp.get("lp_warmstart_steps", 0)
    if _ws_steps > 0:
        print(f"\n{'='*65}")
        print(f"LP Expert Warm-Start (target: {_ws_steps} transitions)")
        print(f"{'='*65}")
        _lp_oracle = PerfectForesightLP(env)
        print(f"  Oracle economics check: LMP_max={_lp_oracle.lmp_max:.2f} $/MWh, "
              f"deepest profitable segment={_lp_oracle.deepest_profitable_seg}")
        print(f"  Oracle SoC band: [{_lp_oracle.SoC_min:.1f}, {_lp_oracle.SoC_max:.1f}] MWh, "
              f"c_dis≈{_lp_oracle.c_dis:.2f} $/MWh delivered")
        _t_ws = time.time()
        n_ws = lp_warmstart(
            env, buffer, _lp_oracle,
            n_target=_ws_steps,
            n_step=hp.get("n_step", 1),
            gamma=hp["gamma"],
        )
        print(f"  Added {n_ws} LP transitions in {time.time()-_t_ws:.1f}s  "
              f"(buffer fill: {len(buffer)}/{hp['buffer_size']})")
        _bc_steps = int(hp.get("warmstart_bc_steps", 0))
        if _bc_steps > 0 and n_ws > 0:
            _t_bc = time.time()
            _bc = warmstart_actor_bc(
                agent=agent,
                buffer=buffer,
                n_expert=n_ws,
                n_steps=_bc_steps,
                batch_size=int(hp.get("warmstart_bc_batch_size", hp["batch_size"])),
            )
            print(
                "  Actor BC pretrain: "
                f"steps={_bc['n_steps']}, "
                f"loss { _bc['loss_init']:.4f} -> { _bc['loss_final']:.4f} "
                f"(mean={_bc['loss_mean']:.4f}) in {time.time()-_t_bc:.1f}s"
            )

    # ---- SAC training ----
    print(f"\n{'='*65}")
    print(f"SAC Training ({hp['train_episodes']} episodes)")
    print(f"{'='*65}")

    checkpoint_dir = os.path.join(OUTPUTS_DIR, "step4_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log = {
        "train_returns":   [],
        "critic_losses":   [],
        "actor_losses":    [],
        "alpha_values":    [],
        "proj_penalties":  [],   # ||a_raw - a_proj||^2; trends → 0 as actor learns feasibility
        "eval_returns":    [],   # list of dicts: {episode, eval_mean, eval_std, ...}
        "episodes":        [],
    }

    # Early-stopping state
    patience          = hp.get("early_stop_patience",  4)
    min_delta         = hp.get("early_stop_min_delta",  0.001)
    best_eval         = -np.inf   # all-time best eval_mean (gates checkpoint save)
    best_eval_patience = -np.inf  # best eval used for patience counting (requires min_delta)
    no_improve        = 0         # consecutive evals without min_delta improvement
    best_ep           = 0         # episode at which best_eval was achieved

    t_train_start = time.time()

    for ep in range(1, hp["train_episodes"] + 1):

        # 1. Collect one episode (N-step transitions stored in buffer)
        ep_return, _, n_collected = run_episode_collect(
            agent, env, buffer,
            n_step=hp.get("n_step", 1),
            gamma=hp["gamma"],
        )

        # 2a. Anneal PER IS exponent β: 0.4 → 1.0 over full training run.
        #     Early training: β low → biased-but-stable gradients.
        #     Late training: β → 1 → fully unbiased IS correction (Schaul 2016 §3.4).
        buffer.anneal_beta(ep / hp["train_episodes"])

        # 2b. Gradient updates (one per collected transition, once the
        #     buffer has enough diverse transitions to form stable TD targets)
        _min_fill = hp.get("min_buffer_fill", hp["batch_size"])
        c_losses, a_losses, alphas, proj_pens = [], [], [], []
        for _ in range(n_collected):
            if len(buffer) >= _min_fill:
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
            eval_m = _run_evaluation(agent, env)
            log["eval_returns"].append({"episode": ep, **eval_m})

            # --- Checkpoint: save whenever this is the all-time best eval ---
            # Decoupled from patience so we never miss the true global maximum.
            if eval_m["eval_mean"] > best_eval:
                best_eval = eval_m["eval_mean"]
                best_ep   = ep
                best_path = os.path.join(checkpoint_dir, "best", "agent")
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                agent.save(best_path)
                tag = " (BEST)"
            else:
                tag = ""

            # --- Patience: only reset when improvement exceeds min_delta ---
            # This prevents micro-improvements (noise) from resetting patience
            # indefinitely while still detecting genuine slow progress.
            if eval_m["eval_mean"] > best_eval_patience + min_delta:
                best_eval_patience = eval_m["eval_mean"]
                no_improve = 0
            else:
                no_improve += 1
                tag += f" (no improve {no_improve}/{patience})"

            # Always save periodic checkpoint for later analysis
            ckpt_path = os.path.join(checkpoint_dir, f"ep{ep:04d}", "agent")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            agent.save(ckpt_path)

            elapsed = time.time() - t_train_start
            print(
                f"  Ep {ep:4d}/{hp['train_episodes']} | "
                f"train_ret={ep_return:+.4f} | "
                f"eval={eval_m['eval_mean']:+.4f}±{eval_m['eval_std']:.4f} | "
                f"alpha={log['alpha_values'][-1]:.4f} | "
                f"critic_loss={log['critic_losses'][-1]:.4f} | "
                f"{elapsed:.0f}s{tag}"
            )

            # --- Stop if patience exceeded ---
            if no_improve >= patience:
                print(f"\n  Early stop at ep {ep}: no min_delta improvement for "
                      f"{patience} evals. "
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

    # ---- Save training log ----
    log_path = os.path.join(OUTPUTS_DIR, "step4_training_log.npy")
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
    """
    Print a comprehensive post-training results table:
      - Training return statistics (full run + final 100 eps)
      - Evaluation return statistics on all test starts
    """
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
        eval_means = [e["eval_mean"] for e in log["eval_returns"]]
        best_idx   = int(np.argmax(eval_means))
        n_test     = log["eval_returns"][0].get("n_eval", "?")
        print(f"\nEvaluation (every {hp['eval_freq']} eps, all {n_test} test starts):")
        print(f"  Best eval mean return   : {eval_means[best_idx]:+.4f}  "
              f"(ep {log['eval_returns'][best_idx]['episode']})")
        print(f"  Final eval mean return  : {eval_means[-1]:+.4f}")
        print(f"  Final eval std          : {log['eval_returns'][-1]['eval_std']:.4f}")
        print(f"  Final eval min/max      : "
              f"{log['eval_returns'][-1]['eval_min']:+.4f} / "
              f"{log['eval_returns'][-1]['eval_max']:+.4f}")

    print("=" * 65)


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    """
    Entry point: instantiate environment, train SAC, print summary.

    Usage:
        python 4.SACAgent.py
    """
    # Ensure outputs directory exists
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("\n" + "=" * 65)
    print("NC-SafeRL  STEP 4: SAC Agent Training")
    print("RTS-GMLC 73-bus | BESS bus 111 | TF 2.x CPU")
    print("=" * 65)

    # ---- Create environment ----
    env = BESSEnv(seed=HP["seed"])

    # ---- E0: PTDF Binding Pre-Training Diagnostic ----
    # Validates that transmission constraints actively limit BESS dispatch at
    # this bus, justifying the NC safety layer over battery-only projection.
    # If PTDF constraints bind < 5% of hours, the ablation gap will be tiny.
    # Run once before training; results saved to outputs/ptdf_binding_stats.npy.
    ptdf_binding_analysis(env, save_dir=OUTPUTS_DIR)

    # ---- Train ----
    # CONVERGENCE MONITORING NOTE
    # ----------------------------
    # Training episode returns are intentionally NOISY: each episode is sampled
    # from a random training week, and weeks differ significantly in LMP level
    # (e.g., winter peak $140/MWh vs. summer shoulder $35/MWh). A rolling mean
    # of training returns (see log["train_returns"]) is useful for stability
    # monitoring but is NOT the convergence criterion.
    #
    # The convergence plot for the paper is: eval_mean vs. episode.
    # _run_evaluation() evaluates the DETERMINISTIC policy on ALL ~95 held-out
    # test weeks every eval_freq=100 episodes. Averaging over all test starts
    # eliminates week-selection variance, giving a stable, reproducible signal.
    # Early stopping triggers after `early_stop_patience` consecutive evals
    # (= patience * eval_freq episodes) without min_delta improvement.
    #
    # Best checkpoint = argmax(eval_mean) — loaded from outputs/step4_checkpoints/best/
    t0 = time.time()
    agent, log = train_sac(env, HP)
    total_time = time.time() - t0

    print(f"\nTotal wall-clock time: {total_time/60:.1f} min")

    # ---- Results summary ----
    print_results_summary(log, agent, env, HP)

    # Load best checkpoint before exporting "final" weights.
    best_prefix = os.path.join(OUTPUTS_DIR, "step4_checkpoints", "best", "agent")
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
    print("\nStep 4 complete.  Ready for Step 5 (analysis / paper figures).")


if __name__ == "__main__":
    main()
