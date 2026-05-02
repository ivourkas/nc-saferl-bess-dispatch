"""
Step 6: Forecast-augmented Soft Actor-Critic baseline for NC-SafeRL.

This script keeps Step 4 (reactive SAC) and Step 5 (forecast-augmented MPC)
unchanged and adds a third benchmark:

  - the same bounded SAC architecture as Step 4,
  - augmented with the same causal train-only forecasts used by Step 5,
  - delivered through an observation wrapper rather than by modifying the env.

The resulting comparison ladder is:

  Step 4  : reactive SAC
  Step 5  : forecast-augmented MPC/MILP
  Step 6  : forecast-augmented SAC
"""

# ================================================================
# IMPORTS
# ================================================================

import os
import math
import json
import time
import copy
import argparse
import datetime
import importlib.util

import numpy as np
from gymnasium import spaces


# ================================================================
# DYNAMIC IMPORTS  (filenames start with digits)
# ================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(module_name: str, filename: str):
    path = os.path.join(_HERE, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


step4 = _load_module("step4_sac", "4.SACAgent.py")
step5 = _load_module("step5_mpc", "5.OptimizationBaseline.py")

BESSEnv = step4.BESSEnv
ptdf_binding_analysis = step4.ptdf_binding_analysis
tf = step4.tf


# Keep Step 6 artifacts separate from Step 4 outputs.
STEP6_WORKSPACE_DIR = os.path.join(_HERE, "outputs", "step6_workspace")
STEP6_FINAL_DIR = os.path.join(_HERE, "outputs", "step6_final")
step4.OUTPUTS_DIR = STEP6_WORKSPACE_DIR


# ================================================================
# HYPERPARAMETERS
# ================================================================

HP = dict(step4.HP)
HP.update({
    "future_hidden_dim": 128,
    "forecast_k_neighbors": int(step5.CFG["k_neighbors"]),
    "forecast_eps": float(step5.CFG["forecast_eps"]),
    "forecast_use_mask": True,
    "forecast_exclude_current_train_episode": True,
})

FORECAST_HORIZON_HOURS = step5.FORECAST_HORIZON_HOURS

LOG_STD_MIN = step4.LOG_STD_MIN
LOG_STD_MAX = step4.LOG_STD_MAX
BASE_OBS_DIM = 38


# ================================================================
# FORECASTER
# ================================================================

class AuditedAnalogForecaster(step5.AnalogForecaster):
    """
    Step-5-style analog forecaster with two additions:
      1. optional exclusion of a training-episode hour range, and
      2. metadata that can be saved as an anti-leakage audit trail.
    """

    def forecast(
        self,
        x_now: np.ndarray,
        horizon: int,
        exclude_lo: int | None = None,
        exclude_hi: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        horizon = int(horizon)

        mask = np.ones(len(self.anchor_hours), dtype=bool)
        if exclude_lo is not None and exclude_hi is not None:
            mask &= ~((self.anchor_hours >= int(exclude_lo)) & (self.anchor_hours < int(exclude_hi)))
            if not np.any(mask):
                mask[:] = True

        candidate_idx = np.flatnonzero(mask)

        x_norm = (x_now - self.mean) / self.std
        x_norm_sq = float(np.dot(x_norm, x_norm))

        X_norm = self.X_norm[candidate_idx]
        X_norm_sq = self.X_norm_sq[candidate_idx]

        d2 = X_norm_sq + x_norm_sq - 2.0 * (X_norm @ x_norm)
        d2 = np.maximum(d2, 0.0)

        k = min(self.k, len(candidate_idx))
        local_idx = np.argpartition(d2, k - 1)[:k]
        local_idx = local_idx[np.argsort(d2[local_idx])]
        idx = candidate_idx[local_idx]

        d = np.sqrt(d2[local_idx])
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
            "exclude_lo": None if exclude_lo is None else int(exclude_lo),
            "exclude_hi": None if exclude_hi is None else int(exclude_hi),
            "train_end": int(self.train_end),
        }
        return lmp_hat, lo_hat, hi_hat, meta


# ================================================================
# FORECAST-AUGMENTED ENV WRAPPER
# ================================================================

class ForecastAugmentedEnv:
    """
    Observation wrapper that appends:
      - normalized future LMP forecast      (H_max,)
      - normalized future p_lo_net forecast (H_max,)
      - normalized future p_hi_net forecast (H_max,)
      - horizon mask                        (H_max,)

    The base 38-dimensional observation remains unchanged at the front.
    """

    def __init__(
        self,
        base_env: BESSEnv,
        forecaster: AuditedAnalogForecaster,
        include_horizon_mask: bool = True,
        exclude_current_train_episode: bool = True,
        future_horizon: int | None = None,
    ):
        self.base_env = base_env
        self.forecaster = forecaster
        self.include_horizon_mask = bool(include_horizon_mask)
        self.exclude_current_train_episode = bool(exclude_current_train_episode)

        self.base_obs_dim = int(base_env.observation_space.shape[0])
        self.future_horizon = int(future_horizon) if future_horizon is not None else int(base_env.EPS_LEN)
        self.future_channels = 4 if self.include_horizon_mask else 3
        self.forecast_dim = self.future_horizon * self.future_channels
        self.obs_dim = self.base_obs_dim + self.forecast_dim

        low = np.full(self.obs_dim, -1.0, dtype=np.float32)
        high = np.full(self.obs_dim, 1.0, dtype=np.float32)
        if self.include_horizon_mask:
            mask_start = self.base_obs_dim + 3 * self.future_horizon
            low[mask_start:] = 0.0
            high[mask_start:] = 1.0

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = base_env.action_space
        self.metadata = getattr(base_env, "metadata", {})

        self._episode_anchor_exclude = None
        self._last_forecast_meta = None

    def __getattr__(self, name):
        return getattr(self.base_env, name)

    def get_last_forecast_meta(self) -> dict | None:
        if self._last_forecast_meta is None:
            return None
        return copy.deepcopy(self._last_forecast_meta)

    def _forecast_obs(self, base_obs: np.ndarray) -> np.ndarray:
        H = min(int(self.base_env.EPS_LEN - self.base_env.step_in_ep), self.future_horizon)
        if H <= 0:
            self._last_forecast_meta = None
            return np.zeros(self.obs_dim, dtype=np.float32)

        x_now = self.forecaster.feature_from_env(self.base_env)

        exclude_lo = None
        exclude_hi = None
        if self.exclude_current_train_episode and self._episode_anchor_exclude is not None:
            exclude_lo, exclude_hi = self._episode_anchor_exclude

        lmp_hat, lo_hat, hi_hat, meta = self.forecaster.forecast(
            x_now,
            H,
            exclude_lo=exclude_lo,
            exclude_hi=exclude_hi,
        )

        lmp_hat[0] = float(self.base_env.lmps[self.base_env.global_t, self.base_env.bess_col])
        # Match Step 5: use exact current safe interval at h = 0.
        _, p_lo_now, p_hi_now = self.base_env._safety_project(0.0)
        lo_hat[0] = float(p_lo_now)
        hi_hat[0] = float(p_hi_now)

        lmp_pad = np.zeros(self.future_horizon, dtype=np.float32)
        lo_pad = np.zeros(self.future_horizon, dtype=np.float32)
        hi_pad = np.zeros(self.future_horizon, dtype=np.float32)
        mask = np.zeros(self.future_horizon, dtype=np.float32)

        lmp_pad[:H] = (lmp_hat / self.base_env.LMP_NORM).astype(np.float32)
        lo_pad[:H] = (lo_hat / self.base_env.P_MAX).astype(np.float32)
        hi_pad[:H] = (hi_hat / self.base_env.P_MAX).astype(np.float32)
        mask[:H] = 1.0

        extras = [lmp_pad, lo_pad, hi_pad]
        if self.include_horizon_mask:
            extras.append(mask)

        meta.update({
            "remaining_horizon": H,
            "global_hour": int(self.base_env.global_t),
            "mask_active_steps": int(H),
        })
        self._last_forecast_meta = meta

        aug = np.concatenate([base_obs.astype(np.float32), *extras]).astype(np.float32)
        return np.clip(aug, -1.0, 1.0)

    def reset(self, seed=None, options=None):
        base_obs, info = self.base_env.reset(seed=seed, options=options)
        if options is None or "start_t" not in options:
            if self.base_env._mode == "train" and self.exclude_current_train_episode:
                self._episode_anchor_exclude = (
                    int(self.base_env.global_t),
                    int(self.base_env.global_t + self.base_env.EPS_LEN),
                )
            else:
                self._episode_anchor_exclude = None
        else:
            self._episode_anchor_exclude = None

        return self._forecast_obs(base_obs), info

    def step(self, action):
        base_obs, rew, terminated, truncated, info = self.base_env.step(action)
        if terminated or truncated:
            self._last_forecast_meta = None
            aug_obs = np.zeros(self.obs_dim, dtype=np.float32)
        else:
            aug_obs = self._forecast_obs(base_obs)
        return aug_obs, rew, terminated, truncated, info


# ================================================================
# FORECAST-AUGMENTED NETWORKS
# ================================================================

class ForecastAugmentedActor(tf.keras.Model):
    """
    Step-4-style bounded actor with an extra Conv1D branch over forecast inputs.
    """

    _BASE_LMP_START = 1
    _BASE_LMP_END = 26
    _BASE_SCALAR_IDX = [0] + list(range(26, 38))
    _P_LO_IDX = 32
    _P_HI_IDX = 33

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        future_len: int = 168,
        future_channels: int = 4,
        future_hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.future_len = int(future_len)
        self.future_channels = int(future_channels)

        self.lmp_conv_short = tf.keras.layers.Conv1D(
            32, kernel_size=4, strides=1, padding="causal", activation="relu"
        )
        self.lmp_conv_long = tf.keras.layers.Conv1D(
            32, kernel_size=8, strides=1, padding="causal", activation="relu"
        )
        self.lmp_pool_short = tf.keras.layers.GlobalAveragePooling1D()
        self.lmp_pool_long = tf.keras.layers.GlobalAveragePooling1D()
        self.lmp_dense = tf.keras.layers.Dense(32, activation="relu")

        self.scalar_dense = tf.keras.layers.Dense(32, activation="relu")

        self.future_conv_short = tf.keras.layers.Conv1D(
            32, kernel_size=6, strides=1, padding="causal", activation="relu"
        )
        self.future_conv_long = tf.keras.layers.Conv1D(
            32, kernel_size=12, strides=1, padding="causal", activation="relu"
        )
        self.future_pool_short = tf.keras.layers.GlobalAveragePooling1D()
        self.future_pool_long = tf.keras.layers.GlobalAveragePooling1D()
        self.future_dense = tf.keras.layers.Dense(future_hidden_dim, activation="relu")

        self.joint_ln = tf.keras.layers.LayerNormalization()
        self.trunk1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.trunk2 = tf.keras.layers.Dense(hidden_dim, activation="relu")

        self.mu_layer = tf.keras.layers.Dense(act_dim)
        self.log_std_layer = tf.keras.layers.Dense(
            act_dim,
            kernel_initializer="zeros",
            bias_initializer=tf.keras.initializers.Constant(LOG_STD_MIN + 2.0),
        )

        self._scalar_idx_tf = tf.constant(self._BASE_SCALAR_IDX, dtype=tf.int32)

    def _split_obs(self, obs: tf.Tensor):
        base_obs = obs[:, :BASE_OBS_DIM]
        future_flat = obs[:, BASE_OBS_DIM:]
        off = self.future_len
        channels = [
            future_flat[:, 0:off],
            future_flat[:, off:2 * off],
            future_flat[:, 2 * off:3 * off],
        ]
        if self.future_channels == 4:
            channels.append(future_flat[:, 3 * off:4 * off])
        future = tf.stack(channels, axis=-1)
        return base_obs, future

    def _encode(self, obs: tf.Tensor) -> tf.Tensor:
        base_obs, future = self._split_obs(obs)

        lmp = tf.expand_dims(base_obs[:, self._BASE_LMP_START:self._BASE_LMP_END], axis=-1)
        short_e = self.lmp_pool_short(self.lmp_conv_short(lmp))
        long_e = self.lmp_pool_long(self.lmp_conv_long(lmp))
        lmp_e = self.lmp_dense(tf.concat([short_e, long_e], axis=-1))

        scalar = tf.gather(base_obs, self._scalar_idx_tf, axis=1)
        scalar_e = self.scalar_dense(scalar)

        fut_short = self.future_pool_short(self.future_conv_short(future))
        fut_long = self.future_pool_long(self.future_conv_long(future))
        fut_e = self.future_dense(tf.concat([fut_short, fut_long], axis=-1))

        joint = self.joint_ln(tf.concat([lmp_e, scalar_e, fut_e], axis=-1))
        return self.trunk2(self.trunk1(joint))

    def _distribution_params(self, obs: tf.Tensor):
        _LOGIT_CLIP = 3.0
        h = self._encode(obs)
        mu_raw = self.mu_layer(h)
        mu = _LOGIT_CLIP * tf.tanh(mu_raw)
        log_std = tf.clip_by_value(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def call(self, obs):
        mu, log_std = self._distribution_params(obs)
        std = tf.exp(log_std)

        base_obs = obs[:, :BASE_OBS_DIM]
        p_lo_norm = base_obs[:, self._P_LO_IDX:self._P_LO_IDX + 1]
        p_hi_norm = base_obs[:, self._P_HI_IDX:self._P_HI_IDX + 1]
        center = (p_hi_norm + p_lo_norm) * 0.5
        half_width = (p_hi_norm - p_lo_norm) * 0.5

        eps = tf.random.normal(tf.shape(mu))
        z = mu + std * eps

        log_pi_normal = (
            -0.5 * (eps ** 2 + tf.math.log(2.0 * math.pi))
            - log_std
        )
        log_pi_normal = tf.reduce_sum(log_pi_normal, axis=-1, keepdims=True)

        squash_correction = tf.reduce_sum(
            tf.math.log(1.0 - tf.tanh(z) ** 2 + 1e-6),
            axis=-1, keepdims=True,
        )

        half_width_safe = tf.maximum(half_width, 0.1)
        log_pi = log_pi_normal - tf.math.log(half_width_safe) - squash_correction

        action = center + half_width * tf.tanh(z)
        return action, log_pi

    def get_deterministic_action(self, obs):
        mu, _ = self._distribution_params(obs)
        base_obs = obs[:, :BASE_OBS_DIM]
        p_lo_norm = base_obs[:, self._P_LO_IDX:self._P_LO_IDX + 1]
        p_hi_norm = base_obs[:, self._P_HI_IDX:self._P_HI_IDX + 1]
        center = (p_hi_norm + p_lo_norm) * 0.5
        half_width = (p_hi_norm - p_lo_norm) * 0.5
        return center + half_width * tf.tanh(mu)


class ForecastAugmentedTwinQCritic(tf.keras.Model):
    """Step-4 twin critic with a separate forecast branch."""

    _BASE_LMP_START = 1
    _BASE_LMP_END = 26
    _BASE_SCALAR_IDX = [0] + list(range(26, 38))

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        future_len: int = 168,
        future_channels: int = 4,
        future_hidden_dim: int = 128,
    ):
        super().__init__()
        self.future_len = int(future_len)
        self.future_channels = int(future_channels)

        # Q1
        self.q1_lmp_conv_short = tf.keras.layers.Conv1D(32, 4, padding="causal", activation="relu")
        self.q1_lmp_conv_long = tf.keras.layers.Conv1D(32, 8, padding="causal", activation="relu")
        self.q1_lmp_pool_short = tf.keras.layers.GlobalAveragePooling1D()
        self.q1_lmp_pool_long = tf.keras.layers.GlobalAveragePooling1D()
        self.q1_lmp_dense = tf.keras.layers.Dense(32, activation="relu")
        self.q1_scalar = tf.keras.layers.Dense(32, activation="relu")
        self.q1_future_conv_short = tf.keras.layers.Conv1D(32, 6, padding="causal", activation="relu")
        self.q1_future_conv_long = tf.keras.layers.Conv1D(32, 12, padding="causal", activation="relu")
        self.q1_future_pool_short = tf.keras.layers.GlobalAveragePooling1D()
        self.q1_future_pool_long = tf.keras.layers.GlobalAveragePooling1D()
        self.q1_future_dense = tf.keras.layers.Dense(future_hidden_dim, activation="relu")
        self.q1_joint_ln = tf.keras.layers.LayerNormalization()
        self.q1_fc1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.q1_fc2 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.q1_out = tf.keras.layers.Dense(1)

        # Q2
        self.q2_lmp_conv_short = tf.keras.layers.Conv1D(32, 4, padding="causal", activation="relu")
        self.q2_lmp_conv_long = tf.keras.layers.Conv1D(32, 8, padding="causal", activation="relu")
        self.q2_lmp_pool_short = tf.keras.layers.GlobalAveragePooling1D()
        self.q2_lmp_pool_long = tf.keras.layers.GlobalAveragePooling1D()
        self.q2_lmp_dense = tf.keras.layers.Dense(32, activation="relu")
        self.q2_scalar = tf.keras.layers.Dense(32, activation="relu")
        self.q2_future_conv_short = tf.keras.layers.Conv1D(32, 6, padding="causal", activation="relu")
        self.q2_future_conv_long = tf.keras.layers.Conv1D(32, 12, padding="causal", activation="relu")
        self.q2_future_pool_short = tf.keras.layers.GlobalAveragePooling1D()
        self.q2_future_pool_long = tf.keras.layers.GlobalAveragePooling1D()
        self.q2_future_dense = tf.keras.layers.Dense(future_hidden_dim, activation="relu")
        self.q2_joint_ln = tf.keras.layers.LayerNormalization()
        self.q2_fc1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.q2_fc2 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.q2_out = tf.keras.layers.Dense(1)

        self._scalar_idx_tf = tf.constant(self._BASE_SCALAR_IDX, dtype=tf.int32)

    def _split_obs(self, obs: tf.Tensor):
        base_obs = obs[:, :BASE_OBS_DIM]
        future_flat = obs[:, BASE_OBS_DIM:]
        off = self.future_len
        channels = [
            future_flat[:, 0:off],
            future_flat[:, off:2 * off],
            future_flat[:, 2 * off:3 * off],
        ]
        if self.future_channels == 4:
            channels.append(future_flat[:, 3 * off:4 * off])
        future = tf.stack(channels, axis=-1)
        return base_obs, future

    def _encode_one(
        self,
        obs,
        act,
        lmp_cs, lmp_cl, lmp_ps, lmp_pl, lmp_d,
        scalar_d,
        fut_cs, fut_cl, fut_ps, fut_pl, fut_d,
        joint_ln, fc1, fc2, out,
    ):
        base_obs, future = self._split_obs(obs)

        lmp = tf.expand_dims(base_obs[:, self._BASE_LMP_START:self._BASE_LMP_END], axis=-1)
        short_e = lmp_ps(lmp_cs(lmp))
        long_e = lmp_pl(lmp_cl(lmp))
        lmp_e = lmp_d(tf.concat([short_e, long_e], axis=-1))

        scalar = tf.gather(base_obs, self._scalar_idx_tf, axis=1)
        scalar_e = scalar_d(tf.concat([scalar, act], axis=-1))

        fut_short = fut_ps(fut_cs(future))
        fut_long = fut_pl(fut_cl(future))
        fut_e = fut_d(tf.concat([fut_short, fut_long], axis=-1))

        joint = joint_ln(tf.concat([lmp_e, scalar_e, fut_e], axis=-1))
        return out(fc2(fc1(joint)))

    def call(self, inputs):
        obs, act = inputs
        q1 = self._encode_one(
            obs, act,
            self.q1_lmp_conv_short, self.q1_lmp_conv_long,
            self.q1_lmp_pool_short, self.q1_lmp_pool_long,
            self.q1_lmp_dense, self.q1_scalar,
            self.q1_future_conv_short, self.q1_future_conv_long,
            self.q1_future_pool_short, self.q1_future_pool_long,
            self.q1_future_dense,
            self.q1_joint_ln, self.q1_fc1, self.q1_fc2, self.q1_out,
        )
        q2 = self._encode_one(
            obs, act,
            self.q2_lmp_conv_short, self.q2_lmp_conv_long,
            self.q2_lmp_pool_short, self.q2_lmp_pool_long,
            self.q2_lmp_dense, self.q2_scalar,
            self.q2_future_conv_short, self.q2_future_conv_long,
            self.q2_future_pool_short, self.q2_future_pool_long,
            self.q2_future_dense,
            self.q2_joint_ln, self.q2_fc1, self.q2_fc2, self.q2_out,
        )
        return q1, q2


# ================================================================
# FORECAST-AUGMENTED SAC AGENT
# ================================================================

class ForecastAugmentedSACAgent:
    """Step-4 SAC agent with a forecast branch in actor and critics."""

    def __init__(self, obs_dim: int, act_dim: int, hp: dict):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = hp["gamma"]
        self.tau = hp["tau"]
        self.batch_size = hp["batch_size"]

        hidden_dim = hp["hidden_dim"]
        future_hidden_dim = int(hp.get("future_hidden_dim", 128))
        future_len = int(hp["forecast_horizon"])
        future_channels = int(hp["forecast_feature_channels"])

        self.actor = ForecastAugmentedActor(
            obs_dim, act_dim, hidden_dim,
            future_len=future_len,
            future_channels=future_channels,
            future_hidden_dim=future_hidden_dim,
        )
        self.critic = ForecastAugmentedTwinQCritic(
            obs_dim, act_dim, hidden_dim,
            future_len=future_len,
            future_channels=future_channels,
            future_hidden_dim=future_hidden_dim,
        )
        self.critic_target = ForecastAugmentedTwinQCritic(
            obs_dim, act_dim, hidden_dim,
            future_len=future_len,
            future_channels=future_channels,
            future_hidden_dim=future_hidden_dim,
        )

        _dummy_obs = tf.zeros((1, obs_dim), dtype=tf.float32)
        _dummy_act = tf.zeros((1, act_dim), dtype=tf.float32)
        self.actor(_dummy_obs)
        self.critic([_dummy_obs, _dummy_act])
        self.critic_target([_dummy_obs, _dummy_act])

        self._tf_act_stoch = tf.function(lambda obs: self.actor(obs), reduce_retracing=True)
        self._tf_act_det = tf.function(
            lambda obs: self.actor.get_deterministic_action(obs),
            reduce_retracing=True,
        )

        self.critic_target.set_weights(self.critic.get_weights())

        lr = hp["lr"]
        alpha_lr = float(hp.get("alpha_lr", lr))
        self.actor_opt = tf.keras.optimizers.Adam(lr, global_clipnorm=2.0)
        self.critic_opt = tf.keras.optimizers.Adam(lr, global_clipnorm=2.0)
        self.alpha_opt = tf.keras.optimizers.Adam(alpha_lr, global_clipnorm=0.5)

        _init_alpha = float(hp.get("init_alpha", 0.01))
        self.log_alpha = tf.Variable(math.log(_init_alpha), trainable=True, dtype=tf.float32)
        self.target_entropy = tf.constant(
            hp["target_entropy_scale"] * float(act_dim), dtype=tf.float32
        )
        self._log_alpha_min = tf.constant(math.log(float(hp.get("alpha_min", 0.0005))), dtype=tf.float32)
        self._log_alpha_max = tf.constant(math.log(float(hp.get("alpha_max", 1.0))), dtype=tf.float32)

        for _opt, _vars in (
            (self.actor_opt, self.actor.trainable_variables),
            (self.critic_opt, self.critic.trainable_variables),
            (self.alpha_opt, [self.log_alpha]),
        ):
            try:
                _opt.build(_vars)
            except Exception:
                pass

        self.proj_lambda = tf.constant(hp.get("proj_lambda", 0.1), dtype=tf.float32)
        self.n_step = int(hp.get("n_step", 1))
        self.gamma_n = tf.constant(self.gamma ** self.n_step, dtype=tf.float32)

    @property
    def alpha(self) -> tf.Tensor:
        return tf.exp(self.log_alpha)

    @tf.function
    def _update_critic(
        self,
        obs: tf.Tensor,
        act: tf.Tensor,
        rew: tf.Tensor,
        next_obs: tf.Tensor,
        done: tf.Tensor,
        is_weights: tf.Tensor,
    ):
        next_act, next_log_pi = self.actor(next_obs)
        next_log_pi = tf.clip_by_value(next_log_pi, -20.0, 2.0)

        q1_next, q2_next = self.critic_target([next_obs, next_act])
        min_q_next = tf.minimum(q1_next, q2_next)
        q_target = tf.stop_gradient(
            tf.clip_by_value(
                rew + self.gamma_n * (1.0 - done) * (min_q_next - self.alpha * next_log_pi),
                -100.0, 100.0,
            )
        )

        with tf.GradientTape() as tape:
            q1, q2 = self.critic([obs, act])
            per_sample_loss = 0.5 * is_weights * (
                tf.square(q1 - q_target) + tf.square(q2 - q_target)
            )
            critic_loss = tf.reduce_mean(per_sample_loss)

        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))

        td_errors = tf.squeeze(
            0.5 * (tf.abs(q1 - q_target) + tf.abs(q2 - q_target)),
            axis=1,
        )
        return critic_loss, td_errors

    @tf.function
    def _update_actor(self, obs: tf.Tensor):
        with tf.GradientTape() as tape:
            act, log_pi = self.actor(obs)
            log_pi = tf.clip_by_value(log_pi, -20.0, 2.0)
            q1, q2 = self.critic([obs, act])
            min_q = tf.minimum(q1, q2)
            proj_pen = tf.constant(0.0)
            actor_loss = tf.reduce_mean(self.alpha * log_pi - min_q)

        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))
        return actor_loss, proj_pen

    @tf.function
    def _update_alpha(self, obs: tf.Tensor):
        with tf.GradientTape() as tape:
            _, log_pi = self.actor(obs)
            log_pi = tf.clip_by_value(log_pi, -20.0, 0.0)
            alpha_loss = -tf.reduce_mean(
                self.log_alpha * tf.stop_gradient(log_pi + self.target_entropy)
            )

        grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_opt.apply_gradients(zip(grads, [self.log_alpha]))
        self.log_alpha.assign(
            tf.clip_by_value(self.log_alpha, self._log_alpha_min, self._log_alpha_max)
        )
        return alpha_loss

    def _soft_update_target(self) -> None:
        tau = self.tau
        for target_var, online_var in zip(
            self.critic_target.variables, self.critic.variables
        ):
            target_var.assign(tau * online_var + (1.0 - tau) * target_var)

    def train_step(self, buffer: step4.ReplayBuffer) -> dict:
        (obs, act, rew, next_obs, done,
         _bounds, _next_bounds,
         sample_indices, is_weights) = buffer.sample(self.batch_size)

        obs_tf = obs.astype(np.float32)
        act_tf = act.astype(np.float32)
        rew_tf = rew.astype(np.float32)
        next_obs_tf = next_obs.astype(np.float32)
        done_tf = done.astype(np.float32)
        is_weights_tf = is_weights.astype(np.float32)

        critic_loss, td_errors = self._update_critic(
            obs_tf, act_tf, rew_tf, next_obs_tf, done_tf, is_weights_tf
        )
        actor_loss, proj_pen = self._update_actor(obs_tf)
        alpha_loss = self._update_alpha(obs_tf)
        self._soft_update_target()

        buffer.update_priorities(sample_indices, td_errors.numpy())
        return {
            "critic_loss": float(critic_loss),
            "actor_loss": float(actor_loss),
            "alpha_loss": float(alpha_loss),
            "alpha": float(self.alpha),
            "proj_pen": float(proj_pen),
        }

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_tf = obs[np.newaxis].astype(np.float32)
        if deterministic:
            act_tf = self._tf_act_det(obs_tf)
        else:
            act_tf, _ = self._tf_act_stoch(obs_tf)
        return act_tf.numpy()[0]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.actor.save_weights(path + "_actor.weights.h5")
        self.critic.save_weights(path + "_critic.weights.h5")
        np.save(path + "_log_alpha.npy", self.log_alpha.numpy())

    def load(self, path: str) -> None:
        self.actor.load_weights(path + "_actor.weights.h5")
        self.critic.load_weights(path + "_critic.weights.h5")
        self.critic_target.set_weights(self.critic.get_weights())
        self.log_alpha.assign(float(np.load(path + "_log_alpha.npy")))


# ================================================================
# EVALUATION / REPORTING
# ================================================================

def run_audited_evaluation(
    agent: ForecastAugmentedSACAgent,
    env: ForecastAugmentedEnv,
    starts: np.ndarray | None = None,
    audit_max_episodes: int = 3,
    audit_max_steps: int = 12,
) -> tuple[dict, list[dict]]:
    if starts is None:
        starts = env.test_starts

    returns = []
    soc_ends = []
    soc_mins = []
    revenues = []
    deg_costs = []
    n_discharge = []
    n_charge = []
    n_idle = []

    audit_records = []

    for ep_idx, start_t in enumerate(starts, start=1):
        obs, _ = env.reset(options={"start_t": int(start_t)})
        ep_return = 0.0
        ep_rev = 0.0
        ep_deg = 0.0
        ep_n_dis = 0
        ep_n_ch = 0
        ep_n_idle = 0
        ep_soc_min = env.soc

        for step_idx in range(env.EPS_LEN):
            forecast_meta = env.get_last_forecast_meta()
            act = np.clip(agent.act(obs, deterministic=True), -1.0, 1.0)
            obs, rew, terminated, truncated, info = env.step(act)

            ep_return += rew
            ep_rev += info["revenue_$"]
            ep_deg += info["deg_cost_$"]
            ep_soc_min = min(ep_soc_min, info["soc_mwh"])

            p_safe = info["p_safe_mw"]
            if p_safe > 0.5:
                ep_n_dis += 1
            elif p_safe < -0.5:
                ep_n_ch += 1
            else:
                ep_n_idle += 1

            if (
                forecast_meta is not None
                and ep_idx <= audit_max_episodes
                and step_idx < audit_max_steps
            ):
                audit_records.append({
                    "episode_index": int(ep_idx),
                    "start_t": int(start_t),
                    "step_index": int(step_idx),
                    "global_hour": int(forecast_meta["global_hour"]),
                    "remaining_horizon": int(forecast_meta["remaining_horizon"]),
                    "exclude_lo": forecast_meta["exclude_lo"],
                    "exclude_hi": forecast_meta["exclude_hi"],
                    "neighbor_hours": forecast_meta["neighbor_hours"],
                    "neighbor_distances": forecast_meta["neighbor_distances"],
                    "neighbor_weights": forecast_meta["neighbor_weights"],
                    "train_end": int(forecast_meta["train_end"]),
                })

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

    returns = np.array(returns)
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
    }
    return metrics, audit_records


def print_step6_summary(log: dict, final_eval: dict, hp: dict, env: ForecastAugmentedEnv) -> None:
    selection_label = log.get("selection_split", "test")
    eval_means = [e["eval_mean"] for e in log["eval_returns"]]
    best_idx = int(np.argmax(eval_means)) if eval_means else -1
    oracle_eval = log.get("oracle_eval") or {}
    oracle_mean = oracle_eval.get("oracle_eval_mean")
    best_capture = (
        eval_means[best_idx] / oracle_mean
        if best_idx >= 0 and oracle_mean is not None and oracle_mean > 1e-8
        else np.nan
    )
    final_capture = (
        final_eval["eval_mean"] / oracle_mean
        if oracle_mean is not None and oracle_mean > 1e-8
        else np.nan
    )

    print("\n" + "=" * 65)
    print("NC-SafeRL  STEP 6  FORECAST-AUGMENTED SAC SUMMARY")
    print("=" * 65)
    print(f"  Observation dim      : {env.observation_space.shape[0]}")
    print(f"  Base obs dim         : {BASE_OBS_DIM}")
    print(f"  Forecast horizon     : {env.future_horizon}")
    print(f"  Forecast channels    : {env.future_channels}")
    print(f"  K neighbors          : {hp['forecast_k_neighbors']}")
    print(f"  Horizon mask         : {hp['forecast_use_mask']}")
    print("")
    if best_idx >= 0:
        print(f"  Best {selection_label} mean : {eval_means[best_idx]:+.4f} "
              f"(ep {log['eval_returns'][best_idx]['episode']})")
        if oracle_mean is not None:
            print(f"  Best oracle capture  : {best_capture * 100:.1f}%")
    print(f"  Final test mean      : {final_eval['eval_mean']:+.4f}")
    print(f"  Final test std       : {final_eval['eval_std']:.4f}")
    print(f"  Final test min/max   : {final_eval['eval_min']:+.4f} / {final_eval['eval_max']:+.4f}")
    if oracle_mean is not None:
        print(f"  Final oracle capture : {final_capture * 100:.1f}%")
    print(f"  Profit mean          : ${final_eval['profit_mean']:.0f}")
    print(f"  Revenue mean         : ${final_eval['revenue_mean']:.0f}")
    print(f"  Deg cost mean        : ${final_eval['deg_cost_mean']:.0f}")
    print(f"  SoC end / min mean   : {final_eval['soc_end_mean']:.1f} / {final_eval['soc_min_mean']:.1f} MWh")
    print(f"  Action mix           : dis={final_eval['frac_discharge']:.0%} "
          f"ch={final_eval['frac_charge']:.0%} idle={final_eval['frac_idle']:.0%}")
    print("=" * 65)


def save_step6_outputs(config: dict, log: dict, final_eval: dict, audit_records: list[dict]) -> None:
    os.makedirs(STEP6_FINAL_DIR, exist_ok=True)

    metrics_path = os.path.join(STEP6_FINAL_DIR, "forecast_sac_metrics.json")
    audit_path = os.path.join(STEP6_FINAL_DIR, "forecast_audit.npy")
    log_path = os.path.join(STEP6_FINAL_DIR, "training_log.npy")

    audit_summary = {
        "n_records": int(len(audit_records)),
        "all_neighbors_in_train": bool(all(
            all(h < rec["train_end"] for h in rec["neighbor_hours"])
            for rec in audit_records
        )),
    }

    payload = {
        "config": config,
        "final_eval": final_eval,
        "oracle_eval": log.get("oracle_eval", {}),
        "run_id": log.get("run_id"),
        "audit_summary": audit_summary,
    }

    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    np.save(audit_path, audit_records, allow_pickle=True)
    np.save(log_path, log, allow_pickle=True)

    print(f"\nSaved Step 6 metrics -> {metrics_path}")
    print(f"Saved Step 6 audit   -> {audit_path}")
    print(f"Saved Step 6 log     -> {log_path}")
    if audit_records:
        print(f"Forecast audit: all neighbors in training block = {audit_summary['all_neighbors_in_train']}")


# ================================================================
# CLI
# ================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 6 forecast-augmented SAC baseline.")
    parser.add_argument("--train_episodes", type=int, default=HP["train_episodes"])
    parser.add_argument("--eval_freq", type=int, default=HP["eval_freq"])
    parser.add_argument("--lp_warmstart_steps", type=int, default=HP["lp_warmstart_steps"])
    parser.add_argument("--lp_bc_steps", type=int, default=HP["lp_bc_steps"])
    parser.add_argument("--batch_size", type=int, default=HP["batch_size"])
    parser.add_argument("--min_buffer_fill", type=int, default=HP["min_buffer_fill"])
    parser.add_argument("--hidden_dim", type=int, default=HP["hidden_dim"])
    parser.add_argument("--future_hidden_dim", type=int, default=HP["future_hidden_dim"])
    parser.add_argument("--k_neighbors", type=int, default=HP["forecast_k_neighbors"])
    parser.add_argument("--max_test_starts", type=int, default=None,
                        help="Optional limit for evaluation starts (useful for smoke tests).")
    parser.add_argument("--audit_episodes", type=int, default=3)
    parser.add_argument("--audit_steps", type=int, default=12)
    parser.add_argument("--skip_ptdf_analysis", action="store_true")
    return parser.parse_args()


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    args = parse_args()
    os.makedirs(STEP6_WORKSPACE_DIR, exist_ok=True)
    os.makedirs(STEP6_FINAL_DIR, exist_ok=True)

    print("\n" + "=" * 65)
    print("NC-SafeRL  STEP 6: Forecast-Augmented SAC")
    print("=" * 65)

    base_env = BESSEnv(seed=HP["seed"], validation_days=HP.get("validation_days", 30))
    if args.max_test_starts is not None:
        base_env.test_starts = base_env.test_starts[:args.max_test_starts]

    if not args.skip_ptdf_analysis:
        ptdf_binding_analysis(base_env, save_dir=STEP6_WORKSPACE_DIR)

    forecaster = AuditedAnalogForecaster(
        env=base_env,
        horizon=FORECAST_HORIZON_HOURS,
        k_neighbors=args.k_neighbors,
        eps=HP["forecast_eps"],
    )

    env = ForecastAugmentedEnv(
        base_env=base_env,
        forecaster=forecaster,
        include_horizon_mask=HP["forecast_use_mask"],
        exclude_current_train_episode=HP["forecast_exclude_current_train_episode"],
        future_horizon=FORECAST_HORIZON_HOURS,
    )

    hp = dict(HP)
    hp.update({
        "train_episodes": int(args.train_episodes),
        "eval_freq": int(args.eval_freq),
        "lp_warmstart_steps": int(args.lp_warmstart_steps),
        "lp_bc_steps": int(args.lp_bc_steps),
        "batch_size": int(args.batch_size),
        "min_buffer_fill": int(args.min_buffer_fill),
        "hidden_dim": int(args.hidden_dim),
        "future_hidden_dim": int(args.future_hidden_dim),
        "forecast_k_neighbors": int(args.k_neighbors),
        "forecast_horizon": int(env.future_horizon),
        "forecast_feature_channels": int(env.future_channels),
    })

    print(f"\nForecast-augmented observation:")
    print(f"  Base dim:           {BASE_OBS_DIM}")
    print(f"  Forecast horizon:   {env.future_horizon}")
    print(f"  Forecast channels:  {env.future_channels} (LMP, p_lo, p_hi, mask)")
    print(f"  Total obs dim:      {env.observation_space.shape[0]}")
    print(f"  Train-only anchors: {len(forecaster.anchor_hours)}")
    print(f"  Exclude current training episode anchors: {env.exclude_current_train_episode}")

    # Reuse the Step-4 training/evaluation scaffold with the new agent.
    step4.SACAgent = ForecastAugmentedSACAgent

    t0 = time.time()
    agent, log = step4.train_sac(env, hp)
    total_time = time.time() - t0
    print(f"\nTotal wall-clock time: {total_time / 60:.1f} min")

    # Reload the best checkpoint saved by the shared Step-4 training loop.
    run_dir = os.path.join(STEP6_WORKSPACE_DIR, "runs", log["run_id"])
    best_prefix = os.path.join(run_dir, "checkpoints", "best", "agent")
    if os.path.exists(best_prefix + "_actor.weights.h5"):
        agent.load(best_prefix)
        print(f"\nLoaded best checkpoint -> {best_prefix}")

    final_path = os.path.join(STEP6_FINAL_DIR, "agent")
    agent.save(final_path)
    print(f"\nFinal agent saved -> {final_path}")

    # Rebuild forecaster with full pre-test library (train + val) for test evaluation.
    # During training the forecaster was restricted to train-only anchors to prevent
    # val-split leakage into observations.  At test time the agent weights are frozen,
    # so enriching the KNN library matches real deployment conditions (all history
    # before the test block is available) and is consistent with run_seed_sweep.py.
    _saved_fte = base_env.forecast_train_end
    base_env.forecast_train_end = base_env.test_start_hour
    eval_forecaster = AuditedAnalogForecaster(
        env=base_env,
        horizon=FORECAST_HORIZON_HOURS,
        k_neighbors=args.k_neighbors,
        eps=HP["forecast_eps"],
    )
    base_env.forecast_train_end = _saved_fte  # restore for any subsequent use
    eval_env = ForecastAugmentedEnv(
        base_env=base_env,
        forecaster=eval_forecaster,
        include_horizon_mask=HP["forecast_use_mask"],
        exclude_current_train_episode=False,  # no training loop; exclusion not needed
        future_horizon=FORECAST_HORIZON_HOURS,
    )

    final_eval, audit_records = run_audited_evaluation(
        agent,
        eval_env,
        starts=base_env.test_starts,
        audit_max_episodes=args.audit_episodes,
        audit_max_steps=args.audit_steps,
    )

    config = {
        "hp": hp,
        "forecast_horizon": env.future_horizon,
        "forecast_channels": env.future_channels,
        "base_obs_dim": BASE_OBS_DIM,
        "total_obs_dim": int(env.observation_space.shape[0]),
        "n_val_starts": int(len(env.val_starts)),
        "n_test_starts": int(len(env.test_starts)),
        "workspace_dir": STEP6_WORKSPACE_DIR,
    }

    print_step6_summary(log, final_eval, hp, env)
    save_step6_outputs(config, log, final_eval, audit_records)

    print("\nStep 6 complete.")


if __name__ == "__main__":
    main()
