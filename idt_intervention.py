"""
IDT Intervention Wrapper
========================
Provides five intervention knobs for correcting/mitigating perturbations:

Wrappers (filtering):
1. Observation smoothing - exponential moving average on observations
2. Action smoothing - exponential moving average on actions
3. Action clipping - limits action magnitude

Holds (probabilistic skip):
4. Observation hold probability - chance to reuse previous observation
5. Action hold probability - chance to reuse previous action

Usage:
    from idt_intervention import IDTInterventionWrapper

    env = IDTInterventionWrapper(
        env,
        obs_smoothing=0.3,   # 0 = off, 1 = max smoothing
        act_smoothing=0.3,   # 0 = off, 1 = max smoothing
        act_clip=0.8,        # 1 = no clipping, 0.5 = clip to 50% of max
        obs_hold_prob=0.1,   # 0 = normal, 0.1 = 10% chance to hold
        act_hold_prob=0.1,   # 0 = normal, 0.1 = 10% chance to hold
        enabled=False,       # Start disabled
    )

    # Enable intervention at episode 60
    env.set_enabled(True)

    # Dynamically adjust holds
    env.set_obs_hold_prob(0.2)
    env.set_act_hold_prob(0.15)
"""

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


class IDTInterventionWrapper(VecEnvWrapper):
    """
    VecEnv wrapper that applies intervention corrections.

    Five knobs:

    WRAPPERS (filtering):
    - obs_smoothing: Exponential moving average on observations (0-1)
        0 = no smoothing (raw obs)
        1 = max smoothing (very slow response)

    - act_smoothing: Exponential moving average on actions (0-1)
        0 = no smoothing (raw actions)
        1 = max smoothing (very slow response)

    - act_clip: Action magnitude scaling (0-1)
        1 = no clipping (full action range)
        0.5 = clip to 50% of action space

    HOLDS (probabilistic skip):
    - obs_hold_prob: Probability to reuse previous observation (0-1)
        0 = normal (fresh obs every step)
        0.1 = 10% chance to hold previous obs

    - act_hold_prob: Probability to reuse previous action (0-1)
        0 = normal (fresh action every step)
        0.1 = 10% chance to hold previous action

    Tracks regime:
    - 0: intervention off
    - 1: intervention active
    """

    def __init__(
            self,
            venv,
            obs_smoothing: float = 0.0,
            act_smoothing: float = 0.0,
            act_clip: float = 1.0,
            obs_hold_prob: float = 0.0,
            act_hold_prob: float = 0.0,
            enabled: bool = True,
    ):
        super().__init__(venv)

        # Validate parameters
        assert 0.0 <= obs_smoothing <= 1.0, "obs_smoothing must be in [0, 1]"
        assert 0.0 <= act_smoothing <= 1.0, "act_smoothing must be in [0, 1]"
        assert 0.0 < act_clip <= 1.0, "act_clip must be in (0, 1]"
        assert 0.0 <= obs_hold_prob <= 1.0, "obs_hold_prob must be in [0, 1]"
        assert 0.0 <= act_hold_prob <= 1.0, "act_hold_prob must be in [0, 1]"

        # Wrapper parameters
        self.obs_smoothing = obs_smoothing
        self.act_smoothing = act_smoothing
        self.act_clip = act_clip

        # Hold parameters (probability-based)
        self.obs_hold_prob = obs_hold_prob
        self.act_hold_prob = act_hold_prob

        self.enabled = enabled

        # State for smoothing (exponential moving average)
        self.obs_ema = None
        self.act_ema = None

        # State for holds (store previous values)
        self.held_obs = None
        self.held_action = None

        # Regime tracking
        self.regime = np.zeros(self.num_envs, dtype=int)

        # Convert smoothing parameter to EMA alpha
        # smoothing=0 -> alpha=1 (no smoothing, instant response)
        # smoothing=1 -> alpha=0.05 (heavy smoothing, slow response)
        self.obs_alpha = 1.0 - (obs_smoothing * 0.95)
        self.act_alpha = 1.0 - (act_smoothing * 0.95)

        print(f"IDTIntervention: obs_smooth={obs_smoothing:.2f} (α={self.obs_alpha:.2f}), "
              f"act_smooth={act_smoothing:.2f} (α={self.act_alpha:.2f}), "
              f"act_clip={act_clip:.2f}, "
              f"obs_hold_prob={obs_hold_prob:.2f}, act_hold_prob={act_hold_prob:.2f}")

    def set_enabled(self, enabled: bool):
        """Enable or disable intervention."""
        self.enabled = enabled
        if enabled:
            self.regime[:] = 1
        else:
            self.regime[:] = 0

    def set_obs_smoothing(self, value: float):
        """Update observation smoothing (0-1)."""
        assert 0.0 <= value <= 1.0
        self.obs_smoothing = value
        self.obs_alpha = 1.0 - (value * 0.95)

    def set_act_smoothing(self, value: float):
        """Update action smoothing (0-1)."""
        assert 0.0 <= value <= 1.0
        self.act_smoothing = value
        self.act_alpha = 1.0 - (value * 0.95)

    def set_act_clip(self, value: float):
        """Update action clipping (0-1)."""
        assert 0.0 < value <= 1.0
        self.act_clip = value

    def set_obs_hold_prob(self, value: float):
        """Update observation hold probability."""
        assert 0.0 <= value <= 1.0
        self.obs_hold_prob = value

    def set_act_hold_prob(self, value: float):
        """Update action hold probability."""
        assert 0.0 <= value <= 1.0
        self.act_hold_prob = value

    def _smooth_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply exponential moving average to observations."""
        if not self.enabled or self.obs_smoothing == 0:
            return obs

        if self.obs_ema is None:
            self.obs_ema = obs.copy()
        else:
            # EMA: new_ema = alpha * obs + (1 - alpha) * old_ema
            self.obs_ema = self.obs_alpha * obs + (1 - self.obs_alpha) * self.obs_ema

        return self.obs_ema.copy()

    def _hold_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply probabilistic observation hold."""
        if not self.enabled or self.obs_hold_prob == 0:
            self.held_obs = obs.copy()  # Always store latest
            return obs

        # If no held obs yet, store current
        if self.held_obs is None:
            self.held_obs = obs.copy()
            return obs

        # Probabilistically decide: hold or update
        if np.random.random() < self.obs_hold_prob:
            # Hold: return previous obs
            return self.held_obs.copy()
        else:
            # Update: store new obs and return it
            self.held_obs = obs.copy()
            return obs

    def _smooth_action(self, actions: np.ndarray) -> np.ndarray:
        """Apply exponential moving average to actions."""
        if not self.enabled or self.act_smoothing == 0:
            return actions

        if self.act_ema is None:
            self.act_ema = actions.copy()
        else:
            # EMA: new_ema = alpha * action + (1 - alpha) * old_ema
            self.act_ema = self.act_alpha * actions + (1 - self.act_alpha) * self.act_ema

        return self.act_ema.copy()

    def _clip_action(self, actions: np.ndarray) -> np.ndarray:
        """Clip action magnitude."""
        if not self.enabled or self.act_clip == 1.0:
            return actions

        # Scale actions by clip factor
        return actions * self.act_clip

    def _hold_action(self, actions: np.ndarray) -> np.ndarray:
        """Apply probabilistic action hold."""
        if not self.enabled or self.act_hold_prob == 0:
            self.held_action = actions.copy()  # Always store latest
            return actions

        # If no held action yet, store current
        if self.held_action is None:
            self.held_action = actions.copy()
            return actions

        # Probabilistically decide: hold or update
        if np.random.random() < self.act_hold_prob:
            # Hold: return previous action
            return self.held_action.copy()
        else:
            # Update: store new action and return it
            self.held_action = actions.copy()
            return actions

    def reset(self):
        """Reset environment and intervention state."""
        obs = self.venv.reset()

        # Reset all state
        self.obs_ema = None
        self.act_ema = None
        self.held_obs = None
        self.held_action = None

        # Apply observation interventions to reset obs
        obs = self._smooth_obs(obs)
        obs = self._hold_obs(obs)

        return obs

    def step_async(self, actions):
        """Process actions before sending to environment."""
        # Apply action interventions (order: hold -> smooth -> clip)
        actions = self._hold_action(actions)
        actions = self._smooth_action(actions)
        actions = self._clip_action(actions)

        self.venv.step_async(actions)

    def step_wait(self):
        """Process observations after receiving from environment."""
        obs, rewards, dones, infos = self.venv.step_wait()

        # Reset state for done environments
        for i, done in enumerate(dones):
            if done:
                self.obs_ema = None
                self.act_ema = None
                self.held_obs = None
                self.held_action = None

        # Apply observation interventions (order: smooth -> hold)
        obs = self._smooth_obs(obs)
        obs = self._hold_obs(obs)

        # Update regime based on enabled state
        if self.enabled:
            self.regime[:] = 1
        else:
            self.regime[:] = 0

        # Add intervention info to infos
        for i, info in enumerate(infos):
            if isinstance(info, dict):
                info["intervention_regime"] = int(self.regime[i])
                info["obs_smoothing"] = self.obs_smoothing if self.enabled else 0.0
                info["act_smoothing"] = self.act_smoothing if self.enabled else 0.0
                info["act_clip"] = self.act_clip if self.enabled else 1.0
                info["obs_hold_prob"] = self.obs_hold_prob if self.enabled else 0.0
                info["act_hold_prob"] = self.act_hold_prob if self.enabled else 0.0

        return obs, rewards, dones, infos