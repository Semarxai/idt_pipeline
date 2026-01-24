"""
HalfCheetah PPO/SAC Training and Evaluation
============================================
Research-grade implementation with:
- Corrected PPO hyperparameters
- Multi-seed support
- Noise injection for both PPO and SAC
- Full trajectory logging
- IDT Intervention (obs/action smoothing, action clipping)

All models saved with "claude" prefix for identification.
"""

import os
import csv
import gymnasium as gym
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, List

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnvWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from idt_intervention import IDTInterventionWrapper

# =============================================================================
# TRAINING LOGGER CALLBACK
# =============================================================================

class TrainingLoggerCallback(BaseCallback):
    """
    Logs episode rewards during training to a CSV file.
    Columns: timestep, episode, reward, length
    """

    def __init__(self, log_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.file = None
        self.writer = None

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.file = open(self.log_path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timestep", "episode", "reward", "length"])
        print(f"Training log: {self.log_path}")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1
                reward = info["episode"]["r"]
                length = info["episode"]["l"]
                self.writer.writerow([
                    self.num_timesteps,
                    self.episode_count,
                    reward,
                    length
                ])
                self.file.flush()
                if self.verbose > 0 and self.episode_count % 100 == 0:
                    print(f"Episode {self.episode_count}: reward={reward:.1f}")
        return True

    def _on_training_end(self) -> None:
        if self.file:
            self.file.close()
        print(f"Training log saved: {self.log_path}")


class FullTrajectoryLoggerCallback(BaseCallback):
    """
    Logs EVERY step during training to a CSV file.
    Columns: timestep, episode, t, s_0...s_n, a_0...a_m, reward, done, s_next_0...s_next_n

    WARNING: This creates very large files and slows training significantly.
    """

    def __init__(self, log_path: str, obs_dim: int, act_dim: int, verbose: int = 1):
        super().__init__(verbose)
        self.log_path = log_path
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.file = None
        self.writer = None
        self.episode_count = 0
        self.step_in_episode = 0
        self.last_obs = None

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.file = open(self.log_path, "w", newline="")
        self.writer = csv.writer(self.file)

        # Build header
        header = ["timestep", "episode", "t"]
        header += [f"s_{i}" for i in range(self.obs_dim)]
        header += [f"a_{i}" for i in range(self.act_dim)]
        header += ["reward", "done"]
        header += [f"s_next_{i}" for i in range(self.obs_dim)]
        self.writer.writerow(header)

        print(f"Full trajectory log: {self.log_path}")
        print(f"WARNING: This will create a large file and slow training.")

    def _on_step(self) -> bool:
        # Get current data from locals
        obs = self.locals.get("obs_tensor")
        if obs is None:
            obs = self.locals.get("new_obs")

        new_obs = self.locals.get("new_obs")
        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos", [])

        # Handle tensor conversion
        if hasattr(obs, 'cpu'):
            obs = obs.cpu().numpy()
        if hasattr(new_obs, 'cpu'):
            new_obs = new_obs.cpu().numpy()
        if hasattr(actions, 'cpu'):
            actions = actions.cpu().numpy()
        if hasattr(rewards, 'cpu'):
            rewards = rewards.cpu().numpy()
        if hasattr(dones, 'cpu'):
            dones = dones.cpu().numpy()

        # Ensure arrays
        obs = np.atleast_2d(obs)
        new_obs = np.atleast_2d(new_obs)
        actions = np.atleast_2d(actions)
        rewards = np.atleast_1d(rewards)
        dones = np.atleast_1d(dones)

        # Log each environment (typically just 1)
        for i in range(len(rewards)):
            row = [self.num_timesteps, self.episode_count, self.step_in_episode]
            row.extend(obs[i].flatten().tolist())
            row.extend(actions[i].flatten().tolist())
            row.append(float(rewards[i]))
            row.append(bool(dones[i]))
            row.extend(new_obs[i].flatten().tolist())
            self.writer.writerow(row)

            self.step_in_episode += 1

            # Check for episode end
            if dones[i]:
                self.episode_count += 1
                self.step_in_episode = 0
                if self.verbose > 0 and self.episode_count % 100 == 0:
                    print(f"Episode {self.episode_count} logged")

        # Flush periodically
        if self.num_timesteps % 1000 == 0:
            self.file.flush()

        return True

    def _on_training_end(self) -> None:
        if self.file:
            self.file.close()
        print(f"Full trajectory log saved: {self.log_path}")
        print(f"Total episodes: {self.episode_count}")


def get_training_log_dir(algorithm: str) -> str:
    """Get training log directory"""
    return f"training_logs_{algorithm.lower()}_claude"


# =============================================================================
# CORRECTED PPO HYPERPARAMETERS
# =============================================================================

PPO_HYPERPARAMS = {
    "learning_rate": 2.5e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 5,  # KEY FIX: reduced from 20
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005,  # KEY FIX: increased from 0.0004
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

PPO_POLICY_KWARGS = {
    "activation_fn": nn.ReLU,
    "net_arch": {"pi": [256, 256], "vf": [256, 256]},
    "ortho_init": True,
    "log_std_init": -1,  # KEY FIX: increased from -2
}


# =============================================================================
# FILE PATH FUNCTIONS (with claude naming)
# =============================================================================

def get_model_path(algorithm: str, seed: int) -> str:
    """Get model path: ppo_claude_seed0.zip or sac_claude_seed0.zip"""
    return f"{algorithm.lower()}_claude_seed{seed}.zip"


def get_stats_path(algorithm: str, seed: int) -> str:
    """Get normalization stats path"""
    return f"vec_normalize_{algorithm.lower()}_claude_seed{seed}.pkl"


def get_checkpoint_dir(algorithm: str, seed: int) -> str:
    """Get checkpoint directory"""
    return f"checkpoints_{algorithm.lower()}_claude_seed{seed}"


def get_log_dir(algorithm: str) -> str:
    """Get log directory"""
    return f"logs_{algorithm.lower()}_claude"


# =============================================================================
# ENVIRONMENT WRAPPERS
# =============================================================================

class NoisyEnvWrapper(VecEnvWrapper):
    """
    VecEnv wrapper that injects observation or action noise after a given
    step index in each episode.

    - noise_type: "none", "obs", or "act"
    - noise_level: standard deviation of Gaussian noise (in normalized units)
    - perturb_start_step: timestep in each episode when noise turns on

    Tracks a "regime" flag:
    - 0: noise inactive
    - 1: noise active
    """

    def __init__(
            self,
            venv,
            noise_type: str = "none",
            noise_level: float = 0.0,
            perturb_start_step: int = 0,
    ):
        super().__init__(venv)
        assert noise_type in ("none", "obs", "act"), "Invalid noise_type"
        self.noise_type = noise_type
        self.noise_level = float(noise_level)
        self.perturb_start_step = int(perturb_start_step)
        self.step_in_episode = np.zeros(self.num_envs, dtype=int)
        self.regime = np.zeros(self.num_envs, dtype=int)
        self.enabled = True  # Can be toggled per episode

    def set_enabled(self, enabled: bool):
        """Enable or disable noise injection."""
        self.enabled = enabled

    def _noise_active_mask(self) -> np.ndarray:
        if not self.enabled:
            return np.zeros(self.num_envs, dtype=bool)
        return self.step_in_episode >= self.perturb_start_step

    def reset(self):
        self.step_in_episode[:] = 0
        self.regime[:] = 0
        obs = self.venv.reset()
        return obs

    def step_async(self, actions):
        if self.noise_type == "act" and self.noise_level > 0.0:
            noisy_actions = np.array(actions, copy=True)
            active = self._noise_active_mask()
            if active.any():
                noise = np.random.normal(
                    loc=0.0,
                    scale=self.noise_level,
                    size=noisy_actions.shape,
                )
                noisy_actions[active] += noise[active]
                if isinstance(self.action_space, gym.spaces.Box):
                    low = self.action_space.low
                    high = self.action_space.high
                    noisy_actions = np.clip(noisy_actions, low, high)
            self.venv.step_async(noisy_actions)
            return
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.step_in_episode += 1

        for i, done in enumerate(dones):
            if done:
                self.step_in_episode[i] = 0
                self.regime[i] = 0

        active = self._noise_active_mask()
        self.regime[active] = 1
        self.regime[~active] = 0

        # Add observation noise if configured
        if self.noise_type == "obs" and self.noise_level > 0.0:
            if self.num_envs == 1:
                if active[0]:
                    noise = np.random.normal(
                        loc=0.0,
                        scale=self.noise_level,
                        size=obs.shape,
                    )
                    obs = obs + noise
            else:
                if active.any():
                    noise = np.random.normal(
                        loc=0.0,
                        scale=self.noise_level,
                        size=obs.shape,
                    )
                    obs[active] = obs[active] + noise[active]

        # Add regime info to infos
        for i, info in enumerate(infos):
            if isinstance(info, dict):
                info["regime"] = int(self.regime[i])
                info["noise_type"] = self.noise_type
                info["noise_level"] = self.noise_level

        return obs, rewards, dones, infos


class ForceInjectionWrapper(VecEnvWrapper):
    """
    VecEnv wrapper that injects external forces to MuJoCo bodies.

    - body_name: "torso", "ffoot" (front foot/tip), "bfoot" (back foot)
    - force_magnitude: force in Newtons (can be negative)
    - force_direction: "x", "y", "z" or combo like "xz"
    - injection_interval: apply force every N steps
    - injection_duration: how many steps to apply force

    Tracks regime:
    - 0: no force
    - 1: force active
    """

    # HalfCheetah body names
    BODY_NAMES = ["torso", "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]

    def __init__(
            self,
            venv,
            body_name: str = "torso",
            force_magnitude: float = 10.0,
            force_direction: str = "x",
            injection_interval: int = 100,
            injection_duration: int = 10,
            enabled: bool = True,
    ):
        super().__init__(venv)

        assert body_name in self.BODY_NAMES, f"body_name must be one of {self.BODY_NAMES}"

        self.body_name = body_name
        self.force_magnitude = force_magnitude
        self.force_direction = force_direction
        self.injection_interval = injection_interval
        self.injection_duration = injection_duration
        self.enabled = enabled

        self.step_in_episode = np.zeros(self.num_envs, dtype=int)
        self.regime = np.zeros(self.num_envs, dtype=int)

        # Get body ID from MuJoCo model
        self.body_id = None
        self._init_body_id()

    def _init_body_id(self):
        """Get body ID from underlying environment."""
        try:
            # Access the underlying gym environment
            env = self.venv.envs[0] if hasattr(self.venv, 'envs') else self.venv
            while hasattr(env, 'env'):
                env = env.env

            if hasattr(env, 'model'):
                self.body_id = env.model.body(self.body_name).id
                print(f"ForceInjection: body '{self.body_name}' has ID {self.body_id}")
        except Exception as e:
            print(f"Warning: Could not get body ID for '{self.body_name}': {e}")
            self.body_id = 1  # Default to torso (usually ID 1)

    def set_enabled(self, enabled: bool):
        """Enable or disable force injection."""
        self.enabled = enabled

    def _get_force_vector(self) -> np.ndarray:
        """Create 6D force vector [fx, fy, fz, tx, ty, tz]."""
        force = np.zeros(6)

        # Apply force based on direction
        if "x" in self.force_direction:
            force[0] = self.force_magnitude
        if "y" in self.force_direction:
            force[1] = self.force_magnitude
        if "z" in self.force_direction:
            force[2] = self.force_magnitude

        return force

    def _should_apply_force(self, step: int) -> bool:
        """Check if force should be active at this step."""
        if not self.enabled:
            return False

        # Check if we're in an injection window
        cycle_position = step % self.injection_interval
        return cycle_position < self.injection_duration

    def _apply_force(self, env, apply: bool):
        """Apply or remove force from MuJoCo simulation."""
        try:
            # Navigate to base env
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env

            if hasattr(base_env, 'data') and self.body_id is not None:
                if apply:
                    base_env.data.xfrc_applied[self.body_id] = self._get_force_vector()
                else:
                    base_env.data.xfrc_applied[self.body_id] = np.zeros(6)
        except Exception as e:
            pass  # Silently fail if can't apply force

    def reset(self):
        self.step_in_episode[:] = 0
        self.regime[:] = 0
        obs = self.venv.reset()
        return obs

    def step_async(self, actions):
        # Apply force before step
        for i in range(self.num_envs):
            step = self.step_in_episode[i]
            should_apply = self._should_apply_force(step)

            env = self.venv.envs[i] if hasattr(self.venv, 'envs') else self.venv
            self._apply_force(env, should_apply)
            self.regime[i] = 1 if should_apply else 0

        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.step_in_episode += 1

        # Reset step counter on episode end
        for i, done in enumerate(dones):
            if done:
                self.step_in_episode[i] = 0
                self.regime[i] = 0
                # Clear force on reset
                env = self.venv.envs[i] if hasattr(self.venv, 'envs') else self.venv
                self._apply_force(env, False)

        # Add regime info to infos
        for i, info in enumerate(infos):
            if isinstance(info, dict):
                info["force_regime"] = int(self.regime[i])
                info["force_body"] = self.body_name
                info["force_magnitude"] = self.force_magnitude if self.regime[i] else 0.0

        return obs, rewards, dones, infos


class GravityPerturbationWrapper(VecEnvWrapper):
    """
    VecEnv wrapper that modifies gravity in MuJoCo simulation.

    - gravity_scale: 1.0 = normal (-9.81), 1.2 = 20% heavier, 0.8 = 20% lighter
    - enabled: toggle on/off

    Tracks regime:
    - 0: normal gravity
    - 1: perturbed gravity
    """

    DEFAULT_GRAVITY = -9.81

    def __init__(
            self,
            venv,
            gravity_scale: float = 1.0,
            enabled: bool = True,
    ):
        super().__init__(venv)

        self.gravity_scale = gravity_scale
        self.enabled = enabled
        self.regime = np.zeros(self.num_envs, dtype=int)

        # Store original gravity
        self.original_gravity = self.DEFAULT_GRAVITY
        self._init_gravity()

    def _init_gravity(self):
        """Get original gravity from environment."""
        try:
            env = self.venv.envs[0] if hasattr(self.venv, 'envs') else self.venv
            while hasattr(env, 'env'):
                env = env.env

            if hasattr(env, 'model'):
                self.original_gravity = env.model.opt.gravity[2]
                print(f"GravityPerturbation: original gravity = {self.original_gravity}")
        except Exception as e:
            print(f"Warning: Could not get gravity: {e}")

    def set_enabled(self, enabled: bool):
        """Enable or disable gravity perturbation."""
        self.enabled = enabled
        self._apply_gravity()

    def _apply_gravity(self):
        """Apply gravity modification to all environments."""
        try:
            for i in range(self.num_envs):
                env = self.venv.envs[i] if hasattr(self.venv, 'envs') else self.venv
                while hasattr(env, 'env'):
                    env = env.env

                if hasattr(env, 'model'):
                    if self.enabled:
                        env.model.opt.gravity[2] = self.original_gravity * self.gravity_scale
                        self.regime[i] = 1
                    else:
                        env.model.opt.gravity[2] = self.original_gravity
                        self.regime[i] = 0
        except Exception as e:
            print(f"Warning: Could not apply gravity: {e}")

    def reset(self):
        obs = self.venv.reset()
        self._apply_gravity()
        return obs

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        # Re-apply gravity after any reset
        for i, done in enumerate(dones):
            if done:
                self._apply_gravity()

        # Add regime info to infos
        for i, info in enumerate(infos):
            if isinstance(info, dict):
                info["gravity_regime"] = int(self.regime[i])
                info["gravity_scale"] = self.gravity_scale if self.enabled else 1.0

        return obs, rewards, dones, infos


# =============================================================================
# PPO FUNCTIONS
# =============================================================================

def train_ppo(
        seed: int = 0,
        total_timesteps: int = 5_000_000,
        checkpoint_freq: int = 500_000,
        log_training: str = "none",  # "none", "episode", or "full"
) -> Tuple[PPO, Optional[str]]:
    """
    Train a PPO model from scratch with corrected hyperparameters.

    Args:
        seed: Random seed
        total_timesteps: Training steps
        checkpoint_freq: Save checkpoint every N steps
        log_training: "none", "episode" (rewards only), or "full" (all 44 columns per step)

    Saves:
    - Model to ppo_claude_seed{seed}.zip
    - VecNormalize stats to vec_normalize_ppo_claude_seed{seed}.pkl
    - Checkpoints to checkpoints_ppo_claude_seed{seed}/
    - Training log (if enabled) to training_logs_ppo_claude/

    Returns:
        (model, training_log_path or None)
    """
    model_path = get_model_path("ppo", seed)
    stats_path = get_stats_path("ppo", seed)
    checkpoint_dir = get_checkpoint_dir("ppo", seed)

    print("=" * 60)
    print(f"Training PPO - Seed {seed}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Log training: {log_training}")
    print("=" * 60)
    print(f"Key settings: n_epochs={PPO_HYPERPARAMS['n_epochs']}, "
          f"ent_coef={PPO_HYPERPARAMS['ent_coef']}")
    print(f"Model will be saved to: {model_path}")
    print("=" * 60)

    # Create environment
    def make_env():
        env = gym.make("HalfCheetah-v4")
        env.reset(seed=seed)
        env = Monitor(env)  # Tracks episode stats for logging
        return env

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    # Create model
    model = PPO(
        "MlpPolicy",
        vec_env,
        **PPO_HYPERPARAMS,
        policy_kwargs=PPO_POLICY_KWARGS,
        verbose=1,
        seed=seed,
        device="mps",

    )

    # Setup callbacks
    callbacks = []

    # Checkpointing
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix=f"ppo_claude_seed{seed}",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Training logger
    training_log_path = None
    if log_training == "episode":
        log_dir = get_training_log_dir("ppo")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_log_path = os.path.join(log_dir, f"ppo_seed{seed}_episodes_{timestamp}.csv")
        training_logger = TrainingLoggerCallback(training_log_path)
        callbacks.append(training_logger)
    elif log_training == "full":
        log_dir = get_training_log_dir("ppo")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_log_path = os.path.join(log_dir, f"ppo_seed{seed}_full_{timestamp}.csv")
        obs_dim = vec_env.observation_space.shape[0]
        act_dim = vec_env.action_space.shape[0]
        training_logger = FullTrajectoryLoggerCallback(training_log_path, obs_dim, act_dim)
        callbacks.append(training_logger)

    # Train
    print(f"\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    print("Training finished.")

    # Save
    model.save(model_path)
    vec_env.save(stats_path)
    print(f"Model saved to: {model_path}")
    print(f"Stats saved to: {stats_path}")

    vec_env.close()
    return model, training_log_path


def continue_training_ppo(
        seed: int = 0,
        total_timesteps: int = 1_000_000,
        log_training: str = "none",  # "none", "episode", or "full"
) -> Tuple[PPO, Optional[str]]:
    """
    Continue training an existing PPO model.
    If no model exists, trains from scratch.

    Args:
        log_training: "none", "episode" (rewards only), or "full" (all 44 columns per step)

    Returns:
        (model, training_log_path or None)
    """
    model_path = get_model_path("ppo", seed)
    stats_path = get_stats_path("ppo", seed)

    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        print(f"No existing model for seed {seed}. Training from scratch...")
        return train_ppo(seed=seed, total_timesteps=total_timesteps, log_training=log_training)

    print("=" * 60)
    print(f"Continuing PPO training - Seed {seed}")
    print(f"Additional timesteps: {total_timesteps:,}")
    print(f"Log training: {log_training}")
    print("=" * 60)

    # Create environment
    def make_env():
        env = gym.make("HalfCheetah-v4")
        env.reset(seed=seed)
        env = Monitor(env)  # Tracks episode stats for logging
        return env

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(stats_path, vec_env)
    vec_env.training = True
    vec_env.norm_reward = True

    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, env=vec_env)

    # Override with corrected hyperparameters
    model.n_epochs = PPO_HYPERPARAMS["n_epochs"]
    model.ent_coef = PPO_HYPERPARAMS["ent_coef"]

    # Setup callbacks
    callbacks = []
    training_log_path = None
    if log_training == "episode":
        log_dir = get_training_log_dir("ppo")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_log_path = os.path.join(log_dir, f"ppo_seed{seed}_continue_episodes_{timestamp}.csv")
        training_logger = TrainingLoggerCallback(training_log_path)
        callbacks.append(training_logger)
    elif log_training == "full":
        log_dir = get_training_log_dir("ppo")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_log_path = os.path.join(log_dir, f"ppo_seed{seed}_continue_full_{timestamp}.csv")
        obs_dim = vec_env.observation_space.shape[0]
        act_dim = vec_env.action_space.shape[0]
        training_logger = FullTrajectoryLoggerCallback(training_log_path, obs_dim, act_dim)
        callbacks.append(training_logger)

    # Train
    print(f"\nContinuing training...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks if callbacks else None, progress_bar=True)
    print("Training finished.")

    # Save
    model.save(model_path)
    vec_env.save(stats_path)
    print(f"Model saved to: {model_path}")

    vec_env.close()
    return model, training_log_path


def evaluate_ppo(
        seed: int = 0,
        num_episodes: int = 50,
) -> Tuple[List[float], float]:
    """
    Evaluate PPO model without logging or noise.

    Returns:
        (list of episode rewards, average reward)
    """
    model_path = get_model_path("ppo", seed)
    stats_path = get_stats_path("ppo", seed)

    print(f"Loading PPO model from '{model_path}'...")
    model = PPO.load(model_path)

    # Create environment
    def make_env():
        env = gym.make("HalfCheetah-v4", render_mode=None)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    # Evaluate
    all_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]

        all_rewards.append(total_reward)
        print(f"Episode {i + 1}: reward = {total_reward:.2f}")

    avg_reward = np.mean(all_rewards)
    print("-" * 50)
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")

    env.close()
    return all_rewards, avg_reward


def evaluate_and_log_ppo(
        seed: int = 0,
        num_episodes: int = 50,
        noise_type: str = "none",
        noise_level: float = 0.0,
        perturb_start_step: int = 0,
        run_name: Optional[str] = None,
        # Force injection parameters
        force_enabled: bool = False,
        force_body: str = "torso",
        force_magnitude: float = 10.0,
        force_direction: str = "x",
        force_interval: int = 100,
        force_duration: int = 10,
        # Gravity perturbation parameters
        gravity_enabled: bool = False,
        gravity_scale: float = 1.0,
        # Episode-based perturbation start
        perturb_start_episode: int = 0,
) -> Tuple[List[float], float, str]:
    """
    Evaluate PPO model with optional noise/force/gravity and log trajectories to CSV.

    Args:
        seed: Model seed
        num_episodes: Number of evaluation episodes
        noise_type: "none", "obs", or "act"
        noise_level: Noise standard deviation
        perturb_start_step: When to start noise within episode
        run_name: Custom name for log file
        force_enabled: Enable force injection
        force_body: Body to apply force ("torso", "ffoot", "bfoot", etc.)
        force_magnitude: Force in Newtons
        force_direction: Direction ("x", "y", "z", or combo)
        force_interval: Apply force every N steps
        force_duration: Duration of force application in steps
        gravity_enabled: Enable gravity perturbation
        gravity_scale: Gravity multiplier (1.0 = normal, 1.2 = 20% heavier)
        perturb_start_episode: Episode number when perturbations begin (0 = from start)

    Returns:
        (list of episode rewards, average reward, path to log file)
    """
    model_path = get_model_path("ppo", seed)
    stats_path = get_stats_path("ppo", seed)
    log_dir = get_log_dir("ppo")

    os.makedirs(log_dir, exist_ok=True)

    # Generate descriptive filename
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ep_suffix = f"_from_ep{perturb_start_episode}" if perturb_start_episode > 0 else ""
        if gravity_enabled:
            gravity_pct = int(gravity_scale * 100)
            run_name = f"ppo_seed{seed}_gravity_{gravity_pct}_percent{ep_suffix}_{timestamp}"
        elif force_enabled:
            run_name = f"ppo_seed{seed}_force_{force_body}_{int(force_magnitude)}N{ep_suffix}_{timestamp}"
        elif noise_type != "none":
            noise_pct = int(noise_level * 100)
            run_name = f"ppo_seed{seed}_{noise_type}_noise_{noise_pct:02d}_percent{ep_suffix}_{timestamp}"
        else:
            run_name = f"ppo_seed{seed}_no_noise_{timestamp}"

    log_path = os.path.join(log_dir, f"{run_name}.csv")

    print("=" * 60)
    print(f"Evaluating PPO - Seed {seed}")
    print(f"Noise: {noise_type}, Level: {noise_level}, Start step: {perturb_start_step}")
    if force_enabled:
        print(
            f"Force: {force_magnitude}N on {force_body} ({force_direction}), every {force_interval} steps for {force_duration} steps")
    if gravity_enabled:
        print(f"Gravity: {gravity_scale}x ({gravity_scale * 100:.0f}% of normal)")
    if perturb_start_episode > 0:
        print(f"Perturbations start at episode {perturb_start_episode}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)

    # Load model
    model = PPO.load(model_path)

    # Create environment
    def make_env():
        env = gym.make("HalfCheetah-v4", render_mode=None)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    # Wrap with noise (initially disabled if perturb_start_episode > 0)
    noise_wrapper = NoisyEnvWrapper(
        env,
        noise_type=noise_type,
        noise_level=noise_level,
        perturb_start_step=perturb_start_step,
    )
    noise_wrapper.set_enabled(perturb_start_episode == 0 and noise_type != "none")

    # Wrap with force injection (initially disabled if perturb_start_episode > 0)
    force_wrapper = ForceInjectionWrapper(
        noise_wrapper,
        body_name=force_body,
        force_magnitude=force_magnitude,
        force_direction=force_direction,
        injection_interval=force_interval,
        injection_duration=force_duration,
        enabled=(perturb_start_episode == 0 and force_enabled),
    )

    # Wrap with gravity perturbation (initially disabled if perturb_start_episode > 0)
    gravity_wrapper = GravityPerturbationWrapper(
        force_wrapper,
        gravity_scale=gravity_scale,
        enabled=(perturb_start_episode == 0 and gravity_enabled),
    )

    env = gravity_wrapper  # Final wrapped env

    # Get dimensions
    obs = env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # CSV header - add force_regime and gravity_regime columns
    header = (
            ["episode", "t", "regime", "force_regime", "gravity_regime"]
            + [f"s_{i}" for i in range(obs_dim)]
            + [f"a_{i}" for i in range(act_dim)]
            + ["reward", "done"]
            + [f"s_next_{i}" for i in range(obs_dim)]
    )

    episode_returns = []

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ep in range(num_episodes):
            # Enable/disable perturbations based on episode number
            if ep == perturb_start_episode and perturb_start_episode > 0:
                print(f"--- Enabling perturbations at episode {ep} ---")
                if noise_type != "none":
                    noise_wrapper.set_enabled(True)
                if force_enabled:
                    force_wrapper.set_enabled(True)
                if gravity_enabled:
                    gravity_wrapper.set_enabled(True)

            obs = env.reset()
            done = False
            ep_return = 0.0
            t = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, dones, infos = env.step(action)

                r = float(reward[0])
                done = bool(dones[0])
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                regime = int(info.get("regime", 0))
                force_regime = int(info.get("force_regime", 0))
                gravity_regime = int(info.get("gravity_regime", 0))

                row = (
                        [ep, t, regime, force_regime, gravity_regime]
                        + obs[0].tolist()
                        + action[0].tolist()
                        + [r, done]
                        + next_obs[0].tolist()
                )
                writer.writerow(row)

                ep_return += r
                obs = next_obs
                t += 1

            episode_returns.append(ep_return)
            perturb_status = "🔴" if (ep >= perturb_start_episode and (
                        force_enabled or noise_type != "none" or gravity_enabled)) else "🟢"
            print(f"[PPO] Episode {ep + 1}: reward = {ep_return:.2f} {perturb_status}")

    avg_return = float(np.mean(episode_returns))
    print("-" * 50)
    print(f"[PPO] Average reward: {avg_return:.2f}")
    print(f"Log saved to: {log_path}")

    env.close()
    return episode_returns, avg_return, log_path


# =============================================================================
# SAC FUNCTIONS
# =============================================================================

def train_sac(
        seed: int = 0,
        total_timesteps: int = 5_000_000,
        checkpoint_freq: int = 500_000,
        log_training: str = "none",  # "none", "episode", or "full"
) -> Tuple[SAC, Optional[str]]:
    """
    Train a SAC model from scratch.

    Args:
        log_training: "none", "episode" (rewards only), or "full" (all 44 columns per step)

    Saves:
    - Model to sac_claude_seed{seed}.zip
    - VecNormalize stats to vec_normalize_sac_claude_seed{seed}.pkl
    - Training log (if enabled) to training_logs_sac_claude/

    Returns:
        (model, training_log_path or None)
    """
    model_path = get_model_path("sac", seed)
    stats_path = get_stats_path("sac", seed)
    checkpoint_dir = get_checkpoint_dir("sac", seed)

    print("=" * 60)
    print(f"Training SAC - Seed {seed}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Log training: {log_training}")
    print(f"Model will be saved to: {model_path}")
    print("=" * 60)

    # Create environment
    def make_env():
        env = gym.make("HalfCheetah-v4")
        env.reset(seed=seed)
        env = Monitor(env)  # Tracks episode stats for logging
        return env

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    # Create model
    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10_000,
        ent_coef="auto",
        target_update_interval=1,
        verbose=1,
        seed=seed,
        device="mps",
    )

    # Setup callbacks
    callbacks = []

    # Checkpointing
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix=f"sac_claude_seed{seed}",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Training logger
    training_log_path = None
    if log_training == "episode":
        log_dir = get_training_log_dir("sac")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_log_path = os.path.join(log_dir, f"sac_seed{seed}_episodes_{timestamp}.csv")
        training_logger = TrainingLoggerCallback(training_log_path)
        callbacks.append(training_logger)
    elif log_training == "full":
        log_dir = get_training_log_dir("sac")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_log_path = os.path.join(log_dir, f"sac_seed{seed}_full_{timestamp}.csv")
        obs_dim = vec_env.observation_space.shape[0]
        act_dim = vec_env.action_space.shape[0]
        training_logger = FullTrajectoryLoggerCallback(training_log_path, obs_dim, act_dim)
        callbacks.append(training_logger)

    # Train
    print(f"\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    print("Training finished.")

    # Save
    model.save(model_path)
    vec_env.save(stats_path)
    print(f"Model saved to: {model_path}")
    print(f"Stats saved to: {stats_path}")

    vec_env.close()
    return model, training_log_path


def continue_training_sac(
        seed: int = 0,
        total_timesteps: int = 1_000_000,
        log_training: str = "none",  # "none", "episode", or "full"
) -> Tuple[SAC, Optional[str]]:
    """
    Continue training an existing SAC model.
    If no model exists, trains from scratch.

    Args:
        log_training: "none", "episode" (rewards only), or "full" (all 44 columns per step)

    Returns:
        (model, training_log_path or None)
    """
    model_path = get_model_path("sac", seed)
    stats_path = get_stats_path("sac", seed)

    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        print(f"No existing model for seed {seed}. Training from scratch...")
        return train_sac(seed=seed, total_timesteps=total_timesteps, log_training=log_training)

    print("=" * 60)
    print(f"Continuing SAC training - Seed {seed}")
    print(f"Additional timesteps: {total_timesteps:,}")
    print(f"Log training: {log_training}")
    print("=" * 60)

    # Create environment
    def make_env():
        env = gym.make("HalfCheetah-v4")
        env.reset(seed=seed)
        env = Monitor(env)  # Tracks episode stats for logging
        return env

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(stats_path, vec_env)
    vec_env.training = True

    # Load model
    print(f"Loading model from {model_path}...")
    model = SAC.load(model_path, env=vec_env)

    # Setup callbacks
    callbacks = []
    training_log_path = None
    if log_training == "episode":
        log_dir = get_training_log_dir("sac")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_log_path = os.path.join(log_dir, f"sac_seed{seed}_continue_episodes_{timestamp}.csv")
        training_logger = TrainingLoggerCallback(training_log_path)
        callbacks.append(training_logger)
    elif log_training == "full":
        log_dir = get_training_log_dir("sac")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_log_path = os.path.join(log_dir, f"sac_seed{seed}_continue_full_{timestamp}.csv")
        obs_dim = vec_env.observation_space.shape[0]
        act_dim = vec_env.action_space.shape[0]
        training_logger = FullTrajectoryLoggerCallback(training_log_path, obs_dim, act_dim)
        callbacks.append(training_logger)

    # Train
    print(f"\nContinuing training...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks if callbacks else None, progress_bar=True)
    print("Training finished.")

    # Save
    model.save(model_path)
    vec_env.save(stats_path)
    print(f"Model saved to: {model_path}")

    vec_env.close()
    return model, training_log_path


def evaluate_sac(
        seed: int = 0,
        num_episodes: int = 50,
) -> Tuple[List[float], float]:
    """
    Evaluate SAC model without logging or noise.

    Returns:
        (list of episode rewards, average reward)
    """
    model_path = get_model_path("sac", seed)
    stats_path = get_stats_path("sac", seed)

    print(f"Loading SAC model from '{model_path}'...")
    model = SAC.load(model_path)

    # Create environment
    def make_env():
        env = gym.make("HalfCheetah-v4", render_mode=None)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    # Evaluate
    all_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]

        all_rewards.append(total_reward)
        print(f"[SAC] Episode {i + 1}: reward = {total_reward:.2f}")

    avg_reward = np.mean(all_rewards)
    print("-" * 50)
    print(f"[SAC] Average reward over {num_episodes} episodes: {avg_reward:.2f}")

    env.close()
    return all_rewards, avg_reward


def evaluate_and_log_sac(
        seed: int = 0,
        num_episodes: int = 50,
        noise_type: str = "none",
        noise_level: float = 0.0,
        perturb_start_step: int = 0,
        run_name: Optional[str] = None,
        # Force injection parameters
        force_enabled: bool = False,
        force_body: str = "torso",
        force_magnitude: float = 10.0,
        force_direction: str = "x",
        force_interval: int = 100,
        force_duration: int = 10,
        # Gravity perturbation parameters
        gravity_enabled: bool = False,
        gravity_scale: float = 1.0,
        # Episode-based perturbation start
        perturb_start_episode: int = 0,
        # IDT Intervention parameters
        intervention_enabled: bool = False,
        obs_smoothing: float = 0.0,
        act_smoothing: float = 0.0,
        act_clip: float = 1.0,
        obs_hold_prob: float = 0.0,
        act_hold_prob: float = 0.0,
        correction_start_episode: int = 0,
) -> Tuple[List[float], float, str]:
    """
    Evaluate SAC model with optional noise/force/gravity and IDT intervention.

    Three-phase evaluation:
    - Phase 1 (ep 0 to perturb_start_episode): Baseline
    - Phase 2 (perturb_start_episode to correction_start_episode): Perturbed
    - Phase 3 (correction_start_episode onwards): Corrected (intervention active)

    Args:
        seed: Model seed
        num_episodes: Number of evaluation episodes
        noise_type: "none", "obs", or "act"
        noise_level: Noise standard deviation
        perturb_start_step: When to start noise within episode
        run_name: Custom name for log file
        force_enabled: Enable force injection
        force_body: Body to apply force ("torso", "ffoot", "bfoot", etc.)
        force_magnitude: Force in Newtons
        force_direction: Direction ("x", "y", "z", or combo)
        force_interval: Apply force every N steps
        force_duration: Duration of force application in steps
        gravity_enabled: Enable gravity perturbation
        gravity_scale: Gravity multiplier (1.0 = normal, 1.2 = 20% heavier)
        perturb_start_episode: Episode number when perturbations begin (0 = from start)
        intervention_enabled: Enable IDT intervention
        obs_smoothing: Observation smoothing intensity (0-1)
        act_smoothing: Action smoothing intensity (0-1)
        act_clip: Action clipping scale (0-1, 1=no clip)
        obs_hold_prob: Observation hold probability (0-1), 0=normal
        act_hold_prob: Action hold probability (0-1), 0=normal
        correction_start_episode: Episode when intervention begins

    Returns:
        (list of episode rewards, average reward, path to log file)
    """
    model_path = get_model_path("sac", seed)
    stats_path = get_stats_path("sac", seed)
    log_dir = get_log_dir("sac")

    os.makedirs(log_dir, exist_ok=True)

    # Generate descriptive filename
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ep_suffix = f"_from_ep{perturb_start_episode}" if perturb_start_episode > 0 else ""
        corr_suffix = f"_corr_ep{correction_start_episode}" if intervention_enabled and correction_start_episode > 0 else ""
        if gravity_enabled:
            gravity_pct = int(gravity_scale * 100)
            run_name = f"sac_seed{seed}_gravity_{gravity_pct}_percent{ep_suffix}{corr_suffix}_{timestamp}"
        elif force_enabled:
            run_name = f"sac_seed{seed}_force_{force_body}_{int(force_magnitude)}N{ep_suffix}{corr_suffix}_{timestamp}"
        elif noise_type != "none":
            noise_pct = int(noise_level * 100)
            run_name = f"sac_seed{seed}_{noise_type}_noise_{noise_pct:02d}_percent{ep_suffix}{corr_suffix}_{timestamp}"
        else:
            run_name = f"sac_seed{seed}_no_noise_{timestamp}"

    log_path = os.path.join(log_dir, f"{run_name}.csv")

    print("=" * 60)
    print(f"Evaluating SAC - Seed {seed}")
    print(f"Noise: {noise_type}, Level: {noise_level}, Start step: {perturb_start_step}")
    if force_enabled:
        print(
            f"Force: {force_magnitude}N on {force_body} ({force_direction}), every {force_interval} steps for {force_duration} steps")
    if gravity_enabled:
        print(f"Gravity: {gravity_scale}x ({gravity_scale * 100:.0f}% of normal)")
    if perturb_start_episode > 0:
        print(f"Perturbations start at episode {perturb_start_episode}")
    if intervention_enabled:
        print(
            f"IDT Intervention: obs_smooth={obs_smoothing}, act_smooth={act_smoothing}, act_clip={act_clip}, obs_hold_prob={obs_hold_prob}, act_hold_prob={act_hold_prob}")
        print(f"Correction starts at episode {correction_start_episode}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)

    # Load model
    model = SAC.load(model_path)

    # Create environment
    def make_env():
        env = gym.make("HalfCheetah-v4", render_mode=None)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    # Wrap with noise (initially disabled if perturb_start_episode > 0)
    noise_wrapper = NoisyEnvWrapper(
        env,
        noise_type=noise_type,
        noise_level=noise_level,
        perturb_start_step=perturb_start_step,
    )
    noise_wrapper.set_enabled(perturb_start_episode == 0 and noise_type != "none")

    # Wrap with force injection (initially disabled if perturb_start_episode > 0)
    force_wrapper = ForceInjectionWrapper(
        noise_wrapper,
        body_name=force_body,
        force_magnitude=force_magnitude,
        force_direction=force_direction,
        injection_interval=force_interval,
        injection_duration=force_duration,
        enabled=(perturb_start_episode == 0 and force_enabled),
    )

    # Wrap with gravity perturbation (initially disabled if perturb_start_episode > 0)
    gravity_wrapper = GravityPerturbationWrapper(
        force_wrapper,
        gravity_scale=gravity_scale,
        enabled=(perturb_start_episode == 0 and gravity_enabled),
    )

    # Wrap with IDT intervention (initially disabled)
    intervention_wrapper = IDTInterventionWrapper(
        gravity_wrapper,
        obs_smoothing=obs_smoothing,
        act_smoothing=act_smoothing,
        act_clip=act_clip,
        obs_hold_prob=obs_hold_prob,
        act_hold_prob=act_hold_prob,
        enabled=False,  # Always start disabled, enable at correction_start_episode
    )

    env = intervention_wrapper  # Final wrapped env

    # Get dimensions
    obs = env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # CSV header - include intervention_regime
    header = (
            ["episode", "t", "regime", "force_regime", "gravity_regime", "intervention_regime"]
            + [f"s_{i}" for i in range(obs_dim)]
            + [f"a_{i}" for i in range(act_dim)]
            + ["reward", "done"]
            + [f"s_next_{i}" for i in range(obs_dim)]
    )

    episode_returns = []

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ep in range(num_episodes):
            # Phase transitions
            # Enable perturbations at perturb_start_episode
            if ep == perturb_start_episode and perturb_start_episode > 0:
                print(f"--- 🔴 Enabling perturbations at episode {ep} ---")
                if noise_type != "none":
                    noise_wrapper.set_enabled(True)
                if force_enabled:
                    force_wrapper.set_enabled(True)
                if gravity_enabled:
                    gravity_wrapper.set_enabled(True)

            # Enable intervention at correction_start_episode
            if ep == correction_start_episode and intervention_enabled and correction_start_episode > 0:
                print(f"--- 🔧 Enabling IDT intervention at episode {ep} ---")
                intervention_wrapper.set_enabled(True)

            obs = env.reset()
            done = False
            ep_return = 0.0
            t = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, dones, infos = env.step(action)

                r = float(reward[0])
                done = bool(dones[0])
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                regime = int(info.get("regime", 0))
                force_regime = int(info.get("force_regime", 0))
                gravity_regime = int(info.get("gravity_regime", 0))
                intervention_regime = int(info.get("intervention_regime", 0))

                row = (
                        [ep, t, regime, force_regime, gravity_regime, intervention_regime]
                        + obs[0].tolist()
                        + action[0].tolist()
                        + [r, done]
                        + next_obs[0].tolist()
                )
                writer.writerow(row)

                ep_return += r
                obs = next_obs
                t += 1

            episode_returns.append(ep_return)

            # Status indicator
            has_perturbation = (
                        ep >= perturb_start_episode and (force_enabled or noise_type != "none" or gravity_enabled))
            has_intervention = (ep >= correction_start_episode and intervention_enabled)

            if has_intervention:
                status = "🔧"  # Corrected
            elif has_perturbation:
                status = "🔴"  # Perturbed
            else:
                status = "🟢"  # Baseline

            print(f"[SAC] Episode {ep + 1}: reward = {ep_return:.2f} {status}")

    avg_return = float(np.mean(episode_returns))
    print("-" * 50)
    print(f"[SAC] Average reward: {avg_return:.2f}")
    print(f"Log saved to: {log_path}")

    env.close()
    return episode_returns, avg_return, log_path


"""
Dynamic IDT Evaluation Function
================================
Add this function to app.py after the existing evaluate_and_log_sac function.

This version uses real-time P monitoring to dynamically control intervention.
"""


# Add this import at top of app.py:
# from idt_monitor import RealTimeIDTMonitor

"""
IDT Evaluation Function with P Monitoring
==========================================
This function ALWAYS computes P, regardless of intervention mode.

Replace the existing evaluate_and_log_sac_dynamic in app.py with this version.
"""


# Add this import at top of app.py:
# from idt_monitor import RealTimeIDTMonitor

def evaluate_and_log_sac_with_P(
        seed: int = 0,
        num_episodes: int = 50,
        noise_type: str = "none",
        noise_level: float = 0.0,
        perturb_start_step: int = 0,
        run_name: Optional[str] = None,
        # Perturbation parameters
        perturb_start_episode: int = 15,
        # Intervention mode: "off", "static", "dynamic"
        intervention_mode: str = "off",
        # IDT Intervention parameters (for static/dynamic modes)
        obs_smoothing: float = 0.3,
        act_smoothing: float = 0.0,
        act_clip: float = 1.0,
        obs_hold_prob: float = 0.0,
        act_hold_prob: float = 0.0,
        # Static mode: fixed episode to start intervention
        correction_start_episode: int = 35,
        # Dynamic control parameters
        baseline_path: str = "sac3_baseline.json",
        p_threshold_low: float = 0.28,
        p_threshold_high: float = 0.32,
        buffer_size: int = 500,
        compute_interval: int = 50,
) -> Tuple[List[float], float, str]:
    """
    Evaluate SAC with P monitoring (always computed).

    Intervention modes:
    - "off": No intervention, P computed for observation
    - "static": Intervention starts at correction_start_episode
    - "dynamic": Intervention based on P thresholds

    Args:
        seed: Model seed
        num_episodes: Number of evaluation episodes
        noise_type: "none", "obs", or "act"
        noise_level: Noise standard deviation
        perturb_start_episode: When noise begins
        intervention_mode: "off", "static", or "dynamic"
        obs_smoothing: Intervention intensity (0-1)
        correction_start_episode: For static mode - when intervention starts
        baseline_path: Path to baseline.json
        p_threshold_low: For dynamic mode - enable intervention below this P
        p_threshold_high: For dynamic mode - disable intervention above this P
        buffer_size: Rolling buffer size for P computation
        compute_interval: Compute P every N steps

    Returns:
        (episode rewards, avg reward, log path)
    """
    from idt_monitor import RealTimeIDTMonitor

    model_path = get_model_path("sac", seed)
    stats_path = get_stats_path("sac", seed)
    log_dir = get_log_dir("sac")

    os.makedirs(log_dir, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        noise_pct = int(noise_level * 100)
        smooth_pct = int(obs_smoothing * 100)
        run_name = f"sac_seed{seed}_{intervention_mode}_{noise_type}_{noise_pct}pct_low{p_threshold_low}_high{p_threshold_high}_smooth{smooth_pct}pct_{timestamp}"

    log_path = os.path.join(log_dir, f"{run_name}.csv")

    print("=" * 60)
    print(f"IDT Evaluation - Seed {seed}")
    print(f"Mode: {intervention_mode.upper()}")
    print(f"Noise: {noise_type} {noise_level * 100:.1f}% from episode {perturb_start_episode}")
    if intervention_mode == "dynamic":
        print(f"P thresholds: low={p_threshold_low}, high={p_threshold_high}")
    elif intervention_mode == "static":
        print(f"Intervention starts at episode {correction_start_episode}")
    print(f"Intervention settings: obs_smooth={obs_smoothing}, act_smooth={act_smoothing}")
    print("=" * 60)

    # Load model
    model = SAC.load(model_path)

    # Create environment
    def make_env():
        env = gym.make("HalfCheetah-v4", render_mode=None)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    # Wrap with noise (initially disabled)
    noise_wrapper = NoisyEnvWrapper(
        env,
        noise_type=noise_type,
        noise_level=noise_level,
        perturb_start_step=perturb_start_step,
    )
    noise_wrapper.set_enabled(False)

    # Wrap with IDT intervention (initially disabled)
    intervention_wrapper = IDTInterventionWrapper(
        noise_wrapper,
        obs_smoothing=obs_smoothing,
        act_smoothing=act_smoothing,
        act_clip=act_clip,
        obs_hold_prob=obs_hold_prob,
        act_hold_prob=act_hold_prob,
        enabled=False,
    )

    env = intervention_wrapper

    # Create real-time monitor (ALWAYS - for P computation)
    monitor = RealTimeIDTMonitor(
        baseline_path=baseline_path,
        buffer_size=buffer_size,
        p_threshold_low=p_threshold_low,
        p_threshold_high=p_threshold_high,
        compute_interval=compute_interval,
    )

    # Get dimensions
    obs = env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # CSV header with P column
    header = (
            ["episode", "t", "regime", "intervention_regime", "P"]
            + [f"s_{i}" for i in range(obs_dim)]
            + [f"a_{i}" for i in range(act_dim)]
            + ["reward", "done"]
            + [f"s_next_{i}" for i in range(obs_dim)]
    )

    episode_returns = []

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ep in range(num_episodes):
            # Enable noise at perturb_start_episode
            if ep == perturb_start_episode and noise_type != "none":
                print(f"--- 🔴 Episode {ep}: Enabling {noise_type} noise ---")
                noise_wrapper.set_enabled(True)

            # Static mode: enable intervention at correction_start_episode
            if intervention_mode == "static" and ep == correction_start_episode:
                print(f"--- 🔧 Episode {ep}: Enabling intervention (static mode) ---")
                intervention_wrapper.set_enabled(True)

            obs = env.reset()
            done = False
            ep_return = 0.0
            t = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, dones, infos = env.step(action)

                r = float(reward[0])
                done = bool(dones[0])
                info = infos[0] if isinstance(infos, (list, tuple)) else infos

                # Add to monitor buffer
                monitor.add(obs[0], action[0], next_obs[0])

                # ALWAYS compute P when it's time
                current_P = None
                if monitor.should_compute():
                    current_P = monitor.compute_P()

                    # Dynamic mode: adjust intervention based on P (only after perturbation)
                    if intervention_mode == "dynamic" and ep >= perturb_start_episode:
                        should_intervene = monitor.should_intervene()

                        if should_intervene and not intervention_wrapper.enabled:
                            print(f"    🔧 Step {t}: P={current_P:.3f} < {p_threshold_low} → Intervention ON")
                            intervention_wrapper.set_enabled(True)
                        elif not should_intervene and intervention_wrapper.enabled:
                            print(f"    ✅ Step {t}: P={current_P:.3f} > {p_threshold_high} → Intervention OFF")
                            intervention_wrapper.set_enabled(False)

                regime = int(info.get("regime", 0))
                intervention_regime = int(info.get("intervention_regime", 0))

                row = (
                        [ep, t, regime, intervention_regime, current_P if current_P else ""]
                        + obs[0].tolist()
                        + action[0].tolist()
                        + [r, done]
                        + next_obs[0].tolist()
                )
                writer.writerow(row)

                ep_return += r
                obs = next_obs
                t += 1

            episode_returns.append(ep_return)

            # Status indicator
            if intervention_wrapper.enabled:
                status = "🔧"
            elif ep >= perturb_start_episode:
                status = "🔴"
            else:
                status = "🟢"
            P_str = f"P={monitor.current_P:.3f}" if monitor.current_P else "P=..."
            print(f"[SAC] Episode {ep + 1}: reward = {ep_return:.2f} {status} {P_str}")

    avg_return = float(np.mean(episode_returns))
    print("-" * 50)
    print(f"[SAC] Average reward: {avg_return:.2f}")
    print(f"Log saved to: {log_path}")

    # Summary
    P_history = monitor.get_P_history()
    if P_history:
        P_values = [p for _, p in P_history]
        print(f"P range: {min(P_values):.3f} - {max(P_values):.3f}")

    env.close()
    return episode_returns, avg_return, log_path


# =============================================================================
# MAIN (for command-line usage)_old_before_adding_evaluate_dynamics
# =============================================================================

# if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HalfCheetah PPO/SAC Training")
    parser.add_argument("--algo", type=str, choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--episodes", type=int, default=50)

    args = parser.parse_args()

    if args.algo == "ppo":
        if args.mode == "train":
            train_ppo(seed=args.seed, total_timesteps=args.timesteps)
        else:
            evaluate_ppo(seed=args.seed, num_episodes=args.episodes)
    else:
        if args.mode == "train":
            train_sac(seed=args.seed, total_timesteps=args.timesteps)
        else:
            evaluate_sac(seed=args.seed, num_episodes=args.episodes)

if __name__ == "__main__":
    evaluate_and_log_sac_dynamic(
        seed=3,
        num_episodes=50,
        noise_type="obs",
        noise_level=0.03,
        perturb_start_episode=15,
        obs_smoothing=0.3,
        baseline_path="sac3_baseline.json",
        p_threshold_low=0.25,
        p_threshold_high=0.35,
    )