"""
HalfCheetah Control Panel - Streamlit UI
=========================================
Research-grade UI with:
- Seed selection (0, 1, 2)
- Noise injection for both PPO and SAC
- Full trajectory logging
- Claude-named model files
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from app import (
    train_ppo,
    continue_training_ppo,
    evaluate_ppo,
    evaluate_and_log_ppo,
    train_sac,
    continue_training_sac,
    evaluate_sac,
    evaluate_and_log_sac,
    evaluate_and_log_sac_with_P,
    get_model_path,
    get_stats_path,
)
import os

"""
Test Log Function
==================
Add this to ui.py to log all test settings and results.

1. Add the function after imports
2. Call it after evaluation completes
"""

import os
from datetime import datetime


def log_test_results(
        log_file: str,
        # Model settings
        algorithm: str,
        seed: int,
        num_episodes: int,
        # Noise settings
        noise_type: str,
        noise_level: float,
        perturb_start_episode: int,
        # Intervention settings
        intervention_mode: str,
        obs_smoothing: float,
        act_smoothing: float,
        act_clip: float,
        obs_hold_prob: float,
        act_hold_prob: float,
        correction_start_episode: int,
        # P settings
        num_bins: int,
        buffer_size: int,
        p_threshold_low: float,
        p_threshold_high: float,
        baseline_path: str,
        # Results - Rewards
        avg_reward: float,
        min_reward: float,
        max_reward: float,
        # Results - Phase Analysis
        baseline_avg: float,
        perturbed_avg: float,
        reward_change_pct: float,
        # Results - P Statistics
        p_mean: float,
        p_min: float,
        p_max: float,
        # Results - Intervention
        intervention_pct: float,
):
    """Append test results to test_log.csv"""

    test_log_path = "test_log.csv"

    # Check if file exists to write header
    write_header = not os.path.exists(test_log_path)

    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # All fields
    fields = [
        # Run info
        "timestamp", "log_file",
        # Model
        "algorithm", "seed", "num_episodes",
        # Noise
        "noise_type", "noise_level", "perturb_start_episode",
        # Intervention
        "intervention_mode", "obs_smoothing", "act_smoothing", "act_clip",
        "obs_hold_prob", "act_hold_prob", "correction_start_episode",
        # P settings
        "num_bins", "buffer_size", "p_threshold_low", "p_threshold_high", "baseline_path",
        # Results - Rewards
        "avg_reward", "min_reward", "max_reward",
        # Results - Phase Analysis
        "baseline_avg", "perturbed_avg", "reward_change_pct",
        # Results - P Statistics
        "p_mean", "p_min", "p_max",
        # Results - Intervention
        "intervention_pct",
    ]

    values = [
        # Run info
        timestamp, log_file,
        # Model
        algorithm, seed, num_episodes,
        # Noise
        noise_type, noise_level, perturb_start_episode,
        # Intervention
        intervention_mode, obs_smoothing, act_smoothing, act_clip,
        obs_hold_prob, act_hold_prob, correction_start_episode,
        # P settings
        num_bins, buffer_size, p_threshold_low, p_threshold_high, baseline_path,
        # Results - Rewards
        round(avg_reward, 2), round(min_reward, 2), round(max_reward, 2),
        # Results - Phase Analysis
        round(baseline_avg, 2), round(perturbed_avg, 2), round(reward_change_pct, 2),
        # Results - P Statistics
        round(p_mean, 4), round(p_min, 4), round(p_max, 4),
        # Results - Intervention
        round(intervention_pct, 2),
    ]

    with open(test_log_path, "a") as f:
        if write_header:
            f.write(",".join(fields) + "\n")
        f.write(",".join(str(v) for v in values) + "\n")

    print(f"Test logged to: {test_log_path}")


# =============================================================================
# WHERE TO ADD IN ui.py:
# =============================================================================
#
# After the P Statistics section (around line 700-720), add:
#
#                         # Log test results
#                         log_test_results(
#                             log_file=log_path,
#                             algorithm=algorithm,
#                             seed=seed,
#                             num_episodes=int(num_episodes),
#                             noise_type=noise_type_param,
#                             noise_level=float(noise_level),
#                             perturb_start_episode=int(perturb_start_episode),
#                             intervention_mode=intervention_mode,
#                             obs_smoothing=float(obs_smoothing),
#                             act_smoothing=float(act_smoothing),
#                             act_clip=float(act_clip),
#                             obs_hold_prob=float(obs_hold_prob),
#                             act_hold_prob=float(act_hold_prob),
#                             correction_start_episode=int(correction_start_episode),
#                             num_bins=num_bins,
#                             buffer_size=int(buffer_size),
#                             p_threshold_low=float(p_threshold_low),
#                             p_threshold_high=float(p_threshold_high),
#                             baseline_path=baseline_path,
#                             avg_reward=avg_return,
#                             min_reward=min(episode_returns),
#                             max_reward=max(episode_returns),
#                             baseline_avg=baseline_avg if perturb_start_episode > 0 else avg_return,
#                             perturbed_avg=perturbed_avg if perturb_start_episode > 0 else avg_return,
#                             reward_change_pct=pct_change if perturb_start_episode > 0 else 0.0,
#                             p_mean=df_p['P'].mean(),
#                             p_min=df_p['P'].min(),
#                             p_max=df_p['P'].max(),
#                             intervention_pct=intervention_pct,
#                         )


st.title("🐆 HalfCheetah Control Panel")
st.caption("Research-grade PPO/SAC training and evaluation")

# =============================================================================
# SIDEBAR - Algorithm and Seed Selection
# =============================================================================

st.sidebar.header("Configuration")

algorithm = st.sidebar.selectbox(
    "Algorithm",
    ["PPO", "SAC"],
    help="PPO uses corrected hyperparameters (n_epochs=5, ent_coef=0.005)"
)

seed = st.sidebar.selectbox(
    "Seed",
    [0, 1, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15],
    index=3,  # Default to seed 3
    help="Random seed for reproducibility."
)

st.sidebar.markdown("---")

# Show model status
model_path = get_model_path(algorithm, seed)
stats_path = get_stats_path(algorithm, seed)
model_exists = os.path.exists(model_path) and os.path.exists(stats_path)

if model_exists:
    st.sidebar.success(f"✅ {algorithm} seed {seed} model exists")
else:
    st.sidebar.warning(f"⚠️ {algorithm} seed {seed} not trained yet")

st.sidebar.markdown("---")
st.sidebar.markdown("**File naming:**")
st.sidebar.code(f"{algorithm.lower()}_claude_seed{seed}.zip")

# =============================================================================
# MAIN PANEL - Tabs
# =============================================================================

tab_train, tab_eval = st.tabs(["🏋️ Training", "📊 Evaluation"])

# =============================================================================
# TRAINING TAB
# =============================================================================

with tab_train:
    st.header(f"Train {algorithm} - Seed {seed}")

    # Training settings
    col1, col2 = st.columns(2)

    with col1:
        training_timesteps = st.number_input(
            "Training timesteps",
            min_value=100_000,
            max_value=10_000_000,
            value=5_000_000,
            step=500_000,
            help="5M recommended for publication quality"
        )

    with col2:
        train_mode = st.radio(
            "Training mode",
            ["Train from scratch", "Continue training"],
            index=0,
            help="'From scratch' creates a new model. 'Continue' adds steps to existing."
        )

    # Log training option
    log_training_option = st.selectbox(
        "📝 Training log level",
        ["None", "Episode only (fast)", "Full trajectory (slow)"],
        index=0,
        help="None: No logging. Episode: Just rewards per episode. Full: All 44 columns per step (WARNING: very slow, ~10x slower)"
    )

    # Map to parameter value
    log_training_map = {
        "None": "none",
        "Episode only (fast)": "episode",
        "Full trajectory (slow)": "full"
    }
    log_training = log_training_map[log_training_option]

    if log_training == "full":
        st.warning("⚠️ Full trajectory logging will make training ~10x slower and create very large files!")

    # Info box
    if algorithm == "PPO":
        with st.expander("ℹ️ PPO Hyperparameters (Corrected)"):
            st.markdown("""
            | Parameter | Old (Bad) | New (Fixed) |
            |-----------|-----------|-------------|
            | n_epochs | 20 | **5** |
            | ent_coef | 0.0004 | **0.005** |
            | log_std_init | -2 | **-1** |
            | n_steps | 512 | **2048** |
            """)

    # Train button
    if st.button("🚀 Start Training", type="primary"):
        if train_mode == "Train from scratch" and model_exists:
            st.warning(f"This will overwrite the existing {algorithm} seed {seed} model!")

        start_time = time.time()

        with st.spinner(f"Training {algorithm} seed {seed} for {training_timesteps:,} steps..."):
            if algorithm == "PPO":
                if train_mode == "Train from scratch":
                    model, training_log_path = train_ppo(seed=seed, total_timesteps=int(training_timesteps),
                                                         log_training=log_training)
                else:
                    model, training_log_path = continue_training_ppo(seed=seed, total_timesteps=int(training_timesteps),
                                                                     log_training=log_training)
            else:  # SAC
                if train_mode == "Train from scratch":
                    model, training_log_path = train_sac(seed=seed, total_timesteps=int(training_timesteps),
                                                         log_training=log_training)
                else:
                    model, training_log_path = continue_training_sac(seed=seed, total_timesteps=int(training_timesteps),
                                                                     log_training=log_training)

        elapsed = time.time() - start_time
        st.success(f"✅ Training complete! Time: {elapsed / 60:.1f} minutes")

        if training_log_path:
            st.info(f"📁 Training log saved to: `{training_log_path}`")

        st.balloons()

# =============================================================================
# EVALUATION TAB
# =============================================================================

with tab_eval:
    st.header(f"Evaluate {algorithm} - Seed {seed}")

    if not model_exists:
        st.error(f"❌ No trained model found for {algorithm} seed {seed}. Train first!")
    else:
        # Evaluation settings
        col1, col2 = st.columns(2)

        with col1:
            num_episodes = st.number_input(
                "Number of episodes",
                min_value=1,
                max_value=1000,
                value=50,
                step=10,
                help="50 episodes = 50,000 steps of data"
            )

        with col2:
            log_trajectories = st.checkbox(
                "📝 Log trajectories to CSV",
                value=True,
                help="Save (s, a, r, s') for each step"
            )

        st.markdown("---")

        # Noise settings
        st.subheader("Noise Injection")

        col1, col2, col3 = st.columns(3)

        with col1:
            noise_type_ui = st.selectbox(
            "Noise type",
            ["None", "Observation", "Action"],
            index=["None", "Observation", "Action"].index(
                st.session_state.get("noise_type", "None")
            ),
            key="noise_type"
        )
        
    with col2:
        noise_level = st.number_input(
            "Noise level (std)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("noise_level", 0.0),
            step=0.01,
            format="%.2f",
            help="0.05 = 5%, 0.10 = 10%, etc."
        )
        st.session_state["noise_level"] = noise_level

        with col3:
            perturb_start_step = st.number_input(
                "Start step",
                min_value=0,
                max_value=1000,
                value=0,
                step=50,
                help="Step in episode when noise begins (0 = from start)"
            )

        # Quick noise presets
        st.caption("Quick presets:")
        preset_cols = st.columns(6)

        with preset_cols[0]:
            if st.button("No noise"):
                st.session_state["noise_type"] = "None"
                st.session_state["noise_level"] = 0.0
                st.rerun()

       with preset_cols[1]:
            if st.button("Obs 5%"):
                st.session_state["noise_type"] = "Observation"
                st.session_state["noise_level"] = 0.05
                st.rerun()
        
        with preset_cols[2]:
            if st.button("Obs 10%"):
                st.session_state["noise_type"] = "Observation"
                st.session_state["noise_level"] = 0.10
                st.rerun()
        
        with preset_cols[3]:
            if st.button("Obs 20%"):
                st.session_state["noise_type"] = "Observation"
                st.session_state["noise_level"] = 0.20
                st.rerun()
        
        with preset_cols[4]:
            if st.button("Act 10%"):
                st.session_state["noise_type"] = "Action"
                st.session_state["noise_level"] = 0.10
                st.rerun()
        
        with preset_cols[5]:
            if st.button("Act 20%"):
                st.session_state["noise_type"] = "Action"
                st.session_state["noise_level"] = 0.20
                st.rerun()

        # Map UI to parameter
        noise_type_map = {"None": "none", "Observation": "obs", "Action": "act"}
        noise_type_param = noise_type_map[noise_type_ui]

        st.markdown("---")

        # Force injection settings
        st.subheader("Force Injection (Physical Perturbation)")

        force_enabled = st.checkbox(
            "💥 Enable force injection",
            value=False,
            help="Apply external force to robot body during evaluation"
        )

        if force_enabled:
            col1, col2, col3 = st.columns(3)

            with col1:
                force_body = st.selectbox(
                    "Target body",
                    ["torso", "ffoot", "bfoot", "fthigh", "bthigh", "fshin", "bshin"],
                    index=0,
                    help="Which body part to apply force to"
                )

            with col2:
                force_magnitude = st.number_input(
                    "Force (N)",
                    min_value=-50.0,
                    max_value=50.0,
                    value=10.0,
                    step=5.0,
                    help="Positive = forward, Negative = backward"
                )

            with col3:
                force_direction = st.selectbox(
                    "Direction",
                    ["x", "y", "z", "xz"],
                    index=0,
                    help="x=forward/back, y=left/right, z=up/down"
                )

            col1, col2 = st.columns(2)

            with col1:
                force_interval = st.number_input(
                    "Injection interval (steps)",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10,
                    help="Apply force every N steps"
                )

            with col2:
                force_duration = st.number_input(
                    "Force duration (steps)",
                    min_value=1,
                    max_value=50,
                    value=10,
                    step=1,
                    help="How long to apply force"
                )

            st.caption(
                f"Force will be applied for {force_duration} steps every {force_interval} steps = {1000 // force_interval} times per episode")
        else:
            force_body = "torso"
            force_magnitude = 10.0
            force_direction = "x"
            force_interval = 100
            force_duration = 10

        st.markdown("---")

        # Gravity perturbation settings
        st.subheader("Gravity Perturbation (Physics Change)")

        gravity_enabled = st.checkbox(
            "🌍 Enable gravity perturbation",
            value=False,
            help="Modify gravity to simulate different environments"
        )

        if gravity_enabled:
            gravity_scale = st.slider(
                "Gravity scale",
                min_value=0.5,
                max_value=1.5,
                value=1.2,
                step=0.1,
                help="1.0 = normal (-9.81 m/s²), 1.2 = 20% heavier, 0.8 = 20% lighter"
            )
            st.caption(f"Gravity: {gravity_scale * -9.81:.2f} m/s² ({gravity_scale * 100:.0f}% of normal)")
        else:
            gravity_scale = 1.0

        st.markdown("---")

        # IDT Intervention settings
        st.subheader("🔧 IDT Intervention (Correction)")

        intervention_mode = st.radio(
            "Intervention Mode",
            ["Off", "Static", "Dynamic"],
            index=0,
            horizontal=True,
            help="Off=no intervention, Static=fixed episodes, Dynamic=auto based on P"
        )

        if intervention_mode != "Off":
            st.markdown("**Wrappers (Filtering)**")
            col1, col2, col3 = st.columns(3)

            with col1:
                obs_smoothing = st.slider(
                    "Observation Smoothing",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1,
                    help="0 = no smoothing, 1 = max smoothing (EMA filter)"
                )

            with col2:
                act_smoothing = st.slider(
                    "Action Smoothing",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1,
                    help="0 = no smoothing, 1 = max smoothing (EMA filter)"
                )

            with col3:
                act_clip = st.slider(
                    "Action Clipping",
                    min_value=0.1,
                    max_value=1.0,
                    value=1.0,
                    step=0.1,
                    help="1 = no clipping, 0.5 = clip to 50% of max action"
                )

            st.markdown("**Holds (Probabilistic Skip)**")
            col1, col2 = st.columns(2)

            with col1:
                obs_hold_prob = st.slider(
                    "Observation Hold Probability",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.0,
                    step=0.05,
                    help="0 = normal, 0.1 = 10% chance to reuse previous obs"
                )

            with col2:
                act_hold_prob = st.slider(
                    "Action Hold Probability",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.0,
                    step=0.05,
                    help="0 = normal, 0.1 = 10% chance to reuse previous action"
                )

            # Dynamic mode settings
            if intervention_mode == "Dynamic":
                st.markdown("**Dynamic P Thresholds**")
                col1, col2, col3 = st.columns(3)

                with col1:
                    p_threshold_low = st.slider(
                        "P Threshold Low",
                        min_value=0.20,
                        max_value=0.50,
                        value=0.28,
                        step=0.02,
                        help="Enable intervention when P drops below this"
                    )

                with col2:
                    p_threshold_high = st.slider(
                        "P Threshold High",
                        min_value=0.30,
                        max_value=0.60,
                        value=0.32,
                        step=0.02,
                        help="Disable intervention when P rises above this"
                    )

                with col3:
                    buffer_size = st.number_input(
                        "Buffer Size",
                        min_value=100,
                        max_value=1000,
                        value=500,
                        step=100,
                        help="Rolling buffer for P computation"
                    )
            else:
                p_threshold_low = 0.28
                p_threshold_high = 0.32
                buffer_size = 500

            # P Computation Settings (for all modes)
            st.markdown("**P Computation Settings**")
            col1, col2 = st.columns(2)

            with col1:
                num_bins = st.selectbox(
                    "Number of Bins",
                    [3, 4, 5],
                    index=0,
                    help="Discretization bins for P calculation"
                )

            with col2:
                baseline_seed = st.selectbox(
                    "Baseline Seed",
                    [3, 5, 7],
                    index=0,
                    help="Which seed's baseline to use"
                )

            # Auto-generate baseline path
            baseline_path = f"sac{baseline_seed}_baseline_{num_bins}bins.json"
            st.caption(f"Using baseline: `{baseline_path}`")
        else:
            obs_smoothing = 0.0
            act_smoothing = 0.0
            act_clip = 1.0
            obs_hold_prob = 0.0
            act_hold_prob = 0.0
            p_threshold_low = 0.28
            p_threshold_high = 0.32
            buffer_size = 500

            # P Computation Settings (even for Off mode, P is still computed)
            st.markdown("**P Computation Settings**")
            col1, col2 = st.columns(2)

            with col1:
                num_bins = st.selectbox(
                    "Number of Bins",
                    [3, 4, 5],
                    index=0,
                    help="Discretization bins for P calculation"
                )

            with col2:
                baseline_seed = st.selectbox(
                    "Baseline Seed",
                    [3, 5, 7],
                    index=0,
                    help="Which seed's baseline to use"
                )

            # Auto-generate baseline path
            baseline_path = f"sac{baseline_seed}_baseline_{num_bins}bins.json"
            st.caption(f"Using baseline: `{baseline_path}`")

        st.markdown("---")

        # Timing (Perturbation + Correction)
        st.subheader("⏱️ Phase Timing")

        col1, col2 = st.columns(2)

        with col1:
            perturb_start_episode = st.number_input(
                "Start perturbations at episode",
                min_value=0,
                max_value=int(num_episodes) - 1,
                value=15,
                step=5,
                help="Episode when perturbations begin"
            )

        with col2:
            if intervention_mode == "Static":
                correction_start_episode = st.number_input(
                    "Start correction at episode",
                    min_value=perturb_start_episode,
                    max_value=int(num_episodes) - 1,
                    value=min(35, int(num_episodes) - 1),
                    step=5,
                    help="Episode when IDT intervention begins"
                )
            else:
                correction_start_episode = 0

        # Visual timeline
        if intervention_mode == "Static" and perturb_start_episode > 0 and correction_start_episode > perturb_start_episode:
            st.info(
                f"🟢 Episodes 1-{perturb_start_episode}: Baseline\n"
                f"🔴 Episodes {perturb_start_episode + 1}-{correction_start_episode}: Perturbed\n"
                f"🔧 Episodes {correction_start_episode + 1}-{int(num_episodes)}: Corrected"
            )
        elif intervention_mode == "Dynamic" and perturb_start_episode > 0:
            st.info(
                f"🟢 Episodes 1-{perturb_start_episode}: Baseline\n"
                f"🔴 Episodes {perturb_start_episode + 1}+: Perturbed\n"
                f"🔧 Intervention: AUTO (P < {p_threshold_low} → ON, P > {p_threshold_high} → OFF)"
            )
        elif perturb_start_episode > 0:
            baseline_eps = perturb_start_episode
            st.info(
                f"🟢 Episodes 1-{baseline_eps}: Baseline\n🔴 Episodes {baseline_eps + 1}-{int(num_episodes)}: Perturbed")
        else:
            st.info("All episodes will have perturbations enabled (if any selected)")

        st.markdown("---")

        # Run evaluation button
        if st.button("▶️ Run Evaluation", type="primary"):
            with st.spinner(f"Evaluating {algorithm} seed {seed}..."):
                if log_trajectories:
                    if algorithm == "PPO":
                        episode_returns, avg_return, log_path = evaluate_and_log_ppo(
                            seed=seed,
                            num_episodes=int(num_episodes),
                            noise_type=noise_type_param,
                            noise_level=float(noise_level),
                            perturb_start_step=int(perturb_start_step),
                            force_enabled=force_enabled,
                            force_body=force_body,
                            force_magnitude=float(force_magnitude),
                            force_direction=force_direction,
                            force_interval=int(force_interval),
                            force_duration=int(force_duration),
                            gravity_enabled=gravity_enabled,
                            gravity_scale=float(gravity_scale),
                            perturb_start_episode=int(perturb_start_episode),
                        )
                    else:
                        # SAC - always use function with P monitoring
                        episode_returns, avg_return, log_path = evaluate_and_log_sac_with_P(
                            seed=seed,
                            num_episodes=int(num_episodes),
                            noise_type=noise_type_param,
                            noise_level=float(noise_level),
                            perturb_start_step=int(perturb_start_step),
                            perturb_start_episode=int(perturb_start_episode),
                            intervention_mode=intervention_mode.lower(),
                            obs_smoothing=float(obs_smoothing),
                            act_smoothing=float(act_smoothing),
                            act_clip=float(act_clip),
                            obs_hold_prob=float(obs_hold_prob),
                            act_hold_prob=float(act_hold_prob),
                            correction_start_episode=int(correction_start_episode),
                            baseline_path=baseline_path,
                            p_threshold_low=float(p_threshold_low),
                            p_threshold_high=float(p_threshold_high),
                            buffer_size=int(buffer_size),
                        )
                else:
                    if algorithm == "PPO":
                        episode_returns, avg_return = evaluate_ppo(
                            seed=seed,
                            num_episodes=int(num_episodes),
                        )
                    else:
                        episode_returns, avg_return = evaluate_sac(
                            seed=seed,
                            num_episodes=int(num_episodes),
                        )
                    log_path = None

            # Results
            st.success("✅ Evaluation complete!")

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Reward", f"{avg_return:.1f}")
            with col2:
                st.metric("Min", f"{min(episode_returns):.1f}")
            with col3:
                st.metric("Max", f"{max(episode_returns):.1f}")







            # Phase Analysis for all SAC modes
            if algorithm == "SAC" and perturb_start_episode > 0:
                st.subheader("📊 Phase Analysis")

                baseline_rewards = episode_returns[:perturb_start_episode]
                perturbed_rewards = episode_returns[perturb_start_episode:]

                col1, col2, col3 = st.columns(3)
                with col1:
                    baseline_avg = sum(baseline_rewards) / len(baseline_rewards) if baseline_rewards else 0
                    st.metric("Baseline Avg", f"{baseline_avg:.1f}",
                              help=f"Episodes 0-{perturb_start_episode - 1}")
                with col2:
                    perturbed_avg = sum(perturbed_rewards) / len(perturbed_rewards) if perturbed_rewards else 0
                    delta = perturbed_avg - baseline_avg
                    st.metric("Perturbed+Intervention Avg", f"{perturbed_avg:.1f}",
                              delta=f"{delta:.1f}",
                              delta_color="normal",
                              help=f"Episodes {perturb_start_episode}+")
                with col3:
                    if baseline_avg > 0:
                        pct_change = (perturbed_avg - baseline_avg) / baseline_avg * 100
                        st.metric("Change", f"{pct_change:.1f}%")

            # Chart - Episode Returns with trend line
            st.subheader("Episode Returns")

            # Create chart with rolling average
            fig_reward, ax_reward = plt.subplots(figsize=(12, 4))
            episodes = list(range(len(episode_returns)))
            ax_reward.bar(episodes, episode_returns, alpha=0.5, color='steelblue', label='Episode Return')

            # Rolling average (window=5)
            if len(episode_returns) >= 5:
                rolling_avg = pd.Series(episode_returns).rolling(window=5, center=True).mean()
                ax_reward.plot(episodes, rolling_avg, 'r-', linewidth=2, label='Rolling Avg (5 ep)')

            # Mark perturbation start
            if perturb_start_episode > 0:
                ax_reward.axvline(x=perturb_start_episode, color='orange', linestyle='--',
                                  label=f'Perturbation Start (ep {perturb_start_episode})')

            ax_reward.set_xlabel('Episode')
            ax_reward.set_ylabel('Return')
            ax_reward.set_title('Episode Returns with Trend')
            ax_reward.legend()
            ax_reward.grid(True, alpha=0.3)
            st.pyplot(fig_reward)

            # P Chart for all SAC modes (P is always computed now)
            if log_path and algorithm == "SAC":
                st.subheader("📈 P Values Over Time")
                try:
                    df = pd.read_csv(log_path)
                    df_p = df[df['P'].notna() & (df['P'] != '')].copy()
                    if len(df_p) > 0:
                        df_p['P'] = df_p['P'].astype(float)

                        # Aggregate P by episode (mean P per episode)
                        p_by_episode = df_p.groupby('episode')['P'].mean()
                        p_min_by_episode = df_p.groupby('episode')['P'].min()

                        # Create P chart with thresholds
                        fig, ax = plt.subplots(figsize=(12, 4))
                        ax.plot(p_by_episode.index, p_by_episode.values, 'b-', linewidth=1.5, marker='o', markersize=3,
                                label='P (episode avg)')
                        ax.plot(p_min_by_episode.index, p_min_by_episode.values, 'r-', linewidth=1, marker='v',
                                markersize=2, alpha=0.7, label='P min')
                        ax.axhline(y=p_threshold_low, color='r', linestyle='--', label=f'Low ({p_threshold_low})')
                        ax.axhline(y=p_threshold_high, color='g', linestyle='--', label=f'High ({p_threshold_high})')
                        if perturb_start_episode > 0:
                            ax.axvline(x=perturb_start_episode, color='orange', linestyle='--', alpha=0.7,
                                       label=f'Perturbation (ep {perturb_start_episode})')
                        ax.set_ylim(0.30, 0.38)
                        ax.set_xlabel('Episode')
                        ax.set_ylabel('P')
                        ax.set_title(f'Predictive Coherence (P) - {intervention_mode} Mode')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)

                        # Intervention chart - aggregate by episode (% ON per episode)
                        if intervention_mode != "Off":
                            st.subheader("🔧 Intervention Status")
                            intervention_by_episode = df.groupby('episode')['intervention_regime'].mean() * 100

                            fig2, ax2 = plt.subplots(figsize=(12, 2))
                            ax2.bar(intervention_by_episode.index, intervention_by_episode.values, alpha=0.7,
                                    color='orange')
                            if perturb_start_episode > 0:
                                ax2.axvline(x=perturb_start_episode, color='red', linestyle='--', alpha=0.7)
                            ax2.set_ylabel('% ON')
                            ax2.set_xlabel('Episode')
                            ax2.set_title('Intervention Active (% of steps per episode)')
                            ax2.set_ylim(0, 105)
                            st.pyplot(fig2)

                        # Summary stats
                        st.markdown("**P Statistics:**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("P Min", f"{df_p['P'].min():.3f}")
                        with col2:
                            st.metric("P Max", f"{df_p['P'].max():.3f}")
                        with col3:
                            st.metric("P Mean", f"{df_p['P'].mean():.3f}")
                        with col4:
                            intervention_pct = df['intervention_regime'].sum() / len(df) * 100
                            st.metric("Intervention %", f"{intervention_pct:.1f}%")

                            # Log test results
                            log_test_results(
                                log_file=log_path,
                                algorithm=algorithm,
                                seed=seed,
                                num_episodes=int(num_episodes),
                                noise_type=noise_type_param,
                                noise_level=float(noise_level),
                                perturb_start_episode=int(perturb_start_episode),
                                intervention_mode=intervention_mode,
                                obs_smoothing=float(obs_smoothing),
                                act_smoothing=float(act_smoothing),
                                act_clip=float(act_clip),
                                obs_hold_prob=float(obs_hold_prob),
                                act_hold_prob=float(act_hold_prob),
                                correction_start_episode=int(correction_start_episode),
                                num_bins=num_bins,
                                buffer_size=int(buffer_size),
                                p_threshold_low=float(p_threshold_low),
                                p_threshold_high=float(p_threshold_high),
                                baseline_path=baseline_path,
                                avg_reward=avg_return,
                                min_reward=min(episode_returns),
                                max_reward=max(episode_returns),
                                baseline_avg=baseline_avg if perturb_start_episode > 0 else avg_return,
                                perturbed_avg=perturbed_avg if perturb_start_episode > 0 else avg_return,
                                reward_change_pct=pct_change if perturb_start_episode > 0 else 0.0,
                                p_mean=df_p['P'].mean(),
                                p_min=df_p['P'].min(),
                                p_max=df_p['P'].max(),
                                intervention_pct=intervention_pct,
                            )


                    else:
                        st.warning("No P values found in log file")
                except Exception as e:
                    st.error(f"Error loading P data: {e}")

            # Log file info
            if log_path:
                st.markdown("---")
                st.subheader("📁 Log File")
                st.code(log_path)
                st.caption(f"Contains {int(num_episodes) * 1000:,} rows ({int(num_episodes)} eps × 1000 steps)")

            # Download log file
                with open(log_path, 'r') as f:
                    st.download_button(
                        label="📥 Download Log CSV",
                        data=f.read(),
                        file_name=log_path.split('/')[-1],
                        mime="text/csv"
                    )
            # Download test log
                if os.path.exists("test_log.csv"):
                    with open("test_log.csv", 'r') as f:
                        st.download_button(
                            label="📥 Download Test Log",
                            data=f.read(),
                            file_name="test_log.csv",
                            mime="text/csv"
                        )
# =============================================================================
# FOOTER
# =============================================================================

