# IDT Monitoring System

**Information Digital Twin (IDT) for Real-Time RL Agent Robustness Monitoring**

## Overview

A framework for monitoring reinforcement learning agent robustness using information-theoretic metrics. The system computes Predictive Coherence (P) in real-time to detect when an agent's behavior degrades under perturbations, and can automatically trigger corrective interventions.

## Live Demo

🚀 **[Launch App](https://idtapp-ekhkszfprr3wsdxpcxzv7e.streamlit.app/)**

## Capabilities

### Evaluation
- Pre-trained SAC/PPO agents (HalfCheetah-v4)
- Multiple seeds available (3, 5, 7, etc.)
- Real-time P monitoring with visualization

### Perturbations
- Observation noise (1-20%)
- Action noise (1-20%)
- Configurable start episode

### IDT Intervention (Correction)
- **Observation smoothing** — EMA filter on observations
- **Action smoothing** — EMA filter on actions
- **Action clipping** — Scale action magnitude
- **Hold probabilities** — Skip % of observations/actions

### Intervention Modes
| Mode | Description |
|------|-------------|
| Off | No intervention, P still computed for monitoring |
| Static | Intervention starts at fixed episode |
| Dynamic | Auto-trigger based on P thresholds |

### Visualization
- Episode returns with rolling average
- P values over time (average + minimum)
- Intervention status per episode
- Phase analysis (baseline vs perturbed)

### Export
- Download trajectory logs (CSV)
- Download test summary log (CSV)

### Log Files

**Trajectory Log** (per run)
Contains step-by-step data: episode, step, S, A, S', reward, P, intervention_regime

### Log Files

**Trajectory Log** (per run)

The trajectory log is the core output of the IDT monitoring system. Every computation — normalization, binning, windowing, entropy calculation — exists to transform raw agent behavior into this structured record of state-action-next_state transitions.

**What it captures:**
- Complete record of agent-environment interaction
- Every decision the agent made (actions)
- Every observation the agent received (states)
- Every outcome that resulted (next states, rewards)
- Real-time P values measuring behavioral coherence
- Intervention status at each moment

**Why it matters:**
- **Reproducibility**: Full trace allows exact replay of what happened
- **Diagnosis**: Pinpoint exactly when and where behavior degraded
- **Analysis**: Compute any information-theoretic metric offline
- **Comparison**: Compare runs across seeds, noise levels, interventions
- **Validation**: Verify that P correctly detects perturbations
- **Research**: Source data for publications and deeper analysis

**Columns:**
| Column | Description |
|--------|-------------|
| episode | Episode number (0 to N) |
| t | Step within episode (0 to 1000) |
| regime | Perturbation active (0=off, 1=on) |
| intervention_regime | Intervention active (0=off, 1=on) |
| P | Predictive coherence at this step (computed every 50 steps) |
| s_0 to s_16 | 17-dimensional state observation |
| a_0 to a_5 | 6-dimensional action |
| reward | Step reward |
| done | Episode termination flag |
| s_next_0 to s_next_16 | Next state observation |

**How to use:**
- Open in Excel/Python for custom analysis
- Feed into `analyze_trajectory.py` for full entropy breakdown
- Plot state trajectories to visualize behavior
- Compare P patterns across different conditions
- Build flight envelopes from multiple runs

**Test Log** (test_log.csv)

Tracks all experiment runs with settings and summary results:

| Category | Fields |
|----------|--------|
| Run info | timestamp, log_file |
| Model | algorithm, seed, num_episodes |
| Noise | noise_type, noise_level, perturb_start_episode |
| Intervention | mode, obs_smoothing, act_smoothing, act_clip |
| P settings | num_bins, buffer_size, p_threshold_low, p_threshold_high |
| Results | avg_reward, baseline_avg, perturbed_avg, P_mean, P_min, intervention_pct |

Use this to compare experiments without opening individual trajectory files.

**Test Log** (test_log.csv)
Tracks all runs with settings and results:

| Category | Fields |
|----------|--------|
| Run info | timestamp, log_file |
| Model | algorithm, seed, num_episodes |
| Noise | noise_type, noise_level, perturb_start_episode |
| Intervention | mode, obs_smoothing, act_smoothing, act_clip |
| P settings | num_bins, buffer_size, p_threshold_low, p_threshold_high |
| Results | avg_reward, baseline_avg, perturbed_avg, P_mean, P_min, intervention_pct |


## How to Use

1. **Select Model**: Choose algorithm (SAC/PPO) and seed in sidebar
2. **Go to Evaluation Tab**
3. **Enable "Log trajectories"**
4. **Configure Perturbation**:
   - Set noise type and level
   - Set perturbation start episode
5. **Configure Intervention** (optional):
   - Select mode (Off/Static/Dynamic)
   - Adjust smoothing parameters
   - Set P thresholds for Dynamic mode
6. **Set P Computation**:
   - Number of bins (3, 4, or 5)
   - Buffer size (default 500)
7. **Run Evaluation**
8. **Analyze Results**:
   - View charts
   - Download logs

## Technical Details

### Predictive Coherence (P)

P measures how well the agent's state-action pairs predict next states:

```
P = MI(S,A ; S') / H_Total

Where:
- MI(S,A ; S') = H(S,A) + H(S') - H(S,A,S')
- H_Total = H(S) + H(A) + H(S')
```

**Interpretation:**
- P ≈ 0.33: Healthy baseline behavior
- P drops: Agent behavior degrading
- P near 0: Unpredictable/chaotic behavior

### Body-Part Grouping (HalfCheetah)

States are grouped by body part for entropy computation:

| Group | State Indices |
|-------|---------------|
| Back leg | 2, 3, 4, 11, 12, 13 |
| Front leg | 5, 6, 7, 14, 15, 16 |
| Tip | 0, 1, 8, 9, 10 |

### Binning

- **Method**: Equal-width bins on z-scored data
- **Options**: 3, 4, or 5 bins per dimension
- **Baseline**: Pre-computed bin edges from clean runs

### Dynamic Intervention

Uses hysteresis thresholds:
- P < threshold_low → Intervention ON
- P > threshold_high → Intervention OFF
- Between thresholds → No change

This prevents rapid on/off oscillation.

## Files

| File | Purpose |
|------|---------|
| ui.py | Streamlit interface |
| app.py | Training/evaluation functions |
| idt_monitor.py | Real-time P computation |
| idt_intervention.py | Correction wrappers |
| analyze_trajectory.py | Offline analysis |

## Citation

If you use this work, please cite:

```
@software{idt_monitoring,
  title={IDT Monitoring System},
  author={Semarx Research},
  year={2025},
  url={https://github.com/Semarxai/idt_app}
}
```

## License

Proprietary - All rights reserved.

---

**Semarx Research** | [semarx.com](https://www.semarx.com)
