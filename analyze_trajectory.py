"""
Trajectory Analysis Pipeline
=============================
Local Python implementation of BigQuery + Spark pipeline.

Usage:
    python analyze_trajectory.py <trajectory_csv> [--baseline baseline.json] [--window 500] [--stride 50]

Example:
    python analyze_trajectory.py logs_sac_claude/sac_seed3_force_torso_18N_from_ep50.csv
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime

# =============================================================================
# BODY-PART GROUPING (from BigQuery Pipeline PDF)
# =============================================================================

# Correct mappings from PDF
BODY_PARTS = {
    "back_leg": [2, 3, 4, 11, 12, 13],  # s_2, s_3, s_4, s_11, s_12, s_13
    "front_leg": [5, 6, 7, 14, 15, 16],  # s_5, s_6, s_7, s_14, s_15, s_16
    "tip": [0, 1, 8, 9, 10],  # s_0, s_1, s_8, s_9, s_10
}

ACTION_INDICES = [0, 1, 2, 3, 4, 5]  # a_0 through a_5


# =============================================================================
# ENTROPY CALCULATION
# =============================================================================

def entropy(series: pd.Series) -> float:
    """
    Calculate Shannon entropy in bits.
    H(X) = -Σ p(x) * log2(p(x))
    """
    if len(series) == 0:
        return 0.0

    counts = series.value_counts()
    probs = counts / counts.sum()

    # Filter out zeros to avoid log(0)
    probs = probs[probs > 0]

    return -np.sum(probs * np.log2(probs))


# =============================================================================
# BASELINE MANAGEMENT
# =============================================================================

def load_baseline(baseline_path: str) -> Dict:
    """Load baseline parameters (mean, std, bin_edges)."""
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")

    with open(baseline_path, 'r') as f:
        baseline = json.load(f)

    print(f"Loaded baseline from: {baseline_path}")
    return baseline


def compute_baseline(df: pd.DataFrame, num_bins: int = 3) -> Dict:
    """
    Compute baseline parameters from data.

    Returns dict with:
    - s_mean, s_std (17 values each)
    - a_mean, a_std (6 values each)
    - s_bin_edges (17 x (num_bins-1) edges)
    - a_bin_edges (6 x (num_bins-1) edges)
    """
    baseline = {}

    # State columns
    s_cols = [f"s_{i}" for i in range(17)]
    s_data = df[s_cols]
    baseline["s_mean"] = s_data.mean().tolist()
    baseline["s_std"] = s_data.std().tolist()

    # Action columns
    a_cols = [f"a_{i}" for i in range(6)]
    a_data = df[a_cols]
    baseline["a_mean"] = a_data.mean().tolist()
    baseline["a_std"] = a_data.std().tolist()

    # Z-score the data first for bin edge calculation
    s_z = (s_data - s_data.mean()) / s_data.std()
    a_z = (a_data - a_data.mean()) / a_data.std()

    # Compute bin edges for z-scored data (equal-width bins)
    s_bin_edges = []
    for col in s_z.columns:
        col_min = s_z[col].min()
        col_max = s_z[col].max()
        col_range = col_max - col_min

        # 2 edges for 3 bins
        edge1 = col_min + col_range / 3
        edge2 = col_min + 2 * col_range / 3
        s_bin_edges.append([float(edge1), float(edge2)])

    a_bin_edges = []
    for col in a_z.columns:
        col_min = a_z[col].min()
        col_max = a_z[col].max()
        col_range = col_max - col_min

        edge1 = col_min + col_range / 3
        edge2 = col_min + 2 * col_range / 3
        a_bin_edges.append([float(edge1), float(edge2)])

    baseline["s_bin_edges"] = s_bin_edges
    baseline["a_bin_edges"] = a_bin_edges
    baseline["num_bins"] = num_bins

    return baseline


def save_baseline(baseline: Dict, baseline_path: str):
    """Save baseline parameters to JSON."""
    os.makedirs(os.path.dirname(baseline_path) or ".", exist_ok=True)

    with open(baseline_path, 'w') as f:
        json.dump(baseline, f, indent=2)

    print(f"Baseline saved to: {baseline_path}")


# =============================================================================
# DATA PROCESSING
# =============================================================================

def z_score_normalize(df: pd.DataFrame, baseline: Dict) -> pd.DataFrame:
    """Apply z-score normalization using baseline mean/std."""
    df = df.copy()

    # Normalize s_0...s_16
    for i in range(17):
        col = f"s_{i}"
        if col in df.columns:
            mean = baseline["s_mean"][i]
            std = baseline["s_std"][i]
            std = std if std > 0 else 1.0
            df[f"{col}_z"] = (df[col] - mean) / std

    # Normalize a_0...a_5
    for i in range(6):
        col = f"a_{i}"
        if col in df.columns:
            mean = baseline["a_mean"][i]
            std = baseline["a_std"][i]
            std = std if std > 0 else 1.0
            df[f"{col}_z"] = (df[col] - mean) / std

    # Normalize s_next_0...s_next_16 (using same params as s)
    for i in range(17):
        col = f"s_next_{i}"
        if col in df.columns:
            mean = baseline["s_mean"][i]
            std = baseline["s_std"][i]
            std = std if std > 0 else 1.0
            df[f"{col}_z"] = (df[col] - mean) / std

    return df


def discretize(df: pd.DataFrame, baseline: Dict) -> pd.DataFrame:
    """Apply discretization using baseline bin edges."""
    df = df.copy()

    def apply_bins(value, edges):
        """Assign bin label 1, 2, or 3 based on edges."""
        if value <= edges[0]:
            return 1
        elif value <= edges[1]:
            return 2
        else:
            return 3

    # Discretize s_0...s_16
    for i in range(17):
        z_col = f"s_{i}_z"
        if z_col in df.columns:
            edges = baseline["s_bin_edges"][i]
            df[f"s_{i}_b"] = df[z_col].apply(lambda x: apply_bins(x, edges))

    # Discretize a_0...a_5
    for i in range(6):
        z_col = f"a_{i}_z"
        if z_col in df.columns:
            edges = baseline["a_bin_edges"][i]
            df[f"a_{i}_b"] = df[z_col].apply(lambda x: apply_bins(x, edges))

    # Discretize s_next_0...s_next_16 (using same edges as s)
    for i in range(17):
        z_col = f"s_next_{i}_z"
        if z_col in df.columns:
            edges = baseline["s_bin_edges"][i]
            df[f"s_next_{i}_b"] = df[z_col].apply(lambda x: apply_bins(x, edges))

    return df


def compose_system(df: pd.DataFrame) -> pd.DataFrame:
    """Create system-level S, A, S' compositions."""
    df = df.copy()

    # Back leg: s_2, s_3, s_4, s_11, s_12, s_13
    df["back_leg"] = (
            df["s_2_b"].astype(str) +
            df["s_3_b"].astype(str) +
            df["s_4_b"].astype(str) +
            df["s_11_b"].astype(str) +
            df["s_12_b"].astype(str) +
            df["s_13_b"].astype(str)
    )

    # Front leg: s_5, s_6, s_7, s_14, s_15, s_16
    df["front_leg"] = (
            df["s_5_b"].astype(str) +
            df["s_6_b"].astype(str) +
            df["s_7_b"].astype(str) +
            df["s_14_b"].astype(str) +
            df["s_15_b"].astype(str) +
            df["s_16_b"].astype(str)
    )

    # Tip: s_0, s_1, s_8, s_9, s_10
    df["tip"] = (
            df["s_0_b"].astype(str) +
            df["s_1_b"].astype(str) +
            df["s_8_b"].astype(str) +
            df["s_9_b"].astype(str) +
            df["s_10_b"].astype(str)
    )

    # Actions
    df["actions"] = (
            df["a_0_b"].astype(str) +
            df["a_1_b"].astype(str) +
            df["a_2_b"].astype(str) +
            df["a_3_b"].astype(str) +
            df["a_4_b"].astype(str) +
            df["a_5_b"].astype(str)
    )

    # Back leg next
    df["back_leg_next"] = (
            df["s_next_2_b"].astype(str) +
            df["s_next_3_b"].astype(str) +
            df["s_next_4_b"].astype(str) +
            df["s_next_11_b"].astype(str) +
            df["s_next_12_b"].astype(str) +
            df["s_next_13_b"].astype(str)
    )

    # Front leg next
    df["front_leg_next"] = (
            df["s_next_5_b"].astype(str) +
            df["s_next_6_b"].astype(str) +
            df["s_next_7_b"].astype(str) +
            df["s_next_14_b"].astype(str) +
            df["s_next_15_b"].astype(str) +
            df["s_next_16_b"].astype(str)
    )

    # Tip next
    df["tip_next"] = (
            df["s_next_0_b"].astype(str) +
            df["s_next_1_b"].astype(str) +
            df["s_next_8_b"].astype(str) +
            df["s_next_9_b"].astype(str) +
            df["s_next_10_b"].astype(str)
    )

    # System-level compositions
    df["S"] = df["back_leg"] + "|" + df["front_leg"] + "|" + df["tip"]
    df["A"] = df["actions"]
    df["S_next"] = df["back_leg_next"] + "|" + df["front_leg_next"] + "|" + df["tip_next"]

    # Joint distributions
    df["S_A"] = df["S"] + "||" + df["A"]
    df["A_Snext"] = df["A"] + "||" + df["S_next"]
    df["S_Snext"] = df["S"] + "||" + df["S_next"]
    df["S_A_Snext"] = df["S"] + "||" + df["A"] + "||" + df["S_next"]

    return df


# =============================================================================
# WINDOWING AND METRICS
# =============================================================================

def compute_window_metrics(window_df: pd.DataFrame) -> Dict:
    """Compute all entropy metrics for a single window."""

    # Marginal entropies
    H_S = entropy(window_df["S"])
    H_A = entropy(window_df["A"])
    H_Snext = entropy(window_df["S_next"])

    # Pairwise joint entropies
    H_S_A = entropy(window_df["S_A"])
    H_A_Snext = entropy(window_df["A_Snext"])
    H_S_Snext = entropy(window_df["S_Snext"])

    # Triple joint entropy
    H_S_A_Snext = entropy(window_df["S_A_Snext"])

    # Derived metrics
    H_Total = H_S + H_A + H_Snext

    # MI(S,A;S') = H(S,A) + H(S') - H(S,A,S')
    MI_SA_Snext = H_S_A + H_Snext - H_S_A_Snext

    # MI(S;A) = H(S) + H(A) - H(S,A)
    MI_S_A = H_S + H_A - H_S_A

    # P = MI(S,A;S') / H_Total (normalized predictive information)
    P = MI_SA_Snext / H_Total if H_Total > 0 else 0.0

    # Hf (forward) = H(S'|S,A) = H(S,A,S') - H(S,A)
    Hf = H_S_A_Snext - H_S_A

    # Hb (backward) = H(S,A|S') = H(S,A,S') - H(S')
    Hb = H_S_A_Snext - H_Snext

    # ΔH = Hf - Hb
    delta_H = Hf - Hb

    # Mean reward in window
    mean_reward = window_df["reward"].mean() if "reward" in window_df.columns else 0.0

    # Force regime (fraction of window with force active)
    force_regime_frac = window_df["force_regime"].mean() if "force_regime" in window_df.columns else 0.0

    return {
        "H_S": H_S,
        "H_A": H_A,
        "H_Snext": H_Snext,
        "H_S_A": H_S_A,
        "H_A_Snext": H_A_Snext,
        "H_S_Snext": H_S_Snext,
        "H_S_A_Snext": H_S_A_Snext,
        "H_Total": H_Total,
        "MI_SA_Snext": MI_SA_Snext,
        "MI_S_A": MI_S_A,
        "P": P,
        "Hf": Hf,
        "Hb": Hb,
        "delta_H": delta_H,
        "mean_reward": mean_reward,
        "force_regime_frac": force_regime_frac,
    }


def sliding_window_analysis(
        df: pd.DataFrame,
        window_size: int = 500,
        stride: int = 50,
) -> pd.DataFrame:
    """
    Compute metrics over sliding windows.
    Step-based (not episode-based) for real-time applicability.
    """
    results = []

    total_steps = len(df)
    num_windows = (total_steps - window_size) // stride + 1

    print(f"Total steps: {total_steps}")
    print(f"Window size: {window_size}, Stride: {stride}")
    print(f"Number of windows: {num_windows}")

    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size

        window_df = df.iloc[start_idx:end_idx]

        metrics = compute_window_metrics(window_df)
        metrics["window_id"] = i
        metrics["step_start"] = start_idx
        metrics["step_end"] = end_idx

        # Get episode range in this window
        if "episode" in window_df.columns:
            metrics["episode_start"] = int(window_df["episode"].iloc[0])
            metrics["episode_end"] = int(window_df["episode"].iloc[-1])

        results.append(metrics)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_windows} windows")

    return pd.DataFrame(results)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def analyze_trajectory(
        csv_path: str,
        baseline_path: str,  # REQUIRED - no default
        window_size: int = 500,
        stride: int = 50,
        output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Full analysis pipeline.

    Args:
        csv_path: Path to trajectory CSV
        baseline_path: Path to baseline.json (REQUIRED - create with create_baseline.py)
        window_size: Window size in steps
        stride: Stride in steps
        output_path: Output path for metrics CSV (auto-generated if None)

    Returns:
        DataFrame with window metrics
    """
    print("=" * 60)
    print("Trajectory Analysis Pipeline")
    print("=" * 60)

    # Load trajectory data
    print(f"\nLoading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # Load baseline - MUST exist, no auto-create
    if not os.path.exists(baseline_path):
        print("\n" + "=" * 60)
        print("ERROR: Baseline file not found!")
        print(f"Expected: {baseline_path}")
        print("")
        print("You must create a baseline first using create_baseline.py:")
        print("  python create_baseline.py <clean_csv> <baseline_name.json>")
        print("=" * 60)
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")

    baseline = load_baseline(baseline_path)
    print(f"Using baseline: {baseline_path}")
    if "_source_file" in baseline:
        print(f"  (created from: {baseline['_source_file']})")

    # Step 1: Z-score normalization
    print("\nStep 1: Z-score normalization...")
    df = z_score_normalize(df, baseline)

    # Step 2: Discretization
    print("Step 2: Discretization...")
    df = discretize(df, baseline)

    # Step 3: System composition
    print("Step 3: System composition...")
    df = compose_system(df)

    # Step 4: Sliding window analysis
    print(f"\nStep 4: Sliding window analysis (window={window_size}, stride={stride})...")
    metrics_df = sliding_window_analysis(df, window_size, stride)

    # Save results
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_dir = os.path.dirname(csv_path) or "."
        output_path = os.path.join(output_dir, f"{base_name}_metrics.csv")

    metrics_df.to_csv(output_path, index=False)
    print(f"\nMetrics saved to: {output_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Windows analyzed: {len(metrics_df)}")
    print(f"P mean: {metrics_df['P'].mean():.4f}")
    print(f"P std:  {metrics_df['P'].std():.4f}")
    print(f"P min:  {metrics_df['P'].min():.4f}")
    print(f"P max:  {metrics_df['P'].max():.4f}")

    if "force_regime_frac" in metrics_df.columns:
        baseline_windows = metrics_df[metrics_df["force_regime_frac"] == 0]
        perturbed_windows = metrics_df[metrics_df["force_regime_frac"] > 0]

        if len(baseline_windows) > 0 and len(perturbed_windows) > 0:
            print(f"\nBaseline windows (no force): {len(baseline_windows)}")
            print(f"  P mean: {baseline_windows['P'].mean():.4f}")
            print(f"\nPerturbed windows (with force): {len(perturbed_windows)}")
            print(f"  P mean: {perturbed_windows['P'].mean():.4f}")
            print(f"\nP shift: {perturbed_windows['P'].mean() - baseline_windows['P'].mean():.4f}")

    return metrics_df


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze trajectory CSV and compute information metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze with a baseline
    python analyze_trajectory.py logs_sac_claude/sac_seed3_force.csv --baseline sac3_baseline.json

    # Custom window settings
    python analyze_trajectory.py data.csv --baseline sac3_baseline.json --window 300 --stride 50

NOTE: You must first create a baseline using create_baseline.py:
    python create_baseline.py logs_sac_claude/clean_run.csv sac3_baseline.json
        """
    )

    parser.add_argument("csv_path", help="Path to trajectory CSV file")
    parser.add_argument("--baseline", required=True,
                        help="Path to baseline.json (REQUIRED - create with create_baseline.py)")
    parser.add_argument("--window", type=int, default=500, help="Window size in steps (default: 500)")
    parser.add_argument("--stride", type=int, default=50, help="Stride in steps (default: 50)")
    parser.add_argument("--output", default=None, help="Output path for metrics CSV (default: auto)")

    args = parser.parse_args()

    analyze_trajectory(
        csv_path=args.csv_path,
        baseline_path=args.baseline,
        window_size=args.window,
        stride=args.stride,
        output_path=args.output,
    )
if __name__ == "__main__":
    analyze_trajectory(
        csv_path="/logs_ppo_claude/ppo35_p/ppo_seed35_obs_noise_04_percent_from_ep15_20260102_232053.csv",
        baseline_path="sac3_baseline.json",
    )

