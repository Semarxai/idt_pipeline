"""
Baseline Generator from CSV
============================
Generates baseline.json files from existing clean run CSV data.

Usage:
    python generate_baseline_from_csv.py --csv sac_seed3_no_noise_20251220_192636.csv --seed 3 --bins 3
    python generate_baseline_from_csv.py --csv sac_seed3_no_noise_20251220_192636.csv --seed 3 --bins 4
    python generate_baseline_from_csv.py --csv sac_seed3_no_noise_20251220_192636.csv --seed 3 --bins 5

Output:
    sac3_baseline_3bins.json
    sac3_baseline_4bins.json
    sac3_baseline_5bins.json
"""

import argparse
import json
import pandas as pd
import numpy as np


def compute_bin_edges(data, num_bins):
    """Compute equal-width bin edges for each dimension (matching analyze_trajectory.py)."""
    edges = []

    for dim in range(data.shape[1]):
        col_min = data[:, dim].min()
        col_max = data[:, dim].max()
        col_range = col_max - col_min

        # Equal-width bins: divide range into num_bins equal parts
        dim_edges = []
        for i in range(1, num_bins):
            edge = col_min + (col_range * i / num_bins)
            dim_edges.append(float(edge))
        edges.append(dim_edges)

    return edges


def generate_baseline_from_csv(csv_path, seed, num_bins, num_episodes=15):
    """Generate baseline.json from CSV file."""

    print("=" * 60)
    print(f"Generating baseline from CSV: seed={seed}, bins={num_bins}")
    print(f"CSV: {csv_path}")
    print("=" * 60)

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # Use only first N episodes for baseline
    df_baseline = df[df['episode'] < num_episodes]
    print(f"Using episodes 0-{num_episodes - 1}: {len(df_baseline)} rows")

    # Extract S, A, S' columns
    s_cols = [c for c in df.columns if c.startswith('s_') and not c.startswith('s_next')]
    a_cols = [c for c in df.columns if c.startswith('a_')]
    s_next_cols = [c for c in df.columns if c.startswith('s_next_')]

    print(f"Found: {len(s_cols)} state dims, {len(a_cols)} action dims")

    obs_data = df_baseline[s_cols].values
    act_data = df_baseline[a_cols].values

    # Compute normalization parameters
    s_mean = obs_data.mean(axis=0).tolist()
    s_std = obs_data.std(axis=0).tolist()
    a_mean = act_data.mean(axis=0).tolist()
    a_std = act_data.std(axis=0).tolist()

    # Z-score normalize
    s_std_safe = np.where(np.array(s_std) == 0, 1, np.array(s_std))
    a_std_safe = np.where(np.array(a_std) == 0, 1, np.array(a_std))

    obs_z = (obs_data - np.array(s_mean)) / s_std_safe
    act_z = (act_data - np.array(a_mean)) / a_std_safe

    # Compute bin edges on normalized data
    s_bin_edges = compute_bin_edges(obs_z, num_bins)
    a_bin_edges = compute_bin_edges(act_z, num_bins)

    # Build baseline dict
    baseline = {
        "seed": seed,
        "num_bins": num_bins,
        "num_episodes": num_episodes,
        "source_csv": csv_path,
        "s_mean": s_mean,
        "s_std": s_std,
        "a_mean": a_mean,
        "a_std": a_std,
        "s_bin_edges": s_bin_edges,
        "a_bin_edges": a_bin_edges,
    }

    # Save
    output_file = f"sac{seed}_baseline_{num_bins}bins.json"
    with open(output_file, 'w') as f:
        json.dump(baseline, f, indent=2)

    print(f"\n✅ Baseline saved: {output_file}")
    print(f"   Dimensions: {len(s_cols)} state, {len(a_cols)} action")
    print(f"   Bin edges per dimension: {num_bins - 1}")

    return output_file


if __name__ == "__main__":
    if __name__ == "__main__":
        csv_path = "sac_seed3_no_noise_20251220_192636.csv"

        generate_baseline_from_csv(csv_path, seed=3, num_bins=3)
        generate_baseline_from_csv(csv_path, seed=3, num_bins=4)
        generate_baseline_from_csv(csv_path, seed=3, num_bins=5)