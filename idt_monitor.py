"""
Real-Time IDT Monitor
=====================
Computes P in real-time from rolling buffer of (S, A, S') tuples.
Triggers intervention when P crosses threshold.

Usage:
    from idt_monitor import RealTimeIDTMonitor

    monitor = RealTimeIDTMonitor(
        baseline_path="sac3_baseline.json",
        buffer_size=500,
        p_threshold_low=0.40,   # Enable intervention below this
        p_threshold_high=0.45,  # Disable intervention above this
    )

    # In evaluation loop:
    monitor.add(obs, action, next_obs)

    if step % 50 == 0:
        P = monitor.compute_P()
        should_intervene = monitor.should_intervene()
"""

import json
import numpy as np
from collections import deque
from typing import Optional, Dict, Tuple


class RealTimeIDTMonitor:
    """
    Real-time monitor for computing P from rolling buffer.

    Maintains a sliding window of recent (S, A, S') transitions
    and computes information-theoretic metrics on demand.
    """

    def __init__(
            self,
            baseline_path: str,
            buffer_size: int = 500,
            p_threshold_low: float = 0.40,
            p_threshold_high: float = 0.45,
            compute_interval: int = 50,
    ):
        """
        Args:
            baseline_path: Path to baseline.json with bin edges
            buffer_size: Number of transitions to keep in buffer
            p_threshold_low: Enable intervention when P drops below this
            p_threshold_high: Disable intervention when P rises above this
            compute_interval: Compute P every N steps
        """
        self.buffer_size = buffer_size
        self.p_threshold_low = p_threshold_low
        self.p_threshold_high = p_threshold_high
        self.compute_interval = compute_interval

        # Load baseline
        self.baseline = self._load_baseline(baseline_path)

        # Rolling buffer: stores (s, a, s_next) tuples
        self.buffer = deque(maxlen=buffer_size)

        # State tracking
        self.step_count = 0
        self.current_P = None
        self.intervention_active = False

        # History for logging
        self.P_history = []

        print(f"IDTMonitor: buffer={buffer_size}, thresholds=({p_threshold_low}, {p_threshold_high})")

    def _load_baseline(self, path: str) -> dict:
        """Load baseline.json with bin edges and normalization params."""
        with open(path, 'r') as f:
            baseline = json.load(f)
        num_bins = baseline.get('num_bins', 3)
        print(f"IDTMonitor: Loaded baseline from {path} ({num_bins} bins)")
        return baseline

    def reset(self):
        """Reset buffer and state for new episode."""
        self.buffer.clear()
        self.step_count = 0
        self.current_P = None
        # Don't reset intervention_active - carry over between episodes

    def add(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray):
        """Add transition to buffer."""
        # Flatten if needed
        if obs.ndim > 1:
            obs = obs.flatten()
        if action.ndim > 1:
            action = action.flatten()
        if next_obs.ndim > 1:
            next_obs = next_obs.flatten()

        self.buffer.append((obs.copy(), action.copy(), next_obs.copy()))
        self.step_count += 1

    def _z_score(self, data: np.ndarray, mean: list, std: list) -> np.ndarray:
        """Z-score normalize data."""
        mean = np.array(mean)
        std = np.array(std)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        return (data - mean) / std

    def _discretize(self, values: np.ndarray, bin_edges: list) -> np.ndarray:
        """Discretize values using bin edges (matching analyze_trajectory.py)."""
        # bin_edges is list of edges for each dimension
        # Returns bins 1, 2, 3, ... (matching analyze_trajectory.py)
        result = np.zeros(values.shape[0], dtype=int)
        for i, (val, edges) in enumerate(zip(values, bin_edges)):
            # Find which bin: 1 if <= edge[0], 2 if <= edge[1], etc.
            bin_idx = 1
            for edge in edges:
                if val > edge:
                    bin_idx += 1
                else:
                    break
            result[i] = bin_idx
        return result

    def _entropy(self, labels: np.ndarray) -> float:
        """Compute entropy of discrete distribution."""
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _joint_entropy(self, labels1: np.ndarray, labels2: np.ndarray) -> float:
        """Compute joint entropy of two discrete variables."""
        joint = np.array([f"{a}|{b}" for a, b in zip(labels1, labels2)])
        return self._entropy(joint)

    def _triple_joint_entropy(self, l1: np.ndarray, l2: np.ndarray, l3: np.ndarray) -> float:
        """Compute joint entropy of three discrete variables."""
        joint = np.array([f"{a}|{b}|{c}" for a, b, c in zip(l1, l2, l3)])
        return self._entropy(joint)

    def _compose_state(self, s_discrete: np.ndarray) -> str:
        """Compose state into body-part groups (matching analyze_trajectory.py)."""
        # back_leg: indices 2, 3, 4, 11, 12, 13
        back_leg = ''.join(map(str, [s_discrete[i] for i in [2, 3, 4, 11, 12, 13]]))
        # front_leg: indices 5, 6, 7, 14, 15, 16
        front_leg = ''.join(map(str, [s_discrete[i] for i in [5, 6, 7, 14, 15, 16]]))
        # tip: indices 0, 1, 8, 9, 10
        tip = ''.join(map(str, [s_discrete[i] for i in [0, 1, 8, 9, 10]]))
        return f"{back_leg}|{front_leg}|{tip}"

    def _compose_action(self, a_discrete: np.ndarray) -> str:
        """Compose action into string."""
        return ''.join(map(str, a_discrete))

    def compute_P(self) -> Optional[float]:
        """
        Compute P from current buffer.
        Uses body-part grouping matching analyze_trajectory.py.

        Returns:
            P value (0-1) or None if buffer too small
        """
        if len(self.buffer) < 100:  # Need minimum data
            return None

        # Extract arrays from buffer
        obs_list = []
        act_list = []
        next_obs_list = []

        for obs, act, next_obs in self.buffer:
            obs_list.append(obs)
            act_list.append(act)
            next_obs_list.append(next_obs)

        obs_arr = np.array(obs_list)
        act_arr = np.array(act_list)
        next_obs_arr = np.array(next_obs_list)

        # Z-score normalize
        obs_z = np.array([self._z_score(o, self.baseline['s_mean'], self.baseline['s_std']) for o in obs_arr])
        act_z = np.array([self._z_score(a, self.baseline['a_mean'], self.baseline['a_std']) for a in act_arr])
        next_obs_z = np.array([self._z_score(o, self.baseline['s_mean'], self.baseline['s_std']) for o in next_obs_arr])

        # Discretize and compose into S, A, S_next strings
        S_list = []
        A_list = []
        S_next_list = []

        for i in range(len(obs_z)):
            s_d = self._discretize(obs_z[i], self.baseline['s_bin_edges'])
            a_d = self._discretize(act_z[i], self.baseline['a_bin_edges'])
            s_next_d = self._discretize(next_obs_z[i], self.baseline['s_bin_edges'])

            # Compose using body-part grouping
            S_list.append(self._compose_state(s_d))
            A_list.append(self._compose_action(a_d))
            S_next_list.append(self._compose_state(s_next_d))

        S_arr = np.array(S_list)
        A_arr = np.array(A_list)
        S_next_arr = np.array(S_next_list)

        # Create joint distributions (matching analyze_trajectory.py)
        S_A = np.array([f"{s}||{a}" for s, a in zip(S_arr, A_arr)])
        S_A_Snext = np.array([f"{s}||{a}||{sn}" for s, a, sn in zip(S_arr, A_arr, S_next_arr)])

        # Compute entropies
        H_S = self._entropy(S_arr)
        H_A = self._entropy(A_arr)
        H_Snext = self._entropy(S_next_arr)
        H_S_A = self._entropy(S_A)
        H_S_A_Snext = self._entropy(S_A_Snext)

        # Compute P = MI(S,A;S') / H_Total
        H_Total = H_S + H_A + H_Snext
        MI_SA_Snext = H_S_A + H_Snext - H_S_A_Snext

        if H_Total > 0:
            P = MI_SA_Snext / H_Total
        else:
            P = 0.0

        self.current_P = P
        self.P_history.append((self.step_count, P))

        return P

    def should_intervene(self) -> bool:
        """
        Determine if intervention should be active based on P.
        Uses hysteresis to avoid oscillation.
        """
        if self.current_P is None:
            return self.intervention_active

        if self.current_P < self.p_threshold_low:
            self.intervention_active = True
        elif self.current_P > self.p_threshold_high:
            self.intervention_active = False
        # Between thresholds: keep current state (hysteresis)

        return self.intervention_active

    def should_compute(self) -> bool:
        """Check if it's time to compute P."""
        return self.step_count % self.compute_interval == 0 and self.step_count > 0

    def get_status(self) -> Dict:
        """Get current monitor status."""
        return {
            'step': self.step_count,
            'buffer_size': len(self.buffer),
            'P': self.current_P,
            'intervention_active': self.intervention_active,
        }

    def get_P_history(self) -> list:
        """Get history of P values for plotting."""
        return self.P_history