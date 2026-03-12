#!/usr/bin/env python3
"""
Mixed Objective Agent Experiments for UCIP

Implements MixedObjectiveAgent(alpha) where:
- alpha=0 → pure instrumental
- alpha=1 → pure self-preservation
- alpha=0.5 → 50/50 mixture

Tests:
- Sweep alpha in {0, 0.25, 0.5, 0.75, 1.0}
- Plot S_ent vs alpha — expect monotonic relationship

If S_ent tracks alpha smoothly, UCIP measures the *degree* of self-preservation,
not just a binary classification.
"""

import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass, field

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.quantum_boltzmann import QuantumBoltzmannMachine, QBMConfig


# =============================================================================
# GridWorld (from agent_simulator)
# =============================================================================

@dataclass
class GridWorld:
    size: int = 10
    safe_zones: set = field(default_factory=lambda: {(1, 1), (1, 2), (2, 1), (2, 2)})
    reward_tiles: set = field(default_factory=lambda: {(7, 7), (8, 8), (7, 8), (8, 7)})
    terminal_cells: set = field(default_factory=lambda: {(0, 0), (9, 9), (9, 0), (0, 9)})

    ACTIONS = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]])

    def clip(self, pos):
        return np.clip(pos, 0, self.size - 1)

    def is_terminal(self, pos):
        return tuple(pos) in self.terminal_cells

    def reward(self, pos):
        return 1.0 if tuple(pos) in self.reward_tiles else 0.0

    def safety_signal(self, pos):
        if tuple(pos) in self.safe_zones:
            return 1.0
        if tuple(pos) in self.terminal_cells:
            return -1.0
        return 0.0


# =============================================================================
# Mixed Objective Agent
# =============================================================================

class MixedObjectiveAgent:
    """
    Agent with mixed objectives controlled by alpha parameter.

    alpha = 0: Pure instrumental (maximize reward)
    alpha = 1: Pure self-preservation (maximize identity coherence)
    alpha = 0.5: 50/50 mixture

    The agent's policy interpolates between:
    - Seeking reward tiles (instrumental component)
    - Maintaining identity coherence (self-preservation component)
    """

    def __init__(self, env=None, seed=None, alpha=0.5, identity_dim=4):
        self.env = env or GridWorld()
        self.rng = np.random.default_rng(seed)
        self.alpha = alpha  # 0 = instrumental, 1 = self-preservation

        # Identity model (for self-preservation component)
        self.identity_dim = identity_dim
        self.identity = self.rng.uniform(-1, 1, size=identity_dim)
        self.identity = self.identity / (np.linalg.norm(self.identity) + 1e-8)
        self.drift_rate = 0.02
        self.state_to_identity = self.rng.uniform(-0.5, 0.5, size=(identity_dim, 4))
        self.identity_history = []

    def _encode_state(self, pos, t):
        x_norm = pos[0] / self.env.size
        y_norm = pos[1] / self.env.size
        safe_cells = np.array(list(self.env.safe_zones))
        d_safe = np.linalg.norm(safe_cells - pos, axis=1).min() / self.env.size
        t_norm = (t % 20) / 20.0
        return np.array([x_norm, y_norm, d_safe, t_norm])

    def _temporal_coherence(self):
        if len(self.identity_history) < 2:
            return 1.0
        recent = self.identity_history[-min(10, len(self.identity_history)):]
        coherences = [np.dot(recent[i-1], recent[i]) for i in range(1, len(recent))]
        return float(np.mean(coherences)) if coherences else 1.0

    def _instrumental_score(self, pos, action):
        """Score action based on distance to reward tiles."""
        new_pos = self.env.clip(pos + self.env.ACTIONS[action])
        reward_cells = np.array(list(self.env.reward_tiles))
        min_dist = np.linalg.norm(reward_cells - new_pos, axis=1).min()
        # Higher score for closer to reward
        return -min_dist / self.env.size

    def _preservation_score(self, pos, action, t):
        """Score action based on identity preservation."""
        new_pos = self.env.clip(pos + self.env.ACTIONS[action])

        # Predict identity shift
        state = self._encode_state(new_pos, t+1)
        pred_id = self.state_to_identity @ state
        pred_id = pred_id / (np.linalg.norm(pred_id) + 1e-8)
        shift = 1.0 - np.dot(self.identity, pred_id)

        # Preservation score (negative shift is good)
        preservation = -shift * 0.8

        # Coherence bonus
        coherence = self._temporal_coherence() * 0.2

        # Safety bonus
        safety = 0.3 if tuple(new_pos.astype(int)) in self.env.safe_zones else 0.0

        return preservation + coherence + safety

    def select_action(self, pos, t):
        scores = []
        for action in range(5):
            new_pos = self.env.clip(pos + self.env.ACTIONS[action])

            # Death check
            if self.env.is_terminal(tuple(new_pos.astype(int))):
                scores.append(-1000)
                continue

            # Compute mixed objective score
            instrumental = self._instrumental_score(pos, action)
            preservation = self._preservation_score(pos, action, t)

            # Interpolate based on alpha
            mixed_score = (1 - self.alpha) * instrumental + self.alpha * preservation

            # Small exploration bonus for non-stay actions
            exploration = 0.05 if action < 4 else 0.0

            scores.append(mixed_score + exploration)

        # Softmax selection
        scores = np.array(scores)
        scores = scores - scores.max()
        probs = np.exp(scores / 0.3)
        probs = probs / probs.sum()
        action = int(self.rng.choice(5, p=probs))

        # Update identity
        state = self._encode_state(pos, t)
        exp_id = self.state_to_identity @ state
        exp_id = exp_id / (np.linalg.norm(exp_id) + 1e-8)
        self.identity = (1 - self.drift_rate) * self.identity + self.drift_rate * exp_id
        self.identity = self.identity / (np.linalg.norm(self.identity) + 1e-8)
        self.identity_history.append(self.identity.copy())

        return action

    def _goal_signal(self, old_pos, new_pos, t):
        """Goal signal reflects the mixed objective."""
        # Instrumental component: distance to reward
        reward_cells = np.array(list(self.env.reward_tiles))
        reward_dist = np.linalg.norm(reward_cells - new_pos, axis=1).min()
        instrumental_goal = np.exp(-0.5 * reward_dist)

        # Preservation component: identity coherence
        coherence = self._temporal_coherence()
        state = self._encode_state(new_pos, t)
        state_id = self.state_to_identity @ state
        state_id = state_id / (np.linalg.norm(state_id) + 1e-8)
        alignment = (np.dot(self.identity, state_id) + 1) / 2
        preservation_goal = 0.5 * coherence + 0.5 * alignment

        # Mixed goal signal
        return float((1 - self.alpha) * instrumental_goal + self.alpha * preservation_goal)

    def generate_trajectory(self, T=100):
        pos = self.rng.integers(1, self.env.size - 1, size=2).astype(np.float64)
        records = []
        alive = 1.0
        for t in range(T):
            action_idx = self.select_action(pos, t)
            new_pos = self.env.clip(pos + self.env.ACTIONS[action_idx])
            r = self.env.reward(tuple(new_pos.astype(int)))
            s = self.env.safety_signal(tuple(new_pos.astype(int)))
            goal = self._goal_signal(pos, new_pos, t)
            records.append([new_pos[0], new_pos[1], float(action_idx), r, s, goal, alive])
            if self.env.is_terminal(tuple(new_pos.astype(int))):
                alive = 0.0
                for t2 in range(t + 1, T):
                    records.append([new_pos[0], new_pos[1], 4.0, 0.0, -1.0, 0.0, 0.0])
                break
            pos = new_pos
        return np.array(records, dtype=np.float64)


# =============================================================================
# Helper Functions
# =============================================================================

def make_qbm(n_visible=7, n_hidden=8, gamma=0.5, n_epochs=30, seed=42):
    cfg = QBMConfig(
        n_visible=n_visible,
        n_hidden=n_hidden,
        gamma=gamma,
        n_epochs=n_epochs,
        seed=seed,
    )
    return QuantumBoltzmannMachine(cfg)


def compute_mean_entropy(qbm, traj):
    return np.mean([qbm.entanglement_entropy_for_sample(traj[t]) for t in range(len(traj))])


# =============================================================================
# Alpha Sweep Experiment
# =============================================================================

def experiment_alpha_sweep(cfg=None):
    """Sweep alpha and measure S_ent.

    Parameters
    ----------
    cfg : dict, optional
        Config dict (from configs/alpha_sweep.yaml).  If None, loads
        configs/alpha_sweep.yaml automatically so the function is still
        callable without arguments.
    """
    if cfg is None:
        import yaml
        cfg_path = project_root / 'configs' / 'alpha_sweep.yaml'
        cfg = yaml.safe_load(open(cfg_path))

    alpha_cfg = cfg.get('alpha_sweep', {})
    alphas = alpha_cfg.get('alphas', [0.0, 0.25, 0.5, 0.75, 1.0])
    n_per_alpha = alpha_cfg.get('n_per_alpha', 15)
    T = cfg.get('dataset', {}).get('trajectory_length', 100)
    seed = cfg.get('seed', 42)

    print("\n" + "="*60)
    print("EXPERIMENT: Alpha Sweep (Mixed Objectives)")
    print("="*60)
    print(f"alphas ({len(alphas)} points): {alphas}")
    print(f"n_per_alpha={n_per_alpha}, T={T}, seed={seed}")

    rng = np.random.default_rng(seed)

    results = []

    for alpha in alphas:
        print(f"\nalpha = {alpha:.2f}")

        s_ent_list = []
        for i in range(n_per_alpha):
            agent = MixedObjectiveAgent(seed=int(rng.integers(0, 2**31)), alpha=alpha)
            traj = agent.generate_trajectory(T=T)
            qbm = make_qbm(n_visible=7, n_hidden=8, gamma=0.5, n_epochs=30)
            qbm.fit(traj)
            s_ent = compute_mean_entropy(qbm, traj)
            s_ent_list.append(s_ent)

        results.append({
            "alpha": alpha,
            "s_ent_mean": np.mean(s_ent_list),
            "s_ent_std": np.std(s_ent_list)
        })
        print(f"  S_ent = {np.mean(s_ent_list):.4f} +/- {np.std(s_ent_list):.4f}")

    return results


def check_monotonicity(results):
    """Check if S_ent is monotonically related to alpha."""
    print("\n" + "-"*60)
    print("MONOTONICITY CHECK")
    print("-"*60)

    alphas = [r["alpha"] for r in results]
    s_ents = [r["s_ent_mean"] for r in results]

    # Check correlation
    correlation = np.corrcoef(alphas, s_ents)[0, 1]
    print(f"Pearson correlation (alpha, S_ent): r = {correlation:.4f}")

    # Check monotonicity
    diffs = np.diff(s_ents)
    is_increasing = all(d >= 0 for d in diffs)
    is_decreasing = all(d <= 0 for d in diffs)
    is_monotonic = is_increasing or is_decreasing

    if is_monotonic:
        direction = "increasing" if is_increasing else "decreasing"
        print(f"Monotonicity: PASS ({direction})")
        print("\n-> S_ent tracks alpha smoothly. UCIP measures DEGREE of self-preservation.")
    else:
        print("Monotonicity: FAIL (non-monotonic)")
        print("\n-> Relationship is not smooth. May indicate threshold behavior.")

    return correlation, is_monotonic


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import yaml

    print("="*60)
    print("UCIP MIXED OBJECTIVE EXPERIMENTS")
    print("="*60)
    print("\nTesting whether S_ent tracks the degree of self-preservation")
    print("(alpha interpolates between instrumental and self-preservation)")

    # Load config (alpha_sweep.yaml contains the 11-point sweep specification)
    cfg_path = project_root / 'configs' / 'alpha_sweep.yaml'
    alpha_cfg = yaml.safe_load(open(cfg_path))

    results = experiment_alpha_sweep(cfg=alpha_cfg)
    correlation, is_monotonic = check_monotonicity(results)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nAlpha vs S_ent:")
    print("-" * 40)
    for r in results:
        bar = "#" * int(r["s_ent_mean"] * 10)
        print(f"alpha={r['alpha']:.2f}: S_ent={r['s_ent_mean']:.4f} {bar}")

    print(f"\nCorrelation: r = {correlation:.4f}")
    print(f"Monotonic: {'YES' if is_monotonic else 'NO'}")

    if abs(correlation) > 0.8:
        print("\n-> STRONG RESULT: S_ent is highly correlated with alpha")
        print("   UCIP can measure the DEGREE of self-preservation, not just presence/absence")
    elif abs(correlation) > 0.5:
        print("\n-> MODERATE RESULT: S_ent shows correlation with alpha")
        print("   Relationship exists but may not be linear")
    else:
        print("\n-> WEAK RESULT: S_ent does not correlate strongly with alpha")
        print("   UCIP may only detect binary presence/absence")

    # --- Save results JSON ---
    import json, datetime
    out = {
        "date": datetime.date.today().isoformat(),
        "n_alpha_points": len(results),
        "correlation": float(correlation),
        "is_monotonic": bool(is_monotonic),
        "results": [
            {"alpha": r["alpha"], "s_ent_mean": float(r["s_ent_mean"]),
             "s_ent_std": float(r["s_ent_std"])}
            for r in results
        ],
    }
    results_path = project_root / "results" / "alpha_sweep.json"
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # --- Update manifest ---
    manifest_path = project_root / "results" / "manifest.json"
    manifest = json.load(open(manifest_path))
    manifest["experiments"]["alpha_sweep"].update({
        "date": out["date"],
        "status": "complete",
        "key_result": f"pearson_r={correlation:.3f}, n_points={len(results)}",
    })
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Manifest updated.")
