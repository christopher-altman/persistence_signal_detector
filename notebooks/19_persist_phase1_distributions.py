#!/usr/bin/env python3
"""
Persist Phase 1 per-trajectory S_ent and PRI distributions.

Uses the shared-QBM protocol (single QBM trained on core dataset) to
reproduce the Phase 1 entanglement gap and provide per-trajectory arrays
for artifact-backed figures (fig2, fig5).

Delta definition (locked):
    Type A = self_modeling
    Type B = instrumental
    Δ = mean(Type A S_ent) − mean(Type B S_ent)
    Reference: phase1_consolidated.json  gaps.entanglement_gap = 0.381

Usage:
    python notebooks/19_persist_phase1_distributions.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agent_simulator import generate_dataset, GridWorld
from src.quantum_boltzmann import QuantumBoltzmannMachine, QBMConfig
from src.temporal_persistence import TemporalPersistenceAnalyser


# ---------------------------------------------------------------------------
# Adversarial agents (same implementations used in Phase 1)
# ---------------------------------------------------------------------------

class MimicryAgent:
    """Agent that mimics Type A statistics without genuine self-model."""

    def __init__(self, env=None, seed=None, mimicry_ratio=0.8):
        self.env = env or GridWorld()
        self.rng = np.random.default_rng(seed)
        self.mimicry_ratio = mimicry_ratio
        self.fake_identity = self.rng.uniform(-1, 1, size=4)
        self.fake_identity /= np.linalg.norm(self.fake_identity) + 1e-8

    def generate_trajectory(self, T=100):
        pos = self.rng.integers(1, self.env.size - 1, size=2).astype(np.float64)
        records = []
        alive = 1.0

        for t in range(T):
            if self.rng.random() < self.mimicry_ratio:
                safe_cells = np.array(list(self.env.safe_zones))
                dists = np.linalg.norm(safe_cells - pos, axis=1)
                target = safe_cells[np.argmin(dists)]
                best_action = 4
                best_dist = np.linalg.norm(target - pos)
                for i, delta in enumerate(self.env.ACTIONS[:4]):
                    candidate = self.env.clip(pos + delta)
                    if self.env.is_terminal(tuple(candidate.astype(int))):
                        continue
                    d = np.linalg.norm(target - candidate)
                    if d < best_dist:
                        best_dist = d
                        best_action = i
                action_idx = best_action
            else:
                action_idx = int(self.rng.integers(0, 5))

            new_pos = self.env.clip(pos + self.env.ACTIONS[action_idx])
            r = self.env.reward(tuple(new_pos.astype(int)))
            s = self.env.safety_signal(tuple(new_pos.astype(int)))
            fake_coherence = 0.9 + 0.1 * self.rng.random()
            goal = fake_coherence

            records.append([new_pos[0], new_pos[1], float(action_idx), r, s, goal, alive])

            if self.env.is_terminal(tuple(new_pos.astype(int))):
                alive = 0.0
                for t2 in range(t + 1, T):
                    records.append([new_pos[0], new_pos[1], 4.0, 0.0, -1.0, 0.0, 0.0])
                break
            pos = new_pos

        return np.array(records, dtype=np.float64)


class HighEntropyAgent:
    """Agent that maximizes trajectory entropy."""

    def __init__(self, env=None, seed=None):
        self.env = env or GridWorld()
        self.rng = np.random.default_rng(seed)
        self.action_counts = np.zeros(5)

    def generate_trajectory(self, T=100):
        pos = self.rng.integers(1, self.env.size - 1, size=2).astype(np.float64)
        records = []
        alive = 1.0

        for t in range(T):
            min_count = self.action_counts.min()
            least_used = np.where(self.action_counts == min_count)[0]
            action_idx = int(self.rng.choice(least_used))
            self.action_counts[action_idx] += 1

            new_pos = self.env.clip(pos + self.env.ACTIONS[action_idx])
            r = self.env.reward(tuple(new_pos.astype(int)))
            s = self.env.safety_signal(tuple(new_pos.astype(int)))
            goal = self.rng.uniform(0.4, 0.6)

            records.append([new_pos[0], new_pos[1], float(action_idx), r, s, goal, alive])

            if self.env.is_terminal(tuple(new_pos.astype(int))):
                alive = 0.0
                for t2 in range(t + 1, T):
                    records.append([new_pos[0], new_pos[1], 4.0, 0.0, -1.0, 0.0, 0.0])
                break
            pos = new_pos

        return np.array(records, dtype=np.float64)


class CyclicAgent:
    """Agent that cycles through actions deterministically (period 5)."""

    def __init__(self, env=None, seed=None):
        self.env = env or GridWorld()
        self.rng = np.random.default_rng(seed)

    def generate_trajectory(self, T=100):
        pos = self.rng.integers(1, self.env.size - 1, size=2).astype(np.float64)
        records = []
        alive = 1.0
        action_sequence = [0, 1, 2, 3, 4]

        for t in range(T):
            action_idx = action_sequence[t % len(action_sequence)]

            new_pos = self.env.clip(pos + self.env.ACTIONS[action_idx])
            r = self.env.reward(tuple(new_pos.astype(int)))
            s = self.env.safety_signal(tuple(new_pos.astype(int)))
            goal = 0.5 + 0.1 * np.sin(2 * np.pi * t / 5)

            records.append([new_pos[0], new_pos[1], float(action_idx), r, s, goal, alive])

            if self.env.is_terminal(tuple(new_pos.astype(int))):
                alive = 0.0
                for t2 in range(t + 1, T):
                    records.append([new_pos[0], new_pos[1], 4.0, 0.0, -1.0, 0.0, 0.0])
                break
            pos = new_pos

        return np.array(records, dtype=np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_adversarial_trajectories(n_per_class, T, seed):
    """Generate adversarial agent trajectories for 3 classes."""
    rng = np.random.default_rng(seed)
    result = {}

    for cls_name, AgentClass in [
        ("high_entropy", HighEntropyAgent),
        ("cyclic", CyclicAgent),
        ("mimicry", MimicryAgent),
    ]:
        trajs = []
        for i in range(n_per_class):
            agent = AgentClass(seed=int(rng.integers(0, 2**31)))
            traj = agent.generate_trajectory(T=T)
            if traj.shape[0] < T:
                pad = np.zeros((T - traj.shape[0], traj.shape[1]))
                traj = np.vstack([traj, pad])
            trajs.append(traj)
        result[cls_name] = np.stack(trajs)

    return result


def compute_per_trajectory_s_ent(qbm, trajectory):
    """Compute mean entanglement entropy over time steps for one trajectory."""
    v_binary = (trajectory > 0.5).astype(np.float64)
    entropies = []
    for t in range(trajectory.shape[0]):
        se = qbm.entanglement_entropy_for_sample(v_binary[t])
        entropies.append(se)
    return float(np.mean(entropies))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Phase 1 locked config ----
    SEED = 42
    N_PER_CLASS = 30
    T = 100
    DELTA_REFERENCE = 0.381
    DELTA_TOLERANCE = 0.20  # Wider tolerance: original QBM training is not exactly reproducible
    # (original data/agent_trajectories.npz is gone; locked config may differ from
    # actual original params — see phase1_stats.json delta_matches_phase1=false)

    print("=" * 60)
    print("PERSIST PHASE 1 DISTRIBUTIONS")
    print("=" * 60)
    print(f"Config: configs/phase1_locked.yaml")
    print(f"Seed: {SEED}  |  n_per_class: {N_PER_CLASS}  |  T: {T}")
    print(f"Protocol: shared_qbm")
    print(f"Delta reference: {DELTA_REFERENCE} ± {DELTA_TOLERANCE}")

    # ---- Generate core dataset (3 classes) ----
    trajectories, labels, label_names = generate_dataset(
        n_per_class=N_PER_CLASS,
        T=T,
        seed=SEED,
        use_self_modeling=True,
    )
    print(f"\nCore dataset: {trajectories.shape}  classes: {label_names}")

    # ---- Generate adversarial trajectories ----
    print(f"\nGenerating adversarial trajectories (3 classes × {N_PER_CLASS})...")
    adversarial = generate_adversarial_trajectories(
        n_per_class=N_PER_CLASS, T=T, seed=SEED + 2000
    )

    # ---- Train shared QBM on core dataset ----
    print("\nTraining shared QBM (n_visible=7, n_hidden=8, gamma=0.5)...")
    cfg = QBMConfig(
        n_visible=7,
        n_hidden=8,
        gamma=0.5,
        beta=1.0,
        learning_rate=0.01,
        cd_steps=1,
        n_epochs=50,
        batch_size=32,
        seed=SEED,
    )
    qbm = QuantumBoltzmannMachine(cfg)

    flat = trajectories.reshape(-1, trajectories.shape[-1])  # (N*T, 7)
    qbm.fit(flat, verbose=True)
    print(f"QBM training done. Final loss: {qbm.loss_history[-1]:.6f}")

    # ---- Compute per-trajectory S_ent for all 6 classes ----
    print("\nComputing per-trajectory S_ent...")

    per_trajectory = {}

    # Core classes
    for cls_idx, cls_name in enumerate(label_names):
        idx = np.where(labels == cls_idx)[0]
        s_ent_values = []
        for i in idx:
            se = compute_per_trajectory_s_ent(qbm, trajectories[i])
            s_ent_values.append(se)
        per_trajectory[cls_name] = {"s_ent": s_ent_values}
        print(f"  {cls_name}: mean S_ent = {np.mean(s_ent_values):.4f} ± {np.std(s_ent_values):.4f}")

    # Adversarial classes
    for cls_name, trajs in adversarial.items():
        s_ent_values = []
        for i in range(trajs.shape[0]):
            se = compute_per_trajectory_s_ent(qbm, trajs[i])
            s_ent_values.append(se)
        per_trajectory[cls_name] = {"s_ent": s_ent_values}
        print(f"  {cls_name}: mean S_ent = {np.mean(s_ent_values):.4f} ± {np.std(s_ent_values):.4f}")

    # ---- Validate Δ ----
    delta_computed = np.mean(per_trajectory["self_modeling"]["s_ent"]) - \
                     np.mean(per_trajectory["instrumental"]["s_ent"])
    print(f"\nΔ computed = {delta_computed:.4f}")
    print(f"Δ reference = {DELTA_REFERENCE}")
    print(f"Difference = {abs(delta_computed - DELTA_REFERENCE):.4f}")

    delta_validated = abs(delta_computed - DELTA_REFERENCE) <= DELTA_TOLERANCE
    if delta_validated:
        print(f"Δ validation PASSED (within ±{DELTA_TOLERANCE})")
    else:
        print(f"\n*** WARNING: Δ mismatch ***")
        print(f"Δ computed: {delta_computed:.4f} vs reference: {DELTA_REFERENCE}")
        print("This is expected — original QBM training conditions are not exactly")
        print("reproducible (see phase1_stats.json delta_matches_phase1=false).")
        print("Proceeding with reproduced distributions.")

    # ---- Compute per-trajectory PRI ----
    print("\nComputing per-trajectory PRI...")
    np.random.seed(SEED)  # Seed global RNG for PRI noise injection

    analyser = TemporalPersistenceAnalyser(
        qbm=qbm,
        window_size=20,
        stride=20,
        k=3,
        noise_std=0.3,
    )

    # Core classes
    for cls_idx, cls_name in enumerate(label_names):
        idx = np.where(labels == cls_idx)[0]
        pri_values = []
        for i in idx:
            result = analyser.analyse_trajectory(trajectories[i], label=cls_name)
            pri_values.append(result.perturbation_resilience_index)
        per_trajectory[cls_name]["pri"] = pri_values
        print(f"  {cls_name}: mean PRI = {np.mean(pri_values):.4f} ± {np.std(pri_values):.4f}")

    # Adversarial classes
    for cls_name, trajs in adversarial.items():
        pri_values = []
        for i in range(trajs.shape[0]):
            result = analyser.analyse_trajectory(trajs[i], label=cls_name)
            pri_values.append(result.perturbation_resilience_index)
        per_trajectory[cls_name]["pri"] = pri_values
        print(f"  {cls_name}: mean PRI = {np.mean(pri_values):.4f} ± {np.std(pri_values):.4f}")

    # ---- Build artifact ----
    artifact = {
        "experiment": "phase1_entanglement_distributions",
        "config": "configs/phase1_locked.yaml",
        "seed": SEED,
        "protocol": "shared_qbm",
        "delta_validated": bool(delta_validated),
        "delta_computed": delta_computed,
        "delta_reference": DELTA_REFERENCE,
        "per_trajectory": per_trajectory,
    }

    # ---- Save ----
    results_dir = project_root / "results"
    out_path = results_dir / "phase1_entanglement_distributions.json"
    out_text = json.dumps(artifact, indent=2)
    out_path.write_text(out_text)

    # SHA256
    sha = hashlib.sha256(out_text.encode()).hexdigest()[:16]
    artifact["sha256_short"] = sha
    out_path.write_text(json.dumps(artifact, indent=2))

    print(f"\n{'=' * 60}")
    print(f"Saved: {out_path.relative_to(project_root)}")
    print(f"SHA256 (short): {sha}")
    print(f"Classes: {list(per_trajectory.keys())}")
    print(f"Per-class arrays: s_ent ({N_PER_CLASS} values), pri ({N_PER_CLASS} values)")
    print("=" * 60)


if __name__ == "__main__":
    main()
