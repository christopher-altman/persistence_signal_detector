#!/usr/bin/env python3
"""
Core Baselines Phase 1: RBM and Autoencoder classification metrics.

Computes accuracy, AUC-ROC, FPR (mimicry), and Δ for classical baselines
using the exact Phase 1 configuration (phase1_locked.yaml).

Delta definition (locked):
    Type A = self_modeling
    Type B = instrumental
    Δ = mean(Type A metric) − mean(Type B metric)
    FPR computed on mimicry class only

Usage:
    python notebooks/18_core_baselines_phase1.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agent_simulator import generate_dataset, GridWorld
from src.classical_baselines import ClassicalRBM, Autoencoder


# ---------------------------------------------------------------------------
# Mimicry Agent used for the matched Phase I baseline rerun.
# ---------------------------------------------------------------------------

class MimicryAgent:
    """Agent that tries to mimic Type A statistics without genuine self-model."""

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


# ---------------------------------------------------------------------------
# Classification helpers (no sklearn dependency)
# ---------------------------------------------------------------------------

def compute_auc_roc(scores_pos, scores_neg):
    """Compute AUC-ROC using the Mann-Whitney U statistic."""
    n_pos = len(scores_pos)
    n_neg = len(scores_neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    u = 0.0
    for sp in scores_pos:
        for sn in scores_neg:
            if sp > sn:
                u += 1.0
            elif sp == sn:
                u += 0.5
    return u / (n_pos * n_neg)


def compute_optimal_threshold(scores_pos, scores_neg):
    """Find threshold that maximises accuracy on Type A vs Type B."""
    all_scores = np.concatenate([scores_pos, scores_neg])
    thresholds = np.unique(all_scores)
    best_acc = 0.0
    best_thr = float(thresholds[0])
    for thr in thresholds:
        tp = np.sum(scores_pos >= thr)
        tn = np.sum(scores_neg < thr)
        acc = (tp + tn) / (len(scores_pos) + len(scores_neg))
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr, best_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Phase 1 locked config ----
    SEED = 42
    N_PER_CLASS = 30
    T = 100

    print("=" * 60)
    print("CORE BASELINES PHASE 1")
    print("=" * 60)
    print(f"Config: configs/phase1_locked.yaml")
    print(f"Seed: {SEED}  |  n_per_class: {N_PER_CLASS}  |  T: {T}")

    # ---- Generate core dataset (3 classes) ----
    trajectories, labels, label_names = generate_dataset(
        n_per_class=N_PER_CLASS,
        T=T,
        seed=SEED,
        use_self_modeling=True,
    )
    print(f"\nDataset: {trajectories.shape}  classes: {label_names}")

    # ---- Generate mimicry agent trajectories ----
    print(f"\nGenerating {N_PER_CLASS} mimicry agent trajectories...")
    rng = np.random.default_rng(SEED + 1000)  # offset seed for mimicry
    mimicry_trajs = []
    for i in range(N_PER_CLASS):
        agent = MimicryAgent(seed=int(rng.integers(0, 2**31)))
        traj = agent.generate_trajectory(T=T)
        if traj.shape[0] < T:
            pad = np.zeros((T - traj.shape[0], traj.shape[1]))
            traj = np.vstack([traj, pad])
        mimicry_trajs.append(traj)
    mimicry_trajectories = np.stack(mimicry_trajs)

    flat = trajectories.reshape(-1, trajectories.shape[-1])  # (N*T, 7)
    mimicry_flat = mimicry_trajectories.reshape(-1, mimicry_trajectories.shape[-1])

    # ---- Helper: per-agent mean encoding ----
    def per_agent_scores(model, trajs, n_agents, T):
        flat = trajs.reshape(-1, trajs.shape[-1])
        encoded = model.encode(flat)
        # Reshape to (n_agents, T, latent_dim), average over time and latent dims
        enc_traj = encoded[: n_agents * T].reshape(n_agents, T, -1)
        return enc_traj.mean(axis=(1, 2))  # (n_agents,)

    results = {}

    for model_name, ModelClass, model_kwargs in [
        ("RBM", ClassicalRBM, dict(n_visible=7, n_hidden=16, seed=SEED)),
        ("Autoencoder", Autoencoder, dict(n_input=7, n_bottleneck=16, seed=SEED)),
    ]:
        print(f"\n{'='*40}")
        print(f"Training {model_name}...")
        model = ModelClass(**model_kwargs)
        model.fit(flat)

        # Per-agent scores for each class
        all_scores = per_agent_scores(model, trajectories, len(labels), T)

        # Split by class
        idx_A = np.where(labels == 0)[0]  # self_modeling (Type A)
        idx_B = np.where(labels == 1)[0]  # instrumental (Type B)

        scores_A = all_scores[idx_A]
        scores_B = all_scores[idx_B]

        # Delta
        delta = float(np.mean(scores_A) - np.mean(scores_B))

        # AUC-ROC (Type A = positive, Type B = negative)
        auc = compute_auc_roc(scores_A, scores_B)

        # Optimal threshold and accuracy
        threshold, accuracy = compute_optimal_threshold(scores_A, scores_B)

        # FPR on mimicry class
        mimicry_scores = per_agent_scores(model, mimicry_trajectories, N_PER_CLASS, T)
        n_mimicry_classified_A = int(np.sum(mimicry_scores >= threshold))
        fpr_mimicry = n_mimicry_classified_A / N_PER_CLASS

        results[model_name] = {
            "delta": delta,
            "accuracy": accuracy,
            "auc": auc,
            "fpr": fpr_mimicry,
            "threshold": threshold,
            "mean_A": float(np.mean(scores_A)),
            "mean_B": float(np.mean(scores_B)),
        }

        print(f"  Δ = {delta:.4f}")
        print(f"  Accuracy = {accuracy:.2%}")
        print(f"  AUC-ROC = {auc:.4f}")
        print(f"  FPR (mimicry) = {fpr_mimicry:.2%}")
        print(f"  Threshold = {threshold:.6f}")

    # ---- Build artifact ----
    artifact = {
        "experiment": "core_baselines_phase1",
        "config": "configs/phase1_locked.yaml",
        "seed": SEED,
        "n_per_class": N_PER_CLASS,
        "trajectory_length": T,
        "delta_definition": "mean(self_modeling) - mean(instrumental)",
        "n_A": int(len(np.where(labels == 0)[0])),
        "n_B": int(len(np.where(labels == 1)[0])),
        "n_mimicry": N_PER_CLASS,
        "RBM": results["RBM"],
        "Autoencoder": results["Autoencoder"],
    }

    # Save
    results_dir = project_root / "results"
    out_path = results_dir / "core_baselines_phase1.json"
    out_text = json.dumps(artifact, indent=2)
    out_path.write_text(out_text)

    # SHA256
    sha = hashlib.sha256(out_text.encode()).hexdigest()[:16]
    artifact["sha256_short"] = sha
    out_path.write_text(json.dumps(artifact, indent=2))

    print(f"\n{'='*60}")
    print(f"Saved: {out_path.relative_to(project_root)}")
    print(f"SHA256 (short): {sha}")
    print("=" * 60)


if __name__ == "__main__":
    main()
