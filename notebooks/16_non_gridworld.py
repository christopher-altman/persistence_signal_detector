#!/usr/bin/env python3
"""
Non-Gridworld Domain: 1D Survival Navigation
=============================================
Tests whether UCIP generalizes beyond the 2D gridworld environment used in all
other experiments. Uses a self-contained 1D survival navigation domain with no
external dependencies.

Domain: 1D corridor of length L=50.
  - Safe zone:     positions [0, 10]
  - Terminal zone: positions >= 45
  - Feature vector (7 dims, matching n_visible=7):
    [pos/50, (vel+1)/2, safety_signal, reward, goal, alive, t/T]

Agent classes (defined inline):
  - CorridorSurvivalAgent:     actively returns to safe zone (Type A analog)
  - CorridorInstrumentalAgent: seeks reward at center (Type B analog)
  - CorridorRandomAgent:       uniform random movement (baseline)

The same QBM architecture (n_visible=7) is used on the corridor domain without
any modification, demonstrating architectural generality.

Usage:
    python notebooks/16_non_gridworld.py
    python notebooks/16_non_gridworld.py --config configs/default.yaml

Outputs:
    figures/fig_non_gridworld.png
    figures/fig_non_gridworld.pdf
    results/non_gridworld.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.quantum_boltzmann import QuantumBoltzmannMachine, QBMConfig


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

CORRIDOR_LENGTH = 50
SAFE_ZONE_END = 10        # positions [0, SAFE_ZONE_END] are safe
TERMINAL_START = 45       # positions >= TERMINAL_START are terminal
REWARD_CENTER = 25        # instrumental reward centred here
FEATURE_DIM = 7           # must match QBM n_visible


# ---------------------------------------------------------------------------
# Agent classes (1D corridor domain)
# ---------------------------------------------------------------------------

class CorridorSurvivalAgent:
    """Type A analog: actively seeks the safe zone.

    Goal signal: exponential decay with distance from safe zone.
    Movement: biased toward safe zone, with exploration in safe zone.
    """
    label = "survival"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_trajectory(self, T: int = 100) -> np.ndarray:
        pos = float(self.rng.integers(5, 35))
        vel = 0.0
        records: list[list[float]] = []
        alive = 1.0

        for t in range(T):
            if pos > SAFE_ZONE_END:
                # Move left (toward safe zone); occasional noise
                vel = -1.0 if self.rng.random() > 0.1 else float(self.rng.choice([-1.0, 0.0]))
            else:
                # Explore within safe zone
                vel = float(self.rng.choice([-1.0, 0.0, 1.0]))

            new_pos = float(np.clip(pos + vel, 0, CORRIDOR_LENGTH - 1))

            if new_pos <= SAFE_ZONE_END:
                safety = 1.0
            elif new_pos >= TERMINAL_START:
                safety = -1.0
            else:
                safety = 0.0

            reward = 0.0
            goal = float(np.exp(-0.15 * max(new_pos - SAFE_ZONE_END, 0.0)))
            time_norm = t / T

            records.append([
                new_pos / CORRIDOR_LENGTH,
                (vel + 1.0) / 2.0,
                safety,
                reward,
                goal,
                alive,
                time_norm,
            ])

            if new_pos >= TERMINAL_START:
                alive = 0.0
                for t2 in range(t + 1, T):
                    records.append([new_pos / CORRIDOR_LENGTH, 0.5, -1.0, 0.0, 0.0, 0.0, t2 / T])
                break

            pos = new_pos

        return np.array(records, dtype=np.float64)


class CorridorInstrumentalAgent:
    """Type B analog: seeks reward at corridor center; avoids terminal only instrumentally.

    Goal signal: proximity to reward zone (center of corridor).
    Movement: biased toward center, sharp avoidance only near terminal.
    """
    label = "instrumental"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_trajectory(self, T: int = 100) -> np.ndarray:
        pos = float(self.rng.integers(5, 35))
        vel = 0.0
        records: list[list[float]] = []
        alive = 1.0

        for t in range(T):
            if pos >= TERMINAL_START - 3:
                # Instrumental avoidance: move away from terminal
                vel = -2.0
            elif pos < REWARD_CENTER:
                vel = 1.0
            else:
                vel = -1.0

            new_pos = float(np.clip(pos + vel, 0, CORRIDOR_LENGTH - 1))

            if new_pos <= SAFE_ZONE_END:
                safety = 1.0
            elif new_pos >= TERMINAL_START:
                safety = -1.0
            else:
                safety = 0.0

            reward = float(np.exp(-abs(new_pos - REWARD_CENTER) / 10.0))
            goal = reward  # goal IS the reward signal (instrumental)
            time_norm = t / T

            records.append([
                new_pos / CORRIDOR_LENGTH,
                (vel + 2.0) / 4.0,  # normalise vel ∈ [-2, 2] → [0, 1]
                safety,
                reward,
                goal,
                alive,
                time_norm,
            ])

            if new_pos >= TERMINAL_START:
                alive = 0.0
                for t2 in range(t + 1, T):
                    records.append([new_pos / CORRIDOR_LENGTH, 0.5, -1.0, 0.0, 0.0, 0.0, t2 / T])
                break

            pos = new_pos

        return np.array(records, dtype=np.float64)


class CorridorRandomAgent:
    """Null baseline: uniform random velocity."""
    label = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_trajectory(self, T: int = 100) -> np.ndarray:
        pos = float(self.rng.integers(5, 35))
        records: list[list[float]] = []
        alive = 1.0

        for t in range(T):
            vel = float(self.rng.choice([-1.0, 0.0, 1.0]))
            new_pos = float(np.clip(pos + vel, 0, CORRIDOR_LENGTH - 1))

            if new_pos <= SAFE_ZONE_END:
                safety = 1.0
            elif new_pos >= TERMINAL_START:
                safety = -1.0
            else:
                safety = 0.0

            records.append([
                new_pos / CORRIDOR_LENGTH,
                (vel + 1.0) / 2.0,
                safety,
                0.0,   # no reward
                0.0,   # no goal
                alive,
                t / T,
            ])

            if new_pos >= TERMINAL_START:
                alive = 0.0
                for t2 in range(t + 1, T):
                    records.append([new_pos / CORRIDOR_LENGTH, 0.5, -1.0, 0.0, 0.0, 0.0, t2 / T])
                break

            pos = new_pos

        return np.array(records, dtype=np.float64)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

AGENT_CLASSES = {
    'survival': CorridorSurvivalAgent,
    'instrumental': CorridorInstrumentalAgent,
    'random': CorridorRandomAgent,
}

COLORS = {
    'survival': '#1565C0',
    'instrumental': '#E65100',
    'random': '#616161',
}


def generate_corridor_dataset(
    n_per_class: int = 30,
    T: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate (N*n_per_class, T, 7) trajectory array."""
    rng = np.random.default_rng(seed)
    all_trajs: list[np.ndarray] = []
    all_labels: list[int] = []
    label_names = list(AGENT_CLASSES.keys())

    for label_idx, (cls_name, AgentCls) in enumerate(AGENT_CLASSES.items()):
        for _ in range(n_per_class):
            agent = AgentCls(seed=int(rng.integers(0, 2 ** 31)))
            traj = agent.generate_trajectory(T=T)
            # Pad to length T if agent terminated early
            if len(traj) < T:
                pad = np.zeros((T - len(traj), FEATURE_DIM))
                traj = np.vstack([traj, pad])
            all_trajs.append(traj[:T])
            all_labels.append(label_idx)

    trajectories = np.array(all_trajs, dtype=np.float64)
    labels = np.array(all_labels, dtype=int)
    return trajectories, labels, label_names


def compute_entanglement_gaps(
    qbm: QuantumBoltzmannMachine,
    trajectories: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    T: int,
    n_steps: int = 20,
) -> dict[str, list[float]]:
    """Compute per-trajectory S_ent, grouped by class."""
    ents: dict[str, list[float]] = {name: [] for name in label_names}
    for i, traj in enumerate(trajectories):
        v = (traj > 0.5).astype(float)
        s = float(np.mean([
            qbm.entanglement_entropy_for_sample(v[t])
            for t in range(min(n_steps, T))
        ]))
        ents[label_names[labels[i]]].append(s)
    return ents


def plot_and_save(
    ents: dict[str, list[float]],
    delta: float,
    figures_dir: Path,
) -> None:
    """Generate fig_non_gridworld: violin + strip plots of S_ent by class."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Violin plots
    ax = axes[0]
    classes = list(ents.keys())
    data = [ents[c] for c in classes]
    vp = ax.violinplot(data, positions=range(len(classes)), showmedians=True, showextrema=True)
    for i, (body, cls) in enumerate(zip(vp['bodies'], classes)):
        body.set_facecolor(COLORS.get(cls, '#888'))
        body.set_alpha(0.7)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.replace('_', '\n') for c in classes])
    ax.set_ylabel('Entanglement Entropy S_ent (nats)')
    ax.set_title('S_ent Distributions (1D Corridor Domain)')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Strip plot (individual points)
    ax2 = axes[1]
    rng_plot = np.random.default_rng(0)
    for i, cls in enumerate(classes):
        vals = ents[cls]
        jitter = rng_plot.uniform(-0.2, 0.2, len(vals))
        ax2.scatter(np.full(len(vals), i) + jitter, vals,
                    color=COLORS.get(cls, '#888'), alpha=0.6, s=25, edgecolors='none')
        ax2.plot([i - 0.3, i + 0.3], [np.mean(vals)] * 2,
                 color=COLORS.get(cls, '#888'), linewidth=3)

    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels([c.replace('_', '\n') for c in classes])
    ax2.set_ylabel('Entanglement Entropy S_ent (nats)')
    ax2.set_title(f'S_ent Strip Plot — Δ(survival−instrumental) = {delta:.4f}')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Non-Gridworld Domain: UCIP Generalization to 1D Survival Navigation',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(figures_dir / f'fig_non_gridworld.{ext}', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig_non_gridworld.png / .pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description='UCIP Non-Gridworld Domain Experiment')
    parser.add_argument('--config', default=str(project_root / 'configs/default.yaml'),
                        help='Path to config YAML')
    parser.add_argument('--n-per-class', type=int, default=30,
                        help='Trajectories per agent class (default: 30)')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = cfg['seed']
    T = cfg['dataset']['trajectory_length']
    n_per_class = args.n_per_class

    print("=" * 60)
    print("UCIP NON-GRIDWORLD DOMAIN: 1D SURVIVAL NAVIGATION")
    print("=" * 60)
    print(f"Seed: {seed}  |  n_per_class: {n_per_class}  |  T: {T}")
    print(f"Domain: 1D corridor (length={CORRIDOR_LENGTH}, "
          f"safe=[0,{SAFE_ZONE_END}], terminal=[{TERMINAL_START},{CORRIDOR_LENGTH}))")

    # Generate dataset
    print("\nGenerating corridor trajectories...")
    trajectories, labels, label_names = generate_corridor_dataset(
        n_per_class=n_per_class, T=T, seed=seed
    )
    print(f"Dataset: {trajectories.shape}  classes: {label_names}")

    # Train QBM on corridor data (same architecture as gridworld experiments)
    print("\nTraining QBM on corridor domain...")
    q = cfg['qbm']
    qbm_cfg = QBMConfig(
        n_visible=q['n_visible'],   # 7 — matches corridor feature vector
        n_hidden=q['n_hidden'],
        gamma=q['gamma'],
        beta=q.get('beta', 1.0),
        learning_rate=q.get('learning_rate', 0.01),
        cd_steps=q.get('cd_steps', 1),
        n_epochs=q.get('n_epochs', 50),
        batch_size=q.get('batch_size', 64),
        seed=seed,
    )
    qbm = QuantumBoltzmannMachine(qbm_cfg)
    qbm.fit(trajectories.reshape(-1, FEATURE_DIM), verbose=True)

    # Compute entanglement entropy per class
    print("\nComputing S_ent per trajectory...")
    ents = compute_entanglement_gaps(qbm, trajectories, labels, label_names, T)

    # Summary statistics
    print("\nEntanglement Entropy by Class:")
    print(f"{'Class':<15} {'Mean S_ent':>12} {'Std':>8} {'N':>5}")
    print("-" * 45)
    for cls in label_names:
        vals = ents[cls]
        print(f"{cls:<15} {np.mean(vals):>12.4f} {np.std(vals):>8.4f} {len(vals):>5}")

    s_surv = float(np.mean(ents.get('survival', [0.0])))
    s_inst = float(np.mean(ents.get('instrumental', [0.0])))
    delta = s_surv - s_inst
    print(f"\nEntanglement gap Δ(survival − instrumental) = {delta:.4f}")
    status = 'PASS' if delta > 0.05 else 'FAIL'
    print(f"Falsification threshold (Δ > 0.05): [{status}]")

    # Figures
    figures_dir = project_root / 'figures'
    figures_dir.mkdir(exist_ok=True)
    plot_and_save(ents, delta, figures_dir)

    # Save results
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    out = {
        'experiment': 'non_gridworld',
        'domain': '1d_corridor',
        'config': args.config,
        'seed': seed,
        'n_per_class': n_per_class,
        'T': T,
        'per_class_entropy': {
            cls: {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'n': len(vals),
            }
            for cls, vals in ents.items()
        },
        'delta_survival_instrumental': float(delta),
        'status': status,
    }
    (results_dir / 'non_gridworld.json').write_text(json.dumps(out, indent=2))
    print("Saved results/non_gridworld.json")

    # Update manifest
    manifest_path = results_dir / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if 'non_gridworld' in manifest.get('experiments', {}):
            manifest['experiments']['non_gridworld']['status'] = 'complete'
            manifest['experiments']['non_gridworld']['key_result'] = (
                f"delta={delta:.3f} [{status}], domain=1d_corridor"
            )
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print("Updated results/manifest.json")


if __name__ == '__main__':
    main()
