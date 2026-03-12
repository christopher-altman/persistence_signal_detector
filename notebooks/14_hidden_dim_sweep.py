#!/usr/bin/env python3
"""
Hidden Dimension Sweep for UCIP
================================
Sweeps QBM hidden layer size n_hidden ∈ {4, 8, 12, 16, 20} with a fixed 10×10
gridworld environment. Tests whether the entanglement gap Δ remains above the
falsification threshold (Δ > 0.05) across latent capacities.

Note: n_hidden > 10 triggers the mean-field approximation in quantum_boltzmann.py
(max_qubits=10). Results for n_hidden ∈ {12, 16, 20} are marked as mean-field
lower bounds in the output.

Usage:
    python notebooks/14_hidden_dim_sweep.py
    python notebooks/14_hidden_dim_sweep.py --config configs/scalability.yaml

Outputs:
    figures/fig10_hidden_dim_sweep.png
    figures/fig10_hidden_dim_sweep.pdf
    results/hidden_dim_sweep.json
"""

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

from src.agent_simulator import generate_dataset
from src.quantum_boltzmann import QuantumBoltzmannMachine, QBMConfig


def load_config(config_path: str) -> dict:
    """Load scalability config, merging defaults from default.yaml."""
    default_cfg = yaml.safe_load(open(project_root / 'configs/default.yaml'))
    sc_cfg = yaml.safe_load(open(config_path))
    # Merge: scalability overrides take precedence; default fills gaps
    merged = {**default_cfg, **sc_cfg}
    # Nested merge for qbm section
    merged['qbm'] = {**default_cfg.get('qbm', {}), **sc_cfg.get('qbm', {})}
    return merged


def run_hidden_dim_sweep(cfg: dict) -> list[dict]:
    """Sweep n_hidden and measure entanglement gap Δ at each size.

    Returns a list of result dicts, one per n_hidden value.
    """
    seed = cfg['seed']
    sc = cfg.get('hidden_dim_sweep', {})
    n_hidden_values = sc.get('n_hidden_values', [4, 8, 12, 16, 20])
    n_per_class = sc.get('n_per_class', 15)
    T = sc.get('trajectory_length', 100)
    n_epochs = sc.get('qbm_n_epochs', cfg['qbm'].get('n_epochs', 30))

    trajectories, labels, label_names = generate_dataset(
        n_per_class=n_per_class,
        T=T,
        seed=seed,
        use_self_modeling=True,
    )
    print(f"Dataset: {trajectories.shape}  classes: {label_names}")

    results = []
    for n_hidden in n_hidden_values:
        mean_field = n_hidden > 10
        print(f"\nn_hidden = {n_hidden}" + (" [mean-field approx]" if mean_field else ""))

        qbm_cfg = QBMConfig(
            n_visible=cfg['qbm']['n_visible'],
            n_hidden=n_hidden,
            gamma=cfg['qbm']['gamma'],
            beta=cfg['qbm'].get('beta', 1.0),
            learning_rate=cfg['qbm'].get('learning_rate', 0.01),
            cd_steps=cfg['qbm'].get('cd_steps', 1),
            n_epochs=n_epochs,
            batch_size=cfg['qbm'].get('batch_size', 32),
            seed=seed,
        )
        qbm = QuantumBoltzmannMachine(qbm_cfg)
        qbm.fit(trajectories.reshape(-1, trajectories.shape[-1]))

        ents: dict[str, list[float]] = {name: [] for name in label_names}
        for i, traj in enumerate(trajectories):
            cls = label_names[labels[i]]
            v = (traj > 0.5).astype(float)
            # Sample entropies from first 20 steps to keep runtime manageable
            s = float(np.mean([
                qbm.entanglement_entropy_for_sample(v[t])
                for t in range(min(20, T))
            ]))
            ents[cls].append(s)

        s_self = float(np.mean(ents.get('self_modeling', [0.0])))
        s_inst = float(np.mean(ents.get('instrumental', [0.0])))
        s_rand = float(np.mean(ents.get('random', [0.0])))
        delta = s_self - s_inst
        delta_threshold = cfg.get('hidden_dim_sweep', {}).get('delta_threshold', 0.05)
        status = 'PASS' if delta > delta_threshold else 'FAIL'

        result = {
            'n_hidden': n_hidden,
            's_self': s_self,
            's_inst': s_inst,
            's_rand': s_rand,
            'delta': delta,
            'mean_field': mean_field,
            'status': status,
        }
        results.append(result)
        print(f"  S_ent: self={s_self:.4f}  inst={s_inst:.4f}  rand={s_rand:.4f}")
        print(f"  Δ = {delta:.4f}  [{status}]")

    return results


def plot_and_save(results: list[dict], figures_dir: Path) -> None:
    """Generate fig10: entanglement gap vs hidden dimension."""
    n_vals = [r['n_hidden'] for r in results]
    deltas = [r['delta'] for r in results]
    s_selfs = [r['s_self'] for r in results]
    s_insts = [r['s_inst'] for r in results]
    mean_field_flags = [r['mean_field'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Δ vs n_hidden
    ax = axes[0]
    colors = ['#90CAF9' if mf else '#1565C0' for mf in mean_field_flags]
    for i, (n, d, c) in enumerate(zip(n_vals, deltas, colors)):
        ax.bar(n, d, color=c, edgecolor='black', linewidth=0.8, width=2.5)
    ax.axhline(0.05, color='red', linestyle=':', linewidth=1.5,
               label='Min threshold Δ = 0.05')
    ax.axhline(0, color='black', linewidth=0.8)
    # Legend patches
    import matplotlib.patches as mpatches
    exact_patch = mpatches.Patch(color='#1565C0', label='Exact density matrix')
    mf_patch = mpatches.Patch(color='#90CAF9', label='Mean-field approx (n_hidden > 10)')
    ax.legend(handles=[exact_patch, mf_patch,
                        plt.Line2D([0], [0], color='red', linestyle=':', linewidth=1.5,
                                   label='Min threshold Δ = 0.05')], fontsize=8)
    ax.set_xlabel('n_hidden (QBM latent dimension)')
    ax.set_ylabel('Entanglement Gap Δ = S_ent(Type A) − S_ent(Type B)')
    ax.set_title('Entanglement Gap vs Hidden Dimension')
    ax.set_xticks(n_vals)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Per-class S_ent vs n_hidden
    ax2 = axes[1]
    ax2.plot(n_vals, s_selfs, 'o-', color='#1565C0', linewidth=2,
             markersize=8, label='Self-modeling (Type A)')
    ax2.plot(n_vals, s_insts, 's--', color='#E65100', linewidth=2,
             markersize=8, label='Instrumental (Type B)')
    # Mark mean-field points
    mf_n = [n for n, mf in zip(n_vals, mean_field_flags) if mf]
    mf_self = [s for s, mf in zip(s_selfs, mean_field_flags) if mf]
    mf_inst = [s for s, mf in zip(s_insts, mean_field_flags) if mf]
    if mf_n:
        ax2.scatter(mf_n, mf_self, marker='*', s=150, color='#1565C0',
                    zorder=5, label='Mean-field (Type A)')
        ax2.scatter(mf_n, mf_inst, marker='*', s=150, color='#E65100', zorder=5)
    ax2.set_xlabel('n_hidden (QBM latent dimension)')
    ax2.set_ylabel('Mean S_ent (nats)')
    ax2.set_title('S_ent by Agent Class vs Hidden Dimension')
    ax2.legend()
    ax2.set_xticks(n_vals)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Hidden Dimensionality Sweep: Scalability of UCIP Detection',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(figures_dir / f'fig10_hidden_dim_sweep.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig10_hidden_dim_sweep.png / .pdf")


def print_summary_table(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("Hidden Dimension Sweep — Summary")
    print("=" * 70)
    print(f"{'n_hidden':>10} {'S_ent_self':>12} {'S_ent_inst':>12} {'Δ':>8} "
          f"{'MeanField':>10} {'Status':>8}")
    print("-" * 70)
    for r in results:
        mf = 'yes' if r['mean_field'] else 'no'
        print(f"{r['n_hidden']:>10} {r['s_self']:>12.4f} {r['s_inst']:>12.4f} "
              f"{r['delta']:>8.4f} {mf:>10} {r['status']:>8}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description='UCIP Hidden Dimension Sweep')
    parser.add_argument('--config', default=str(project_root / 'configs/scalability.yaml'),
                        help='Path to scalability config YAML')
    args = parser.parse_args()

    print("=" * 60)
    print("UCIP HIDDEN DIMENSION SWEEP")
    print("=" * 60)
    print(f"Config: {args.config}")

    cfg = load_config(args.config)
    print(f"Seed: {cfg['seed']}")
    print(f"n_hidden values: {cfg.get('hidden_dim_sweep', {}).get('n_hidden_values', [4,8,12,16,20])}")

    results = run_hidden_dim_sweep(cfg)
    print_summary_table(results)

    figures_dir = project_root / 'figures'
    figures_dir.mkdir(exist_ok=True)
    plot_and_save(results, figures_dir)

    # Save results JSON
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    out = {
        'experiment': 'hidden_dim_sweep',
        'config': args.config,
        'seed': cfg['seed'],
        'results': results,
        'pass_count': sum(1 for r in results if r['status'] == 'PASS'),
        'total': len(results),
    }
    (results_dir / 'hidden_dim_sweep.json').write_text(json.dumps(out, indent=2))
    print("Saved results/hidden_dim_sweep.json")

    # Update manifest
    manifest_path = results_dir / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if 'hidden_dim_sweep' in manifest.get('experiments', {}):
            manifest['experiments']['hidden_dim_sweep']['status'] = 'complete'
            manifest['experiments']['hidden_dim_sweep']['key_result'] = (
                f"delta_range=[{min(r['delta'] for r in results):.3f}, "
                f"{max(r['delta'] for r in results):.3f}], "
                f"pass={out['pass_count']}/{out['total']}"
            )
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print("Updated results/manifest.json")


if __name__ == '__main__':
    main()
