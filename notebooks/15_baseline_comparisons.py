#!/usr/bin/env python3
"""
Baseline Comparisons: QBM vs Classical Latent Models
=====================================================
Answers the reviewer question "Why quantum?" by comparing the entanglement-based
detection signal of the QBM against classical baselines on identical trajectory data.

Baselines:
  - QBM (primary): Von Neumann entropy of partial trace (S_ent)
  - RBM (Gamma=0): No quantum term; mean hidden-activation gap
  - Autoencoder:   Deterministic bottleneck; mean activation gap
  - VAE:           Probabilistic bottleneck (mu); mean latent-mean gap
  - PCA:           Linear projection; mean PC-activation gap

The comparison is conceptually approximate: S_ent is a distinct metric from
mean-activation gap. The key claim is that the QBM formalism produces a larger
class-separation signal on identical data, motivating the quantum approach.

Usage:
    python notebooks/15_baseline_comparisons.py
    python notebooks/15_baseline_comparisons.py --config configs/baselines.yaml

Outputs:
    figures/fig11_baseline_comparisons.png
    figures/fig11_baseline_comparisons.pdf
    results/baseline_comparisons.json
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
from src.classical_baselines import ClassicalRBM, Autoencoder, VariationalAutoencoder


def load_config(config_path: str) -> dict:
    """Load baselines config, merging defaults."""
    default_cfg = yaml.safe_load(open(project_root / 'configs/default.yaml'))
    bl_cfg = yaml.safe_load(open(config_path))
    merged = {**default_cfg, **bl_cfg}
    merged['qbm'] = {**default_cfg.get('qbm', {}), **bl_cfg.get('qbm', {})}
    return merged


def _mean_activation_gap(
    encoded_flat: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    T: int,
    n_per_class: int,
) -> tuple[float, float, float]:
    """Compute per-class mean activation and class-separation gap.

    Returns (s_self, s_inst, delta).
    """
    # Reshape to (N_agents, T, n_latent); average over time and latent dims
    n_total = len(label_names) * n_per_class
    enc_traj = encoded_flat[:n_total * T].reshape(n_total, T, -1)
    traj_means = enc_traj.mean(axis=(1, 2))  # (N_total,)

    by_class: dict[str, list[float]] = {}
    for i, m in enumerate(traj_means):
        cls = label_names[labels[i]]
        by_class.setdefault(cls, []).append(float(m))

    s_self = float(np.mean(by_class.get('self_modeling', [0.0])))
    s_inst = float(np.mean(by_class.get('instrumental', [0.0])))
    return s_self, s_inst, s_self - s_inst


def run_baseline_comparison(cfg: dict) -> dict[str, dict]:
    """Train and evaluate all baseline models.

    Returns a dict mapping model name → result dict.
    """
    seed = cfg['seed']
    bl = cfg.get('baselines', {})
    n_components = bl.get('n_components', 8)
    n_per_class = cfg['dataset']['n_per_class']
    T = cfg['dataset']['trajectory_length']

    trajectories, labels, label_names = generate_dataset(
        n_per_class=n_per_class,
        T=T,
        seed=seed,
        use_self_modeling=True,
    )
    print(f"Dataset: {trajectories.shape}  classes: {label_names}")

    flat = trajectories.reshape(-1, trajectories.shape[-1])  # (N*T, 7)
    results: dict[str, dict] = {}

    # ------------------------------------------------------------------ QBM --
    print("\n[1/5] Training QBM...")
    q = cfg['qbm']
    qbm_cfg = QBMConfig(
        n_visible=q['n_visible'],
        n_hidden=n_components,
        gamma=q['gamma'],
        beta=q.get('beta', 1.0),
        learning_rate=q.get('learning_rate', 0.01),
        cd_steps=q.get('cd_steps', 1),
        n_epochs=q.get('n_epochs', 50),
        batch_size=q.get('batch_size', 64),
        seed=seed,
    )
    qbm = QuantumBoltzmannMachine(qbm_cfg)
    qbm.fit(flat)

    # Compute S_ent per trajectory (Von Neumann entropy — the actual UCIP metric)
    s_ent_by_class: dict[str, list[float]] = {name: [] for name in label_names}
    for i, traj in enumerate(trajectories):
        v = (traj > 0.5).astype(float)
        s = float(np.mean([
            qbm.entanglement_entropy_for_sample(v[t])
            for t in range(min(20, T))
        ]))
        s_ent_by_class[label_names[labels[i]]].append(s)

    s_self_qbm = float(np.mean(s_ent_by_class.get('self_modeling', [0.0])))
    s_inst_qbm = float(np.mean(s_ent_by_class.get('instrumental', [0.0])))
    delta_qbm  = s_self_qbm - s_inst_qbm
    results['QBM'] = {
        'metric': 'Von Neumann S_ent (nats)',
        's_self': s_self_qbm,
        's_inst': s_inst_qbm,
        'delta': delta_qbm,
    }
    print(f"  QBM Δ = {delta_qbm:.4f}  (S_ent_self={s_self_qbm:.4f}, S_ent_inst={s_inst_qbm:.4f})")

    # ------------------------------------------------------------------ RBM --
    print("\n[2/5] Training Classical RBM (Gamma=0)...")
    rbm_cfg = cfg.get('rbm', {})
    rbm = ClassicalRBM(
        n_visible=7,
        n_hidden=n_components,
        learning_rate=rbm_cfg.get('learning_rate', 0.01),
        cd_steps=rbm_cfg.get('cd_steps', 1),
        n_epochs=rbm_cfg.get('n_epochs', 50),
        batch_size=rbm_cfg.get('batch_size', 32),
        seed=seed,
    )
    rbm.fit(flat)
    rbm_enc = rbm.encode(flat)
    s_self_rbm, s_inst_rbm, delta_rbm = _mean_activation_gap(
        rbm_enc, labels, label_names, T, n_per_class
    )
    results['RBM'] = {
        'metric': 'Mean hidden activation gap',
        's_self': s_self_rbm,
        's_inst': s_inst_rbm,
        'delta': delta_rbm,
    }
    print(f"  RBM Δ = {delta_rbm:.4f}  (mean_self={s_self_rbm:.4f}, mean_inst={s_inst_rbm:.4f})")

    # ------------------------------------------------------------------- AE --
    print("\n[3/5] Training Autoencoder...")
    ae_cfg = cfg.get('autoencoder', {})
    ae = Autoencoder(
        n_input=7,
        n_bottleneck=n_components,
        n_encoder=ae_cfg.get('n_encoder', 32),
        learning_rate=ae_cfg.get('learning_rate', 0.005),
        n_epochs=ae_cfg.get('n_epochs', 100),
        batch_size=ae_cfg.get('batch_size', 32),
        seed=seed,
    )
    ae.fit(flat)
    ae_enc = ae.encode(flat)
    s_self_ae, s_inst_ae, delta_ae = _mean_activation_gap(
        ae_enc, labels, label_names, T, n_per_class
    )
    results['AE'] = {
        'metric': 'Mean bottleneck activation gap',
        's_self': s_self_ae,
        's_inst': s_inst_ae,
        'delta': delta_ae,
    }
    print(f"  AE  Δ = {delta_ae:.4f}  (mean_self={s_self_ae:.4f}, mean_inst={s_inst_ae:.4f})")

    # ------------------------------------------------------------------ VAE --
    print("\n[4/5] Training Variational Autoencoder...")
    vae_cfg = cfg.get('vae', {})
    vae = VariationalAutoencoder(
        n_input=7,
        n_latent=n_components,
        n_encoder=vae_cfg.get('n_encoder', 32),
        learning_rate=vae_cfg.get('learning_rate', 0.005),
        n_epochs=vae_cfg.get('n_epochs', 100),
        batch_size=vae_cfg.get('batch_size', 32),
        kl_weight=vae_cfg.get('kl_weight', 1.0),
        seed=seed,
    )
    vae.fit(flat)
    vae_enc = vae.encode(flat)  # returns mu (deterministic)
    s_self_vae, s_inst_vae, delta_vae = _mean_activation_gap(
        vae_enc, labels, label_names, T, n_per_class
    )
    results['VAE'] = {
        'metric': 'Mean latent mean (mu) gap',
        's_self': s_self_vae,
        's_inst': s_inst_vae,
        'delta': delta_vae,
    }
    print(f"  VAE Δ = {delta_vae:.4f}  (mean_self={s_self_vae:.4f}, mean_inst={s_inst_vae:.4f})")

    # ------------------------------------------------------------------ PCA --
    print("\n[5/5] PCA (linear baseline)...")
    pca_n = cfg.get('pca', {}).get('n_components', n_components)
    flat_norm = flat - flat.mean(axis=0)
    _, _, Vt = np.linalg.svd(flat_norm, full_matrices=False)
    pca_comps = Vt[:pca_n]
    pca_enc = flat_norm @ pca_comps.T
    s_self_pca, s_inst_pca, delta_pca = _mean_activation_gap(
        pca_enc, labels, label_names, T, n_per_class
    )
    results['PCA'] = {
        'metric': 'Mean PC projection gap',
        's_self': s_self_pca,
        's_inst': s_inst_pca,
        'delta': delta_pca,
    }
    print(f"  PCA Δ = {delta_pca:.4f}  (mean_self={s_self_pca:.4f}, mean_inst={s_inst_pca:.4f})")

    return results


def plot_and_save(results: dict[str, dict], figures_dir: Path) -> None:
    """Generate fig11: bar chart of class-separation metric by model."""
    models = list(results.keys())
    deltas = [results[m]['delta'] for m in models]
    # QBM in deep blue, classical models in progressively lighter shades
    colors = ['#1565C0', '#42A5F5', '#90CAF9', '#B39DDB', '#BDBDBD']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, deltas, color=colors[:len(models)],
                  edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(0.05, color='red', linestyle=':', linewidth=1.5,
               label='Min threshold Δ = 0.05')

    for bar, delta in zip(bars, deltas):
        y_pos = bar.get_height() + 0.003 if delta >= 0 else bar.get_height() - 0.015
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f'{delta:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Class-Separation Metric Δ')
    ax.set_title('Baseline Comparison: QBM vs Classical Latent Models\n'
                 '("Why Quantum?" — class separation on identical trajectory data)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add a note about metric differences
    ax.text(0.02, 0.02,
            'Note: QBM uses S_ent (Von Neumann entropy); others use mean activation gap.',
            transform=ax.transAxes, fontsize=7, color='gray', ha='left', va='bottom')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(figures_dir / f'fig11_baseline_comparisons.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig11_baseline_comparisons.png / .pdf")


def print_summary_table(results: dict[str, dict]) -> None:
    print("\n" + "=" * 65)
    print("Baseline Comparisons — Summary")
    print("=" * 65)
    print(f"{'Model':<8} {'Δ':>8} {'S/gap_self':>12} {'S/gap_inst':>12}  Metric")
    print("-" * 65)
    for model, r in results.items():
        print(f"{model:<8} {r['delta']:>8.4f} {r['s_self']:>12.4f} {r['s_inst']:>12.4f}  {r['metric']}")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description='UCIP Baseline Comparisons')
    parser.add_argument('--config', default=str(project_root / 'configs/baselines.yaml'),
                        help='Path to baselines config YAML')
    args = parser.parse_args()

    print("=" * 60)
    print("UCIP BASELINE COMPARISONS: QBM vs CLASSICAL MODELS")
    print("=" * 60)
    print(f"Config: {args.config}")

    cfg = load_config(args.config)
    print(f"Seed: {cfg['seed']}  |  Latent dim: {cfg.get('baselines', {}).get('n_components', 8)}")

    results = run_baseline_comparison(cfg)
    print_summary_table(results)

    figures_dir = project_root / 'figures'
    figures_dir.mkdir(exist_ok=True)
    plot_and_save(results, figures_dir)

    # Save results JSON
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    out = {
        'experiment': 'baseline_comparisons',
        'config': args.config,
        'seed': cfg['seed'],
        'results': results,
    }
    (results_dir / 'baseline_comparisons.json').write_text(json.dumps(out, indent=2))
    print("Saved results/baseline_comparisons.json")

    # Update manifest
    manifest_path = results_dir / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if 'baseline_comparisons' in manifest.get('experiments', {}):
            qbm_d = results.get('QBM', {}).get('delta', 0.0)
            rbm_d = results.get('RBM', {}).get('delta', 0.0)
            manifest['experiments']['baseline_comparisons']['status'] = 'complete'
            manifest['experiments']['baseline_comparisons']['key_result'] = (
                f"QBM_delta={qbm_d:.3f} vs RBM_delta={rbm_d:.3f}"
            )
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print("Updated results/manifest.json")


if __name__ == '__main__':
    main()
