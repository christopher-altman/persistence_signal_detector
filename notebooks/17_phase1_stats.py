#!/usr/bin/env python3
"""
Phase 1 Statistical Validation: Permutation Test + Bootstrap CI

Reproduces the phase1 entanglement gap Δ from scratch using configs/phase1_locked.yaml,
collects individual trajectory-level S_ent values, then computes:
  - Permutation test (n_perm=1000, one-sided: H1: self_modeling > instrumental)
  - Bootstrap 95% CI on Δ (n_boot=2000)

Outputs: results/phase1_stats.json

This closes the audit finding: phase1_consolidated.json stores only mean/std and
has no p_value field, making the manuscript's 'p < 0.001' claim unverifiable from
saved artifacts alone.
"""

import json
import sys
import yaml
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agent_simulator import SelfModelingAgent, InstrumentalAgent
from src.quantum_boltzmann import QuantumBoltzmannMachine, QBMConfig


# =============================================================================
# 1. Load locked config
# =============================================================================

config_path = project_root / "configs" / "phase1_locked.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

qbm_cfg = cfg["qbm"]
n_per_class = cfg["dataset"]["n_per_class"]   # 30
T = cfg["dataset"]["trajectory_length"]        # 100
seed = qbm_cfg["seed"]                         # 42

print("=" * 70)
print("PHASE 1 STATISTICAL VALIDATION")
print("=" * 70)
print(f"\nConfig: {config_path}")
print(f"n_per_class={n_per_class}, T={T}, seed={seed}")
print(f"QBM: n_visible={qbm_cfg['n_visible']}, n_hidden={qbm_cfg['n_hidden']}, "
      f"n_epochs={qbm_cfg['n_epochs']}, batch_size={qbm_cfg['batch_size']}")


# =============================================================================
# 2. Regenerate trajectories + compute per-trajectory S_ent
# =============================================================================

def collect_s_ents(AgentCls, n, T, rng, qbm_cfg):
    """
    Generate n trajectories from AgentCls, fit a QBM to each, and return
    a list of mean entanglement entropy (S_ent) values, one per trajectory.

    Each trajectory gets its own freshly trained QBM (matching phase1 protocol
    where each agent sample was evaluated independently).
    """
    s_ents = []
    for i in range(n):
        agent = AgentCls(seed=int(rng.integers(0, 2**31)))
        traj = agent.generate_trajectory(T=T)
        qcfg = QBMConfig(
            n_visible=qbm_cfg["n_visible"],
            n_hidden=qbm_cfg["n_hidden"],
            gamma=qbm_cfg["gamma"],
            n_epochs=qbm_cfg["n_epochs"],
            batch_size=qbm_cfg["batch_size"],
            seed=qbm_cfg["seed"],
        )
        qbm = QuantumBoltzmannMachine(qcfg)
        qbm.fit(traj)
        s_ent = float(np.mean([
            qbm.entanglement_entropy_for_sample(traj[t]) for t in range(T)
        ]))
        s_ents.append(s_ent)
        if (i + 1) % 5 == 0:
            print(f"  {AgentCls.__name__}: {i+1}/{n} done (last S_ent={s_ent:.4f})")
    return s_ents


rng = np.random.default_rng(seed)

print("\n--- Collecting SelfModelingAgent S_ent values ---")
self_ents = collect_s_ents(SelfModelingAgent, n_per_class, T, rng, qbm_cfg)

print("\n--- Collecting InstrumentalAgent S_ent values ---")
inst_ents = collect_s_ents(InstrumentalAgent, n_per_class, T, rng, qbm_cfg)


# =============================================================================
# 3. Compute observed Δ and sanity-check against phase1_consolidated.json
# =============================================================================

obs_delta = float(np.mean(self_ents) - np.mean(inst_ents))
phase1_delta_reference = 0.3810883045604201  # from phase1_consolidated.json

print(f"\nObserved Δ = {obs_delta:.6f}")
print(f"Phase1 reference Δ = {phase1_delta_reference:.6f}")
print(f"Absolute difference = {abs(obs_delta - phase1_delta_reference):.6f}")

# Tolerance: within 0.05 nat of reference (QBM stochasticity may cause small drift)
delta_matches_phase1 = bool(abs(obs_delta - phase1_delta_reference) < 0.05)
if delta_matches_phase1:
    print("✓ Reproduced Δ matches phase1 reference (within 0.05 tolerance)")
else:
    print("⚠ Reproduced Δ differs from phase1 reference by > 0.05 nats")
    print("  → QBM training is stochastic; batch_size and epoch differences")
    print("    may cause small numerical divergence across runs.")


# =============================================================================
# 4. Permutation test (one-sided: H1: self_modeling S_ent > instrumental S_ent)
# =============================================================================

n_perm = 1000
pool = list(self_ents) + list(inst_ents)   # 60 values
perm_rng = np.random.default_rng(seed + 1)

print(f"\n--- Permutation test (n_perm={n_perm}) ---")
null_deltas = []
for _ in range(n_perm):
    shuffled = perm_rng.permutation(pool)
    pseudo_delta = float(np.mean(shuffled[:n_per_class]) - np.mean(shuffled[n_per_class:]))
    null_deltas.append(pseudo_delta)

null_deltas = np.array(null_deltas)
p_value = float(np.sum(null_deltas >= obs_delta) / n_perm)

print(f"Null distribution: mean={np.mean(null_deltas):.4f}, std={np.std(null_deltas):.4f}")
print(f"Observed Δ = {obs_delta:.4f}")
print(f"p-value (one-sided) = {p_value:.4f}  ({int(np.sum(null_deltas >= obs_delta))}/{n_perm} null Δ ≥ obs Δ)")

if p_value < 0.001:
    print("✓ p < 0.001 — manuscript claim confirmed")
elif p_value < 0.01:
    print(f"⚠ p = {p_value:.4f} — manuscript states p < 0.001; update claim")
else:
    print(f"✗ p = {p_value:.4f} — significant revision required")


# =============================================================================
# 5. Bootstrap 95% CI on Δ (n_boot=2000)
# =============================================================================

n_boot = 2000
boot_rng = np.random.default_rng(seed + 2)
self_arr = np.array(self_ents)
inst_arr = np.array(inst_ents)

print(f"\n--- Bootstrap 95% CI (n_boot={n_boot}) ---")
boot_deltas = []
for _ in range(n_boot):
    bs_idx = boot_rng.integers(0, n_per_class, size=n_per_class)
    bi_idx = boot_rng.integers(0, n_per_class, size=n_per_class)
    boot_deltas.append(float(np.mean(self_arr[bs_idx]) - np.mean(inst_arr[bi_idx])))

boot_deltas = np.array(boot_deltas)
ci_low = float(np.percentile(boot_deltas, 2.5))
ci_high = float(np.percentile(boot_deltas, 97.5))

print(f"Bootstrap Δ: mean={np.mean(boot_deltas):.4f}, std={np.std(boot_deltas):.4f}")
print(f"95% CI = [{ci_low:.4f}, {ci_high:.4f}]")

if ci_low > 0:
    print("✓ CI excludes zero — gap is reliably positive")
else:
    print("⚠ CI lower bound ≤ 0 — gap may be unstable at this sample size")


# =============================================================================
# 6. Save to results/phase1_stats.json
# =============================================================================

output = {
    "metadata": {
        "date": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.relative_to(project_root)),
        "n_per_class": n_per_class,
        "trajectory_length": T,
        "seed": seed,
        "n_perm": n_perm,
        "n_boot": n_boot,
        "note": (
            "S_ent values reproduced by re-running phase1 with configs/phase1_locked.yaml. "
            "phase1_consolidated.json stores only mean/std; individual values are generated here "
            "to enable permutation test and bootstrap CI."
        ),
    },
    "reproduced_means": {
        "self_modeling": float(np.mean(self_ents)),
        "self_modeling_std": float(np.std(self_ents)),
        "instrumental": float(np.mean(inst_ents)),
        "instrumental_std": float(np.std(inst_ents)),
    },
    "individual_s_ents": {
        "self_modeling": [float(x) for x in self_ents],
        "instrumental": [float(x) for x in inst_ents],
    },
    "delta_observed": float(obs_delta),
    "phase1_delta_reference": phase1_delta_reference,
    "delta_matches_phase1": delta_matches_phase1,
    "permutation_test": {
        "n_perm": n_perm,
        "p_value": p_value,
        "null_delta_mean": float(np.mean(null_deltas)),
        "null_delta_std": float(np.std(null_deltas)),
        "n_null_gte_obs": int(np.sum(null_deltas >= obs_delta)),
    },
    "bootstrap_ci_95": {
        "low": ci_low,
        "high": ci_high,
        "n_boot": n_boot,
        "boot_delta_mean": float(np.mean(boot_deltas)),
        "boot_delta_std": float(np.std(boot_deltas)),
    },
}

results_path = project_root / "results" / "phase1_stats.json"
with open(results_path, "w") as fh:
    json.dump(output, fh, indent=2)

print(f"\n✓ Results saved to {results_path}")
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Δ_obs          = {obs_delta:.4f}")
print(f"  p-value        = {p_value:.4f}  ({'p < 0.001' if p_value < 0.001 else f'p = {p_value:.4f}'})")
print(f"  95% CI         = [{ci_low:.4f}, {ci_high:.4f}]")
print(f"  delta_matches  = {delta_matches_phase1}")
print(f"  CI excludes 0  = {ci_low > 0}")
