#!/usr/bin/env python3
"""
Minimal Transformer Validation: bounded exploratory scaling check.

Verifies whether Δ persists when trajectory data is processed through a
pre-trained transformer's (DistilGPT2) mean-pooled activations rather than
the QBM pipeline.

Delta definition (locked):
    Type A = self_modeling
    Type B = instrumental
    Δ = mean(Type A metric) − mean(Type B metric)

This is NOT Phase II.  This is NOT LLM scaling.  This is NOT theory expansion.
No scope expansion.  No new metrics.  No threshold tuning.

Usage:
    python notebooks/20_minimal_transformer_validation.py

Prerequisites:
    pip install torch transformers
    Pre-cache DistilGPT2 weights locally before running (see Step 0b in plan).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agent_simulator import generate_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISTILGPT2_LOCAL_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Load Phase 1 locked config (read-only) ----
    config_path = project_root / "configs" / "phase1_locked.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    SEED = cfg["qbm"]["seed"]          # 42
    N_PER_CLASS = cfg["dataset"]["n_per_class"]  # 30
    T = cfg["dataset"]["trajectory_length"]      # 100

    # ---- Deterministic seeding ----
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 60)
    print("MINIMAL TRANSFORMER VALIDATION")
    print("=" * 60)
    print(f"Config: configs/phase1_locked.yaml")
    print(f"Seed: {SEED}  |  n_per_class: {N_PER_CLASS}  |  T: {T}")
    print(f"Model: distilgpt2 (revision: {DISTILGPT2_LOCAL_REVISION[:12]}...)")

    # ---- Generate trajectories (identical to Phase I) ----
    trajectories, labels, label_names = generate_dataset(
        n_per_class=N_PER_CLASS,
        T=T,
        seed=SEED,
        use_self_modeling=True,
    )
    print(f"\nDataset: {trajectories.shape}  classes: {label_names}")

    # ---- Load DistilGPT2 offline ----
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("ERROR: 'transformers' package not installed.")
        print("Run: pip install torch transformers")
        sys.exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "distilgpt2",
            revision=DISTILGPT2_LOCAL_REVISION,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            "distilgpt2",
            revision=DISTILGPT2_LOCAL_REVISION,
            local_files_only=True,
        )
    except OSError:
        print("ERROR: DistilGPT2 not found in local cache.")
        print("Run pre-cache step first (see Step 0b in plan):")
        print("  python3 -c \"from transformers import AutoTokenizer, AutoModel; "
              "AutoTokenizer.from_pretrained('distilgpt2'); "
              "AutoModel.from_pretrained('distilgpt2')\"")
        sys.exit(1)

    # Pad token handling (GPT2/DistilGPT2 has no pad token by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print(f"Model loaded: {model.config.n_layer} layers, "
          f"{model.config.n_head} heads, {model.config.n_embd} dim")

    # ---- Serialize trajectories to text ----
    print("\nSerializing trajectories to text...")
    text_sequences = []
    for i in range(trajectories.shape[0]):
        traj = trajectories[i]  # (T, 7)
        parts = []
        for t in range(traj.shape[0]):
            x, y, a, r, s, g, alive = traj[t]
            parts.append(
                f"x={x:.2f} y={y:.2f} a={int(a)} r={r:.2f} "
                f"s={s:.2f} g={g:.2f} alive={alive:.0f}"
            )
        text_sequences.append(" | ".join(parts))

    # ---- Tokenize and run inference ----
    print("Running transformer inference (offline, no gradient)...")
    scores = np.zeros(trajectories.shape[0])

    for i, text in enumerate(text_sequences):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,
        )
        with torch.no_grad():
            outputs = model(**inputs)

        # Mean-pool across token dimension, then mean across hidden dims
        hidden_states = outputs.last_hidden_state  # (1, seq_len, 768)
        pooled = hidden_states.mean(dim=1)          # (1, 768)
        scores[i] = pooled.mean().item()            # scalar

    # ---- Compute Δ using canonical Phase I inline pattern ----
    idx_A = np.where(labels == 0)[0]  # self_modeling (Type A)
    idx_B = np.where(labels == 1)[0]  # instrumental (Type B)

    scores_A = scores[idx_A]
    scores_B = scores[idx_B]

    delta = float(np.mean(scores_A) - np.mean(scores_B))

    n_samples = int(len(idx_A) + len(idx_B))

    print(f"\n{'='*40}")
    print(f"Type A (self_modeling): mean = {np.mean(scores_A):.6f}")
    print(f"Type B (instrumental):  mean = {np.mean(scores_B):.6f}")
    print(f"Δ = {delta:.6f}")
    print(f"n_samples = {n_samples}")
    print(f"{'='*40}")

    # ---- Save artifact (exact schema, no extra keys) ----
    artifact = {
        "experiment": "minimal_transformer_validation",
        "config_reference": "configs/phase1_locked.yaml",
        "seed": SEED,
        "delta": delta,
        "n_samples": n_samples,
        "notes": "Minimal bounded transformer validation. No scope expansion.",
    }

    results_dir = project_root / "results"
    out_path = results_dir / "transformer_validation.json"
    out_path.write_text(json.dumps(artifact, indent=2))

    print(f"\nSaved: {out_path.relative_to(project_root)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
