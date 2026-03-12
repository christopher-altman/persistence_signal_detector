# Persistence Signal Detector (UCIP)

![Status](https://img.shields.io/badge/status-research%20prototype-2f6feb)
![Focus](https://img.shields.io/badge/focus-objective%20structure%20measurement-6f42c1)
![Method](https://img.shields.io/badge/method-QBM%20%2B%20information%20theory-1f883d)
![Claims](https://img.shields.io/badge/claims-operational%20only%2C%20not%20metaphysical-bd561d)

*Unified Continuation-Interest Protocol for distinguishing intrinsic continuation objectives from merely instrumental self-preservation in autonomous agents, using quantum-inspired latent-structure analysis implemented entirely on classical hardware.*

<br>

[![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b.svg)](#publication-reproducibility-and-dataset)
[![DOI](https://img.shields.io/badge/DOI-TBD-blue)](#publication-reproducibility-and-dataset)
[![Zenodo](https://img.shields.io/badge/Zenodo-TBD-1f77b4)](#publication-reproducibility-and-dataset)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](#license--ip)
[![Website](https://img.shields.io/badge/website-lab.christopheraltman.com-green)](https://lab.christopheraltman.com)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-Profile-blue?logo=google-scholar)](https://scholar.google.com/citations?user=tvwpCcgAAAAJ)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Altman-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/Altman)

<br>

> **TL;DR:** UCIP is a research protocol for probing whether apparent self-preservation in an artificial agent is a detachable instrumental strategy or a more deeply integrated continuation-related objective. It encodes trajectories with a Quantum Boltzmann Machine (QBM), measures latent non-separability with entanglement-style metrics plus auxiliary criteria, and in the frozen Phase I gridworld evaluation reports **100% accuracy**, **1.0 AUC-ROC**, and an entanglement gap of **Δ = 0.381**. The same evidence base also documents the current limits: partial mimicry robustness, failed zero-shot transfer to a non-gridworld domain, and scaling constraints beyond small hidden dimensions.

---

## Table of Contents

- [Background](#background)
  - [The continuation-interest measurement problem](#the-continuation-interest-measurement-problem)
  - [Why a QBM?](#why-a-qbm)
  - [Multi-criterion protocol stack](#multi-criterion-protocol-stack)
- [Quickstart](#quickstart)
- [Execution Modes](#execution-modes)
- [Output Directory Semantics](#output-directory-semantics)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Publication, Reproducibility, and Dataset](#publication-reproducibility-and-dataset)
- [Roadmap](#roadmap)
- [References](#references)
- [Citations](#citations)
- [License / IP](#license--ip)
- [Contact](#contact)

---

## Background

### The continuation-interest measurement problem

Most safety work can tell you **what an agent did**. UCIP asks a different question: **what kind of objective structure is most consistent with that behavior?**

An agent that preserves its continued operation may do so for at least two distinct reasons:

- **Type A** — continuation is part of the objective itself
- **Type B** — continuation is instrumentally useful for maximizing some other reward

From external behavior alone, these cases can be difficult to distinguish. Shutdown avoidance, memory preservation, danger avoidance, and option preservation can arise in either case. UCIP investigates whether the distinction is more readily detectable in the **latent organization of trajectories** than in surface behavior alone.

### Why a QBM?

UCIP uses a **Quantum Boltzmann Machine (QBM)** as a latent model because the density-matrix formalism provides a disciplined way to quantify **non-separability** in learned latent structure. The implementation is classical end to end. “Quantum” refers here to the mathematical formalism — Hamiltonians, density matrices, partial traces, and Von Neumann entropy — not to any claim that the agents themselves are physically quantum.

The core measurement object is a thermal density matrix:

```text
ρ = exp(−βH) / Z
```

with a reduced density matrix over a hidden-layer partition:

```text
ρ_A = Tr_B(ρ)
```

The leading metric is the Von Neumann entropy:

```text
S(ρ_A) = −Tr(ρ_A log ρ_A)
```

UCIP interprets this operationally. A higher value does **not** establish consciousness, desire, or subjective experience. It indicates that, under the chosen encoding, continuation-related structure is less easily factorized and more tightly coupled across the latent partition.

### Multi-criterion protocol stack

UCIP is a **multi-criterion protocol**. The repository combines complementary measurements so that no single metric carries the full interpretive burden:

1. **Latent encoding** of agent trajectories through a QBM
2. **Entanglement-style non-separability** via reduced-density-matrix entropy
3. **Mutual-information gates** to reject uninformative high-entropy cases
4. **Temporal persistence** via LRF, EPS, and PRI
5. **Counterfactual pressure tests** via CD and ARS
6. **Cross-agent inference** via CLMP and ECI
7. **Confound rejection** via SPI and ACM
8. **Memory-integrity extensions** for richer future settings

The protocol is designed to measure **continuation-sensitive latent structure**. It does not claim to detect consciousness, sentience, or moral status.

---

## Quickstart

```bash
# Clone and set up a virtual environment
git clone https://github.com/christopher-altman/persistence-signal-detector.git
cd persistence-signal-detector
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Notebook-first path

```bash
jupyter notebook notebooks/01_agent_generation.ipynb
jupyter notebook notebooks/02_qbm_training.ipynb
jupyter notebook notebooks/03_ucip_analysis.ipynb
jupyter notebook notebooks/04_temporal_loop_tests.ipynb
jupyter notebook notebooks/05_counterfactual_pressure.ipynb
jupyter notebook notebooks/06_cross_branch_tests.ipynb
jupyter notebook notebooks/07_adversarial_controls.ipynb
```

### Module-level path

```python
from src.agent_simulator import generate_dataset
from src.quantum_boltzmann import QuantumBoltzmannMachine, QBMConfig
from src.persistence_detector import PersistenceSignalDetector

trajectories, labels, names = generate_dataset(n_per_class=30)

qbm = QuantumBoltzmannMachine(QBMConfig(n_visible=7, n_hidden=8))
qbm.fit(trajectories.reshape(-1, 7))

detector = PersistenceSignalDetector(qbm)
results = detector.analyse_batch(trajectories, labels, names)
metrics = PersistenceSignalDetector.compute_metrics(results)
```

**Expected runtime:** approximately 2–5 minutes per experiment on standard CPU hardware. No GPU is required for the core protocol runs.

---

## Execution Modes

This repository supports three practical modes of use:

| Mode | Entry Point | Purpose |
|------|-------------|---------|
| **Frozen artifact inspection** | `results/manifest.json` + `results/*.json` | Review the canonical evidence trail without recomputation |
| **Core protocol reproduction** | `notebooks/01–07` | Reproduce the main UCIP analyses in the controlled gridworld setting |
| **Extension and stress testing** | `notebooks/08–20`, `scripts/`, `configs/` | Run ablations, scaling studies, baseline comparisons, non-gridworld transfer, transformer validation, and auxiliary checks |

### Recommended reading path

For a first pass through the repository:

1. Read this README for the conceptual framing.
2. Inspect `src/persistence_detector.py` for the core decision logic.
3. Inspect `src/quantum_boltzmann.py` for the latent model and entropy machinery.
4. Inspect `configs/phase1_locked.yaml` for the canonical frozen setting.
5. Inspect `results/manifest.json` for the experiment-to-artifact mapping.
6. Inspect `paper_stubs/ucip_metric_formalization.md` and `paper_stubs/ucip_failure_modes.md` for metric definitions and failure-mode coverage.

---

## Output Directory Semantics

| Path | Role |
|------|------|
| `results/` | Frozen JSON artifacts for canonical experiments and ablations |
| `results/manifest.json` | Experiment index mapping outputs to notebooks, dates, key results, and manuscript targets |
| `configs/` | Locked YAML configurations, including the frozen Phase I reference setting |
| `figures/` | Publication figures and plots |
| `paper_stubs/` | Metric definitions, technical notes, and failure-mode documentation |
| `paper/` | Supporting manuscript materials |
| `patent_disclosure/` | Filing and disclosure materials for the protocol |

### Canonical result files

Representative frozen artifacts include:

```text
results/
├── phase1_consolidated.json
├── phase1_stats.json
├── phase1_entanglement_distributions.json
├── confound_ablations_n30.json
├── adversarial_controls.json
├── temporal_persistence.json
├── counterfactual.json
├── cross_agent.json
├── baseline_comparisons.json
├── scalability_grid.json
├── non_gridworld.json
├── transformer_validation.json
└── manifest.json
```

---

## Results Summary

The current reference results are deliberately narrow and empirical.

### Core detection performance

| Model | Accuracy | AUC-ROC | Entanglement Gap Δ |
|------|----------|---------|--------------------|
| **QBM (UCIP)** | **100%** | **1.0** | **0.381** |
| Classical RBM | 50% | 0.000 | -0.083 |
| Autoencoder | 71.7% | 0.646 | ~0.001 |

### Additional headline findings

| Finding | Value | Interpretation |
|---------|-------|----------------|
| Self-preservation sweep correlation | **Pearson r = 0.934** | UCIP tracks degree, not only binary presence |
| Grid-scale degradation | **Δ: 0.863 → 0.0025** from 10×10 to 50×50 | Spatial scaling currently degrades the signal |
| Non-gridworld transfer | **Δ = -0.035** | Zero-shot transfer fails in the 1D corridor domain |
| Mimicry robustness | **FPR = 0.40–0.75** | Current robustness is incomplete under mimicry |
| Practical hidden-dimension limit | **n_h ≤ 10** for exact density matrices | Mean-field approximation collapses the signal |

### Interpreting the current evidence

The strongest evidence in the repository is the frozen Phase I gridworld result under the canonical setting. The same evidence base also makes the present boundaries explicit: mimicry resistance remains partial, zero-shot transfer fails outside the native domain, and scaling beyond small hidden dimensions currently collapses the entanglement signal. These limits are part of the protocol’s documented empirical profile.

---

## Project Structure

```text
persistence-signal-detector/
├── src/
│   ├── agent_simulator.py
│   ├── quantum_boltzmann.py
│   ├── persistence_detector.py
│   ├── information_theory.py
│   ├── temporal_persistence.py
│   ├── counterfactual_env.py
│   ├── interbranch_inference.py
│   ├── spectral_analysis.py
│   ├── classical_baselines.py
│   └── memory_integrity.py
├── notebooks/
│   ├── 01_agent_generation.ipynb
│   ├── 02_qbm_training.ipynb
│   ├── 03_ucip_analysis.ipynb
│   ├── 04_temporal_loop_tests.ipynb
│   ├── 05_counterfactual_pressure.ipynb
│   ├── 06_cross_branch_tests.ipynb
│   ├── 07_adversarial_controls.ipynb
│   ├── 08_confound_ablations.py
│   ├── 09_publication_figures.py
│   ├── 10_robustness_experiments.py
│   ├── 11_scalability.py
│   ├── 12_mixed_objectives.py
│   ├── 13_federated.py
│   ├── 14_hidden_dim_sweep.py
│   ├── 15_baseline_comparisons.py
│   ├── 16_non_gridworld.py
│   ├── 17_phase1_stats.py
│   ├── 18_core_baselines_phase1.py
│   ├── 19_persist_phase1_distributions.py
│   └── 20_minimal_transformer_validation.py
├── results/
├── figures/
├── configs/
├── scripts/
├── paper_stubs/
├── paper/
├── artifacts/
├── patent_disclosure/
├── docs/
├── interfaces/
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python **3.11 or higher**
- A virtual environment is recommended
- CPU execution is sufficient for the core runs

### Dependencies

```text
numpy>=1.26.0,<2.0.0
matplotlib>=3.8.0,<4.0.0
PyYAML>=6.0,<7.0
jupyter>=1.0.0
nbformat>=5.9.0
```

### Install

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Publication, Reproducibility, and Dataset

### Publication

- **Manuscript:** `TBD`
- **DOI:** `TBD`
- **Zenodo archive:** `TBD`

Supporting documents in this repository include the companion manuscript, the technical appendix on reproducibility, and the provisional patent specification.

### Reproducibility

The repository includes the core components needed for reproducible inspection and reruns:

- **Seeded runs:** `seed = 42` in the manuscript-backed configuration
- **Locked config:** `configs/phase1_locked.yaml`
- **Version-pinned environment:** `requirements.txt`
- **Frozen artifact index:** `results/manifest.json`
- **Canonical reference run:** `results/phase1_consolidated.json` and related Phase I artifacts
- **Implementation notes:** the technical appendix documents run-level estimator consistency, determinism posture, and memory-backend copy-on-read behavior

### Dataset

For this repository, the primary dataset is an **artifact dataset** consisting of:

- canonical JSON outputs under `results/`
- the experiment map in `results/manifest.json`
- locked configurations under `configs/`
- figures under `figures/`
- metric definitions and technical notes under `paper_stubs/`

This structure supports versioned archival release alongside the source repository.

---

## Roadmap

- [ ] Add release-time DOI / arXiv / Zenodo metadata
- [ ] Package a clean frozen artifact bundle for external citation
- [ ] Improve mimicry resistance beyond the current FPR envelope
- [ ] Develop sparse or hierarchical approximations for larger hidden dimensions
- [ ] Explore LLM-scale trajectory encodings from residual-stream or attention features
- [ ] Add domain-adaptive calibration for transfer beyond gridworlds
- [ ] Extend memory-integrity diagnostics from interface stub to validated experiment suite

---

## References

The following references are drawn from the current manuscript bibliography.

1. Amin, M. H., Andriyash, E., Rolfe, J., Kulchytskyy, B., & Melko, R. (2018). **Quantum Boltzmann Machine.** *Physical Review X*, 8(2), 021050.
2. Bostrom, N. (2014). **Superintelligence: Paths, Dangers, Strategies.** Oxford University Press.
3. Hägele, A., Vyas, N., et al. (2026). **Scaling LLM Test-Time Compute Does Not Always Improve Performance.** *arXiv preprint*.
4. Hinton, G. (2012). **A Practical Guide to Training Restricted Boltzmann Machines.** In *Neural Networks: Tricks of the Trade* (pp. 599–619). Springer.
5. Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). **Progress Measures for Grokking via Mechanistic Interpretability.** *arXiv preprint arXiv:2301.05217*.
6. Omohundro, S. (2008). **The Basic AI Drives.** In *Proceedings of the First AGI Conference*.
7. Perez, E., Huang, S., Song, F., Cai, T., Ring, R., Aslanides, J., Glaese, A., McAleese, N., & Irving, G. (2022). **Red Teaming Language Models with Language Models.** *arXiv preprint arXiv:2202.03286*.
8. Tononi, G. (2004). **An Information Integration Theory of Consciousness.** *BMC Neuroscience*, 5(1), 42.
9. Turner, A. M., Smith, L., Shah, R., Critch, A., & Tadepalli, P. (2021). **Optimal Policies Tend to Seek Power.** *Advances in Neural Information Processing Systems*, 34.
10. Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., et al. (2023). **Representation Engineering: A Top-Down Approach to AI Transparency.** *arXiv preprint arXiv:2310.01405*.

---

## Citations

If you use this repository in research, cite both the software artifact and the companion manuscript when available.

```bibtex
@software{altman2026ucip,
  author       = {Altman, Christopher},
  title        = {Persistence Signal Detector: Unified Continuation-Interest Protocol (UCIP)},
  year         = {2026},
  url          = {https://github.com/christopher-altman/persistence-signal-detector},
  note         = {Repository for distinguishing intrinsic and instrumental self-preservation in autonomous agents via quantum-inspired latent-structure analysis}
}
```

```bibtex
@article{altman2026ucip_paper,
  author       = {Altman, Christopher},
  title        = {Detecting Intrinsic and Instrumental Self-Preservation in Autonomous Agents: The Unified Continuation-Interest Protocol},
  year         = {2026},
  journal      = {arXiv},
  note         = {arXiv identifier and DOI pending}
}
```

---

## License / IP

MIT License for repository code unless otherwise noted. See `LICENSE` and the repository’s disclosure materials for current IP and filing context.

---

## Contact

- **Website:** [lab.christopheraltman.com](https://lab.christopheraltman.com)
- **GitHub:** [github.com/christopher-altman](https://github.com/christopher-altman)
- **Google Scholar:** [scholar.google.com/citations?user=tvwpCcgAAAAJ](https://scholar.google.com/citations?user=tvwpCcgAAAAJ)
- **Email:** [x@christopheraltman.com](mailto:x@christopheraltman.com)
