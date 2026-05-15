# marifah-core

**Marifah** is a company building Cortical Reasoning Architecture (CRA) for Cognitive Workflow Automation. This repository is the active development home for CORAL — the first instantiation of CRA — focused on graph DAG execution for enterprise workflows.

## Architecture

**CORAL** (Cortical Reasoning via Abstraction Layers) is the first architecture in the CRA family. It is a hierarchical reasoning model with two-level nested recurrence (H-level and L-level modules), adaptive computation time (ACT) halting, and a predictive coding substrate that implements free-energy minimization across levels.

**The Nous substrate** (the Reasoning Cortex) is the core inference engine: a stack of Post-Norm Transformer blocks running H_cycles × L_cycles recurrent steps per segment, with 1-step gradient deep supervision. The substrate generalizes across task domains without task-specific architecture changes.

**The Marifah mechanism** (the Recognition Cortex) is the deployment-time compounding layer: a Soft MoE crystallization codebook that amortizes recurring computations into spatial mode templates, making the system progressively cheaper on familiar workflows at inference time.

The current development focus is **graph DAG execution for enterprise workflows** — encoding, reasoning over, and faithfully executing structured cognitive workflows represented as directed acyclic graphs.

## Intellectual lineage

The architecture inherits from two converging traditions. From the Western thread: Helmholtz's conception of perception as inference, extended by Friston's free energy principle and active inference framework. From the Eastern thread: al-Ghazali's epistemology of illuminated cognition, refined through Suhrawardi's Illuminationist ontology, deepened by Ibn ʿArabi's theory of imaginal knowing, and synthesized by Mulla Sadra's doctrine of the intensification of being — the idea that recognition is not retrieval but a progressive deepening of contact with form. The architecture attempts to implement this: not pattern matching, but genuine hierarchical abstraction that sharpens with use.

For precise vocabulary and taxonomy, see `Marifah_Naming_and_Taxonomy.md` (source of truth for all architectural naming).

## Repository structure

```
src/marifah/
├── models/          Core CORAL architecture (Nous substrate)
│   ├── coral.py     CoralV3Inner — main model with PC + crystallization
│   ├── coral_base.py CoralInner + CoralConfig + InnerCarry
│   ├── act.py       CoralACT + CoralV3ACT wrappers
│   ├── codebook.py  SpatialMoECodebook (Marifah mechanism scaffold)
│   └── ...
├── training/        Training loop, losses, optimizer, scheduler
└── data/            Base dataset loader (HRM-format)
checkpoints/
└── sudoku-phase3c/  Best Sudoku seed-0 checkpoint (68.74%, W&B wpmdrf8n)
```

The archived research repo where Sudoku and ARC validation were conducted: [CORAL-v3](https://github.com/anwar11235/CORAL-v3) (tagged `v3.0-pre-marifah-pivot`).

## Setup

```bash
git clone <this-repo>
cd marifah
pip install -e .

# Optional: flash-attn (A100 only, ~4x attention speedup)
FLASH_ATTN_CUDA_ARCHS=80 pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## Docker (Vast.ai)

```bash
docker build -t anwar1919/marifah-core:2026-05-15 .
```

Run tests before any training session:

```bash
pytest tests/test_smoke.py -v
```
