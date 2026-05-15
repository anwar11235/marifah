# Session 01 — Repository Bootstrap

**Date:** 2026-05-15  
**Branch:** `session-1/repo-bootstrap`  
**Source repo:** CORAL-v3 @ commit `85f3a73` (master HEAD)  
**Source tag created:** `v3.0-pre-marifah-pivot` (pushed to CORAL-v3 origin)

---

## Goal

Bootstrap the `marifah-core` repository by selectively porting the core CORAL architecture from CORAL-v3 master, excluding all ARC-specific code. Establish the foundation for graph DAG execution work.

## Pre-flight actions

1. Verified CORAL-v3 was on `arc/eval-max-examples` branch; ported from master using `git show master:<path>`.
2. Dropped 2 stashes from CORAL-v3 (old housekeeping; user confirmed disposable).
3. Tagged CORAL-v3 master HEAD `85f3a73` as `v3.0-pre-marifah-pivot` and pushed to origin.

## Files ported

### Models

| Source (CORAL-v3) | Destination (marifah) | Notes |
|---|---|---|
| `coral/models/coral_v3.py` | `src/marifah/models/coral.py` | `_v3` suffix dropped; class `CoralV3Inner` kept |
| `coral/models/coral_base.py` | `src/marifah/models/coral_base.py` | `CoralConfig`, `CoralInner`, `InnerCarry` |
| `coral/training/act.py` | `src/marifah/models/act.py` | Moved from `training/` to `models/`; `CoralACT`, `CoralV3ACT` |
| `coral/models/crystallization.py` | `src/marifah/models/codebook.py` | Renamed; `SpatialMoECodebook`, `CrystallizationBuffer` |
| `coral/models/prediction.py` | `src/marifah/models/prediction.py` | `PredictionNet`, `PrecisionNet` |
| `coral/models/reasoning_module.py` | `src/marifah/models/reasoning_module.py` | `ReasoningModule` |
| `coral/models/transformer_block.py` | `src/marifah/models/transformer_block.py` | `TransformerBlock`, `TransformerBlockConfig` |
| `coral/models/layers.py` | `src/marifah/models/layers.py` | `CastedLinear`, `CastedEmbedding`, `Attention`, `SwiGLU`, `RotaryEmbedding` |
| `coral/models/sparse_embedding.py` | `src/marifah/models/sparse_embedding.py` | `CastedSparseEmbedding`, distributed variant |
| `coral/models/columnar.py` | `src/marifah/models/columnar.py` | Kept for checkpoint key compat; Phase 2 STUBBED (NotImplementedError) |
| `coral/models/common.py` | `src/marifah/utils/common.py` | Moved to `utils/`; `trunc_normal_init_`, `rms_norm` |

### Training

| Source (CORAL-v3) | Destination (marifah) | Notes |
|---|---|---|
| `coral/training/losses.py` | `src/marifah/training/losses.py` | `ACTLossHead`, `CoralV3LossHead`, `stablemax_cross_entropy` |
| `coral/training/scheduler.py` | `src/marifah/training/scheduler.py` | Exact copy |
| `coral/training/adam_atan2.py` | `src/marifah/training/adam_atan2.py` | Pure-PyTorch AdamATan2 fallback |
| `scripts/train.py` | `src/marifah/training/train.py` | Hydra config_path updated (3 levels up); log prefix changed to `[marifah]` |

### Data

| Source (CORAL-v3) | Destination (marifah) | Notes |
|---|---|---|
| `coral/data/puzzle_dataset.py` | `src/marifah/data/base_dataset.py` | Renamed; generic HRM base only; Sudoku subclass not ported |

### Project files

| Source | Destination | Notes |
|---|---|---|
| `requirements.txt` | `requirements.txt` | Package name references updated |
| `.gitignore` | `.gitignore` | Rescoped for marifah structure |
| `Dockerfile` | `Dockerfile` | Image tag → `anwar1919/marifah-core:2026-05-15`; ARC submodule clones removed |
| `pyproject.toml` | `pyproject.toml` | Package name: `coral` → `marifah`; `where = ["src"]` (src layout) |

## Files NOT ported

- `coral/data/build_arc_dataset.py`, `build_sudoku_dataset.py` — dataset builders stay in CORAL-v3
- All ARC-specific tests, scripts, and configs
- `.gitmodules` and ARC/ConceptARC submodules
- v2 metric-shape mechanism (stays in CORAL-v3; Run B verdict pending)
- R0/R0.5 probe code

## Import changes (all ported files)

All internal `coral.*` imports replaced with `marifah.*`. No external dependency changes. The `train.py` Hydra `config_path` changed from `"../configs"` (one level from `scripts/`) to `"../../../configs"` (three levels from `src/marifah/training/`).

## Checkpoint

Sudoku Phase 3c seed-0 checkpoint copied from disk to `checkpoints/sudoku-phase3c/`:

- `phase3c_canonical_seed0_best.pt` — 51 keys; best accuracy 68.74%; W&B run `wpmdrf8n`
- `config.yaml` — run config from W&B artifact `run-20260426_183033-wpmdrf8n/files/`

Checkpoint verified: loads cleanly under `torch.load(..., weights_only=False, map_location="cpu")`. Key structure begins with `model.inner.H_init`, `model.inner.L_init`, etc. (51 keys total). Bit-identical to CORAL-v3 warm-start.

## Verification

| Check | Status |
|---|---|
| `git status` clean on `session-1/repo-bootstrap` | ✅ |
| `pip install -e .` | ✅ |
| `pytest tests/test_smoke.py` — 9/9 passed (3.48s, CPU) | ✅ |
| Smoke test: import, instantiate, forward (base + v3+PC), ACT forward, backward + grad check, no-NaN loss | ✅ |
| Sudoku Phase 3c checkpoint loads, 51 keys, correct structure | ✅ |
| README.md coherent | ✅ |
| CLAUDE.md populated | ✅ |
| Tag `v3.0-pre-marifah-pivot` exists on CORAL-v3 and pushed | ✅ |
| Docker build verified | ⚠️ Deferred — no Docker daemon on Windows dev machine; verify on first Vast.ai instance |

## Decisions made

1. **`act.py` moved to `models/`** — target structure in session prompt placed it there; keeps inference-critical code co-located with model files.
2. **`coral_v3.py` → `coral.py`** — `_v3` was version-tagging not needed in a repo that IS v3.
3. **`crystallization.py` → `codebook.py`** — better reflects scope: just the Marifah mechanism scaffold.
4. **`puzzle_dataset.py` → `base_dataset.py`** — emphasizes generic base; Sudoku specifics stay in CORAL-v3.
5. **Source of truth for porting: CORAL-v3 master** — CORAL-v3 was on `arc/eval-max-examples` branch; porting used master (Sudoku Phase 3c work) throughout.

## Deviations from session prompt

- **Docker build not verified locally** — Windows; no Docker daemon. Dockerfile is structurally correct (flash-attn pin verified, ARC submodule clones removed). Verify on first Vast.ai provisioning.
- **`Marifah_Naming_and_Taxonomy.md` not found locally** — README references it. Must be made available before Session 2.
- **Additional model files ported beyond prompt's explicit list** — `coral_base.py`, `reasoning_module.py`, `prediction.py`, `sparse_embedding.py`, `columnar.py` are all direct import dependencies of `coral.py` and `act.py`. Porting them was required for the smoke tests to pass.
- **Company naming** — session prompt says "Marifah is the company"; game plan doc says "Aktuator". Followed the session prompt. Needs deliberate resolution before external messaging.

## Open items for subsequent sessions

- **Graph adapter (Session 2)** — no graph-specific code exists yet. Designs in `CORAL_DAG_Codebook_Design.md` and `CORAL_Synthetic_DAG_Benchmark_Spec.md`.
- **Synthetic DAG benchmark generator (Session 2)** — spec in `CORAL_Synthetic_DAG_Benchmark_Spec.md`.
- **HMSC codebook module (Session 3+)** — `SpatialMoECodebook` is scaffolding; real Marifah mechanism designed in `CORAL_DAG_Codebook_Design.md`.
- **W&B workspace migration** — `train.py` still points at `aktuator-ai` workspace. Separate decision.
- **Docker image push** — tagged `anwar1919/marifah-core:2026-05-15`; push on first Vast.ai provisioning.
- **`seq_len` config consideration** — codebook scaffold has implicit coupling to Sudoku `seq_len=81`. Graph adapter will need to reconfigure `seq_len` to match DAG node count.
- **Company naming resolution** — Marifah vs Aktuator. Clarify before any external communication.

## Commits

```
1a864c9  feat: port CORAL architecture from CORAL-v3 master (session-1/repo-bootstrap)
f1745f3  docs: add README and CLAUDE.md (session-1/repo-bootstrap)
<this>   docs: session-01 summary
```
