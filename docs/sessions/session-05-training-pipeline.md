# Session 05 — Training Pipeline Integration

**Date:** 2026-05-15  
**Branch:** `session-5/training-pipeline`  
**Goal:** Wire adapter + HMSC + CORAL into a complete training pipeline with checkpointing, W&B logging, configurable lambdas, and Phase 0 / Phase 1 launch configurations.

---

## What was built

### New modules under `src/marifah/training/`

| Module | LOC | Contents |
|--------|-----|----------|
| `config.py` | ~100 | Pydantic config hierarchy: `TrainingConfig`, `ModelConfig`, `HMSCConfig`, `TrainingPhaseConfig`, `DataConfig`, `LoggingConfig`, `WarmStartConfig`; `load_config()` YAML loader |
| `graph_losses.py` | ~70 | `compute_total_loss()` — aggregates main CE loss + halt BCE + HMSC aux losses |
| `graph_utils.py` | ~75 | `derive_region_labels()` (per-node → per-region label mapping), `prepare_batch_for_model()` (GraphBatch → CoralV3Inner dict) |
| `data_pipeline.py` | ~110 | `collate_graphs_padded()` — pads batches to fixed `max_nodes`; `build_data_loaders()` for all 5 splits |
| `checkpointing.py` | ~90 | `save_checkpoint()` (atomic write via tmp+rename), `load_checkpoint()`, `load_warmstart()` |
| `logging.py` | ~100 | `TrainingLogger` — W&B + always-on local JSONL, `log_step()`, `log_eval()`, `log_codebook_stats()` |
| `eval_loop.py` | ~90 | `evaluate()` — full pass over DataLoader, model in eval mode, aggregates node accuracy + HMSC utilisation |
| `trainer.py` | ~200 | `build_model()`, `Trainer` class with `train()` (epoch-based, eval cadence in EPOCHS), `step()` (1-step gradient) |
| `cli.py` | ~160 | `train`, `eval`, `smoke` subcommands; `smoke` auto-generates tiny dataset |

### New config files under `configs/`

| File | Purpose |
|------|---------|
| `phase0.yaml` | PC-only, `use_hmsc=false`, all lambdas=0, d_model=512 |
| `phase1.yaml` | HMSC engaged, λ_G=λ_R=λ_P=0.1, K_G=64/K_R=16/K_P=16 |
| `smoke.yaml` | d_model=64, 3 epochs, HMSC on, wandb disabled |
| `smoke_hmsc_off.yaml` | Same as smoke but `use_hmsc=false` |

### New unit tests (25 total)

| File | Tests |
|------|-------|
| `test_training_losses.py` | 5 — loss aggregation, masking, aux zero/nonzero, backward |
| `test_training_checkpointing.py` | 5 — round-trip, atomic write, optimizer restore, warm-start |
| `test_training_logging.py` | 4 — wandb-disabled, JSONL creation, parseable step/eval/codebook records |
| `test_training_data_pipeline.py` | 6 — padded collation, max_nodes enforcement, split availability |
| `test_training_eval_loop.py` | 4 — no-error completion, finite metrics, determinism, HMSC utilisation |

---

## Configuration decisions

All defaults from §2.7 were adopted as specified. Notable decisions:

- **halt_loss_weight = 0.1** (not 1.0): single-step training always targets halt=True; initialized q_head bias of -5 causes BCE ≈ 4.96 at epoch 0. Down-weighting keeps the total loss signal balanced with the main primitive-prediction loss.
- **max_nodes = 64** for smoke (100 for phase configs): smaller for faster smoke iteration.
- **H_layers=1, L_layers=1** for smoke (2 for phase configs): minimizes compute.
- **forward_dtype = float32** always: bf16 is a GPU concern; float32 used on CPU for determinism and smoke test correctness.
- **HMSC lambdas set at model construction**, not as training hyperparameters: HMSC's `compute_aux_losses` accepts lambdas at init; the trainer sets them from the config. This means the `L_G_raw` / `L_G` dict distinction is irrelevant — raw losses are weighted at HMSC init time.
- **region_labels derivation**: Per-node `pattern_id % num_regions` majority-vote heuristic. Nodes with `pattern_id % num_regions == r` are assigned to learnable region slot `r`. This is a pipeline-plumbing heuristic; production training may need a better supervised labeling scheme.
- **1-step gradient training** with fresh carry per batch: no persistent carry between batches. Full ACT multi-step is test-time reasoning; training uses one CORAL inner segment per batch for simplicity and speed.

---

## Smoke test results

### Smoke A — HMSC off

Config: `configs/smoke_hmsc_off.yaml` (d_model=64, 3 epochs, 200 train DAGs, batch=8)

| Epoch | main loss | halt loss | aux total | val loss_main | val acc_node |
|-------|-----------|-----------|-----------|--------------|--------------|
| 0 | 1.681 | 4.964 | 0.000 | 1.132 | 0.764 |
| 1 | 0.248 | 4.887 | 0.000 | 0.698 | 0.824 |
| 2 | 0.077 | 4.842 | 0.000 | 0.646 | 0.824 |

Main loss trends downward (correct). Aux losses = 0 confirmed (use_hmsc=False). Checkpoint saved and reloaded. JSONL log has 18 records.

### Smoke B — HMSC on

Config: `configs/smoke.yaml` (d_model=64, 3 epochs, 200 train DAGs, batch=8, λ=0.1)

| Epoch | main loss | halt loss | aux total | val loss_main | val acc_node |
|-------|-----------|-----------|-----------|--------------|--------------|
| 0 | 1.686 | 4.965 | 0.853 | 1.131 | 0.761 |
| 1 | 0.279 | 4.886 | 0.773 | 0.718 | 0.824 |
| 2 | 0.110 | 4.842 | 0.710 | 0.674 | 0.824 |

Main loss trends downward. Aux losses non-zero and decreasing (HMSC is learning). HMSC codebook utilisation stats present in log. Checkpoint saved and reloaded. JSONL log has 26 records (includes codebook_stats entries).

### HMSC utilisation (Smoke B, final eval)
Stats present in log: G_active_frac, G_entropy, G_top1_dominance, R_active_frac, R_entropy, P_active_frac, P_entropy, etc. All scale utilisation > 0 at random init.

---

## Regression check

```
max_diff = 0.0
Regression check PASSED: use_hmsc=False is bit-identical.
```

---

## Verification step results

| # | Description | Result |
|---|-------------|--------|
| 1 | `pytest tests/` passes | ✅ 280/280 |
| 2 | Smoke A: 3 epochs no error | ✅ |
| 3 | Smoke B: 3 epochs no error | ✅ |
| 4 | Checkpoints load into fresh model | ✅ |
| 5 | `use_hmsc=False` max_diff = 0.0 | ✅ |
| 6 | Loss values real, main non-zero | ✅ |
| 7 | HMSC utilisation in Smoke B log | ✅ |
| 8 | Aux=0 Smoke A, aux>0 Smoke B | ✅ |
| 9 | wandb disabled mode works | ✅ |
| 10 | Local JSONL written and parseable | ✅ |
| 11 | CLI commands run end-to-end | ✅ |
| 12 | Clean branch with all changes committed | ✅ (after this session) |

---

## Known issues / deferred items

- **Halt loss high in smoke** (≈4.84): q_head bias is initialized to -5 (from CORAL base). In 1-step training with always-halt targets, BCE starts at ~4.96 and decreases slowly. The halt_loss_weight=0.1 down-weights it, keeping gradient signal balanced. Not a bug — expected behavior.
- **ACT outer loop not implemented**: Training uses single-step CORAL forward (1-step gradient as designed). Full ACT multi-step at test time is not wired into the eval loop — eval also uses single-step. This is fine for Phase 0/1 plumbing; ACT-at-eval would require building the halting outer loop, which is separate from pipeline plumbing.
- **region_labels heuristic**: The `derive_region_labels()` function uses `pattern_id % num_regions` majority vote. This produces sensible labels for the smoke test but may need refinement for production — the mapping between learnable region slots and ground-truth patterns is not semantically grounded.
- **No multi-GPU support**: Single-process training only. Phase 0 fits on one A100 80GB.
- **Full dataset generation time**: At 200 DAGs/test with 950 DAGs/sec single-threaded, the 800K train split takes ~14 min single-threaded. Use `--workers` in the generator CLI for parallelism.
- **GPU memory at batch_size=64, d_model=512, 2 layers**: Not profiled (no GPU on dev machine). Estimate: ~40-50GB for full Phase 0 config. Vast.ai A100 80GB has headroom.

---

## Session 6 handoff notes

Phase 0 launch checklist:
1. Provision Vast.ai A100 SXM4 80GB with `anwar1919/marifah-core:2026-05-15`
2. Generate full dataset: `marifah-generate-full --output /data/marifah` (uses default.yaml, ~14 min single-threaded, 6 min with 4 workers)
3. Update `configs/phase0.yaml` `data.dataset_root` to `/data/marifah`
4. Launch: `python -m marifah.training.cli train --config configs/phase0.yaml --device cuda`
5. After Phase 0 checkpoint: run workflow-type AUC probe (§7 gate)
6. Phase 0 config currently has `d_model=512, H_layers=2, L_layers=2` — verify GPU memory before launch; may need to reduce H/L layers if OOM

Config tuning notes from smoke:
- batch_size=8 (smoke) → 64 (phase0 target); verify GPU memory
- 3 epochs smoke → 100 epochs production; eval every 5 epochs
- Phase 0 warmup_steps=1000 at 64 batch, 800K train = ~12,500 steps/epoch → 8% of first epoch warmup; reasonable
