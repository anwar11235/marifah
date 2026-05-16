# Session 6 Vast.ai Runbook

**Purpose:** Provision a Vast.ai A100 SXM4 80GB instance, verify the container image, generate the full synthetic dataset, and run the warm-start comparison that resolves codebook OD7 (Sudoku Phase 3c vs. from-scratch init).

**Expected total runtime on instance:** ~5–6 hours
- Container verification: ~5 min
- Dataset generation: ~30–60 min (server-grade CPUs)
- Warm-start cold run (5K steps): ~1–2 hours
- Warm-start warm run (5K steps): ~1–2 hours
- Probe runs: ~10 min total

---

## 1. Provision instance

**Specs required:**
- GPU: A100 SXM4 80GB (required for Phase 0 GPU-memory compatibility check)
- Image: `anwar1919/coral-v3:2026-04-20`
- Storage: ≥ 50 GB (dataset ~4–5 GB, checkpoints ~1–2 GB)

Provision from the Vast.ai console with the above image. Note the instance IP.

---

## 2. Setup

### 2a. Install runtime deps

```bash
cd /workspace/marifah-core
pip install -r requirements.txt --break-system-packages
pip install -e . --break-system-packages
```

### 2b. Verify deps

```bash
python -c "
import torch, scipy, pyarrow, yaml, networkx, pydantic, wandb, einops, sklearn
import flash_attn
print('all deps OK')
print('  torch      :', torch.__version__)
print('  flash_attn :', flash_attn.__version__)
print('  cuda       :', torch.cuda.is_available(), torch.version.cuda if torch.cuda.is_available() else 'n/a')
"
```

```bash
python -m pytest tests/ -x --timeout=300 -q
```

**Stop here if anything fails.** Report the error before proceeding.

### 2c. Copy Sudoku Phase 3c checkpoint

The checkpoint must be at `checkpoints/sudoku-phase3c/phase3c_canonical_seed0_best.pt` relative to repo root.

Option A — if checkpoint is in the repo (it's tracked by git):
```bash
ls checkpoints/sudoku-phase3c/
# Expected: config.yaml  phase3c_canonical_seed0_best.pt
```

Option B — rsync from local:
```bash
# Run on your local machine:
rsync -av checkpoints/sudoku-phase3c/ vast:/workspace/marifah-core/checkpoints/sudoku-phase3c/
```

Verify the checkpoint loads:
```bash
python -c "
import torch
ckpt = torch.load('checkpoints/sudoku-phase3c/phase3c_canonical_seed0_best.pt', map_location='cpu', weights_only=False)
print('Checkpoint loaded. Keys (first 5):', list(ckpt.keys())[:5])
print('Total keys:', len(ckpt))
"
```

### 2d. W&B login

```bash
wandb login
```

The warmstart configs use `wandb_mode: online`. If you prefer offline, override:
```bash
export WANDB_MODE=offline
```

---

## 3. Generate full synthetic dataset

```bash
python -m marifah.data.synthetic.cli generate-full \
    --config configs/default.yaml \
    --output /workspace/data/marifah_full_dataset \
    --workers 8
```

**Expected time:** 30–60 min on server-grade CPU with 8 workers.
**Expected output:** ~800K train DAGs + validation/test splits.

Verify:
```bash
python -m marifah.data.synthetic.cli validate-dataset /workspace/data/marifah_full_dataset
```

Expected: `PASSED` for all splits. If validation fails, stop and report.

Spot-check workflow frequency distribution:
```bash
python -c "
import json
with open('/workspace/data/marifah_full_dataset/manifest.json') as f:
    m = json.load(f)
print('Splits and sizes:')
for k, v in m.get('split_sizes', {}).items():
    print(f'  {k}: {v}')
"
```

Confirm 5 frequency tiers spanning ~3 orders of magnitude (workflow type counts from ~50 to ~100,000).

---

## 4. Run cold-start comparison

```bash
python -m marifah.training.cli train \
    --config configs/warmstart_cold.yaml \
    --device cuda
```

**Expected duration:** ~1–2 hours (5000 steps at batch_size=64).

The run exits when `max_steps=5000` is hit. Final checkpoint saved to `checkpoints/warmstart_cold/final.pt`.

Monitor GPU memory:
```bash
# In a separate terminal while training:
watch -n 30 nvidia-smi
```

Record peak GPU memory usage and report it back.

---

## 5. Run warm-start comparison

```bash
python -m marifah.training.cli train \
    --config configs/warmstart_warm.yaml \
    --device cuda
```

This is identical to the cold run except it loads the Sudoku Phase 3c checkpoint before training. The config references `checkpoints/sudoku-phase3c/phase3c_canonical_seed0_best.pt`.

**Expected duration:** ~1–2 hours (5000 steps at batch_size=64).

Final checkpoint saved to `checkpoints/warmstart_warm/final.pt`.

---

## 6. Run probes on both checkpoints

```bash
mkdir -p results

python scripts/warmstart_probe.py \
    --checkpoint checkpoints/warmstart_cold/final.pt \
    --dataset /workspace/data/marifah_full_dataset \
    --split val \
    --config configs/warmstart_cold.yaml \
    --output results/cold_results.json \
    --max_samples 1000 \
    --device cuda

python scripts/warmstart_probe.py \
    --checkpoint checkpoints/warmstart_warm/final.pt \
    --dataset /workspace/data/marifah_full_dataset \
    --split val \
    --config configs/warmstart_warm.yaml \
    --output results/warm_results.json \
    --max_samples 1000 \
    --device cuda
```

**Expected time:** ~5 min each.

Preview results inline:
```bash
python -c "
import json
for name in ['cold', 'warm']:
    with open(f'results/{name}_results.json') as f:
        r = json.load(f)
    print(f'=== {name} ===')
    print(f'  workflow_type_auc : {r[\"workflow_type_auc\"][\"auc\"]:.4f}')
    print(f'  mean_edit_distance: {r[\"execution_faithfulness\"][\"mean_edit_distance\"]:.4f}')
    print(f'  failure_rate      : {r[\"execution_faithfulness\"][\"failure_rate\"]:.4f}')
"
```

---

## 7. Share results back

Transfer these back to your local machine:

```bash
# Run on your local machine:
mkdir -p results
rsync -av vast:/workspace/marifah-core/results/ results/
```

Also report:
1. **Container verification result** — pass or fail; any error messages if fail
2. **Dataset generation time** — actual wall-clock time on the instance
3. **Dataset validation output** — copy of the `validate-dataset` output
4. **Peak GPU memory** — observed during warmstart_cold and warmstart_warm training
5. **Any operational issues** — unexpected errors, interruptions, config problems

---

## Troubleshooting

### OOM during training
If training OOMs at batch_size=64, reduce to 32:
```bash
# Override in CLI (if supported) or edit the YAML before training
```

Report the OOM so Phase 0 config can be adjusted before the main launch.

### Dataset generation timeout
If generation is slow with 8 workers, try `--workers 16`. Note actual throughput (DAGs/sec) for Session 7 planning.

### Checkpoint not found for warm run
Verify: `ls checkpoints/sudoku-phase3c/phase3c_canonical_seed0_best.pt`
If missing, copy from local per §2c above.

### W&B connectivity issues
Set `WANDB_MODE=offline` and sync after the run completes:
```bash
wandb sync checkpoints/warmstart_cold/wandb/
wandb sync checkpoints/warmstart_warm/wandb/
```
