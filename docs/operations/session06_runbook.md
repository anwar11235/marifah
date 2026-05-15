# Session 6 Vast.ai Runbook

**Purpose:** Provision a Vast.ai 2× A100 SXM4 40GB instance (Quebec, CA), verify the container image, generate the full synthetic dataset, and run the warm-start comparison (cold + warm in parallel) that resolves codebook OD7 (Sudoku Phase 3c vs. from-scratch init).

**Expected total runtime on instance:** ~3–4 hours (parallel warm-start)
- Container setup: ~5 min
- Dataset generation: ~15–30 min on 64 server-grade CPUs
- Warm-start runs (parallel on 2 GPUs): ~1–2 hours wall-clock
- Probe runs: ~10 min total
- Fallback to single-GPU sequential: ~5–6 hours total

---

## 1. Provision instance

**Specs:**
- GPU: 2× A100 SXM4 40GB (Quebec, CA — supports parallel cold/warm runs)
- Image: `anwar1919/coral-v3:2026-04-20`
- Storage: ≥ 50 GB (dataset ~4–5 GB, checkpoints ~1–2 GB)

Provision from the Vast.ai console with the above image. Note the instance IP.

> **Note on GPU memory:** Each A100 SXM4 is 40GB (not 80GB). The warm-start comparison runs at `batch_size=64, d_model=512` — if OOM occurs on 40GB, reduce to `batch_size=32` and report. Phase 0 main training (Session 7) may need an 80GB instance depending on observed memory.

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

## 3. Verify dual-GPU configuration

Before launching parallel runs, confirm both GPUs are visible and `CUDA_VISIBLE_DEVICES` is respected:

```bash
# Confirm both GPUs are visible
nvidia-smi
# Expected output: two A100 SXM4 entries (GPU 0 and GPU 1)
```

```bash
# Confirm CUDA_VISIBLE_DEVICES is respected
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(f'GPU count with CVD=0: {torch.cuda.device_count()}')"
# Expected: 1

CUDA_VISIBLE_DEVICES=1 python -c "import torch; print(f'GPU count with CVD=1: {torch.cuda.device_count()}')"
# Expected: 1
```

**If either check fails:** fall back to sequential execution — skip to the "Fallback: Sequential warm-start runs" section below and run the two training jobs one after the other.

---

## 4. Generate full synthetic dataset

```bash
python -m marifah.data.synthetic.cli generate-full \
    --config configs/default.yaml \
    --output /workspace/data/marifah_full_dataset \
    --workers 16
```

**Expected time:** 15–30 min on 64-core server-grade CPU with 16 workers. If slow, try `--workers 32`.

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

## 5. Run warm-start runs in parallel

Both runs launch simultaneously, each pinned to a different GPU via `CUDA_VISIBLE_DEVICES`. Total wall-clock: ~1–2 hours (whichever run takes longer), vs. ~3–4 hours sequential.

```bash
mkdir -p logs

# Launch cold run on GPU 0 (background)
CUDA_VISIBLE_DEVICES=0 python -m marifah.training.cli train \
    --config configs/warmstart_cold.yaml \
    --device cuda > logs/warmstart_cold.log 2>&1 &
COLD_PID=$!
echo "Cold run started, PID=$COLD_PID, log=logs/warmstart_cold.log"

# Launch warm run on GPU 1 (background)
CUDA_VISIBLE_DEVICES=1 python -m marifah.training.cli train \
    --config configs/warmstart_warm.yaml \
    --device cuda > logs/warmstart_warm.log 2>&1 &
WARM_PID=$!
echo "Warm run started, PID=$WARM_PID, log=logs/warmstart_warm.log"

# Wait for both to finish and capture exit codes
wait $COLD_PID
COLD_EXIT=$?
wait $WARM_PID
WARM_EXIT=$?

echo "Cold run exit code: $COLD_EXIT"
echo "Warm run exit code: $WARM_EXIT"

if [ $COLD_EXIT -ne 0 ] || [ $WARM_EXIT -ne 0 ]; then
    echo "ERROR: At least one run failed. Check logs."
    exit 1
fi

echo "Both runs completed successfully."
```

### Monitor progress (optional — open a second SSH session)

```bash
# Tail both logs simultaneously
tail -f logs/warmstart_cold.log logs/warmstart_warm.log

# Or check GPU utilization to confirm both GPUs are busy
watch -n 5 nvidia-smi
```

Expected outputs when complete:
- `checkpoints/warmstart_cold/final.pt`
- `checkpoints/warmstart_warm/final.pt`

---

## Fallback: Sequential warm-start runs

Use this section if the dual-GPU verification (§3) failed, or if you want to run sequentially for any reason.

```bash
# Cold run first (~1–2 hours)
python -m marifah.training.cli train \
    --config configs/warmstart_cold.yaml \
    --device cuda

# Warm run second (~1–2 hours)
python -m marifah.training.cli train \
    --config configs/warmstart_warm.yaml \
    --device cuda
```

Total sequential wall-clock: ~3–4 hours instead of ~1–2 hours parallel.

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
2. **Dual-GPU verification result** — both GPUs visible and `CUDA_VISIBLE_DEVICES` respected?
3. **Execution mode used** — parallel or sequential fallback?
4. **Dataset generation time** — actual wall-clock time and throughput (DAGs/sec)
5. **Dataset validation output** — copy of the `validate-dataset` output
6. **Peak GPU memory** — observed per GPU during training (from `nvidia-smi`)
7. **Wall-clock for parallel runs** — time from first launch to both `wait` calls completing
8. **Any operational issues** — unexpected errors, interruptions, config problems

---

## Troubleshooting

### OOM during training
If training OOMs at `batch_size=64`, reduce to `batch_size=32` by editing the YAML before relaunching:
```bash
sed -i 's/batch_size: 64/batch_size: 32/' configs/warmstart_cold.yaml
sed -i 's/batch_size: 64/batch_size: 32/' configs/warmstart_warm.yaml
```

Report the OOM and the batch_size that worked so Phase 0 config can be adjusted before the main launch.

### One parallel run failed
Check the log for the failed run:
```bash
cat logs/warmstart_cold.log   # or warmstart_warm.log
```

If the failure is recoverable (e.g., dataset path wrong), fix and re-run that config alone:
```bash
CUDA_VISIBLE_DEVICES=0 python -m marifah.training.cli train \
    --config configs/warmstart_cold.yaml --device cuda
```

### Dataset generation timeout
If generation is slow with 16 workers, try `--workers 32`. Note actual throughput (DAGs/sec) for Session 7 planning.

### Checkpoint not found for warm run
Verify: `ls checkpoints/sudoku-phase3c/phase3c_canonical_seed0_best.pt`
If missing, copy from local per §2c above.

### W&B connectivity issues
Set `WANDB_MODE=offline` before launching runs and sync after:
```bash
export WANDB_MODE=offline
# ... run training ...
wandb sync checkpoints/warmstart_cold/wandb/
wandb sync checkpoints/warmstart_warm/wandb/
```
