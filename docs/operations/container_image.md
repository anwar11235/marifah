# Container Image: anwar1919/coral-v3:2026-04-20

**Image tag:** `anwar1919/coral-v3:2026-04-20`
**Base:** PyTorch + CUDA, NCCL, flash-attn pre-installed.
**Strategy:** Reuse existing image; install Session 2–5 deps at runtime via pip.

---

## New dependencies added in Sessions 2–5

Sessions 2–5 added the following packages not in the original image:

| Package | Session | Purpose |
|---------|---------|---------|
| `scipy` | 3 | Laplacian positional encoding (eigsh) |
| `pyarrow` | 2 | Parquet read/write for synthetic dataset |
| `pyyaml` | 5 | YAML config loading |
| `networkx` | 2 | DAG topology construction |
| `pydantic` | 5 | Training config validation |
| `wandb` | 5 | W&B experiment logging |
| `einops` | 4 | HMSC tensor operations |
| `scikit-learn` | 6 | Linear probe for workflow-type AUC |

---

## Provisioning workflow on Vast.ai

### 1. Pull image

```bash
docker pull anwar1919/coral-v3:2026-04-20
```

Estimated time: ~2 min on a clean Vast.ai instance.

### 2. Launch container and mount repo

When provisioning the Vast.ai instance, set the image to `anwar1919/coral-v3:2026-04-20`. The Vast.ai interface allows mounting a volume or cloning the repo at launch.

Alternatively, after launch:

```bash
# Option A: clone from GitHub
git clone https://github.com/anwar11235/marifah /workspace/marifah-core
cd /workspace/marifah-core

# Option B: rsync from local
rsync -av --exclude='.git' /path/to/local/marifah-core/ vast:/workspace/marifah-core/
```

### 3. Install runtime deps

```bash
cd /workspace/marifah-core
pip install -r requirements.txt --break-system-packages
pip install -e . --break-system-packages
```

Expected time: ~30–60s.

### 4. Verify deps

Run both of these before proceeding to dataset generation or training:

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

**Expected outcome:** All imports resolve; test suite passes (280/280 tests). GPU tests that are skipped on CPU will run on Vast.ai with CUDA.

If anything fails, stop and report before proceeding.

---

## Verification status

| Field | Value |
|-------|-------|
| Validation date | TBD — pending Vast.ai execution (Session 6 §2.5) |
| Validation outcome | TBD |
| Validated by | — |
| Image approved for Phase 0 | TBD |
| Notes | — |

*This section is updated by the user after executing the §2.5 Vast.ai verification step.*

---

## Notes

- If `flash_attn` import fails, the code falls back to PyTorch SDPA automatically (see `attention.py`). Training will be slower but correct.
- If any dep cannot be runtime-installed (e.g., incompatible CUDA version), escalate before rebuilding the image.
- Image rebuild is explicitly out of scope for Session 6 (see prompt §3 architectural constraints).
