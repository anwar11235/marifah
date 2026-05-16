"""End-to-end pipeline verification script for use on Vast.ai before Phase 0 launch.

Runs four phases in under 5 minutes on an A100 SXM4 80GB and emits a single
PASS/FAIL summary.  Do NOT run locally (no GPU, no full dataset).

Usage:
    python scripts/verify_pipeline_e2e.py \\
        --dataset /workspace/data/marifah_graph_v1 \\
        --config configs/warmstart_cold.yaml \\
        [--device cuda] \\
        [--output /workspace/verify_results.json]

Phases:
  1. Minimal training   — 4 steps, batch_size=4, smoke check forward+backward
  2. Eval pass          — one val pass on the trained checkpoint
  3. Probe pass         — carry-state extraction + faithfulness on ≤16 samples
  4. Summary            — PASS/FAIL table printed and optionally written to JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("verify_e2e")

_PASS = "PASS"
_FAIL = "FAIL"


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E2E pipeline verification — run on Vast before Phase 0 launch"
    )
    p.add_argument("--dataset", required=True, help="Dataset root (must contain train/val splits)")
    p.add_argument("--config", required=True, help="TrainingConfig YAML (e.g. configs/warmstart_cold.yaml)")
    p.add_argument("--device", default="cuda", help="Device: cuda or cpu (default: cuda)")
    p.add_argument("--output", default=None, help="Optional JSON output path for results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------


def _phase1_train(
    config_path: str,
    dataset_root: str,
    device: torch.device,
    ckpt_dir: str,
) -> Tuple[str, Dict[str, Any]]:
    """Run 4 training steps; return (status, info)."""
    from marifah.training.config import load_config
    from marifah.training.trainer import build_model, Trainer
    from marifah.training.data_pipeline import build_data_loaders
    from marifah.training.logging import TrainingLogger

    cfg = load_config(config_path)
    cfg.data.dataset_root = dataset_root
    cfg.training.max_steps = 4
    cfg.training.batch_size = 4
    cfg.training.eval_interval_epochs = 999   # skip epoch eval in training loop
    cfg.logging.checkpoint_dir = ckpt_dir
    cfg.logging.wandb_mode = "disabled"
    cfg.logging.heartbeat_interval_steps = 0

    torch.manual_seed(0)
    model = build_model(cfg, device)
    loaders = build_data_loaders(cfg)
    log_obj = TrainingLogger(cfg)
    trainer = Trainer(model, cfg, loaders, log_obj, device)

    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0

    final_ckpt = str(Path(ckpt_dir) / "final.pt")
    if not Path(final_ckpt).exists():
        return _FAIL, {"error": f"final.pt not written to {ckpt_dir}", "elapsed_s": elapsed}

    logger.info("Phase 1 done: %.2fs  final.pt=%s", elapsed, final_ckpt)
    return _PASS, {"elapsed_s": round(elapsed, 2), "checkpoint": final_ckpt}


def _phase2_eval(
    config_path: str,
    dataset_root: str,
    checkpoint: str,
    device: torch.device,
) -> Tuple[str, Dict[str, Any]]:
    """Load checkpoint, run one val pass, return (status, metrics)."""
    from marifah.training.config import load_config
    from marifah.training.trainer import build_model
    from marifah.training.checkpointing import load_checkpoint
    from marifah.training.data_pipeline import build_data_loaders
    from marifah.training.eval_loop import evaluate

    cfg = load_config(config_path)
    cfg.data.dataset_root = dataset_root
    cfg.training.batch_size = 4
    cfg.logging.wandb_mode = "disabled"

    model = build_model(cfg, device)
    load_checkpoint(checkpoint, model)
    loaders = build_data_loaders(cfg)
    val_loader = loaders.get("val")
    if val_loader is None:
        return _FAIL, {"error": "val split not found"}

    t0 = time.perf_counter()
    metrics = evaluate(model, val_loader, cfg, device)
    elapsed = time.perf_counter() - t0

    # Sanity: main loss must be finite
    loss_main = metrics.get("loss_main", float("nan"))
    if not (loss_main == loss_main):  # NaN check
        return _FAIL, {"error": f"eval loss_main is NaN", "elapsed_s": elapsed}

    logger.info("Phase 2 done: %.2fs  loss_main=%.4f  acc=%.4f",
                elapsed, loss_main, metrics.get("accuracy_node", float("nan")))
    return _PASS, {
        "elapsed_s": round(elapsed, 2),
        "loss_main": round(loss_main, 6),
        "accuracy_node": round(metrics.get("accuracy_node", 0.0), 6),
    }


def _phase3_probe(
    config_path: str,
    dataset_root: str,
    checkpoint: str,
    device: torch.device,
    max_samples: int = 16,
) -> Tuple[str, Dict[str, Any]]:
    """Run carry extraction + faithfulness probe; return (status, results)."""
    import sys, os
    scripts_dir = str(Path(__file__).parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from marifah.training.config import load_config
    from marifah.training.trainer import build_model
    from marifah.training.checkpointing import load_checkpoint
    from marifah.training.data_pipeline import build_data_loaders
    from warmstart_probe import _extract_carry_states, compute_execution_faithfulness

    cfg = load_config(config_path)
    cfg.data.dataset_root = dataset_root
    cfg.training.batch_size = 4
    cfg.logging.wandb_mode = "disabled"

    model = build_model(cfg, device)
    load_checkpoint(checkpoint, model)
    model.eval()

    loaders = build_data_loaders(cfg)
    val_loader = loaders.get("val")
    if val_loader is None:
        return _FAIL, {"error": "val split not found"}

    t0 = time.perf_counter()

    carry_states, labels = _extract_carry_states(model, val_loader, cfg, device, max_samples)

    # Reload val loader for faithfulness (carry extraction consumed it)
    loaders2 = build_data_loaders(cfg)
    faithfulness = compute_execution_faithfulness(
        model, loaders2["val"], cfg, device, max_samples
    )
    elapsed = time.perf_counter() - t0

    # Sanity: carry states must be finite
    import numpy as np
    if not np.isfinite(carry_states).all():
        return _FAIL, {"error": "carry_states contain non-finite values", "elapsed_s": elapsed}

    mean_ed = faithfulness.get("mean_edit_distance", float("nan"))
    logger.info("Phase 3 done: %.2fs  n=%d  mean_edit_dist=%.4f",
                elapsed, faithfulness.get("n_samples", 0), mean_ed)
    return _PASS, {
        "elapsed_s": round(elapsed, 2),
        "carry_shape": list(carry_states.shape),
        "n_labels": int(len(labels)),
        "mean_edit_distance": round(float(mean_ed), 6) if mean_ed == mean_ed else None,
        "failure_rate": round(faithfulness.get("failure_rate", 0.0), 6),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary(results: Dict[str, Dict[str, Any]]) -> bool:
    """Print PASS/FAIL table; return True iff all phases passed."""
    print("\n" + "=" * 60)
    print("  END-TO-END PIPELINE VERIFICATION SUMMARY")
    print("=" * 60)
    all_pass = True
    for phase, (status, info) in results.items():
        mark = "✓" if status == _PASS else "✗"
        print(f"  {mark} {phase:30s}  {status}")
        if status != _PASS:
            all_pass = False
            print(f"      error: {info.get('error', '?')}")
    print("=" * 60)
    verdict = "PASS — pipeline verified, safe to launch Phase 0." if all_pass \
        else "FAIL — one or more phases failed; do NOT launch Phase 0."
    print(f"\n  VERDICT: {verdict}\n")
    return all_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    logger.info("Device: %s | config: %s | dataset: %s", device, args.config, args.dataset)

    results: Dict[str, Tuple[str, Dict[str, Any]]] = {}

    with tempfile.TemporaryDirectory(prefix="marifah_verify_") as tmpdir:
        ckpt_dir = str(Path(tmpdir) / "ckpts")

        # Phase 1
        logger.info("=== Phase 1: Minimal training ===")
        status1, info1 = _phase1_train(args.config, args.dataset, device, ckpt_dir)
        results["Phase 1: training (4 steps)"] = (status1, info1)

        checkpoint = info1.get("checkpoint", "")

        # Phase 2 — only if Phase 1 produced a checkpoint
        logger.info("=== Phase 2: Eval pass ===")
        if status1 == _PASS and checkpoint:
            status2, info2 = _phase2_eval(args.config, args.dataset, checkpoint, device)
        else:
            status2, info2 = _FAIL, {"error": "skipped — Phase 1 failed"}
        results["Phase 2: eval pass"] = (status2, info2)

        # Phase 3 — only if Phase 1 produced a checkpoint
        logger.info("=== Phase 3: Probe pass ===")
        if status1 == _PASS and checkpoint:
            status3, info3 = _phase3_probe(args.config, args.dataset, checkpoint, device)
        else:
            status3, info3 = _FAIL, {"error": "skipped — Phase 1 failed"}
        results["Phase 3: probe (n=16)"] = (status3, info3)

    # Phase 4: summary
    all_pass = _print_summary(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            phase: {"status": status, **info}
            for phase, (status, info) in results.items()
        }
        payload["verdict"] = "PASS" if all_pass else "FAIL"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Results written to %s", out_path)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
