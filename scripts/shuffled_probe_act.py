"""Shuffled-primitive probe with ACT-iterated z_H extraction.

Identical to shuffled_probe.py except z_H is extracted after running CoralV3ACT for
halt_max_steps iterations (4 in eval mode) instead of a single CoralV3Inner pass.

Runs two passes:
  1. Un-shuffled: full ACT iteration → z_H, probe workflow_type_id (auc_unshuffled)
  2. Shuffled: primitives randomized within each DAG → full ACT iteration → z_H, probe (auc_shuffled)

Path A criterion (same as shuffled_probe.py): auc_shuffled >= 0.65

Usage:
    python scripts/shuffled_probe_act.py \\
        --checkpoint checkpoints/warmstart_cold/final.pt \\
        --dataset /workspace/data/marifah_graph_v1 \\
        --split val \\
        --config configs/warmstart_cold.yaml \\
        --output results/shuffled_probe_act_cold.json \\
        [--max_samples 1000] \\
        [--shuffle_seed 42] \\
        [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# ACT model builder + loader (delegate to delta_probe_act)
# ---------------------------------------------------------------------------


def _build_act_model(config: Any, device: torch.device) -> Any:
    from delta_probe_act import _build_act_model as _b
    return _b(config, device)


def _load_checkpoint_into_act(act_model: Any, checkpoint_path: str, device: torch.device) -> None:
    from delta_probe_act import _load_checkpoint_into_act as _l
    _l(act_model, checkpoint_path, device)


# ---------------------------------------------------------------------------
# ACT-iterated z_H extraction (with optional primitive shuffling)
# ---------------------------------------------------------------------------


def _extract_z_H_pooled_act(
    act_model: Any,
    data_loader: Any,
    config: Any,
    device: torch.device,
    max_samples: int,
    shuffle_primitives: bool = False,
    shuffle_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Run ACT-iterated forward; return (pooled_z_H, labels, n_steps_used).

    Args:
        shuffle_primitives: If True, shuffle primitives within each DAG before forward.
        shuffle_seed:       Seed for shuffle RNG (for reproducibility).
    """
    from shuffled_probe import _shuffle_batch_primitives
    from marifah.training.graph_utils import prepare_batch_for_model

    rng_shuf = np.random.RandomState(shuffle_seed)
    z_H_list: List[np.ndarray] = []
    labels_list: List[int] = []
    step_counts: List[int] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            if shuffle_primitives:
                batch = _shuffle_batch_primitives(batch, device, rng_shuf)

            coral_batch = prepare_batch_for_model(batch, config, device)

            # Initialize ACT carry (all halted=True → fresh start)
            act_carry = act_model.initial_carry(coral_batch)

            # Iterate; in eval mode always runs halt_max_steps times
            steps_run = 0
            for step in range(config.model.halt_max_steps):
                act_carry, _ = act_model(act_carry, coral_batch)
                steps_run = step + 1
                if act_carry.halted.all():
                    break

            step_counts.append(steps_run)

            # Extract z_H from final ACT carry
            z_H = act_carry.inner_carry.z_H  # (B, max_nodes, d_model)

            node_mask = batch.node_mask.to(device)
            mask_f = node_mask.unsqueeze(-1).float()
            node_counts = mask_f.sum(dim=1).clamp(min=1)
            pooled = (z_H.float() * mask_f).sum(dim=1) / node_counts  # (B, d_model)

            z_H_list.append(pooled.cpu().numpy())
            wf_ids = (batch.workflow_type_id - 1).clamp(min=0).cpu().numpy()
            labels_list.extend(wf_ids.tolist())

            if len(labels_list) >= max_samples:
                break

    z_H_arr = np.concatenate(z_H_list, axis=0)[:max_samples]
    labels = np.array(labels_list[:max_samples], dtype=np.int64)
    n_steps_max = max(step_counts) if step_counts else 0
    return z_H_arr, labels, n_steps_max


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def _probe_auc(
    z_H: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
    label: str = "",
) -> Dict[str, Any]:
    from warmstart_probe import compute_workflow_type_auc
    result = compute_workflow_type_auc(z_H, labels, seed=seed)
    if label:
        logger.info(
            "%s AUC=%.4f  accuracy=%.4f  classes=%d  n_train=%d  n_test=%d",
            label, result["auc"], result["accuracy"],
            result["n_classes_present_in_test"], result["n_train"], result["n_test"],
        )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shuffled-primitive probe (ACT-iterated): substrate uses structure, not just primitive identity"
    )
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--dataset", required=True, help="Dataset root directory")
    p.add_argument("--split", default="val", help="Which split to evaluate (default: val)")
    p.add_argument("--config", required=True, help="TrainingConfig YAML")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--max_samples", type=int, default=1000,
                   help="Max DAGs to evaluate (default: 1000)")
    p.add_argument("--shuffle_seed", type=int, default=42,
                   help="RNG seed for primitive shuffling (default: 42)")
    p.add_argument("--device", default="cpu", help="Device: cpu or cuda (default: cpu)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from marifah.training.config import load_config
    from marifah.training.data_pipeline import build_data_loaders

    config = load_config(args.config)
    device = torch.device(args.device)

    logger.info("Building ACT model from %s", args.config)
    act_model = _build_act_model(config, device)

    logger.info("Loading checkpoint into act_model.inner: %s", args.checkpoint)
    _load_checkpoint_into_act(act_model, args.checkpoint, device)

    config.data.dataset_root = args.dataset

    # ---- Un-shuffled pass (normal ACT forward) ----------------------------------
    logger.info("Un-shuffled pass: extracting ACT-iterated z_H with normal inputs ...")
    loaders = build_data_loaders(config)
    loader = loaders.get(args.split)
    if loader is None:
        logger.error("Split '%s' not found — aborting.", args.split)
        sys.exit(1)

    z_H_normal, labels_normal, n_steps_unshuffled = _extract_z_H_pooled_act(
        act_model, loader, config, device, args.max_samples, shuffle_primitives=False
    )
    n = len(labels_normal)
    logger.info("Un-shuffled: %d samples  z_H shape=%s  n_unique_labels=%d  n_act_steps=%d",
                n, z_H_normal.shape, len(set(labels_normal.tolist())), n_steps_unshuffled)

    res_unshuffled = _probe_auc(z_H_normal, labels_normal, label="[ACT unshuffled]")

    # ---- Shuffled pass -------------------------------------------------------
    logger.info("Shuffled pass: extracting ACT-iterated z_H with primitives shuffled within DAGs ...")
    loaders2 = build_data_loaders(config)
    loader2 = loaders2.get(args.split)
    if loader2 is None:
        logger.error("Split '%s' not found for shuffled pass — aborting.", args.split)
        sys.exit(1)

    z_H_shuffled, labels_shuffled, n_steps_shuffled = _extract_z_H_pooled_act(
        act_model, loader2, config, device, args.max_samples,
        shuffle_primitives=True, shuffle_seed=args.shuffle_seed,
    )
    logger.info("Shuffled: %d samples  z_H shape=%s  n_act_steps=%d",
                len(labels_shuffled), z_H_shuffled.shape, n_steps_shuffled)

    res_shuffled = _probe_auc(z_H_shuffled, labels_shuffled, label="[ACT shuffled]")

    auc_drop = res_unshuffled["auc"] - res_shuffled["auc"]

    results: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "split": args.split,
        "config": str(args.config),
        "max_samples": args.max_samples,
        "n_extracted": n,
        "device": args.device,
        "shuffle_seed": args.shuffle_seed,
        "n_act_steps_unshuffled": n_steps_unshuffled,
        "n_act_steps_shuffled": n_steps_shuffled,
        "unshuffled": res_unshuffled,
        "shuffled": res_shuffled,
        "auc_drop": round(auc_drop, 6),
        "verdict_inputs": {
            "auc_shuffled": res_shuffled["auc"],
            "auc_unshuffled": res_unshuffled["auc"],
            "auc_drop": round(auc_drop, 6),
            "path_a_threshold": 0.65,
            "path_a_met": res_shuffled["auc"] >= 0.65,
            "path_b_threshold": 0.55,
            "note": (
                "ACT-iterated variant. Path A requires auc_shuffled >= 0.65 (pre-reg amendment 2026-05-17). "
                "Path B: auc_shuffled < 0.55. "
                "Large auc_drop means substrate reads primitive identity, not structure."
            ),
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results written to %s", args.output)
    logger.info("=== SUMMARY ===")
    logger.info("  ACT steps used (both passes): %d", n_steps_unshuffled)
    logger.info("  AUC (ACT unshuffled inputs): %.4f", res_unshuffled["auc"])
    logger.info("  AUC (ACT shuffled inputs):   %.4f  [Path A requires >= 0.65]",
                res_shuffled["auc"])
    logger.info("  AUC drop (primitive leak):   %.4f", auc_drop)


if __name__ == "__main__":
    main()
