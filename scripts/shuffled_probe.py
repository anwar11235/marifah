"""Shuffled-primitive probe: tests whether substrate uses graph structure, not just primitive identity.

Runs the model forward pass twice on each batch:
  1. Normal inputs — extracts z_H, probes workflow_type_id (auc_unshuffled)
  2. Shuffled inputs — primitive_assignments[b,:] permuted within each DAG, node_features[b,:,0]
     permuted in lockstep — extracts z_H, probes workflow_type_id (auc_shuffled)

The shuffle preserves the multiset of primitives per DAG but destroys which-primitive-is-at-
which-node, eliminating position-specific primitive identity while keeping graph topology intact.

Interpretation:
  auc_shuffled ≈ 0.50 → substrate reads workflow type purely from primitive identity (leakage)
  auc_shuffled >= 0.55 → substrate uses graph structure beyond primitive identity
  auc_shuffled >= 0.65 → strong evidence substrate uses structural organization (Path A criterion)

Path A criterion: auc_shuffled >= 0.65 (pre-registered amendment 2026-05-17)

Usage:
    python scripts/shuffled_probe.py \\
        --checkpoint checkpoints/warmstart_cold/final.pt \\
        --dataset /workspace/data/marifah_graph_v1 \\
        --split val \\
        --config configs/warmstart_cold.yaml \\
        --output results/shuffled_probe_cold.json \\
        [--max_samples 1000] \\
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
# Shuffling utility
# ---------------------------------------------------------------------------


def _shuffle_batch_primitives(
    batch: Any,
    device: torch.device,
    rng: np.random.RandomState,
) -> Any:
    """Return a copy of batch with primitives shuffled within each DAG.

    For each item b in the batch:
      - Generate a random permutation of real (non-padded) node positions
      - Apply it to node_features[b, :, 0] (primitive id as first feature component)
      - Apply the same permutation to primitive_assignments[b, :]
      - All other fields (graph structure, attention_mask, node_mask, pos_encoding,
        all other node_feature columns) are left unchanged

    This preserves the multiset of primitives per DAG but destroys position-specific
    primitive identity while keeping graph topology intact.
    """
    import copy

    # Clone the tensors we need to modify
    node_features = batch.node_features.clone()  # (B, max_nodes, feat_dim)
    primitive_assignments = batch.primitive_assignments.clone()  # (B, max_nodes)
    node_mask = batch.node_mask  # (B, max_nodes) bool

    B = node_features.shape[0]
    for b in range(B):
        real_indices = node_mask[b].nonzero(as_tuple=True)[0].cpu().numpy()
        if len(real_indices) < 2:
            continue
        perm = rng.permutation(len(real_indices))
        perm_indices = real_indices[perm]

        # Shuffle primitive id (first feature component)
        orig_prim_feat = node_features[b, real_indices, 0].clone()
        node_features[b, real_indices, 0] = orig_prim_feat[perm]

        # Shuffle primitive_assignments in lockstep
        orig_prim_assign = primitive_assignments[b, real_indices].clone()
        primitive_assignments[b, real_indices] = orig_prim_assign[perm]

    from marifah.data.adapter.batch_format import GraphBatch
    return GraphBatch(
        node_features=node_features,
        attention_mask=batch.attention_mask,
        node_mask=batch.node_mask,
        pos_encoding=batch.pos_encoding,
        workflow_type_id=batch.workflow_type_id,
        region_assignments=batch.region_assignments,
        primitive_assignments=primitive_assignments,
        halt_step=batch.halt_step,
        execution_trace=batch.execution_trace,
        cycle_annotations=batch.cycle_annotations,
    )


# ---------------------------------------------------------------------------
# Carry extraction
# ---------------------------------------------------------------------------


def _extract_z_H_pooled(
    model: torch.nn.Module,
    data_loader: Any,
    config: Any,
    device: torch.device,
    max_samples: int,
    shuffle_primitives: bool = False,
    shuffle_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model forward; return (pooled_z_H, labels).

    Args:
        shuffle_primitives: If True, shuffle primitives within each DAG before forward pass.
        shuffle_seed:       Seed for shuffle RNG (for reproducibility).
    """
    from marifah.training.graph_utils import prepare_batch_for_model
    from marifah.models.coral_base import InnerCarry

    rng = np.random.RandomState(shuffle_seed)
    model.eval()
    z_H_list: List[np.ndarray] = []
    labels_list: List[int] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            if shuffle_primitives:
                batch = _shuffle_batch_primitives(batch, device, rng)

            B = batch.batch_size
            carry = InnerCarry.zeros(B, config.model, device)
            coral_batch = prepare_batch_for_model(batch, config, device)
            result = model(carry, coral_batch, is_last_segment=True)

            out_carry = result[0]
            z_H = out_carry.z_H  # (B, max_nodes, d_model)

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
    return z_H_arr, labels


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
        description="Shuffled-primitive probe: substrate uses structure, not just primitive identity"
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
    from marifah.training.trainer import build_model
    from marifah.training.checkpointing import load_warmstart
    from marifah.training.data_pipeline import build_data_loaders

    config = load_config(args.config)
    device = torch.device(args.device)

    logger.info("Building model from %s", args.config)
    model = build_model(config, device)

    logger.info("Loading checkpoint: %s", args.checkpoint)
    load_warmstart(args.checkpoint, model)
    model.eval()

    config.data.dataset_root = args.dataset

    # ---- Un-shuffled pass (normal forward) --------------------------------
    logger.info("Un-shuffled pass: extracting z_H with normal inputs ...")
    loaders = build_data_loaders(config)
    loader = loaders.get(args.split)
    if loader is None:
        logger.error("Split '%s' not found — aborting.", args.split)
        sys.exit(1)

    z_H_normal, labels_normal = _extract_z_H_pooled(
        model, loader, config, device, args.max_samples, shuffle_primitives=False
    )
    n = len(labels_normal)
    logger.info("Un-shuffled: %d samples  z_H shape=%s  n_unique_labels=%d",
                n, z_H_normal.shape, len(set(labels_normal.tolist())))

    res_unshuffled = _probe_auc(z_H_normal, labels_normal, label="[unshuffled]")

    # ---- Shuffled pass (primitives randomized within each DAG) ------------
    logger.info("Shuffled pass: extracting z_H with primitives shuffled within DAGs ...")
    loaders2 = build_data_loaders(config)
    loader2 = loaders2.get(args.split)
    if loader2 is None:
        logger.error("Split '%s' not found for shuffled pass — aborting.", args.split)
        sys.exit(1)

    z_H_shuffled, labels_shuffled = _extract_z_H_pooled(
        model, loader2, config, device, args.max_samples,
        shuffle_primitives=True, shuffle_seed=args.shuffle_seed,
    )
    logger.info("Shuffled: %d samples  z_H shape=%s", len(labels_shuffled), z_H_shuffled.shape)

    res_shuffled = _probe_auc(z_H_shuffled, labels_shuffled, label="[shuffled]")

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
                "Path A requires auc_shuffled >= 0.65 (pre-reg amendment 2026-05-17). "
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
    logger.info("  AUC (unshuffled inputs): %.4f", res_unshuffled["auc"])
    logger.info("  AUC (shuffled inputs):   %.4f  [Path A requires >= 0.65]",
                res_shuffled["auc"])
    logger.info("  AUC drop (primitive leak estimate): %.4f", auc_drop)


if __name__ == "__main__":
    main()
