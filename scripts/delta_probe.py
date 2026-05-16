"""Δ-probe: measures information added by the substrate beyond raw inputs.

Trains two linear probes on the same stratified split:
  B1 (baseline): mean-pooled node_features → workflow_type_id
  B2 (substrate): mean-pooled z_H          → workflow_type_id

Δ = AUC(B2) - AUC(B1)

Interpretation:
  Δ ≈ 0   → substrate adds nothing beyond raw inputs; not doing useful work
  Δ > 0.10 → substrate adds meaningful information; organizing beyond inputs
  Δ < 0   → substrate destroying information (degenerate training)

Path A criterion: Δ ≥ 0.10 (pre-registered amendment 2026-05-17)

Usage:
    python scripts/delta_probe.py \\
        --checkpoint checkpoints/warmstart_cold/final.pt \\
        --dataset /workspace/data/marifah_graph_v1 \\
        --split val \\
        --config configs/warmstart_cold.yaml \\
        --output results/delta_probe_cold.json \\
        [--max_samples 1000] \\
        [--bootstrap 100] \\
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


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _extract_features_and_labels(
    model: torch.nn.Module,
    data_loader: Any,
    config: Any,
    device: torch.device,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (node_features_pooled, z_H_pooled, labels) in one forward pass.

    Returns:
        node_feats : float32 array (N, node_feat_dim) — mean-pooled raw inputs
        z_H_states : float32 array (N, d_model)       — mean-pooled substrate carry
        labels     : int64 array (N,)                  — workflow_type_id (0-indexed)
    """
    from marifah.training.graph_utils import prepare_batch_for_model
    from marifah.models.coral_base import InnerCarry

    model.eval()
    node_feats_list: List[np.ndarray] = []
    z_H_list: List[np.ndarray] = []
    labels_list: List[int] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            B = batch.batch_size

            carry = InnerCarry.zeros(B, config.model, device)
            coral_batch = prepare_batch_for_model(batch, config, device)
            result = model(carry, coral_batch, is_last_segment=True)

            out_carry = result[0]
            z_H = out_carry.z_H  # (B, max_nodes, d_model)

            node_mask = batch.node_mask.to(device)  # (B, max_nodes) bool
            mask_f = node_mask.unsqueeze(-1).float()
            node_counts = mask_f.sum(dim=1).clamp(min=1)

            # Pool z_H
            pooled_z = (z_H.float() * mask_f).sum(dim=1) / node_counts  # (B, d_model)
            z_H_list.append(pooled_z.cpu().numpy())

            # Pool node_features (raw inputs before embedding)
            raw_feats = batch.node_features.to(device).float()  # (B, max_nodes, feat_dim)
            raw_mask = node_mask.unsqueeze(-1).float()
            raw_counts = raw_mask.sum(dim=1).clamp(min=1)
            pooled_raw = (raw_feats * raw_mask).sum(dim=1) / raw_counts  # (B, feat_dim)
            node_feats_list.append(pooled_raw.cpu().numpy())

            wf_ids = (batch.workflow_type_id - 1).clamp(min=0).cpu().numpy()
            labels_list.extend(wf_ids.tolist())

            if len(labels_list) >= max_samples:
                break

    node_feats = np.concatenate(node_feats_list, axis=0)[:max_samples]
    z_H_arr = np.concatenate(z_H_list, axis=0)[:max_samples]
    labels = np.array(labels_list[:max_samples], dtype=np.int64)
    return node_feats, z_H_arr, labels


# ---------------------------------------------------------------------------
# Probe: linear classifier with stratified split
# ---------------------------------------------------------------------------

# Import compute_workflow_type_auc from warmstart_probe for consistency
sys.path.insert(0, str(Path(__file__).parent))


def _probe_auc(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
    test_fraction: float = 0.3,
) -> Dict[str, Any]:
    """Train LogisticRegression probe; return AUC dict compatible with warmstart_probe."""
    from warmstart_probe import compute_workflow_type_auc
    return compute_workflow_type_auc(features, labels, test_fraction=test_fraction, seed=seed)


def _bootstrap_delta(
    node_feats: np.ndarray,
    z_H: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int,
    base_seed: int = 0,
) -> Optional[Dict[str, float]]:
    """Bootstrap the Δ distribution (resample carry+input features together).

    Returns dict with mean, std, ci_low, ci_high at 95% CI, or None if < 10 succeed.
    """
    rng = np.random.RandomState(base_seed)
    n = len(labels)
    deltas: List[float] = []

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            res_baseline = _probe_auc(node_feats[idx], labels[idx], seed=base_seed + i)
            res_substrate = _probe_auc(z_H[idx], labels[idx], seed=base_seed + i)
            deltas.append(res_substrate["auc"] - res_baseline["auc"])
        except (RuntimeError, ValueError):
            continue

    if len(deltas) < 10:
        logger.warning("Bootstrap delta: only %d/%d succeeded — returning None.",
                       len(deltas), n_bootstrap)
        return None

    arr = np.array(deltas)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "n_successful": len(deltas),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Δ-probe: substrate AUC minus baseline AUC")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--dataset", required=True, help="Dataset root directory")
    p.add_argument("--split", default="val", help="Which split to evaluate (default: val)")
    p.add_argument("--config", required=True, help="TrainingConfig YAML")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--max_samples", type=int, default=1000,
                   help="Max DAGs to evaluate (default: 1000)")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="Bootstrap iterations for 95%% CI on Δ (0 = skip)")
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
    loaders = build_data_loaders(config)
    loader = loaders.get(args.split)
    if loader is None:
        logger.error("Split '%s' not found — aborting.", args.split)
        sys.exit(1)

    logger.info("Extracting features: split=%s  max_samples=%d  device=%s",
                args.split, args.max_samples, args.device)
    node_feats, z_H, labels = _extract_features_and_labels(
        model, loader, config, device, args.max_samples
    )
    n = len(labels)
    logger.info("Extracted %d samples — node_feats shape=%s  z_H shape=%s  n_unique_labels=%d",
                n, node_feats.shape, z_H.shape, len(set(labels.tolist())))

    assert len(z_H) == len(node_feats) == len(labels), "Shape mismatch after extraction"

    logger.info("Probing node_features (baseline B1) ...")
    res_baseline = _probe_auc(node_feats, labels)

    logger.info("Probing z_H (substrate B2) ...")
    res_substrate = _probe_auc(z_H, labels)

    delta = res_substrate["auc"] - res_baseline["auc"]

    bootstrap_results = None
    if args.bootstrap > 0:
        logger.info("Bootstrapping Δ (%d iterations) ...", args.bootstrap)
        bootstrap_results = _bootstrap_delta(node_feats, z_H, labels,
                                             n_bootstrap=args.bootstrap)
        if bootstrap_results:
            logger.info("Δ bootstrap: mean=%.4f  std=%.4f  95%% CI=[%.4f, %.4f]",
                        bootstrap_results["mean"], bootstrap_results["std"],
                        bootstrap_results["ci_low"], bootstrap_results["ci_high"])

    results: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "split": args.split,
        "config": str(args.config),
        "max_samples": args.max_samples,
        "n_extracted": n,
        "device": args.device,
        "baseline": res_baseline,
        "substrate": res_substrate,
        "delta": round(delta, 6),
        "verdict_inputs": {
            "delta": round(delta, 6),
            "auc_baseline": res_baseline["auc"],
            "auc_substrate": res_substrate["auc"],
            "path_a_threshold": 0.10,
            "path_a_met": delta >= 0.10,
            "note": "Path A requires delta >= 0.10 (pre-reg amendment 2026-05-17)",
        },
    }
    if bootstrap_results:
        results["delta_bootstrap"] = bootstrap_results

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results written to %s", args.output)
    logger.info("=== SUMMARY ===")
    logger.info("  AUC (baseline node_features): %.4f", res_baseline["auc"])
    logger.info("  AUC (substrate z_H):          %.4f", res_substrate["auc"])
    logger.info("  Δ = substrate - baseline:     %.4f  [Path A requires >= 0.10]",
                delta)
    if bootstrap_results:
        logger.info("  Δ 95%% CI:                    [%.4f, %.4f]",
                    bootstrap_results["ci_low"], bootstrap_results["ci_high"])


if __name__ == "__main__":
    main()
