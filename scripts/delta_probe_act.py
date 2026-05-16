"""Δ-probe with ACT-iterated z_H: measures substrate information beyond inputs after full ACT.

Identical to delta_probe.py except z_H is extracted after running CoralV3ACT for
halt_max_steps iterations (4 in eval mode) instead of a single CoralV3Inner pass.

Comparison with delta_probe.py reveals whether:
  - ACT-iterated delta > 1-step delta: substrate can organize representations at 4-step
    inference even though training used only 1-step gradient. Training is the bottleneck.
  - ACT-iterated delta ≈ 1-step delta: ACT iteration does not help; substrate cannot
    organize from this task/training combination. Path B: deeper redesign needed.

Path A criterion (same as delta_probe.py): delta >= 0.10

Usage:
    python scripts/delta_probe_act.py \\
        --checkpoint checkpoints/warmstart_cold/final.pt \\
        --dataset /workspace/data/marifah_graph_v1 \\
        --split val \\
        --config configs/warmstart_cold.yaml \\
        --output results/delta_probe_act_cold.json \\
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

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# ACT model builder + checkpoint loader
# ---------------------------------------------------------------------------


def _build_act_model(config: Any, device: torch.device) -> Any:
    """Build CoralV3ACT wrapping a CoralV3Inner with the same architecture as the trainer."""
    from marifah.models.coral_base import CoralConfig
    from marifah.models.act import CoralV3ACT

    coral_cfg = CoralConfig(
        batch_size=config.training.batch_size,
        seq_len=config.model.max_nodes,
        vocab_size=config.model.vocab_size,
        H_cycles=config.model.H_cycles,
        L_cycles=config.model.L_cycles,
        H_layers=config.model.H_layers,
        L_layers=config.model.L_layers,
        hidden_size=config.model.d_model,
        num_heads=config.model.num_heads,
        use_predictive_coding=True,
        use_hmsc=False,
        forward_dtype=config.model.forward_dtype,
        halt_max_steps=config.model.halt_max_steps,
        halt_exploration_prob=config.model.halt_exploration_prob,
    )
    return CoralV3ACT(coral_cfg).to(device)


def _load_checkpoint_into_act(
    act_model: Any,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """Load a trainer-saved CoralV3Inner checkpoint into act_model.inner.

    The trainer saves flat CoralV3Inner state_dicts. If the checkpoint was saved
    from an ACT-wrapped model (keys prefixed with 'inner.'), that prefix is stripped
    before loading.

    After loading, verifies that at least one substrate weight changed.
    """
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict", payload)

    # Strip 'inner.' prefix if checkpoint came from an ACT-wrapped model
    if state_dict and all(k.startswith("inner.") for k in state_dict):
        logger.info("Stripping 'inner.' prefix from checkpoint keys (ACT-wrapped save)")
        state_dict = {k[len("inner."):]: v for k, v in state_dict.items()}

    # Snapshot a substrate weight for change-detection
    substrate_key = next(
        (k for k in act_model.inner.state_dict() if "H_level" in k and k.endswith(".weight")),
        None,
    )
    pre_sum: Optional[float] = None
    if substrate_key is not None:
        pre_sum = float(act_model.inner.state_dict()[substrate_key].sum())

    incompat = act_model.inner.load_state_dict(state_dict, strict=False)
    if incompat.missing_keys:
        logger.warning("Missing keys in checkpoint: first 5: %s", incompat.missing_keys[:5])
    if incompat.unexpected_keys:
        logger.warning("Unexpected keys in checkpoint: first 5: %s", incompat.unexpected_keys[:5])

    # Verify at least one substrate parameter actually changed
    if substrate_key is not None and pre_sum is not None:
        post_sum = float(act_model.inner.state_dict()[substrate_key].sum())
        if pre_sum == post_sum:
            raise RuntimeError(
                f"ACT inner load: substrate parameter '{substrate_key}' unchanged after load. "
                "Probable key-naming mismatch. All checkpoint keys may have been dropped."
            )
        logger.info("ACT inner load OK — substrate weight changed (%.4f → %.4f)", pre_sum, post_sum)

    act_model.to(device)
    act_model.eval()


# ---------------------------------------------------------------------------
# Feature extraction with ACT iteration
# ---------------------------------------------------------------------------


def _extract_features_and_labels_act(
    act_model: Any,
    data_loader: Any,
    config: Any,
    device: torch.device,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, Dict[int, int]]:
    """Run ACT-iterated forward; return (node_features_projected, z_H_pooled, labels, n_steps, step_hist).

    In eval mode, CoralV3ACT always runs halt_max_steps iterations (halted = is_last_step only).
    Returns:
        node_feats      : float32 (N, d_model) — dim-matched projected raw inputs
        z_H_states      : float32 (N, d_model) — mean-pooled ACT-iterated carry
        labels          : int64 (N,)            — workflow_type_id (0-indexed)
        n_steps_used    : int — actual steps run (should equal halt_max_steps in eval)
        step_hist       : {n_steps: count} across all batches
    """
    from marifah.training.graph_utils import prepare_batch_for_model

    node_feats_list: List[np.ndarray] = []
    z_H_list: List[np.ndarray] = []
    labels_list: List[int] = []
    step_counts: List[int] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            coral_batch = prepare_batch_for_model(batch, config, device)

            # Initialize ACT carry (halted=True → all sequences reset on first call)
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

            B = batch.batch_size
            node_mask = batch.node_mask.to(device)
            mask_f = node_mask.unsqueeze(-1).float()
            node_counts = mask_f.sum(dim=1).clamp(min=1)

            pooled_z = (z_H.float() * mask_f).sum(dim=1) / node_counts  # (B, d_model)
            z_H_list.append(pooled_z.cpu().numpy())

            # Pool raw node_features for dim-matched baseline
            raw_feats = batch.node_features.to(device).float()
            raw_mask = node_mask.unsqueeze(-1).float()
            raw_counts = raw_mask.sum(dim=1).clamp(min=1)
            pooled_raw = (raw_feats * raw_mask).sum(dim=1) / raw_counts  # (B, feat_dim)
            node_feats_list.append(pooled_raw.cpu().numpy())

            wf_ids = (batch.workflow_type_id - 1).clamp(min=0).cpu().numpy()
            labels_list.extend(wf_ids.tolist())

            if len(labels_list) >= max_samples:
                break

    node_feats_raw = np.concatenate(node_feats_list, axis=0)[:max_samples]
    z_H_arr = np.concatenate(z_H_list, axis=0)[:max_samples]
    labels = np.array(labels_list[:max_samples], dtype=np.int64)

    # Project node_features to match z_H dimensionality (removes dim-inflation artifact)
    raw_dim = node_feats_raw.shape[-1]
    d_model = z_H_arr.shape[-1]
    if raw_dim != d_model:
        rng_proj = torch.Generator()
        rng_proj.manual_seed(0)  # FIXED projection seed; never randomize this
        proj = torch.randn(raw_dim, d_model, generator=rng_proj)
        proj = proj / proj.norm(dim=0, keepdim=True)  # unit-norm columns
        node_feats = (torch.from_numpy(node_feats_raw) @ proj).numpy()
    else:
        node_feats = node_feats_raw

    n_steps_max = max(step_counts) if step_counts else 0
    step_hist: Dict[int, int] = {}
    for s in step_counts:
        step_hist[s] = step_hist.get(s, 0) + 1

    return node_feats, z_H_arr, labels, n_steps_max, step_hist


# ---------------------------------------------------------------------------
# Probe + bootstrap (delegate to delta_probe.py functions)
# ---------------------------------------------------------------------------


def _probe_auc(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
    test_fraction: float = 0.3,
) -> Dict[str, Any]:
    from warmstart_probe import compute_workflow_type_auc
    return compute_workflow_type_auc(features, labels, test_fraction=test_fraction, seed=seed)


def _bootstrap_delta(
    node_feats: np.ndarray,
    z_H: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int,
    base_seed: int = 0,
) -> Optional[Dict[str, float]]:
    from delta_probe import _bootstrap_delta as _bd
    return _bd(node_feats, z_H, labels, n_bootstrap=n_bootstrap, base_seed=base_seed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Δ-probe (ACT-iterated): substrate AUC minus dim-matched baseline AUC")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--dataset", required=True, help="Dataset root directory")
    p.add_argument("--split", default="val", help="Which split to evaluate (default: val)")
    p.add_argument("--config", required=True, help="TrainingConfig YAML")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--max_samples", type=int, default=1000,
                   help="Max DAGs to evaluate (default: 1000)")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="Bootstrap iterations for 95%% CI on delta (0 = skip)")
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
    loaders = build_data_loaders(config)
    loader = loaders.get(args.split)
    if loader is None:
        logger.error("Split '%s' not found — aborting.", args.split)
        sys.exit(1)

    logger.info("Extracting ACT-iterated features: split=%s  max_samples=%d  device=%s",
                args.split, args.max_samples, args.device)
    node_feats, z_H, labels, n_steps_used, step_hist = _extract_features_and_labels_act(
        act_model, loader, config, device, args.max_samples
    )
    n = len(labels)
    logger.info("Extracted %d samples — node_feats shape=%s  z_H shape=%s  "
                "n_unique_labels=%d  n_act_steps=%d",
                n, node_feats.shape, z_H.shape, len(set(labels.tolist())), n_steps_used)

    assert len(z_H) == len(node_feats) == len(labels), "Shape mismatch after extraction"

    logger.info("Probing dim-matched node_features (baseline B1) ...")
    res_baseline = _probe_auc(node_feats, labels)

    logger.info("Probing ACT-iterated z_H (substrate B2) ...")
    res_substrate = _probe_auc(z_H, labels)

    delta = res_substrate["auc"] - res_baseline["auc"]

    bootstrap_results = None
    if args.bootstrap > 0:
        logger.info("Bootstrapping delta (%d iterations) ...", args.bootstrap)
        bootstrap_results = _bootstrap_delta(node_feats, z_H, labels, n_bootstrap=args.bootstrap)
        if bootstrap_results:
            logger.info("delta bootstrap: mean=%.4f  std=%.4f  95%% CI=[%.4f, %.4f]",
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
        "n_act_steps_used": n_steps_used,
        "act_halt_distribution": {str(k): v for k, v in step_hist.items()},
        "baseline": res_baseline,
        "substrate": res_substrate,
        "delta": round(delta, 6),
        "verdict_inputs": {
            "delta": round(delta, 6),
            "auc_baseline": res_baseline["auc"],
            "auc_substrate": res_substrate["auc"],
            "path_a_threshold": 0.10,
            "path_a_met": delta >= 0.10,
            "note": (
                "ACT-iterated variant. Path A requires delta >= 0.10 (pre-reg amendment 2026-05-17). "
                "Baseline uses dim-matched random projection of node_features to eliminate "
                "dimensional-inflation artifact (Amendment 2)."
            ),
        },
    }
    if bootstrap_results:
        results["delta_bootstrap"] = bootstrap_results

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results written to %s", args.output)
    logger.info("=== SUMMARY ===")
    logger.info("  ACT steps used:               %d / %d", n_steps_used, config.model.halt_max_steps)
    logger.info("  AUC (dim-matched node_feats): %.4f", res_baseline["auc"])
    logger.info("  AUC (ACT-iterated z_H):       %.4f", res_substrate["auc"])
    logger.info("  delta = substrate - baseline:  %.4f  [Path A requires >= 0.10]", delta)
    if bootstrap_results:
        logger.info("  delta 95%% CI:                 [%.4f, %.4f]",
                    bootstrap_results["ci_low"], bootstrap_results["ci_high"])


if __name__ == "__main__":
    main()
