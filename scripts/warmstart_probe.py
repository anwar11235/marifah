"""Warm-start probe: workflow-type AUC and execution faithfulness on a checkpoint.

Measures two metrics on a given checkpoint to resolve OD7
(Sudoku Phase 3c warm-start vs. from-scratch comparison):

  1. workflow_type_auc  — linear-probe AUC on pooled carry state → workflow_type_id
  2. execution_faithfulness — mean per-step edit distance vs. ground-truth trace
                              (lower is better; 0.0 = perfect)

Usage:
    python scripts/warmstart_probe.py \\
        --checkpoint checkpoints/warmstart_cold/final.pt \\
        --dataset /workspace/data/marifah_full_dataset \\
        --split val \\
        --config configs/warmstart_cold.yaml \\
        --output results/cold_results.json \\
        [--max_samples 1000] \\
        [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Carry construction helper
# ---------------------------------------------------------------------------


def _make_carry(B: int, config: Any, device: torch.device) -> Any:
    """Construct a zero carry with correct dtype; falls back to float32 on CPU when bf16 configured.

    bf16 is not fully supported on all CPU paths (flash-attn requires CUDA).
    When --device cpu is used with forward_dtype=bfloat16, we warn and use float32.
    """
    from marifah.models.coral_base import InnerCarry

    dtype_override = None
    if device.type == "cpu" and getattr(config.model, "forward_dtype", "float32") == "bfloat16":
        logger.warning(
            "CPU device with forward_dtype=bfloat16: using float32 for carry "
            "(bf16 not fully supported on all CPU paths — GPU required for flash-attn)"
        )
        dtype_override = torch.float32
    return InnerCarry.zeros(B, config.model, device, dtype_override=dtype_override)


# ---------------------------------------------------------------------------
# Carry-state extraction
# ---------------------------------------------------------------------------


def _extract_carry_states(
    model: torch.nn.Module,
    data_loader: Any,
    config: Any,
    device: torch.device,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model in eval mode; collect (pooled carry state, workflow_type_id) pairs.

    Returns:
        carry_states : float32 array of shape (N, d_model)
        labels       : int array of shape (N,) — workflow_type_id (0-indexed)
    """
    from marifah.training.graph_utils import prepare_batch_for_model

    model.eval()
    carry_states: List[np.ndarray] = []
    labels: List[int] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            B = batch.batch_size

            carry = _make_carry(B, config, device)

            coral_batch = prepare_batch_for_model(batch, config, device)
            result = model(carry, coral_batch, is_last_segment=True)

            # result[0] is the output InnerCarry with z_H being the final carry state.
            out_carry = result[0]
            z_H = out_carry.z_H  # (B, max_nodes, d_model)

            # Mask out padding positions before pooling.
            node_mask = batch.node_mask.to(device)  # (B, max_nodes) bool
            mask_f = node_mask.unsqueeze(-1).float()  # (B, max_nodes, 1)
            node_counts = mask_f.sum(dim=1).clamp(min=1)  # (B, 1)
            pooled = (z_H * mask_f).sum(dim=1) / node_counts  # (B, d_model)

            carry_states.append(pooled.cpu().float().numpy())

            # workflow_type_id is 1-indexed in the dataset; convert to 0-indexed.
            wf_ids = (batch.workflow_type_id - 1).clamp(min=0).cpu().numpy()
            labels.extend(wf_ids.tolist())

            if len(labels) >= max_samples:
                break

    carry_arr = np.concatenate(carry_states, axis=0)[:max_samples]
    label_arr = np.array(labels[:max_samples], dtype=np.int64)
    return carry_arr, label_arr


# ---------------------------------------------------------------------------
# Workflow-type AUC probe
# ---------------------------------------------------------------------------


def compute_workflow_type_auc(
    carry_states: np.ndarray,
    labels: np.ndarray,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> Dict[str, float]:
    """Train a linear probe on carry states; report multi-class OvR AUC.

    Uses sklearn LogisticRegression with a stratified train/test split.
    AUC failure raises RuntimeError — no silent fallback to accuracy.

    Returns dict with:
        auc                             — macro-averaged OvR AUC (gating metric §7.1)
        accuracy                        — probe accuracy on held-out test set
        n_classes                       — distinct classes present in test set
        n_classes_present_in_test       — same (alias for clarity in reports)
        n_classes_filtered_single_sample — classes dropped for having only 1 sample
        n_train                         — training set size
        n_test                          — test set size
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    unique_classes, class_counts = np.unique(labels, return_counts=True)

    # Stratification requires >= 2 samples per class; filter singletons first.
    if class_counts.min() < 2:
        n_single = int((class_counts < 2).sum())
        logger.warning(
            "Stratified split impossible: %d class(es) have only 1 sample. "
            "Filtering them out. AUC computed on remaining classes only.",
            n_single,
        )
        valid_classes = unique_classes[class_counts >= 2]
        valid_mask = np.isin(labels, valid_classes)
        if valid_mask.sum() < 10:
            raise RuntimeError(
                f"Too few samples after filtering single-instance classes: {valid_mask.sum()}"
            )
        labels_filtered = labels[valid_mask]
        carry_states_filtered = carry_states[valid_mask]
        logger.info(
            "Filtered to %d valid samples across %d classes (was %d / %d).",
            valid_mask.sum(), len(valid_classes), len(labels), len(unique_classes),
        )
        n_classes_filtered = n_single
    else:
        labels_filtered = labels
        carry_states_filtered = carry_states
        n_classes_filtered = 0

    X_train, X_test, y_train, y_test = train_test_split(
        carry_states_filtered, labels_filtered,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels_filtered,
    )
    n_train = len(y_train)
    n_test = len(y_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=500, C=1.0, random_state=seed)
    clf.fit(X_train_s, y_train)

    assert clf.classes_.shape[0] >= 2, (
        f"Classifier trained on only {clf.classes_.shape[0]} class(es) — AUC undefined."
    )

    y_prob = clf.predict_proba(X_test_s)
    y_pred = clf.predict(X_test_s)

    classes_in_test = np.unique(y_test)
    n_unique_in_test = len(classes_in_test)

    if n_unique_in_test < 2:
        raise RuntimeError(
            f"Only {n_unique_in_test} class(es) in y_test after stratified split. "
            f"AUC undefined. Check label distribution."
        )

    # Restrict probabilities to classes present in test to avoid OvR NaN.
    class_mask = np.isin(clf.classes_, classes_in_test)
    y_prob_restricted = y_prob[:, class_mask]
    classes_restricted = clf.classes_[class_mask]

    try:
        auc = float(roc_auc_score(
            y_test, y_prob_restricted,
            multi_class="ovr",
            average="macro",
            labels=classes_restricted,
        ))
        if np.isnan(auc):
            raise RuntimeError("roc_auc_score returned NaN even with stratified split")
    except ValueError as exc:
        raise RuntimeError(f"AUC computation failed: {exc}") from exc

    accuracy = float((y_pred == y_test).mean())

    logger.info(
        "Workflow-type AUC=%.4f  accuracy=%.4f  classes=%d  n_train=%d  n_test=%d",
        auc, accuracy, n_unique_in_test, n_train, n_test,
    )
    return {
        "auc": auc,
        "accuracy": accuracy,
        "n_classes": int(n_unique_in_test),
        "n_classes_present_in_test": int(n_unique_in_test),
        "n_classes_filtered_single_sample": n_classes_filtered,
        "n_train": int(n_train),
        "n_test": int(n_test),
    }


def _bootstrap_auc(
    carry_states: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 100,
    base_seed: int = 0,
) -> Optional[Dict[str, float]]:
    """Bootstrap AUC by resampling carry/label pairs with replacement.

    Returns dict with mean, std, ci_low, ci_high at 95% CI, plus n_successful.
    Returns None if fewer than 10 bootstrap iterations succeeded.

    Note: carry_states and labels must already be extracted — this function
    does NOT re-run model inference.
    """
    rng = np.random.RandomState(base_seed)
    n = len(labels)
    aucs: List[float] = []
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            result = compute_workflow_type_auc(
                carry_states[idx], labels[idx],
                test_fraction=0.3, seed=base_seed + i,
            )
            aucs.append(result["auc"])
        except (RuntimeError, ValueError):
            continue
    if len(aucs) < 10:
        logger.warning(
            "Bootstrap: only %d/%d iterations succeeded — returning None.", len(aucs), n_bootstrap
        )
        return None
    aucs_arr = np.array(aucs)
    return {
        "mean": float(aucs_arr.mean()),
        "std": float(aucs_arr.std()),
        "ci_low": float(np.percentile(aucs_arr, 2.5)),
        "ci_high": float(np.percentile(aucs_arr, 97.5)),
        "n_successful": len(aucs),
    }


# ---------------------------------------------------------------------------
# Execution faithfulness probe
# ---------------------------------------------------------------------------


def _compute_sequence_edit_distance(seq_a: List[Any], seq_b: List[Any]) -> int:
    """Standard Levenshtein edit distance between two sequences."""
    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[m]


def compute_execution_faithfulness(
    model: torch.nn.Module,
    data_loader: Any,
    config: Any,
    device: torch.device,
    max_samples: int,
) -> Dict[str, float]:
    """Compute per-step edit distance between predicted and ground-truth traces.

    The model's "predicted trace" is operationalized as the sequence of top-1
    primitive predictions at each real (non-padded) node.  The ground-truth
    is the `primitive_assignments` from the batch labels.

    Returns dict with:
        mean_edit_distance           — mean over DAGs (normalized by trace length)
        failure_rate                 — fraction of DAGs with any error
        catastrophic_failure_rate    — fraction with edit distance > 50% of trace length
        n_samples                    — number of DAGs evaluated
    """
    from marifah.training.graph_utils import prepare_batch_for_model

    model.eval()
    edit_distances: List[float] = []
    n_evaluated = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            B = batch.batch_size

            carry = _make_carry(B, config, device)

            coral_batch = prepare_batch_for_model(batch, config, device)
            result = model(carry, coral_batch, is_last_segment=True)
            logits = result[1]  # (B, max_nodes, vocab_size)

            pred_ids = logits.argmax(dim=-1)  # (B, max_nodes) — predicted primitive IDs
            gt_ids = batch.primitive_assignments  # (B, max_nodes) — -1 for padding
            node_mask = batch.node_mask  # (B, max_nodes) bool

            pred_cpu = pred_ids.cpu()
            gt_cpu = gt_ids.cpu()
            mask_cpu = node_mask.cpu()

            for b in range(B):
                real_mask = mask_cpu[b]  # (max_nodes,) bool
                pred_seq = pred_cpu[b][real_mask].tolist()
                gt_seq = gt_cpu[b][real_mask].tolist()
                # Filter out -1 padding in gt (shouldn't be any under the mask, but be safe)
                gt_seq = [x for x in gt_seq if x >= 0]

                if len(gt_seq) == 0:
                    continue

                dist = _compute_sequence_edit_distance(pred_seq, gt_seq)
                normalized = dist / len(gt_seq)
                edit_distances.append(normalized)
                n_evaluated += 1

                if n_evaluated >= max_samples:
                    break

            if n_evaluated >= max_samples:
                break

    if not edit_distances:
        return {"mean_edit_distance": float("nan"), "failure_rate": float("nan"),
                "catastrophic_failure_rate": float("nan"), "n_samples": 0}

    arr = np.array(edit_distances)
    mean_ed = float(arr.mean())
    failure_rate = float((arr > 0).mean())
    catastrophic_rate = float((arr > 0.5).mean())

    logger.info(
        "Execution faithfulness — mean_edit_dist=%.4f  failure_rate=%.4f  catastrophic=%.4f  n=%d",
        mean_ed, failure_rate, catastrophic_rate, n_evaluated,
    )
    return {
        "mean_edit_distance": mean_ed,
        "failure_rate": failure_rate,
        "catastrophic_failure_rate": catastrophic_rate,
        "n_samples": n_evaluated,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Warm-start probe: workflow-type AUC + faithfulness")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    p.add_argument("--dataset", required=True, help="Dataset root directory")
    p.add_argument("--split", default="val", help="Which split to evaluate (default: val)")
    p.add_argument("--config", required=True, help="TrainingConfig YAML used to build the model")
    p.add_argument("--output", required=True, help="Output JSON path for probe results")
    p.add_argument("--max_samples", type=int, default=1000,
                   help="Max DAGs to evaluate (default: 1000 for speed)")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="Bootstrap iterations for 95%% CI on AUC (0 = skip, default: 0)")
    p.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Config + model ---------------------------------------------------
    from marifah.training.config import load_config, TrainingConfig
    from marifah.training.trainer import build_model
    from marifah.training.checkpointing import load_warmstart

    config = load_config(args.config)
    device = torch.device(args.device)

    logger.info("Building model from config %s", args.config)
    model = build_model(config, device)

    logger.info("Loading checkpoint: %s", args.checkpoint)
    load_warmstart(args.checkpoint, model)
    model.eval()

    # ---- Data loader -------------------------------------------------------
    from marifah.training.data_pipeline import build_data_loaders

    # Override dataset_root to the provided path
    config.data.dataset_root = args.dataset
    loaders = build_data_loaders(config)
    loader = loaders.get(args.split)
    if loader is None:
        logger.error("Split '%s' not found under %s — aborting.", args.split, args.dataset)
        sys.exit(1)

    logger.info("Evaluating split='%s'  max_samples=%d  device=%s",
                args.split, args.max_samples, args.device)

    # ---- Carry-state extraction for AUC probe -----------------------------
    logger.info("Extracting carry states for workflow-type AUC probe ...")
    carry_states, labels = _extract_carry_states(
        model, loader, config, device, args.max_samples
    )
    n_extracted = carry_states.shape[0]
    if n_extracted < args.max_samples:
        logger.info(
            "Carry states: shape=%s  n_unique_labels=%d  "
            "(extracted %d < max_samples=%d — dataset smaller than requested)",
            carry_states.shape, len(set(labels)), n_extracted, args.max_samples,
        )
    else:
        logger.info("Carry states: shape=%s  n_unique_labels=%d",
                    carry_states.shape, len(set(labels)))
    assert carry_states.shape[0] == len(labels), (
        f"carry_states rows ({carry_states.shape[0]}) != labels length ({len(labels)})"
    )

    # ---- Workflow-type AUC ------------------------------------------------
    logger.info("Computing workflow-type AUC ...")
    auc_results = compute_workflow_type_auc(carry_states, labels)

    # ---- Bootstrap CI (optional) ------------------------------------------
    bootstrap_results = None
    if args.bootstrap > 0:
        logger.info("Computing bootstrap AUC CI (%d iterations) ...", args.bootstrap)
        bootstrap_results = _bootstrap_auc(carry_states, labels, n_bootstrap=args.bootstrap)
        if bootstrap_results is not None:
            logger.info(
                "Bootstrap AUC: mean=%.4f  std=%.4f  95%% CI=[%.4f, %.4f]  n_ok=%d",
                bootstrap_results["mean"], bootstrap_results["std"],
                bootstrap_results["ci_low"], bootstrap_results["ci_high"],
                bootstrap_results["n_successful"],
            )
        else:
            logger.warning("Bootstrap returned None — too few successful iterations.")

    # ---- Execution faithfulness -------------------------------------------
    logger.info("Computing execution faithfulness ...")
    # Reload data_loader since we consumed it above
    loaders2 = build_data_loaders(config)
    loader2 = loaders2[args.split]
    faithfulness_results = compute_execution_faithfulness(
        model, loader2, config, device, args.max_samples
    )

    # ---- Combine and emit -------------------------------------------------
    results: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "split": args.split,
        "config": str(args.config),
        "max_samples": args.max_samples,
        "n_extracted": n_extracted,
        "device": args.device,
        "workflow_type_auc": auc_results,
        "execution_faithfulness": faithfulness_results,
    }
    if bootstrap_results is not None:
        results["workflow_type_auc_bootstrap"] = bootstrap_results

    # Decision summary per OD7 criteria (pre-registered in §2.6)
    auc = auc_results["auc"]
    mean_ed = faithfulness_results["mean_edit_distance"]
    results["decision_inputs"] = {
        "workflow_type_auc": auc,
        "mean_edit_distance": mean_ed,
        "n_classes_in_probe": auc_results["n_classes_present_in_test"],
        "n_classes_filtered": auc_results["n_classes_filtered_single_sample"],
        "note": (
            "AUC >= 0.6 + mean_edit_distance <= 0.1 → warm-start preferable. "
            "See session-06-warmstart-verdict.md for full decision table."
        ),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results written to %s", args.output)
    logger.info("=== SUMMARY ===")
    logger.info("  workflow_type_auc : %.4f  (classes in test=%d, filtered=%d)",
                auc, auc_results["n_classes_present_in_test"],
                auc_results["n_classes_filtered_single_sample"])
    if bootstrap_results is not None:
        logger.info("  auc 95%% CI      : [%.4f, %.4f]",
                    bootstrap_results["ci_low"], bootstrap_results["ci_high"])
    logger.info("  mean_edit_distance: %.4f", mean_ed)
    logger.info("  failure_rate      : %.4f", faithfulness_results["failure_rate"])


if __name__ == "__main__":
    main()
