"""Checkpoint save/load with atomic write and full training state.

Supports:
  - Full saves: model + optimizer + scheduler + step/epoch + config + metadata
  - Partial loads: model-only (warm-start) or full resume
  - Atomic write: save to tmp then rename — protects against corrupt files on interrupt
  - Cross-architecture remap: sudoku_to_marifah strips prefix/compile artifacts
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint remap
# ---------------------------------------------------------------------------


def remap_sudoku_checkpoint(
    source_state_dict: Dict[str, torch.Tensor],
    target_model: nn.Module,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Remap a Sudoku Phase 3c checkpoint to marifah key naming.

    Three sources of key drift between Sudoku and marifah:
    1. Prefix ``model.inner.`` — Sudoku wrapped the substrate in a parent module.
    2. ``_orig_mod.`` infix — Sudoku was saved while torch.compile'd.
    3. Shape mismatches — vocab_size=11 (Sudoku) vs 10 (marifah) on embed/lm_head.

    CORAL-v3-only keys (prediction_net.*, precision_net.*, moe_codebook.*) are
    filtered because they are not present in the marifah substrate.

    Args:
        source_state_dict: State dict from the Sudoku checkpoint.
        target_model: The marifah model instance to load into.

    Returns:
        (remapped_dict, manifest) where manifest has:
          - ``loaded``: list of (original_key, remapped_key) pairs
          - ``dropped_not_in_target``: list of (original_key, remapped_key) pairs
          - ``dropped_shape_mismatch``: list of (original_key, src_shape, tgt_shape) tuples
    """
    target_sd = target_model.state_dict()

    manifest: Dict[str, Any] = {
        "loaded": [],
        "dropped_not_in_target": [],
        "dropped_shape_mismatch": [],
    }
    remapped: Dict[str, torch.Tensor] = {}

    for orig_key, tensor in source_state_dict.items():
        # Step 1: strip model.inner. prefix
        key = re.sub(r"^model\.inner\.", "", orig_key)
        # Step 2: strip _orig_mod. from any position in the path
        key = re.sub(r"^_orig_mod\.", "", key)
        key = re.sub(r"\._orig_mod\.", ".", key)

        # Step 3: filter keys not in target
        if key not in target_sd:
            manifest["dropped_not_in_target"].append((orig_key, key))
            logger.debug("Dropped (not in target): %s -> %s", orig_key, key)
            continue

        # Step 4: filter shape mismatches
        target_shape = target_sd[key].shape
        if tensor.shape != target_shape:
            manifest["dropped_shape_mismatch"].append((orig_key, tuple(tensor.shape), tuple(target_shape)))
            logger.warning(
                "Shape mismatch (dropped): %s  source=%s  target=%s",
                orig_key, tuple(tensor.shape), tuple(target_shape),
            )
            continue

        remapped[key] = tensor
        manifest["loaded"].append((orig_key, key))

    n_loaded = len(manifest["loaded"])
    n_not_in_target = len(manifest["dropped_not_in_target"])
    n_shape = len(manifest["dropped_shape_mismatch"])
    logger.info(
        "Remap 'sudoku_to_marifah': %d loaded | %d dropped (not in target) | %d dropped (shape mismatch)",
        n_loaded, n_not_in_target, n_shape,
    )
    if manifest["loaded"]:
        sample = [f"{o}->{n}" for o, n in manifest["loaded"][:5]]
        logger.info("Sample loaded keys: %s", sample)

    return remapped, manifest


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[Any],
    step: int,
    epoch: int,
    config_dict: Dict[str, Any],
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a full training checkpoint atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "epoch": epoch,
        "config": config_dict,
        "extra_metadata": extra_metadata or {},
    }

    dir_name = os.path.dirname(os.path.abspath(path))
    with tempfile.NamedTemporaryFile(dir=dir_name, delete=False, suffix=".pt.tmp") as tmp:
        tmp_path = tmp.name
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def _find_substrate_key(model: nn.Module) -> Optional[str]:
    """Return the first H_level weight key for change-detection snapshotting."""
    return next(
        (k for k in model.state_dict() if k.startswith("H_level") and k.endswith(".weight")),
        None,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    remap: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Always loads model weights. Loads optimizer/scheduler state only if those
    objects are provided (and the checkpoint contains them).

    Args:
        path:      Path to checkpoint file.
        model:     Model to load weights into.
        optimizer: If provided, restore optimizer state (full resume).
        scheduler: If provided, restore scheduler state.
        remap:     If ``'sudoku_to_marifah'``, apply key remapping before load.
                   Required when loading a Sudoku Phase 3c checkpoint into marifah.

    Returns:
        Metadata dict: {'step': int, 'epoch': int, 'config': dict, 'extra_metadata': dict}

    Raises:
        RuntimeError: If no substrate parameters were updated after load, indicating
                      a probable key-naming mismatch.
        ValueError:   If an unknown remap strategy is specified.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)

    # Bare state_dict format (from old train.py saves)
    if "model_state_dict" not in payload:
        state_dict = payload
        is_legacy = True
    else:
        state_dict = payload["model_state_dict"]
        is_legacy = False

    # Apply cross-architecture remap
    if remap is not None:
        if remap == "sudoku_to_marifah":
            state_dict, _manifest = remap_sudoku_checkpoint(state_dict, model)
        else:
            raise ValueError(f"Unknown remap strategy: {remap!r}. Supported: 'sudoku_to_marifah'")

    # Snapshot substrate param for change-detection
    substrate_key = _find_substrate_key(model)
    pre_sum: Optional[float] = None
    if substrate_key is not None:
        pre_sum = float(model.state_dict()[substrate_key].sum())

    # Load weights
    incompat = model.load_state_dict(state_dict, strict=False)

    # Warn on dropped keys (addresses Audit Finding 1 from Session 6)
    if incompat.missing_keys:
        logger.warning(
            "load_state_dict: %d missing_keys (in model, not in checkpoint). First 5: %s",
            len(incompat.missing_keys), incompat.missing_keys[:5],
        )
    if incompat.unexpected_keys:
        logger.warning(
            "load_state_dict: %d unexpected_keys (in checkpoint, not in model). First 5: %s",
            len(incompat.unexpected_keys), incompat.unexpected_keys[:5],
        )

    # Verify that at least one substrate parameter actually changed.
    # A no-change result means the checkpoint produced zero key matches —
    # almost certainly a key-naming mismatch. Raise rather than silently
    # train from random init while believing a warm-start happened.
    if substrate_key is not None and pre_sum is not None:
        post_sum = float(model.state_dict()[substrate_key].sum())
        if pre_sum == post_sum:
            raise RuntimeError(
                f"Checkpoint load completed but substrate parameter '{substrate_key}' "
                "was not updated. Probable key-naming mismatch — all checkpoint keys "
                "were dropped as unexpected. If loading a Sudoku Phase 3c checkpoint, "
                "set remap='sudoku_to_marifah'."
            )

    if is_legacy:
        return {"step": 0, "epoch": 0, "config": {}, "extra_metadata": {}}

    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    return {
        "step": payload.get("step", 0),
        "epoch": payload.get("epoch", 0),
        "config": payload.get("config", {}),
        "extra_metadata": payload.get("extra_metadata", {}),
    }


def load_warmstart(
    path: str,
    model: nn.Module,
    remap: Optional[str] = None,
) -> Dict[str, Any]:
    """Load model weights from a checkpoint without restoring optimizer state.

    Used for warm-start from a prior checkpoint (OD7 comparison).
    The optimizer is always freshly initialized in this mode.

    Args:
        path:  Path to checkpoint file.
        model: Model to load weights into.
        remap: Key-remap strategy. Pass ``'sudoku_to_marifah'`` when loading
               the Sudoku Phase 3c checkpoint into a marifah model.
    """
    return load_checkpoint(path, model, optimizer=None, scheduler=None, remap=remap)
