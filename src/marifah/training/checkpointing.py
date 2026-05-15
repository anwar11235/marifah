"""Checkpoint save/load with atomic write and full training state.

Supports:
  - Full saves: model + optimizer + scheduler + step/epoch + config + metadata
  - Partial loads: model-only (warm-start) or full resume
  - Atomic write: save to tmp then rename — protects against corrupt files on interrupt
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.optim import Optimizer


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
    """Save a full training checkpoint atomically.

    Writes to a temp file in the same directory, then renames — so an interrupted
    save never corrupts the previous checkpoint.

    Args:
        path:          Destination file path (e.g., checkpoints/phase0/step_1000.pt)
        model:         Model to save (state_dict)
        optimizer:     Optimizer state
        scheduler:     LR scheduler (optional)
        step:          Global training step
        epoch:         Current epoch number
        config_dict:   Training config as a plain dict (for reproducibility)
        extra_metadata: Any extra info to embed (eval metrics, utilization snapshot, etc.)
    """
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

    # Atomic write: temp file in same directory, then rename
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


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Always loads model weights.  Loads optimizer/scheduler state only if those
    objects are provided (and the checkpoint contains them).

    Args:
        path:      Path to checkpoint file
        model:     Model to load weights into
        optimizer: If provided, restore optimizer state (full resume)
        scheduler: If provided, restore scheduler state

    Returns:
        Metadata dict: {'step': int, 'epoch': int, 'config': dict, 'extra_metadata': dict}
    """
    payload = torch.load(path, map_location="cpu")

    # Handle checkpoints that are bare state_dicts (from old save_checkpoint in train.py)
    if "model_state_dict" not in payload:
        model.load_state_dict(payload, strict=False)
        return {"step": 0, "epoch": 0, "config": {}, "extra_metadata": {}}

    model.load_state_dict(payload["model_state_dict"], strict=False)

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
) -> Dict[str, Any]:
    """Load model weights from a checkpoint without restoring optimizer state.

    Used for Phase 1 warm-start from Phase 0 checkpoint (OD7 comparison).
    The optimizer is always freshly initialized in this mode.
    """
    return load_checkpoint(path, model, optimizer=None, scheduler=None)
