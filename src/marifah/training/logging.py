"""Training logger: W&B integration + local JSONL fallback.

Behaviours:
  - wandb_mode=online  → logs to W&B + local JSONL
  - wandb_mode=offline → logs to local wandb files + local JSONL
  - wandb_mode=disabled → local JSONL only; no wandb import side-effects

Always writes a JSONL file at <checkpoint_dir>/<run_name>.jsonl.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Dict, Optional

from marifah.training.config import TrainingConfig

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Structured logging to W&B (optional) and a local JSONL file."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self._wandb_enabled = config.logging.wandb_mode != "disabled"
        self._run_name = config.logging.run_name or self._generate_run_name(config)
        self._wandb = None

        # Set up local JSONL file
        log_dir = config.logging.checkpoint_dir
        os.makedirs(log_dir, exist_ok=True)
        self._jsonl_path = os.path.join(log_dir, f"{self._run_name}.jsonl")
        self._jsonl_file = open(self._jsonl_path, "a", encoding="utf-8")

        if self._wandb_enabled:
            try:
                import wandb
                self._wandb = wandb
                wandb.init(
                    project=config.logging.wandb_project,
                    name=self._run_name,
                    config=config.model_dump(),
                    mode=config.logging.wandb_mode,
                    settings=wandb.Settings(_disable_stats=True),
                )
                logger.info("W&B run initialised: %s", self._run_name)
            except Exception as exc:
                logger.warning("W&B init failed (%s). Falling back to local logging only.", exc)
                self._wandb = None
                self._wandb_enabled = False

    @property
    def run_name(self) -> str:
        return self._run_name

    @property
    def jsonl_path(self) -> str:
        return self._jsonl_path

    def log_step(
        self,
        step: int,
        losses: Dict[str, float],
        learning_rate: float,
        utilization: Optional[Dict] = None,
        extra: Optional[Dict] = None,
    ) -> None:
        """Log one training step."""
        record: Dict = {
            "_type": "step",
            "step": step,
            "timestamp": time.time(),
            "learning_rate": learning_rate,
            **{f"train/{k}": v for k, v in losses.items()},
        }
        if utilization:
            record.update({f"util/{k}": v for k, v in utilization.items()})
        if extra:
            record.update(extra)

        self._write_jsonl(record)
        if self._wandb is not None:
            self._wandb.log(record, step=step)

    def log_eval(
        self,
        step: int,
        split_name: str,
        metrics: Dict[str, float],
    ) -> None:
        """Log evaluation metrics for a split."""
        record: Dict = {
            "_type": "eval",
            "step": step,
            "split": split_name,
            "timestamp": time.time(),
            **{f"eval/{split_name}/{k}": v for k, v in metrics.items()},
        }
        self._write_jsonl(record)
        if self._wandb is not None:
            self._wandb.log(record, step=step)

    def log_codebook_stats(self, step: int, hmsc_outputs: Dict) -> None:
        """Emit HMSC codebook utilisation stats."""
        util = hmsc_outputs.get("codebook_utilization", {})
        if not util:
            return
        record: Dict = {
            "_type": "codebook_stats",
            "step": step,
            "timestamp": time.time(),
            **{f"codebook/{k}": float(v) if hasattr(v, "item") else v
               for k, v in util.items()},
        }
        self._write_jsonl(record)
        if self._wandb is not None:
            self._wandb.log(record, step=step)

    def finish(self) -> None:
        """Flush and close logger resources."""
        try:
            self._jsonl_file.flush()
            self._jsonl_file.close()
        except Exception:
            pass
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass

    def _write_jsonl(self, record: Dict) -> None:
        try:
            self._jsonl_file.write(json.dumps(record, default=_json_default) + "\n")
            self._jsonl_file.flush()
        except Exception as exc:
            logger.warning("JSONL write failed: %s", exc)

    @staticmethod
    def _generate_run_name(config: TrainingConfig) -> str:
        tag = uuid.uuid4().hex[:6]
        return f"{config.experiment.name}_phase{config.experiment.phase}_{tag}"


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
