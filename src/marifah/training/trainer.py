"""Main Trainer class: orchestrates training loop, eval, and checkpointing.

Uses CoralV3Inner directly to access PredMetrics (including hmsc_aux_losses)
that are not forwarded through CoralV3ACT.  Implements a 1-step-gradient
training loop: one CORAL inner segment per batch, fresh carry each step.

Discipline rules enforced:
  - eval_interval is in EPOCHS, not steps (CORAL discipline rule)
  - HMSC lambdas set at model-build time via HMSC constructor
  - use_hmsc=False preserves bit-identical output to the pre-HMSC baseline
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from marifah.training.config import TrainingConfig
from marifah.training.graph_losses import compute_total_loss
from marifah.training.graph_utils import prepare_batch_for_model, derive_region_labels
from marifah.training.checkpointing import save_checkpoint, load_checkpoint, load_warmstart
from marifah.training.logging import TrainingLogger
from marifah.training.eval_loop import evaluate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_model(config: TrainingConfig, device: torch.device) -> nn.Module:
    """Construct a CoralV3Inner for graph DAG training.

    Builds with use_predictive_coding=True (required for _forward_with_pc path
    which returns PredMetrics and routes HMSC).  HMSC is attached post-init with
    parameters from config (including training lambdas).
    """
    from marifah.models.coral_base import CoralConfig
    from marifah.models.coral import CoralV3Inner
    from marifah.models.hmsc.hmsc import HMSC

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
        use_hmsc=False,                  # set explicitly below
        forward_dtype="float32",
        halt_max_steps=config.model.halt_max_steps,
        halt_exploration_prob=config.model.halt_exploration_prob,
    )

    model = CoralV3Inner(coral_cfg).to(device)

    if config.model.use_hmsc:
        hcfg = config.model.hmsc
        assert hcfg is not None, "model.hmsc config section required when use_hmsc=True"
        hmsc = HMSC(
            K_G=hcfg.K_G,
            K_R=hcfg.K_R,
            K_P=hcfg.K_P,
            d_model=config.model.d_model,
            d_G=hcfg.d_G,
            d_R=hcfg.d_R,
            d_P=hcfg.d_P,
            d_output=config.model.d_model,
            num_workflow_types=hcfg.num_workflow_types,
            num_pattern_types=hcfg.num_pattern_types,
            num_primitives=hcfg.num_primitives,
            num_regions=hcfg.num_regions,
            composition_method=hcfg.composition_method,
            lambda_G=config.training.lambda_G,
            lambda_R=config.training.lambda_R,
            lambda_P=config.training.lambda_P,
            p_discreteness=hcfg.p_discreteness,
        ).to(device)
        model.hmsc = hmsc

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model built: %d params | use_hmsc=%s | d_model=%d | seq_len=%d",
                n_params, config.model.use_hmsc, config.model.d_model, config.model.max_nodes)
    return model


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Orchestrates graph DAG training with eval and checkpoint management."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        data_loaders: Dict[str, Optional[DataLoader]],
        logger_obj: TrainingLogger,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.data_loaders = data_loaders
        self.logger = logger_obj
        self.device = device

        self.global_step = 0
        self.start_epoch = 0
        self._total_steps = 1  # updated in train()

        try:
            from adam_atan2_pytorch import AdamAtan2 as AdamATan2
        except ImportError:
            from marifah.training.adam_atan2 import AdamATan2

        self.optimizer = AdamATan2(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

    def _compute_lr(self) -> float:
        from marifah.training.scheduler import cosine_schedule_with_warmup_lr_lambda
        return cosine_schedule_with_warmup_lr_lambda(
            current_step=self.global_step,
            base_lr=self.config.training.learning_rate,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=max(self._total_steps, self.config.training.warmup_steps + 1),
            min_ratio=0.1,
        )

    def _set_lr(self) -> float:
        lr = self._compute_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def train(self) -> None:
        """Main training loop (epoch-based eval — CORAL discipline rule)."""
        train_loader = self.data_loaders.get("train")
        val_loader = self.data_loaders.get("val")
        if train_loader is None:
            raise RuntimeError("No train DataLoader — check dataset_root in config.")

        cfg = self.config.training
        self._total_steps = max(
            len(train_loader) * cfg.max_epochs,
            cfg.warmup_steps + 1,
        )

        best_val_loss: Optional[float] = None
        patience_counter = 0
        _stop_training = False

        for epoch in range(self.start_epoch, cfg.max_epochs):
            if _stop_training:
                break
            self.model.train()
            epoch_losses: Dict[str, float] = {}

            for batch in train_loader:
                step_losses = self.step(batch)
                for k, v in step_losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v

                if self.global_step % self.config.logging.log_interval_steps == 0:
                    self.logger.log_step(self.global_step, step_losses, self._compute_lr())

                self.global_step += 1

                if (cfg.save_every_n_steps is not None
                        and self.global_step % cfg.save_every_n_steps == 0):
                    self._save(
                        os.path.join(self.config.logging.checkpoint_dir,
                                     f"step_{self.global_step:07d}.pt"),
                        epoch,
                    )

                if cfg.max_steps is not None and self.global_step >= cfg.max_steps:
                    logger.info("max_steps=%d reached at epoch %d; stopping.", cfg.max_steps, epoch)
                    _stop_training = True
                    break

            n_steps = max(len(train_loader), 1)
            logger.info(
                "[epoch %d] main=%.4f halt=%.4f aux=%.4f",
                epoch,
                epoch_losses.get("main", 0.0) / n_steps,
                epoch_losses.get("halt", 0.0) / n_steps,
                epoch_losses.get("aux_total", 0.0) / n_steps,
            )

            # Eval cadence: every eval_interval_epochs EPOCHS (not steps)
            if (epoch + 1) % cfg.eval_interval_epochs == 0 and val_loader is not None:
                eval_metrics = evaluate(self.model, val_loader, self.config, self.device)
                self.logger.log_eval(self.global_step, "val", eval_metrics)
                logger.info(
                    "[epoch %d val] loss_main=%.4f acc=%.4f",
                    epoch,
                    eval_metrics.get("loss_main", float("nan")),
                    eval_metrics.get("accuracy_node", float("nan")),
                )

                ckpt = os.path.join(self.config.logging.checkpoint_dir, f"epoch_{epoch:04d}.pt")
                self._save(ckpt, epoch, extra={"eval": eval_metrics})

                val_loss = eval_metrics.get("loss_main", float("inf"))
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save(
                        os.path.join(self.config.logging.checkpoint_dir, "best.pt"),
                        epoch,
                        extra={"eval": eval_metrics},
                    )
                else:
                    patience_counter += 1

                if (cfg.early_stopping_patience is not None
                        and patience_counter >= cfg.early_stopping_patience):
                    logger.info("Early stopping at epoch %d.", epoch)
                    break

            self.model.train()

        final = os.path.join(self.config.logging.checkpoint_dir, "final.pt")
        last_epoch = min(self.start_epoch + cfg.max_epochs - 1, cfg.max_epochs - 1)
        self._save(final, last_epoch)
        self.logger.finish()

    def step(self, batch) -> Dict[str, float]:
        """One forward-backward-update step."""
        from marifah.models.coral_base import InnerCarry

        graph_batch = batch.to(self.device)
        B = graph_batch.batch_size
        max_nodes = self.config.model.max_nodes
        d_model = self.config.model.d_model

        carry = InnerCarry(
            z_H=torch.zeros(B, max_nodes, d_model, dtype=torch.float32, device=self.device),
            z_L=torch.zeros(B, max_nodes, d_model, dtype=torch.float32, device=self.device),
        )

        coral_batch = prepare_batch_for_model(graph_batch, self.config, self.device)

        result = self.model(carry, coral_batch, is_last_segment=True)
        logits = result[1]
        q_tuple = result[2]
        pred_metrics = result[3] if len(result) > 3 else None
        q_halt_logits = q_tuple[0] if isinstance(q_tuple, tuple) else None

        loss_dict = compute_total_loss(
            logits=logits,
            q_halt_logits=q_halt_logits,
            pred_metrics=pred_metrics,
            graph_batch=graph_batch,
            config=self.config,
            device=self.device,
        )

        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
        self._set_lr()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if (pred_metrics is not None
                and pred_metrics.hmsc_utilization is not None
                and self.global_step % self.config.logging.utilization_interval_steps == 0):
            self.logger.log_codebook_stats(
                self.global_step,
                {"codebook_utilization": pred_metrics.hmsc_utilization},
            )

        return {k: float(v.item()) if hasattr(v, "item") else float(v)
                for k, v in loss_dict.items()}

    def _save(
        self,
        path: str,
        epoch: int,
        extra: Optional[Dict] = None,
    ) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=None,
            step=self.global_step,
            epoch=epoch,
            config_dict=self.config.model_dump(),
            extra_metadata=extra,
        )
        logger.info("Checkpoint saved: %s", path)

    def resume(self, checkpoint_path: str) -> None:
        meta = load_checkpoint(checkpoint_path, self.model, self.optimizer)
        self.global_step = meta.get("step", 0)
        self.start_epoch = meta.get("epoch", 0)
        logger.info("Resumed from %s (step=%d, epoch=%d)",
                    checkpoint_path, self.global_step, self.start_epoch)

    def warmstart(self, checkpoint_path: str) -> None:
        load_warmstart(checkpoint_path, self.model)
        logger.info("Warm-started from %s", checkpoint_path)
