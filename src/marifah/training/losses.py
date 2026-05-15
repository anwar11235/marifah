"""Loss functions and ACT loss head for CORAL.

stablemax_cross_entropy is the default for small-sample experiments.
ACTLossHead wraps CoralACT; CoralV3LossHead extends it with free energy terms.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID: int = -100


# ---------------------------------------------------------------------------
# Stablemax
# ---------------------------------------------------------------------------


def _stablemax_s(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    return torch.where(x < 0, 1.0 / (1.0 - x + epsilon), x + 1.0)


def _log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    s_x = _stablemax_s(x)
    return torch.log(s_x / s_x.sum(dim=dim, keepdim=True))


def stablemax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_LABEL_ID,
) -> torch.Tensor:
    logprobs = _log_stablemax(logits.to(torch.float64), dim=-1)
    valid_mask = labels != ignore_index
    safe_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs,
        index=safe_labels.to(torch.long).unsqueeze(-1),
        dim=-1,
    ).squeeze(-1)
    return -torch.where(valid_mask, prediction_logprobs, torch.zeros_like(prediction_logprobs))


def softmax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_LABEL_ID,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.to(torch.float32).view(-1, logits.shape[-1]),
        labels.to(torch.long).view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(labels.shape)


# ---------------------------------------------------------------------------
# ACT loss head
# ---------------------------------------------------------------------------


class ACTLossHead(nn.Module):
    """Wraps CoralACT and computes training loss + evaluation metrics.

    total = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
    """

    def __init__(self, model: nn.Module, loss_type: str) -> None:
        super().__init__()
        self.model = model
        _loss_fns = {
            "stablemax_cross_entropy": stablemax_cross_entropy,
            "softmax_cross_entropy": softmax_cross_entropy,
        }
        if loss_type not in _loss_fns:
            raise ValueError(f"Unknown loss_type: {loss_type!r}. Choose from {list(_loss_fns)}")
        self.loss_fn = _loss_fns[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore[operator]

    @property
    def puzzle_emb(self):
        return self.model.puzzle_emb  # type: ignore[attr-defined]

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            preds_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = preds_correct.sum(-1) == loss_counts

            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics: Dict[str, torch.Tensor] = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (preds_correct.to(torch.float32) / loss_divisor).sum(-1),
                    torch.zeros_like(loss_counts, dtype=torch.float32),
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
                ).sum(),
                "steps": torch.where(
                    valid_metrics,
                    new_carry.steps.to(torch.float32),
                    torch.zeros_like(new_carry.steps, dtype=torch.float32),
                ).sum(),
            }

        lm_loss = (
            self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor
        ).sum()

        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )

        metrics["lm_loss"] = lm_loss.detach()
        metrics["q_halt_loss"] = q_halt_loss.detach()

        q_continue_loss: torch.Tensor = torch.tensor(0.0, device=lm_loss.device)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum",
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()


# ---------------------------------------------------------------------------
# Predictive coding loss (Phase 1)
# ---------------------------------------------------------------------------


def predictive_coding_loss(
    epsilon: torch.Tensor,
    pi: torch.Tensor,
    lambda_pred: float = 0.1,
    lambda_pi: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Free energy loss terms for predictive coding.

    pi_reg uses a log-normal prior centred at π=1 (symmetric log-space penalty).
    """
    pred_loss = lambda_pred * 0.5 * (pi * epsilon ** 2).sum(dim=-1).mean()
    pi_reg = lambda_pi * 0.5 * (torch.log(pi + 1e-8) ** 2).sum(dim=-1).mean()
    return pred_loss, pi_reg


# ---------------------------------------------------------------------------
# Load-balancing loss (Phase 2)
# ---------------------------------------------------------------------------


def load_balancing_loss(
    all_routing_logits: list,
    S: int,
) -> torch.Tensor:
    avg_probs = torch.stack(
        [F.softmax(l, dim=-1) for l in all_routing_logits]
    ).mean(dim=(0, 1))
    return S * (avg_probs * torch.log(avg_probs * S + 1e-8)).sum()


# ---------------------------------------------------------------------------
# CoralV3LossHead (Phase 1)
# ---------------------------------------------------------------------------


class CoralV3LossHead(nn.Module):
    """Loss head for CoralV3ACT — extends ACTLossHead with free energy terms."""

    def __init__(self, model: nn.Module, loss_type: str) -> None:
        super().__init__()
        self.model = model
        _loss_fns = {
            "stablemax_cross_entropy": stablemax_cross_entropy,
            "softmax_cross_entropy": softmax_cross_entropy,
        }
        if loss_type not in _loss_fns:
            raise ValueError(f"Unknown loss_type: {loss_type!r}. Choose from {list(_loss_fns)}")
        self.loss_fn = _loss_fns[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore[operator]

    @property
    def puzzle_emb(self):
        return self.model.puzzle_emb  # type: ignore[attr-defined]

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            preds_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = preds_correct.sum(-1) == loss_counts

            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics: Dict[str, torch.Tensor] = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (preds_correct.to(torch.float32) / loss_divisor).sum(-1),
                    torch.zeros_like(loss_counts, dtype=torch.float32),
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
                ).sum(),
                "steps": torch.where(
                    valid_metrics,
                    new_carry.steps.to(torch.float32),
                    torch.zeros_like(new_carry.steps, dtype=torch.float32),
                ).sum(),
            }

        lm_loss = (
            self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor
        ).sum()

        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )

        metrics["lm_loss"] = lm_loss.detach()
        metrics["q_halt_loss"] = q_halt_loss.detach()

        q_continue_loss: torch.Tensor = torch.tensor(0.0, device=lm_loss.device)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum",
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

        if "epsilon_final" in outputs and outputs["epsilon_final"] is not None:
            lambda_pred = self.model.config.lambda_pred  # type: ignore[attr-defined]
            lambda_pi = self.model.config.lambda_pi      # type: ignore[attr-defined]
            pred_loss, pi_reg = predictive_coding_loss(
                outputs["epsilon_final"],
                outputs["pi_final"],
                lambda_pred=lambda_pred,
                lambda_pi=lambda_pi,
            )
            total_loss = total_loss + pred_loss + pi_reg
            metrics["pred_loss"] = pred_loss.detach()
            metrics["pi_reg"] = pi_reg.detach()

        for key in ("prediction_error", "precision_mean"):
            if key in outputs:
                metrics[key] = outputs[key].detach()
        if "pi_final" in outputs and outputs["pi_final"] is not None:
            metrics["precision_std"] = outputs["pi_final"].std().detach()

        routing_logits_H = outputs.get("routing_logits_H")
        routing_logits_L = outputs.get("routing_logits_L")
        if routing_logits_H is not None and routing_logits_L is not None:
            all_routing_logits = routing_logits_H + routing_logits_L
            S = self.model.config.num_columns  # type: ignore[attr-defined]
            lambda_balance = self.model.config.lambda_balance  # type: ignore[attr-defined]
            balance_loss = load_balancing_loss(all_routing_logits, S)
            total_loss = total_loss + lambda_balance * balance_loss
            metrics["load_balance_loss"] = balance_loss.detach()
            with torch.no_grad():
                entropies = []
                for logits in all_routing_logits:
                    probs = F.softmax(logits, dim=-1)
                    entropies.append(-(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean())
                metrics["router_entropy"] = torch.stack(entropies).mean()
            with torch.no_grad():
                avg_probs = torch.stack(
                    [F.softmax(l, dim=-1) for l in all_routing_logits]
                ).mean(dim=(0, 1))
                for col_idx in range(avg_probs.shape[0]):
                    metrics[f"col_{col_idx}_freq"] = avg_probs[col_idx]

        moe_recon_loss = outputs.get("moe_recon_loss")
        moe_lb_loss = outputs.get("moe_lb_loss")
        if moe_recon_loss is not None:
            metrics["crystal/recon_loss"] = moe_recon_loss.detach()
        if moe_recon_loss is not None and moe_lb_loss is not None:
            cfg = self.model.config  # type: ignore[attr-defined]
            total_loss = (
                total_loss
                + cfg.lambda_moe_recon * moe_recon_loss
                + cfg.lambda_moe_balance * moe_lb_loss
            )
            metrics["crystal/lb_loss"] = moe_lb_loss.detach()

        moe_pt = outputs.get("moe_passthrough_weight")
        if moe_pt is not None:
            metrics["crystal/mean_passthrough_weight"] = moe_pt.detach()
            metrics["crystal/mean_codebook_weight"] = (1.0 - moe_pt).detach()
        moe_entropy = outputs.get("moe_routing_entropy")
        if moe_entropy is not None:
            metrics["crystal/routing_entropy"] = moe_entropy.detach()
        moe_util = outputs.get("moe_codebook_util_frac")
        if moe_util is not None:
            metrics["crystal/codebook_utilisation_frac"] = moe_util.detach()

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
