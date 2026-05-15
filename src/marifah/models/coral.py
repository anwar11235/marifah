"""CoralV3Inner — CORAL base with precision-weighted predictive coding
and Phase 3b Soft MoE Crystallization.

Phase 1 — Predictive coding:
    H predicts L's state; L receives that prediction; H receives the
    precision-weighted prediction error (free-energy minimisation).

Phase 2 — Columnar routing (STUBBED):
    agate-cuckoo and curly-manatee both collapsed without convergence.
    Dispatch paths that enable columnar routing raise NotImplementedError.
    Reviving columnar routing requires its own redesign; stubs are kept so
    the dispatch-table structure is preserved for future use.

Phase 3b — Soft MoE Crystallization:
    A SpatialMoECodebook routes each L-state through K_modes full-spatial
    codebook experts plus one passthrough expert (the recurrence output z_L_rec).
    The convex combination z_L_out = w_pt*z_L_rec + (1-w_pt)*z_bypass replaces
    the raw L-level output at EVERY H-cycle.

Dispatch table:
  pc=F, cr=F, cry=F  →  super().forward() (CoralInner, unchanged, 3-tuple)
  pc=T, cr=F, cry=*  →  _forward_with_pc()        ← ONLY active path
  pc=F, cr=T, …     →  NotImplementedError        (columnar routing stubbed)
  pc=T, cr=T, …     →  NotImplementedError        (columnar routing stubbed)
  pc=F, cr=F, cry=T →  NotImplementedError        (cry without PC unsupported)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from marifah.models.coral_base import CoralConfig, CoralInner, InnerCarry
from marifah.models.columnar import ColumnarReasoningModule
from marifah.models.codebook import (
    CrystallizationBuffer,
    SpatialMoECodebook,
)
from marifah.models.prediction import PredictionNet, PrecisionNet
from marifah.models.transformer_block import TransformerBlockConfig


# ---------------------------------------------------------------------------
# Metrics carrier
# ---------------------------------------------------------------------------


@dataclass
class PredMetrics:
    """Statistics collected during one inner forward pass."""

    pred_error_norms: List[torch.Tensor]
    precision_means: List[torch.Tensor]
    epsilon_final: Optional[torch.Tensor]
    pi_final: Optional[torch.Tensor]
    routing_logits_H: Optional[List[torch.Tensor]] = field(default=None)
    routing_logits_L: Optional[List[torch.Tensor]] = field(default=None)
    moe_recon_loss: Optional[torch.Tensor] = field(default=None)
    moe_lb_loss: Optional[torch.Tensor] = field(default=None)
    moe_passthrough_weight: float = field(default=1.0)
    moe_routing_entropy: Optional[float] = field(default=None)
    moe_codebook_util_frac: Optional[float] = field(default=None)


# ---------------------------------------------------------------------------
# CoralV3Inner
# ---------------------------------------------------------------------------


class CoralV3Inner(CoralInner):
    """CORAL inner model — extends CoralInner with Phase 1 and Phase 3b mechanisms."""

    def __init__(self, config: CoralConfig) -> None:
        super().__init__(config)

        self._crystal_bootstrap_active: bool = config.crystal_bootstrap_steps > 0

        if config.use_predictive_coding:
            dim = config.hidden_size
            self.prediction_net = PredictionNet(h_dim=dim, l_dim=dim)
            self.precision_net = PrecisionNet(dim=dim)

        if config.use_columnar_routing:
            block_cfg = TransformerBlockConfig(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                expansion=config.expansion,
                rms_norm_eps=config.rms_norm_eps,
            )
            self.H_level = ColumnarReasoningModule(
                config=block_cfg,
                num_layers=config.H_layers,
                S=config.num_columns,
                k=config.active_columns,
            )
            self.L_level = ColumnarReasoningModule(
                config=block_cfg,
                num_layers=config.L_layers,
                S=config.num_columns,
                k=config.active_columns,
            )

        if config.use_crystallization:
            self.moe_codebook = SpatialMoECodebook(config, seq_len=self.total_seq_len)
            self.moe_codebook.bootstrap_mask_router(self._crystal_bootstrap_active)
            self.crystal_buffer = CrystallizationBuffer(
                capacity=config.crystal_buffer_capacity,
            )

    def _apply_moe_mixing(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        z_L_rec: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.config.use_crystallization or self._crystal_bootstrap_active:
            return z_L_rec, None, None

        w, z_bypass, _key = self.moe_codebook(z_H, z_L)
        w_pt = w[:, -1]
        z_L_out = w_pt[:, None, None] * z_L_rec + (1.0 - w_pt[:, None, None]) * z_bypass
        return z_L_out, w, z_bypass

    def _compute_moe_losses(
        self,
        z_L_final: torch.Tensor,
        w: Optional[torch.Tensor],
        z_bypass: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.config.use_crystallization:
            return None, None
        if w is None or z_bypass is None:
            return None, None
        L_recon, L_lb = self.moe_codebook.moe_losses(z_L_final, w, z_bypass)
        if not self.training:
            return L_recon.detach(), None
        return L_recon, L_lb

    @torch.compiler.disable(recursive=False)
    def _maybe_record_crystal(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        is_last_h: bool = False,
        is_last_segment: bool = False,
    ) -> None:
        if not is_last_h or not is_last_segment:
            return
        if not self.config.use_crystallization or not self.training:
            return
        if not self._crystal_bootstrap_active:
            return

        h_pool = self.moe_codebook.proj_h(z_H).mean(dim=1)
        l_pool = self.moe_codebook.proj_l(z_L).mean(dim=1)
        key = torch.cat([h_pool, l_pool], dim=-1)
        pooled_z_L = z_L.mean(dim=1)
        self.crystal_buffer.add(key, pooled_z_L, z_L_spatial=z_L)

    def consolidate_codebook(self, is_first_consolidation: bool = False) -> Optional[int]:
        if not self.config.use_crystallization:
            return None

        k_modes = self.config.moe_num_modes

        if is_first_consolidation and self.crystal_buffer.size < int(
            0.8 * self.crystal_buffer.capacity
        ):
            return None

        result = self.crystal_buffer.consolidate_spatial(k_modes)
        if result is None:
            return None

        centroids, utilization = result
        device = self.moe_codebook.codebook_values.device
        dtype = self.moe_codebook.codebook_values.dtype
        self.moe_codebook.codebook_values.data = centroids.to(device=device, dtype=dtype)

        self.crystal_buffer.clear()

        self._crystal_bootstrap_active = False
        self.moe_codebook.bootstrap_mask_router(False)

        return utilization

    def forward(
        self,
        carry: InnerCarry,
        batch: Dict[str, torch.Tensor],
        is_last_segment: bool = False,
    ) -> Tuple:
        pc = self.config.use_predictive_coding
        cr = self.config.use_columnar_routing
        cry = self.config.use_crystallization

        if not pc and not cr and not cry:
            return super().forward(carry, batch)
        elif pc and not cr:
            return self._forward_with_pc(carry, batch, is_last_segment=is_last_segment)
        else:
            raise NotImplementedError(
                "columnar routing disabled pending Phase 2 redesign; "
                "use pc=True, cr=False for Phase 3b."
            )

    def _forward_with_pc(
        self,
        carry: InnerCarry,
        batch: Dict[str, torch.Tensor],
        is_last_segment: bool = False,
    ) -> Tuple[InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], PredMetrics]:
        """Inner forward with predictive coding and optional Soft MoE crystallization."""
        cos_sin = self._cos_sin()
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch.get("puzzle_identifiers"),
        )

        pred_error_norms: List[torch.Tensor] = []
        precision_means: List[torch.Tensor] = []

        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L
            xi: Optional[torch.Tensor] = None

            for h_step in range(self.config.H_cycles):
                for l_step in range(self.config.L_cycles):
                    is_last_l = (
                        h_step == self.config.H_cycles - 1
                        and l_step == self.config.L_cycles - 1
                    )
                    if not is_last_l:
                        mu_L = self.prediction_net(z_H)
                        z_L_rec = self.L_level(z_L, mu_L + input_embeddings, cos_sin=cos_sin)
                        z_L, _, _ = self._apply_moe_mixing(z_H, z_L, z_L_rec)
                        epsilon = z_L - mu_L
                        pi = self.precision_net(z_L)
                        xi = pi * epsilon
                        pred_error_norms.append(epsilon.norm(dim=-1).mean())
                        precision_means.append(pi.mean())

                if not (h_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, xi, cos_sin=cos_sin)  # type: ignore[arg-type]
                    self._maybe_record_crystal(
                        z_H, z_L,
                        is_last_h=(h_step == self.config.H_cycles - 2),
                        is_last_segment=is_last_segment,
                    )

        assert not z_H.requires_grad and not z_L.requires_grad

        mu_L = self.prediction_net(z_H)
        z_L_rec = self.L_level(z_L, mu_L + input_embeddings, cos_sin=cos_sin)
        z_L, w, z_bypass = self._apply_moe_mixing(z_H, z_L, z_L_rec)

        epsilon_final = z_L - mu_L
        pi_final = self.precision_net(z_L)
        xi = pi_final * epsilon_final

        moe_recon_loss, moe_lb_loss = self._compute_moe_losses(z_L, w, z_bypass)

        z_H = self.H_level(z_H, xi, cos_sin=cos_sin)

        pred_error_norms.append(epsilon_final.detach().norm(dim=-1).mean())
        precision_means.append(pi_final.detach().mean())

        moe_pt_weight: float = 1.0
        moe_routing_entropy: Optional[float] = None
        moe_codebook_util_frac: Optional[float] = None
        if w is not None:
            moe_pt_weight = float(w[:, -1].mean().item())
            with torch.no_grad():
                w_f = w.float()
                eps = 1e-10
                moe_routing_entropy = float(
                    -(w_f * torch.log(w_f + eps)).sum(dim=-1).mean().item()
                )
                K_modes = self.config.moe_num_modes
                w_cb_mean = w_f[:, :K_modes].mean(dim=0)
                moe_codebook_util_frac = float(
                    (w_cb_mean > 0.01).float().mean().item()
                )

        new_carry = InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        pred_metrics = PredMetrics(
            pred_error_norms=pred_error_norms,
            precision_means=precision_means,
            epsilon_final=epsilon_final,
            pi_final=pi_final,
            moe_recon_loss=moe_recon_loss,
            moe_lb_loss=moe_lb_loss,
            moe_passthrough_weight=moe_pt_weight,
            moe_routing_entropy=moe_routing_entropy,
            moe_codebook_util_frac=moe_codebook_util_frac,
        )
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), pred_metrics
