"""Top-level HMSC module: composes G, R, P codebooks + composition + auxiliary heads.

Default Session 4 configuration (all committed, not tunable this session):
  K_G=64, K_R=16, K_P=16
  d_G=512, d_R=256, d_P=128
  d_output=512 (= d_model)
  num_regions=8
  composition_method="sum"
  p_discreteness="soft"
  lambda_G=lambda_R=lambda_P=0.0

Per-step crystallization: HMSC.forward() is called at every ACT step.
Auxiliary losses are accumulated across steps by the caller.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from marifah.models.hmsc.global_codebook import GlobalCodebook
from marifah.models.hmsc.regional_codebook import RegionalCodebook
from marifah.models.hmsc.perposition_codebook import PerPositionCodebook
from marifah.models.hmsc.composition import HMSCComposition
from marifah.models.hmsc.auxiliary_heads import (
    GlobalAuxHead,
    RegionalAuxHead,
    PerPositionAuxHead,
    compute_aux_losses,
)

# Committed Session 4 defaults (per design §13.1 / session prompt §2.5)
_UTIL_THRESHOLD: float = 0.01   # entry considered "active" if weight > this


class HMSC(nn.Module):
    """Hierarchical Multi-Scale Codebook — the Marifah mechanism (Recognition Cortex)."""

    def __init__(
        self,
        K_G: int = 64,
        K_R: int = 16,
        K_P: int = 16,
        d_model: int = 512,
        d_G: int = 512,
        d_R: int = 256,
        d_P: int = 128,
        d_output: int = 512,
        num_workflow_types: int = 50,
        num_pattern_types: int = 12,
        num_primitives: int = 10,
        num_regions: int = 8,
        composition_method: str = "sum",
        lambda_G: float = 0.0,
        lambda_R: float = 0.0,
        lambda_P: float = 0.0,
        p_discreteness: str = "soft",
    ) -> None:
        super().__init__()
        self.lambda_G = lambda_G
        self.lambda_R = lambda_R
        self.lambda_P = lambda_P
        self.num_regions = num_regions
        self.K_G = K_G
        self.K_R = K_R
        self.K_P = K_P

        self.global_cb = GlobalCodebook(K_G=K_G, d_model=d_model, d_G=d_G)
        self.regional_cb = RegionalCodebook(K_R=K_R, d_model=d_model, d_R=d_R, num_regions=num_regions)
        self.perpos_cb = PerPositionCodebook(
            K_P=K_P, d_model=d_model, d_P=d_P, discreteness=p_discreteness
        )
        self.composition = HMSCComposition(
            d_G=d_G, d_R=d_R, d_P=d_P, d_output=d_output, method=composition_method
        )

        self.g_head = GlobalAuxHead(K_G=K_G, num_workflow_types=num_workflow_types)
        self.r_head = RegionalAuxHead(K_R=K_R, num_regions=num_regions, num_pattern_types=num_pattern_types)
        self.p_head = PerPositionAuxHead(K_P=K_P, num_primitives=num_primitives)

    def _utilization_stats(
        self,
        g_weights: torch.Tensor,    # (B, K_G) softmax
        r_weights: torch.Tensor,    # (B, num_regions, K_R) softmax
        p_weights: torch.Tensor,    # (B, N, K_P) softmax
        node_mask: torch.Tensor,    # (B, N)
    ) -> Dict[str, float]:
        """Compute per-scale utilization: active entries, entropy, top-1 dominance."""
        stats: Dict[str, float] = {}
        eps = 1e-10

        def _stats(w: torch.Tensor, name: str) -> None:
            # w: (..., K) flattened to (M, K)
            flat = w.reshape(-1, w.shape[-1]).float()
            active = (flat > _UTIL_THRESHOLD).float().mean(dim=0)  # (K,)
            stats[f"{name}_active_frac"] = float(active.mean().item())
            stats[f"{name}_active_count"] = float((active > 0).sum().item())
            entropy = -(flat * torch.log(flat + eps)).sum(dim=-1).mean()
            stats[f"{name}_entropy"] = float(entropy.item())
            top1_dom = (flat.max(dim=-1).values > 0.5).float().mean()
            stats[f"{name}_top1_dominance"] = float(top1_dom.item())

        _stats(g_weights, "G")
        _stats(r_weights, "R")

        # P: only over real nodes
        mask = node_mask.bool()
        if mask.any():
            p_real = p_weights[mask]    # (real_nodes, K_P)
            _stats(p_real, "P")
        else:
            _stats(p_weights.reshape(-1, self.K_P), "P")

        return stats

    def forward(
        self,
        carry_state: torch.Tensor,                          # (B, N, d_model)
        node_mask: torch.Tensor,                            # (B, N)
        workflow_labels: Optional[torch.Tensor] = None,    # (B,)
        region_labels: Optional[torch.Tensor] = None,      # (B, num_regions)
        primitive_labels: Optional[torch.Tensor] = None,   # (B, N)
    ) -> Dict:
        """
        Returns dict with:
            'composed': (B, N, d_output) — augmented carry, passed downstream
            'g_logits': (B, K_G) raw G routing logits
            'r_logits': (B, num_regions, K_R) raw R routing logits
            'p_attention': (B, N, K_P) P-codebook attention weights
            'aux_losses': dict from compute_aux_losses (zeros if no labels / lambda=0)
            'codebook_utilization': per-scale utilization stats
        """
        B, N, D = carry_state.shape

        # --- Three-scale forward ---
        G_mode, g_logits = self.global_cb(carry_state, node_mask)       # (B, N, d_G), (B, K_G)
        R_mode, r_logits = self.regional_cb(carry_state, node_mask)     # (B, N, d_R), (B, R, K_R)
        P_mode, p_attn = self.perpos_cb(carry_state, node_mask)         # (B, N, d_P), (B, N, K_P)

        # --- Composition ---
        composed = self.composition(G_mode, R_mode, P_mode, carry_state=carry_state)  # (B, N, d_output)

        # --- Auxiliary losses (computed always; lambda=0 → zero gradient) ---
        with torch.no_grad():
            g_weights = torch.softmax(g_logits, dim=-1)   # for utilization only
            r_weights = torch.softmax(r_logits, dim=-1)

        aux_losses = compute_aux_losses(
            g_logits=g_logits,
            r_logits=r_logits,
            p_attn=p_attn,
            g_head=self.g_head,
            r_head=self.r_head,
            p_head=self.p_head,
            workflow_labels=workflow_labels,
            region_labels=region_labels,
            primitive_labels=primitive_labels,
            node_mask=node_mask,
            lambda_G=self.lambda_G,
            lambda_R=self.lambda_R,
            lambda_P=self.lambda_P,
        )

        # --- Utilization stats (no grad, for logging) ---
        with torch.no_grad():
            util_stats = self._utilization_stats(g_weights, r_weights, p_attn, node_mask)

        return {
            "composed": composed,
            "g_logits": g_logits,
            "r_logits": r_logits,
            "p_attention": p_attn,
            "aux_losses": aux_losses,
            "codebook_utilization": util_stats,
        }
