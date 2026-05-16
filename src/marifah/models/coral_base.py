"""CORAL base inner model — faithful reproduction of HierarchicalReasoningModel_ACTV1_Inner.

Implements the core hierarchical reasoning computation without the ACT wrapper.
Architecture: H-level and L-level ReasoningModules, each with 4 TransformerBlocks.
Nested recurrence: H_cycles outer × L_cycles inner per segment.
1-step gradient: all but the final (L, H) step run under torch.no_grad().
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from marifah.utils.common import trunc_normal_init_
from marifah.models.layers import CastedEmbedding, CastedLinear, CosSin, RotaryEmbedding
from marifah.models.reasoning_module import ReasoningModule
from marifah.models.sparse_embedding import CastedSparseEmbedding
from marifah.models.transformer_block import TransformerBlock, TransformerBlockConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class CoralConfig(BaseModel):
    """Configuration for the CORAL base inner model."""

    batch_size: int
    seq_len: int

    vocab_size: int
    num_puzzle_identifiers: int = 0
    puzzle_emb_ndim: int = 0

    H_cycles: int = 2
    L_cycles: int = 2
    H_layers: int = 4
    L_layers: int = 4

    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4.0

    pos_encodings: str = "rope"
    rope_theta: float = 10000.0

    rms_norm_eps: float = 1e-5
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1

    forward_dtype: str = "bfloat16"

    # Phase 1: predictive coding
    use_predictive_coding: bool = False
    lambda_pred: float = 0.1
    lambda_pi: float = 0.01

    # Phase 2: sparse columnar routing
    use_columnar_routing: bool = False
    num_columns: int = 8
    active_columns: int = 2
    lambda_balance: float = 0.01

    # Phase 3: recognition-gated crystallization
    use_crystallization: bool = False
    codebook_size: int = 256
    crystal_proj_dim: int = 128
    crystal_buffer_capacity: int = 10000
    crystal_consolidation_interval: int = 10
    crystal_bootstrap_steps: int = 5000

    # Phase 3b: Soft MoE spatial codebook
    moe_num_modes: int = 32
    lambda_moe_recon: float = 0.1
    lambda_moe_balance: float = 0.01

    # Session 4: HMSC (Hierarchical Multi-Scale Codebook)
    use_hmsc: bool = False


# ---------------------------------------------------------------------------
# Carry
# ---------------------------------------------------------------------------


@dataclass
class InnerCarry:
    """Recurrent state passed between ACT segments. Both tensors are always detached."""

    z_H: torch.Tensor  # [B, total_seq_len, hidden_size]
    z_L: torch.Tensor  # [B, total_seq_len, hidden_size]

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        model_config: Any,
        device: torch.device,
        dtype_override: Optional[torch.dtype] = None,
    ) -> "InnerCarry":
        """Construct a zero-initialized carry with dtype from model_config.

        Single source of truth for carry dtype: reads model_config.forward_dtype
        so all call sites stay in sync when the config changes.

        Args:
            batch_size:    B dimension
            model_config:  ModelConfig with .forward_dtype, .max_nodes, .d_model
            device:        Target device
            dtype_override: If provided, use this dtype instead of model_config.forward_dtype.
                           Used for CPU fallback when bf16 is not fully supported.
        """
        dtype = dtype_override if dtype_override is not None else getattr(torch, model_config.forward_dtype)
        return cls(
            z_H=torch.zeros(batch_size, model_config.max_nodes, model_config.d_model, dtype=dtype, device=device),
            z_L=torch.zeros(batch_size, model_config.max_nodes, model_config.d_model, dtype=dtype, device=device),
        )


# ---------------------------------------------------------------------------
# Inner model
# ---------------------------------------------------------------------------


class CoralInner(nn.Module):
    """CORAL base inner loop (no ACT wrapper).

    One forward call runs H_cycles × L_cycles recurrent steps with 1-step gradient.
    Only the very last L-step and H-step are in the computation graph.
    """

    def __init__(self, config: CoralConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype: torch.dtype = getattr(torch, config.forward_dtype)

        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.puzzle_emb_len: int = -(config.puzzle_emb_ndim // -config.hidden_size) if config.puzzle_emb_ndim > 0 else 0
        self.total_seq_len: int = config.seq_len + self.puzzle_emb_len

        self.embed_tokens = CastedEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore[union-attr]

        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                num_embeddings=config.num_puzzle_identifiers,
                embedding_dim=config.puzzle_emb_ndim,
                batch_size=config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        block_cfg = TransformerBlockConfig(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expansion=config.expansion,
            rms_norm_eps=config.rms_norm_eps,
        )
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=self.total_seq_len,
                base=config.rope_theta,
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                num_embeddings=self.total_seq_len,
                embedding_dim=config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise ValueError(f"Unknown pos_encodings: {config.pos_encodings!r}")

        self.H_level = ReasoningModule(
            layers=[TransformerBlock(block_cfg) for _ in range(config.H_layers)]
        )
        self.L_level = ReasoningModule(
            layers=[TransformerBlock(block_cfg) for _ in range(config.L_layers)]
        )

        self.H_init: torch.Tensor
        self.L_init: torch.Tensor
        self.register_buffer(
            "H_init",
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1.0),
            persistent=True,
        )
        self.register_buffer(
            "L_init",
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1.0),
            persistent=True,
        )

    def _cos_sin(self) -> Optional[CosSin]:
        if hasattr(self, "rotary_emb"):
            return self.rotary_emb()
        return None

    def _input_embeddings(
        self,
        inputs: torch.Tensor,
        puzzle_identifiers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embedding = self.embed_tokens(inputs.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2,
            )

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
            )

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device: torch.device = None) -> InnerCarry:
        return InnerCarry(
            z_H=torch.empty(
                batch_size, self.total_seq_len, self.config.hidden_size,
                dtype=self.forward_dtype, device=device,
            ),
            z_L=torch.empty(
                batch_size, self.total_seq_len, self.config.hidden_size,
                dtype=self.forward_dtype, device=device,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: InnerCarry) -> InnerCarry:
        flag = reset_flag.view(-1, 1, 1)
        return InnerCarry(
            z_H=torch.where(flag, self.H_init, carry.z_H),
            z_L=torch.where(flag, self.L_init, carry.z_L),
        )

    def forward(
        self,
        carry: InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        cos_sin = self._cos_sin()
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch.get("puzzle_identifiers"),
        )

        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for h_step in range(self.config.H_cycles):
                for l_step in range(self.config.L_cycles):
                    is_last_l = (h_step == self.config.H_cycles - 1) and (l_step == self.config.L_cycles - 1)
                    if not is_last_l:
                        z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)

                if not (h_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        assert not z_H.requires_grad and not z_L.requires_grad

        z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
        z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        new_carry = InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])
