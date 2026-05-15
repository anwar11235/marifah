"""Graph-aware attention functions for the graph adapter.

Two callable attention paths are provided:

  sdpa_with_bias(q, k, v, attention_mask)
      Standard PyTorch SDPA with a (B, N, N) additive-bias mask.
      CPU and CUDA compatible.  Used as the fallback when flash-attn is
      unavailable or for debug/testing.

  flash_varlen(q_packed, k_packed, v_packed, cu_seqlens, max_seqlen)
      Flash-attention varlen kernel for packed variable-length graph sequences.
      CUDA-only; falls back to sdpa_with_bias on CPU.
      Uses flash_attn_interface (fa3) if available, then flash_attn (fa2).

Config flag: attention_backend ∈ {"sdpa", "flash_varlen"} controls which path
the GraphAttentionLayer uses.  Default: "flash_varlen" (with SDPA fallback).

Salvage report
--------------
Salvage 1 — arc/padding-attention-mask (c7e784d):
  Source: coral/models/layers.py Attention.forward(), SDPA branch.
  Ported: additive-bias mask construction (masked_fill -inf convention),
          SDPA call with attn_mask, layout handling (transpose [B,S,H,D]→[B,H,S,D]).
  Adapted: source used a 1D [B, S] padding mask; here the mask is a pre-built
           2D [B, N, N] edge-induced mask passed directly as attn_mask to SDPA.
  Not ported: RoPE, QKV projection, the non-masked flash path (no graph-specific
              assumptions needed there).

Salvage 2 — arc/flash-attn-varlen (28af53a):
  Source: coral/models/layers.py Attention.forward(), CUDA masked branch.
  Ported: _get_flash_attn_varlen_func() lazy importer, cu_seqlens computation,
          pack (boolean index → q_packed) / unpack (scatter back to full shape).
  Adapted: source packed based on a [B, S] boolean mask (ARC padding context);
           here the caller computes cu_seqlens from node counts directly.
           Graph boundary isolation (causal=False, full attention within each graph).
  Not ported: ARC grid-shape assumptions; ConceptARC references.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_FLASH_ATTN_BACKEND = "flash_varlen"    # default; callers may override


def _get_flash_attn_varlen_func():
    """Return flash_attn_varlen_func or None if unavailable (CUDA-only).

    Preference order:
      1. flash_attn_interface (fa3, Hopper GPUs)
      2. flash_attn           (fa2, Ampere and older)
      3. None — callers must fall back to SDPA on CPU.
    """
    try:
        from flash_attn_interface import flash_attn_varlen_func  # type: ignore[import]
        return flash_attn_varlen_func
    except ImportError:
        pass
    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore[import]
        return flash_attn_varlen_func
    except ImportError:
        pass
    return None


def sdpa_with_bias(
    q: torch.Tensor,           # (B, N, n_heads, head_dim) or (B, n_heads, N, head_dim)
    k: torch.Tensor,           # same layout as q
    v: torch.Tensor,           # same layout as q
    attention_mask: Optional[torch.Tensor] = None,  # (B, N, N) additive bias
    qkv_layout: str = "bshd",  # "bshd" = [B, S, H, D]; "bhsd" = [B, H, S, D]
) -> torch.Tensor:
    """SDPA with optional (B, N, N) additive-bias attention mask.

    Mask convention (from salvage 1, arc/padding-attention-mask c7e784d):
      0.0  = attention allowed
      -inf = attention blocked

    Returns tensor in the same qkv_layout as input.
    """
    if qkv_layout == "bshd":
        # Transpose to [B, H, S, D] for SDPA
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
    else:
        q_t, k_t, v_t = q, k, v

    if attention_mask is not None:
        # attention_mask: (B, N, N) — add a head dimension
        # SDPA expects attn_mask broadcastable to [B, H, N, N]
        bias = attention_mask.unsqueeze(1)  # (B, 1, N, N)
        bias = bias.to(q_t.dtype)
    else:
        bias = None

    out = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=bias, is_causal=False)

    if qkv_layout == "bshd":
        out = out.transpose(1, 2)  # back to [B, S, H, D]
    return out


def flash_varlen(
    q_packed: torch.Tensor,    # (total_nodes, n_heads, head_dim) — packed
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    cu_seqlens: torch.Tensor,  # (B+1,) int32 — cumulative node counts
    max_seqlen: int,
    causal: bool = False,
) -> torch.Tensor:
    """Flash-attention varlen kernel for packed variable-length graph sequences.

    Full attention within each graph (graph boundaries encoded in cu_seqlens).
    CUDA-only: falls back silently to SDPA-with-full-mask on CPU.

    Returns (total_nodes, n_heads, head_dim) packed output.

    Salvage 2 (arc/flash-attn-varlen 28af53a): cu_seqlens computation, pack/unpack
    pattern, and fa2/fa3 dispatch are direct ports.
    """
    fn = _get_flash_attn_varlen_func()
    if fn is not None and q_packed.is_cuda:
        out = fn(
            q=q_packed, k=k_packed, v=v_packed,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=causal,
        )
        if isinstance(out, tuple):  # fa3 returns (out, softmax_lse)
            out = out[0]
        return out

    # CPU fallback: reconstruct full [B, N, H, D] representation, run SDPA,
    # then repack.  Not efficient; used only for CPU smoke tests.
    B = cu_seqlens.shape[0] - 1
    n_heads = q_packed.shape[1]
    head_dim = q_packed.shape[2]

    # Unpack to [B, max_seqlen, H, D]
    q_full = torch.zeros(B, max_seqlen, n_heads, head_dim, dtype=q_packed.dtype, device=q_packed.device)
    k_full = torch.zeros_like(q_full)
    v_full = torch.zeros_like(q_full)
    for i in range(B):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        n = end - start
        q_full[i, :n] = q_packed[start:end]
        k_full[i, :n] = k_packed[start:end]
        v_full[i, :n] = v_packed[start:end]

    out_full = sdpa_with_bias(q_full, k_full, v_full, attention_mask=None)  # (B, N, H, D)

    # Repack
    out_packed = torch.zeros_like(q_packed)
    for i in range(B):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        n = end - start
        out_packed[start:end] = out_full[i, :n]

    return out_packed


# ---------------------------------------------------------------------------
# GraphAttentionLayer — single self-attention layer for graph nodes
# ---------------------------------------------------------------------------

class GraphAttentionLayer(nn.Module):
    """Multi-head self-attention layer operating on graph node sequences.

    Parameters
    ----------
    d_model:
        Input and output feature dimension.
    n_heads:
        Number of attention heads.
    attention_backend:
        "sdpa" — use SDPA with additive bias mask (default on CPU).
        "flash_varlen" — use flash-attn varlen path; falls back to SDPA on CPU.
    attention_direction:
        Determines how the attention_mask was built; informational only here.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        attention_backend: str = "flash_varlen",
        attention_direction: str = "directed",
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_backend = attention_backend
        self.attention_direction = attention_direction

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,                          # (B, N_max, d_model)
        attention_mask: Optional[torch.Tensor] = None,  # (B, N_max, N_max) additive bias
        node_counts: Optional[torch.Tensor] = None,     # (B,) int — actual node counts for varlen
    ) -> torch.Tensor:
        """Returns (B, N_max, d_model)."""
        B, N, _ = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0]   # (B, N, H, D)
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        use_varlen = (
            self.attention_backend == "flash_varlen"
            and node_counts is not None
            and x.is_cuda
            and _get_flash_attn_varlen_func() is not None
        )

        if use_varlen:
            # Pack valid nodes across the batch
            cu_seqlens = torch.cat([
                torch.zeros(1, dtype=torch.int32, device=x.device),
                node_counts.cumsum(0, dtype=torch.int32),
            ])
            total = int(cu_seqlens[-1].item())
            max_n = int(node_counts.max().item())

            # Flatten and index by valid positions
            flat_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
            for i, nc in enumerate(node_counts):
                flat_mask[i, :nc] = True

            q_p = q[flat_mask]  # (total, H, D)
            k_p = k[flat_mask]
            v_p = v[flat_mask]

            out_p = flash_varlen(q_p, k_p, v_p, cu_seqlens, max_n)

            out = torch.zeros(B, N, self.n_heads, self.head_dim, dtype=x.dtype, device=x.device)
            out[flat_mask] = out_p
        else:
            # SDPA path with additive bias mask
            out = sdpa_with_bias(q, k, v, attention_mask=attention_mask)  # (B, N, H, D)

        out = out.reshape(B, N, self.d_model)
        return self.out_proj(out)
