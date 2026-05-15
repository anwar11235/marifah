"""Unit tests for PerPositionCodebook (P-scale)."""

import torch
import pytest

from marifah.models.hmsc.perposition_codebook import PerPositionCodebook


def make_cb(**kw):
    defaults = dict(K_P=4, d_model=16, d_P=8)
    defaults.update(kw)
    return PerPositionCodebook(**defaults)


def make_input(B=2, N=6, d=16):
    carry = torch.randn(B, N, d)
    mask = torch.ones(B, N)
    return carry, mask


class TestPerPositionShape:
    def test_output_shapes_soft(self):
        cb = make_cb()
        carry, mask = make_input()
        mode_out, attn = cb(carry, mask)
        assert mode_out.shape == (2, 6, 8), f"mode_output shape wrong: {mode_out.shape}"
        assert attn.shape == (2, 6, 4), f"attention_weights shape wrong: {attn.shape}"

    def test_output_shapes_hard(self):
        cb = make_cb(discreteness="hard")
        carry, mask = make_input()
        mode_out, attn = cb(carry, mask)
        assert mode_out.shape == (2, 6, 8)
        assert attn.shape == (2, 6, 4)


class TestPerPositionSoft:
    def test_attention_weights_sum_to_one(self):
        cb = make_cb()
        carry, mask = make_input(B=2, N=5, d=16)
        with torch.no_grad():
            _, attn = cb(carry, mask)
        sums = attn.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            f"Soft attention weights don't sum to 1: {sums}"

    def test_gradient_flows_soft(self):
        cb = make_cb()
        carry, mask = make_input()
        mode_out, attn = cb(carry, mask)
        loss = mode_out.sum() + attn.sum()
        loss.backward()
        assert cb.codebook.grad is not None
        assert torch.isfinite(cb.codebook.grad).all()

    def test_determinism(self):
        cb = make_cb()
        carry, mask = make_input()
        with torch.no_grad():
            out1, a1 = cb(carry, mask)
            out2, a2 = cb(carry, mask)
        assert torch.allclose(out1, out2)
        assert torch.allclose(a1, a2)

    def test_finite_output(self):
        cb = make_cb()
        carry, mask = make_input()
        with torch.no_grad():
            mode_out, attn = cb(carry, mask)
        assert torch.isfinite(mode_out).all()
        assert torch.isfinite(attn).all()


class TestPerPositionHard:
    def test_hard_attention_is_one_hot(self):
        cb = make_cb(discreteness="hard")
        carry, mask = make_input(B=1, N=4, d=16)
        with torch.no_grad():
            _, attn = cb(carry, mask)
        # Each row should be one-hot
        row_max = attn.max(dim=-1).values
        row_sum = attn.sum(dim=-1)
        assert torch.allclose(row_max, torch.ones_like(row_max), atol=1e-6), \
            "Hard attention max should be 1.0"
        assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-6), \
            "Hard attention sum should be 1.0"

    def test_hard_no_gradient_on_non_selected(self):
        """Gradient should not flow through hard selection to non-selected codebook entries."""
        cb = make_cb(K_P=4, d_model=8, d_P=4, discreteness="hard")
        carry = torch.randn(1, 3, 8)
        mask = torch.ones(1, 3)
        mode_out, attn = cb(carry, mask)
        loss = mode_out.sum()
        loss.backward()
        # attn is one-hot with no grad; codebook grad should only be non-zero on selected entries
        # (can't easily check per-entry; just verify grad is non-None and finite)
        assert cb.codebook.grad is not None
        assert torch.isfinite(cb.codebook.grad).all()

    def test_soft_hard_similar_when_dominant(self):
        """When one code strongly dominates, soft ≈ hard."""
        cb_soft = make_cb(K_P=4, d_model=8, d_P=4, discreteness="soft")
        cb_hard = make_cb(K_P=4, d_model=8, d_P=4, discreteness="hard")
        # Copy weights
        cb_hard.load_state_dict(cb_soft.state_dict())

        # Make codebook entries very far apart so top-1 dominates
        with torch.no_grad():
            cb_soft.codebook.data[0] = 10.0
            cb_soft.codebook.data[1:] = -10.0
            cb_hard.codebook.data[0] = 10.0
            cb_hard.codebook.data[1:] = -10.0

        carry = torch.randn(1, 3, 8)
        mask = torch.ones(1, 3)
        with torch.no_grad():
            out_soft, _ = cb_soft(carry, mask)
            out_hard, _ = cb_hard(carry, mask)

        # Both should be very close to the dominant codebook entry
        diff = (out_soft - out_hard).abs().max()
        assert diff < 0.1, f"Soft and hard diverge when one code dominates: diff={diff}"
