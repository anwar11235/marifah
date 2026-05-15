"""Unit tests for HMSCComposition."""

import torch
import pytest

from marifah.models.hmsc.composition import HMSCComposition


def make_comp(**kw):
    defaults = dict(d_G=16, d_R=8, d_P=4, d_output=16, method="sum")
    defaults.update(kw)
    return HMSCComposition(**defaults)


def make_modes(B=2, N=5, d_G=16, d_R=8, d_P=4):
    return (
        torch.randn(B, N, d_G),
        torch.randn(B, N, d_R),
        torch.randn(B, N, d_P),
    )


class TestCompositionShape:
    def test_sum_output_shape(self):
        comp = make_comp()
        G, R, P = make_modes()
        out = comp(G, R, P)
        assert out.shape == (2, 5, 16)

    def test_gated_output_shape(self):
        comp = make_comp(method="gated")
        G, R, P = make_modes()
        carry = torch.randn(2, 5, 16)
        out = comp(G, R, P, carry_state=carry)
        assert out.shape == (2, 5, 16)


class TestCompositionSum:
    def test_sum_is_additive(self):
        """Sum output should equal proj_G(G) + proj_R(R) + proj_P(P)."""
        comp = make_comp()
        G, R, P = make_modes()
        with torch.no_grad():
            out = comp(G, R, P)
            expected = comp.proj_G(G) + comp.proj_R(R) + comp.proj_P(P)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_gradient_flows_all_projections(self):
        comp = make_comp()
        G, R, P = make_modes()
        out = comp(G, R, P)
        loss = out.sum()
        loss.backward()
        assert comp.proj_G.weight.grad is not None
        assert comp.proj_R.weight.grad is not None
        assert comp.proj_P.weight.grad is not None
        assert torch.isfinite(comp.proj_G.weight.grad).all()
        assert torch.isfinite(comp.proj_R.weight.grad).all()
        assert torch.isfinite(comp.proj_P.weight.grad).all()

    def test_finite_output(self):
        comp = make_comp()
        G, R, P = make_modes()
        with torch.no_grad():
            out = comp(G, R, P)
        assert torch.isfinite(out).all()


class TestCompositionGated:
    def test_gate_weights_positive_sum_to_one(self):
        comp = make_comp(method="gated")
        G, R, P = make_modes()
        carry = torch.randn(2, 5, 16)
        with torch.no_grad():
            # Inspect gate weights directly
            gate_logits = comp.gate_net(carry.float())
            import torch.nn.functional as F
            alphas = F.softmax(gate_logits, dim=-1)
        assert (alphas >= 0).all(), "Gate weights should be non-negative"
        sums = alphas.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            f"Gate weights should sum to 1: {sums}"

    def test_gated_gradient_flows(self):
        comp = make_comp(method="gated")
        G, R, P = make_modes()
        carry = torch.randn(2, 5, 16)
        out = comp(G, R, P, carry_state=carry)
        loss = out.sum()
        loss.backward()
        assert comp.proj_G.weight.grad is not None
        assert comp.proj_R.weight.grad is not None
        assert comp.proj_P.weight.grad is not None

    def test_gated_requires_carry(self):
        comp = make_comp(method="gated")
        G, R, P = make_modes()
        with pytest.raises(AssertionError, match="carry_state required"):
            comp(G, R, P)

    def test_invalid_method(self):
        with pytest.raises(AssertionError, match="method must be"):
            HMSCComposition(d_G=8, d_R=4, d_P=2, d_output=8, method="invalid")
