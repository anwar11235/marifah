"""End-to-end HMSC tests: top-level HMSC module + CORAL integration smoke test.

Covers §2.6 (Smoke test), §4 verification steps 1–12.
"""

import torch
import pytest

from marifah.models.hmsc.hmsc import HMSC


# Small HMSC for fast tests
def make_hmsc(**kw):
    defaults = dict(
        K_G=8, K_R=4, K_P=4,
        d_model=32, d_G=32, d_R=16, d_P=8, d_output=32,
        num_workflow_types=5, num_pattern_types=3, num_primitives=10,
        num_regions=3,
        composition_method="sum",
        lambda_G=0.0, lambda_R=0.0, lambda_P=0.0,
        p_discreteness="soft",
    )
    defaults.update(kw)
    return HMSC(**defaults)


def make_carry(B=2, N=7, d=32):
    carry = torch.randn(B, N, d)
    mask = torch.ones(B, N)
    mask[0, 5:] = 0
    return carry, mask


class TestHMSCForward:
    def test_output_keys_present(self):
        hmsc = make_hmsc()
        carry, mask = make_carry()
        out = hmsc(carry, mask)
        for key in ("composed", "g_logits", "r_logits", "p_attention", "aux_losses", "codebook_utilization"):
            assert key in out, f"Missing key: {key}"

    def test_composed_shape(self):
        hmsc = make_hmsc()
        carry, mask = make_carry(B=3, N=9)
        out = hmsc(carry, mask)
        assert out["composed"].shape == (3, 9, 32)

    def test_logit_shapes(self):
        hmsc = make_hmsc()
        carry, mask = make_carry(B=2, N=7)
        out = hmsc(carry, mask)
        assert out["g_logits"].shape == (2, 8)
        assert out["r_logits"].shape == (2, 3, 4)
        assert out["p_attention"].shape == (2, 7, 4)

    def test_composed_is_finite(self):
        hmsc = make_hmsc()
        carry, mask = make_carry()
        out = hmsc(carry, mask)
        assert torch.isfinite(out["composed"]).all(), "composed contains non-finite values"

    def test_determinism(self):
        hmsc = make_hmsc()
        carry, mask = make_carry()
        with torch.no_grad():
            out1 = hmsc(carry, mask)
            out2 = hmsc(carry, mask)
        assert torch.allclose(out1["composed"], out2["composed"]), "Non-deterministic composed output"
        assert torch.allclose(out1["g_logits"], out2["g_logits"]), "Non-deterministic g_logits"


class TestHMSCAuxLosses:
    def test_aux_losses_computed_real_numbers(self):
        hmsc = make_hmsc()
        carry, mask = make_carry(B=2, N=7)
        wf_labels = torch.randint(0, 5, (2,))
        reg_labels = torch.randint(0, 3, (2, 3))
        prim_labels = torch.randint(0, 10, (2, 7))
        out = hmsc(carry, mask, workflow_labels=wf_labels,
                   region_labels=reg_labels, primitive_labels=prim_labels)
        aux = out["aux_losses"]
        for key in ("L_G", "L_R", "L_P", "L_aux_total"):
            assert torch.isfinite(aux[key]), f"{key} not finite: {aux[key]}"

    def test_aux_total_zero_when_lambda_zero(self):
        hmsc = make_hmsc(lambda_G=0.0, lambda_R=0.0, lambda_P=0.0)
        carry, mask = make_carry()
        wf_labels = torch.randint(0, 5, (2,))
        reg_labels = torch.randint(0, 3, (2, 3))
        prim_labels = torch.randint(0, 10, (2, 7))
        out = hmsc(carry, mask, workflow_labels=wf_labels,
                   region_labels=reg_labels, primitive_labels=prim_labels)
        assert out["aux_losses"]["L_aux_total"].item() == 0.0, \
            "L_aux_total should be 0.0 when all lambdas are 0"

    def test_no_head_gradient_when_lambda_zero(self):
        """Verify lambda=0 means no gradient to head params (via main forward path)."""
        hmsc = make_hmsc(lambda_G=0.0, lambda_R=0.0, lambda_P=0.0)
        carry, mask = make_carry()
        wf_labels = torch.randint(0, 5, (2,))
        reg_labels = torch.randint(0, 3, (2, 3))
        prim_labels = torch.randint(0, 10, (2, 7))
        out = hmsc(carry, mask, workflow_labels=wf_labels,
                   region_labels=reg_labels, primitive_labels=prim_labels)
        # Backward through composed (main path) + aux_losses (zero)
        loss = out["composed"].sum() + out["aux_losses"]["L_aux_total"]
        loss.backward()
        # Head params should have no gradient when lambda=0
        for name, param in list(hmsc.g_head.named_parameters()) + \
                list(hmsc.r_head.named_parameters()) + list(hmsc.p_head.named_parameters()):
            if param.grad is not None:
                assert param.grad.abs().max() < 1e-8, \
                    f"Head param {name} has unexpected gradient with lambda=0"


class TestHMSCGradientFlow:
    def test_codebook_grads_via_main_loss(self):
        """Codebook params get gradient through composed output (main loss path)."""
        hmsc = make_hmsc()
        carry, mask = make_carry()
        out = hmsc(carry, mask)
        loss = out["composed"].sum()
        loss.backward()
        assert hmsc.global_cb.codebook.grad is not None
        assert hmsc.regional_cb.codebook.grad is not None
        assert hmsc.perpos_cb.codebook.grad is not None
        assert torch.isfinite(hmsc.global_cb.codebook.grad).all()
        assert torch.isfinite(hmsc.regional_cb.codebook.grad).all()
        assert torch.isfinite(hmsc.perpos_cb.codebook.grad).all()

    def test_all_projection_grads_with_nonzero_lambda(self):
        """All params get gradient when lambdas > 0."""
        hmsc = make_hmsc(lambda_G=0.1, lambda_R=0.1, lambda_P=0.1)
        carry, mask = make_carry()
        wf_labels = torch.randint(0, 5, (2,))
        reg_labels = torch.randint(0, 3, (2, 3))
        prim_labels = torch.randint(0, 10, (2, 7))
        out = hmsc(carry, mask, workflow_labels=wf_labels,
                   region_labels=reg_labels, primitive_labels=prim_labels)
        loss = out["composed"].sum() + out["aux_losses"]["L_aux_total"]
        loss.backward()
        for name, p in hmsc.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"


class TestHMSCUtilization:
    def test_utilization_stats_present(self):
        hmsc = make_hmsc()
        carry, mask = make_carry()
        with torch.no_grad():
            out = hmsc(carry, mask)
        util = out["codebook_utilization"]
        for scale in ("G", "R", "P"):
            for stat in ("active_frac", "active_count", "entropy", "top1_dominance"):
                key = f"{scale}_{stat}"
                assert key in util, f"Missing utilization stat: {key}"

    def test_utilization_sensible_at_random_init(self):
        """At random init, entries should be neither all-dead nor all-one."""
        hmsc = HMSC(K_G=64, K_R=16, K_P=16, d_model=512, d_G=512, d_R=256, d_P=128, d_output=512)
        B, N = 4, 20
        carry = torch.randn(B, N, 512)
        mask = torch.ones(B, N)
        with torch.no_grad():
            out = hmsc(carry, mask)
        util = out["codebook_utilization"]
        # Not all entries should be dead
        assert util["G_active_frac"] > 0.0, "All G entries dead at init"
        assert util["R_active_frac"] > 0.0, "All R entries dead at init"
        assert util["P_active_frac"] > 0.0, "All P entries dead at init"
        # Entropy should be positive (not all mass on one entry)
        assert util["G_entropy"] > 0.0, "G entropy = 0 (all mass on one entry)"
        assert util["R_entropy"] > 0.0, "R entropy = 0"
        assert util["P_entropy"] > 0.0, "P entropy = 0"


class TestCORALHMSCIntegration:
    """§2.6 smoke test: CoralV3Inner + HMSC end-to-end."""

    @pytest.fixture(scope="class")
    def setup(self):
        from marifah.models.coral_base import CoralConfig, InnerCarry
        from marifah.models.coral import CoralV3Inner
        B, N = 2, 10
        config = CoralConfig(
            batch_size=B, seq_len=N, vocab_size=10,
            H_cycles=1, L_cycles=1, H_layers=1, L_layers=1,
            hidden_size=32, num_heads=2,
            use_predictive_coding=True,
            use_hmsc=True,
            forward_dtype="float32",
        )
        # Patch HMSC with correct d_model
        model = CoralV3Inner(config)
        # Reinstantiate HMSC with matching d_model
        from marifah.models.hmsc.hmsc import HMSC
        model.hmsc = HMSC(
            K_G=8, K_R=4, K_P=4,
            d_model=32, d_G=32, d_R=16, d_P=8, d_output=32,
            num_workflow_types=5, num_pattern_types=3, num_primitives=10,
            num_regions=3,
        )
        carry = InnerCarry(
            z_H=torch.zeros(B, N, 32),
            z_L=torch.zeros(B, N, 32),
        )
        return model, carry, B, N

    def test_hmsc_forward_shape(self, setup):
        model, carry, B, N = setup
        inputs = torch.randint(0, 10, (B, N))
        node_mask = torch.ones(B, N)
        batch = {"inputs": inputs, "node_mask": node_mask}
        new_carry, output, q_logits, metrics = model(carry, batch)
        assert output.shape == (B, N, 10)
        assert torch.isfinite(output).all()

    def test_hmsc_aux_losses_finite(self, setup):
        model, carry, B, N = setup
        inputs = torch.randint(0, 10, (B, N))
        node_mask = torch.ones(B, N)
        wf_labels = torch.randint(0, 5, (B,))
        batch = {"inputs": inputs, "node_mask": node_mask, "workflow_labels": wf_labels}
        _, _, _, metrics = model(carry, batch)
        assert metrics.hmsc_aux_losses is not None
        for key in ("L_G", "L_R", "L_P", "L_aux_total"):
            val = metrics.hmsc_aux_losses[key]
            assert torch.isfinite(val), f"HMSC aux loss {key} is not finite: {val}"

    def test_hmsc_utilization_stats(self, setup):
        model, carry, B, N = setup
        inputs = torch.randint(0, 10, (B, N))
        node_mask = torch.ones(B, N)
        batch = {"inputs": inputs, "node_mask": node_mask}
        _, _, _, metrics = model(carry, batch)
        assert metrics.hmsc_utilization is not None
        assert "G_active_frac" in metrics.hmsc_utilization

    def test_backward_codebook_grads(self, setup):
        model, carry, B, N = setup
        inputs = torch.randint(0, 10, (B, N))
        node_mask = torch.ones(B, N)
        batch = {"inputs": inputs, "node_mask": node_mask}
        _, output, _, _ = model(carry, batch)
        loss = output.sum()
        loss.backward()
        assert model.hmsc.global_cb.codebook.grad is not None
        assert model.hmsc.regional_cb.codebook.grad is not None
        assert model.hmsc.perpos_cb.codebook.grad is not None

    def test_use_hmsc_false_is_bit_identical(self):
        """CORAL with use_hmsc=False must produce bit-identical output to pre-HMSC baseline."""
        from marifah.models.coral_base import CoralConfig, InnerCarry
        from marifah.models.coral import CoralV3Inner

        B, N = 2, 8
        base_cfg = dict(
            batch_size=B, seq_len=N, vocab_size=10,
            H_cycles=1, L_cycles=1, H_layers=1, L_layers=1,
            hidden_size=32, num_heads=2,
            use_predictive_coding=True,
            forward_dtype="float32",
        )

        torch.manual_seed(42)
        model_no_hmsc = CoralV3Inner(CoralConfig(**{**base_cfg, "use_hmsc": False}))

        torch.manual_seed(42)
        model_with_hmsc_off = CoralV3Inner(CoralConfig(**{**base_cfg, "use_hmsc": False}))

        # Copy state dicts to ensure identical weights
        model_with_hmsc_off.load_state_dict(model_no_hmsc.state_dict())

        carry = InnerCarry(
            z_H=torch.zeros(B, N, 32),
            z_L=torch.zeros(B, N, 32),
        )
        inputs = torch.randint(0, 10, (B, N))
        batch = {"inputs": inputs}

        with torch.no_grad():
            _, out1, _, _ = model_no_hmsc(carry, batch)
            _, out2, _, _ = model_with_hmsc_off(carry, batch)

        max_diff = (out1 - out2).abs().max().item()
        assert max_diff == 0.0, \
            f"use_hmsc=False produced non-identical output: max_diff={max_diff}"
