"""Smoke test for ported CORAL model code.

Verifies that the ported code:
  - Imports cleanly
  - Instantiates a minimal model without errors
  - Runs a forward pass on synthetic data (shape compatibility)
  - Produces output of correct shape with no NaNs
  - Supports backward pass with gradient flow

Does NOT verify model correctness — only that the port is clean.
"""

import torch
import pytest

from marifah.models.coral_base import CoralConfig, CoralInner, InnerCarry
from marifah.models.coral import CoralV3Inner, PredMetrics
from marifah.models.act import CoralACT, CoralV3ACT, ACTCarry
from marifah.training.losses import ACTLossHead, CoralV3LossHead


# Minimal config for fast instantiation
SMALL_CONFIG = dict(
    batch_size=2,
    seq_len=16,
    vocab_size=32,
    num_puzzle_identifiers=0,
    puzzle_emb_ndim=0,
    H_cycles=1,
    L_cycles=1,
    H_layers=1,
    L_layers=1,
    hidden_size=64,
    num_heads=4,
    expansion=2.0,
    halt_max_steps=2,
    halt_exploration_prob=0.0,
    forward_dtype="float32",
)


def make_batch(config_dict: dict) -> dict:
    B = config_dict["batch_size"]
    S = config_dict["seq_len"]
    V = config_dict["vocab_size"]
    inputs = torch.randint(0, V, (B, S))
    labels = torch.randint(0, V, (B, S))
    return {"inputs": inputs, "labels": labels}


class TestCoralInnerSmoke:
    def test_import(self):
        assert CoralInner is not None
        assert CoralV3Inner is not None

    def test_instantiate_base(self):
        cfg = CoralConfig(**SMALL_CONFIG)
        model = CoralInner(cfg)
        assert model is not None

    def test_instantiate_v3(self):
        cfg = CoralConfig(**{**SMALL_CONFIG, "use_predictive_coding": True})
        model = CoralV3Inner(cfg)
        assert model is not None

    def test_forward_base(self):
        cfg = CoralConfig(**SMALL_CONFIG)
        model = CoralInner(cfg)
        model.eval()

        B = cfg.batch_size
        batch = make_batch(SMALL_CONFIG)
        carry = InnerCarry(
            z_H=torch.zeros(B, cfg.seq_len, cfg.hidden_size),
            z_L=torch.zeros(B, cfg.seq_len, cfg.hidden_size),
        )

        new_carry, output, (q_halt, q_cont) = model(carry, batch)

        assert output.shape == (B, cfg.seq_len, cfg.vocab_size)
        assert q_halt.shape == (B,)
        assert q_cont.shape == (B,)
        assert not torch.isnan(output).any(), "NaN in forward output"
        assert not torch.isnan(q_halt).any(), "NaN in q_halt"

    def test_forward_v3_pc(self):
        cfg = CoralConfig(**{**SMALL_CONFIG, "use_predictive_coding": True})
        model = CoralV3Inner(cfg)
        model.eval()

        B = cfg.batch_size
        batch = make_batch({**SMALL_CONFIG, "use_predictive_coding": True})
        carry = InnerCarry(
            z_H=torch.zeros(B, cfg.seq_len, cfg.hidden_size),
            z_L=torch.zeros(B, cfg.seq_len, cfg.hidden_size),
        )

        result = model(carry, batch)
        new_carry, output, (q_halt, q_cont), pred_metrics = result

        assert output.shape == (B, cfg.seq_len, cfg.vocab_size)
        assert isinstance(pred_metrics, PredMetrics)
        assert not torch.isnan(output).any(), "NaN in PC forward output"


class TestACTSmoke:
    def test_coral_act_forward(self):
        cfg = CoralConfig(**SMALL_CONFIG)
        model = CoralACT(cfg)
        model.eval()

        batch = make_batch(SMALL_CONFIG)
        carry = model.initial_carry(batch)

        new_carry, outputs = model(carry=carry, batch=batch)
        logits = outputs["logits"]

        assert logits.shape == (cfg.batch_size, cfg.seq_len, cfg.vocab_size)
        assert not torch.isnan(logits).any(), "NaN in ACT logits"

    def test_coral_v3_act_forward(self):
        cfg_dict = {**SMALL_CONFIG, "use_predictive_coding": True}
        cfg = CoralConfig(**cfg_dict)
        model = CoralV3ACT(cfg)
        model.eval()

        batch = make_batch(cfg_dict)
        carry = model.initial_carry(batch)

        new_carry, outputs = model(carry=carry, batch=batch)
        logits = outputs["logits"]

        assert logits.shape == (cfg.batch_size, cfg.seq_len, cfg.vocab_size)
        assert not torch.isnan(logits).any(), "NaN in V3 ACT logits"

    def test_backward_pass(self):
        cfg = CoralConfig(**SMALL_CONFIG)
        model = CoralACT(cfg)
        loss_head = ACTLossHead(model, loss_type="softmax_cross_entropy")
        loss_head.train()

        batch = make_batch(SMALL_CONFIG)
        carry = loss_head.initial_carry(batch)

        new_carry, loss, metrics, _, all_halted = loss_head(
            carry=carry, batch=batch, return_keys=[]
        )

        assert loss.requires_grad or loss.item() >= 0
        loss.backward()

        grad_found = any(
            p.grad is not None
            for p in model.parameters()
        )
        assert grad_found, "No gradients flowed to model parameters"

    def test_no_nan_in_loss(self):
        cfg = CoralConfig(**SMALL_CONFIG)
        model = CoralACT(cfg)
        loss_head = ACTLossHead(model, loss_type="softmax_cross_entropy")
        loss_head.train()

        batch = make_batch(SMALL_CONFIG)
        carry = loss_head.initial_carry(batch)

        new_carry, loss, metrics, _, _ = loss_head(
            carry=carry, batch=batch, return_keys=[]
        )

        assert not torch.isnan(loss), f"NaN in loss: {loss}"
