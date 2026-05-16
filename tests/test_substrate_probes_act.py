"""Tests for ACT-iterated probe mechanics.

Validates:
  1. In eval mode, CoralV3ACT runs exactly halt_max_steps iterations
  2. ACT-iterated z_H differs from single-step z_H on the same inputs
  3. All sequences are marked halted after halt_max_steps

All tests run on CPU with tiny models (no GPU, no checkpoint I/O).
Integration tests (real checkpoint + real data) are marked @pytest.mark.gpu.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# ---------------------------------------------------------------------------
# ACT iteration mechanics
# ---------------------------------------------------------------------------

class TestActIterationMechanics:
    """Verify ACT iterates correctly in eval mode."""

    def test_act_runs_halt_max_steps_in_eval_mode(self):
        """In eval mode, CoralV3ACT must run exactly halt_max_steps iterations."""
        from marifah.models.coral_base import CoralConfig
        from marifah.models.act import CoralV3ACT

        halt_max_steps = 4
        config = CoralConfig(
            batch_size=2, seq_len=8, vocab_size=4,
            H_cycles=1, L_cycles=1, H_layers=1, L_layers=1,
            hidden_size=32, num_heads=2,
            halt_max_steps=halt_max_steps,
            halt_exploration_prob=0.0,
            forward_dtype="float32",
        )
        model = CoralV3ACT(config)
        model.eval()

        batch = {"inputs": torch.randint(0, 4, (2, 8))}
        act_carry = model.initial_carry(batch)

        steps_run = 0
        with torch.no_grad():
            for _ in range(halt_max_steps + 2):  # extra iterations to confirm it stops
                act_carry, _ = model(act_carry, batch)
                steps_run += 1
                if act_carry.halted.all():
                    break

        assert steps_run == halt_max_steps, (
            f"Expected exactly {halt_max_steps} steps in eval mode, got {steps_run}. "
            "ACT should halt all sequences at is_last_step in eval mode."
        )

    def test_all_sequences_halted_after_halt_max_steps(self):
        """After halt_max_steps calls, all samples must be halted in eval mode."""
        from marifah.models.coral_base import CoralConfig
        from marifah.models.act import CoralV3ACT

        config = CoralConfig(
            batch_size=4, seq_len=6, vocab_size=4,
            H_cycles=1, L_cycles=1, H_layers=1, L_layers=1,
            hidden_size=16, num_heads=2,
            halt_max_steps=3,
            halt_exploration_prob=0.0,
            forward_dtype="float32",
        )
        model = CoralV3ACT(config)
        model.eval()

        batch = {"inputs": torch.randint(0, 4, (4, 6))}
        act_carry = model.initial_carry(batch)

        with torch.no_grad():
            for _ in range(config.halt_max_steps):
                act_carry, _ = model(act_carry, batch)

        assert act_carry.halted.all(), (
            f"All sequences should be halted after halt_max_steps={config.halt_max_steps} "
            f"in eval mode. halted={act_carry.halted}"
        )
        assert int(act_carry.steps.max().item()) == config.halt_max_steps, (
            f"steps should equal halt_max_steps. Got {act_carry.steps.max().item()}"
        )

    def test_act_z_H_differs_from_single_step_z_H(self):
        """ACT-iterated z_H must differ from single CoralV3Inner forward pass.

        ACT applies the inner model halt_max_steps times; single-step applies it once.
        Even with the same weights, the recurrent carry evolves — so outputs must differ.
        """
        from marifah.models.coral_base import CoralConfig, InnerCarry
        from marifah.models.act import CoralV3ACT  # noqa: F401
        from marifah.models.coral import CoralV3Inner

        torch.manual_seed(42)
        config = CoralConfig(
            batch_size=2, seq_len=8, vocab_size=4,
            H_cycles=1, L_cycles=1, H_layers=1, L_layers=1,
            hidden_size=32, num_heads=2,
            halt_max_steps=4,
            halt_exploration_prob=0.0,
            forward_dtype="float32",
        )
        B = 2

        # Build inner model (single-step)
        inner_model = CoralV3Inner(config)
        inner_model.eval()

        # Build ACT model with IDENTICAL weights
        act_model = CoralV3ACT(config)
        act_model.inner.load_state_dict(inner_model.state_dict())
        act_model.eval()

        batch = {"inputs": torch.randint(0, 4, (B, 8))}

        # Single-step z_H (use empty_carry from the model itself — CoralConfig uses seq_len/hidden_size,
        # not max_nodes/d_model, so InnerCarry.zeros with CoralConfig would fail)
        with torch.no_grad():
            carry_single = InnerCarry(
                z_H=torch.zeros(B, 8, 32, dtype=torch.float32),
                z_L=torch.zeros(B, 8, 32, dtype=torch.float32),
            )
            result_single = inner_model(carry_single, batch, is_last_segment=True)
            z_H_single = result_single[0].z_H  # (B, N, d)

        # ACT-iterated z_H
        with torch.no_grad():
            act_carry = act_model.initial_carry(batch)
            for _ in range(config.halt_max_steps):
                act_carry, _ = act_model(act_carry, batch)
                if act_carry.halted.all():
                    break
            z_H_act = act_carry.inner_carry.z_H  # (B, N, d)

        max_diff = (z_H_act - z_H_single).abs().max().item()
        assert max_diff > 1e-6, (
            f"ACT-iterated z_H should differ from single-step z_H (max diff={max_diff:.2e}). "
            "If zero, ACT is not applying recurrent refinement."
        )

    def test_act_steps_counter_increments_correctly(self):
        """ACT steps counter should go 1, 2, ..., halt_max_steps across calls."""
        from marifah.models.coral_base import CoralConfig
        from marifah.models.act import CoralV3ACT

        config = CoralConfig(
            batch_size=2, seq_len=4, vocab_size=4,
            H_cycles=1, L_cycles=1, H_layers=1, L_layers=1,
            hidden_size=16, num_heads=2,
            halt_max_steps=4,
            halt_exploration_prob=0.0,
            forward_dtype="float32",
        )
        model = CoralV3ACT(config)
        model.eval()

        batch = {"inputs": torch.randint(0, 4, (2, 4))}
        act_carry = model.initial_carry(batch)

        with torch.no_grad():
            for expected_step in range(1, config.halt_max_steps + 1):
                act_carry, _ = model(act_carry, batch)
                actual_steps = int(act_carry.steps.max().item())
                assert actual_steps == expected_step, (
                    f"After iteration {expected_step}, steps should be {expected_step}, got {actual_steps}"
                )
