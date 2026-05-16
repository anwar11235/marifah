"""Tests for checkpointing.py — remap, load guards, and save/load round-trips."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict

import pytest
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_model() -> nn.Module:
    """Minimal 1-layer H_level substrate for testing."""
    from marifah.models.coral_base import CoralConfig, CoralInner

    cfg = CoralConfig(
        batch_size=2,
        seq_len=8,
        vocab_size=10,
        H_cycles=1,
        L_cycles=1,
        H_layers=1,
        L_layers=1,
        hidden_size=32,
        num_heads=2,
    )
    return CoralInner(cfg)


def _save_state_dict(state_dict: Dict[str, torch.Tensor], path: str) -> None:
    """Save a bare state_dict (legacy format, no model_state_dict wrapper)."""
    torch.save(state_dict, path)


def _save_full_checkpoint(
    model: nn.Module, path: str, step: int = 0, epoch: int = 0
) -> None:
    from marifah.training.checkpointing import save_checkpoint

    class _FakeOpt:
        def state_dict(self):
            return {}

    save_checkpoint(
        path=path,
        model=model,
        optimizer=_FakeOpt(),  # type: ignore[arg-type]
        scheduler=None,
        step=step,
        epoch=epoch,
        config_dict={},
    )


# ---------------------------------------------------------------------------
# Remap tests
# ---------------------------------------------------------------------------


class TestRemapSudokuCheckpoint:
    def test_strips_model_inner_prefix(self):
        """Keys starting with model.inner. should have that prefix removed."""
        from marifah.training.checkpointing import remap_sudoku_checkpoint

        model = _tiny_model()
        source = {
            "model.inner.H_level.layers.0.self_attn.qkv_proj.weight": torch.zeros(
                model.state_dict()["H_level.layers.0.self_attn.qkv_proj.weight"].shape
            )
        }
        remapped, manifest = remap_sudoku_checkpoint(source, model)
        assert "H_level.layers.0.self_attn.qkv_proj.weight" in remapped
        assert len(manifest["loaded"]) == 1

    def test_strips_orig_mod_infix(self):
        """_orig_mod. segments (torch.compile artifacts) must be stripped."""
        from marifah.training.checkpointing import remap_sudoku_checkpoint

        model = _tiny_model()
        target_key = "H_level.layers.0.self_attn.qkv_proj.weight"
        src_shape = model.state_dict()[target_key].shape

        # Simulate: model.inner.H_level._orig_mod.layers.0.self_attn.qkv_proj.weight
        source = {
            "model.inner.H_level._orig_mod.layers.0.self_attn.qkv_proj.weight": torch.zeros(src_shape)
        }
        remapped, manifest = remap_sudoku_checkpoint(source, model)
        assert target_key in remapped
        assert len(manifest["loaded"]) == 1
        assert len(manifest["dropped_not_in_target"]) == 0

    def test_filters_shape_mismatch(self, caplog):
        """Keys with wrong shape must be filtered and logged as WARNING."""
        import logging
        from marifah.training.checkpointing import remap_sudoku_checkpoint

        model = _tiny_model()
        target_key = "H_level.layers.0.self_attn.qkv_proj.weight"
        wrong_shape = (999, 999)

        source = {
            "model.inner." + target_key: torch.zeros(wrong_shape),
        }
        with caplog.at_level(logging.WARNING, logger="marifah.training.checkpointing"):
            remapped, manifest = remap_sudoku_checkpoint(source, model)

        assert target_key not in remapped
        assert len(manifest["dropped_shape_mismatch"]) == 1
        orig_key, src_shape, tgt_shape = manifest["dropped_shape_mismatch"][0]
        assert src_shape == wrong_shape

    def test_filters_keys_not_in_target(self):
        """Keys that don't exist in the target model must be filtered out."""
        from marifah.training.checkpointing import remap_sudoku_checkpoint

        model = _tiny_model()
        source = {
            "prediction_net.fc.weight": torch.zeros(16, 16),
            "precision_net.linear.bias": torch.zeros(16),
        }
        remapped, manifest = remap_sudoku_checkpoint(source, model)
        assert len(remapped) == 0
        assert len(manifest["dropped_not_in_target"]) == 2
        assert len(manifest["loaded"]) == 0

    def test_manifest_loaded_entries_are_correct(self):
        """manifest['loaded'] must list (original_key, remapped_key) pairs."""
        from marifah.training.checkpointing import remap_sudoku_checkpoint

        model = _tiny_model()
        target_key = "H_level.layers.0.self_attn.qkv_proj.weight"
        orig_key = "model.inner." + target_key

        source = {orig_key: torch.zeros(model.state_dict()[target_key].shape)}
        _, manifest = remap_sudoku_checkpoint(source, model)

        assert manifest["loaded"] == [(orig_key, target_key)]


# ---------------------------------------------------------------------------
# load_checkpoint guard tests
# ---------------------------------------------------------------------------


class TestLoadCheckpointGuards:
    def test_missing_unexpected_keys_are_logged(self, caplog, tmp_path):
        """When all keys match, no missing/unexpected warnings should appear."""
        import logging
        from marifah.training.checkpointing import load_checkpoint

        model_a = _tiny_model()
        model_b = _tiny_model()   # different random init → params WILL change on load
        ckpt = str(tmp_path / "match.pt")
        _save_full_checkpoint(model_a, ckpt)
        with caplog.at_level(logging.WARNING, logger="marifah.training.checkpointing"):
            load_checkpoint(ckpt, model_b)
        warning_texts = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("missing_keys" in t or "unexpected_keys" in t for t in warning_texts)

    def test_load_checkpoint_raises_on_zero_param_change(self, tmp_path):
        """Loading a checkpoint with no matching keys must raise RuntimeError."""
        from marifah.training.checkpointing import load_checkpoint

        model = _tiny_model()
        # A state_dict with keys that don't exist in the model (no substrate keys)
        foreign_state = {
            "prediction_net.fc.weight": torch.randn(16, 16),
            "moe_codebook.centroids": torch.randn(64, 32),
        }
        ckpt = str(tmp_path / "foreign.pt")
        _save_state_dict(foreign_state, ckpt)

        with pytest.raises(RuntimeError, match="not updated"):
            load_checkpoint(ckpt, model)

    def test_load_checkpoint_succeeds_when_keys_match(self, tmp_path):
        """A checkpoint with matching keys must load without error."""
        from marifah.training.checkpointing import load_checkpoint

        model_a = _tiny_model()
        model_b = _tiny_model()

        ckpt = str(tmp_path / "good.pt")
        _save_full_checkpoint(model_a, ckpt, step=10, epoch=1)

        meta = load_checkpoint(ckpt, model_b)
        assert meta["step"] == 10
        assert meta["epoch"] == 1

    def test_remap_sudoku_to_marifah_integration(self, tmp_path):
        """With remap, a synthetic 'Sudoku-style' checkpoint should load substrate params.

        Simulates the Sudoku key structure (model.inner.* + _orig_mod.) without
        needing the real checkpoint file.
        """
        from marifah.training.checkpointing import load_checkpoint

        model = _tiny_model()
        sd = model.state_dict()

        # Build a fake "Sudoku" checkpoint: wrap all keys with model.inner._orig_mod.
        sudoku_state: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            # Insert both prefix and _orig_mod. in the path
            parts = k.split(".")
            # e.g. H_level.layers.0.X → model.inner.H_level._orig_mod.layers.0.X
            sudoku_key = "model.inner." + parts[0] + "._orig_mod." + ".".join(parts[1:])
            sudoku_state[sudoku_key] = v.clone() + 1.0  # offset so they differ from fresh init

        ckpt = str(tmp_path / "fake_sudoku.pt")
        torch.save(sudoku_state, ckpt)

        model_fresh = _tiny_model()
        load_checkpoint(ckpt, model_fresh, remap="sudoku_to_marifah")

        # Verify substrate weights loaded (offset by 1.0 applied)
        substrate_key = next(
            k for k in model_fresh.state_dict() if k.startswith("H_level") and k.endswith(".weight")
        )
        expected = sd[substrate_key] + 1.0
        assert torch.allclose(model_fresh.state_dict()[substrate_key].float(), expected.float())

    @pytest.mark.skipif(
        not Path("checkpoints/sudoku-phase3c/phase3c_canonical_seed0_best.pt").exists(),
        reason="Sudoku Phase 3c checkpoint not available locally",
    )
    def test_load_real_sudoku_checkpoint_with_remap(self):
        """Full integration: load actual Sudoku Phase 3c ckpt with remap.

        Asserts that >= 4 H_level + >= 4 L_level layer weight tensors differ
        from fresh random init after the remap load.
        """
        from marifah.models.coral_base import CoralConfig, CoralInner
        from marifah.training.checkpointing import load_checkpoint

        cfg = CoralConfig(
            batch_size=2,
            seq_len=100,
            vocab_size=10,
            H_cycles=1,
            L_cycles=1,
            H_layers=4,
            L_layers=4,
            hidden_size=512,
            num_heads=8,
        )
        model_init = CoralInner(cfg)
        init_sd = {k: v.clone() for k, v in model_init.state_dict().items()}

        model_warm = CoralInner(cfg)
        load_checkpoint(
            "checkpoints/sudoku-phase3c/phase3c_canonical_seed0_best.pt",
            model_warm,
            remap="sudoku_to_marifah",
        )

        changed_H = sum(
            1 for i in range(4)
            for k, v in model_warm.state_dict().items()
            if f"H_level.layers.{i}" in k and "weight" in k
            and not torch.allclose(v.float(), init_sd[k].float())
        )
        changed_L = sum(
            1 for i in range(4)
            for k, v in model_warm.state_dict().items()
            if f"L_level.layers.{i}" in k and "weight" in k
            and not torch.allclose(v.float(), init_sd[k].float())
        )
        assert changed_H >= 4, f"Expected >= 4 changed H_level layers, got {changed_H}"
        assert changed_L >= 4, f"Expected >= 4 changed L_level layers, got {changed_L}"
