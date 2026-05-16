"""Regression test for carry dtype propagation through all forward-path call sites.

Bug class: carry tensors (z_H, z_L in InnerCarry) constructed with hardcoded
dtype=torch.float32 instead of reading config.model.forward_dtype. When
forward_dtype=bfloat16 (GPU runs), fp32 carry flows into flash_attn_func →
RuntimeError: FlashAttention only supports fp16 and bf16.

Rounds patched:
  Round 1 (c92cfcb): trainer.build_model, trainer.step
  Round 2 (07f8341-era): eval_loop.evaluate, warmstart_probe._extract_carry_states,
                         warmstart_probe.compute_execution_faithfulness

This test exercises ALL four carry-construction call sites in a single parametrized
run, asserting that every InnerCarry constructed during the forward pass has the
expected dtype.

Maintainer note — verifying this test has teeth:
  Before merging a carry-construction change, manually revert ONE call site to
  hardcoded float32 (e.g., in eval_loop.py revert `InnerCarry.zeros(...)` to
  the old `InnerCarry(z_H=torch.zeros(..., dtype=torch.float32, ...))`),
  then run:
      pytest tests/test_dtype_propagation.py -v
  The test MUST fail with an AssertionError showing the wrong dtype.
  Restore the site, re-run, confirm green. This was verified during development.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers — tiny dataset + config
# ---------------------------------------------------------------------------


def _generate_tiny_dataset(root: Path, n: int = 30) -> None:
    from marifah.data.synthetic.generator import DagGenerator
    from marifah.data.synthetic.storage import write_split
    from marifah.data.synthetic.vertical_config import GeneratorConfig, SplitSizes, _hash_config

    cfg = GeneratorConfig(seed=888)
    cfg.split_sizes = SplitSizes(
        train=n, val=n, test_id=5, test_ood_size=5, test_ood_composition=5
    )
    cfg.config_hash = _hash_config(cfg)
    gen = DagGenerator(cfg)
    for split, seed_off in [("train", 0), ("val", 10_000)]:
        recs = gen.generate_split(split, n, seed_offset=seed_off)
        write_split(recs, root, split)


@pytest.fixture(scope="module")
def tiny_ds_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("dtype_ds")
    _generate_tiny_dataset(root)
    return root


def _make_config(forward_dtype: str, dataset_root: str, ckpt_dir: str):
    from marifah.training.config import (
        TrainingConfig, ModelConfig, TrainingPhaseConfig,
        DataConfig, LoggingConfig, ExperimentConfig,
    )
    return TrainingConfig(
        experiment=ExperimentConfig(seed=0),
        model=ModelConfig(
            d_model=32,
            num_heads=2,
            H_cycles=1,
            L_cycles=1,
            H_layers=1,
            L_layers=1,
            max_nodes=32,
            vocab_size=10,
            forward_dtype=forward_dtype,
            halt_max_steps=2,
        ),
        training=TrainingPhaseConfig(
            batch_size=2,
            max_epochs=1,
            max_steps=2,
            eval_interval_epochs=1,
        ),
        data=DataConfig(dataset_root=dataset_root, num_workers=0, drop_last=False),
        logging=LoggingConfig(
            wandb_mode="disabled",
            checkpoint_dir=ckpt_dir,
            heartbeat_interval_steps=0,
        ),
    )


# ---------------------------------------------------------------------------
# Monkey-patcher
# ---------------------------------------------------------------------------


class CarryDtypeRecorder:
    """Context manager that patches InnerCarry.__init__ to record z_H dtypes."""

    def __init__(self):
        self.recorded: List[torch.dtype] = []
        self._orig_init = None

    def __enter__(self):
        import marifah.models.coral_base as _coral

        self._orig_init = _coral.InnerCarry.__init__
        recorded = self.recorded

        def _recording_init(self_carry, z_H, z_L):
            self._orig_init(self_carry, z_H, z_L)
            recorded.append(z_H.dtype)

        _coral.InnerCarry.__init__ = _recording_init
        return self

    def __exit__(self, *_):
        import marifah.models.coral_base as _coral
        _coral.InnerCarry.__init__ = self._orig_init


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fwd_dtype,expected_dtype",
    [
        ("float32", torch.float32),
        pytest.param(
            "bfloat16",
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason=(
                    "bfloat16 carry test requires CUDA (flash-attn path). "
                    "Run on Vast before merging any carry-construction change."
                ),
            ),
        ),
    ],
)
def test_carry_dtype_propagates_through_all_call_sites(
    fwd_dtype: str,
    expected_dtype: torch.dtype,
    tiny_ds_dir: Path,
    tmp_path: Path,
):
    """Assert every InnerCarry constructed across all four call sites uses the expected dtype.

    Call sites tested:
      1. trainer.step          — training forward pass
      2. eval_loop.evaluate    — evaluation forward pass
      3. warmstart_probe._extract_carry_states     — probe carry extraction
      4. warmstart_probe.compute_execution_faithfulness — probe faithfulness
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = str(tmp_path / "ckpts")
    config = _make_config(fwd_dtype, str(tiny_ds_dir), ckpt_dir)

    from marifah.training.trainer import build_model, Trainer
    from marifah.training.data_pipeline import build_data_loaders
    from marifah.training.logging import TrainingLogger
    from marifah.training.eval_loop import evaluate
    from scripts.warmstart_probe import _extract_carry_states, compute_execution_faithfulness

    torch.manual_seed(0)
    model = build_model(config, device)
    loaders = build_data_loaders(config)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    logger_obj = TrainingLogger(config)
    trainer = Trainer(model, config, loaders, logger_obj, device)

    batch = next(iter(train_loader))

    with CarryDtypeRecorder() as rec:
        # Site 1: trainer.step
        trainer.step(batch)

        # Site 2: eval_loop.evaluate (runs on val_loader)
        evaluate(model, val_loader, config, device)

        # Site 3: warmstart_probe._extract_carry_states
        _extract_carry_states(model, val_loader, config, device, max_samples=4)

        # Site 4: warmstart_probe.compute_execution_faithfulness
        compute_execution_faithfulness(model, val_loader, config, device, max_samples=4)

    logger_obj.finish()

    assert len(rec.recorded) >= 4, (
        f"Expected at least 4 carry constructions, got {len(rec.recorded)}. "
        "A call site may not be creating InnerCarry at all."
    )
    for i, dtype in enumerate(rec.recorded):
        assert dtype == expected_dtype, (
            f"Carry construction #{i} has dtype={dtype}, expected {expected_dtype}. "
            f"config.model.forward_dtype='{fwd_dtype}'. "
            "A call site is using hardcoded float32 — check eval_loop.py, "
            "trainer.py, and warmstart_probe.py."
        )


def test_inner_carry_zeros_factory_uses_config_dtype():
    """InnerCarry.zeros reads dtype from model_config.forward_dtype, not hardcoded."""
    from marifah.models.coral_base import InnerCarry

    class _FakeConfig:
        forward_dtype = "float32"
        max_nodes = 8
        d_model = 16

    carry = InnerCarry.zeros(2, _FakeConfig(), torch.device("cpu"))
    assert carry.z_H.dtype == torch.float32
    assert carry.z_L.dtype == torch.float32

    class _FakeConfigBf16:
        forward_dtype = "bfloat16"
        max_nodes = 8
        d_model = 16

    carry_bf = InnerCarry.zeros(2, _FakeConfigBf16(), torch.device("cpu"))
    assert carry_bf.z_H.dtype == torch.bfloat16
    assert carry_bf.z_L.dtype == torch.bfloat16


def test_inner_carry_zeros_dtype_override():
    """dtype_override takes precedence over model_config.forward_dtype."""
    from marifah.models.coral_base import InnerCarry

    class _FakeConfig:
        forward_dtype = "bfloat16"
        max_nodes = 4
        d_model = 8

    carry = InnerCarry.zeros(1, _FakeConfig(), torch.device("cpu"), dtype_override=torch.float32)
    assert carry.z_H.dtype == torch.float32, "dtype_override should win over model_config"
