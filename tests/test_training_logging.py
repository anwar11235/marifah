"""Unit tests for TrainingLogger (wandb=disabled mode)."""

import json
import os
import tempfile
import pytest

from marifah.training.config import TrainingConfig
from marifah.training.logging import TrainingLogger


def _make_config(tmp_dir: str) -> TrainingConfig:
    return TrainingConfig(**{
        "experiment": {"name": "test_run", "phase": 0, "seed": 0},
        "logging": {
            "wandb_mode": "disabled",
            "checkpoint_dir": tmp_dir,
            "log_interval_steps": 1,
            "utilization_interval_steps": 1,
        },
    })


class TestTrainingLogger:
    def test_instantiates_without_wandb(self, tmp_path):
        cfg = _make_config(str(tmp_path))
        logger = TrainingLogger(cfg)
        assert logger.run_name is not None
        logger.finish()

    def test_jsonl_created(self, tmp_path):
        cfg = _make_config(str(tmp_path))
        logger = TrainingLogger(cfg)
        logger.log_step(0, {"total": 1.0, "main": 0.9}, learning_rate=1e-4)
        logger.finish()
        assert os.path.exists(logger.jsonl_path)

    def test_log_step_parseable(self, tmp_path):
        cfg = _make_config(str(tmp_path))
        logger = TrainingLogger(cfg)
        logger.log_step(1, {"total": 2.5, "main": 2.0, "halt": 0.5}, learning_rate=5e-5)
        logger.finish()

        with open(logger.jsonl_path) as f:
            records = [json.loads(line) for line in f if line.strip()]

        assert len(records) >= 1
        r = records[0]
        assert r["step"] == 1
        assert r["_type"] == "step"
        assert "train/total" in r

    def test_log_eval_parseable(self, tmp_path):
        cfg = _make_config(str(tmp_path))
        logger = TrainingLogger(cfg)
        logger.log_eval(10, "val", {"loss_main": 1.2, "accuracy_node": 0.7})
        logger.finish()

        with open(logger.jsonl_path) as f:
            records = [json.loads(line) for line in f if line.strip()]

        eval_records = [r for r in records if r.get("_type") == "eval"]
        assert len(eval_records) >= 1
        assert eval_records[0]["split"] == "val"
        assert "eval/val/loss_main" in eval_records[0]

    def test_log_codebook_stats(self, tmp_path):
        import torch
        cfg = _make_config(str(tmp_path))
        logger = TrainingLogger(cfg)
        util = {"G_active_frac": 0.8, "R_active_frac": 0.5}
        logger.log_codebook_stats(5, {"codebook_utilization": util})
        logger.finish()

        with open(logger.jsonl_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        cb_records = [r for r in records if r.get("_type") == "codebook_stats"]
        assert len(cb_records) >= 1
        assert "codebook/G_active_frac" in cb_records[0]
