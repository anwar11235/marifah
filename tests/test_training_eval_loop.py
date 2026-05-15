"""Unit tests for eval_loop.evaluate."""

import pytest
import torch
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eval_dataset_dir(tmp_path_factory):
    from marifah.data.synthetic.generator import DagGenerator
    from marifah.data.synthetic.storage import write_split
    from marifah.data.synthetic.vertical_config import (
        GeneratorConfig, SplitSizes, _hash_config
    )
    root = tmp_path_factory.mktemp("eval_ds")
    cfg = GeneratorConfig(seed=55)
    cfg.split_sizes = SplitSizes(train=10, val=10, test_id=0, test_ood_size=0, test_ood_composition=0)
    cfg.config_hash = _hash_config(cfg)
    gen = DagGenerator(cfg)
    for split, n, off in [("train", 10, 0), ("val", 10, 10_000)]:
        records = gen.generate_split(split, n, seed_offset=off)
        write_split(records, root, split)
    return root


def _make_config(dataset_root: str, use_hmsc: bool = False):
    from marifah.training.config import TrainingConfig
    hmsc = None
    if use_hmsc:
        hmsc = {
            "K_G": 4, "K_R": 2, "K_P": 2,
            "d_G": 16, "d_R": 8, "d_P": 4,
            "num_regions": 2,
            "composition_method": "sum",
            "p_discreteness": "soft",
            "num_workflow_types": 50, "num_pattern_types": 12, "num_primitives": 10,
        }
    raw = {
        "experiment": {"name": "eval_test", "phase": 0, "seed": 0},
        "model": {
            "d_model": 16, "num_heads": 2,
            "H_cycles": 1, "L_cycles": 1, "H_layers": 1, "L_layers": 1,
            "vocab_size": 10, "max_nodes": 32,
            "use_hmsc": use_hmsc, "hmsc": hmsc,
            "halt_max_steps": 2, "halt_exploration_prob": 0.0,
        },
        "training": {
            "batch_size": 4, "max_epochs": 1, "eval_interval_epochs": 1,
            "learning_rate": 1e-4, "warmup_steps": 1,
            "lambda_G": 0.1 if use_hmsc else 0.0,
            "lambda_R": 0.1 if use_hmsc else 0.0,
            "lambda_P": 0.1 if use_hmsc else 0.0,
            "drop_last": False,
        },
        "data": {
            "dataset_root": dataset_root,
            "num_workers": 0, "pin_memory": False,
        },
        "logging": {"wandb_mode": "disabled", "checkpoint_dir": "/tmp/eval_test_ckpt/"},
    }
    return TrainingConfig(**raw)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_completes_without_error(self, eval_dataset_dir):
        from marifah.training.trainer import build_model
        from marifah.training.data_pipeline import build_data_loaders
        from marifah.training.eval_loop import evaluate

        cfg = _make_config(str(eval_dataset_dir))
        device = torch.device("cpu")
        model = build_model(cfg, device)
        loaders = build_data_loaders(cfg)
        val_loader = loaders["val"]
        assert val_loader is not None

        metrics = evaluate(model, val_loader, cfg, device)
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_all_metrics_real(self, eval_dataset_dir):
        from marifah.training.trainer import build_model
        from marifah.training.data_pipeline import build_data_loaders
        from marifah.training.eval_loop import evaluate
        import math

        cfg = _make_config(str(eval_dataset_dir))
        device = torch.device("cpu")
        model = build_model(cfg, device)
        loaders = build_data_loaders(cfg)
        metrics = evaluate(model, loaders["val"], cfg, device)

        for k, v in metrics.items():
            assert not math.isnan(v), f"Metric {k} is NaN"
            assert not math.isinf(v), f"Metric {k} is inf"

    def test_deterministic(self, eval_dataset_dir):
        from marifah.training.trainer import build_model
        from marifah.training.data_pipeline import build_data_loaders
        from marifah.training.eval_loop import evaluate

        cfg = _make_config(str(eval_dataset_dir))
        device = torch.device("cpu")
        model = build_model(cfg, device)
        loaders = build_data_loaders(cfg)
        val_loader = loaders["val"]

        m1 = evaluate(model, val_loader, cfg, device)
        m2 = evaluate(model, val_loader, cfg, device)
        for k in m1:
            assert abs(m1[k] - m2[k]) < 1e-6, f"Non-deterministic metric: {k}"

    def test_hmsc_utilisation_in_metrics(self, eval_dataset_dir):
        from marifah.training.trainer import build_model
        from marifah.training.data_pipeline import build_data_loaders
        from marifah.training.eval_loop import evaluate

        cfg = _make_config(str(eval_dataset_dir), use_hmsc=True)
        device = torch.device("cpu")
        model = build_model(cfg, device)
        loaders = build_data_loaders(cfg)
        metrics = evaluate(model, loaders["val"], cfg, device)

        # At least one codebook utilisation stat should appear
        util_keys = [k for k in metrics if k.startswith("codebook_")]
        assert len(util_keys) > 0, f"No codebook utilisation metrics found. Keys: {list(metrics.keys())}"
