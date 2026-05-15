"""Unit tests for data_pipeline: build_data_loaders and padded collation."""

import pytest
import torch
from pathlib import Path

from marifah.training.data_pipeline import collate_graphs_padded, build_data_loaders
from marifah.training.config import TrainingConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_dataset_dir(tmp_path_factory):
    """Generate a tiny dataset for testing."""
    from marifah.data.synthetic.generator import DagGenerator
    from marifah.data.synthetic.storage import write_split
    from marifah.data.synthetic.vertical_config import (
        GeneratorConfig, SplitSizes, _hash_config
    )
    root = tmp_path_factory.mktemp("dp_ds")
    cfg = GeneratorConfig(seed=99)
    cfg.split_sizes = SplitSizes(train=20, val=10, test_id=5, test_ood_size=5, test_ood_composition=5)
    cfg.config_hash = _hash_config(cfg)
    gen = DagGenerator(cfg)
    for split, n, off in [("train", 20, 0), ("val", 10, 10_000)]:
        records = gen.generate_split(split, n, seed_offset=off)
        write_split(records, root, split)
    return root


def _make_config(dataset_root: str, use_hmsc: bool = False) -> TrainingConfig:
    raw = {
        "experiment": {"name": "dp_test", "phase": 0, "seed": 42},
        "model": {
            "d_model": 32, "num_heads": 2, "vocab_size": 10,
            "max_nodes": 64, "use_hmsc": use_hmsc,
        },
        "training": {
            "batch_size": 4, "max_epochs": 1, "eval_interval_epochs": 1,
            "learning_rate": 1e-4, "warmup_steps": 1,
        },
        "data": {
            "dataset_root": dataset_root,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
        },
        "logging": {"wandb_mode": "disabled", "checkpoint_dir": "/tmp/dp_test_ckpt/"},
    }
    return TrainingConfig(**raw)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCollateGraphsPadded:
    def test_pads_to_max_nodes(self):
        from marifah.data.adapter.dataset import GraphDAGDataset
        from marifah.data.adapter.collate import collate_graphs
        from marifah.data.synthetic.generator import DagGenerator
        from marifah.data.synthetic.storage import write_split
        from marifah.data.synthetic.vertical_config import (
            GeneratorConfig, SplitSizes, _hash_config
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = GeneratorConfig(seed=1)
            cfg.split_sizes = SplitSizes(train=10, val=0, test_id=0, test_ood_size=0, test_ood_composition=0)
            cfg.config_hash = _hash_config(cfg)
            gen = DagGenerator(cfg)
            records = gen.generate_split("train", 10, seed_offset=0)
            write_split(records, root, "train")

            ds = GraphDAGDataset(root / "train", max_nodes=64)
            items = [ds[i] for i in range(min(4, len(ds)))]
            batch = collate_graphs_padded(items, max_nodes=64)

            assert batch.node_features.shape[1] == 64
            assert batch.attention_mask.shape == (len(items), 64, 64)
            assert batch.node_mask.shape == (len(items), 64)
            assert batch.pos_encoding.shape[1] == 64

    def test_raises_if_batch_exceeds_max_nodes(self, tmp_path):
        from marifah.data.adapter.dataset import GraphDAGDataset
        from marifah.data.synthetic.generator import DagGenerator
        from marifah.data.synthetic.storage import write_split
        from marifah.data.synthetic.vertical_config import (
            GeneratorConfig, SplitSizes, _hash_config
        )
        cfg = GeneratorConfig(seed=2)
        cfg.split_sizes = SplitSizes(train=5, val=0, test_id=0, test_ood_size=0, test_ood_composition=0)
        cfg.config_hash = _hash_config(cfg)
        gen = DagGenerator(cfg)
        records = gen.generate_split("train", 5, seed_offset=0)
        write_split(records, tmp_path, "train")

        ds = GraphDAGDataset(tmp_path / "train", max_nodes=200)
        items = [ds[i] for i in range(min(2, len(ds)))]
        with pytest.raises(ValueError, match="max_nodes"):
            collate_graphs_padded(items, max_nodes=2)  # intentionally too small


class TestBuildDataLoaders:
    def test_returns_train_and_val(self, tiny_dataset_dir):
        cfg = _make_config(str(tiny_dataset_dir))
        loaders = build_data_loaders(cfg)
        assert "train" in loaders
        assert "val" in loaders
        assert loaders["train"] is not None
        assert loaders["val"] is not None

    def test_train_batch_shapes(self, tiny_dataset_dir):
        cfg = _make_config(str(tiny_dataset_dir))
        loaders = build_data_loaders(cfg)
        batch = next(iter(loaders["train"]))
        assert batch.node_features.shape[1] == cfg.model.max_nodes
        assert batch.node_mask.shape[1] == cfg.model.max_nodes

    def test_missing_split_returns_none(self, tiny_dataset_dir):
        cfg = _make_config(str(tiny_dataset_dir))
        loaders = build_data_loaders(cfg)
        # test_ood_size was not generated in the fixture
        assert loaders["test_ood_size"] is None
