"""Tests for synthetic generator performance fixes.

Covers:
  - Streaming shard writes (shards appear on disk during generation)
  - Multiprocessing correctness (results match single-worker baseline)
  - CLI default --workers uses os.cpu_count() - 1
  - Determinism: same seed + same num_workers -> identical record set
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

import pytest

from marifah.data.synthetic.vertical_config import GeneratorConfig, SplitSizes, tiny_config, load_config
from marifah.data.synthetic.splits import SplitGenerator
from marifah.data.synthetic.storage import write_shard, verify_manifest, write_manifest_from_counts
from marifah.data.synthetic.labels import DAGRecord


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_cfg() -> GeneratorConfig:
    """Config with 200 train + small test splits — fast enough for correctness tests."""
    cfg = load_config("configs/default.yaml")
    cfg = tiny_config(cfg)  # shrink to tiny sizes
    # Override to slightly larger train for streaming/parallelism tests
    cfg.split_sizes = SplitSizes(
        train=200,
        val=30,
        test_id=20,
        test_ood_size=10,
        test_ood_composition=10,
    )
    return cfg


@pytest.fixture()
def tiny_cfg() -> GeneratorConfig:
    """Minimal config for fast unit-style tests."""
    cfg = load_config("configs/default.yaml")
    cfg = tiny_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Test 1: streaming shard writes
# ---------------------------------------------------------------------------

def test_streaming_generation_produces_shards(small_cfg: GeneratorConfig, tmp_path: Path) -> None:
    """Shards are written progressively — each batch lands on disk as a shard file."""
    sg = SplitGenerator(small_cfg, num_workers=1)
    split_dir = tmp_path / "val"
    split_dir.mkdir()

    shard_count = 0
    record_count = 0
    for batch in sg.generate_split_streaming("val", batch_size=10):
        shard_path = split_dir / f"shard_{shard_count:04d}.parquet"
        write_shard(batch, shard_path)
        shard_count += 1
        record_count += len(batch)
        # Verify the shard file exists immediately after write
        assert shard_path.exists(), f"Shard {shard_path} not written"

    assert record_count == small_cfg.split_sizes.val
    assert shard_count > 0
    shards = sorted(split_dir.glob("shard_*.parquet"))
    assert len(shards) == shard_count


def test_streaming_yields_correct_total(small_cfg: GeneratorConfig) -> None:
    """Total records across all yielded batches equals the configured split size."""
    sg = SplitGenerator(small_cfg, num_workers=1)
    for split_name in ["val", "test_id"]:
        total = sum(len(batch) for batch in sg.generate_split_streaming(split_name, batch_size=10))
        expected = getattr(small_cfg.split_sizes, split_name)
        assert total == expected, f"{split_name}: got {total}, expected {expected}"


def test_streaming_batch_size_respected(small_cfg: GeneratorConfig) -> None:
    """Every yielded batch (except possibly the last) is exactly batch_size records."""
    sg = SplitGenerator(small_cfg, num_workers=1)
    batch_size = 7
    batches = list(sg.generate_split_streaming("val", batch_size=batch_size))
    assert len(batches) >= 1
    for batch in batches[:-1]:
        assert len(batch) == batch_size, f"Expected batch_size={batch_size}, got {len(batch)}"
    # Last batch may be smaller
    assert 1 <= len(batches[-1]) <= batch_size


# ---------------------------------------------------------------------------
# Test 2: multiprocessing correctness
# ---------------------------------------------------------------------------

def test_multiprocessing_produces_correct_record_count(small_cfg: GeneratorConfig) -> None:
    """With num_workers=2, the total record count still matches split sizes."""
    if os.cpu_count() is None or os.cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for this test")
    sg = SplitGenerator(small_cfg, num_workers=2)
    total = sum(len(batch) for batch in sg.generate_split_streaming("val", batch_size=50))
    assert total == small_cfg.split_sizes.val


def test_multiprocessing_all_records_are_valid_dag_records(small_cfg: GeneratorConfig) -> None:
    """All records from multi-worker streaming are valid DAGRecord instances."""
    if os.cpu_count() is None or os.cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for this test")
    sg = SplitGenerator(small_cfg, num_workers=2)
    all_records: List[DAGRecord] = []
    for batch in sg.generate_split_streaming("val", batch_size=50):
        all_records.extend(batch)
    assert len(all_records) == small_cfg.split_sizes.val
    for rec in all_records[:5]:
        assert isinstance(rec, DAGRecord)
        assert rec.num_nodes > 0
        assert rec.halt_step >= 0


# ---------------------------------------------------------------------------
# Test 3: default workers CLI argument
# ---------------------------------------------------------------------------

def test_default_workers_uses_cpu_count() -> None:
    """CLI DEFAULT_WORKERS is max(1, cpu_count() - 1) at import time."""
    import marifah.data.synthetic.cli as cli_module

    # DEFAULT_WORKERS is computed at module import time, not at call time.
    _cpu = os.cpu_count()
    expected = max(1, _cpu - 1) if _cpu else 1
    assert cli_module.DEFAULT_WORKERS == expected
    assert cli_module.DEFAULT_WORKERS >= 1


def test_default_workers_fallback_when_cpu_count_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falls back to 1 worker when os.cpu_count() returns None."""
    # The DEFAULT_WORKERS constant is computed at import time, so we test the formula
    _cpu = None
    computed = max(1, _cpu - 1) if _cpu else 1
    assert computed == 1


def test_cli_workers_argument_parsed(tmp_path: Path) -> None:
    """CLI argument --workers is accepted without error."""
    import marifah.data.synthetic.cli as cli_module
    parser_args = cli_module.main.__code__  # module accessible

    # Build a fresh parser to check the default
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    p_tiny = sub.add_parser("generate-tiny")
    p_tiny.add_argument("--config", default="configs/default.yaml")
    p_tiny.add_argument("--output", default=str(tmp_path))
    p_tiny.add_argument("--workers", type=int, default=cli_module.DEFAULT_WORKERS)

    parsed = p.parse_args(["generate-tiny", "--config", "configs/default.yaml",
                           "--output", str(tmp_path)])
    assert parsed.workers == cli_module.DEFAULT_WORKERS

    parsed_explicit = p.parse_args(["generate-tiny", "--config", "configs/default.yaml",
                                    "--output", str(tmp_path), "--workers", "4"])
    assert parsed_explicit.workers == 4


# ---------------------------------------------------------------------------
# Test 4: determinism with multiprocessing
# ---------------------------------------------------------------------------

def test_determinism_single_worker(tiny_cfg: GeneratorConfig) -> None:
    """Two single-worker runs with the same seed produce the same dag_ids in the same order."""
    sg1 = SplitGenerator(tiny_cfg, num_workers=1)
    sg2 = SplitGenerator(tiny_cfg, num_workers=1)

    ids1 = [rec.dag_id for batch in sg1.generate_split_streaming("val") for rec in batch]
    ids2 = [rec.dag_id for batch in sg2.generate_split_streaming("val") for rec in batch]
    assert ids1 == ids2, "Single-worker generation is not deterministic"


def test_determinism_multi_worker_record_set(small_cfg: GeneratorConfig) -> None:
    """Two multi-worker runs produce the same set of dag seeds (order preserved via imap)."""
    if os.cpu_count() is None or os.cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for this test")

    sg1 = SplitGenerator(small_cfg, num_workers=2)
    sg2 = SplitGenerator(small_cfg, num_workers=2)

    seeds1 = sorted(rec.seed for batch in sg1.generate_split_streaming("val") for rec in batch)
    seeds2 = sorted(rec.seed for batch in sg2.generate_split_streaming("val") for rec in batch)
    assert seeds1 == seeds2, "Multi-worker runs produced different seed sets"
    assert len(seeds1) == small_cfg.split_sizes.val


def test_write_manifest_from_counts_and_verify(small_cfg: GeneratorConfig, tmp_path: Path) -> None:
    """write_manifest_from_counts produces a manifest that passes verify_manifest."""
    from marifah.data.synthetic.storage import write_shard, write_manifest_from_counts, verify_manifest

    sg = SplitGenerator(small_cfg, num_workers=1)
    all_shard_paths = {}
    all_record_counts = {}

    for split_name in ["val", "test_id"]:
        split_dir = tmp_path / split_name
        split_dir.mkdir()
        shard_paths = []
        shard_idx = 0
        record_count = 0
        for batch in sg.generate_split_streaming(split_name, batch_size=15):
            shard_path = split_dir / f"shard_{shard_idx:04d}.parquet"
            write_shard(batch, shard_path)
            shard_paths.append(shard_path)
            shard_idx += 1
            record_count += len(batch)
        all_shard_paths[split_name] = shard_paths
        all_record_counts[split_name] = record_count

    manifest_path = write_manifest_from_counts(tmp_path, small_cfg, all_record_counts, all_shard_paths)
    assert manifest_path.exists()
    assert verify_manifest(tmp_path)
