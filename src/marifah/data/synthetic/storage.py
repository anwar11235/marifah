"""§11 — Storage: parquet writing, manifest emission, dataset directory structure."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from marifah.data.synthetic.labels import DAGRecord
from marifah.data.synthetic.vertical_config import GeneratorConfig


SHARD_SIZE = 10_000  # DAGs per parquet shard


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_PARQUET_SCHEMA = pa.schema([
    pa.field("dag_id",               pa.string()),
    pa.field("workflow_type_id",     pa.int32()),
    pa.field("split",                pa.string()),
    pa.field("seed",                 pa.int64()),
    pa.field("nodes",                pa.string()),   # JSON
    pa.field("edges",                pa.string()),   # JSON
    pa.field("region_assignments",   pa.string()),   # JSON
    pa.field("primitive_assignments",pa.string()),   # JSON
    pa.field("execution_trace",      pa.string()),   # JSON
    pa.field("halt_step",            pa.int32()),
    pa.field("ood_flags",            pa.string()),   # JSON
    pa.field("num_nodes",            pa.int32()),
    pa.field("num_edges",            pa.int32()),
    pa.field("frequency_tier",       pa.string()),
])


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

def write_shard(
    records: List[DAGRecord],
    shard_path: Path,
) -> None:
    """Write a single shard of records to shard_path."""
    rows = [r.to_parquet_row() for r in records]
    table = pa.Table.from_pylist(rows, schema=_PARQUET_SCHEMA)
    pq.write_table(table, shard_path, compression="snappy")


def write_split(
    records: List[DAGRecord],
    output_dir: Path,
    split: str,
) -> List[Path]:
    """Write a split's records as sharded parquet files.  Returns shard paths."""
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    shards: List[Path] = []
    for shard_idx, start in enumerate(range(0, max(len(records), 1), SHARD_SIZE)):
        chunk = records[start: start + SHARD_SIZE]
        if not chunk:
            break
        shard_path = split_dir / f"shard_{shard_idx:04d}.parquet"
        write_shard(chunk, shard_path)
        shards.append(shard_path)

    return shards


def read_split(split_dir: Path) -> List[Dict[str, Any]]:
    """Read all shards in a split directory and return rows as dicts."""
    rows: List[Dict[str, Any]] = []
    for shard in sorted(split_dir.glob("shard_*.parquet")):
        table = pq.read_table(shard)
        rows.extend(table.to_pydict())
    return rows


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    output_dir: Path,
    config: GeneratorConfig,
    split_records: Dict[str, List[DAGRecord]],
    shard_paths: Dict[str, List[Path]],
    generator_version: str = "session-2",
) -> Path:
    """Write manifest.json and generator_config.json to output_dir."""
    # Statistical summary
    summary: Dict[str, Any] = {}
    for split, records in split_records.items():
        if not records:
            summary[split] = {"count": 0}
            continue
        node_counts = [r.num_nodes for r in records]
        summary[split] = {
            "count": len(records),
            "mean_nodes": sum(node_counts) / len(node_counts),
            "max_nodes": max(node_counts),
            "min_nodes": min(node_counts),
        }

    # Shard hashes
    shard_hashes: Dict[str, List[str]] = {}
    for split, paths in shard_paths.items():
        shard_hashes[split] = [_sha256_file(p) for p in paths]

    manifest = {
        "generator_version": generator_version,
        "config_hash": config.config_hash,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": config.seed,
        "splits": summary,
        "shard_hashes": shard_hashes,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    config_path = output_dir / "generator_config.json"
    with open(config_path, "w") as fh:
        json.dump({
            "primitives": config.primitive_names,
            "allow_cycles": config.allow_cycles,
            "ood_holdout_fraction": config.ood_holdout_fraction,
            "ood_size_scale_min": config.ood_size_scale_min,
            "ood_size_scale_max": config.ood_size_scale_max,
            "seed": config.seed,
            "config_hash": config.config_hash,
            "split_sizes": {
                "train": config.split_sizes.train,
                "val": config.split_sizes.val,
                "test_id": config.split_sizes.test_id,
                "test_ood_size": config.split_sizes.test_ood_size,
                "test_ood_composition": config.split_sizes.test_ood_composition,
            },
        }, fh, indent=2)

    return manifest_path


def write_manifest_from_counts(
    output_dir: Path,
    config: GeneratorConfig,
    split_counts: Dict[str, int],
    split_shard_paths: Dict[str, List[Path]],
    generator_version: str = "session-2",
) -> Path:
    """Write manifest.json from record counts and shard paths.

    Streaming alternative to write_manifest — does not require all records
    in memory.  Summary stats (mean/max/min nodes) are omitted; only count
    is recorded per split.
    """
    summary: Dict[str, Any] = {split: {"count": count} for split, count in split_counts.items()}

    shard_hashes: Dict[str, List[str]] = {}
    for split, paths in split_shard_paths.items():
        shard_hashes[split] = [_sha256_file(p) for p in paths]

    manifest = {
        "generator_version": generator_version,
        "config_hash": config.config_hash,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": config.seed,
        "splits": summary,
        "shard_hashes": shard_hashes,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    config_path = output_dir / "generator_config.json"
    with open(config_path, "w") as fh:
        json.dump({
            "primitives": config.primitive_names,
            "allow_cycles": config.allow_cycles,
            "ood_holdout_fraction": config.ood_holdout_fraction,
            "ood_size_scale_min": config.ood_size_scale_min,
            "ood_size_scale_max": config.ood_size_scale_max,
            "seed": config.seed,
            "config_hash": config.config_hash,
            "split_sizes": {
                "train": config.split_sizes.train,
                "val": config.split_sizes.val,
                "test_id": config.split_sizes.test_id,
                "test_ood_size": config.split_sizes.test_ood_size,
                "test_ood_composition": config.split_sizes.test_ood_composition,
            },
        }, fh, indent=2)

    return manifest_path


def load_manifest(output_dir: Path) -> Dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path) as fh:
        return json.load(fh)


def verify_manifest(output_dir: Path) -> bool:
    """Verify shard hashes from manifest.  Returns True if all match."""
    manifest = load_manifest(output_dir)
    for split, hashes in manifest.get("shard_hashes", {}).items():
        split_dir = output_dir / split
        shards = sorted(split_dir.glob("shard_*.parquet"))
        if len(shards) != len(hashes):
            return False
        for shard_path, expected_hash in zip(shards, hashes):
            if _sha256_file(shard_path) != expected_hash:
                return False
    return True
