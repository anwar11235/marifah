"""CLI entry point: python -m marifah.data.synthetic.cli <command> [args]

Commands
--------
generate-tiny   Generate ~1K DAGs for smoke testing.
generate-full   Generate the full dataset per split_sizes in config.
validate-dataset  Run well-formedness and distribution audits on a dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from marifah.data.synthetic.generator import DagGenerator
from marifah.data.synthetic.splits import SplitGenerator
from marifah.data.synthetic.storage import (
    load_manifest,
    verify_manifest,
    write_manifest,
    write_split,
)
from marifah.data.synthetic.validate import (
    DistributionAudit,
    ValidationResult,
    audit_distribution,
    spot_check_traces,
    validate_record,
)
from marifah.data.synthetic.vertical_config import GeneratorConfig, load_config, tiny_config
from marifah.data.synthetic.workflows import validate_coverage


def _generate(
    config: GeneratorConfig,
    output_dir: Path,
    num_workers: int,
    label: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    sg = SplitGenerator(config, num_workers=num_workers)

    all_records = {}
    all_shards = {}

    for split_name in ["train", "val", "test_id", "test_ood_size", "test_ood_composition"]:
        print(f"  Generating {split_name}...", flush=True)
        records = getattr(sg, f"generate_{split_name.replace('-', '_')}")()
        all_records[split_name] = records
        shards = write_split(records, output_dir, split_name)
        all_shards[split_name] = shards
        print(f"    -> {len(records)} records, {len(shards)} shards", flush=True)

    manifest_path = write_manifest(output_dir, config, all_records, all_shards)
    elapsed = time.time() - t0
    total = sum(len(v) for v in all_records.values())
    print(f"\n{label}: {total} DAGs in {elapsed:.1f}s -> {output_dir}")
    print(f"Manifest: {manifest_path}")


def cmd_generate_tiny(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cfg = tiny_config(cfg)
    print(f"Tiny config: seed={cfg.seed} hash={cfg.config_hash}")
    print(f"Split sizes: {cfg.split_sizes}")
    _generate(cfg, Path(args.output), num_workers=args.workers, label="generate-tiny")
    return 0


def cmd_generate_full(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(f"Full config: seed={cfg.seed} hash={cfg.config_hash}")
    print(f"Split sizes: {cfg.split_sizes}")
    _generate(cfg, Path(args.output), num_workers=args.workers, label="generate-full")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"ERROR: dataset path not found: {dataset_path}", file=sys.stderr)
        return 1

    errors_total = 0

    # Check workflow coverage constraints
    ok, msg = validate_coverage()
    if not ok:
        print(f"FAIL coverage: {msg}", file=sys.stderr)
        errors_total += 1
    else:
        print(f"OK coverage: {msg}")

    # Per-split validation
    for split in ["train", "val", "test_id", "test_ood_size", "test_ood_composition"]:
        split_dir = dataset_path / split
        if not split_dir.exists():
            print(f"WARN: split directory missing: {split_dir}")
            continue

        import pyarrow.parquet as pq, json as _json

        rows = []
        for shard in sorted(split_dir.glob("shard_*.parquet")):
            t = pq.read_table(shard)
            rows.extend(t.to_pylist())
        if not rows:
            print(f"WARN: {split} has no records")
            continue

        # Reconstruct minimal records for validation (fast path)
        n_errors = 0
        sample_size = min(50, len(rows))

        # Spot checks on a sample
        from marifah.data.synthetic.labels import (
            DAGRecord, EdgeRecord, NodeRecord, RegionAssignment
        )

        sample_records = []
        for row in rows[:sample_size]:
            nodes_raw = _json.loads(row["nodes"])
            edges_raw = _json.loads(row["edges"])
            ra_raw = _json.loads(row["region_assignments"])
            pa_raw = _json.loads(row["primitive_assignments"])
            trace_raw = _json.loads(row["execution_trace"])

            rec = DAGRecord(
                dag_id=row["dag_id"],
                workflow_type_id=row["workflow_type_id"],
                split=row["split"],
                seed=row["seed"],
                nodes=[NodeRecord(n["node_id"], n["primitive"], n["primitive_name"],
                                  n["attributes"]) for n in nodes_raw],
                edges=[EdgeRecord(e["src"], e["dst"], e.get("branch")) for e in edges_raw],
                region_assignments=[RegionAssignment(r["node_id"], r["pattern_id"],
                                                     r["pattern_instance_id"]) for r in ra_raw],
                primitive_assignments=pa_raw,
                execution_trace=trace_raw,
                halt_step=row["halt_step"],
                ood_flags=_json.loads(row["ood_flags"]),
                frequency_tier=row.get("frequency_tier", ""),
            )
            sample_records.append(rec)

            vr = validate_record(rec)
            if not vr.ok:
                n_errors += 1
                if args.verbose:
                    print(f"  {rec.dag_id}: {vr.errors}")

        # Executor consistency spot check
        spot_result = spot_check_traces(sample_records, n=min(10, len(sample_records)))
        spot_ok = spot_result.ok

        status = "OK" if n_errors == 0 else "FAIL"
        spot_status = "OK" if spot_ok else "FAIL"
        print(f"{status} {split}: {len(rows)} records, {n_errors}/{sample_size} bad")
        print(f"  spot-check traces: {spot_status}"
              + (f" ({spot_result.errors[0]})" if spot_result.errors else ""))
        errors_total += n_errors + (0 if spot_ok else 1)

    # Manifest verification
    if (dataset_path / "manifest.json").exists():
        manifest_ok = verify_manifest(dataset_path)
        print(f"{'OK' if manifest_ok else 'FAIL'} manifest hash verification")
        if not manifest_ok:
            errors_total += 1
    else:
        print("WARN: no manifest.json found")

    if errors_total == 0:
        print("\nvalidate-dataset: PASSED")
        return 0
    else:
        print(f"\nvalidate-dataset: FAILED ({errors_total} errors)", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="marifah.data.synthetic.cli")
    sub = parser.add_subparsers(dest="command")

    p_tiny = sub.add_parser("generate-tiny", help="Generate ~1K DAGs for smoke testing")
    p_tiny.add_argument("--config", required=True, help="Path to YAML config")
    p_tiny.add_argument("--output", required=True, help="Output directory")
    p_tiny.add_argument("--workers", type=int, default=1)

    p_full = sub.add_parser("generate-full", help="Generate full dataset")
    p_full.add_argument("--config", required=True, help="Path to YAML config")
    p_full.add_argument("--output", required=True, help="Output directory")
    p_full.add_argument("--workers", type=int, default=1)

    p_val = sub.add_parser("validate-dataset", help="Validate a generated dataset")
    p_val.add_argument("dataset_path", help="Path to dataset root")
    p_val.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "generate-tiny":
        return cmd_generate_tiny(args)
    elif args.command == "generate-full":
        return cmd_generate_full(args)
    elif args.command == "validate-dataset":
        return cmd_validate(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
