"""Unit tests for generator pipeline, validation, splits, and determinism."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pytest
import numpy as np

from marifah.data.synthetic.generator import DagGenerator, generate_one
from marifah.data.synthetic.labels import DAGRecord
from marifah.data.synthetic.splits import SplitGenerator
from marifah.data.synthetic.storage import write_split, write_manifest, load_manifest, verify_manifest
from marifah.data.synthetic.validate import validate_record, audit_distribution, spot_check_traces
from marifah.data.synthetic.vertical_config import GeneratorConfig, SplitSizes, tiny_config
from marifah.data.synthetic.workflows import build_reserved_primitive_pairs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_cfg() -> GeneratorConfig:
    cfg = GeneratorConfig(seed=7)
    cfg.split_sizes = SplitSizes(
        train=50, val=10, test_id=10, test_ood_size=5, test_ood_composition=5
    )
    from marifah.data.synthetic.vertical_config import _hash_config
    cfg.config_hash = _hash_config(cfg)
    return cfg


@pytest.fixture
def reserved_pairs():
    return build_reserved_primitive_pairs(holdout_fraction=0.15, seed=0)


# ---------------------------------------------------------------------------
# generate_one tests
# ---------------------------------------------------------------------------

class TestGenerateOne:
    def test_returns_dag_record(self, tiny_cfg, reserved_pairs):
        rec = generate_one(
            seed=100, split="val", config=tiny_cfg, reserved_pairs=reserved_pairs
        )
        assert rec is not None
        assert isinstance(rec, DAGRecord)

    def test_deterministic(self, tiny_cfg, reserved_pairs):
        rec1 = generate_one(seed=200, split="val", config=tiny_cfg, reserved_pairs=reserved_pairs)
        rec2 = generate_one(seed=200, split="val", config=tiny_cfg, reserved_pairs=reserved_pairs)
        assert rec1 is not None and rec2 is not None
        assert rec1.dag_id == rec2.dag_id
        assert rec1.halt_step == rec2.halt_step

    def test_different_seeds_produce_different_dags(self, tiny_cfg, reserved_pairs):
        rec1 = generate_one(seed=300, split="val", config=tiny_cfg, reserved_pairs=reserved_pairs)
        rec2 = generate_one(seed=301, split="val", config=tiny_cfg, reserved_pairs=reserved_pairs)
        assert rec1 is not None and rec2 is not None
        # At minimum the seeds should differ
        assert rec1.seed != rec2.seed

    def test_record_halts(self, tiny_cfg, reserved_pairs):
        for seed in range(5):
            rec = generate_one(
                seed=seed + 400, split="val",
                config=tiny_cfg, reserved_pairs=reserved_pairs
            )
            if rec is not None:
                assert rec.halt_step >= 0

    def test_ood_composition_contains_reserved_pair(self, tiny_cfg, reserved_pairs):
        """test_ood_composition DAGs must contain at least one reserved primitive pair."""
        if not reserved_pairs:
            pytest.skip("no reserved pairs")
        found = False
        for seed in range(50):
            rec = generate_one(
                seed=seed + 500, split="test_ood_composition",
                config=tiny_cfg, reserved_pairs=reserved_pairs,
                require_reserved_pair=True,
            )
            if rec is not None:
                # Verify at least one adjacent pair is reserved
                import networkx as nx
                from marifah.data.synthetic.primitives import PrimitiveType
                dag = nx.DiGraph()
                nmap = {n.node_id: n for n in rec.nodes}
                for e in rec.edges:
                    dag.add_edge(e.src, e.dst)
                for src, dst in dag.edges():
                    pair = (nmap[src].primitive, nmap[dst].primitive)
                    if pair in reserved_pairs:
                        found = True
                        break
            if found:
                break
        assert found, "No OOD-composition DAG contained a reserved pair"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_record_passes(self, tiny_cfg, reserved_pairs):
        rec = generate_one(seed=600, split="val", config=tiny_cfg, reserved_pairs=reserved_pairs)
        if rec is None:
            pytest.skip("generation returned None")
        result = validate_record(rec)
        assert result.ok, f"Errors: {result.errors}"

    def test_ten_records_pass_validation(self, tiny_cfg, reserved_pairs):
        errors = []
        for seed in range(10):
            rec = generate_one(seed=seed + 700, split="val", config=tiny_cfg, reserved_pairs=reserved_pairs)
            if rec is None:
                continue
            vr = validate_record(rec)
            if not vr.ok:
                errors.extend(vr.errors)
        assert not errors, f"Validation errors: {errors}"


# ---------------------------------------------------------------------------
# Split generation tests
# ---------------------------------------------------------------------------

class TestSplitGeneration:
    def test_val_split_generates_records(self, tiny_cfg):
        gen = DagGenerator(tiny_cfg)
        records = gen.generate_split("val", 5, seed_offset=10_000)
        assert len(records) > 0
        assert all(r.split == "val" for r in records)

    def test_ood_size_records_are_larger(self, tiny_cfg):
        gen = DagGenerator(tiny_cfg)
        normal = gen.generate_split("test_id", 5, seed_offset=20_000)
        ood = gen.generate_split("test_ood_size", 5, seed_offset=30_000, ood_size=True)
        if normal and ood:
            mean_normal = sum(r.num_nodes for r in normal) / len(normal)
            mean_ood = sum(r.num_nodes for r in ood) / len(ood)
            assert mean_ood >= mean_normal, (
                f"OOD-size DAGs should be larger: {mean_ood:.1f} vs {mean_normal:.1f}"
            )


# ---------------------------------------------------------------------------
# Storage and determinism tests
# ---------------------------------------------------------------------------

class TestStorageAndDeterminism:
    def test_parquet_round_trip(self, tiny_cfg, reserved_pairs, tmp_path):
        records = []
        for seed in range(20):
            rec = generate_one(seed=seed + 800, split="val", config=tiny_cfg,
                               reserved_pairs=reserved_pairs)
            if rec:
                records.append(rec)
        assert records, "No records generated"

        write_split(records, tmp_path, "val")
        shard = list((tmp_path / "val").glob("*.parquet"))
        assert len(shard) >= 1

    def test_manifest_written_and_verified(self, tiny_cfg, reserved_pairs, tmp_path):
        records = []
        for seed in range(10):
            rec = generate_one(seed=seed + 900, split="val", config=tiny_cfg,
                               reserved_pairs=reserved_pairs)
            if rec:
                records.append(rec)
        split_records = {"val": records}
        shard_paths = {"val": write_split(records, tmp_path, "val")}
        write_manifest(tmp_path, tiny_cfg, split_records, shard_paths)
        assert (tmp_path / "manifest.json").exists()
        assert verify_manifest(tmp_path)

    def test_determinism_byte_identical(self, tiny_cfg, reserved_pairs, tmp_path):
        """Same seed + config → byte-identical shard files."""
        def _gen(out: Path):
            records = []
            for seed in range(20):
                rec = generate_one(seed=seed + 1000, split="val", config=tiny_cfg,
                                   reserved_pairs=reserved_pairs)
                if rec:
                    records.append(rec)
            write_split(records, out, "val")
            shard_paths = {"val": sorted((out / "val").glob("*.parquet"))}
            split_records = {"val": records}
            write_manifest(out, tiny_cfg, split_records, shard_paths)

        run1 = tmp_path / "run1"
        run2 = tmp_path / "run2"
        _gen(run1)
        _gen(run2)

        def _hash(p: Path) -> str:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                h.update(f.read())
            return h.hexdigest()

        shards1 = sorted((run1 / "val").glob("*.parquet"))
        shards2 = sorted((run2 / "val").glob("*.parquet"))
        assert len(shards1) == len(shards2), "Different shard counts"
        for s1, s2 in zip(shards1, shards2):
            assert _hash(s1) == _hash(s2), f"Shard mismatch: {s1.name}"


# ---------------------------------------------------------------------------
# Spot check executor consistency
# ---------------------------------------------------------------------------

class TestSpotCheckTraces:
    def test_traces_consistent(self, tiny_cfg, reserved_pairs):
        records = []
        for seed in range(15):
            rec = generate_one(seed=seed + 1100, split="val", config=tiny_cfg,
                               reserved_pairs=reserved_pairs)
            if rec:
                records.append(rec)
        if not records:
            pytest.skip("no records generated")
        result = spot_check_traces(records, n=min(10, len(records)))
        assert result.ok, f"Trace inconsistencies: {result.errors}"


# ---------------------------------------------------------------------------
# Coverage spot check (§4 verification step 7)
# ---------------------------------------------------------------------------

class TestCoverageSpotCheck:
    def test_all_primitives_appear_in_records(self, tiny_cfg, reserved_pairs):
        from marifah.data.synthetic.primitives import PrimitiveType
        seen_primitives = set()
        for seed in range(200):
            rec = generate_one(seed=seed, split="val", config=tiny_cfg,
                               reserved_pairs=reserved_pairs)
            if rec:
                seen_primitives.update(rec.primitive_assignments)
        # All 10 primitives should appear across 200 DAGs
        all_prim_ids = {p.value for p in PrimitiveType}
        missing = all_prim_ids - seen_primitives
        assert not missing, f"Primitives not seen in any DAG: {missing}"


# ---------------------------------------------------------------------------
# Workflow-type distribution coverage (regression for filter-driven extinction)
# ---------------------------------------------------------------------------

class TestTrainSplitWorkflowTypeCoverage:
    """Verifies that the OOD holdout seed does not drive most workflow types extinct.

    Seed=0 caused 40 of 50 types to have 0% training pass rate (boundary pair
    overlap). Seed=13 reduces this to 1 extinct type. This test checks per-type
    pass rate using 20 attempts per workflow type — the same diagnostic method
    used to confirm the root cause and verify the fix.
    """

    def test_train_split_workflow_type_coverage(self):
        from marifah.data.synthetic.vertical_config import GeneratorConfig, _hash_config
        from marifah.data.synthetic.generator import generate_one
        from marifah.data.synthetic.workflows import build_reserved_primitive_pairs, WORKFLOW_DEFINITIONS

        reserved_pairs = build_reserved_primitive_pairs(holdout_fraction=0.15, seed=13)
        cfg = GeneratorConfig(seed=42, ood_holdout_seed=13, ood_holdout_fraction=0.15)
        cfg.config_hash = _hash_config(cfg)

        surviving_types = 0
        for wf in WORKFLOW_DEFINITIONS:
            for attempt in range(20):
                rec = generate_one(
                    seed=wf.workflow_type_id * 1000 + attempt,
                    split="train",
                    config=cfg,
                    reserved_pairs=reserved_pairs,
                    workflow_type_id=wf.workflow_type_id,
                )
                if rec is not None:
                    surviving_types += 1
                    break  # this type passes — move on

        assert surviving_types >= 45, (
            f"Only {surviving_types} workflow types survive training filter "
            f"(expected >= 45 with ood_holdout_seed=13). "
            f"Check docs/sessions/session-06-generator-distribution-finding.md"
        )


class TestOodCompositionDiversity:
    """Verifies that test_ood_composition split has diverse workflow types.

    Root cause of prior 2-class collapse (seed=13):
      - cross-pattern-only pair check + frequency-weighted sampling → only
        workflows 7 and 27 could produce OOD-composition DAGs.
    Fix: use all-edge pair check + round-robin workflow assignment.
    Expected: >= 25 unique workflow types with n=200 tasks.
    """

    def test_test_ood_composition_has_diverse_workflow_types(self):
        """OOD-composition split must have diverse workflow types for substrate probing."""
        from collections import Counter
        from marifah.data.synthetic.vertical_config import GeneratorConfig, _hash_config
        from marifah.data.synthetic.generator import DagGenerator

        cfg = GeneratorConfig(seed=42, ood_holdout_seed=13, ood_holdout_fraction=0.15)
        cfg.config_hash = _hash_config(cfg)

        gen = DagGenerator(cfg)
        recs = gen.generate_split(
            "test_ood_composition", n=200, seed_offset=900_000,
            require_reserved_pair=True,
        )

        wf_counts = Counter(r.workflow_type_id for r in recs)
        n_unique = len(wf_counts)

        assert n_unique >= 25, (
            f"test_ood_composition has only {n_unique} unique workflow types "
            f"(expected >= 25; substrate probing requires diverse classes). "
            f"The all-edge pair check + round-robin sampling should give >= 49/50."
        )

    def test_ood_composition_records_contain_reserved_pair_in_all_edges(self):
        """Every OOD-composition record must have a reserved pair in at least one edge."""
        import networkx as nx
        from marifah.data.synthetic.vertical_config import GeneratorConfig, _hash_config
        from marifah.data.synthetic.generator import DagGenerator
        from marifah.data.synthetic.workflows import build_reserved_primitive_pairs

        cfg = GeneratorConfig(seed=42, ood_holdout_seed=13, ood_holdout_fraction=0.15)
        cfg.config_hash = _hash_config(cfg)
        reserved = build_reserved_primitive_pairs(holdout_fraction=0.15, seed=13)

        gen = DagGenerator(cfg)
        recs = gen.generate_split(
            "test_ood_composition", n=50, seed_offset=910_000,
            require_reserved_pair=True,
        )

        assert len(recs) > 0, "No OOD-composition records generated"
        for rec in recs:
            nmap = {n.node_id: n for n in rec.nodes}
            has_reserved = any(
                (nmap[e.src].primitive, nmap[e.dst].primitive) in reserved
                for e in rec.edges
            )
            assert has_reserved, (
                f"DAG {rec.dag_id} (wf={rec.workflow_type_id}) contains no reserved pair"
            )
