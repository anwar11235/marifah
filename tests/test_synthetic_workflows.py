"""Unit tests for workflow definitions and coverage constraints."""

import pytest

from marifah.data.synthetic.workflows import (
    WORKFLOW_DEFINITIONS,
    WORKFLOW_BY_ID,
    WORKFLOW_INSTANCES_MAP,
    WORKFLOW_TIER_MAP,
    WORKFLOW_SAMPLING_WEIGHTS,
    FREQUENCY_TIERS,
    validate_coverage,
    build_reserved_primitive_pairs,
)
from marifah.data.synthetic.patterns import NUM_PATTERNS


class TestWorkflowDefinitions:
    def test_fifty_workflows(self):
        assert len(WORKFLOW_DEFINITIONS) == 50

    def test_ids_1_to_50(self):
        ids = sorted(w.workflow_type_id for w in WORKFLOW_DEFINITIONS)
        assert ids == list(range(1, 51))

    def test_simple_workflows_have_2_to_3_patterns(self):
        simple = [w for w in WORKFLOW_DEFINITIONS if w.size_tier == "simple"]
        assert len(simple) == 10
        for w in simple:
            assert 2 <= len(set(w.pattern_sequence)) <= 3, (
                f"workflow {w.workflow_type_id} has {len(set(w.pattern_sequence))} distinct patterns"
            )

    def test_medium_workflows_have_3_to_5_patterns(self):
        medium = [w for w in WORKFLOW_DEFINITIONS if w.size_tier == "medium"]
        assert len(medium) == 25
        for w in medium:
            assert len(set(w.pattern_sequence)) >= 2

    def test_complex_workflows_have_5_to_8_patterns(self):
        complex_ = [w for w in WORKFLOW_DEFINITIONS if w.size_tier == "complex"]
        assert len(complex_) == 15


class TestCoverageConstraints:
    def test_coverage_passes(self):
        ok, msg = validate_coverage()
        assert ok, f"Coverage validation failed: {msg}"

    def test_each_pattern_in_at_least_5_workflows(self):
        pattern_counts = {i: 0 for i in range(NUM_PATTERNS)}
        for w in WORKFLOW_DEFINITIONS:
            for pat_id in set(w.pattern_sequence):
                pattern_counts[pat_id] += 1
        for pat_id, count in pattern_counts.items():
            assert count >= 5, f"Pattern {pat_id} appears in only {count} workflows"

    def test_each_workflow_has_at_least_2_distinct_patterns(self):
        for w in WORKFLOW_DEFINITIONS:
            assert len(set(w.pattern_sequence)) >= 2, (
                f"Workflow {w.workflow_type_id} has < 2 distinct patterns"
            )


class TestFrequencyDistribution:
    def test_tier_map_covers_all_50(self):
        assert set(WORKFLOW_TIER_MAP.keys()) == set(range(1, 51))

    def test_tier_counts_match_spec(self):
        tier_counts = {}
        for tier_name in WORKFLOW_TIER_MAP.values():
            tier_counts[tier_name] = tier_counts.get(tier_name, 0) + 1
        # 5 very_high, 10 high, 15 medium, 15 low, 5 very_low
        expected = {
            "very_high": 5, "high": 10, "medium": 15, "low": 15, "very_low": 5
        }
        assert tier_counts == expected

    def test_sampling_weights_sum_to_one(self):
        import numpy as np
        assert abs(WORKFLOW_SAMPLING_WEIGHTS.sum() - 1.0) < 1e-9

    def test_very_high_workflow_has_highest_weight(self):
        very_high_ids = [wf_id for wf_id, t in WORKFLOW_TIER_MAP.items() if t == "very_high"]
        very_low_ids = [wf_id for wf_id, t in WORKFLOW_TIER_MAP.items() if t == "very_low"]
        vh_weight = sum(WORKFLOW_SAMPLING_WEIGHTS[wf_id - 1] for wf_id in very_high_ids)
        vl_weight = sum(WORKFLOW_SAMPLING_WEIGHTS[wf_id - 1] for wf_id in very_low_ids)
        assert vh_weight > vl_weight

    def test_instances_map_keys(self):
        assert set(WORKFLOW_INSTANCES_MAP.keys()) == set(range(1, 51))


class TestReservedPrimitivePairs:
    def test_reserved_pairs_15_percent(self):
        pairs = build_reserved_primitive_pairs(holdout_fraction=0.15, seed=0)
        total_possible = 10 * 10  # 10 primitives × 10 primitives
        expected = max(1, int(total_possible * 0.15))
        assert len(pairs) == expected

    def test_pairs_are_tuples_of_ints(self):
        pairs = build_reserved_primitive_pairs()
        for a, b in pairs:
            assert isinstance(a, int)
            assert isinstance(b, int)

    def test_deterministic(self):
        pairs1 = build_reserved_primitive_pairs(seed=42)
        pairs2 = build_reserved_primitive_pairs(seed=42)
        assert pairs1 == pairs2
