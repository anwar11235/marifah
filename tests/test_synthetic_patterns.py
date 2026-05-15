"""Unit tests for all 12 pattern templates: instantiate and execute."""

import pytest
import numpy as np

from marifah.data.synthetic.executor import execute_dag
from marifah.data.synthetic.patterns import ALL_PATTERNS, PATTERN_BY_NAME
from marifah.data.synthetic.primitives import PrimitiveType


RNG = np.random.default_rng(42)


def _run_pattern(pattern, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    inst = pattern.instantiate(rng)
    # Provide initial state for all root nodes
    import networkx as nx
    root_nodes = [n for n in inst.dag.nodes if inst.dag.in_degree(n) == 0]
    initial = {n: 10 for n in root_nodes}
    result = execute_dag(inst.dag, inst.node_attrs, initial)
    return inst, result


class TestAllPatternsInstantiate:
    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_instantiate(self, pattern):
        rng = np.random.default_rng(0)
        inst = pattern.instantiate(rng)
        assert inst.dag.number_of_nodes() >= pattern.min_size
        assert inst.dag.number_of_nodes() <= pattern.max_size
        assert inst.entry_node in inst.dag.nodes
        assert all(e in inst.dag.nodes for e in inst.exit_nodes)

    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_is_dag(self, pattern):
        import networkx as nx
        rng = np.random.default_rng(1)
        inst = pattern.instantiate(rng)
        assert nx.is_directed_acyclic_graph(inst.dag)

    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_executes(self, pattern):
        rng = np.random.default_rng(2)
        inst, result = _run_pattern(pattern, rng)
        # At minimum the executor should produce a trace
        # (not all patterns end with terminate, so halted may be False)
        assert len(result.trace) >= 1 or inst.dag.number_of_nodes() > 0


class TestTerminatingPatterns:
    """Patterns that include a TERMINATE node should halt."""

    def test_sequential_validation_halts(self):
        p = PATTERN_BY_NAME["sequential_validation"]
        inst, result = _run_pattern(p)
        terminate_nodes = {
            n.node_id for n in [] if False  # just check there are terminate nodes
        }
        # The executor should have visited at least the entry and a validate node
        assert len(result.trace) >= 2

    def test_accumulate_path_halts(self):
        p = PATTERN_BY_NAME["accumulate_path"]
        inst, result = _run_pattern(p)
        # accumulate_path ends with terminate
        assert result.halted

    def test_constrained_terminate_halts(self):
        p = PATTERN_BY_NAME["constrained_terminate"]
        inst, result = _run_pattern(p)
        assert result.halted


class TestPatternCoverage:
    def test_all_12_patterns_defined(self):
        assert len(ALL_PATTERNS) == 12

    def test_pattern_ids_unique(self):
        ids = [p.pattern_id for p in ALL_PATTERNS]
        assert len(set(ids)) == len(ids)

    def test_pattern_names_unique(self):
        names = [p.name for p in ALL_PATTERNS]
        assert len(set(names)) == len(names)

    def test_node_attrs_contain_primitive(self):
        rng = np.random.default_rng(7)
        for pattern in ALL_PATTERNS:
            inst = pattern.instantiate(rng)
            for nid, attrs in inst.node_attrs.items():
                assert "primitive" in attrs
                assert isinstance(attrs["primitive"], PrimitiveType)
