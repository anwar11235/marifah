"""Unit tests for every primitive (§6.3 gate)."""

import pytest
import numpy as np

from marifah.data.synthetic.primitives import (
    PrimitiveType,
    apply_primitive,
    sample_attrs,
)

RNG = np.random.default_rng(0)


class TestConditional:
    def test_positive_branch1(self):
        attrs = {"condition": "positive"}
        r = apply_primitive(PrimitiveType.CONDITIONAL, 5, attrs)
        assert r.branch_taken == 1
        assert r.output_state == 1

    def test_positive_branch0(self):
        attrs = {"condition": "positive"}
        r = apply_primitive(PrimitiveType.CONDITIONAL, -3, attrs)
        assert r.branch_taken == 0

    def test_even_condition(self):
        r = apply_primitive(PrimitiveType.CONDITIONAL, 4, {"condition": "even"})
        assert r.branch_taken == 1
        r2 = apply_primitive(PrimitiveType.CONDITIONAL, 3, {"condition": "even"})
        assert r2.branch_taken == 0

    def test_zero_condition(self):
        r = apply_primitive(PrimitiveType.CONDITIONAL, 0, {"condition": "zero"})
        assert r.branch_taken == 1

    def test_non_negative(self):
        r = apply_primitive(PrimitiveType.CONDITIONAL, 0, {"condition": "non_negative"})
        assert r.branch_taken == 1
        r2 = apply_primitive(PrimitiveType.CONDITIONAL, -1, {"condition": "non_negative"})
        assert r2.branch_taken == 0


class TestAggregate:
    def test_sum(self):
        r = apply_primitive(PrimitiveType.AGGREGATE, [1, 2, 3], {"agg_fn": "sum"})
        assert r.output_state == 6

    def test_count(self):
        r = apply_primitive(PrimitiveType.AGGREGATE, [10, 20, 30], {"agg_fn": "count"})
        assert r.output_state == 3

    def test_max(self):
        r = apply_primitive(PrimitiveType.AGGREGATE, [5, 2, 9], {"agg_fn": "max"})
        assert r.output_state == 9

    def test_min(self):
        r = apply_primitive(PrimitiveType.AGGREGATE, [5, 2, 9], {"agg_fn": "min"})
        assert r.output_state == 2

    def test_empty(self):
        r = apply_primitive(PrimitiveType.AGGREGATE, [], {"agg_fn": "sum"})
        assert r.output_state == 0

    def test_single(self):
        r = apply_primitive(PrimitiveType.AGGREGATE, 7, {"agg_fn": "sum"})
        assert r.output_state == 7


class TestLookup:
    def test_basic(self):
        attrs = {"table": {0: 42, 1: 99}}
        r = apply_primitive(PrimitiveType.LOOKUP, 0, attrs)
        assert r.output_state == 42

    def test_key_wrapping(self):
        attrs = {"table": {0: 10, 1: 20}}
        r = apply_primitive(PrimitiveType.LOOKUP, 2, attrs)  # 2 % 2 = 0
        assert r.output_state == 10

    def test_missing_key_defaults(self):
        attrs = {"table": {5: 77}}
        r = apply_primitive(PrimitiveType.LOOKUP, 0, attrs)  # 0 % 1 = 0, key 0 missing
        assert r.output_state == 0  # default


class TestCompare:
    def test_less(self):
        r = apply_primitive(PrimitiveType.COMPARE, [3, 7], {})
        assert r.output_state == -1

    def test_equal(self):
        r = apply_primitive(PrimitiveType.COMPARE, [5, 5], {})
        assert r.output_state == 0

    def test_greater(self):
        r = apply_primitive(PrimitiveType.COMPARE, [9, 3], {})
        assert r.output_state == 1


class TestTransform:
    def test_increment(self):
        r = apply_primitive(PrimitiveType.TRANSFORM, 5, {"transform_fn": "increment"})
        assert r.output_state == 6

    def test_double(self):
        r = apply_primitive(PrimitiveType.TRANSFORM, 4, {"transform_fn": "double"})
        assert r.output_state == 8

    def test_negate(self):
        r = apply_primitive(PrimitiveType.TRANSFORM, 3, {"transform_fn": "negate"})
        assert r.output_state == -3

    def test_absolute(self):
        r = apply_primitive(PrimitiveType.TRANSFORM, -7, {"transform_fn": "absolute"})
        assert r.output_state == 7

    def test_square(self):
        r = apply_primitive(PrimitiveType.TRANSFORM, 3, {"transform_fn": "square"})
        assert r.output_state == 9


class TestValidate:
    def test_positive_true(self):
        r = apply_primitive(PrimitiveType.VALIDATE, 5, {"constraint": "positive"})
        assert r.output_state is True

    def test_positive_false(self):
        r = apply_primitive(PrimitiveType.VALIDATE, -1, {"constraint": "positive"})
        assert r.output_state is False

    def test_non_negative_zero(self):
        r = apply_primitive(PrimitiveType.VALIDATE, 0, {"constraint": "non_negative"})
        assert r.output_state is True

    def test_even(self):
        r = apply_primitive(PrimitiveType.VALIDATE, 4, {"constraint": "even"})
        assert r.output_state is True

    def test_in_range(self):
        r = apply_primitive(PrimitiveType.VALIDATE, 50, {"constraint": "in_range_0_100"})
        assert r.output_state is True
        r2 = apply_primitive(PrimitiveType.VALIDATE, 101, {"constraint": "in_range_0_100"})
        assert r2.output_state is False


class TestRoute:
    def test_basic_routing(self):
        attrs = {"num_branches": 3}
        r = apply_primitive(PrimitiveType.ROUTE, 7, attrs)  # 7 % 3 = 1
        assert r.branch_taken == 1

    def test_zero_routes_to_zero(self):
        attrs = {"num_branches": 4}
        r = apply_primitive(PrimitiveType.ROUTE, 0, attrs)
        assert r.branch_taken == 0

    def test_negative_handled(self):
        attrs = {"num_branches": 2}
        r = apply_primitive(PrimitiveType.ROUTE, -3, attrs)  # abs(-3) % 2 = 1
        assert r.branch_taken == 1


class TestTerminate:
    def test_passthrough(self):
        r = apply_primitive(PrimitiveType.TERMINATE, 42, {})
        assert r.output_state == 42
        assert r.branch_taken is None

    def test_any_state(self):
        r = apply_primitive(PrimitiveType.TERMINATE, [1, 2], {})
        assert r.output_state == [1, 2]


class TestAccumulate:
    def test_step(self):
        r = apply_primitive(PrimitiveType.ACCUMULATE, 10, {"step_value": 5})
        assert r.output_state == 15

    def test_default_step(self):
        r = apply_primitive(PrimitiveType.ACCUMULATE, 0, {})
        assert r.output_state == 1


class TestNop:
    def test_identity(self):
        r = apply_primitive(PrimitiveType.NOP, 99, {})
        assert r.output_state == 99

    def test_list_identity(self):
        r = apply_primitive(PrimitiveType.NOP, [1, 2, 3], {})
        assert r.output_state == [1, 2, 3]


class TestSampleAttrs:
    def test_all_primitives_sample(self):
        rng = np.random.default_rng(42)
        for prim in PrimitiveType:
            kwargs = {"num_branches": 3} if prim == PrimitiveType.ROUTE else {}
            attrs = sample_attrs(prim, rng, **kwargs)
            assert isinstance(attrs, dict)
