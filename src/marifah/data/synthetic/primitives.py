"""§2 — Primitive vocabulary: 10 generic primitives + augmentation slot."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# State type
# ---------------------------------------------------------------------------

State = Union[int, float, bool, list, dict]

# ---------------------------------------------------------------------------
# Primitive type enumeration
# ---------------------------------------------------------------------------

class PrimitiveType(enum.IntEnum):
    CONDITIONAL = 0
    AGGREGATE   = 1
    LOOKUP      = 2
    COMPARE     = 3
    TRANSFORM   = 4
    VALIDATE    = 5
    ROUTE       = 6
    TERMINATE   = 7
    ACCUMULATE  = 8
    NOP         = 9

NUM_BASE_PRIMITIVES = 10

PRIMITIVE_NAMES: Dict[PrimitiveType, str] = {
    PrimitiveType.CONDITIONAL: "conditional",
    PrimitiveType.AGGREGATE:   "aggregate",
    PrimitiveType.LOOKUP:      "lookup",
    PrimitiveType.COMPARE:     "compare",
    PrimitiveType.TRANSFORM:   "transform",
    PrimitiveType.VALIDATE:    "validate",
    PrimitiveType.ROUTE:       "route",
    PrimitiveType.TERMINATE:   "terminate",
    PrimitiveType.ACCUMULATE:  "accumulate",
    PrimitiveType.NOP:         "nop",
}

NAME_TO_PRIMITIVE: Dict[str, PrimitiveType] = {v: k for k, v in PRIMITIVE_NAMES.items()}

# ---------------------------------------------------------------------------
# Primitive result
# ---------------------------------------------------------------------------

@dataclass
class PrimitiveResult:
    output_state: State
    branch_taken: Optional[int] = None  # set by conditional / route


# ---------------------------------------------------------------------------
# Attribute schemas per primitive
# ---------------------------------------------------------------------------

# conditional attrs: {"condition": str}  where condition ∈ CONDITION_TYPES
CONDITION_TYPES = ("positive", "non_negative", "even", "odd", "zero")

# aggregate attrs: {"agg_fn": str}  where agg_fn ∈ AGGREGATE_FNS
AGGREGATE_FNS = ("sum", "count", "max", "min", "mean")

# lookup attrs: {"table": {int: int}}  — mapping key→value
# The executor normalises the key via key % len(table)

# compare attrs: {} — no extra; compare takes exactly 2 predecessor states

# transform attrs: {"transform_fn": str}  where transform_fn ∈ TRANSFORM_FNS
TRANSFORM_FNS = ("increment", "decrement", "double", "halve", "negate", "absolute", "square")

# validate attrs: {"constraint": str}  where constraint ∈ CONSTRAINT_TYPES
CONSTRAINT_TYPES = ("positive", "non_negative", "even", "non_zero", "in_range_0_100")

# route attrs: {"num_branches": int}  — output branch = int(state) % num_branches

# terminate attrs: {} — no extra

# accumulate attrs: {"step_value": int}  — output = input + step_value
#   Represents a single step in an accumulate_path

# nop attrs: {} — no extra


# ---------------------------------------------------------------------------
# Augmentation slot (§9.3)
# ---------------------------------------------------------------------------

# Vertical-specific primitive augmentations are registered here at config
# load time.  Keys are string names; values are callables matching the
# apply_primitive signature.
_AUGMENTED_PRIMITIVES: Dict[str, Any] = {}


def register_augmented_primitive(name: str, apply_fn: Any) -> None:
    """Register a vertical-specific primitive augmentation."""
    _AUGMENTED_PRIMITIVES[name] = apply_fn


# ---------------------------------------------------------------------------
# Core primitive application
# ---------------------------------------------------------------------------

def _to_numeric(state: State) -> Union[int, float]:
    if isinstance(state, bool):
        return int(state)
    if isinstance(state, (int, float)):
        return state
    if isinstance(state, list) and len(state) > 0:
        return _to_numeric(state[0])
    return 0


def _apply_conditional(state: State, attrs: dict) -> PrimitiveResult:
    condition = attrs.get("condition", "positive")
    v = _to_numeric(state)
    if condition == "positive":
        branch = 1 if v > 0 else 0
    elif condition == "non_negative":
        branch = 1 if v >= 0 else 0
    elif condition == "even":
        branch = 1 if int(v) % 2 == 0 else 0
    elif condition == "odd":
        branch = 1 if int(v) % 2 != 0 else 0
    elif condition == "zero":
        branch = 1 if v == 0 else 0
    else:
        branch = 0
    return PrimitiveResult(output_state=branch, branch_taken=branch)


def _apply_aggregate(states: List[State], attrs: dict) -> PrimitiveResult:
    agg_fn = attrs.get("agg_fn", "sum")
    nums = [_to_numeric(s) for s in states]
    if not nums:
        return PrimitiveResult(output_state=0)
    if agg_fn == "sum":
        result: Union[int, float] = sum(nums)
    elif agg_fn == "count":
        result = len(nums)
    elif agg_fn == "max":
        result = max(nums)
    elif agg_fn == "min":
        result = min(nums)
    elif agg_fn == "mean":
        result = sum(nums) / len(nums)
    else:
        result = sum(nums)
    return PrimitiveResult(output_state=int(result) if isinstance(result, float) and result == int(result) else result)


def _apply_lookup(state: State, attrs: dict) -> PrimitiveResult:
    raw_table: dict = attrs.get("table", {0: 0})
    # JSON round-trips dict keys as strings; normalise back to int.
    table = {int(k): v for k, v in raw_table.items()}
    key = int(_to_numeric(state)) % max(len(table), 1)
    value = table.get(key, 0)
    return PrimitiveResult(output_state=value)


def _apply_compare(state_a: State, state_b: State, attrs: dict) -> PrimitiveResult:
    a = _to_numeric(state_a)
    b = _to_numeric(state_b)
    if a < b:
        result = -1
    elif a > b:
        result = 1
    else:
        result = 0
    return PrimitiveResult(output_state=result)


def _apply_transform(state: State, attrs: dict) -> PrimitiveResult:
    fn = attrs.get("transform_fn", "increment")
    v = _to_numeric(state)
    if fn == "increment":
        out: Union[int, float] = v + 1
    elif fn == "decrement":
        out = v - 1
    elif fn == "double":
        out = v * 2
    elif fn == "halve":
        out = v // 2 if isinstance(v, int) else v / 2
    elif fn == "negate":
        out = -v
    elif fn == "absolute":
        out = abs(v)
    elif fn == "square":
        out = v * v
    else:
        out = v
    return PrimitiveResult(output_state=int(out) if isinstance(out, float) and out == int(out) else out)


def _apply_validate(state: State, attrs: dict) -> PrimitiveResult:
    constraint = attrs.get("constraint", "positive")
    v = _to_numeric(state)
    if constraint == "positive":
        result = bool(v > 0)
    elif constraint == "non_negative":
        result = bool(v >= 0)
    elif constraint == "even":
        result = bool(int(v) % 2 == 0)
    elif constraint == "non_zero":
        result = bool(v != 0)
    elif constraint == "in_range_0_100":
        result = bool(0 <= v <= 100)
    else:
        result = True
    return PrimitiveResult(output_state=result)


def _apply_route(state: State, attrs: dict) -> PrimitiveResult:
    num_branches: int = attrs.get("num_branches", 2)
    v = int(_to_numeric(state))
    branch = abs(v) % num_branches
    return PrimitiveResult(output_state=branch, branch_taken=branch)


def _apply_terminate(state: State, attrs: dict) -> PrimitiveResult:
    return PrimitiveResult(output_state=state)


def _apply_accumulate(state: State, attrs: dict) -> PrimitiveResult:
    step_value: int = attrs.get("step_value", 1)
    v = _to_numeric(state)
    result = v + step_value
    return PrimitiveResult(output_state=int(result))


def _apply_nop(state: State, attrs: dict) -> PrimitiveResult:
    return PrimitiveResult(output_state=state)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

def apply_primitive(
    primitive: PrimitiveType,
    input_state: Union[State, List[State]],
    attrs: dict,
) -> PrimitiveResult:
    """Apply a primitive to its input(s) and return the result.

    For AGGREGATE and COMPARE the caller passes a list of states;
    for all others a single state.
    """
    if primitive == PrimitiveType.CONDITIONAL:
        return _apply_conditional(input_state, attrs)  # type: ignore[arg-type]
    elif primitive == PrimitiveType.AGGREGATE:
        states = input_state if isinstance(input_state, list) else [input_state]
        return _apply_aggregate(states, attrs)
    elif primitive == PrimitiveType.LOOKUP:
        return _apply_lookup(input_state, attrs)  # type: ignore[arg-type]
    elif primitive == PrimitiveType.COMPARE:
        states = input_state if isinstance(input_state, list) else [input_state, input_state]
        a = states[0] if len(states) >= 1 else 0
        b = states[1] if len(states) >= 2 else 0
        return _apply_compare(a, b, attrs)
    elif primitive == PrimitiveType.TRANSFORM:
        return _apply_transform(input_state, attrs)  # type: ignore[arg-type]
    elif primitive == PrimitiveType.VALIDATE:
        return _apply_validate(input_state, attrs)  # type: ignore[arg-type]
    elif primitive == PrimitiveType.ROUTE:
        return _apply_route(input_state, attrs)  # type: ignore[arg-type]
    elif primitive == PrimitiveType.TERMINATE:
        return _apply_terminate(input_state, attrs)  # type: ignore[arg-type]
    elif primitive == PrimitiveType.ACCUMULATE:
        return _apply_accumulate(input_state, attrs)  # type: ignore[arg-type]
    elif primitive == PrimitiveType.NOP:
        return _apply_nop(input_state, attrs)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown primitive: {primitive}")


# ---------------------------------------------------------------------------
# Attribute generation helpers (used by patterns.py during instantiation)
# ---------------------------------------------------------------------------

def sample_conditional_attrs(rng: Any) -> dict:
    return {"condition": rng.choice(list(CONDITION_TYPES))}


def sample_aggregate_attrs(rng: Any) -> dict:
    return {"agg_fn": rng.choice(list(AGGREGATE_FNS))}


def sample_lookup_attrs(rng: Any, table_size: int = 8) -> dict:
    keys = list(range(table_size))
    values = [int(rng.integers(0, 100)) for _ in keys]
    return {"table": dict(zip(keys, values))}


def sample_compare_attrs(rng: Any) -> dict:
    return {}


def sample_transform_attrs(rng: Any) -> dict:
    return {"transform_fn": rng.choice(list(TRANSFORM_FNS))}


def sample_validate_attrs(rng: Any) -> dict:
    return {"constraint": rng.choice(list(CONSTRAINT_TYPES))}


def sample_route_attrs(rng: Any, num_branches: int) -> dict:
    return {"num_branches": num_branches}


def sample_terminate_attrs(rng: Any) -> dict:
    return {}


def sample_accumulate_attrs(rng: Any) -> dict:
    return {"step_value": int(rng.integers(1, 10))}


def sample_nop_attrs(rng: Any) -> dict:
    return {}


_ATTR_SAMPLERS = {
    PrimitiveType.CONDITIONAL: lambda rng, **_: sample_conditional_attrs(rng),
    PrimitiveType.AGGREGATE:   lambda rng, **_: sample_aggregate_attrs(rng),
    PrimitiveType.LOOKUP:      lambda rng, **_: sample_lookup_attrs(rng),
    PrimitiveType.COMPARE:     lambda rng, **_: sample_compare_attrs(rng),
    PrimitiveType.TRANSFORM:   lambda rng, **_: sample_transform_attrs(rng),
    PrimitiveType.VALIDATE:    lambda rng, **_: sample_validate_attrs(rng),
    PrimitiveType.ROUTE:       lambda rng, num_branches=2, **_: sample_route_attrs(rng, num_branches),
    PrimitiveType.TERMINATE:   lambda rng, **_: sample_terminate_attrs(rng),
    PrimitiveType.ACCUMULATE:  lambda rng, **_: sample_accumulate_attrs(rng),
    PrimitiveType.NOP:         lambda rng, **_: sample_nop_attrs(rng),
}


def sample_attrs(primitive: PrimitiveType, rng: Any, **kwargs: Any) -> dict:
    return _ATTR_SAMPLERS[primitive](rng, **kwargs)
