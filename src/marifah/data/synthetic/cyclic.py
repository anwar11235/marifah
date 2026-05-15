"""§10 — Cyclic-mode scaffolding.

Default config: acyclic-only.  Cyclic mode raises NotImplementedError.
The interface is defined here so future sessions can implement it without
changing the caller signature.
"""

from __future__ import annotations

from typing import Any


class CyclicNotImplementedError(NotImplementedError):
    """Raised when cyclic mode is requested but not yet implemented."""


def validate_acyclic_constraint(allow_cycles: bool) -> None:
    """Raise CyclicNotImplementedError if cycles are requested."""
    if allow_cycles:
        raise CyclicNotImplementedError(
            "Cyclic DAG generation is not implemented in this session. "
            "Set allow_cycles=false in config. "
            "Cyclic handling is deferred pending the vertical decision (spec §10.3)."
        )


def annotate_cycle_edges(dag: Any, cycle_edges: list) -> None:  # type: ignore[type-arg]
    """Annotate back-edges with cycle metadata (stub — future implementation)."""
    raise CyclicNotImplementedError("cycle annotation not implemented")


def check_halt_condition(cycle_id: int, state: Any, halt_condition: str) -> bool:
    """Evaluate a cycle halt condition (stub — future implementation)."""
    raise CyclicNotImplementedError("cycle halt condition not implemented")
