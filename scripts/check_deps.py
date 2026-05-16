"""Dependency pre-flight check for Vast.ai instances.

Verifies that all packages required for probe scripts are importable and meet
minimum version requirements.  Exits 0 on success, 1 on any failure.

Usage:
    python scripts/check_deps.py
"""

from __future__ import annotations

import importlib
import sys
from typing import Tuple


_CHECKS: list[Tuple[str, str, str]] = [
    # (import_name, version_attr, min_version)
    ("torch",       "__version__",  "2.4"),
    ("numpy",       "__version__",  "1.24"),
    ("scipy",       "__version__",  "1.10"),
    ("sklearn",     "__version__",  "1.3"),
    ("pyarrow",     "__version__",  "14.0"),
    ("networkx",    "__version__",  "3.0"),
    ("pydantic",    "__version__",  "2.0"),
    ("wandb",       "__version__",  "0.16"),
    ("tqdm",        "__version__",  "4.0"),
]


def _parse_version(v: str) -> Tuple[int, ...]:
    parts = []
    for part in v.split(".")[:3]:
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def main() -> None:
    failures: list[str] = []
    print("Checking dependencies...")

    for import_name, version_attr, min_version in _CHECKS:
        try:
            mod = importlib.import_module(import_name)
        except ImportError as exc:
            failures.append(f"  MISSING  {import_name}: {exc}")
            continue

        actual = getattr(mod, version_attr, "unknown")
        if actual == "unknown":
            print(f"  OK       {import_name} (version unknown)")
            continue

        if _parse_version(actual) < _parse_version(min_version):
            failures.append(
                f"  OLD      {import_name}=={actual}  (requires >={min_version})"
            )
        else:
            print(f"  OK       {import_name}=={actual}")

    # Flash-attn is optional — just report presence
    try:
        import flash_attn  # type: ignore
        print(f"  OK       flash_attn=={getattr(flash_attn, '__version__', '?')} (optional)")
    except ImportError:
        print("  SKIP     flash_attn not installed (optional — SDPA fallback active)")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f)
        print(f"\n{len(failures)} dependency check(s) failed.")
        sys.exit(1)
    else:
        print("\nAll dependency checks passed.")


if __name__ == "__main__":
    main()
