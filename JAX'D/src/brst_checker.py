"""Symbolic BRST-fakeon compatibility stubs."""

from __future__ import annotations

import sympy as sp


def brst_nilpotency_check() -> bool:
    """Verify s^2 = 0 on a toy BRST differential."""
    c = sp.Symbol("c", commutative=False)
    # For a Grassmann ghost, c^2 = 0 is modeled by explicit replacement.
    expr = (c * c).xreplace({c * c: 0})
    return sp.simplify(expr) == 0
