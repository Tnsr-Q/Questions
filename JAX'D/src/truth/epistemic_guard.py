from __future__ import annotations

from enum import Enum, auto

from src.proto.constraint_schema import AssumptionTag, PhysicsPredicate, StatusLevel


class EpistemicBoundary(Enum):
    PROVED = auto()
    VERIFIED = auto()
    DEMONSTRATED = auto()
    SPECULATIVE = auto()
    METAPHYSICAL = auto()


def enforce_boundary(predicate: PhysicsPredicate, claim: str) -> bool:
    if predicate.status == StatusLevel.PENDING and claim in {"PROVED", "VERIFIED"}:
        return False
    if AssumptionTag.A2_FAKEON_VALID not in predicate.assumptions and "fakeon" in claim.lower():
        return False
    return True
