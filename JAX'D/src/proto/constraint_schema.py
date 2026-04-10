from __future__ import annotations

from datetime import datetime
from enum import Enum
import json
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StatusLevel(str, Enum):
    """Epistemic status for physics predicates."""

    PROVED = "PROVED"
    VERIFIED = "VERIFIED"
    CALCULATED = "CALCULATED"
    DEMONSTRATED = "DEMONSTRATED"
    PENDING = "PENDING"


class ConstraintRole(str, Enum):
    """Logical role of a predicate in the verification stack."""

    SELECTOR = "SELECTOR"
    CLOSURE = "CLOSURE"
    CONSISTENCY = "CONSISTENCY"


class AssumptionTag(str, Enum):
    """Explicit assumption boundaries."""

    A1_PERTURBATIVE = "A1_perturbative"
    A2_FAKEON_VALID = "A2_fakeon_valid"
    A3_PALATINI_COMPAT = "A3_palatini"
    A4_SCALE_INVARIANT = "A4_scale_invariant"
    A5_PORTAL_DOMINANCE = "A5_portal_dominance"


class PhysicsPredicate(BaseModel):
    """Typed contract for physics claims in the verification suite."""

    predicate_id: str = Field(pattern=r"^[A-Z][A-Za-z0-9_]+$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    statement: str
    mathematical_form: Optional[str] = None
    assumptions: List[AssumptionTag] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)

    tolerance: float = Field(gt=0)
    residual: Optional[float] = None
    status: StatusLevel = StatusLevel.PENDING
    metadata: Dict[str, Any] = Field(default_factory=dict)

    boundary_checks: Dict[str, bool] = Field(default_factory=dict)

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        json_encoders={np.floating: float, np.integer: int},
    )

    @field_validator("residual")
    @classmethod
    def residual_must_respect_tolerance(cls, value: Optional[float], info):
        tolerance = info.data.get("tolerance")
        if value is not None and tolerance is not None and abs(value) > tolerance * 1.01:
            raise ValueError("Residual exceeds tolerance")
        return value

    @model_validator(mode="after")
    def status_requires_residual(self) -> "PhysicsPredicate":
        if self.status in {StatusLevel.VERIFIED, StatusLevel.CALCULATED} and self.residual is None:
            raise ValueError(f"{self.status.value} requires residual value")
        return self

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True)

    def to_protobuf(self) -> bytes:
        """Serialize to protobuf-compatible bytes (JSON wire fallback)."""
        return json.dumps(self.model_dump(exclude_none=True), default=str).encode("utf-8")

    def check_assumption_boundaries(self) -> Dict[AssumptionTag, bool]:
        """Evaluate tagged assumptions against current numeric metadata."""
        results: Dict[AssumptionTag, bool] = {}

        for tag in self.assumptions:
            if tag == AssumptionTag.A1_PERTURBATIVE:
                f2 = self.metadata.get("f2_value")
                results[tag] = bool(f2 is not None and f2 < 1.0)
            elif tag == AssumptionTag.A2_FAKEON_VALID:
                results[tag] = bool(self.metadata.get("ghost_pole_virtual", False))
            elif tag == AssumptionTag.A3_PALATINI_COMPAT:
                results[tag] = bool(self.metadata.get("palatini_compatible", False))
            elif tag == AssumptionTag.A4_SCALE_INVARIANT:
                results[tag] = bool(self.metadata.get("scale_invariant", False))
            elif tag == AssumptionTag.A5_PORTAL_DOMINANCE:
                results[tag] = bool(self.metadata.get("portal_dominance", False))
            else:
                results[tag] = False

        return results
