from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field


class FakeonCertification(BaseModel):
    Re_alpha_at_M2: float
    fakeon_virtualized: bool
    trajectory: Any
    status: Literal["VERIFIED", "PENDING"]
    t_grid: Any | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class UnifiedMeshResults(BaseModel):
    jax: Any | None = None
    torch: Any | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MeshExecutionScheme(BaseModel):
    mesh_axes: tuple[str, ...] = Field(default=("data",))
    fsdp_enabled: bool = False
    fsdp_sharding_strategy: str = "FULL_SHARD"
    checkpoint_backend: Literal["orbax", "torch"] = "orbax"
