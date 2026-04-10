from __future__ import annotations

from typing import Any

from src.proto.return_schemas import MeshExecutionScheme


def detect_fsdp(model: Any) -> bool:
    """Best-effort detection for a PyTorch Lightning model wrapped in FSDP."""
    cls_name = model.__class__.__name__.lower()
    if "fullyshardeddataparallel" in cls_name or "fsdp" in cls_name:
        return True
    wrapped = getattr(model, "module", None)
    if wrapped is not None:
        wrapped_name = wrapped.__class__.__name__.lower()
        return "fullyshardeddataparallel" in wrapped_name or "fsdp" in wrapped_name
    return False


def unify_jax_fsdp_scheme(mesh_axes: tuple[str, ...] = ("data",), model: Any | None = None) -> dict[str, Any]:
    """Return a normalized scheme spanning JAX mesh config and PL/FSDP state."""
    scheme = MeshExecutionScheme(mesh_axes=mesh_axes)
    if model is not None and detect_fsdp(model):
        scheme = scheme.model_copy(update={"fsdp_enabled": True})
    return scheme.model_dump()
