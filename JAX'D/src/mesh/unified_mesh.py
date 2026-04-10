from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, Optional

import numpy as np

from .topology import DeviceTopology, JAXMeshAdapter, MeshAxis, PyTorchMeshAdapter
from src.proto.return_schemas import MeshExecutionScheme, UnifiedMeshResults
from src.proto.schema_enforcer import enforce_schema

log = logging.getLogger("QUFT_UnifiedMesh")

_JAX_PARAMS = set(inspect.signature(JAXMeshAdapter.__init__).parameters) - {"self"}
_TORCH_PARAMS = set(inspect.signature(PyTorchMeshAdapter.__init__).parameters) - {"self"}


class UnifiedMesh:
    """Framework-agnostic mesh manager for JAX and PyTorch backends."""

    _instance: Optional["UnifiedMesh"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._jax_mesh: Optional[JAXMeshAdapter] = None
        self._torch_mesh: Optional[PyTorchMeshAdapter] = None
        self._active_backend: Optional[str] = None
        self._execution_scheme = MeshExecutionScheme()

    def initialize(self, backend: str = "auto", **kwargs):
        jax_kwargs = {k: v for k, v in kwargs.items() if k in _JAX_PARAMS}
        torch_kwargs = {k: v for k, v in kwargs.items() if k in _TORCH_PARAMS}

        if "mesh_axes" in kwargs:
            self._execution_scheme = self._execution_scheme.model_copy(update={"mesh_axes": tuple(kwargs["mesh_axes"])})

        if backend in {"auto", "jax"}:
            try:
                self._jax_mesh = JAXMeshAdapter(**jax_kwargs)
                log.info("Initialized JAX mesh: %s devices", self._jax_mesh.get_device_count())
                self._active_backend = self._active_backend or "jax"
            except Exception as exc:  # best-effort initialization
                log.warning("JAX mesh initialization skipped: %s", exc)

        if backend in {"auto", "pytorch"}:
            try:
                self._torch_mesh = PyTorchMeshAdapter(**torch_kwargs)
                log.info("Initialized PyTorch mesh: %s ranks", self._torch_mesh.get_world_size())
                self._active_backend = self._active_backend or "pytorch"
            except Exception as exc:  # best-effort initialization
                log.warning("PyTorch mesh initialization skipped: %s", exc)

        if self._active_backend is None:
            raise RuntimeError("No distributed backend available")

    @property
    def active_backend(self) -> Optional[str]:
        return self._active_backend

    def get_topology(self, framework: Optional[str] = None) -> DeviceTopology:
        fw = framework or self._active_backend
        if fw == "jax" and self._jax_mesh:
            return self._jax_mesh
        if fw == "pytorch" and self._torch_mesh:
            return self._torch_mesh
        raise ValueError(f"No topology available for framework: {fw}")

    @enforce_schema(UnifiedMeshResults)
    def co_schedule(
        self,
        jax_fn: Optional[Callable[[Dict[str, Any]], Any]],
        torch_fn: Optional[Callable[[Dict[str, Any]], Any]],
        shared_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self._jax_mesh:
            self._jax_mesh.barrier()
        if self._torch_mesh:
            self._torch_mesh.barrier()

        results: Dict[str, Any] = {}

        if self._jax_mesh and jax_fn:
            import jax.numpy as jnp

            jax_inputs = {
                key: jnp.array(value) if isinstance(value, (list, np.ndarray)) else value
                for key, value in shared_data.items()
            }
            results["jax"] = jax_fn(jax_inputs)

        if self._torch_mesh and torch_fn:
            import torch

            torch_inputs = {
                key: torch.tensor(value) if isinstance(value, (list, np.ndarray)) else value
                for key, value in shared_data.items()
            }
            results["torch"] = torch_fn(torch_inputs)

        if self._jax_mesh:
            self._jax_mesh.barrier()
        if self._torch_mesh:
            self._torch_mesh.barrier()

        return results

    @enforce_schema(UnifiedMeshResults)
    def shard_across_frameworks(
        self, data: Any, axis: MeshAxis, jax_spec: tuple, torch_spec: tuple
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        if self._jax_mesh:
            result["jax"] = self._jax_mesh.shard_tensor(data, axis, jax_spec)

        if self._torch_mesh:
            result["torch"] = self._torch_mesh.shard_tensor(data, axis, torch_spec)

        return result

    def get_execution_scheme(self) -> dict[str, Any]:
        """Return the active JAX/PyTorch-FSDP mesh scheme."""
        return self._execution_scheme.model_dump()
