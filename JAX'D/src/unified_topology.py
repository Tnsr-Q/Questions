"""Hybrid classical/quantum topology manager."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Optional

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np

from src.hessian_qjax import QJAXHessianEstimator


@dataclass
class ClassicalHessianExecutor:
    """Small classical-executor placeholder for routing decisions."""

    mesh: Mesh
    sharding: NamedSharding

    mode: str = "classical"


class HybridTopologyManager:
    """Unified contract for classical (JAX) and optional qjax resources."""

    def __init__(
        self,
        mesh_axes: tuple[str, ...] = ("data",),
        qpu_count: int = 0,
        condition_threshold: float = 100.0,
        qjax_runtime: Optional[ModuleType] = None,
    ):
        self.n_devices = jax.device_count()
        self.n_qpus = max(0, int(qpu_count))
        self.condition_threshold = float(condition_threshold)

        if not mesh_axes:
            raise ValueError("mesh_axes must be a non-empty tuple of mesh axis names.")
        devices = np.array(jax.devices()).reshape((-1,) + (1,) * (len(mesh_axes) - 1))
        self.classical_mesh = Mesh(devices, mesh_axes)
        self.classical_sharding = NamedSharding(self.classical_mesh, PartitionSpec(*mesh_axes))

        self.qjax_runtime = qjax_runtime
        self.quantum_mesh: Optional[Any] = None
        if self.n_qpus > 0:
            self._init_quantum_mesh()

    def _init_quantum_mesh(self) -> None:
        runtime = self.qjax_runtime
        if runtime is None:
            try:
                import qjax as runtime  # type: ignore
            except ModuleNotFoundError:
                return
        if not hasattr(runtime, "list_devices") or not hasattr(runtime, "Mesh"):
            return

        quantum_devices = runtime.list_devices()[: self.n_qpus]
        if not quantum_devices:
            return
        self.quantum_mesh = runtime.Mesh(quantum_devices, ("qubits",))
        self.qjax_runtime = runtime

    def select_hessian_backend(self, condition_number: float) -> str:
        if self.quantum_mesh is not None and condition_number > self.condition_threshold:
            return "quantum"
        return "classical"

    def get_hessian_executor(self, condition_number: float):
        backend = self.select_hessian_backend(condition_number)
        if backend == "quantum":
            return QJAXHessianEstimator(mesh=self.quantum_mesh, runtime=self.qjax_runtime)
        return ClassicalHessianExecutor(mesh=self.classical_mesh, sharding=self.classical_sharding)
