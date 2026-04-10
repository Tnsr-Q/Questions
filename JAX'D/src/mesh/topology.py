from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np


class DeviceType(Enum):
    CPU = auto()
    GPU = auto()
    TPU = auto()
    IPU = auto()


class MeshAxis(Enum):
    DATA = "data"
    MODEL = "model"
    PIPELINE = "pipeline"
    EXPERT = "expert"
    SPECTRAL = "spectral"


class DeviceTopology(ABC):
    """Abstract distributed device coordination API."""

    @abstractmethod
    def get_device_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_device_type(self) -> DeviceType:
        raise NotImplementedError

    @abstractmethod
    def shard_tensor(self, tensor, axis: MeshAxis, partition_spec: Tuple[str, ...]):
        raise NotImplementedError

    @abstractmethod
    def all_reduce(self, tensor, op: str = "sum"):
        raise NotImplementedError

    @abstractmethod
    def barrier(self):
        raise NotImplementedError

    @abstractmethod
    def get_rank(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_world_size(self) -> int:
        raise NotImplementedError


class JAXMeshAdapter(DeviceTopology):
    """Adapter around JAX mesh primitives."""

    def __init__(self, mesh_axes: Tuple[str, ...] = ("data",), devices: Optional[List] = None):
        import jax
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        self.devices = devices or jax.devices()
        if not mesh_axes:
            raise ValueError("mesh_axes must contain at least one axis name")
        n_axes = len(mesh_axes)
        shape = (-1,) + (1,) * (n_axes - 1) if n_axes > 1 else (-1,)
        self.mesh = Mesh(np.array(self.devices).reshape(shape), mesh_axes)
        self.mesh_axes = mesh_axes
        self.PartitionSpec = PartitionSpec
        self.NamedSharding = NamedSharding

    def get_device_count(self) -> int:
        return len(self.devices)

    def get_device_type(self) -> DeviceType:
        import jax

        backend = jax.default_backend()
        if backend == "gpu":
            return DeviceType.GPU
        if backend == "tpu":
            return DeviceType.TPU
        return DeviceType.CPU

    def shard_tensor(self, tensor, axis: MeshAxis, partition_spec: Tuple[str, ...]):
        _ = axis
        from jax import device_put
        from jax.sharding import PartitionSpec

        sharding = self.NamedSharding(self.mesh, PartitionSpec(*partition_spec))
        return device_put(tensor, sharding)

    def all_reduce(self, tensor, op: str = "sum"):
        from jax.lax import pmean, psum

        if op == "sum":
            return psum(tensor, axis_name=self.mesh_axes[0])
        if op == "mean":
            return pmean(tensor, axis_name=self.mesh_axes[0])
        return tensor

    def barrier(self):
        import jax

        jax.device_get(jax.device_put(0))

    def get_rank(self) -> int:
        import jax

        ids = jax.local_device_ids()
        return ids[0] if ids else 0

    def get_world_size(self) -> int:
        return len(self.devices)


class PyTorchMeshAdapter(DeviceTopology):
    """Adapter for PyTorch distributed APIs."""

    def __init__(self, backend: str = "gloo", init_method: Optional[str] = None):
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method=init_method)
        self.dist = dist

    def get_device_count(self) -> int:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 1

    def get_device_type(self) -> DeviceType:
        import torch

        return DeviceType.GPU if torch.cuda.is_available() else DeviceType.CPU

    def shard_tensor(self, tensor, axis: MeshAxis, partition_spec: Tuple[str, ...]):
        _ = (axis, partition_spec)
        return tensor

    def all_reduce(self, tensor, op: str = "sum"):
        if op == "sum":
            self.dist.all_reduce(tensor, op=self.dist.ReduceOp.SUM)
        elif op == "mean":
            self.dist.all_reduce(tensor, op=self.dist.ReduceOp.SUM)
            tensor /= self.dist.get_world_size()
        return tensor

    def barrier(self):
        self.dist.barrier()

    def get_rank(self) -> int:
        return self.dist.get_rank()

    def get_world_size(self) -> int:
        return self.dist.get_world_size()
