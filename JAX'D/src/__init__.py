"""QÜFT-Engine: Proto/Schema & Unified Mesh Architecture."""

from .discovery.theory_space import TheoryHypothesis, TheorySpaceExplorer
from .mesh.topology import DeviceTopology, JAXMeshAdapter, MeshAxis, PyTorchMeshAdapter
from .mesh.unified_mesh import UnifiedMesh
from .proto.constraint_schema import AssumptionTag, ConstraintRole, PhysicsPredicate, StatusLevel
from .proto.registry import PredicateRegistry
from .proto.serializer import Serializer
from .truth.universality_kernel import UniversalityKernel
from .rl_conjecture_loop import AletheiaAgent, TheorySpaceEnv
from .unified_topology import HybridTopologyManager


__version__ = "2.0.0"

__all__ = [
    "PhysicsPredicate",
    "StatusLevel",
    "ConstraintRole",
    "AssumptionTag",
    "PredicateRegistry",
    "Serializer",
    "DeviceTopology",
    "JAXMeshAdapter",
    "PyTorchMeshAdapter",
    "MeshAxis",
    "UnifiedMesh",
    "TheoryHypothesis",
    "TheorySpaceExplorer",
    "UniversalityKernel",
    "HybridTopologyManager",
    "TheorySpaceEnv",
    "AletheiaAgent",
]
