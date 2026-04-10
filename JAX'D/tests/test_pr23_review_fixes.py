"""Tests for PR #23 review fixes: kwargs filtering, IR consistency gating,
registry public accessor, and JAX mesh reshape."""

import inspect
from unittest.mock import MagicMock

import numpy as np

from src.proto.constraint_schema import PhysicsPredicate, StatusLevel
from src.proto.registry import PredicateRegistry
from src.mesh.topology import JAXMeshAdapter, PyTorchMeshAdapter
from src.mesh.unified_mesh import _JAX_PARAMS, _TORCH_PARAMS
from src.truth.universality_kernel import UniversalityKernel


# ---------- PredicateRegistry.predicate_ids ----------


def test_predicate_ids_empty():
    registry = PredicateRegistry()
    assert registry.predicate_ids() == []


def test_predicate_ids_returns_registered_ids():
    registry = PredicateRegistry()
    pred = PhysicsPredicate(
        predicate_id="C_ghost",
        version="1.0.0",
        statement="Ghost-free",
        tolerance=0.01,
        residual=0.005,
        status=StatusLevel.VERIFIED,
    )
    registry.register(pred)
    assert "C_ghost" in registry.predicate_ids()


# ---------- kwargs filtering in UnifiedMesh.initialize ----------


def test_jax_params_contains_expected_keys():
    assert "mesh_axes" in _JAX_PARAMS
    assert "devices" in _JAX_PARAMS
    assert "backend" not in _JAX_PARAMS
    assert "init_method" not in _JAX_PARAMS


def test_torch_params_contains_expected_keys():
    assert "backend" in _TORCH_PARAMS
    assert "init_method" in _TORCH_PARAMS
    assert "mesh_axes" not in _TORCH_PARAMS
    assert "devices" not in _TORCH_PARAMS


# ---------- IR consistency gating in scan_f2_space ----------


def _make_registry_with_predicate():
    registry = PredicateRegistry()
    pred = PhysicsPredicate(
        predicate_id="C_ghost",
        version="1.0.0",
        statement="Ghost-free",
        tolerance=0.01,
        residual=0.005,
        status=StatusLevel.VERIFIED,
    )
    registry.register(pred)
    return registry


def test_ir_inconsistent_candidate_skipped():
    """When _check_ir_consistency returns False the candidate should be skipped."""
    registry = _make_registry_with_predicate()

    rge_solver = MagicMock()
    # Return g_ir too short → _check_ir_consistency returns False
    rge_solver.solve_f2.return_value = {"g_ir": [0.129, 0.5, 0.5]}

    bootstrap_solver = MagicMock()
    bootstrap_solver.solve_at_scale.return_value = {"ghost_residual": 0.0}

    kernel = UniversalityKernel(registry, rge_solver, bootstrap_solver)
    result = kernel.scan_f2_space(np.array([1e-8]))

    assert result["status"] == "NO_SOLUTION"
    # bootstrap_solver should never be called when IR check fails
    bootstrap_solver.solve_at_scale.assert_not_called()


def test_ir_consistent_candidate_processed():
    """When _check_ir_consistency returns True the candidate should be evaluated."""
    registry = _make_registry_with_predicate()

    rge_solver = MagicMock()
    rge_solver.solve_f2.return_value = {
        "g_ir": [0.129, 0.5, 0.5, 0.995],
    }

    bootstrap_solver = MagicMock()
    bootstrap_solver.solve_at_scale.return_value = {"ghost_residual": 0.0}

    kernel = UniversalityKernel(registry, rge_solver, bootstrap_solver)
    result = kernel.scan_f2_space(np.array([1e-8]))

    # bootstrap_solver must have been called for the IR-consistent candidate
    bootstrap_solver.solve_at_scale.assert_called_once()
    assert result["status"] in {"UNIQUE_SOLUTION", "NO_SOLUTION"}


# ---------- JAXMeshAdapter reshape derives from mesh_axes ----------


def test_jax_adapter_init_signature_no_extra_params():
    """Ensure JAXMeshAdapter.__init__ only takes mesh_axes and devices."""
    params = set(inspect.signature(JAXMeshAdapter.__init__).parameters) - {"self"}
    assert params == {"mesh_axes", "devices"}


def test_pytorch_adapter_init_signature_no_extra_params():
    """Ensure PyTorchMeshAdapter.__init__ only takes backend and init_method."""
    params = set(inspect.signature(PyTorchMeshAdapter.__init__).parameters) - {"self"}
    assert params == {"backend", "init_method"}
