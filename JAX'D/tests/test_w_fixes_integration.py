import numpy as np


def test_unify_jax_fsdp_scheme_flags_fsdp_from_class_name():
    from src.mesh.schemes import unify_jax_fsdp_scheme

    FSDPType = type("FullyShardedDataParallel", (), {})
    model = FSDPType()
    scheme = unify_jax_fsdp_scheme(("data", "model"), model=model)

    assert scheme["mesh_axes"] == ("data", "model")
    assert scheme["fsdp_enabled"] is True
    assert scheme["checkpoint_backend"] == "orbax"


def test_orbax_atomic_state_io_json_fallback(tmp_path):
    from src.proto.orbax_atomic import OrbaxAtomicStateIO

    path = tmp_path / "state.json"
    io = OrbaxAtomicStateIO(str(path))
    state = {"mu_global": 0.1, "L_global": 0.5}
    io.save(state)

    restored = io.restore()
    assert restored["mu_global"] == state["mu_global"]
    assert restored["L_global"] == state["L_global"]


def test_jax_hessian_adaptive_k_and_ledger_injection():
    jnp = __import__("jax.numpy", fromlist=["array"])

    from src.hessian_jax import JAXHessianEstimator

    class _Ledger:
        def __init__(self):
            self.updated = False

        def get_tolerance(self, key):
            assert key == "hessian_pl"
            return 1e-5

        def update_from_residual(self, key, residual, solver_id):
            assert key == "hessian_pl"
            assert solver_id == "jax_hessian"
            self.updated = True
            return residual

    ledger = _Ledger()

    def constraint(theta):
        return jnp.array([theta[0] + 2.0 * theta[1], theta[0] - theta[1]])

    est = JAXHessianEstimator(constraint_fn=constraint, weights=jnp.array([1.0, 1.0]), tolerance_ledger=ledger)
    theta = jnp.array([0.2, -0.3])

    eigs = est.lanczos_eigenvalues(theta, k=2, adaptive_k=True, max_iter=10)
    cert = est.verify_pl_condition(theta, loss_val=0.8)

    assert eigs.shape[0] >= 1
    assert np.isfinite(np.asarray(eigs)).all()
    assert "mu_lb" in cert
    assert ledger.updated is True
