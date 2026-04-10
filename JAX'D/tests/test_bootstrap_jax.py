import jax.numpy as jnp

from src.bootstrap_solver import DiscretizedBootstrapSolver
from src.hessian_jax import JAXHessianEstimator


def test_bootstrap_unitarity_and_crossing():
    solver = DiscretizedBootstrapSolver(N_s=64, N_l=4, alpha=0.05)
    res = solver.solve(maxiter=200)

    assert res["success"], "Bootstrap optimization failed"
    assert res["residuals"]["unitarity"] < 1e-4, "Unitarity violation in elastic regime"
    assert res["residuals"]["crossing"] < 1e-3, "Crossing symmetry not satisfied"
    assert res["froissart_check"]["satisfied"], "Froissart-Martin bound violated"


def test_jax_hessian_pl_certification():
    def mock_constraints(theta):
        f2, xi, lam = theta
        return jnp.array(
            [
                (f2 / 1e-8) ** (-0.5),
                jnp.exp(-1.2 * (f2 / 1e-8) ** (-0.5)),
                (lam / 3.2e-32),
            ]
        )

    estimator = JAXHessianEstimator(
        constraint_fn=mock_constraints,
        weights=jnp.array([1.0, 1.0, 1.0]),
        reg_lambda=1e-4,
    )
    theta_star = jnp.array([1e-8, 5e8, 3.2e-32])

    eigvals = estimator.lanczos_eigenvalues(theta_star, k=3)
    mu = float(eigvals[0])
    L = float(eigvals[-1])

    assert 1e-5 < mu < 1e-1, f"PL constant {mu} outside expected range"
    assert L < 1.0, f"Lipschitz constant {L} too large for free-tier stability"
    assert L / mu < 5e4, "Condition number too high for PL convergence"

    loss_val = estimator._loss_fn(theta_star)
    pl_check = estimator.verify_pl_condition(theta_star, loss_val, mu_lb=1e-6)
    assert pl_check["pl_satisfied"], f"PL condition failed: mu_est={pl_check['mu_est']}"
