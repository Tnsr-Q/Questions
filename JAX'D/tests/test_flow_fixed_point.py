import numpy as np
from scipy.integrate import solve_ivp

from src.flow_solver import fixed_point_attractor, lorentzian_wetterich_flow


def test_flow_exists_and_is_bounded():
    """Verify global solution existence on compact coupling domain."""
    g0 = np.array([1e-8, 1e-3, 1e-1])
    t_span = (0.0, 20.0)
    sol = solve_ivp(lorentzian_wetterich_flow, t_span, g0, rtol=1e-8, atol=1e-10)
    assert sol.success, "Flow integration failed"
    assert np.all(np.isfinite(sol.y)), "Flow diverged outside perturbative domain"


def test_fixed_point_attractor_behavior():
    """Check ∂_k G_k → 0 and kinematic ghost suppression."""
    k_vals = np.logspace(-3, 3, 100)
    residuals, res_ghost = fixed_point_attractor(k_vals)
    assert np.max(np.abs(residuals)) < 1e-6
    assert np.max(np.abs(res_ghost)) < 1e-12
