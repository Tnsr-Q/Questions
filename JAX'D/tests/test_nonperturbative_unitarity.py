import numpy as np

from src.bootstrap_solver import check_crossing_symmetry, discretized_bootstrap


def test_bootstrap_unitarity_residual():
    """Im M = M†M on discretized grid."""
    grid = discretized_bootstrap(N_s=50, N_t=30)
    residuals = grid.unitarity_residuals()
    assert np.max(np.abs(residuals)) < 5e-4, "Bootstrap unitarity not converged"


def test_crossing_symmetry():
    """M(s,t) ≈ M(t,s) within numerical tolerance."""
    grid = discretized_bootstrap(N_s=50, N_t=30)
    M_st = grid.amplitude_matrix()
    error = np.max(np.abs(M_st - M_st.T))
    assert error < 1e-5, "Crossing symmetry violated beyond tolerance"
    assert check_crossing_symmetry(M_st)
