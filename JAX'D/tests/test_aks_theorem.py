import numpy as np

from src.spectral_density import compute_froissart_amplitude, compute_inelasticity


def test_inelasticity_profile_unitarity():
    """|S_l(s)| ≤ 1 for s > 4m²."""
    s_vals = np.logspace(1, 4, 200)
    eta = compute_inelasticity(s_vals, l=0, alpha=0.05)
    assert np.all(eta <= 1.0 + 1e-10), "Unitarity violation in inelastic regime"
    assert np.all(eta >= 0.0), "Negative inelasticity (unphysical)"


def test_froissart_martin_bound():
    """σ_tot(s) ≤ C ln²(s/s0)."""
    s_max = 1e8
    sigma_tot = compute_froissart_amplitude(s_max)
    bound = 0.25 * np.log(s_max / 1.0) ** 2
    assert sigma_tot <= bound * (1 + 1e-3), "Froissart bound violated at high energy"
