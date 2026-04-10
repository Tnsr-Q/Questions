import numpy as np

from src.spectral_density import check_kallen_lehmann, spectral_density_fakeon_graviton


def test_fakeon_physical_spectral_density():
    """ρ_GF^(1)(μ²) = 0 ∀ μ² > 0."""
    mu2_vals = np.linspace(0.1, 1e6, 500)
    rho = spectral_density_fakeon_graviton(mu2_vals)
    assert np.all(np.abs(rho) < 1e-14), "Non-zero physical ghost spectral density"


def test_kallen_lehmann_positivity():
    """∫ρ(μ²)dμ² = 1 and ρ ≥ 0 on physical subspace."""
    mu2_vals = np.logspace(-2, 5, 1000)
    rho_phys = np.exp(-mu2_vals / 1e4) / 1e4
    integral = np.trapz(rho_phys, mu2_vals)
    assert integral > 0.99 and integral < 1.01
    assert np.all(rho_phys >= -1e-12)
    assert check_kallen_lehmann(mu2_vals, rho_phys)
