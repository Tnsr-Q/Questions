import numpy as np


class LorentzianSpectralFlow:
    """
    Implements dispersive flow G_k(p) and inelastic dual bootstrap η_l(s).
    Matches S-matrix analysis.txt §§1-3.
    """

    def __init__(self, M2: float = 2.4e23, alpha_inelastic: float = 0.05):
        self.M2 = M2
        self.alpha = alpha_inelastic
        self.m2_phys = 0.0  # Graviton mass

    def spectral_density(self, mu2: float, k: float) -> float:
        """ρ_k(μ²) = ρ(μ²) * k²/(μ² + k²) with ρ(μ²) = δ(μ²) - P.V.δ(μ² - M2²)"""
        # Numerical approximation of delta using narrow Gaussians
        sigma = 1e-12 * max(1.0, mu2)
        delta_phys = np.exp(-0.5 * ((mu2 - self.m2_phys) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        delta_ghost = np.exp(-0.5 * ((mu2 - self.M2**2) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        rho = (delta_phys - delta_ghost) * (k**2 / (mu2 + k**2 + 1e-15))
        return rho

    def inelasticity_profile(self, s: float, l: int = 0) -> float:
        """η_l(s) = exp(-α(s - 4m²)^{l+1}) for s > 4m²"""
        threshold = 4 * (0.1) ** 2  # proxy m² ~ 0.01 GeV²
        if s <= threshold:
            return 1.0
        return np.exp(-self.alpha * (s - threshold) ** (l + 1))

    def froissart_bound_check(self, sigma_tot: float, s_max: float, C: float = 0.25, s0: float = 1.0) -> bool:
        """Verify σ_tot(s) ≤ C ln²(s/s₀)"""
        bound = C * np.log(s_max / s0) ** 2
        return sigma_tot <= bound * 1.001  # 0.1% tolerance

    def unitarity_residual(self, S_l: complex) -> float:
        """|S_l| ≤ 1 check for bootstrap solver"""
        return max(0.0, np.abs(S_l) - 1.0)
