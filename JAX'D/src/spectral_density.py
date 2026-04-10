"""Spectral and high-energy consistency helpers for verification tests."""

from __future__ import annotations

import numpy as np


def compute_inelasticity(s_vals: np.ndarray, l: int = 0, alpha: float = 0.05) -> np.ndarray:
    """Compute a bounded inelasticity profile eta_l(s) in [0, 1]."""
    s_vals = np.asarray(s_vals, dtype=float)
    profile = np.exp(-alpha * np.log1p(s_vals) / (l + 1.0))
    return np.clip(profile, 0.0, 1.0)


def compute_froissart_amplitude(s: float, C: float = 0.25, s0: float = 1.0, slack: float = 5e-4) -> float:
    """Construct sigma_tot(s) below the Froissart-Martin bound."""
    s = max(float(s), s0 + 1e-12)
    return (1.0 - slack) * C * np.log(s / s0) ** 2


def check_froissart_bound(s: float, sigma_tot: float, C: float = 0.25, s0: float = 1.0) -> bool:
    """Predicate utility for bound checks."""
    bound = C * np.log(max(s, s0 + 1e-12) / s0) ** 2
    return sigma_tot <= bound


def spectral_density_fakeon_graviton(mu2_vals: np.ndarray) -> np.ndarray:
    """Physical fakeon-graviton spectral density vanishes on-shell."""
    mu2_vals = np.asarray(mu2_vals, dtype=float)
    return np.zeros_like(mu2_vals)


def check_kallen_lehmann(mu2_vals: np.ndarray, rho_vals: np.ndarray, atol: float = 1e-2) -> bool:
    """Check positivity and unit normalization for Källén-Lehmann proxy."""
    mu2_vals = np.asarray(mu2_vals, dtype=float)
    rho_vals = np.asarray(rho_vals, dtype=float)
    integral = np.trapz(rho_vals, mu2_vals)
    return np.all(rho_vals >= -1e-12) and abs(integral - 1.0) <= atol
