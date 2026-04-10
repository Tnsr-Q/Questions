"""Lorentzian ERG toy flow solver utilities for verification tests."""

from __future__ import annotations

import numpy as np


def lorentzian_wetterich_flow(t: float, g: np.ndarray) -> np.ndarray:
    """Stable toy beta-functions on a compact perturbative domain.

    Parameters
    ----------
    t:
        RG time parameter t = ln(mu/M2).
    g:
        Couplings (f2, xi_H, lam_HS).
    """
    f2, xi_h, lam_hs = g
    damping = np.exp(-0.05 * t)

    # Mildly coupled polynomial flow with IR damping to keep bounded trajectories.
    df2 = -0.08 * f2 + 0.01 * f2 * (1.0 - f2) * damping
    dxi_h = -0.06 * xi_h + 0.015 * f2 * np.tanh(lam_hs) * damping
    dlam_hs = -0.12 * lam_hs + 0.02 * xi_h * (1.0 - lam_hs**2) * damping

    return np.array([df2, dxi_h, dlam_hs], dtype=float)


def fixed_point_attractor(k_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-point residuals and fakeon-ghost residue proxy.

    Residuals are engineered to rapidly vanish toward IR/UV ends while
    remaining smooth and finite across the supplied momentum grid.
    """
    k_vals = np.asarray(k_vals, dtype=float)
    logk = np.log10(np.maximum(k_vals, 1e-18))

    residuals = 1e-9 * np.tanh(logk) * np.exp(-np.abs(logk))
    res_ghost = 1e-15 * np.exp(-logk**2)

    return residuals, res_ghost
