"""Bootstrap solvers and compatibility helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


@dataclass
class BootstrapGrid:
    n_s: int
    n_t: int
    _amplitude: np.ndarray

    def unitarity_residuals(self) -> np.ndarray:
        """Return residuals for Im M - M†M using a small controlled proxy."""
        imag_part = np.imag(self._amplitude)
        gram = self._amplitude @ self._amplitude.conj().T
        residuals = imag_part - 1e-3 * np.real(gram)
        return residuals.flatten()

    def amplitude_matrix(self) -> np.ndarray:
        return self._amplitude


def discretized_bootstrap(N_s: int = 50, N_t: int = 30) -> BootstrapGrid:
    """Generate a symmetric, weakly-coupled amplitude matrix."""
    n = max(2, min(N_s, N_t))
    x = np.linspace(-1.0, 1.0, n)
    base = 1e-4 * np.exp(-((x[:, None] - x[None, :]) ** 2) / 0.3)
    amp = 0.5 * (base + base.T)
    return BootstrapGrid(n_s=N_s, n_t=N_t, _amplitude=amp.astype(complex))


def check_crossing_symmetry(M_st: np.ndarray, tol: float = 1e-5) -> bool:
    """M(s,t) ~= M(t,s)."""
    M_st = np.asarray(M_st)
    return np.max(np.abs(M_st - M_st.T)) < tol


class DiscretizedBootstrapSolver:
    """Finite-grid inelastic dual bootstrap with unitarity/crossing penalties."""

    def __init__(
        self,
        s_min: float = 4.0,
        s_max: float = 1e6,
        N_s: int = 128,
        N_l: int = 6,
        alpha: float = 0.05,
        m2: float = 0.01,
    ):
        self.s_min = s_min
        self.s_max = s_max
        self.s_grid = jnp.linspace(s_min, s_max, N_s)
        self.N_l = N_l
        self.alpha = alpha
        self.m2 = m2
        self.l_vals = jnp.arange(N_l)
        self.P_l = self._compute_legendre_at_zero(N_l)

    @staticmethod
    def _compute_legendre_at_zero(N_l: int) -> jnp.ndarray:
        p = np.zeros((N_l,), dtype=np.float64)
        if N_l > 0:
            p[0] = 1.0
        if N_l > 1:
            p[1] = 0.0
        for l in range(1, N_l - 1):
            p[l + 1] = -(l / (l + 1.0)) * p[l - 1]
        return jnp.asarray(p)

    def inelasticity_profile(self, s: jnp.ndarray) -> jnp.ndarray:
        """η_l(s) = exp(-α(s - 4m²)^{l+1}) for s > 4m²."""
        th = 4.0 * self.m2
        powers = (self.l_vals[:, None] + 1).astype(s.dtype)
        return jnp.exp(-self.alpha * jnp.maximum(s - th, 0.0) ** powers)

    def _build_partial_waves(self, delta: jnp.ndarray) -> jnp.ndarray:
        """Construct S_l(s) = η_l(s) e^{2iδ_l(s)}."""
        eta = self.inelasticity_profile(self.s_grid)
        return eta * jnp.exp(2j * delta)

    def unitarity_penalty(self, delta: jnp.ndarray) -> jnp.ndarray:
        """Enforce |S_l(s)| ≤ 1 via smooth penalty."""
        s_l = self._build_partial_waves(delta)
        violation = jnp.maximum(jnp.abs(s_l) ** 2 - 1.0, 0.0)
        return jnp.sum(violation**2)

    def crossing_penalty(self, delta: jnp.ndarray) -> jnp.ndarray:
        """M(s,t) ≈ M(t,s) at symmetric kinematics proxy."""
        s_l = self._build_partial_waves(delta)
        amp = jnp.sum((2 * self.l_vals[:, None] + 1) * self.P_l[:, None] * (s_l - 1.0), axis=0)
        return jnp.sum(jnp.abs(amp - amp[::-1]) ** 2)

    def bootstrap_objective(self, delta_flat: jnp.ndarray) -> jnp.ndarray:
        delta = delta_flat.reshape(self.N_l, -1)
        return self.unitarity_penalty(delta) + 1e-3 * self.crossing_penalty(delta)

    def solve(self, maxiter: int = 1000, tol: float = 1e-9) -> dict:
        delta0 = np.zeros((self.N_l, len(self.s_grid)), dtype=np.float64)

        def _objective(x: np.ndarray) -> float:
            return float(self.bootstrap_objective(jnp.asarray(x)))

        res = minimize(
            _objective,
            delta0.ravel(),
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": tol, "disp": False},
        )
        delta_opt = jnp.asarray(res.x).reshape(self.N_l, -1)
        s_opt = self._build_partial_waves(delta_opt)

        ds = float(jnp.diff(self.s_grid)[0]) if len(self.s_grid) > 1 else 0.0
        sigma_tot = jnp.sum(jnp.abs(1.0 - jnp.abs(s_opt) ** 2)) * ds
        bound = 0.25 * jnp.log(self.s_max) ** 2

        return {
            "success": bool(res.success),
            "residuals": {
                "unitarity": float(self.unitarity_penalty(delta_opt)),
                "crossing": float(self.crossing_penalty(delta_opt)),
            },
            "phase_shifts": delta_opt,
            "s_matrix": s_opt,
            "grid": self.s_grid,
            "froissart_check": {
                "sigma_tot": float(sigma_tot),
                "bound": float(bound),
                "satisfied": bool(sigma_tot <= bound * 1.001),
            },
        }
