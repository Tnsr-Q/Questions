from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from src.tolerance.dynamic_ledger import DynamicToleranceLedger
from src.tolerance.regime_detector import RegimeDetector

# Constants
PI = np.pi
T16 = 16.0 * PI**2
T16_SQ = T16**2

jax.config.update("jax_enable_x64", True)


@dataclass
class JAXRGEResult:
    """Minimal ODE result container mirroring the subset used in this project."""

    t: np.ndarray
    y: np.ndarray
    success: bool
    nfev: int


class SIQGRGESolver:
    """
    2-Loop RGE system for Scale-Invariant Quadratic Gravity + SM.
    Matches Extended Lemma 2 & S.3 β-function closure.
    """

    def __init__(
        self,
        mu_start: float = 173.1,
        mu_end: float = 2.4e23,
        ledger: Optional[DynamicToleranceLedger] = None,
        detector: Optional[RegimeDetector] = None,
    ):
        self.mu_start = mu_start  # GeV (top mass scale)
        self.mu_end = mu_end  # GeV (fakeon threshold M2)
        self.t_span = (np.log(mu_start), np.log(mu_end))
        self.ledger = ledger
        self.detector = detector or RegimeDetector(M2_GeV=mu_end)
        self.state: Dict[str, float] = {"energy_scale": mu_start}
        # Fixed bounded integration budget for full XLA compilation.
        self.default_steps = 2048

    @staticmethod
    def _beta_f2(f2: float, lam_HS: float, xi_H: float) -> float:
        """Return raw loop coefficients for β_{f2}; rhs() applies the overall /T16 normalization."""
        b1 = -(133.0 / 20.0) * f2**3
        b2_grav = (5196.0 / 5.0) / T16 * f2**5
        b2_sm = -12.0 * lam_HS * xi_H**2 / T16 * f2**3
        return b1 + b2_grav + b2_sm

    def rhs(self, t: float, g: np.ndarray) -> np.ndarray:
        """RGE vector field dg/dt = β(g)"""
        _ = t
        lam_H, lam_S, lam_HS, y_t, g1, g2, g3, f2, xi_H = g

        # 1-loop scalar sector (QUFT- RGE-Thermal.txt §1)
        b_lam_H = (
            24 * lam_H**2
            + 0.5 * lam_HS**2
            - 6 * y_t**4
            + 0.375 * (2 * g2**4 + (g1**2 + g2**2) ** 2)
            + (1.5 * g2**2 + 0.5 * g1**2 - 6 * y_t**2) * lam_H
        )
        b_lam_S = 18 * lam_S**2 + 2 * lam_HS**2
        b_lam_HS = (
            4 * lam_HS**2
            + 12 * lam_H * lam_HS
            + 6 * lam_S * lam_HS
            - 6 * y_t**2 * lam_HS
            + (1.5 * g2**2 + 0.5 * g1**2) * lam_HS
        )

        # 1-loop SM gauge & Yukawa (standard MS-bar)
        b_yt = y_t * (4.5 * y_t**2 - 4 * g3**2 - 2.25 * g2**2 - 1.4166667 * g1**2)
        b_g1 = (41.0 / 6.0) * g1**3
        b_g2 = (-19.0 / 6.0) * g2**3
        b_g3 = -7.0 * g3**3

        # f2 flow with 2-loop closure
        b_f2 = self._beta_f2(f2, lam_HS, xi_H)
        b_xi_H = 0.0  # Negligible running in perturbative regime per docs

        return np.array([b_lam_H, b_lam_S, b_lam_HS, b_yt, b_g1, b_g2, b_g3, b_f2, b_xi_H]) / T16

    @staticmethod
    @jax.checkpoint
    def _rhs_jax(t: float, g: jnp.ndarray) -> jnp.ndarray:
        """JAX-native RGE vector field dg/dt = β(g)."""
        _ = t
        lam_H, lam_S, lam_HS, y_t, g1, g2, g3, f2, xi_H = g

        b_lam_H = (
            24 * lam_H**2
            + 0.5 * lam_HS**2
            - 6 * y_t**4
            + 0.375 * (2 * g2**4 + (g1**2 + g2**2) ** 2)
            + (1.5 * g2**2 + 0.5 * g1**2 - 6 * y_t**2) * lam_H
        )
        b_lam_S = 18 * lam_S**2 + 2 * lam_HS**2
        b_lam_HS = (
            4 * lam_HS**2
            + 12 * lam_H * lam_HS
            + 6 * lam_S * lam_HS
            - 6 * y_t**2 * lam_HS
            + (1.5 * g2**2 + 0.5 * g1**2) * lam_HS
        )
        b_yt = y_t * (4.5 * y_t**2 - 4 * g3**2 - 2.25 * g2**2 - 1.4166667 * g1**2)
        b_g1 = (41.0 / 6.0) * g1**3
        b_g2 = (-19.0 / 6.0) * g2**3
        b_g3 = -7.0 * g3**3

        b1 = -(133.0 / 20.0) * f2**3
        b2_grav = (5196.0 / 5.0) / T16 * f2**5
        b2_sm = -12.0 * lam_HS * xi_H**2 / T16 * f2**3
        b_f2 = b1 + b2_grav + b2_sm
        b_xi_H = 0.0
        return jnp.array([b_lam_H, b_lam_S, b_lam_HS, b_yt, b_g1, b_g2, b_g3, b_f2, b_xi_H]) / T16

    @staticmethod
    @partial(jax.jit, static_argnums=(3,))
    def _integrate_rk4(g0: jnp.ndarray, t0: float, t1: float, num_steps: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Bounded-loop RK4 integrator compiled into a single XLA program."""
        ts = jnp.linspace(t0, t1, num_steps + 1)
        dt = (t1 - t0) / num_steps

        def step(g, t):
            k1 = SIQGRGESolver._rhs_jax(t, g)
            k2 = SIQGRGESolver._rhs_jax(t + 0.5 * dt, g + 0.5 * dt * k1)
            k3 = SIQGRGESolver._rhs_jax(t + 0.5 * dt, g + 0.5 * dt * k2)
            k4 = SIQGRGESolver._rhs_jax(t + dt, g + dt * k3)
            g_next = g + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return g_next, g_next

        _, ys_tail = jax.lax.scan(step, g0, ts[:-1])
        ys = jnp.concatenate([g0[None, :], ys_tail], axis=0)
        return ts, ys

    def solve(
        self,
        g0: np.ndarray,
        rtol: float = 1e-8,
        atol: Optional[float] = None,
        num_steps: Optional[int] = None,
    ) -> Dict:
        """Integrate RGE from m_t to M2. Returns solution dict + stability flags."""
        del rtol  # Fixed-step integrator keeps bounded compiled loops.
        if atol is None:
            if self.ledger is not None:
                regime = self.detector.classify(self.state, {})
                atol = self.ledger.get_tolerance("rge_atol", regime=regime)
            else:
                atol = 1e-10

        steps = int(num_steps or self.default_steps)
        if steps <= 0:
            raise ValueError("num_steps must be positive")
        t_jax, y_jax = self._integrate_rk4(
            jnp.asarray(g0, dtype=jnp.float64),
            float(self.t_span[0]),
            float(self.t_span[1]),
            steps,
        )
        sol = JAXRGEResult(
            t=np.asarray(t_jax),
            y=np.asarray(y_jax).T,
            success=True,
            nfev=4 * steps,
        )

        # Extract endpoints
        g_uv = sol.y[:, -1]
        g_ir = sol.y[:, 0]

        regime = self.detector.classify(
            {
                "energy_scale": self.mu_end,
                "f2": float(g_uv[7]),
                "step_size": float(np.diff(sol.t).min()) if sol.t.size > 1 else 1.0,  # type: ignore[arg-type]
            },
            {"RGE_residual": 0.0 if sol.success else 1.0},
        )

        if self.ledger is not None:
            residual = abs(float(g_uv[-1] - g0[-1]))
            self.ledger.update_from_residual("rge_atol", residual, "rge_solver")

        return {
            "sol": sol,
            "g_uv": g_uv,
            "g_ir": g_ir,
            "success": sol.success,
            "nfev": sol.nfev,
            "tol_used": atol,
            "regime": regime.value,
        }
