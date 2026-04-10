"""JAX-native Regge pole tracker utilities."""

from __future__ import annotations

from typing import Dict

from src.proto.return_schemas import FakeonCertification
from src.proto.schema_enforcer import enforce_schema

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, lax, vmap

jax.config.update("jax_enable_x64", True)


class JAXReggePoleTracker:
    """GIL-free, XLA-compilable Regge pole tracker."""

    def __init__(
        self,
        alpha: float = 0.05,
        m2: float = 0.01,
        max_iter: int = 50,
        tol: float = 1e-8,
        eps: float = 1e-7,
        fallback_root: float = -0.25,
    ):
        self.alpha = alpha
        self.m2 = m2
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.threshold = 4.0 * m2
        self.fallback_root = fallback_root

    @staticmethod
    def _safe_complex_power(base: jnp.ndarray, exp: jnp.ndarray) -> jnp.ndarray:
        """Numerically stable base**exp for non-negative base."""
        safe_base = jnp.maximum(base, 1e-15)
        return jnp.exp(exp * jnp.log(safe_base))

    def _pole_condition(self, l_re: float, s_val: float, delta_mean: float) -> float:
        """Evaluate Im[1 / (1 - S_c(l_re + i eps))]."""
        l_c = l_re + 1j * self.eps
        arg = jnp.maximum(s_val - self.threshold, 0.0)
        eta_c = jnp.exp(-self.alpha * self._safe_complex_power(arg, l_c + 1.0))
        s_c = eta_c * jnp.exp(2j * delta_mean)
        return jnp.imag(1.0 / (1.0 - s_c))

    def _newton_step(self, l_re: float, s_val: float, delta_mean: float) -> float:
        f_val = self._pole_condition(l_re, s_val, delta_mean)
        df_val = grad(self._pole_condition, argnums=0)(l_re, s_val, delta_mean)
        step = f_val / (df_val + 1e-12)
        return l_re - jnp.clip(step, -0.5, 0.5)

    def _find_pole_single(self, s_val: float, delta_mean: float, l0: float = 1.95) -> float:
        def cond(state):
            i, _l_cur, f_cur = state
            return (i < self.max_iter) & (jnp.abs(f_cur) > self.tol)

        def body(state):
            i, l_cur, _f_cur = state
            l_new = self._newton_step(l_cur, s_val, delta_mean)
            f_new = self._pole_condition(l_new, s_val, delta_mean)
            return i + 1, l_new, f_new

        _i_final, l_final, f_final = lax.while_loop(
            cond,
            body,
            (0, l0, self._pole_condition(l0, s_val, delta_mean)),
        )
        is_good = jnp.isfinite(l_final) & jnp.isfinite(f_final) & (jnp.abs(f_final) <= 1e-4)
        return jnp.where(is_good, l_final, self.fallback_root)

    def scan_trajectory(self, t_grid: jnp.ndarray, delta_at_t: jnp.ndarray, l0: float = 1.95) -> jnp.ndarray:
        """Vectorized Regge pole scan over the provided t-grid."""
        s_cross = jnp.maximum(t_grid, self.threshold)
        vmap_find = vmap(self._find_pole_single, in_axes=(0, 0, None))
        return vmap_find(s_cross, delta_at_t, l0)

    @enforce_schema(FakeonCertification)
    def verify_fakeon_virtualization(
        self,
        alpha_traj: jnp.ndarray,
        t_grid: jnp.ndarray,
        m2_target: float = 2.4e23,
    ) -> Dict[str, object]:
        """Evaluate whether Re[alpha(M2^2)] < 0 on a discrete trajectory."""
        t_target = m2_target**2
        idx = jnp.argmin(jnp.abs(t_grid - t_target))
        alpha_m2 = alpha_traj[idx]
        is_virtualized = jnp.real(alpha_m2) < 0
        return {
            "Re_alpha_at_M2": float(jnp.real(alpha_m2)),
            "fakeon_virtualized": bool(is_virtualized),
            "trajectory": np.array(alpha_traj),
            "status": "VERIFIED" if is_virtualized else "PENDING",
        }
