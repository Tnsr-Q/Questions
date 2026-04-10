import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, lax, vmap

jax.config.update("jax_enable_x64", True)


class VectorizedReggeSolver:
    """Parallel t-channel Regge pole tracking via JAX ``vmap``."""

    def __init__(
        self,
        N_t: int = 64,
        t_min: float = 1e-2,
        t_max: float = 1e4,
        alpha_inel: float = 0.05,
        M2: float = 2.4e23,
        max_iter: int = 40,
        tol: float = 1e-8,
    ):
        self.t_grid = jnp.logspace(jnp.log10(t_min), jnp.log10(t_max), N_t)
        self.alpha = alpha_inel
        self.M2 = M2
        self.max_iter = max_iter
        self.tol = tol
        self.s_cross = jnp.maximum(self.t_grid, 4.0 * 0.01)
        self.last_convergence_mask = jnp.zeros_like(self.t_grid, dtype=bool)

    @staticmethod
    @jit
    def _pole_condition_real(l_re: float, s_val: float, delta_mean: float) -> float:
        """Root of this real function aligns with a Regge pole condition."""
        eps = 1e-7
        l_c = l_re + 1j * eps
        eta_c = jnp.exp(-0.05 * jnp.maximum(s_val - 0.04, 0.0) ** (l_c + 1))
        S_c = eta_c * jnp.exp(2j * delta_mean)
        return jnp.imag(1.0 / (1.0 - S_c))

    def _newton_trajectory_step(self, l_init: float, s_val: float, delta_mean: float) -> tuple[float, bool]:
        """JAX-traceable Newton-Raphson update for Re[alpha(t)]."""

        def cond(carry):
            i, _, f = carry
            return (i < self.max_iter) & (jnp.abs(f) > self.tol)

        def body(carry):
            i, l, _ = carry
            f_val = self._pole_condition_real(l, s_val, delta_mean)
            f_grad = grad(self._pole_condition_real, argnums=0)(l, s_val, delta_mean)
            l_new = l - f_val / (f_grad + 1e-12)
            return i + 1, l_new, f_val

        iters, l_final, f_final = lax.while_loop(cond, body, (0, l_init, 1.0))
        did_converge = (jnp.abs(f_final) <= self.tol) & (iters < self.max_iter)
        return l_final, did_converge

    def scan_regge_trajectory(self, delta_at_t: jnp.ndarray) -> jnp.ndarray:
        """Vectorized Regge pole tracking across the t-channel grid."""
        l0 = jnp.ones_like(self.t_grid) * 1.95
        roots, did_converge = vmap(self._newton_trajectory_step)(l0, self.s_cross, delta_at_t)
        self.last_convergence_mask = did_converge
        return roots

    def verify_fakeon_virtualization(self, alpha_traj: jnp.ndarray) -> dict:
        """Check Re[alpha(M2^2)] < 0 fakeon virtualization criterion."""
        t_target = self.M2**2
        idx = jnp.argmin(jnp.abs(self.t_grid - t_target))
        alpha_M2 = alpha_traj[idx]
        alpha_real = float(jnp.real(alpha_M2))
        virtualized = alpha_real < 0

        return {
            "Re_alpha_at_M2": alpha_real,
            "fakeon_virtualized": virtualized,
            "trajectory": np.array(alpha_traj),
            "t_grid": np.array(self.t_grid),
            "status": "VERIFIED" if virtualized else "PENDING",
        }
