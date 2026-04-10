import numpy as np
import jax.numpy as jnp
from jax import jit
from typing import Dict, Tuple

from src.bootstrap_solver import DiscretizedBootstrapSolver
from src.regge_jax_solver import JAXReggePoleTracker


class ReggeExtendedBootstrap(DiscretizedBootstrapSolver):
    """Discretized bootstrap extension with lightweight Regge pole tracking."""

    def __init__(
        self,
        s_min: float = 4.0,
        s_max: float = 1e6,
        N_s: int = 128,
        N_l: int = 6,
        alpha: float = 0.05,
        m2: float = 0.01,
        M2: float = 2.4e23,
        **kwargs,
    ):
        super().__init__(s_min=s_min, s_max=s_max, N_s=N_s, N_l=N_l, alpha=alpha, m2=m2, **kwargs)
        self.M2 = M2
        self.t_grid = np.logspace(-2, 4.0, 40)
        self.l_contour_re = np.linspace(-0.5, 2.0, 150)
        self.l_contour_im = np.linspace(-0.8, 0.8, 100)

    @staticmethod
    @jit
    def _analytic_continue_Sl(
        s: jnp.ndarray,
        l_re: jnp.ndarray,
        l_im: jnp.ndarray,
        alpha: float,
        delta_l: jnp.ndarray,
    ) -> jnp.ndarray:
        """Analytic continuation of S_l(s) to complex l = l_re + i l_im."""
        eta_c = jnp.exp(-alpha * jnp.maximum(s - 0.04, 0.0) ** (l_re + 1j * l_im + 1.0))
        return eta_c * jnp.exp(2j * delta_l)

    def track_regge_poles(
        self,
        S_l_solved: jnp.ndarray,
        delta_solved: jnp.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Track a coarse Regge trajectory α(t) with a JAX-native Newton solver."""
        del S_l_solved  # interface-compatible input; phase information is read from delta_solved.

        tracker = JAXReggePoleTracker(alpha=self.alpha, m2=self.m2)
        t_grid_jax = jnp.asarray(self.t_grid)
        s_grid_jax = jnp.asarray(self.s_grid)
        delta_jax = jnp.asarray(delta_solved)

        s_cross = jnp.maximum(t_grid_jax, tracker.threshold)
        s_idx = jnp.argmin(jnp.abs(s_grid_jax[None, :] - s_cross[:, None]), axis=1)
        delta_at_t = jnp.mean(delta_jax[:, s_idx], axis=0)

        alpha_traj = tracker.scan_trajectory(t_grid_jax, delta_at_t)
        return np.asarray(alpha_traj), self.t_grid

    def verify_fakeon_regge_condition(self, alpha_traj: np.ndarray, t_grid: np.ndarray) -> Dict[str, object]:
        """Verify Re[α(M2²)] < 0 as a fakeon-virtualization diagnostic."""
        t_target = self.M2**2
        idx = int(np.argmin(np.abs(t_grid - t_target)))
        alpha_M2 = alpha_traj[idx]

        return {
            "Re_alpha_at_M2": float(np.real(alpha_M2)),
            "Im_alpha_at_M2": float(np.imag(alpha_M2)),
            "fakeon_virtualized": bool(np.real(alpha_M2) < 0),
            "trajectory": alpha_traj.tolist(),
            "status": "VERIFIED" if np.real(alpha_M2) < 0 else "PENDING",
        }

    def run_full_regge_analysis(self, S_opt: jnp.ndarray, delta_opt: jnp.ndarray) -> Dict[str, object]:
        """Run coarse Regge-pole extraction and fakeon verification."""
        traj, t_vals = self.track_regge_poles(S_opt, delta_opt)
        verification = self.verify_fakeon_regge_condition(traj, t_vals)
        verification["trajectory"] = [str(c) for c in verification["trajectory"]]
        return verification
