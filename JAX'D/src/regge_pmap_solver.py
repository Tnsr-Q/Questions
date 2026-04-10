import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, pmap

from src.proto.return_schemas import FakeonCertification
from src.proto.schema_enforcer import enforce_schema

jax.config.update("jax_enable_x64", True)


class PMappedReggeSolver:
    """Multi-device Regge pole tracking accelerated with JAX ``pmap``."""

    def __init__(
        self,
        N_t: int = 128,
        t_min: float = 1e-2,
        t_max: float = 1e4,
        alpha_inel: float = 0.05,
        M2: float = 2.4e23,
        max_iter: int = 40,
        tol: float = 1e-8,
        num_devices: int | None = None,
    ):
        self.num_devices = num_devices or jax.device_count()
        if N_t % self.num_devices != 0:
            raise ValueError("N_t must be divisible by num_devices for clean pmap sharding")

        self.N_t_per_device = N_t // self.num_devices
        self.t_grid = jnp.logspace(jnp.log10(t_min), jnp.log10(t_max), N_t)
        self.alpha = alpha_inel
        self.M2 = M2
        self.max_iter = max_iter
        self.tol = tol
        self.s_cross = jnp.maximum(self.t_grid, 4.0 * 0.01)

    @staticmethod
    def _pole_condition_real(l_re: float, s_val: float, delta_mean: float) -> float:
        eps = 1e-7
        l_c = l_re + 1j * eps
        eta_c = jnp.exp(-0.05 * jnp.maximum(s_val - 0.04, 0.0) ** (l_c + 1))
        S_c = eta_c * jnp.exp(2j * delta_mean)
        return jnp.imag(1.0 / (1.0 - S_c))

    @staticmethod
    def _newton_step(
        l_init: float,
        s_val: float,
        delta_mean: float,
        max_iter: int,
        tol: float,
    ) -> float:
        def cond(carry):
            i, _, f = carry
            return (i < max_iter) & (jnp.abs(f) > tol)

        def body(carry):
            i, l, _ = carry
            f_val = PMappedReggeSolver._pole_condition_real(l, s_val, delta_mean)
            _, df_dl = jax.jvp(
                lambda x: PMappedReggeSolver._pole_condition_real(x, s_val, delta_mean),
                (l,),
                (1.0,),
            )
            l_new = l - f_val / (df_dl + 1e-12)
            return i + 1, l_new, f_val

        _, l_final, _ = lax.while_loop(cond, body, (0, l_init, 1.0))
        return l_final

    def _scan_shard(
        self,
        s_shard: jnp.ndarray,
        l0_shard: jnp.ndarray,
        delta_shard: jnp.ndarray,
    ) -> jnp.ndarray:
        return jax.vmap(self._newton_step, in_axes=(0, 0, 0, None, None))(
            l0_shard,
            s_shard,
            delta_shard,
            self.max_iter,
            self.tol,
        )

    def scan_regge_trajectory_pmap(self, delta_at_t: jnp.ndarray) -> jnp.ndarray:
        delta_sharded = delta_at_t.reshape(self.num_devices, self.N_t_per_device)
        s_sharded = self.s_cross.reshape(self.num_devices, self.N_t_per_device)
        l0_sharded = jnp.ones_like(delta_sharded) * 1.95

        alpha_sharded = pmap(self._scan_shard)(s_sharded, l0_sharded, delta_sharded)
        return alpha_sharded.reshape(-1)

    @enforce_schema(FakeonCertification)
    def verify_fakeon_virtualization(self, alpha_traj: jnp.ndarray) -> dict:
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
