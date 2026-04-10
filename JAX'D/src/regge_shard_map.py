import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, shard_map
from jax.sharding import Mesh, PartitionSpec

from src.proto.return_schemas import FakeonCertification
from src.proto.schema_enforcer import enforce_schema

jax.config.update("jax_enable_x64", True)


class ShardedReggeSolver:
    """Multi-device Regge pole tracking accelerated with JAX ``shard_map``."""

    def __init__(
        self,
        N_t: int = 128,
        t_min: float = 1e-2,
        t_max: float = 1e4,
        alpha_inel: float = 0.05,
        M2: float = 2.4e23,
        max_iter: int = 40,
        tol: float = 1e-8,
    ):
        devices = jax.devices()
        self.num_devices = len(devices)
        if N_t % self.num_devices != 0:
            raise ValueError("N_t must be divisible by device count for clean sharding")

        self.N_t = N_t
        self.t_grid = jnp.logspace(jnp.log10(t_min), jnp.log10(t_max), N_t)
        self.alpha = alpha_inel
        self.M2 = M2
        self.max_iter = max_iter
        self.tol = tol
        self.s_cross = jnp.maximum(self.t_grid, 4.0 * 0.01)
        self.last_convergence_mask = jnp.zeros_like(self.t_grid, dtype=bool)

        self.mesh = Mesh(np.array(devices), ("dev",))
        self.shard_spec = PartitionSpec("dev")

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
    ) -> tuple[float, bool]:
        def cond(carry):
            i, _, f = carry
            return (i < max_iter) & (jnp.abs(f) > tol)

        def body(carry):
            i, l, _ = carry
            f_val = ShardedReggeSolver._pole_condition_real(l, s_val, delta_mean)
            _, df_dl = jax.jvp(
                lambda x: ShardedReggeSolver._pole_condition_real(x, s_val, delta_mean),
                (l,),
                (1.0,),
            )
            l_new = l - f_val / (df_dl + 1e-12)
            return i + 1, l_new, f_val

        f0 = ShardedReggeSolver._pole_condition_real(l_init, s_val, delta_mean)
        iters, l_final, f_final = lax.while_loop(cond, body, (0, l_init, f0))
        did_converge = (jnp.abs(f_final) <= tol) & (iters < max_iter)
        return l_final, did_converge

    def _process_chunk(
        self,
        s_chunk: jnp.ndarray,
        l0_chunk: jnp.ndarray,
        delta_chunk: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jax.vmap(self._newton_step, in_axes=(0, 0, 0, None, None))(
            l0_chunk,
            s_chunk,
            delta_chunk,
            self.max_iter,
            self.tol,
        )

    def scan_regge_trajectory_sharded(self, delta_at_t: jnp.ndarray) -> jnp.ndarray:
        l0 = jnp.ones_like(self.t_grid) * 1.95
        common_kwargs = dict(
            mesh=self.mesh,
            in_specs=(self.shard_spec, self.shard_spec, self.shard_spec),
            out_specs=(self.shard_spec, self.shard_spec),
        )
        try:
            sharded_fn = shard_map(self._process_chunk, check_vma=False, **common_kwargs)
        except TypeError:
            try:
                sharded_fn = shard_map(self._process_chunk, check_rep=False, **common_kwargs)
            except TypeError:
                sharded_fn = shard_map(self._process_chunk, **common_kwargs)
        roots, did_converge = sharded_fn(self.s_cross, l0, delta_at_t)
        self.last_convergence_mask = did_converge
        return roots

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
