import logging
import time
import tracemalloc

import jax
import jax.numpy as jnp
import jax.profiler
import numpy as np
from jax import lax, shard_map
from jax.sharding import Mesh, PartitionSpec

jax.config.update("jax_enable_x64", True)

log = logging.getLogger("QUFT_ReggeProfiler")


class ProfiledShardedReggeSolver:
    """Sharded Regge solver with JAX and host-memory profiling metadata."""

    def __init__(
        self,
        N_t: int = 128,
        t_min: float = 1e-2,
        t_max: float = 1e4,
        alpha_inel: float = 0.05,
        M2: float = 2.4e23,
        max_iter: int = 40,
        tol: float = 1e-8,
        profile_dir: str = "./jax_regge_profiles",
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
        self.profile_dir = profile_dir

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
    ) -> float:
        def cond(carry):
            i, _, f = carry
            return (i < max_iter) & (jnp.abs(f) > tol)

        def body(carry):
            i, l, _ = carry
            f_val = ProfiledShardedReggeSolver._pole_condition_real(l, s_val, delta_mean)
            _, df_dl = jax.jvp(
                lambda x: ProfiledShardedReggeSolver._pole_condition_real(x, s_val, delta_mean),
                (l,),
                (1.0,),
            )
            l_new = l - f_val / (df_dl + 1e-12)
            return i + 1, l_new, f_val

        f0 = ProfiledShardedReggeSolver._pole_condition_real(l_init, s_val, delta_mean)
        _, l_final, _ = lax.while_loop(cond, body, (0, l_init, f0))
        return l_final

    def _process_chunk(
        self,
        s_chunk: jnp.ndarray,
        l0_chunk: jnp.ndarray,
        delta_chunk: jnp.ndarray,
    ) -> jnp.ndarray:
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
            out_specs=self.shard_spec,
        )
        try:
            sharded_fn = shard_map(self._process_chunk, check_vma=False, **common_kwargs)
        except TypeError:
            try:
                sharded_fn = shard_map(self._process_chunk, check_rep=False, **common_kwargs)
            except TypeError:
                sharded_fn = shard_map(self._process_chunk, **common_kwargs)
        return sharded_fn(self.s_cross, l0, delta_at_t)

    def scan_with_profiler(self, delta_at_t: jnp.ndarray, run_id: str = "default") -> tuple[jnp.ndarray, dict]:
        """Execute the sharded scan while recording JAX and host memory metadata."""
        tracemalloc.start()
        jax.profiler.start_trace(self.profile_dir)

        t0 = time.perf_counter()
        with jax.profiler.StepTraceAnnotation("regge_shard_scan"):
            traj = self.scan_regge_trajectory_sharded(delta_at_t)
        jax.profiler.stop_trace()
        t1 = time.perf_counter()

        jax_mem = sum(arr.nbytes for arr in jax.live_arrays()) / (1024**2)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        meta = {
            "run_id": run_id,
            "wall_time_sec": t1 - t0,
            "peak_host_mem_mb": peak / (1024**2),
            "current_host_mem_mb": current / (1024**2),
            "jax_managed_mem_mb": jax_mem,
            "devices": self.num_devices,
            "trace_dir": self.profile_dir,
            "status": "PROFILING_COMPLETE",
        }
        log.info(
            "[%s] Scan completed in %.3fs | Peak Host: %.1fMB | JAX: %.1fMB",
            run_id,
            meta["wall_time_sec"],
            meta["peak_host_mem_mb"],
            meta["jax_managed_mem_mb"],
        )
        return traj, meta

    def verify_fakeon_virtualization(self, alpha_traj: jnp.ndarray) -> dict:
        t_target = self.M2**2
        idx = jnp.argmin(jnp.abs(self.t_grid - t_target))
        alpha_M2 = alpha_traj[idx]
        alpha_real = float(jnp.real(alpha_M2))
        return {
            "Re_alpha_at_M2": alpha_real,
            "fakeon_virtualized": alpha_real < 0,
            "status": "VERIFIED" if alpha_real < 0 else "PENDING",
        }
