import glob
import logging
import os
from typing import Callable

import jax
from jax import profiler, shard_map
from jax.sharding import Mesh, PartitionSpec

jax.config.update("jax_enable_x64", True)

log = logging.getLogger("QUFT_MemSnapshot")


class MemorySnapshotShardTracker:
    """Wrap jax.shard_map with memory snapshot capture and rotation."""

    def __init__(
        self,
        log_dir: str = "./tb_regge_mem",
        run_id: str = "default",
        snapshot_interval: int = 25,
        max_snapshots: int = 5,
    ):
        self.log_dir = os.path.join(log_dir, run_id, "profiles")
        os.makedirs(self.log_dir, exist_ok=True)
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = max_snapshots
        self.step = 0

    def _cleanup_old_snapshots(self) -> None:
        """Auto-rotate old snapshots to avoid disk bloat."""
        files = sorted(glob.glob(os.path.join(self.log_dir, "mem_step_*.pb.gz")))
        if len(files) <= self.max_snapshots:
            return

        for path in files[: -self.max_snapshots]:
            os.remove(path)
            log.debug("Removed old snapshot: %s", path)

    def _capture_memory(self, step: int) -> None:
        """Capture a TensorBoard-compatible JAX device memory profile."""
        path = os.path.join(self.log_dir, f"mem_step_{step}.pb.gz")
        try:
            profiler.save_device_memory_profile(path)
            log.info("Memory snapshot saved at step %s: %s", step, path)
        except Exception as exc:  # pragma: no cover - best-effort profiling
            log.warning("Memory snapshot failed at step %s: %s", step, exc)

    def profiled_shard_map(
        self,
        func: Callable,
        mesh: Mesh,
        in_specs: PartitionSpec,
        out_specs: PartitionSpec,
        **kwargs,
    ) -> Callable:
        """Create a shard_map wrapper with periodic memory snapshots."""

        @profiler.StepTraceAnnotation("regge_shard_execution")
        def wrapped(*args, **kw):
            if self.step > 0 and self.step % self.snapshot_interval == 0:
                self._capture_memory(self.step)
                self._cleanup_old_snapshots()

            with profiler.TraceAnnotation("local_newton_solve"):
                return func(*args, **kw)

        sharded_fn = shard_map(wrapped, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **kwargs)
        self.step += 1
        return sharded_fn

    def get_tensorboard_command(self) -> str:
        return f"tensorboard --logdir={os.path.dirname(self.log_dir)} --bind_all"
