import json
import logging
import os
import time
from typing import Any

import jax
from jax import profiler, shard_map

jax.config.update("jax_enable_x64", True)

log = logging.getLogger("QUFT_ShardLayout")


class TensorBoardShardLayoutTracker:
    """Track ``shard_map`` layouts and emit TensorBoard-friendly trace metadata."""

    def __init__(self, log_dir: str = "./tb_shard_layout", run_id: str = "default"):
        self.log_dir = os.path.join(log_dir, run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self.metadata_path = os.path.join(self.log_dir, "shard_layouts.json")
        self.step = 0
        self._layouts: list[dict[str, Any]] = []

    def _as_list(self, specs: Any) -> list[Any]:
        if isinstance(specs, (list, tuple)):
            return list(specs)
        return [specs]

    def _log_layout(self, func_name: str, mesh: Any, in_specs: Any, out_specs: Any) -> None:
        layout = {
            "step": self.step,
            "timestamp": time.time(),
            "func": func_name,
            "mesh_shape": dict(mesh.shape),
            "mesh_axes": tuple(mesh.axis_names),
            "in_specs": [str(s) for s in self._as_list(in_specs)],
            "out_specs": [str(s) for s in self._as_list(out_specs)],
        }
        self._layouts.append(layout)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._layouts, f, indent=2)

    @profiler.StepTraceAnnotation("regge_shard_execution")
    def profiled_shard_map(self, func, mesh, in_specs, out_specs, **kwargs):
        func_name = getattr(func, "__name__", f"sharded_fn_{self.step}")
        self._log_layout(func_name, mesh, in_specs, out_specs)

        def wrapped_func(*args, **kw):
            with profiler.TraceAnnotation(f"device_compute/{func_name}"):
                with profiler.TraceAnnotation("local_newton_solve"):
                    return func(*args, **kw)

        sharded_fn = shard_map(wrapped_func, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **kwargs)
        self.step += 1
        return sharded_fn

    def get_trace_command(self) -> str:
        return f"tensorboard --logdir={self.log_dir} --bind_all"
