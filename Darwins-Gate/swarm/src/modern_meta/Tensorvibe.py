“””
TensorCell — Geometry-aware GW strain processing cell for Darwinian swarm.

Changes from previous version:

- Null channel replaced with instantaneous frequency (physically meaningful)
- Projection selection is deterministic (genome param or step counter, not wall-clock)
- Added step counter for reproducible forward passes
- Device coercion consolidated into helper
- solve_physics returns richer diagnostics for Darwinian fitness
- Added training-ready interface (get_geometric_params / from_mlx_output)
- Fitness accumulation uses exponential moving average, not raw sum
  “””

import uuid
import socket
import time
import numpy as np
import vibetensor as vbt
from utils.quaternion import (
generate_geometric_projections,
quaternion_to_matrix,
random_unit_quaternion,
)

class TensorCell:
def **init**(self, genome: dict):
self.genome = genome
self.fitness = 0.0
self._fitness_ema_alpha = genome.get(“fitness_ema_alpha”, 0.1)
self.uuid = str(uuid.uuid4())[:8]
self.pod_name = socket.gethostname()
self.step = 0  # Deterministic counter for reproducible ops

```
    # Initialize Geometric System
    self._setup_geometric_operations()

def _setup_geometric_operations(self):
    """Configure geometry-aware processing based on genome."""
    seed_val = int(self.uuid, 16) % (2**31)  # Keep seed in valid range

    # 1. Rotation Configuration
    if self.genome.get("apply_rotation", False):
        q = self.genome.get("rotation_quaternion")
        if q is None:
            q = random_unit_quaternion(seed=seed_val)
        self.rotation_matrix = quaternion_to_matrix(q)
    else:
        self.rotation_matrix = np.eye(3)

    # 2. Projection System
    if self.genome.get("use_geometric_projections", False):
        self.projections = generate_geometric_projections(
            n_proj=self.genome.get("num_projections", 3),
            rank=self.genome.get("projection_rank", 2),
            seed=seed_val,
        )
    else:
        self.projections = None

# ------------------------------------------------------------------ #
#  Core Physics Solver                                                 #
# ------------------------------------------------------------------ #

def solve_physics(self, inputs: dict) -> dict:
    """
    Run comb/parity kernel on (possibly enhanced) strain data.

    Parameters
    ----------
    inputs : dict
        Must contain 'strain_dlpack' and 'comb_mask_dlpack' as DLPack capsules.

    Returns
    -------
    dict with score, elapsed_ms, partial_fitness, step, diagnostics.
    """
    start = time.perf_counter()  # perf_counter > time.time for benchmarking

    # Decode tensors from DLPack
    strain = vbt.from_dlpack(inputs["strain_dlpack"])
    comb = vbt.from_dlpack(inputs["comb_mask_dlpack"])

    # Apply geometric transformations
    processed_strain, geo_info = self._apply_geometric_enhancements(strain)

    # Build TensorIter Config
    cfg = vbt.core.TensorIterConfig()
    cfg.check_mem_overlap(True)
    cfg.is_reduction(True)

    cfg.add_input(processed_strain)
    cfg.add_input(comb)
    cfg.add_output(vbt.empty_like(processed_strain))

    # Genome-driven tiling
    cfg.set_tile_sizes([
        self.genome.get("tile_m", 128),
        self.genome.get("tile_n", 128),
        self.genome.get("tile_k", 64),
    ])

    prec = self.genome.get("precision", "fp32")
    cfg.set_compute_type(prec)

    iter_obj = cfg.build()

    # Execute Kernel
    score = iter_obj.reduce_op(vbt.mul, processed_strain, comb)

    elapsed_ms = (time.perf_counter() - start) * 1000
    speed_reward = 1000.0 / max(elapsed_ms, 1.0)

    # EMA fitness instead of raw accumulation (prevents runaway values)
    self.fitness = (
        self._fitness_ema_alpha * speed_reward
        + (1.0 - self._fitness_ema_alpha) * self.fitness
    )
    self.step += 1

    return {
        "score": float(score.item()) if hasattr(score, "item") else 0.0,
        "elapsed_ms": elapsed_ms,
        "partial_fitness": self.fitness,
        "step": self.step,
        "geo_info": geo_info,
    }

# ------------------------------------------------------------------ #
#  Geometric Enhancement Pipeline                                      #
# ------------------------------------------------------------------ #

def _apply_geometric_enhancements(self, tensor):
    """
    Apply geometry-aware processing to GW strain.

    Accepts:
      (Batch, Time)       — single-detector raw strain
      (Batch, Time, 1)    — single-detector with trailing dim
      (Batch, 3, Time)    — multi-detector (H1, L1, V1)

    Returns:
      (tensor, info_dict) — processed tensor + diagnostic metadata
    """
    info = {"embedding": "none", "operator": "none", "projection_idx": None}

    # 1. Hilbert Embedding for 1D / single-detector data
    if tensor.ndim == 2 or (tensor.ndim == 3 and tensor.shape[1] > 3):
        # Robust squeeze for (Batch, Time, 1)
        if tensor.ndim == 3 and tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1)
        elif tensor.ndim == 3:
            # (Batch, Channels>3, Time) — unusual shape, flatten channels
            b, c, t = tensor.shape
            tensor = tensor.reshape(b, c * t)

        # Analytic signal via Hilbert transform
        analytic = vbt.hilbert(tensor)

        # 3-channel embedding: [strain, quadrature, instantaneous_freq]
        real_part = analytic.real().unsqueeze(1)       # (B, 1, T) — strain
        imag_part = analytic.imag().unsqueeze(1)       # (B, 1, T) — quadrature

        # Instantaneous frequency: d(phase)/dt
        # This is the chirp's fingerprint — f ∝ (t_merge - t)^{-3/8}
        phase = vbt.angle(analytic)                    # (B, T)
        inst_freq = vbt.diff(phase, dim=-1)            # (B, T-1)
        # Pad to match time dimension (replicate last value)
        inst_freq = vbt.cat([
            inst_freq,
            inst_freq[:, -1:],  # replicate final sample
        ], dim=-1)
        inst_freq = inst_freq.unsqueeze(1)             # (B, 1, T)

        tensor_3d = vbt.cat([real_part, imag_part, inst_freq], dim=1)
        info["embedding"] = "hilbert_3ch"

        # Recursively apply rotation/projection to the 3D embedding
        tensor_out, geo_info = self._apply_geometric_enhancements(tensor_3d)
        geo_info["embedding"] = info["embedding"]
        return tensor_out, geo_info

    # 2. Geometric operations on (Batch, 3, Time) data
    if tensor.ndim == 3 and tensor.shape[1] == 3:
        op_matrix = None

        if self.projections:
            # Deterministic selection: genome param > step counter > fallback 0
            idx = self.genome.get(
                "active_projection_idx", self.step
            ) % len(self.projections)
            op_matrix = self.projections[idx]
            info["operator"] = "projection"
            info["projection_idx"] = int(idx)

        elif self.genome.get("apply_rotation", False):
            op_matrix = self.rotation_matrix
            info["operator"] = "rotation"

        if op_matrix is not None:
            op_matrix = self._ensure_vbt_tensor(op_matrix, tensor)

            # Channel mixing: rotate/project dimension 1
            # (B, 3, T) → (B, T, 3) @ (3, 3)^T → (B, T, 3) → (B, 3, T)
            tensor = vbt.matmul(
                tensor.permute(0, 2, 1),
                op_matrix.T,
            ).permute(0, 2, 1)

    return tensor, info

# ------------------------------------------------------------------ #
#  Training Interface (for Drifting Model / MLX bridge)                #
# ------------------------------------------------------------------ #

def get_geometric_params(self) -> dict:
    """
    Export current geometric configuration as a flat dict.
    Useful for serializing cell state or bridging to MLX training.
    """
    params = {
        "rotation_matrix": self.rotation_matrix.tolist(),
        "has_projections": self.projections is not None,
    }
    if self.projections:
        params["projections"] = [p.tolist() for p in self.projections]
        params["active_projection_idx"] = self.genome.get(
            "active_projection_idx", 0
        )
    return params

def update_from_drifting_output(self, generated_params: dict):
    """
    Accept parameters generated by the Drifting Model and update
    this cell's geometric configuration.

    This is the bridge between gradient-based training (Drifting Model)
    and evolutionary selection (Darwinian overseer).
    """
    if "rotation_quaternion" in generated_params:
        q = np.asarray(generated_params["rotation_quaternion"], dtype=np.float64)
        self.rotation_matrix = quaternion_to_matrix(q)
        self.genome["rotation_quaternion"] = q.tolist()

    if "projection_rank" in generated_params:
        self.genome["projection_rank"] = int(generated_params["projection_rank"])
        # Regenerate projections with new rank
        self._setup_geometric_operations()

# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

@staticmethod
def _ensure_vbt_tensor(array, reference_tensor):
    """Coerce numpy array to vibetensor on the same device/dtype as reference."""
    if isinstance(array, np.ndarray):
        return vbt.tensor(
            array,
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )
    return array

def __repr__(self):
    proj_info = f", projections={len(self.projections)}" if self.projections else ""
    return (
        f"TensorCell(uuid={self.uuid}, fitness={self.fitness:.4f}, "
        f"step={self.step}{proj_info})"
    )
```