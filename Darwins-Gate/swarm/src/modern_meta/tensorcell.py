“””
TensorCell — Geometry-aware GW strain processing cell for Darwinian swarm.

Option A integration: Full-spectrum KK causality enforcement.

Pipeline:
raw strain → geometric (rotation/projection) → FFT(full) → KK enforce → IFFT → comb reduction

CausalEnforcer is genome-evolvable:

- kk_tolerance ∈ [0, 1]: blend measured vs minimum-phase
- enforce_kk: bool safety valve (Darwinian selection can kill KK if it hurts)

omega comes from build_waveform_with_echoes → meta[‘omega_full’] → inputs dict.
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
from causal_enforcer import CausalEnforcer

class TensorCell:
def **init**(self, genome: dict):
self.genome = genome
self.fitness = 0.0
self._fitness_ema_alpha = genome.get(“fitness_ema_alpha”, 0.1)
self.uuid = str(uuid.uuid4())[:8]
self.pod_name = socket.gethostname()
self.step = 0

```
    # Seed RNG from UUID for deterministic mutations/projections
    self.rng = np.random.default_rng(int(self.uuid, 16))

    self._setup_geometric_operations()

    # CausalEnforcer — genome-evolvable KK params
    kk_params = genome.get("kk_params", {})
    self.causal_enforcer = CausalEnforcer(**kk_params)

def _setup_geometric_operations(self):
    """Genome-driven geometry config — evolvable for parity sensitivity."""
    # 1. Rotation: SO(3) channel mixing
    if self.genome.get("apply_rotation", False):
        q = self.genome.get("rotation_quaternion")
        if q is None:
            q = random_unit_quaternion(rng=self.rng)
        self.rotation_matrix = vbt.tensor(quaternion_to_matrix(q))
    else:
        self.rotation_matrix = vbt.eye(3)

    # 2. Projections: low-rank projectors
    if self.genome.get("use_geometric_projections", False):
        self.projections = generate_geometric_projections(
            n_proj=self.genome.get("num_projections", 3),
            rank=self.genome.get("projection_rank", 2),
            rng=self.rng,
        )
        self.projections = [vbt.tensor(P) for P in self.projections]
        self.proj_idx = 0
    else:
        self.projections = None

# ------------------------------------------------------------------ #
#  Core Physics Solver                                                 #
# ------------------------------------------------------------------ #

def solve_physics(self, inputs: dict) -> dict:
    """
    Run comb/parity kernel on causality-enforced, geometry-enhanced strain.

    Parameters
    ----------
    inputs : dict
        Required keys:
          'strain_dlpack'    — DLPack capsule for strain tensor
          'comb_mask_dlpack' — DLPack capsule for comb mask tensor
        Optional keys (needed for KK enforcement):
          'omega'            — full-spectrum angular frequency array (np.ndarray or vbt.tensor)
                               from meta['omega_full']. If absent, KK step is skipped.
          'sample_rate'      — float, Hz (for diagnostics)

    Returns
    -------
    dict with score, elapsed_ms, partial_fitness, step, geo_info, kk_info.
    """
    start = time.perf_counter()

    # Decode tensors from DLPack
    strain = vbt.from_dlpack(inputs["strain_dlpack"])
    comb = vbt.from_dlpack(inputs["comb_mask_dlpack"])

    # ── Phase 1: Geometric enhancements ──────────────────────────
    processed_strain = self._apply_geometric_enhancements(strain)

    # ── Phase 2: KK causality enforcement (full-spectrum FFT) ────
    kk_info = {"applied": False, "kk_tolerance": self.causal_enforcer.kk_tolerance}

    if "omega" in inputs and self.causal_enforcer.enforce_kk:
        omega = inputs["omega"]
        if isinstance(omega, np.ndarray):
            omega = vbt.tensor(omega)

        # Full-spectrum FFT of processed strain
        # For multi-channel (B, 3, T): apply KK per channel
        if processed_strain.ndim == 3:
            causal_channels = []
            for ch in range(processed_strain.shape[1]):
                channel = processed_strain[:, ch, :]         # (B, T)
                # FFT along time axis — process each batch element
                # For batch processing: flatten batch, apply, unflatten
                causal_ch = self._apply_kk_per_channel(channel, omega)
                causal_channels.append(causal_ch.unsqueeze(1))
            processed_strain = vbt.cat(causal_channels, dim=1)  # (B, 3, T)
        elif processed_strain.ndim == 2:
            # (B, T) — single-channel case
            processed_strain = self._apply_kk_per_channel(processed_strain, omega)

        kk_info["applied"] = True

    # ── Phase 3: Comb reduction ──────────────────────────────────
    cfg = vbt.core.TensorIterConfig()
    cfg.check_mem_overlap(True)
    cfg.is_reduction(True)

    cfg.add_input(processed_strain)
    cfg.add_input(comb)
    cfg.add_output(vbt.empty_like(processed_strain[..., 0:1]))

    cfg.set_tile_sizes([
        self.genome.get("tile_m", 128),
        self.genome.get("tile_n", 128),
        self.genome.get("tile_k", 64),
    ])

    prec = self.genome.get("precision", "fp32")
    cfg.set_compute_type(prec)

    iter_obj = cfg.build()
    score = iter_obj.reduce_op(vbt.mul, processed_strain, comb)

    # ── Fitness & diagnostics ────────────────────────────────────
    elapsed_ms = (time.perf_counter() - start) * 1000
    speed_reward = 1000.0 / max(elapsed_ms, 1.0)

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
        "applied_rotation": self.genome.get("apply_rotation", False),
        "applied_projection": self.projections is not None,
        "kk_info": kk_info,
    }

def _apply_kk_per_channel(self, channel: vbt.tensor, omega: vbt.tensor) -> vbt.tensor:
    """
    Apply Kramers-Kronig enforcement to a single (B, T) channel.

    Full-spectrum path:
      channel(t) → FFT → KK enforce → IFFT → causal_channel(t)

    omega must be full-spectrum (N bins from np.fft.fftfreq, not rfftfreq).
    """
    B, T = channel.shape

    # Process each batch element (omega is shared across batch)
    causal_slices = []
    for b in range(B):
        ch_b = channel[b]                           # (T,)
        freq_domain = vbt.fft(ch_b)                 # (T,) complex
        causal_freq = self.causal_enforcer.apply_kramers_kronig(omega, freq_domain)
        causal_time = vbt.real(vbt.ifft(causal_freq))  # Back to time domain, real part
        causal_slices.append(causal_time.unsqueeze(0))

    return vbt.cat(causal_slices, dim=0)            # (B, T)

# ------------------------------------------------------------------ #
#  Geometric Enhancement Pipeline                                      #
# ------------------------------------------------------------------ #

def _apply_geometric_enhancements(self, tensor):
    """
    Apply rotation/projection — safe for 1D/2D/3D strain.

    Accepts:
      (Batch, Time)       — single-detector raw strain
      (Batch, Time, 1)    — single-detector with trailing dim
      (Batch, 3, Time)    — multi-detector (H1, L1, V1) or Hilbert-embedded
    """
    if tensor.ndim < 2:
        return tensor

    # Projections: apply if enabled and last dim matches
    if self.projections:
        P = self.projections[self.proj_idx]
        self.proj_idx = (self.proj_idx + 1) % len(self.projections)

        if tensor.shape[-1] == P.shape[0]:
            return vbt.matmul(tensor, P.T)

    # Rotation: apply SO(3) if last dim is 3
    if self.genome.get("apply_rotation", False) and tensor.shape[-1] == 3:
        return vbt.matmul(tensor, self.rotation_matrix.T)

    return tensor

# ------------------------------------------------------------------ #
#  Training Interface (Drifting Model / MLX bridge)                    #
# ------------------------------------------------------------------ #

def get_geometric_params(self) -> dict:
    """Export current geometric + KK config for serialization or MLX bridge."""
    params = {
        "rotation_matrix": self.rotation_matrix.tolist()
            if hasattr(self.rotation_matrix, 'tolist')
            else self.rotation_matrix,
        "has_projections": self.projections is not None,
        "kk_params": self.causal_enforcer.get_params(),
    }
    if self.projections:
        params["projections"] = [
            p.tolist() if hasattr(p, 'tolist') else p
            for p in self.projections
        ]
        params["active_projection_idx"] = self.genome.get(
            "active_projection_idx", 0
        )
    return params

def update_from_drifting_output(self, generated_params: dict):
    """
    Accept Drifting Model output and update cell geometry + KK config.

    Bridge between gradient-based training (MLX) and
    evolutionary selection (Darwinian overseer).
    """
    if "rotation_quaternion" in generated_params:
        q = np.asarray(generated_params["rotation_quaternion"], dtype=np.float64)
        self.rotation_matrix = vbt.tensor(quaternion_to_matrix(q))
        self.genome["rotation_quaternion"] = q.tolist()

    if "projection_rank" in generated_params:
        self.genome["projection_rank"] = int(generated_params["projection_rank"])
        self._setup_geometric_operations()

    if "kk_params" in generated_params:
        self.causal_enforcer.set_params(generated_params["kk_params"])
        self.genome["kk_params"] = self.causal_enforcer.get_params()

# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def __repr__(self):
    proj_info = f", projections={len(self.projections)}" if self.projections else ""
    kk_info = f", kk={self.causal_enforcer.kk_tolerance}" if self.causal_enforcer.enforce_kk else ""
    return (
        f"TensorCell(uuid={self.uuid}, fitness={self.fitness:.4f}, "
        f"step={self.step}{proj_info}{kk_info})"
    )
```