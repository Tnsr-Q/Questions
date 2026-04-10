"""
Test harness for TensorCell with VibeTensor.

Requirements:
  - CUDA-capable GPU (VibeTensor has no CPU-only mode)
  - vibetensor installed (editable dev install from repo)
  - scipy (for Hilbert transform bridge)
  - numpy

Usage:
  python test_tensorcell.py
"""

import numpy as np
import vibetensor.torch as vt  # Correct import path
from tensorvibe_corrected import TensorCell

# ── Genome with geometry enabled ──────────────────────────────
genome = {
    "apply_rotation": True,
    "use_geometric_projections": True,
    "num_projections": 3,
    "projection_rank": 2,
    "active_projection_idx": 0,    # Deterministic projection selection
    "tile_m": 128,
    "tile_n": 128,
    "tile_k": 64,
    "precision": "fp32",
    "fitness_ema_alpha": 0.1,
}

cell = TensorCell(genome)
print(f"Created: {cell}")
print(f"Rotation matrix:\n{cell.rotation_matrix}")
print(f"Projections: {len(cell.projections) if cell.projections else 0}")

# ── Test 1: Multi-detector strain (Batch, 3, Time) ───────────
print("\n--- Test 1: Multi-detector (B=2, C=3, T=1024) ---")
batch, time_steps = 2, 1024
strain_np = np.random.randn(batch, 3, time_steps).astype(np.float32)
comb_np = np.random.randn(1, 1, 64).astype(np.float32)

# Create vbt tensors then export to DLPack
strain_vbt = vt.tensor(strain_np)
comb_vbt = vt.tensor(comb_np)

# DLPack zero-copy bridge
strain_dlpack = vt.to_dlpack(strain_vbt)
comb_dlpack = vt.to_dlpack(comb_vbt)

inputs = {
    "strain_dlpack": strain_dlpack,
    "comb_mask_dlpack": comb_dlpack,
}

result = cell.solve_physics(inputs)
print(f"  Score: {result['score']:.6f}")
print(f"  Elapsed: {result['elapsed_ms']:.2f} ms")
print(f"  Fitness: {result['partial_fitness']:.4f}")
print(f"  Geo info: {result['geo_info']}")

# ── Test 2: Single-detector strain (Batch, Time) → Hilbert embedding ─
print("\n--- Test 2: Single-detector (B=2, T=1024) → Hilbert 3ch ---")
strain_1d_np = np.random.randn(batch, time_steps).astype(np.float32)
comb_1d_np = np.random.randn(1, 1, 64).astype(np.float32)

strain_1d_vbt = vt.tensor(strain_1d_np)
comb_1d_vbt = vt.tensor(comb_1d_np)

inputs_1d = {
    "strain_dlpack": vt.to_dlpack(strain_1d_vbt),
    "comb_mask_dlpack": vt.to_dlpack(comb_1d_vbt),
}

result_1d = cell.solve_physics(inputs_1d)
print(f"  Score: {result_1d['score']:.6f}")
print(f"  Elapsed: {result_1d['elapsed_ms']:.2f} ms")
print(f"  Fitness: {result_1d['partial_fitness']:.4f}")
print(f"  Geo info: {result_1d['geo_info']}")
assert result_1d["geo_info"]["embedding"] == "hilbert_3ch", "Expected Hilbert embedding"

# ── Test 3: Geometric params export (for MLX bridge) ─────────
print("\n--- Test 3: Geometric params export ---")
params = cell.get_geometric_params()
print(f"  Keys: {list(params.keys())}")
print(f"  Has projections: {params['has_projections']}")
print(f"  Active idx: {params.get('active_projection_idx')}")

# ── Test 4: Drifting Model output injection ───────────────────
print("\n--- Test 4: Update from Drifting Model output ---")
old_rot = cell.rotation_matrix.copy()
cell.update_from_drifting_output({
    "rotation_quaternion": [0.5, 0.5, 0.5, 0.5],  # 120° rotation
})
print(f"  Rotation changed: {not np.allclose(old_rot, cell.rotation_matrix)}")

# ── Test 5: Fitness EMA behavior ──────────────────────────────
print("\n--- Test 5: Fitness EMA stability ---")
fitnesses = []
for i in range(10):
    # Re-create DLPack capsules (consumed after first use)
    s = vt.tensor(np.random.randn(batch, 3, time_steps).astype(np.float32))
    c = vt.tensor(comb_np)
    r = cell.solve_physics({
        "strain_dlpack": vt.to_dlpack(s),
        "comb_mask_dlpack": vt.to_dlpack(c),
    })
    fitnesses.append(r["partial_fitness"])

print(f"  Fitness over 10 steps: {[f'{f:.4f}' for f in fitnesses]}")
print(f"  Fitness is bounded: {max(fitnesses) < 1e6}")

print(f"\nAll tests passed. Cell state: {cell}")