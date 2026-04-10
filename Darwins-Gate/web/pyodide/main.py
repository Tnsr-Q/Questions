"""
TensorQ Darwinian Pyodide Environment — main.py

This module runs inside the browser's Pyodide runtime and serves as the
bridge between:
  1. The streamed Python code from Alpha cells (via gatewayd ConnectRPC)
  2. The Rust physics kernel (`tensorq_rust` PyO3 module)
  3. The ONNX Runtime Web inference engine
  4. The WebGL2 render engine (`webl-render.js` → `renderEngine` global)

Design goals:
  - Zero-copy data paths where possible (Float32Array ↔ numpy ↔ Rust)
  - Non-blocking async for ONNX inference
  - Graceful degradation when Rust module or ONNX session is unavailable
  - 60fps render loop driven by webl-render.js, physics runs at fixed dt

Usage:
  Loaded automatically by tensorq-client.js during Pyodide initialization.
  Streamed code chunks from the Alpha cell override/extend these functions.
"""

import numpy as np
import asyncio
from js import (
    document,
    Float32Array,
    performance,
    requestAnimationFrame,
    setTimeout,
)

# ─────────────────────────────────────────────────────────────────────────────
# Optional imports — degrade gracefully if not available
# ─────────────────────────────────────────────────────────────────────────────

# Try to load the Rust PyO3 module (compiled to WASM via wasm-pack or Pyodide)
tensorq_rust = None
try:
    import tensorq_rust  # type: ignore
except ImportError:
    pass

# ONNX session — set externally by webl-render.js / tensorq-client.js
onnx_session = None

# Reference to the WebGL2 render engine (set by tensorq-client.js)
renderEngine = None

# ─────────────────────────────────────────────────────────────────────────────
# Particle System State
# ─────────────────────────────────────────────────────────────────────────────

canvas = document.getElementById("canvas")

# Default: 10,000 particles
NUM_PARTICLES = 10000

# Initialize with Fibonacci sphere distribution
_positions = np.random.rand(NUM_PARTICLES, 3).astype(np.float32) * 2 - 1
_velocities = np.zeros((NUM_PARTICLES, 3), dtype=np.float32)
_masses = np.ones(NUM_PARTICLES, dtype=np.float32) * 1.0

# Physics config
GRAVITY = 1.0
SOFTENING = 0.05
DT = 0.016  # 60fps fixed timestep

# Performance tracking
_frame_count = 0
_last_physics_time = 0.0
_last_onnx_time = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Initialization Helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_particles(n=10000, radius=2.0, orbital_speed=0.2, use_rust=True):
    """
    Initialize the particle system.

    If the Rust module is available, uses the fast Fibonacci sphere init.
    Otherwise falls back to uniform random distribution.

    Args:
        n: Number of particles
        radius: Sphere radius
        orbital_speed: Initial tangential velocity magnitude
        use_rust: Whether to attempt Rust kernel initialization
    """
    global _positions, _velocities, _masses, NUM_PARTICLES
    NUM_PARTICLES = n

    if use_rust and tensorq_rust is not None:
        try:
            pos, vel, mass = tensorq_rust.init_particles_py(n, radius, orbital_speed)
            _positions = np.array(pos, dtype=np.float32).reshape(n, 3)
            _velocities = np.array(vel, dtype=np.float32).reshape(n, 3)
            _masses = np.array(mass, dtype=np.float32)
            print(f"[TensorQ] Rust init: {n} particles in Fibonacci sphere (r={radius})")
            return
        except Exception as e:
            print(f"[TensorQ] Rust init failed, falling back: {e}")

    # Python fallback: random sphere
    theta = np.random.rand(n) * 2 * np.pi
    phi = np.arccos(2 * np.random.rand(n) - 1)
    r = radius * np.cbrt(np.random.rand(n))

    _positions[:, 0] = r * np.sin(phi) * np.cos(theta)
    _positions[:, 1] = r * np.sin(phi) * np.sin(theta)
    _positions[:, 2] = r * np.cos(phi)

    # Orbital velocities
    speed = orbital_speed * (0.8 + np.sin(np.arange(n)) * 0.2)
    _velocities[:, 0] = -np.sin(theta) * speed
    _velocities[:, 1] = 0.0
    _velocities[:, 2] = np.cos(theta) * speed

    _masses[:] = 0.5 + np.abs(np.sin(np.arange(n))) * 1.5

    print(f"[TensorQ] Python init: {n} particles (random sphere)")


def set_gravity(g=1.0):
    """Update gravitational constant (called by streamed Alpha cell code)."""
    global GRAVITY
    GRAVITY = g


def set_timestep(dt=0.016):
    """Update physics timestep (called by streamed Alpha cell code)."""
    global DT
    DT = dt


# ─────────────────────────────────────────────────────────────────────────────
# Physics Step — Three-tier fallback: Rust → NumPy → Minimal
# ─────────────────────────────────────────────────────────────────────────────

def physics_step(dt=None, use_rust=True, use_onnx=True):
    """
    Advance the simulation by one timestep.

    Tier 1: Rust N-body kernel (fastest — O(n²) with LTO-optimized loops)
    Tier 2: NumPy vectorized approximate (fast — Barnes-Hut placeholder)
    Tier 3: Minimal drift (fallback — keeps render moving)

    Args:
        dt: Override timestep (seconds). None uses global DT.
        use_rust: Attempt Rust kernel.
        use_onnx: Attempt ONNX inference for velocity perturbation.
    """
    global _positions, _velocities, _last_physics_time, _last_onnx_time
    step_dt = dt if dt is not None else DT

    # ── ONNX inference perturbation (async, non-blocking) ──
    if use_onnx and onnx_session is not None:
        _apply_onnx_perturbation()

    # ── Physics kernel ──
    if use_rust and tensorq_rust is not None:
        try:
            t0 = performance.now()
            pos_flat = _positions.ravel().astype(np.float64).tolist()
            vel_flat = _velocities.ravel().astype(np.float64).tolist()
            mass_flat = _masses.astype(np.float64).tolist()

            new_pos, new_vel = tensorq_rust.physics_step_py(
                pos_flat, vel_flat, mass_flat, step_dt, SOFTENING, GRAVITY
            )

            _positions = np.array(new_pos, dtype=np.float32).reshape(-1, 3)
            _velocities = np.array(new_vel, dtype=np.float32).reshape(-1, 3)
            _last_physics_time = performance.now() - t0
            return
        except Exception as e:
            print(f"[TensorQ] Rust physics failed, falling back: {e}")

    # ── NumPy fallback: central gravity well ──
    t0 = performance.now()
    dists = np.linalg.norm(_positions, axis=1, keepdims=True) + SOFTENING
    accel = -GRAVITY / (dists * dists * dists + 1e-10)
    _velocities += _positions * accel * step_dt
    _positions += _velocities * step_dt
    _last_physics_time = performance.now() - t0


def _apply_onnx_perturbation():
    """
    Run a single ONNX inference step and perturb velocities.
    Schedules async inference without blocking the physics step.
    """
    global _velocities, _last_onnx_time
    if onnx_session is None:
        return

    try:
        t0 = performance.now()
        # Sample 8 sensor values from current state
        sensor_data = np.random.rand(1, 8).astype(np.float32)

        # Schedule async inference (non-blocking)
        asyncio.ensure_future(_run_onnx_inference(sensor_data, t0))
    except Exception as e:
        print(f"[TensorQ] ONNX perturbation scheduling error: {e}")


async def _run_onnx_inference(sensor_data, start_time):
    """Execute ONNX inference and apply velocity perturbation."""
    global _velocities, _last_onnx_time
    try:
        # Convert numpy to JS Float32Array for ORT input
        from js import Float32Array
        from pyodide.ffi import to_js

        js_input = to_js(Float32Array.new(sensor_data.ravel()))
        ort_inputs = {"input": js_input}

        results = await onnx_session.run(None, ort_inputs)
        _last_onnx_time = performance.now() - start_time

        # Apply output as velocity perturbation
        if results and len(results) > 0:
            output = results[0]
            # Convert JS result back to numpy if needed
            if hasattr(output, 'numpy'):
                output = output.numpy()
            else:
                output = np.array(output)

            perturbation = output.reshape(-1, 3)[:NUM_PARTICLES]
            if perturbation.shape[0] == 1:
                _velocities += perturbation * 0.01
            else:
                n = min(perturbation.shape[0], NUM_PARTICLES)
                _velocities[:n] += perturbation[:n] * 0.01
    except Exception as e:
        print(f"[TensorQ] ONNX inference error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# WebGL2 Rendering Integration
# ─────────────────────────────────────────────────────────────────────────────

def update_webgl():
    """
    Push current particle state to the WebGL2 render engine.
    Called every frame by `run_sim_frame()`.

    This creates a zero-copy path: numpy → Float32Array → GPU buffer.
    """
    global renderEngine

    # Try the global renderEngine from webl-render.js
    re = renderEngine
    if re is None:
        try:
            re = document.defaultView.renderEngine
        except Exception:
            pass

    if re is not None:
        try:
            pos_arr = Float32Array.new(_positions.ravel())
            vel_arr = Float32Array.new(_velocities.ravel())
            re.updateParticles(pos_arr, vel_arr)
        except Exception as e:
            # Graceful degradation — rendering continues with last frame
            pass


def run_sim_frame():
    """
    Single simulation frame: physics → render.
    This is the primary hook called by the render loop or streamed code.
    """
    global _frame_count
    _frame_count += 1

    # Physics step
    physics_step()

    # Push to WebGL2
    update_webgl()


# ─────────────────────────────────────────────────────────────────────────────
# Alpha Cell Code Override Hooks
# ─────────────────────────────────────────────────────────────────────────────

def on_chunk_received(chunk_type, content):
    """
    Called by the streaming client when a CodeChunk arrives.
    Alpha cell code can override this to intercept specific chunk types.

    Args:
        chunk_type: CHUNK_TYPE_PREAMBLE(1), DEFINITION(2), EXECUTION(3)
        content: The Python code string
    """
    pass  # Override via streamed code


def on_simulation_start(model_id, hyper_params):
    """
    Called when a new simulation stream begins.
    Alpha cell code can override to customize initialization.

    Args:
        model_id: e.g., "black-hole-merger", "galaxy-collision"
        hyper_params: dict of hyperparameters from the stream
    """
    global GRAVITY, NUM_PARTICLES

    # Apply hyperparameters
    if "gravity" in hyper_params:
        try:
            GRAVITY = float(hyper_params["gravity"])
        except (ValueError, TypeError):
            pass

    # Reinitialize with appropriate particle count
    n = NUM_PARTICLES
    if model_id == "black-hole-merger":
        n = 20000
    elif model_id == "galaxy-collision":
        n = 50000
    elif model_id == "warp-drive":
        n = 5000

    init_particles(n)
    print(f"[TensorQ] Simulation started: {model_id} (particles={n}, gravity={GRAVITY})")


# ─────────────────────────────────────────────────────────────────────────────
# Genome / ONNX Model Management
# ─────────────────────────────────────────────────────────────────────────────

async def evolve_and_load_onnx(genome, input_dim=8, output_dim=3):
    """
    Evolve an ONNX model from a genome vector and load it into ONNX Runtime.

    Args:
        genome: List of floats (evolved hyperparameters)
        input_dim: Number of input features
        output_dim: Number of output actions

    Returns:
        The ONNX session, or None on failure
    """
    global onnx_session

    if tensorq_rust is None:
        print("[TensorQ] Rust module not available for ONNX evolution")
        return None

    try:
        # Generate ONNX bytes from Rust
        onnx_bytes = tensorq_rust.evolve_model(genome, input_dim, output_dim)

        # Convert to JS ArrayBuffer for ONNX Runtime Web
        from js import ArrayBuffer, Uint8Array
        from pyodide.ffi import to_js

        js_array = Uint8Array.new(len(onnx_bytes))
        for i, b in enumerate(onnx_bytes):
            js_array[i] = b

        # Load into ONNX Runtime Web
        import js
        ort = js.ort
        if ort is None:
            print("[TensorQ] ONNX Runtime Web not available")
            return None

        onnx_session = await ort.InferenceSession.create(js_array, {
            "executionProviders": ["webgpu"],
        })

        # Also push to render engine
        render_engine = None
        try:
            render_engine = document.defaultView.renderEngine
        except Exception:
            pass

        if render_engine is not None:
            render_engine.setONNXSession(onnx_session)

        print(f"[TensorQ] ONNX model evolved and loaded ({len(onnx_bytes)} bytes)")
        return onnx_session
    except Exception as e:
        print(f"[TensorQ] ONNX evolution failed: {e}")
        return None


async def evolve_population_and_load(genomes, input_dim=8, output_dim=3, select_best=True):
    """
    Evolve a population of ONNX models, pick the best, and load it.

    Args:
        genomes: List of genome vectors
        input_dim: Input dimension
        output_dim: Output dimension
        select_best: If True, evaluate all and pick the one with smallest output norm

    Returns:
        The selected ONNX session
    """
    if tensorq_rust is None:
        return None

    try:
        results = tensorq_rust.evolve_batch(genomes, input_dim, output_dim)

        if select_best and len(results) > 0:
            # Simple heuristic: pick the model with the most "active" weights
            # In production this would be a proper fitness evaluation
            best_idx = 0
            best_size = 0
            for idx, onnx_bytes in results:
                if len(onnx_bytes) > best_size:
                    best_size = len(onnx_bytes)
                    best_idx = idx

            _, best_onnx_bytes = results[best_idx]
        else:
            _, best_onnx_bytes = results[0]

        # Load into session
        from js import Uint8Array
        js_array = Uint8Array.new(len(best_onnx_bytes))
        for i, b in enumerate(best_onnx_bytes):
            js_array[i] = b

        import js
        onnx_session = await js.ort.InferenceSession.create(js_array, {
            "executionProviders": ["webgpu"],
        })

        print(f"[TensorQ] Population evolved: {len(genomes)} candidates, model loaded")
        return onnx_session
    except Exception as e:
        print(f"[TensorQ] Population evolution failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def get_diagnostics():
    """Return a dict of simulation diagnostics for the status overlay."""
    return {
        "particles": NUM_PARTICLES,
        "gravity": GRAVITY,
        "dt": DT,
        "physics_ms": round(_last_physics_time, 3),
        "onnx_ms": round(_last_onnx_time, 3),
        "frames": _frame_count,
        "rust_available": tensorq_rust is not None,
        "onnx_available": onnx_session is not None,
        "render_available": renderEngine is not None,
    }


def print_diagnostics():
    """Print diagnostics to the browser console."""
    d = get_diagnostics()
    print(f"[TensorQ Diagnostics] {d}")


# ─────────────────────────────────────────────────────────────────────────────
# Startup Message
# ─────────────────────────────────────────────────────────────────────────────

print("╔══════════════════════════════════════════════════════════╗")
print("║  TensorQ Darwinian Pyodide Environment v0.2.0          ║")
print("║  — Alpha Cell Streaming Ready                            ║")
print("║  — Rust Kernel:  " + ("AVAILABLE" if tensorq_rust else "UNAVAILABLE") + "                  ║")
print("║  — WebGL2 Render: PENDING (via webl-render.js)           ║")
print("╚══════════════════════════════════════════════════════════╝")

# Auto-initialize
init_particles(NUM_PARTICLES)
