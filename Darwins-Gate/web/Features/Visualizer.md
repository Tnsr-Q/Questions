Changes Summary

    rust/src/lib.rs — Complete Rewrite
     - Removed the broken include_bytes!("../../models/minimal_ppo_policy.onnx")
        that referenced a non-existent file
     - Added a hand-written ONNX v1.0 protobuf serializer that builds
       Tanh(MatMul(input, weight) + bias) graphs in memory — zero external
       dependencies
     - Added physics_step_py() — O(n²) N-body gravitational kernel with
       symplectic Euler integration, LTO-optimized
     - Added init_particles_py() — Fibonacci sphere distribution for uniform
       particle placement
     - Added evolve_batch() — parallel population evolution for GA selection
     - Proper PyO3 signatures with #[pyo3(signature = (...))] for optional
       defaults
     - `cargo.toml` — added [profile.release] with lto = "fat", codegen-units = 
       1, strip = true for maximum WASM speed

    web/pyodide/main.py — Complete Rewrite
     - Three-tier physics fallback: Rust kernel → NumPy vectorized → minimal
       drift
     - Non-blocking ONNX inference: asyncio.ensure_future() schedules inference
       without blocking the 60fps physics loop
     - `renderEngine` bridge: update_webgl() pushes Float32Array
       positions/velocities to the WebGL2 engine
     - Alpha cell hooks: on_chunk_received(), on_simulation_start() for streamed
        code override
     - Population evolution: evolve_and_load_onnx(),
       evolve_population_and_load() for GA-driven model loading
     - Graceful degradation: All optional imports (Rust, ONNX, render) fail
       silently

    web/assets/webl-render.js — Written From Scratch
     - WebGL2 particle renderer with 100k capacity, interleaved VBO
       (pos/vel/mass/color)
     - Animated spatial grid with wave distortion
     - Spherical camera with mouse drag orbit + scroll zoom + touch pinch-zoom
     - `onChunk()` method — receives CodeChunk from streaming client, tracks
       stats
     - `setONNXSession()` — receives loaded models, tracks inference latency
     - Live ESM export of renderEngine singleton for zero-friction cross-module
       access

    web/js/tensorq-client.js — Rewired
     - Fixed circular import via import * as weblRender live binding pattern
     - Proper chunk routing: content/type/sequenceId matching the actual proto
       fields
     - Touch support for mobile orbit/pinch-zoom
     - Graceful standalone mode: runs local simulation when gateway is
       unreachable
     - Loading overlay coordination

    web/index.html — Fixed
     - Was escaped HTML inside <p> tags (Cocoa Text Editor artifact)
     - Now proper HTML5 with canvas, status overlay, loading spinner, ONNX
       Runtime Web CDN script
