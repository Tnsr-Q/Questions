# Additions: wgpu pipelines + wasm-bindgen entry

- **crates/stateviz-gfx-wgpu**: shared device/queue init. `new_headless()` for native, `new_web(canvas)` for WASM.
- **gravity14 pullback pipeline**: `gravity14_wgpu::PullbackPipeline` builds and dispatches the WGSL compute shader for `h = J^T g J`.
- **crates/stateviz-engine-wasm**: exports `init(canvas_id)` over `wasm-bindgen`, sets up WebGPU, and builds the pullback pipeline.

## How to hook up
- Native: call `Gfx::new_headless().await` and use `PullbackPipeline::dispatch()` with your buffers for G/J/H.
- Web: compile `stateviz-engine-wasm` to wasm; in your web app, import the pkg and `await init("canvas-id")`.
- Replace the dummy buffers with your real `wgpu::Buffer` uploads (metric tiles, Jacobians, and output H-store).

This keeps the surface minimal and stable — you can swap in real kernels without changing the Manifold or plugin contracts.
