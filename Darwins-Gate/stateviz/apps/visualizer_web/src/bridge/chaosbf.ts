/**
 * ChaosBF WASM bridge (stub).
 * Bind to the ChaosBF sim's exported functions without re-implementing its host loop.
 * Replace the `// TODO` markers with actual WASM instance exports from your build.
 */

export type ChaosEvent = {
  t: number;
  kind: "replicate" | "mutate" | "crossover";
  src: number;
  dst: number;
  cell: number;
};

export type ChaosBuffers = {
  grid_w: number;
  grid_h: number;
  energy: Float32Array;
  entropy: Float32Array;
  temperature: Float32Array;
  free_energy: Float32Array;
  lambda: Float32Array;
  org_instr_mix: Uint16Array; // length = w*h*8 (8 buckets)
  events: ChaosEvent[];
};

let _wasm: any | null = null;
let _ptrs: { [k: string]: number } | null = null;

export function attachChaosBF(wasmInstance: any) {
  _wasm = wasmInstance;
  // Example expected exports (adapt names to your build):
  // - init_sim(w, h)
  // - step_sim()
  // - get_metrics_ptr(), get_output_ptr(), get_events_ptr()
  // - memory (WebAssembly.Memory)
}

export function snapshotChaosBF(): ChaosBuffers {
  if (!_wasm) throw new Error("ChaosBF WASM not attached");
  // TODO: read pointers & lengths from _wasm, create typed views over memory.buffer
  // For now, return an empty grid so the visualizer compiles.
  return {
    grid_w: 0,
    grid_h: 0,
    energy: new Float32Array(),
    entropy: new Float32Array(),
    temperature: new Float32Array(),
    free_energy: new Float32Array(),
    lambda: new Float32Array(),
    org_instr_mix: new Uint16Array(),
    events: []
  };
}
