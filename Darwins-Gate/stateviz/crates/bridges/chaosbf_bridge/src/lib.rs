use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosEvent {
    pub t: u64,
    pub kind: ChaosEventKind,
    pub src: u32,
    pub dst: u32,
    pub cell: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChaosEventKind { Replicate, Mutate, Crossover }

#[derive(Debug, Clone)]
pub struct ChaosBuffers<'a> {
    pub grid_w: u32,
    pub grid_h: u32,
    pub energy: &'a [f32],
    pub entropy: &'a [f32],
    pub temperature: &'a [f32],
    pub free_energy: &'a [f32],
    pub lambda: &'a [f32],
    /// Length = w*h*8 (8 operator buckets)
    pub org_instr_mix: &'a [u16],
    pub events: Vec<ChaosEvent>,
}

#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("not attached")] NotAttached,
    #[error("bad buffer length for {0}")] BadBuffer(&'static str),
}

/// Host-side handle that knows how to read WASM memory layout.
/// In your app, implement `from_raw` with real pointers/lengths.
pub struct ChaosBFHost {
    pub w: u32,
    pub h: u32,
    // opaque fields for your pointers/offsets
}

impl ChaosBFHost {
    pub fn from_raw(_w: u32, _h: u32) -> Self {
        Self { w: _w, h: _h }
    }

    /// Return borrowed buffers pointing into WASM memory.
    /// Replace the TODOs with your memory reads; keep validation here.
    pub fn snapshot<'a>(&self) -> Result<ChaosBuffers<'a>, BridgeError> {
        let len = (self.w as usize) * (self.h as usize);
        // TODO: replace with actual memory views
        let empty_f32: &'a [f32] = &[];
        let empty_u16: &'a [u16] = &[];

        Ok(ChaosBuffers {
            grid_w: self.w,
            grid_h: self.h,
            energy: empty_f32,
            entropy: empty_f32,
            temperature: empty_f32,
            free_energy: empty_f32,
            lambda: empty_f32,
            org_instr_mix: empty_u16,
            events: Vec::new(),
        })
    }
}
