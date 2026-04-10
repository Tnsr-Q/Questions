/// WGSL std430-like buffer layouts for pullback metrics (documentation-only placeholders).

/// Each entry packs a 3x3 pullback metric h in row-major order.
/// Align to 16 bytes per row when mapping to WGSL storage buffers.
#[repr(C)]
pub struct PullbackH3 {
    pub h00: f32, pub h01: f32, pub h02: f32, pub _pad0: f32,
    pub h10: f32, pub h11: f32, pub h12: f32, pub _pad1: f32,
    pub h20: f32, pub h21: f32, pub h22: f32, pub _pad2: f32,
}
