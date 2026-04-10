#[derive(Clone, Debug)]
pub struct Budget {
    pub gpu_mb: u32,
    pub cpu_ms: u32,
    pub mem_mb: u32,
}
impl Default for Budget {
    fn default() -> Self { Self { gpu_mb: 256, cpu_ms: 4, mem_mb: 512 } }
}
