use crate::env_plugin::EnvMetrics;

#[derive(Default, Clone, Debug)]
pub struct RollingStats {
    pub fps: f32,
    pub gpu_ms: f32,
    pub cpu_ms: f32,
}
impl RollingStats {
    pub fn update(&mut self, fps: f32, gpu_ms: f32, cpu_ms: f32) {
        self.fps = 0.9*self.fps + 0.1*fps;
        self.gpu_ms = 0.9*self.gpu_ms + 0.1*gpu_ms;
        self.cpu_ms = 0.9*self.cpu_ms + 0.1*cpu_ms;
    }
}
pub fn snapshot(name: &str, stats: &RollingStats, mem_mb: u32, dropped: u32, errors: u32) -> EnvMetrics {
    EnvMetrics {
        name: name.to_string(),
        fps: stats.fps, gpu_ms: stats.gpu_ms, cpu_ms: stats.cpu_ms,
        mem_mb, dropped_frames: dropped, errors
    }
}
