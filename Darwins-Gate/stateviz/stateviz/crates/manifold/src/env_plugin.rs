use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EnvState { Detached, Attached, Initialized, Running, Paused, Stopping, Error }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSpec {
    pub namespace: String,
    pub names: Vec<String>, // e.g., ["atlas","hud","glyphs"]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvMetrics {
    pub name: String,
    pub fps: f32,
    pub gpu_ms: f32,
    pub cpu_ms: f32,
    pub mem_mb: u32,
    pub dropped_frames: u32,
    pub errors: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum EnvError {
    #[error("attach failed: {0}")] Attach(String),
    #[error("init failed: {0}")] Init(String),
    #[error("tick failed: {0}")] Tick(String),
    #[error("suspend failed: {0}")] Suspend(String),
    #[error("resume failed: {0}")] Resume(String),
    #[error("detach failed: {0}")] Detach(String),
}

pub struct AttachContext {}
pub struct InitContext {}
pub struct SchedCtx {}

#[derive(Clone)]
pub enum TickResult { Idle, DrewFrame }

#[derive(Clone)]
pub enum ManifoldEvent {
    Key(char),
    BudgetChanged,
    Trim(u8), // 0..3 severity
}

pub trait EnvPlugin: Send {
    fn name(&self) -> &'static str;
    fn declare_layers(&self) -> LayerSpec;
    fn attach(&mut self, _ctx: &mut AttachContext) -> Result<(), EnvError>;
    fn init(&mut self, _cx: &mut InitContext) -> Result<(), EnvError>;
    fn tick(&mut self, _dt: f32, _sched: &mut SchedCtx) -> Result<TickResult, EnvError>;
    fn handle_event(&mut self, _e: ManifoldEvent) -> Result<(), EnvError> { Ok(()) }
    fn suspend(&mut self) -> Result<(), EnvError>;
    fn resume(&mut self) -> Result<(), EnvError>;
    fn detach(&mut self) -> Result<(), EnvError>;
}
