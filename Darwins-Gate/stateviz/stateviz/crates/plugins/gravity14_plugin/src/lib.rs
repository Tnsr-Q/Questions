use stateviz_manifold::env_plugin::*;

pub struct Gravity14;

impl EnvPlugin for Gravity14 {
    fn name(&self) -> &'static str { "gravity14_plugin" }
    fn declare_layers(&self) -> LayerSpec {
        LayerSpec { namespace: "gravity14_plugin".into(), names: vec!["slicer3d", "curvature_lens", "measure_tools"] }
    }
    fn attach(&mut self, _ctx: &mut AttachContext) -> Result<(), EnvError> { Ok(()) }
    fn init(&mut self, _cx: &mut InitContext) -> Result<(), EnvError> { Ok(()) }
    fn tick(&mut self, _dt: f32, _sched: &mut SchedCtx) -> Result<TickResult, EnvError> { Ok(TickResult::Idle) }
    fn suspend(&mut self) -> Result<(), EnvError> { Ok(()) }
    fn resume(&mut self) -> Result<(), EnvError> { Ok(()) }
    fn detach(&mut self) -> Result<(), EnvError> { Ok(()) }
}
