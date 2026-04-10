use stateviz_manifold::env_plugin::*;

pub struct Redshift;

impl EnvPlugin for Redshift {
    fn name(&self) -> &'static str { "redshift_plugin" }
    fn declare_layers(&self) -> LayerSpec {
        LayerSpec { namespace: "redshift_plugin".into(), names: vec!["redshift_atlas", "redshift_residuals", "redshift_ppc"] }
    }
    fn attach(&mut self, _ctx: &mut AttachContext) -> Result<(), EnvError> { Ok(()) }
    fn init(&mut self, _cx: &mut InitContext) -> Result<(), EnvError> { Ok(()) }
    fn tick(&mut self, _dt: f32, _sched: &mut SchedCtx) -> Result<TickResult, EnvError> { Ok(TickResult::Idle) }
    fn suspend(&mut self) -> Result<(), EnvError> { Ok(()) }
    fn resume(&mut self) -> Result<(), EnvError> { Ok(()) }
    fn detach(&mut self) -> Result<(), EnvError> { Ok(()) }
}
