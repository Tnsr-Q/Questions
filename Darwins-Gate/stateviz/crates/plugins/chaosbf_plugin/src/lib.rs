use stateviz_manifold::env_plugin::*;

pub struct ChaosBF;

impl EnvPlugin for ChaosBF {
    fn name(&self) -> &'static str { "chaosbf_plugin" }
    fn declare_layers(&self) -> LayerSpec {
        LayerSpec { namespace: "chaosbf_plugin".into(), names: vec!["chaosbf_atlas", "chaosbf_gauges", "chaosbf_glyphs", "chaosbf_lineage"] }
    }
    fn attach(&mut self, _ctx: &mut AttachContext) -> Result<(), EnvError> { Ok(()) }
    fn init(&mut self, _cx: &mut InitContext) -> Result<(), EnvError> { Ok(()) }
    fn tick(&mut self, _dt: f32, _sched: &mut SchedCtx) -> Result<TickResult, EnvError> { Ok(TickResult::Idle) }
    fn suspend(&mut self) -> Result<(), EnvError> { Ok(()) }
    fn resume(&mut self) -> Result<(), EnvError> { Ok(()) }
    fn detach(&mut self) -> Result<(), EnvError> { Ok(()) }
}
