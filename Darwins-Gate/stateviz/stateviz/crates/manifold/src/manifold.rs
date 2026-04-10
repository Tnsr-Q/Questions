use std::collections::BTreeMap;
use parking_lot::RwLock;
use crate::env_plugin::{EnvPlugin, EnvState, AttachContext, InitContext, SchedCtx, TickResult, ManifoldEvent};
use crate::resource_budget::Budget;
use crate::router::Router;

pub struct Manifold {
    envs: BTreeMap<String, Box<dyn EnvPlugin>>,
    state: BTreeMap<String, EnvState>,
    budget: BTreeMap<String, Budget>,
    router: RwLock<Router>,
}

impl Manifold {
    pub fn new() -> Self {
        Self { envs: BTreeMap::new(), state: BTreeMap::new(), budget: BTreeMap::new(), router: RwLock::new(Router::default()) }
    }

    pub fn register(&mut self, env: Box<dyn EnvPlugin>) {
        let name = env.name().to_string();
        self.budget.insert(name.clone(), Budget::default());
        self.state.insert(name.clone(), EnvState::Detached);
        self.envs.insert(name, env);
    }

    pub fn enable(&mut self, name: &str) -> anyhow::Result<()> {
        let env = self.envs.get_mut(name).ok_or_else(|| anyhow::anyhow!("env not found"))?;
        {
            let mut r = self.router.write();
            r.mount_namespace(name);
        }
        env.attach(&mut AttachContext{})?;
        env.init(&mut InitContext{})?;
        self.state.insert(name.to_string(), EnvState::Running);
        Ok(())
    }

    pub fn disable(&mut self, name: &str) -> anyhow::Result<()> {
        let env = self.envs.get_mut(name).ok_or_else(|| anyhow::anyhow!("env not found"))?;
        env.detach()?;
        {
            let mut r = self.router.write();
            r.unmount_namespace(name);
        }
        self.state.insert(name.to_string(), EnvState::Detached);
        Ok(())
    }

    pub fn tick(&mut self, dt: f32) {
        let mut sched = SchedCtx{};
        for (name, env) in self.envs.iter_mut() {
            if self.state.get(name) != Some(&EnvState::Running) { continue; }
            let _ = env.tick(dt, &mut sched).unwrap_or(TickResult::Idle);
        }
    }

    pub fn event(&mut self, e: ManifoldEvent) {
        for env in self.envs.values_mut() { let _ = env.handle_event(e.clone()); }
    }
}
