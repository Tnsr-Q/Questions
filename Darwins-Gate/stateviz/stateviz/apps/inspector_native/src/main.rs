use stateviz_manifold::manifold::Manifold;
use stateviz_ppf_floquet_plugin::PpfFloquet;
use stateviz_gravity14_plugin::Gravity14;
use stateviz_redshift_plugin::Redshift;
use stateviz_chaosbf_plugin::ChaosBF;

fn main() -> anyhow::Result<()> {
    let mut mf = Manifold::new();
    mf.register(Box::new(PpfFloquet));
    mf.register(Box::new(Gravity14));
    mf.register(Box::new(Redshift));
    mf.register(Box::new(ChaosBF));
    println!("Manifold initialized with 4 plugins registered.");
    // Wire your real UI here (egui/GPUI) to toggle mf.enable()/disable()
    Ok(())
}
