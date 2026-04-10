use wgpu::*;

pub struct Gfx {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

impl Gfx {
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn new_headless() -> anyhow::Result<Self> {
        let instance = Instance::default();
        let adapter = instance.request_adapter(&RequestAdapterOptions {
            compatible_surface: None,
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
        }).await.ok_or_else(|| anyhow::anyhow!("No adapter"))?;

        let (device, queue) = adapter.request_device(&DeviceDescriptor {
            features: Features::empty(),
            limits: Limits::downlevel_defaults(),
            label: Some("stateviz-device"),
        }, None).await?;

        Ok(Self { instance, adapter, device, queue })
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn new_web(canvas: &web_sys::HtmlCanvasElement) -> anyhow::Result<(Self, wgpu::Surface<'static>)> {
        use wasm_bindgen::JsCast;
        let instance = Instance::default();
        let surface = instance.create_surface(canvas).map_err(|e| anyhow::anyhow!(format!("{e:?}")))?;
        let adapter = instance.request_adapter(&RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
        }).await.ok_or_else(|| anyhow::anyhow!("No adapter"))?;

        let (device, queue) = adapter.request_device(&DeviceDescriptor {
            features: Features::empty(),
            limits: Limits::downlevel_defaults(),
            label: Some("stateviz-device-web"),
        }, None).await?;

        Ok((Self { instance, adapter, device, queue }, surface))
    }
}
