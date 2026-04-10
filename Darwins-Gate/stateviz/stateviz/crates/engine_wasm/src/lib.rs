use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{window, HtmlCanvasElement};
use stateviz_gfx_wgpu::Gfx;
use stateviz_gravity14_core::gravity14_wgpu::PullbackPipeline;

#[wasm_bindgen(start)]
pub fn start() {
    // Optional: panic hook or console log setup
}

#[wasm_bindgen]
pub async fn init(canvas_id: String) -> Result<JsValue, JsValue> {
    let doc = window().ok_or("no window")?.document().ok_or("no document")?;
    let canvas = doc.get_element_by_id(&canvas_id).ok_or("canvas not found")?
        .dyn_into::<HtmlCanvasElement>().map_err(|_| "not a canvas")?;
    let (gfx, _surface) = Gfx::new_web(&canvas).await.map_err(|e| JsValue::from_str(&format!("{e:?}")))?;

    // Prepare a dummy pipeline so the binding compiles; you will feed real buffers.
    let shader_src = include_str!("../../gravity14_core/shaders/pullback_metric.wgsl");
    let _pipe = PullbackPipeline::new(&gfx.device, shader_src);

    // Return a handle marker for now; expand to real state handle later.
    Ok(JsValue::from_str("engine-wasm-initialized"))
}
