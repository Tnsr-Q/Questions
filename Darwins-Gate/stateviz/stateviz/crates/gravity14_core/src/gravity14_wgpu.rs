use wgpu::*;
use crate::buffers::PullbackH3;

pub struct PullbackPipeline {
    pub pipeline: ComputePipeline,
    pub layout: PipelineLayout,
    pub bind_group_layout: BindGroupLayout,
}

impl PullbackPipeline {
    pub fn new(device: &Device, shader_src: &str) -> Self {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("pullback_metric.wgsl"),
            source: ShaderSource::Wgsl(shader_src.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("pullback-bgl"),
            entries: &[
                BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ]
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("pullback-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("pullback-pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: "main",
            compilation_options: PipelineCompilationOptions::default(),
        });
        Self { pipeline, layout, bind_group_layout }
    }

    pub fn dispatch(&self,
        device: &Device, queue: &Queue,
        buf_g: &Buffer, buf_j: &Buffer, buf_h: &Buffer,
        tiles: u32
    ) {
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("pullback-bindgroup"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: buf_g.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: buf_j.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: buf_h.as_entire_binding() },
            ]
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: Some("pullback-encoder") });
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("pullback-pass") });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(tiles, 1, 1);
        }
        queue.submit([encoder.finish()]);
    }
}
