use bytemuck::{Pod, Zeroable};
use wgpu;

use crate::effect::PostProcessEffect;
use crate::fullscreen;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct VignetteParams {
    params: [f32; 4], // intensity, radius, softness, _pad
}

pub struct Vignette {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    pub intensity: f32,
    pub radius: f32,
    pub softness: f32,
    pub enabled: bool,
}

impl Vignette {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader_src = format!(
            "{}\n{}",
            fullscreen::FULLSCREEN_VERT_WGSL,
            include_str!("../../shaders/vignette.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fx_vignette_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = fullscreen::texture_uniform_bind_group_layout(
            device,
            "fx_vignette_bgl",
            std::mem::size_of::<VignetteParams>() as u64,
        );

        let pipeline = fullscreen::fullscreen_pipeline(
            device,
            "fx_vignette_pipeline",
            &shader,
            "fs_vignette",
            format,
            &bind_group_layout,
        );

        let sampler = fullscreen::linear_sampler(device);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx_vignette_ub"),
            size: std::mem::size_of::<VignetteParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            uniform_buffer,
            intensity: 0.5,
            radius: 0.4,
            softness: 0.3,
            enabled: false,
        }
    }
}

impl PostProcessEffect for Vignette {
    fn set_params(&mut self, queue: &wgpu::Queue) {
        let params = VignetteParams {
            params: [self.intensity, self.radius, self.softness, 0.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&params));
    }

    fn resize(&mut self, _device: &wgpu::Device, _width: u32, _height: u32) {}

    fn apply(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx_vignette_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        fullscreen::fullscreen_pass(encoder, output, &self.pipeline, &bind_group, "fx_vignette_pass");
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn name(&self) -> &str {
        "Vignette"
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
