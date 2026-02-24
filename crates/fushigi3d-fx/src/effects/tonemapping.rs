use bytemuck::{Pod, Zeroable};
use wgpu;

use crate::effect::PostProcessEffect;
use crate::fullscreen;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TonemapMode {
    Reinhard = 0,
    Aces = 1,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TonemapParams {
    mode_exposure: [f32; 4],
}

pub struct Tonemapping {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    pub mode: TonemapMode,
    pub exposure: f32,
    pub enabled: bool,
}

impl Tonemapping {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader_src = format!(
            "{}\n{}",
            fullscreen::FULLSCREEN_VERT_WGSL,
            include_str!("../../shaders/tonemapping.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fx_tonemap_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = fullscreen::texture_uniform_bind_group_layout(
            device,
            "fx_tonemap_bgl",
            std::mem::size_of::<TonemapParams>() as u64,
        );

        let pipeline = fullscreen::fullscreen_pipeline(
            device,
            "fx_tonemap_pipeline",
            &shader,
            "fs_tonemap",
            format,
            &bind_group_layout,
        );

        let sampler = fullscreen::linear_sampler(device);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx_tonemap_ub"),
            size: std::mem::size_of::<TonemapParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            uniform_buffer,
            mode: TonemapMode::Aces,
            exposure: 1.0,
            enabled: true,
        }
    }
}

impl PostProcessEffect for Tonemapping {
    fn set_params(&mut self, queue: &wgpu::Queue) {
        let params = TonemapParams {
            mode_exposure: [self.mode as u32 as f32, self.exposure, 0.0, 0.0],
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
            label: Some("fx_tonemap_bg"),
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
        fullscreen::fullscreen_pass(encoder, output, &self.pipeline, &bind_group, "fx_tonemap_pass");
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn name(&self) -> &str {
        "Tonemapping"
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
