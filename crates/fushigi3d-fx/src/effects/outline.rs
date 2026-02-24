use bytemuck::{Pod, Zeroable};
use wgpu;

use crate::effect::PostProcessEffect;
use crate::fullscreen;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct OutlineParams {
    params: [f32; 4],  // thickness, threshold, 1/width, 1/height
    color: [f32; 4],   // rgb + strength
}

pub struct Outline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    depth_view: Option<wgpu::TextureView>,
    width: u32,
    height: u32,
    pub thickness: f32,
    pub threshold: f32,
    pub color: [f32; 3],
    pub strength: f32,
    pub enabled: bool,
}

impl Outline {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader_src = format!(
            "{}\n{}",
            fullscreen::FULLSCREEN_VERT_WGSL,
            include_str!("../../shaders/outline.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fx_outline_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Custom layout: color texture + sampler + uniform + depth texture
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fx_outline_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZero::new(
                                std::mem::size_of::<OutlineParams>() as u64,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline = fullscreen::fullscreen_pipeline(
            device,
            "fx_outline_pipeline",
            &shader,
            "fs_outline",
            format,
            &bind_group_layout,
        );

        let sampler = fullscreen::linear_sampler(device);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx_outline_ub"),
            size: std::mem::size_of::<OutlineParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            uniform_buffer,
            depth_view: None,
            width: 1,
            height: 1,
            thickness: 1.0,
            threshold: 0.01,
            color: [0.0, 0.0, 0.0],
            strength: 1.0,
            enabled: false,
        }
    }

    /// Set the depth texture view from the scene render. Must be called each frame.
    pub fn set_depth_view(&mut self, view: wgpu::TextureView) {
        self.depth_view = Some(view);
    }
}

impl PostProcessEffect for Outline {
    fn set_params(&mut self, queue: &wgpu::Queue) {
        let params = OutlineParams {
            params: [
                self.thickness,
                self.threshold,
                1.0 / self.width as f32,
                1.0 / self.height as f32,
            ],
            color: [self.color[0], self.color[1], self.color[2], self.strength],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&params));
    }

    fn resize(&mut self, _device: &wgpu::Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    fn apply(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
    ) {
        let depth_view = match &self.depth_view {
            Some(v) => v,
            None => return,
        };

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx_outline_bg"),
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
            ],
        });
        fullscreen::fullscreen_pass(encoder, output, &self.pipeline, &bind_group, "fx_outline_pass");
    }

    fn enabled(&self) -> bool {
        self.enabled && self.depth_view.is_some()
    }

    fn name(&self) -> &str {
        "Outline"
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
