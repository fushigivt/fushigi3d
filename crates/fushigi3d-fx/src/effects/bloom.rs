use bytemuck::{Pod, Zeroable};
use wgpu;

use crate::effect::PostProcessEffect;
use crate::fullscreen;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BloomParams {
    params: [f32; 4],
}

pub struct Bloom {
    // Threshold pass
    threshold_pipeline: wgpu::RenderPipeline,
    threshold_bgl: wgpu::BindGroupLayout,
    threshold_ub: wgpu::Buffer,
    // Blur pass (shared for horizontal and vertical)
    blur_pipeline: wgpu::RenderPipeline,
    blur_bgl: wgpu::BindGroupLayout,
    blur_ub_h: wgpu::Buffer,
    blur_ub_v: wgpu::Buffer,
    // Combine pass
    combine_pipeline: wgpu::RenderPipeline,
    combine_bgl: wgpu::BindGroupLayout,
    combine_ub: wgpu::Buffer,
    // Half-res ping-pong textures for blur
    half_a: Option<wgpu::Texture>,
    half_b: Option<wgpu::Texture>,
    half_width: u32,
    half_height: u32,
    // Shared
    sampler: wgpu::Sampler,
    format: wgpu::TextureFormat,
    // Params
    pub threshold: f32,
    pub soft_knee: f32,
    pub intensity: f32,
    pub enabled: bool,
}

impl Bloom {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        let sampler = fullscreen::linear_sampler(device);

        // --- Threshold pass ---
        let threshold_src = format!(
            "{}\n{}",
            fullscreen::FULLSCREEN_VERT_WGSL,
            include_str!("../../shaders/bloom_threshold.wgsl")
        );
        let threshold_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fx_bloom_threshold_shader"),
            source: wgpu::ShaderSource::Wgsl(threshold_src.into()),
        });
        let threshold_bgl = fullscreen::texture_uniform_bind_group_layout(
            device,
            "fx_bloom_threshold_bgl",
            std::mem::size_of::<BloomParams>() as u64,
        );
        let threshold_pipeline = fullscreen::fullscreen_pipeline(
            device,
            "fx_bloom_threshold_pipeline",
            &threshold_shader,
            "fs_bloom_threshold",
            format,
            &threshold_bgl,
        );
        let threshold_ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx_bloom_threshold_ub"),
            size: std::mem::size_of::<BloomParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Blur pass ---
        let blur_src = format!(
            "{}\n{}",
            fullscreen::FULLSCREEN_VERT_WGSL,
            include_str!("../../shaders/bloom_blur.wgsl")
        );
        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fx_bloom_blur_shader"),
            source: wgpu::ShaderSource::Wgsl(blur_src.into()),
        });
        let blur_bgl = fullscreen::texture_uniform_bind_group_layout(
            device,
            "fx_bloom_blur_bgl",
            std::mem::size_of::<BloomParams>() as u64,
        );
        let blur_pipeline = fullscreen::fullscreen_pipeline(
            device,
            "fx_bloom_blur_pipeline",
            &blur_shader,
            "fs_bloom_blur",
            format,
            &blur_bgl,
        );
        let blur_ub_h = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx_bloom_blur_h_ub"),
            size: std::mem::size_of::<BloomParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blur_ub_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx_bloom_blur_v_ub"),
            size: std::mem::size_of::<BloomParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Combine pass ---
        let combine_src = format!(
            "{}\n{}",
            fullscreen::FULLSCREEN_VERT_WGSL,
            include_str!("../../shaders/bloom_combine.wgsl")
        );
        let combine_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fx_bloom_combine_shader"),
            source: wgpu::ShaderSource::Wgsl(combine_src.into()),
        });
        // Combine needs scene + uniform + bloom texture
        let combine_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fx_bloom_combine_bgl"),
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
                                std::mem::size_of::<BloomParams>() as u64,
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
        let combine_pipeline = fullscreen::fullscreen_pipeline(
            device,
            "fx_bloom_combine_pipeline",
            &combine_shader,
            "fs_bloom_combine",
            format,
            &combine_bgl,
        );
        let combine_ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx_bloom_combine_ub"),
            size: std::mem::size_of::<BloomParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let half_width = (width / 2).max(1);
        let half_height = (height / 2).max(1);
        let half_a = Some(create_half_texture(device, half_width, half_height, format, "fx_bloom_half_a"));
        let half_b = Some(create_half_texture(device, half_width, half_height, format, "fx_bloom_half_b"));

        Self {
            threshold_pipeline,
            threshold_bgl,
            threshold_ub,
            blur_pipeline,
            blur_bgl,
            blur_ub_h,
            blur_ub_v,
            combine_pipeline,
            combine_bgl,
            combine_ub,
            half_a,
            half_b,
            half_width,
            half_height,
            sampler,
            format,
            threshold: 0.8,
            soft_knee: 0.2,
            intensity: 0.3,
            enabled: false,
        }
    }
}

fn create_half_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

impl PostProcessEffect for Bloom {
    fn set_params(&mut self, queue: &wgpu::Queue) {
        // Threshold params
        let tp = BloomParams {
            params: [self.threshold, self.soft_knee, 0.0, 0.0],
        };
        queue.write_buffer(&self.threshold_ub, 0, bytemuck::bytes_of(&tp));

        // Blur direction params
        let hw = 1.0 / self.half_width as f32;
        let hh = 1.0 / self.half_height as f32;
        let bh = BloomParams {
            params: [hw, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.blur_ub_h, 0, bytemuck::bytes_of(&bh));
        let bv = BloomParams {
            params: [0.0, hh, 0.0, 0.0],
        };
        queue.write_buffer(&self.blur_ub_v, 0, bytemuck::bytes_of(&bv));

        // Combine params
        let cp = BloomParams {
            params: [self.intensity, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.combine_ub, 0, bytemuck::bytes_of(&cp));
    }

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.half_width = (width / 2).max(1);
        self.half_height = (height / 2).max(1);
        self.half_a = Some(create_half_texture(
            device,
            self.half_width,
            self.half_height,
            self.format,
            "fx_bloom_half_a",
        ));
        self.half_b = Some(create_half_texture(
            device,
            self.half_width,
            self.half_height,
            self.format,
            "fx_bloom_half_b",
        ));
    }

    fn apply(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
    ) {
        let half_a = match &self.half_a {
            Some(t) => t,
            None => return,
        };
        let half_b = match &self.half_b {
            Some(t) => t,
            None => return,
        };
        let view_a = half_a.create_view(&Default::default());
        let view_b = half_b.create_view(&Default::default());

        // 1. Threshold: input → half_a
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fx_bloom_threshold_bg"),
                layout: &self.threshold_bgl,
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
                        resource: self.threshold_ub.as_entire_binding(),
                    },
                ],
            });
            fullscreen::fullscreen_pass(
                encoder,
                &view_a,
                &self.threshold_pipeline,
                &bg,
                "fx_bloom_threshold_pass",
            );
        }

        // 2. Horizontal blur: half_a → half_b
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fx_bloom_blur_h_bg"),
                layout: &self.blur_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_a),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.blur_ub_h.as_entire_binding(),
                    },
                ],
            });
            fullscreen::fullscreen_pass(
                encoder,
                &view_b,
                &self.blur_pipeline,
                &bg,
                "fx_bloom_blur_h_pass",
            );
        }

        // 3. Vertical blur: half_b → half_a
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fx_bloom_blur_v_bg"),
                layout: &self.blur_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_b),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.blur_ub_v.as_entire_binding(),
                    },
                ],
            });
            fullscreen::fullscreen_pass(
                encoder,
                &view_a,
                &self.blur_pipeline,
                &bg,
                "fx_bloom_blur_v_pass",
            );
        }

        // 4. Combine: input (scene) + half_a (blurred bloom) → output
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fx_bloom_combine_bg"),
                layout: &self.combine_bgl,
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
                        resource: self.combine_ub.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&view_a),
                    },
                ],
            });
            fullscreen::fullscreen_pass(
                encoder,
                output,
                &self.combine_pipeline,
                &bg,
                "fx_bloom_combine_pass",
            );
        }
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn name(&self) -> &str {
        "Bloom"
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
