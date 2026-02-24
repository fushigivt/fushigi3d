//! wgpu rendering pipeline for VRM models.
//!
//! Manages vertex/index/uniform buffers, offscreen render-to-texture with depth,
//! and per-frame vertex buffer updates from CPU-skinned data.

#![cfg(feature = "native-ui")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use bytemuck::{Pod, Zeroable};
use eframe::wgpu;
use glam::{Mat4, Vec3, Vec4};

use fushigi3d_fx::PostProcessChain;

use super::vrm_loader::VrmModel;

/// Vertex layout matching the shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// Uniform buffer layout matching the shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {
    pub mvp: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
    pub light_dir_0: [f32; 4],
    pub light_dir_1: [f32; 4],
    pub light_dir_2: [f32; 4],
    pub light_col_0: [f32; 4],
    pub light_col_1: [f32; 4],
    pub light_col_2: [f32; 4],
    pub ambient: [f32; 4],
    pub base_color: [f32; 4],
    /// x: 1.0 = sample texture, 0.0 = base_color only
    pub use_texture: [f32; 4],
}

/// One draw call's GPU resources (vertex buffer, index buffer, uniform buffer).
#[allow(dead_code)]
struct DrawCall {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    num_indices: u32,
    num_vertices: u32,
    base_color: [f32; 4],
    has_texture: bool,
}

/// Mutable offscreen state behind a Mutex for resize support.
struct OffscreenState {
    offscreen_texture: wgpu::Texture,
    offscreen_view: wgpu::TextureView,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// Final output texture (post-processing result). Blit reads from this.
    final_texture: wgpu::Texture,
    final_view: wgpu::TextureView,
    blit_bind_group: wgpu::BindGroup,
    offscreen_size: [u32; 2],
    projection_matrix: Mat4,
    view_matrix: Mat4,
    mirrored: bool,
}

/// The VRM renderer. Holds all GPU resources for offscreen rendering.
#[allow(dead_code)]
pub struct VrmRenderer {
    // Pipelines
    scene_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    // Per-primitive draw calls (flattened across all meshes)
    draw_calls: Vec<DrawCall>,
    // Primitive metadata for updating vertex buffers
    prim_map: Vec<PrimRef>,
    // Offscreen render target (mutable for resize)
    offscreen: Mutex<OffscreenState>,
    // Blit resources
    blit_bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    // Scene layout
    scene_bind_group_layout: wgpu::BindGroupLayout,
    // Light directions (normalized, in world space — pointing FROM light TO origin)
    lights: [LightInfo; 3],
    ambient: [f32; 4],
    // GPU textures (kept alive for bind group references)
    _textures: Vec<wgpu::Texture>,
    // Global toggle: use textures vs white shading
    use_textures: AtomicBool,
    // Transparent background mode
    transparent_bg: AtomicBool,
    // Post-processing chain
    fx_chain: Mutex<PostProcessChain>,
}

struct LightInfo {
    direction: Vec3,
    color_intensity: Vec4,
}

/// Tracks which mesh/primitive a draw call corresponds to.
#[allow(dead_code)]
struct PrimRef {
    mesh_idx: usize,
    _prim_idx: usize,
}

impl VrmRenderer {
    /// Create a new renderer from a loaded VRM model.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_format: wgpu::TextureFormat,
        model: &VrmModel,
        width: u32,
        height: u32,
    ) -> Self {
        // Shader modules
        let shader_src = include_str!("shader.wgsl");
        let scene_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vrm_scene_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vrm_blit_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Scene bind group layout (uniform buffer + texture + sampler)
        let scene_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vrm_scene_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let scene_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vrm_scene_pl"),
                bind_group_layouts: &[&scene_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Scene render pipeline (offscreen HDR for post-processing chain)
        let offscreen_format = wgpu::TextureFormat::Rgba16Float;
        let scene_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vrm_scene_pipeline"),
            layout: Some(&scene_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: offscreen_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // VRM models often have double-sided materials
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        // Blit bind group layout (texture + sampler)
        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vrm_blit_bgl"),
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
                ],
            });

        let blit_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vrm_blit_pl"),
                bind_group_layouts: &[&blit_bind_group_layout],
                push_constant_ranges: &[],
            });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vrm_blit_pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_blit"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_blit"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        // Create offscreen textures
        let (offscreen_texture, offscreen_view) =
            create_color_texture(device, width, height, offscreen_format);
        let (depth_texture, depth_view) = create_depth_texture(device, width, height);

        // Create final output texture + post-processing chain
        let (final_texture, final_view) =
            create_color_texture(device, width, height, fushigi3d_fx::HDR_FORMAT);

        let fx_chain = {
            let mut chain = PostProcessChain::new(device, width, height);
            // Effects order: outline → bloom → vignette → color_grading → chromatic_ab → film_grain → tonemapping (last)
            chain.push(Box::new(fushigi3d_fx::effects::outline::Outline::new(device, fushigi3d_fx::HDR_FORMAT)));
            chain.push(Box::new(fushigi3d_fx::effects::bloom::Bloom::new(device, fushigi3d_fx::HDR_FORMAT, width, height)));
            chain.push(Box::new(fushigi3d_fx::effects::vignette::Vignette::new(device, fushigi3d_fx::HDR_FORMAT)));
            chain.push(Box::new(fushigi3d_fx::effects::color_grading::ColorGrading::new(device, fushigi3d_fx::HDR_FORMAT)));
            chain.push(Box::new(fushigi3d_fx::effects::chromatic_ab::ChromaticAb::new(device, fushigi3d_fx::HDR_FORMAT)));
            chain.push(Box::new(fushigi3d_fx::effects::film_grain::FilmGrain::new(device, fushigi3d_fx::HDR_FORMAT)));
            chain.push(Box::new(fushigi3d_fx::effects::tonemapping::Tonemapping::new(device, fushigi3d_fx::HDR_FORMAT)));
            chain
        };

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vrm_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Blit reads from chain output (final_view)
        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vrm_blit_bg"),
            layout: &blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&final_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Create 1x1 white fallback texture for primitives without a texture
        let fallback_texture = create_rgba_texture(device, queue, &[255, 255, 255, 255], 1, 1, "vrm_fallback_tex");
        let fallback_view = fallback_texture.create_view(&Default::default());

        // Create GPU textures for each primitive and build draw calls
        let mut draw_calls = Vec::new();
        let mut prim_map = Vec::new();
        let mut gpu_textures = vec![fallback_texture];

        for (mesh_idx, mesh) in model.meshes.iter().enumerate() {
            for (prim_idx, prim) in mesh.primitives.iter().enumerate() {
                if prim.positions.is_empty() || prim.indices.is_empty() {
                    continue;
                }

                let vertices: Vec<Vertex> = prim
                    .positions
                    .iter()
                    .zip(prim.normals.iter())
                    .zip(prim.uvs.iter())
                    .map(|((p, n), uv)| Vertex {
                        position: [p.x, p.y, p.z],
                        normal: [n.x, n.y, n.z],
                        uv: *uv,
                    })
                    .collect();

                let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("vrm_vb_{}_{}", mesh_idx, prim_idx)),
                    size: (vertices.len() * std::mem::size_of::<Vertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vertices));

                let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("vrm_ib_{}_{}", mesh_idx, prim_idx)),
                    size: (prim.indices.len() * std::mem::size_of::<u32>()) as u64,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&index_buffer, 0, bytemuck::cast_slice(&prim.indices));

                let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("vrm_ub_{}_{}", mesh_idx, prim_idx)),
                    size: std::mem::size_of::<Uniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                // Create or use fallback texture
                let (tex_view, has_texture) = if let Some(ref tex) = prim.texture {
                    let gpu_tex = create_rgba_texture(
                        device, queue, &tex.pixels, tex.width, tex.height,
                        &format!("vrm_tex_{}_{}", mesh_idx, prim_idx),
                    );
                    let view = gpu_tex.create_view(&Default::default());
                    gpu_textures.push(gpu_tex);
                    (view, true)
                } else {
                    (fallback_view.clone(), false)
                };

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("vrm_bg_{}_{}", mesh_idx, prim_idx)),
                    layout: &scene_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&tex_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                    ],
                });

                draw_calls.push(DrawCall {
                    vertex_buffer,
                    index_buffer,
                    uniform_buffer,
                    bind_group,
                    num_indices: prim.indices.len() as u32,
                    num_vertices: prim.positions.len() as u32,
                    base_color: prim.base_color,
                    has_texture,
                });

                prim_map.push(PrimRef {
                    mesh_idx,
                    _prim_idx: prim_idx,
                });
            }
        }

        // Camera: perspective yfov=π/8.5, position (0, 1.33, 0.88), look at (0, 1.33, 0)
        let aspect = width as f32 / height as f32;
        let yfov = std::f32::consts::PI / 8.5;
        let projection_matrix = Mat4::perspective_rh(yfov, aspect, 0.01, 100.0);
        let view_matrix = Mat4::look_at_rh(
            Vec3::new(0.0, 1.33, 0.88),
            Vec3::new(0.0, 1.33, 0.0),
            Vec3::Y,
        );

        // Lights adapted from demo_vrm.py (intensities reduced for non-PBR shader)
        let lights = [
            // Key light: warm, front-right
            LightInfo {
                direction: euler_to_direction(-0.4, 0.5, 0.0),
                color_intensity: Vec4::new(1.0 * 1.2, 0.95 * 1.2, 0.9 * 1.2, 1.0),
            },
            // Fill light: cool, front-left
            LightInfo {
                direction: euler_to_direction(-0.3, -0.6, 0.0),
                color_intensity: Vec4::new(0.8 * 0.5, 0.85 * 0.5, 1.0 * 0.5, 1.0),
            },
            // Rim light: behind
            LightInfo {
                direction: euler_to_direction(0.2, std::f32::consts::PI, 0.0),
                color_intensity: Vec4::new(0.3, 0.3, 0.4, 1.0),
            },
        ];

        Self {
            scene_pipeline,
            blit_pipeline,
            draw_calls,
            prim_map,
            offscreen: Mutex::new(OffscreenState {
                offscreen_texture,
                offscreen_view,
                depth_texture,
                depth_view,
                final_texture,
                final_view,
                blit_bind_group,
                offscreen_size: [width, height],
                projection_matrix,
                view_matrix,
                mirrored: true,
            }),
            blit_bind_group_layout,
            sampler,
            scene_bind_group_layout,
            lights,
            ambient: [0.15, 0.15, 0.18, 1.0],
            _textures: gpu_textures,
            use_textures: AtomicBool::new(true),
            transparent_bg: AtomicBool::new(false),
            fx_chain: Mutex::new(fx_chain),
        }
    }

    /// Resize the offscreen render target if the viewport size changed.
    pub fn resize(&self, device: &wgpu::Device, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        let mut state = self.offscreen.lock().unwrap();

        if state.offscreen_size == [width, height] {
            return;
        }
        state.offscreen_size = [width, height];

        let offscreen_format = wgpu::TextureFormat::Rgba16Float;
        let (tex, view) = create_color_texture(device, width, height, offscreen_format);
        state.offscreen_texture = tex;
        state.offscreen_view = view;

        let (dtex, dview) = create_depth_texture(device, width, height);
        state.depth_texture = dtex;
        state.depth_view = dview;

        // Resize post-processing chain and final texture
        let (ftex, fview) = create_color_texture(device, width, height, fushigi3d_fx::HDR_FORMAT);
        state.final_texture = ftex;
        state.final_view = fview;
        let mut chain = self.fx_chain.lock().unwrap();
        chain.resize(device, width, height);

        // Recreate blit bind group with new final view
        state.blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vrm_blit_bg"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&state.final_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        // Update projection matrix for new aspect ratio
        let aspect = width as f32 / height as f32;
        let yfov = std::f32::consts::PI / 8.5;
        state.projection_matrix = Mat4::perspective_rh(yfov, aspect, 0.01, 100.0);
    }

    /// Set horizontal mirror mode (selfie-style flip).
    pub fn set_mirrored(&self, mirrored: bool) {
        self.offscreen.lock().unwrap().mirrored = mirrored;
    }

    /// Set the camera distance (Z position) for zoom control.
    pub fn set_camera_distance(&self, distance: f32) {
        let mut state = self.offscreen.lock().unwrap();
        state.view_matrix = Mat4::look_at_rh(
            Vec3::new(0.0, 1.33, distance),
            Vec3::new(0.0, 1.33, 0.0),
            Vec3::Y,
        );
    }

    /// Toggle texture rendering on/off (true = textured, false = white/lit only).
    pub fn set_use_textures(&self, enabled: bool) {
        self.use_textures.store(enabled, Ordering::Relaxed);
    }

    /// Toggle transparent background (for OBS window capture with alpha).
    pub fn set_transparent_bg(&self, enabled: bool) {
        self.transparent_bg.store(enabled, Ordering::Relaxed);
    }

    /// Access the post-processing chain for effect parameter adjustments.
    pub fn fx_chain(&self) -> std::sync::MutexGuard<'_, PostProcessChain> {
        self.fx_chain.lock().unwrap()
    }

    /// Update vertex buffers with skinned positions and base normals.
    ///
    /// `skinned_meshes`: per-mesh Vec of per-primitive Vec of skinned positions.
    /// Must match the order of meshes/primitives from the model.
    pub fn update_vertices(
        &self,
        queue: &wgpu::Queue,
        model: &VrmModel,
        skinned_meshes: &[Vec<Vec<glam::Vec3>>],
    ) {
        let mut dc_idx = 0;
        for (mesh_idx, mesh) in model.meshes.iter().enumerate() {
            for (prim_idx, prim) in mesh.primitives.iter().enumerate() {
                if prim.positions.is_empty() || prim.indices.is_empty() {
                    continue;
                }
                if dc_idx >= self.draw_calls.len() {
                    break;
                }

                let positions = &skinned_meshes[mesh_idx][prim_idx];
                let normals = &prim.normals;
                let uvs = &prim.uvs;

                let vertices: Vec<Vertex> = positions
                    .iter()
                    .zip(normals.iter())
                    .zip(uvs.iter())
                    .map(|((p, n), uv)| Vertex {
                        position: [p.x, p.y, p.z],
                        normal: [n.x, n.y, n.z],
                        uv: *uv,
                    })
                    .collect();

                queue.write_buffer(
                    &self.draw_calls[dc_idx].vertex_buffer,
                    0,
                    bytemuck::cast_slice(&vertices),
                );

                dc_idx += 1;
            }
        }
    }

    /// Render the scene offscreen. Call this in `prepare()`.
    pub fn render_offscreen(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let state = self.offscreen.lock().unwrap();
        let model_mat = if state.mirrored {
            Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0))
        } else {
            Mat4::IDENTITY
        };
        let mvp = state.projection_matrix * state.view_matrix * model_mat;
        let use_tex = self.use_textures.load(Ordering::Relaxed);
        let transparent = self.transparent_bg.load(Ordering::Relaxed);

        // Update all uniform buffers
        for dc in &self.draw_calls {
            let tex_flag = if use_tex && dc.has_texture { 1.0f32 } else { 0.0 };
            let uniforms = Uniforms {
                mvp: mvp.to_cols_array_2d(),
                model: model_mat.to_cols_array_2d(),
                light_dir_0: self.lights[0].direction.extend(0.0).to_array(),
                light_dir_1: self.lights[1].direction.extend(0.0).to_array(),
                light_dir_2: self.lights[2].direction.extend(0.0).to_array(),
                light_col_0: self.lights[0].color_intensity.to_array(),
                light_col_1: self.lights[1].color_intensity.to_array(),
                light_col_2: self.lights[2].color_intensity.to_array(),
                ambient: self.ambient,
                base_color: dc.base_color,
                use_texture: [tex_flag, 0.0, 0.0, 0.0],
            };
            queue.write_buffer(&dc.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vrm_offscreen_encoder"),
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vrm_offscreen_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &state.offscreen_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(if transparent {
                            wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }
                        } else {
                            wgpu::Color { r: 0.12, g: 0.12, b: 0.16, a: 1.0 }
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &state.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.scene_pipeline);

            for dc in &self.draw_calls {
                pass.set_bind_group(0, &dc.bind_group, &[]);
                pass.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
                pass.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..dc.num_indices, 0, 0..1);
            }
        }

        // Run post-processing chain: scene HDR → final LDR
        {
            let mut chain = self.fx_chain.lock().unwrap();
            chain.set_params(queue);
            if chain.has_enabled_effects() {
                chain.run(device, &mut encoder, &state.offscreen_view, &state.final_view);
            } else {
                // No effects: copy scene to final so blit has valid data
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &state.offscreen_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &state.final_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: state.offscreen_size[0],
                        height: state.offscreen_size[1],
                        depth_or_array_layers: 1,
                    },
                );
            }
        }

        // Drop lock before submit
        drop(state);
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Blit the offscreen texture to the current render pass. Call this in `paint()`.
    pub fn blit(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        let state = self.offscreen.lock().unwrap();
        render_pass.set_pipeline(&self.blit_pipeline);
        render_pass.set_bind_group(0, Some(&state.blit_bind_group), &[]);
        drop(state);
        render_pass.draw(0..3, 0..1); // fullscreen triangle
    }
}

fn create_color_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vrm_offscreen_color"),
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
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vrm_offscreen_depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

/// Create a GPU texture from RGBA8 pixel data.
fn create_rgba_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &[u8],
    width: u32,
    height: u32,
    label: &str,
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    texture
}

/// Convert euler angles (rx, ry, rz in radians) to a light direction vector.
/// The direction points from the light towards the origin.
fn euler_to_direction(rx: f32, ry: f32, _rz: f32) -> Vec3 {
    // Build rotation from euler angles, then extract -Z direction
    let rot = glam::Mat3::from_rotation_y(ry) * glam::Mat3::from_rotation_x(rx);
    let dir = rot * Vec3::NEG_Z;
    dir.normalize()
}
