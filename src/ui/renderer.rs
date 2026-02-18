//! wgpu rendering pipeline for VRM models.
//!
//! Manages vertex/index/uniform buffers, offscreen render-to-texture with depth,
//! and per-frame vertex buffer updates from CPU-skinned data.

#![cfg(feature = "native-ui")]

use std::sync::Mutex;

use bytemuck::{Pod, Zeroable};
use eframe::wgpu;
use glam::{Mat4, Vec3, Vec4};

use super::vrm_loader::VrmModel;

/// Vertex layout matching the shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

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
}

/// Mutable offscreen state behind a Mutex for resize support.
struct OffscreenState {
    offscreen_texture: wgpu::Texture,
    offscreen_view: wgpu::TextureView,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    blit_bind_group: wgpu::BindGroup,
    offscreen_size: [u32; 2],
    projection_matrix: Mat4,
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
    // Camera
    view_matrix: Mat4,
    // Light directions (normalized, in world space — pointing FROM light TO origin)
    lights: [LightInfo; 3],
    ambient: [f32; 4],
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

        // Scene bind group layout (uniform buffer)
        let scene_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vrm_scene_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let scene_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vrm_scene_pl"),
                bind_group_layouts: &[&scene_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Scene render pipeline (offscreen, Rgba8UnormSrgb)
        let offscreen_format = wgpu::TextureFormat::Rgba8UnormSrgb;
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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vrm_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vrm_blit_bg"),
            layout: &blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&offscreen_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Create draw calls for each primitive of each mesh
        let mut draw_calls = Vec::new();
        let mut prim_map = Vec::new();

        for (mesh_idx, mesh) in model.meshes.iter().enumerate() {
            for (prim_idx, prim) in mesh.primitives.iter().enumerate() {
                if prim.positions.is_empty() || prim.indices.is_empty() {
                    continue;
                }

                let vertices: Vec<Vertex> = prim
                    .positions
                    .iter()
                    .zip(prim.normals.iter())
                    .map(|(p, n)| Vertex {
                        position: [p.x, p.y, p.z],
                        normal: [n.x, n.y, n.z],
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

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("vrm_bg_{}_{}", mesh_idx, prim_idx)),
                    layout: &scene_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    }],
                });

                draw_calls.push(DrawCall {
                    vertex_buffer,
                    index_buffer,
                    uniform_buffer,
                    bind_group,
                    num_indices: prim.indices.len() as u32,
                    num_vertices: prim.positions.len() as u32,
                    base_color: prim.base_color,
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
                blit_bind_group,
                offscreen_size: [width, height],
                projection_matrix,
            }),
            blit_bind_group_layout,
            sampler,
            scene_bind_group_layout,
            view_matrix,
            lights,
            ambient: [0.15, 0.15, 0.18, 1.0],
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

        let offscreen_format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let (tex, view) = create_color_texture(device, width, height, offscreen_format);
        state.offscreen_texture = tex;
        state.offscreen_view = view;

        let (dtex, dview) = create_depth_texture(device, width, height);
        state.depth_texture = dtex;
        state.depth_view = dview;

        // Recreate blit bind group with new texture view
        state.blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vrm_blit_bg"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&state.offscreen_view),
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

                let vertices: Vec<Vertex> = positions
                    .iter()
                    .zip(normals.iter())
                    .map(|(p, n)| Vertex {
                        position: [p.x, p.y, p.z],
                        normal: [n.x, n.y, n.z],
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
        let mvp = state.projection_matrix * self.view_matrix;
        let model_mat = Mat4::IDENTITY;

        // Update all uniform buffers
        for dc in &self.draw_calls {
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
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.12,
                            g: 0.12,
                            b: 0.16,
                            a: 1.0,
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

/// Convert euler angles (rx, ry, rz in radians) to a light direction vector.
/// The direction points from the light towards the origin.
fn euler_to_direction(rx: f32, ry: f32, _rz: f32) -> Vec3 {
    // Build rotation from euler angles, then extract -Z direction
    let rot = glam::Mat3::from_rotation_y(ry) * glam::Mat3::from_rotation_x(rx);
    let dir = rot * Vec3::NEG_Z;
    dir.normalize()
}
