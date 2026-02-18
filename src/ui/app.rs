//! Main egui application with VRM 3D viewport.

use std::sync::Arc;
use std::time::Instant;

use eframe::egui;

use crate::avatar::AvatarState;
use crate::AppState;

use super::animation;
use super::blendshape_map::BlendshapeMapper;
use super::renderer::VrmRenderer;
use super::skinning;
use super::viewport::VrmViewportCallback;
use super::vrm_loader::VrmModel;

/// The native egui application window.
pub struct RustuberApp {
    state: Arc<AppState>,
    /// Broadcast receiver for avatar state updates (sync-safe via try_recv)
    state_rx: tokio::sync::broadcast::Receiver<AvatarState>,
    /// Cached latest avatar state (updated each frame via try_recv)
    cached_avatar: AvatarState,
    /// VRM model (loaded once)
    model: Option<Arc<VrmModel>>,
    /// GPU renderer (created from wgpu render state)
    renderer: Option<Arc<VrmRenderer>>,
    /// ARKit → VRM blendshape mapper
    mapper: Option<BlendshapeMapper>,
    /// Start time for animation clock
    start_time: Instant,
    /// Error message if model failed to load
    load_error: Option<String>,
}

impl RustuberApp {
    pub fn new(cc: &eframe::CreationContext<'_>, state: Arc<AppState>) -> Self {
        let state_rx = state.subscribe_state();
        let start_time = Instant::now();

        let mut app = Self {
            state,
            state_rx,
            cached_avatar: AvatarState::default(),
            model: None,
            renderer: None,
            mapper: None,
            start_time,
            load_error: None,
        };

        // Try to load VRM model and initialize renderer
        app.init_vrm(cc);

        app
    }

    fn init_vrm(&mut self, cc: &eframe::CreationContext<'_>) {
        let render_state = match cc.wgpu_render_state.as_ref() {
            Some(rs) => rs,
            None => {
                self.load_error = Some("wgpu render state not available".to_string());
                return;
            }
        };

        // Load model
        let config = {
            // We can't await here (sync context), so use try_lock or blocking
            // Since this is init, the config won't be contended
            let rt = tokio::runtime::Handle::try_current();
            match rt {
                Ok(handle) => {
                    // We're in a tokio context, use block_in_place
                    tokio::task::block_in_place(|| {
                        handle.block_on(self.state.config.read()).clone()
                    })
                }
                Err(_) => {
                    // Not in tokio context; try to read config path from default
                    crate::config::Config::default()
                }
            }
        };

        let model_path = &config.avatar.vrm.model_path;

        let model = match VrmModel::load(model_path) {
            Ok(m) => {
                tracing::info!(
                    "VRM model loaded: {} meshes, {} morph targets, {} bones",
                    m.meshes.len(),
                    m.morph_target_names.len(),
                    m.bone_to_node.len()
                );
                Arc::new(m)
            }
            Err(e) => {
                self.load_error = Some(format!("Failed to load VRM model: {}", e));
                tracing::error!("{}", self.load_error.as_ref().unwrap());
                return;
            }
        };

        // Create blendshape mapper
        let mapper = BlendshapeMapper::new(&model.morph_target_names);

        // Create renderer
        let device = &render_state.device;
        let queue = &render_state.queue;
        let target_format = render_state.target_format;

        let renderer = Arc::new(VrmRenderer::new(
            device,
            queue,
            target_format,
            &model,
            800,
            600,
        ));

        // Do initial skinning with rest pose
        let rotations = animation::idle_pose(&model);
        let world = skinning::compute_world_transforms(&model, &rotations);

        let mut skinned_meshes = Vec::new();
        for (mesh_idx, _mesh) in model.meshes.iter().enumerate() {
            let base = skinning::base_positions(&model, mesh_idx);
            let skinned = skinning::skin_vertices(&model, mesh_idx, &base, &world);
            skinned_meshes.push(skinned);
        }
        renderer.update_vertices(queue, &model, &skinned_meshes);

        self.model = Some(model);
        self.renderer = Some(renderer);
        self.mapper = Some(mapper);
    }

    /// Launch the native UI window. Blocks until the window is closed.
    pub fn run(state: Arc<AppState>) -> eframe::Result {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_title("rustuber")
                .with_inner_size([800.0, 600.0]),
            ..Default::default()
        };

        eframe::run_native(
            "rustuber",
            options,
            Box::new(move |cc| Ok(Box::new(Self::new(cc, state)))),
        )
    }

    /// Drain the broadcast channel and cache the latest avatar state.
    fn update_cached_state(&mut self) {
        // Drain all pending messages, keep the last one
        loop {
            match self.state_rx.try_recv() {
                Ok(new_state) => {
                    self.cached_avatar = new_state;
                }
                Err(tokio::sync::broadcast::error::TryRecvError::Empty) => break,
                Err(tokio::sync::broadcast::error::TryRecvError::Lagged(n)) => {
                    tracing::debug!("Avatar state receiver lagged by {} messages", n);
                    // Continue draining
                }
                Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
            }
        }
    }

    /// Update skinning for the current frame.
    fn update_skinning(&self, render_state: &eframe::egui_wgpu::RenderState) {
        let model = match &self.model {
            Some(m) => m,
            None => return,
        };
        let renderer = match &self.renderer {
            Some(r) => r,
            None => return,
        };
        let mapper = match &self.mapper {
            Some(m) => m,
            None => return,
        };

        let time = self.start_time.elapsed().as_secs_f32();
        let avatar = &self.cached_avatar;

        // Map blendshapes
        let morph_weights = if avatar.blendshapes().is_empty() {
            vec![0.0f32; mapper.num_targets()]
        } else {
            mapper.map_blendshapes(avatar.blendshapes())
        };

        // Head rotation for tracking (or None for idle animation)
        let head_rot = {
            let r = avatar.head_rotation();
            if r[0].abs() < 0.01 && r[1].abs() < 0.01 && r[2].abs() < 0.01 {
                None
            } else {
                Some(r)
            }
        };

        // Compute animated bone rotations
        let bone_rotations =
            animation::animated_pose(model, time, head_rot, avatar.is_speaking());

        // Forward kinematics
        let world = skinning::compute_world_transforms(model, &bone_rotations);

        // Skin each mesh
        let mut skinned_meshes = Vec::with_capacity(model.meshes.len());
        for (mesh_idx, _mesh) in model.meshes.iter().enumerate() {
            // Face mesh (index 1) gets morph targets applied first
            let base = if mesh_idx == 1 {
                skinning::apply_morph_targets(model, mesh_idx, &morph_weights)
            } else {
                skinning::base_positions(model, mesh_idx)
            };

            let skinned = skinning::skin_vertices(model, mesh_idx, &base, &world);
            skinned_meshes.push(skinned);
        }

        // Upload to GPU
        renderer.update_vertices(&render_state.queue, model, &skinned_meshes);
    }
}

impl eframe::App for RustuberApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Drain latest state from broadcast channel
        self.update_cached_state();

        // Update skinning if we have a renderer
        if let Some(render_state) = frame.wgpu_render_state() {
            self.update_skinning(render_state);
        }

        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.label("rustuber");
                ui.separator();
                ui.label("VTuber engine");
            });
        });

        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.heading("Controls");
            ui.separator();

            // OBS status
            let obs_connected = self
                .state
                .obs_connected
                .load(std::sync::atomic::Ordering::Relaxed);
            ui.horizontal(|ui| {
                ui.label("OBS:");
                if obs_connected {
                    ui.colored_label(egui::Color32::GREEN, "connected");
                } else {
                    ui.colored_label(egui::Color32::RED, "disconnected");
                }
            });

            ui.separator();

            // Avatar state info
            let avatar = &self.cached_avatar;
            ui.label(format!("State: {}", avatar.state_type()));
            ui.label(format!("Mouth: {:.2}", avatar.mouth_open()));
            ui.label(format!("Blink: {:.2}", avatar.blink()));
            ui.label(format!(
                "Head: [{:.1}, {:.1}, {:.1}]",
                avatar.head_rotation()[0],
                avatar.head_rotation()[1],
                avatar.head_rotation()[2],
            ));

            if !avatar.blendshapes().is_empty() {
                ui.label(format!("Blendshapes: {}", avatar.blendshapes().len()));
            }

            if let Some(ref err) = self.load_error {
                ui.separator();
                ui.colored_label(egui::Color32::RED, err);
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(renderer) = &self.renderer {
                let available_size = ui.available_size();
                let (rect, _response) =
                    ui.allocate_exact_size(available_size, egui::Sense::hover());

                let ppp = ctx.pixels_per_point();
                let vp_width = (available_size.x * ppp) as u32;
                let vp_height = (available_size.y * ppp) as u32;

                ui.painter().add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    VrmViewportCallback {
                        renderer: renderer.clone(),
                        viewport_width: vp_width.max(1),
                        viewport_height: vp_height.max(1),
                    },
                ));
            } else {
                ui.heading("Avatar Preview");
                if let Some(ref err) = self.load_error {
                    ui.colored_label(egui::Color32::RED, err);
                } else {
                    ui.label("Loading VRM model...");
                }
            }
        });

        // Repaint continuously for real-time updates
        ctx.request_repaint();
    }
}
