//! Main egui application with VRM 3D viewport and 2D PNGTuber mode.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use eframe::egui;

use crate::avatar::assets::AssetManager;
use crate::avatar::AvatarState;
use crate::config::VadProvider;
use crate::AppState;

use super::animation;
use super::blendshape_map::BlendshapeMapper;
use super::body_ik::BodyIkSetup;
use super::renderer::VrmRenderer;
use super::skinning;
use super::smoothing::{SmoothingMode, TrackingSmoother};
use super::spring_bone::SpringBoneSimulator;
use super::viewport::VrmViewportCallback;
use super::vrm_loader::VrmModel;

/// Display mode for the avatar viewport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewMode {
    /// 3D VRM model rendering
    Vrm3D,
    /// 2D PNGTuber sprite rendering
    PngTuber2D,
}

/// The native egui application window.
pub struct RustuberApp {
    state: Arc<AppState>,
    /// Broadcast receiver for avatar state updates (sync-safe via try_recv)
    state_rx: tokio::sync::broadcast::Receiver<AvatarState>,
    /// Cached latest avatar state (updated each frame via try_recv)
    cached_avatar: AvatarState,
    /// Current display mode (2D or 3D)
    view_mode: ViewMode,
    /// Asset manager for loading PNGTuber sprites
    asset_manager: Option<AssetManager>,
    /// Cached egui textures for 2D sprites (keyed by asset key)
    png_textures: HashMap<String, egui::TextureHandle>,
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
    /// Active catppuccin theme flavor
    theme: catppuccin_egui::Theme,
    /// Cached list of available audio input devices
    audio_devices: Vec<String>,
    /// Currently selected audio device name
    selected_device: String,
    /// Tracking smoother for head rotation and blendshapes
    smoother: TrackingSmoother,
    /// Body IK solver (precomputed from model)
    body_ik: Option<BodyIkSetup>,
    /// Tracking tuning parameters (editable via UI)
    tuning: crate::config::TrackingTuning,
    /// Last frame timestamp for computing dt
    last_frame: Instant,
    /// Currently selected VAD provider
    selected_vad: VadProvider,
    /// Camera distance for 3D viewport zoom (Z position)
    camera_distance: f32,
    /// Head-only mode: skip body IK, use procedural arms only
    head_only: bool,
    /// Manual head rotation override (bypass tracking)
    head_override: bool,
    /// Manual [pitch, yaw, roll] in degrees
    head_override_rot: [f32; 3],
    /// Mirror the viewport horizontally (selfie mode, default true)
    mirrored: bool,
    /// Spring bone physics simulator (hair/cloth)
    spring_sim: Option<SpringBoneSimulator>,
    /// Whether spring bone physics is enabled
    spring_bones_enabled: bool,
}

impl RustuberApp {
    pub fn new(cc: &eframe::CreationContext<'_>, state: Arc<AppState>) -> Self {
        let state_rx = state.subscribe_state();
        let start_time = Instant::now();

        // Load config for asset manager
        let config = {
            let rt = tokio::runtime::Handle::try_current();
            match rt {
                Ok(handle) => tokio::task::block_in_place(|| {
                    handle.block_on(state.config.read()).clone()
                }),
                Err(_) => crate::config::Config::default(),
            }
        };

        let asset_manager = match AssetManager::new(&config.avatar) {
            Ok(am) => {
                tracing::info!("Asset manager loaded: {:?}", am.keys().collect::<Vec<_>>());
                Some(am)
            }
            Err(e) => {
                tracing::warn!("Failed to load asset manager: {}", e);
                None
            }
        };

        let audio_devices = crate::audio::capture::list_input_devices();
        let selected_device = config.audio.device.clone();

        let tuning = config.avatar.tracking.clone();
        let mode = SmoothingMode::from_str(&tuning.smoothing_mode);
        let smoother = TrackingSmoother::new(mode, &tuning);

        let selected_vad = config.vad.provider;

        let mut app = Self {
            state,
            state_rx,
            cached_avatar: AvatarState::default(),
            view_mode: ViewMode::Vrm3D,
            asset_manager,
            png_textures: HashMap::new(),
            model: None,
            renderer: None,
            mapper: None,
            start_time,
            load_error: None,
            theme: catppuccin_egui::LATTE,
            audio_devices,
            selected_device,
            smoother,
            body_ik: None,
            tuning,
            last_frame: Instant::now(),
            selected_vad,
            camera_distance: 0.88,
            head_only: true,
            head_override: false,
            head_override_rot: [0.0; 3],
            mirrored: true,
            spring_sim: None,
            spring_bones_enabled: true,
        };

        // Try to load VRM model and initialize renderer
        app.init_vrm(cc);

        // If VRM failed to load, default to 2D mode
        if app.model.is_none() {
            app.view_mode = ViewMode::PngTuber2D;
        }

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

        self.body_ik = BodyIkSetup::from_model(&model);
        if self.body_ik.is_some() {
            tracing::info!("Body IK setup initialized for arm tracking");
        }

        self.spring_sim = SpringBoneSimulator::new(&model);

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

    /// Render the 2D PNGTuber sprite in the given UI region.
    fn show_pngtuber_panel(&mut self, ui: &mut egui::Ui) {
        let asset_manager = match &self.asset_manager {
            Some(am) => am,
            None => {
                ui.colored_label(self.theme.red, "No assets loaded");
                return;
            }
        };

        // Determine which sprite to show
        let key = self.cached_avatar.asset_key();

        // Fallback chain: exact key → strip "_speaking" suffix → "idle"
        let resolved_key = if asset_manager.has_asset(&key) {
            key.clone()
        } else if key.ends_with("_speaking") {
            let base = key.trim_end_matches("_speaking");
            if asset_manager.has_asset(base) {
                base.to_string()
            } else if asset_manager.has_asset("idle") {
                "idle".to_string()
            } else {
                key.clone()
            }
        } else if asset_manager.has_asset("idle") {
            "idle".to_string()
        } else {
            key.clone()
        };

        // Load texture on first use
        if !self.png_textures.contains_key(&resolved_key) {
            match asset_manager.get_data(&resolved_key) {
                Ok(bytes) => match image::load_from_memory(&bytes) {
                    Ok(img) => {
                        let rgba = img.to_rgba8();
                        let size = [rgba.width() as usize, rgba.height() as usize];
                        let pixels = rgba.into_raw();
                        let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
                        let texture = ui.ctx().load_texture(
                            &resolved_key,
                            color_image,
                            egui::TextureOptions::LINEAR,
                        );
                        self.png_textures.insert(resolved_key.clone(), texture);
                    }
                    Err(e) => {
                        ui.colored_label(
                            self.theme.red,
                            format!("Failed to decode image '{}': {}", resolved_key, e),
                        );
                        return;
                    }
                },
                Err(e) => {
                    ui.colored_label(
                        self.theme.red,
                        format!("Failed to load asset '{}': {}", resolved_key, e),
                    );
                    return;
                }
            }
        }

        if let Some(texture) = self.png_textures.get(&resolved_key) {
            let available = ui.available_size();
            ui.centered_and_justified(|ui| {
                ui.add(
                    egui::Image::new(texture)
                        .fit_to_exact_size(available)
                        .maintain_aspect_ratio(true),
                );
            });
        }
    }

    /// Update skinning for the current frame.
    fn update_skinning(&mut self, render_state: &eframe::egui_wgpu::RenderState) {
        // Clone Arcs so we can borrow &mut self.smoother later
        let model = match &self.model {
            Some(m) => m.clone(),
            None => return,
        };
        let renderer = match &self.renderer {
            Some(r) => r.clone(),
            None => return,
        };

        renderer.set_camera_distance(self.camera_distance);
        renderer.set_mirrored(self.mirrored);
        let mapper = match &self.mapper {
            Some(m) => m,
            None => return,
        };

        // Compute frame dt
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;

        let time = self.start_time.elapsed().as_secs_f32();
        let avatar = &self.cached_avatar;

        // Head rotation: use manual override or tracked data
        let head_rot = if self.head_override {
            Some(self.head_override_rot)
        } else {
            let r = avatar.head_rotation();
            if r[0].abs() < 0.01 && r[1].abs() < 0.01 && r[2].abs() < 0.01 {
                None
            } else {
                let smoothed = self.smoother.smooth_head(r, dt, &self.tuning);
                Some(smoothed)
            }
        };

        // Head yaw in radians for eye occlusion (yaw is index 1, in degrees)
        let head_yaw_rad = head_rot
            .map(|[_, yaw, _]| yaw.to_radians())
            .unwrap_or(0.0);

        // Smooth blendshapes, then map (with head yaw for eye occlusion)
        let morph_weights = if avatar.blendshapes().is_empty() {
            vec![0.0f32; mapper.num_targets()]
        } else {
            let smoothed_bs =
                self.smoother
                    .smooth_blendshapes(avatar.blendshapes(), dt, &self.tuning);
            mapper.map_blendshapes(&smoothed_bs, self.tuning.blendshape_sensitivity, head_yaw_rad)
        };

        // Solve body IK if tracking data is available and not in head-only mode
        let body_ik_rotations = if avatar.has_body_tracking() && !self.head_only {
            let smoothed = self
                .smoother
                .smooth_body_landmarks(avatar.body_landmarks(), dt, &self.tuning);
            let rest_world = skinning::compute_world_transforms(&model, &HashMap::new());
            self.body_ik
                .as_ref()
                .map(|ik| ik.solve(&model, &smoothed, &rest_world))
        } else {
            None
        };

        // Compute animated bone rotations
        let bone_rotations = animation::animated_pose(
            &model,
            time,
            head_rot,
            avatar.is_speaking(),
            &self.tuning,
            body_ik_rotations.as_ref(),
        );

        // Forward kinematics
        let world = skinning::compute_world_transforms(&model, &bone_rotations);

        // Spring bone physics (hair/cloth secondary motion)
        let world = if self.spring_bones_enabled {
            if let Some(sim) = &mut self.spring_sim {
                let spring_rots = sim.step(&model, &world, dt);
                if !spring_rots.is_empty() {
                    let mut final_rots = bone_rotations.clone();
                    final_rots.extend(spring_rots);
                    skinning::compute_world_transforms(&model, &final_rots)
                } else {
                    world
                }
            } else {
                world
            }
        } else {
            world
        };

        // Skin each mesh
        let mut skinned_meshes = Vec::with_capacity(model.meshes.len());
        for (mesh_idx, _mesh) in model.meshes.iter().enumerate() {
            // Face mesh (index 1) gets morph targets applied first
            let base = if mesh_idx == 1 {
                skinning::apply_morph_targets(&model, mesh_idx, &morph_weights)
            } else {
                skinning::base_positions(&model, mesh_idx)
            };

            let skinned = skinning::skin_vertices(&model, mesh_idx, &base, &world);
            skinned_meshes.push(skinned);
        }

        // Upload to GPU
        renderer.update_vertices(&render_state.queue, &model, &skinned_meshes);
    }
}

impl eframe::App for RustuberApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Apply catppuccin theme
        catppuccin_egui::set_theme(ctx, self.theme);
        let theme = self.theme; // Copy for use inside closures

        // Drain latest state from broadcast channel
        self.update_cached_state();

        // Only update skinning in 3D mode
        if self.view_mode == ViewMode::Vrm3D {
            if let Some(render_state) = frame.wgpu_render_state() {
                self.update_skinning(render_state);
            }
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

            // View mode toggle
            ui.horizontal(|ui| {
                ui.label("Mode:");
                ui.selectable_value(&mut self.view_mode, ViewMode::Vrm3D, "3D (VRM)");
                ui.selectable_value(&mut self.view_mode, ViewMode::PngTuber2D, "2D (PNG)");
            });

            if self.view_mode == ViewMode::Vrm3D {
                ui.horizontal(|ui| {
                    ui.label("Zoom:");
                    if ui.button("-").clicked() {
                        self.camera_distance = (self.camera_distance + 0.05).min(3.0);
                    }
                    ui.label(format!("{:.2}", self.camera_distance));
                    if ui.button("+").clicked() {
                        self.camera_distance = (self.camera_distance - 0.05).max(0.3);
                    }
                    ui.separator();
                    ui.checkbox(&mut self.mirrored, "Mirror");
                });
            }

            ui.separator();

            // Theme flavor picker
            ui.horizontal(|ui| {
                ui.label("Theme:");
                ui.selectable_value(&mut self.theme, catppuccin_egui::LATTE, "Latte");
                ui.selectable_value(&mut self.theme, catppuccin_egui::FRAPPE, "Frappe");
                ui.selectable_value(&mut self.theme, catppuccin_egui::MACCHIATO, "Macchiato");
                ui.selectable_value(&mut self.theme, catppuccin_egui::MOCHA, "Mocha");
            });

            ui.separator();

            // Audio input device picker
            ui.label("Audio Input");
            let prev_device = self.selected_device.clone();
            egui::ComboBox::from_id_salt("audio_device")
                .selected_text(&self.selected_device)
                .show_ui(ui, |ui| {
                    for name in &self.audio_devices {
                        ui.selectable_value(&mut self.selected_device, name.clone(), name);
                    }
                });
            if self.selected_device != prev_device {
                let new_device = self.selected_device.clone();
                let state = self.state.clone();
                let rt = tokio::runtime::Handle::current();
                rt.spawn(async move {
                    let mut config = state.config.write().await;
                    config.audio.device = new_device;
                    drop(config);
                    state.signal_audio_restart();
                });
            }

            ui.separator();

            // VAD provider toggle
            ui.label("VAD Engine");
            let prev_vad = self.selected_vad;
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.selected_vad, VadProvider::Silero, "Silero");
                ui.selectable_value(&mut self.selected_vad, VadProvider::Energy, "Energy");
            });
            if self.selected_vad != prev_vad {
                let new_provider = self.selected_vad;
                let state = self.state.clone();
                let rt = tokio::runtime::Handle::current();
                rt.spawn(async move {
                    let mut config = state.config.write().await;
                    config.vad.provider = new_provider;
                    drop(config);
                    state.signal_audio_restart();
                });
            }

            ui.separator();

            // OBS status
            let obs_connected = self
                .state
                .obs_connected
                .load(std::sync::atomic::Ordering::Relaxed);
            ui.horizontal(|ui| {
                ui.label("OBS:");
                if obs_connected {
                    ui.colored_label(theme.green, "connected");
                } else {
                    ui.colored_label(theme.red, "disconnected");
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
            if avatar.has_body_tracking() {
                ui.label(format!("Body landmarks: {}", avatar.body_landmarks().len()));
            }

            ui.separator();

            // Audio meter
            let energy_db = self.state.get_audio_energy_db();
            let vad_conf = self.state.get_audio_vad_confidence();
            ui.label("Audio");
            ui.label(format!("Energy: {:.1} dB", energy_db));
            // Map dB to 0.0–1.0 for the bar: -100 dB → 0.0, 0 dB → 1.0
            let level_frac = ((energy_db + 100.0) / 100.0).clamp(0.0, 1.0);
            let bar_color = if avatar.is_speaking() {
                theme.green
            } else {
                theme.blue
            };
            let bar_width = ui.available_width();
            let (bar_rect, _) = ui.allocate_exact_size(
                egui::vec2(bar_width, 14.0),
                egui::Sense::hover(),
            );
            ui.painter().rect_filled(
                bar_rect,
                2.0,
                theme.surface0,
            );
            let filled = egui::Rect::from_min_size(
                bar_rect.min,
                egui::vec2(bar_rect.width() * level_frac, bar_rect.height()),
            );
            ui.painter().rect_filled(filled, 2.0, bar_color);
            ui.label(format!("VAD: {:.0}%", vad_conf * 100.0));
            ui.horizontal(|ui| {
                ui.label("Speaking:");
                if avatar.is_speaking() {
                    ui.colored_label(theme.green, "yes");
                } else {
                    ui.label("no");
                }
            });

            ui.separator();

            // Tracking tuning controls
            ui.heading("Tracking");

            // Smoothing mode selector
            let current_mode = self.smoother.mode();
            let mut selected_mode = current_mode;
            ui.horizontal(|ui| {
                ui.label("Smoothing:");
                for mode in &SmoothingMode::ALL {
                    ui.selectable_value(&mut selected_mode, *mode, mode.as_str());
                }
            });
            if selected_mode != current_mode {
                self.smoother.set_mode(selected_mode, &self.tuning);
            }

            ui.add(
                egui::Slider::new(&mut self.tuning.head_sensitivity, 0.5..=3.0)
                    .text("Head sensitivity"),
            );
            ui.add(
                egui::Slider::new(&mut self.tuning.blendshape_sensitivity, 0.5..=2.5)
                    .text("Expression sensitivity"),
            );

            // Mode-specific controls
            match self.smoother.mode() {
                SmoothingMode::Spring => {
                    ui.add(
                        egui::Slider::new(&mut self.tuning.head_halflife, 0.01..=0.5)
                            .text("Head halflife"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.tuning.blendshape_halflife, 0.01..=0.5)
                            .text("BS halflife"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.tuning.blink_halflife, 0.01..=0.2)
                            .text("Blink halflife"),
                    );
                }
                SmoothingMode::OneEuro => {
                    ui.add(
                        egui::Slider::new(&mut self.tuning.head_min_cutoff, 0.1..=10.0)
                            .text("Head min cutoff"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.tuning.head_beta, 0.0..=1.0)
                            .text("Head beta"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.tuning.blendshape_min_cutoff, 0.001..=1.0)
                            .logarithmic(true)
                            .text("BS min cutoff"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.tuning.blendshape_beta, 0.0..=30.0)
                            .text("BS beta"),
                    );
                }
                SmoothingMode::None => {}
            }

            ui.separator();
            ui.heading("Head Override");
            ui.checkbox(&mut self.head_override, "Manual head pose");
            if self.head_override {
                ui.add(
                    egui::Slider::new(&mut self.head_override_rot[0], -45.0..=45.0)
                        .text("Pitch"),
                );
                ui.add(
                    egui::Slider::new(&mut self.head_override_rot[1], -45.0..=45.0)
                        .text("Yaw"),
                );
                ui.add(
                    egui::Slider::new(&mut self.head_override_rot[2], -30.0..=30.0)
                        .text("Roll"),
                );
                if ui.button("Reset to zero").clicked() {
                    self.head_override_rot = [0.0; 3];
                }
            }

            ui.separator();
            ui.heading("Physics");
            ui.checkbox(&mut self.spring_bones_enabled, "Spring bones");

            ui.separator();
            ui.heading("Body Tracking");
            ui.checkbox(&mut self.head_only, "Head only (no arms)");
            ui.add_enabled(
                !self.head_only,
                egui::Slider::new(&mut self.tuning.body_blend_factor, 0.0..=1.0)
                    .text("Body blend"),
            );
            match self.smoother.mode() {
                SmoothingMode::Spring => {
                    ui.add(
                        egui::Slider::new(&mut self.tuning.body_halflife, 0.01..=0.5)
                            .text("Body halflife"),
                    );
                }
                _ => {}
            }

            // In 2D mode, show the current asset key
            if self.view_mode == ViewMode::PngTuber2D {
                ui.separator();
                ui.label(format!("Sprite: {}", self.cached_avatar.asset_key()));
            }

            if let Some(ref err) = self.load_error {
                ui.separator();
                ui.colored_label(theme.red, err);
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.view_mode {
                ViewMode::Vrm3D => {
                    if let Some(renderer) = &self.renderer {
                        let available_size = ui.available_size();
                        let (rect, response) =
                            ui.allocate_exact_size(available_size, egui::Sense::click_and_drag());

                        if response.hovered() {
                            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                            if scroll.abs() > 0.0 {
                                self.camera_distance =
                                    (self.camera_distance - scroll * 0.002).clamp(0.3, 3.0);
                            }
                        }

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
                            ui.colored_label(theme.red, err);
                        } else {
                            ui.label("Loading VRM model...");
                        }
                    }
                }
                ViewMode::PngTuber2D => {
                    self.show_pngtuber_panel(ui);
                }
            }
        });

        // Repaint continuously for real-time updates
        ctx.request_repaint();
    }
}
