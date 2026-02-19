//! Main egui application with VRM 3D viewport and 2D PNGTuber mode.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use eframe::egui;

use crate::avatar::assets::AssetManager;
use crate::avatar::AvatarState;
use crate::AppState;

use super::animation;
use super::blendshape_map::BlendshapeMapper;
use super::renderer::VrmRenderer;
use super::skinning;
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
