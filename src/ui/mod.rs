//! Native egui UI for fushigi3d.
//!
//! Provides a desktop window with:
//! - Avatar preview viewport (PNGTuber sprites or future VRM 3D)
//! - Audio level meter + VAD indicator
//! - OBS connection status and controls
//! - Configuration panel
//!
//! Enabled via `--features native-ui`.

mod animation;
mod app;
mod blendshape_map;
mod body_ik;
mod renderer;
mod smoothing;
mod skinning;
mod spring_bone;
mod viewport;
mod vrm_loader;

pub use app::Fushigi3dApp;
