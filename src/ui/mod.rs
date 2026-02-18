//! Native egui UI for rustuber.
//!
//! Provides a desktop window with:
//! - Avatar preview viewport (PNGTuber sprites or future VRM 3D)
//! - Audio level meter + VAD indicator
//! - OBS connection status and controls
//! - Configuration panel
//!
//! Enabled via `--features native-ui`.

mod app;

pub use app::RustuberApp;
