//! Fushigi3D - Headless VTuber/PNGTuber Service
//!
//! A modular Rust service for PNGtuber/VTuber functionality that:
//! - Works fully standalone as a complete PNGTuber application
//! - Outputs to OBS WebSocket and Browser Source (HTTP/SSE)
//! - Supports VMC, OSF, and MediaPipe for external face tracking
//! - Real-time 3D VRM avatar rendering with spring bone physics

pub mod audio;
pub mod avatar;
pub mod config;
pub mod error;
pub mod output;
pub mod tracking;
pub mod web;

#[cfg(feature = "native-ui")]
pub mod ui;

pub use config::Config;
pub use error::{Result, Fushigi3dError};

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock, Notify};

use avatar::AvatarState;

/// Application state shared across all components
#[derive(Debug)]
pub struct AppState {
    /// Current configuration
    pub config: RwLock<Config>,
    /// Current avatar state
    pub avatar_state: RwLock<AvatarState>,
    /// Channel for avatar state updates
    pub state_tx: broadcast::Sender<AvatarState>,
    /// Shutdown signal
    pub shutdown_tx: broadcast::Sender<()>,
    /// OBS connection status
    pub obs_connected: AtomicBool,
    /// OBS scenes list (cached)
    pub obs_scenes: RwLock<Vec<String>>,
    /// OBS reconnect signal
    pub obs_reconnect: Notify,
    /// Config changed signal
    pub config_changed: Notify,
    /// Audio energy level in dB (stored as f32 bits in AtomicU32)
    pub audio_energy_db: AtomicU32,
    /// VAD confidence (0.0–1.0, stored as f32 bits in AtomicU32)
    pub audio_vad_confidence: AtomicU32,
}

impl AppState {
    /// Create a new application state with the given configuration
    pub fn new(config: Config) -> Arc<Self> {
        let (state_tx, _) = broadcast::channel(64);
        let (shutdown_tx, _) = broadcast::channel(1);

        let default_state = config.avatar.default_state.clone();

        Arc::new(Self {
            config: RwLock::new(config),
            avatar_state: RwLock::new(AvatarState::new(&default_state)),
            state_tx,
            shutdown_tx,
            obs_connected: AtomicBool::new(false),
            obs_scenes: RwLock::new(Vec::new()),
            obs_reconnect: Notify::new(),
            config_changed: Notify::new(),
            audio_energy_db: AtomicU32::new(f32::to_bits(-100.0)),
            audio_vad_confidence: AtomicU32::new(f32::to_bits(0.0)),
        })
    }

    /// Update the avatar state and broadcast the change
    pub async fn update_avatar_state(&self, state: AvatarState) {
        let mut current = self.avatar_state.write().await;
        *current = state.clone();
        let _ = self.state_tx.send(state);
    }

    /// Get the current avatar state
    pub async fn get_avatar_state(&self) -> AvatarState {
        self.avatar_state.read().await.clone()
    }

    /// Subscribe to avatar state changes
    pub fn subscribe_state(&self) -> broadcast::Receiver<AvatarState> {
        self.state_tx.subscribe()
    }

    /// Subscribe to shutdown signal
    pub fn subscribe_shutdown(&self) -> broadcast::Receiver<()> {
        self.shutdown_tx.subscribe()
    }

    /// Signal shutdown
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }

    /// Set OBS connection status
    pub fn set_obs_connected(&self, connected: bool) {
        self.obs_connected.store(connected, Ordering::Relaxed);
    }

    /// Update cached OBS scenes
    pub async fn set_obs_scenes(&self, scenes: Vec<String>) {
        let mut current = self.obs_scenes.write().await;
        *current = scenes;
    }

    /// Signal OBS to reconnect
    pub fn signal_obs_reconnect(&self) {
        self.obs_reconnect.notify_one();
    }

    /// Wait for OBS reconnect signal
    pub async fn wait_obs_reconnect(&self) {
        self.obs_reconnect.notified().await;
    }

    /// Signal that config has changed
    pub fn signal_config_changed(&self) {
        self.config_changed.notify_waiters();
    }

    /// Wait for config change signal
    pub async fn wait_config_changed(&self) {
        self.config_changed.notified().await;
    }

    /// Update audio level metrics from the audio pipeline
    pub fn set_audio_levels(&self, energy_db: f32, confidence: f32) {
        self.audio_energy_db.store(f32::to_bits(energy_db), Ordering::Relaxed);
        self.audio_vad_confidence.store(f32::to_bits(confidence), Ordering::Relaxed);
    }

    /// Get the current audio energy level in dB
    pub fn get_audio_energy_db(&self) -> f32 {
        f32::from_bits(self.audio_energy_db.load(Ordering::Relaxed))
    }

    /// Get the current VAD confidence (0.0–1.0)
    pub fn get_audio_vad_confidence(&self) -> f32 {
        f32::from_bits(self.audio_vad_confidence.load(Ordering::Relaxed))
    }
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");
