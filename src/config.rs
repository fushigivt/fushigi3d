//! Configuration parsing and management for Fushigi3D

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::{ConfigError, Fushigi3dError};

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub audio: AudioConfig,
    pub vad: VadConfig,
    pub avatar: AvatarConfig,
    pub obs: ObsConfig,
    pub http: HttpConfig,
    pub vmc: VmcConfig,
    pub osf: OsfConfig,
    pub mediapipe: MediaPipeConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            audio: AudioConfig::default(),
            vad: VadConfig::default(),
            avatar: AvatarConfig::default(),
            obs: ObsConfig::default(),
            http: HttpConfig::default(),
            vmc: VmcConfig::default(),
            osf: OsfConfig::default(),
            mediapipe: MediaPipeConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Fushigi3dError> {
        let contents = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            ConfigError::ReadFile(format!("{}: {}", path.as_ref().display(), e))
        })?;

        Self::from_str(&contents)
    }

    /// Parse configuration from a TOML string
    pub fn from_str(s: &str) -> Result<Self, Fushigi3dError> {
        toml::from_str(s).map_err(|e| ConfigError::Parse(e.to_string()).into())
    }

    /// Load configuration from default paths
    pub fn load() -> Result<Self, Fushigi3dError> {
        // Try config paths in order
        let paths = [
            PathBuf::from("config.toml"),
            PathBuf::from("config/default.toml"),
            dirs_path().join("config.toml"),
        ];

        for path in &paths {
            if path.exists() {
                tracing::info!("Loading config from: {}", path.display());
                return Self::from_file(path);
            }
        }

        tracing::info!("No config file found, using defaults");
        Ok(Self::default())
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), Fushigi3dError> {
        // Validate audio settings
        if self.audio.sample_rate == 0 {
            return Err(ConfigError::InvalidValue {
                field: "audio.sample_rate".to_string(),
                message: "Sample rate must be greater than 0".to_string(),
            }
            .into());
        }

        // Validate VAD settings
        if !(0.0..=1.0).contains(&self.vad.silero_threshold) {
            return Err(ConfigError::InvalidValue {
                field: "vad.silero_threshold".to_string(),
                message: "Threshold must be between 0.0 and 1.0".to_string(),
            }
            .into());
        }

        // Validate OSF settings
        if !(-3..=4).contains(&self.osf.model_quality) {
            return Err(ConfigError::InvalidValue {
                field: "osf.model_quality".to_string(),
                message: "Model quality must be between -3 and 4".to_string(),
            }
            .into());
        }

        if self.osf.auto_launch {
            let path = std::path::Path::new(&self.osf.facetracker_path);
            if !path.exists() {
                tracing::warn!(
                    "OSF auto_launch enabled but facetracker script not found at: {}",
                    self.osf.facetracker_path
                );
            }
        }

        if self.mediapipe.auto_launch {
            let path = std::path::Path::new(&self.mediapipe.tracker_script);
            if !path.exists() {
                tracing::warn!(
                    "MediaPipe auto_launch enabled but tracker script not found at: {}",
                    self.mediapipe.tracker_script
                );
            }
        }

        // Validate HTTP settings
        if self.http.port == 0 {
            return Err(ConfigError::InvalidValue {
                field: "http.port".to_string(),
                message: "Port must be greater than 0".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

/// Audio input configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    /// Enable audio capture
    pub enabled: bool,
    /// Audio device name or "default"
    pub device: String,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels (typically 1 for VAD)
    pub channels: u16,
    /// Buffer size in samples
    pub buffer_size: u32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device: "default".to_string(),
            sample_rate: 16000,
            channels: 1,
            buffer_size: 512,
        }
    }
}

/// Voice Activity Detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VadConfig {
    /// VAD provider: "silero", "webrtc", "energy", or "remote"
    pub provider: VadProvider,
    /// Silero VAD threshold (0.0 - 1.0)
    pub silero_threshold: f32,
    /// WebRTC VAD aggressiveness (0-3)
    pub webrtc_mode: u8,
    /// Energy-based VAD threshold in dB
    pub energy_threshold_db: f32,
    /// Attack time in milliseconds (voice starts)
    pub attack_ms: u32,
    /// Release time in milliseconds (voice ends)
    pub release_ms: u32,
    /// Minimum speech duration in ms to trigger speaking state
    pub min_speech_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            provider: VadProvider::Silero,
            silero_threshold: 0.3,
            webrtc_mode: 2,
            energy_threshold_db: -40.0,
            attack_ms: 50,
            release_ms: 200,
            min_speech_ms: 100,
        }
    }
}

/// VAD provider selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VadProvider {
    /// Silero VAD (neural network based)
    Silero,
    /// WebRTC VAD
    WebRtc,
    /// Simple energy-based detection
    Energy,
}

impl Default for VadProvider {
    fn default() -> Self {
        Self::Silero
    }
}

/// Avatar configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AvatarConfig {
    /// Default state on startup
    pub default_state: String,
    /// Transition duration in milliseconds
    pub transition_ms: u32,
    /// Directory containing avatar assets
    pub assets_dir: PathBuf,
    /// State to image mapping
    pub states: HashMap<String, String>,
    /// Expression to image mapping
    pub expressions: HashMap<String, String>,
    /// VRM 3D model configuration
    pub vrm: VrmConfig,
    /// Tracking smoothing and sensitivity tuning
    pub tracking: TrackingTuning,
}

/// Tracking smoothing and sensitivity tuning parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrackingTuning {
    /// Smoothing algorithm: "spring", "one_euro", or "none"
    #[serde(default = "default_smoothing_mode")]
    pub smoothing_mode: String,

    // --- Head sensitivity ---
    /// Overall head rotation multiplier
    #[serde(default = "default_1_4")]
    pub head_sensitivity: f32,
    /// Per-axis scaling
    #[serde(default = "default_1_0")]
    pub pitch_scale: f32,
    #[serde(default = "default_1_0")]
    pub yaw_scale: f32,
    #[serde(default = "default_1_0")]
    pub roll_scale: f32,

    // --- Blendshape sensitivity ---
    #[serde(default = "default_1_2")]
    pub blendshape_sensitivity: f32,

    // --- Spring halflife (seconds) ---
    #[serde(default = "default_0_08")]
    pub head_halflife: f32,
    #[serde(default = "default_0_12")]
    pub blendshape_halflife: f32,
    #[serde(default = "default_0_04")]
    pub blink_halflife: f32,

    // --- 1-Euro filter params ---
    #[serde(default = "default_1_5")]
    pub head_min_cutoff: f32,
    #[serde(default = "default_0_05")]
    pub head_beta: f32,
    #[serde(default = "default_0_005")]
    pub blendshape_min_cutoff: f32,
    #[serde(default = "default_15_0")]
    pub blendshape_beta: f32,
    #[serde(default = "default_0_8")]
    pub blink_min_cutoff: f32,
    #[serde(default = "default_10_0")]
    pub blink_beta: f32,

    // --- Body tracking ---
    /// Spring halflife for body landmarks (seconds)
    #[serde(default = "default_0_15")]
    pub body_halflife: f32,
    /// Blend factor: 0.0 = pure procedural, 1.0 = pure tracked
    #[serde(default = "default_0_85")]
    pub body_blend_factor: f32,

    // --- Deadzones ---
    #[serde(default = "default_0_5")]
    pub head_deadzone: f32,
    #[serde(default = "default_0_02")]
    pub blendshape_deadzone: f32,

    // --- Expression transitions ---
    #[serde(default = "default_0_4")]
    pub expression_fade_duration: f32,
    #[serde(default = "default_easing")]
    pub expression_easing: String,
}

fn default_smoothing_mode() -> String { "spring".to_string() }
fn default_easing() -> String { "quad_in_out".to_string() }
fn default_1_4() -> f32 { 1.4 }
fn default_1_2() -> f32 { 1.2 }
fn default_1_0() -> f32 { 1.0 }
fn default_0_08() -> f32 { 0.08 }
fn default_0_12() -> f32 { 0.12 }
fn default_0_04() -> f32 { 0.04 }
fn default_1_5() -> f32 { 1.5 }
fn default_0_05() -> f32 { 0.05 }
fn default_0_005() -> f32 { 0.005 }
fn default_15_0() -> f32 { 15.0 }
fn default_0_8() -> f32 { 0.8 }
fn default_10_0() -> f32 { 10.0 }
fn default_0_15() -> f32 { 0.15 }
fn default_0_85() -> f32 { 0.85 }
fn default_0_5() -> f32 { 0.5 }
fn default_0_02() -> f32 { 0.02 }
fn default_0_4() -> f32 { 0.4 }

impl Default for TrackingTuning {
    fn default() -> Self {
        Self {
            smoothing_mode: default_smoothing_mode(),
            head_sensitivity: default_1_4(),
            pitch_scale: default_1_0(),
            yaw_scale: default_1_0(),
            roll_scale: default_1_0(),
            blendshape_sensitivity: default_1_2(),
            head_halflife: default_0_08(),
            blendshape_halflife: default_0_12(),
            blink_halflife: default_0_04(),
            head_min_cutoff: default_1_5(),
            head_beta: default_0_05(),
            blendshape_min_cutoff: default_0_005(),
            blendshape_beta: default_15_0(),
            blink_min_cutoff: default_0_8(),
            blink_beta: default_10_0(),
            body_halflife: default_0_15(),
            body_blend_factor: default_0_85(),
            head_deadzone: default_0_5(),
            blendshape_deadzone: default_0_02(),
            expression_fade_duration: default_0_4(),
            expression_easing: default_easing(),
        }
    }
}

/// VRM 3D model rendering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VrmConfig {
    /// Path to the VRM/GLB model file
    pub model_path: String,
}

impl Default for VrmConfig {
    fn default() -> Self {
        Self {
            model_path: "assets/default/models/long-hair.vrm".to_string(),
        }
    }
}

impl Default for AvatarConfig {
    fn default() -> Self {
        let mut states = HashMap::new();
        states.insert("idle".to_string(), "idle.png".to_string());
        states.insert("speaking".to_string(), "speaking.png".to_string());

        let mut expressions = HashMap::new();
        expressions.insert("happy".to_string(), "expressions/happy.png".to_string());
        expressions.insert(
            "surprised".to_string(),
            "expressions/surprised.png".to_string(),
        );

        Self {
            default_state: "idle".to_string(),
            transition_ms: 100,
            assets_dir: PathBuf::from("./assets/default"),
            states,
            expressions,
            vrm: VrmConfig::default(),
            tracking: TrackingTuning::default(),
        }
    }
}

/// OBS WebSocket configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ObsConfig {
    /// Enable OBS integration
    pub enabled: bool,
    /// OBS WebSocket host
    pub host: String,
    /// OBS WebSocket port
    pub port: u16,
    /// OBS WebSocket password (optional)
    pub password: Option<String>,
    /// Output mode: "scene" or "source"
    pub mode: ObsMode,
    /// Scene name for idle state (scene mode)
    pub idle_scene: Option<String>,
    /// Scene name for speaking state (scene mode)
    pub speaking_scene: Option<String>,
    /// Scene to control (source mode)
    pub scene: Option<String>,
    /// Source name for idle state (source mode)
    pub idle_source: Option<String>,
    /// Source name for speaking state (source mode)
    pub speaking_source: Option<String>,
    /// Reconnect delay in seconds
    pub reconnect_delay_secs: u64,
}

impl Default for ObsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            host: "127.0.0.1".to_string(),
            port: 4455,
            password: None,
            mode: ObsMode::Scene,
            idle_scene: Some("Idle".to_string()),
            speaking_scene: Some("Speaking".to_string()),
            scene: None,
            idle_source: None,
            speaking_source: None,
            reconnect_delay_secs: 5,
        }
    }
}

/// OBS output mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ObsMode {
    /// Switch between scenes
    Scene,
    /// Toggle source visibility within a scene
    Source,
}

impl Default for ObsMode {
    fn default() -> Self {
        Self::Scene
    }
}

/// HTTP server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HttpConfig {
    /// Enable HTTP server
    pub enabled: bool,
    /// HTTP server host
    pub host: String,
    /// HTTP server port
    pub port: u16,
    /// Enable CORS
    pub cors_enabled: bool,
    /// Allowed origins for CORS
    pub cors_origins: Vec<String>,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            host: "127.0.0.1".to_string(),
            port: 8080,
            cors_enabled: true,
            cors_origins: vec!["*".to_string()],
        }
    }
}

/// VMC protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VmcConfig {
    /// Enable VMC receiver
    pub receiver_enabled: bool,
    /// VMC receiver port
    pub receiver_port: u16,
    /// Blend VMC data with VAD state
    pub blend_with_vad: bool,
    /// Expression mappings from blendshapes
    pub expressions: HashMap<String, VmcExpressionMapping>,
}

impl Default for VmcConfig {
    fn default() -> Self {
        Self {
            receiver_enabled: false,
            receiver_port: 39539,
            blend_with_vad: true,
            expressions: HashMap::new(),
        }
    }
}

/// VMC blendshape to expression mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmcExpressionMapping {
    /// Blendshape name to monitor
    pub blendshape: String,
    /// Threshold to trigger expression
    pub threshold: f32,
}

/// OpenSeeFace native protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OsfConfig {
    /// Enable OpenSeeFace tracking
    pub enabled: bool,
    /// UDP port to receive OSF data on
    pub port: u16,
    /// Listen address for UDP socket
    pub listen_address: String,
    /// Auto-launch OpenSeeFace subprocess
    pub auto_launch: bool,
    /// Path to facetracker.py script
    pub facetracker_path: String,
    /// Camera device index
    pub camera_device: u32,
    /// Model quality (-3 to 4, higher = better but slower)
    pub model_quality: i8,
    /// Maximum number of faces to track
    pub max_faces: u32,
    /// Camera capture width
    pub capture_width: u32,
    /// Camera capture height
    pub capture_height: u32,
    /// Camera capture FPS
    pub capture_fps: u32,
    /// Face ID to use from multi-face tracking
    pub face_id: i32,
    /// Blend OSF tracking data with VAD state
    pub blend_with_vad: bool,
    /// Auto-restart subprocess on crash
    pub auto_restart: bool,
    /// Delay before restarting crashed subprocess (seconds)
    pub restart_delay_secs: u64,
}

impl Default for OsfConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            port: 11573,
            listen_address: "127.0.0.1".to_string(),
            auto_launch: false,
            facetracker_path: "OpenSeeFace/facetracker.py".to_string(),
            camera_device: 0,
            model_quality: 3,
            max_faces: 1,
            capture_width: 640,
            capture_height: 480,
            capture_fps: 24,
            face_id: 0,
            blend_with_vad: true,
            auto_restart: true,
            restart_delay_secs: 3,
        }
    }
}

/// MediaPipe face tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MediaPipeConfig {
    /// Enable MediaPipe tracking
    pub enabled: bool,
    /// UDP port to receive MediaPipe data on
    pub port: u16,
    /// Listen address for UDP socket
    pub listen_address: String,
    /// Auto-launch the Python tracker subprocess
    pub auto_launch: bool,
    /// Path to mp_tracker.py script
    pub tracker_script: String,
    /// Camera device index
    pub camera_device: u32,
    /// Camera capture width
    pub capture_width: u32,
    /// Camera capture height
    pub capture_height: u32,
    /// Camera capture FPS
    pub capture_fps: u32,
    /// Directory to store/cache the MediaPipe model file
    pub model_dir: String,
    /// Blend MediaPipe tracking data with VAD state
    pub blend_with_vad: bool,
    /// Enable body pose tracking alongside face tracking
    pub enable_body_tracking: bool,
    /// Auto-restart subprocess on crash
    pub auto_restart: bool,
    /// Delay before restarting crashed subprocess (seconds)
    pub restart_delay_secs: u64,
}

impl Default for MediaPipeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 12346,
            listen_address: "127.0.0.1".to_string(),
            auto_launch: true,
            tracker_script: "scripts/mp_tracker.py".to_string(),
            camera_device: 0,
            capture_width: 640,
            capture_height: 480,
            capture_fps: 30,
            model_dir: ".".to_string(),
            blend_with_vad: true,
            enable_body_tracking: true,
            auto_restart: true,
            restart_delay_secs: 3,
        }
    }
}

/// Integration configuration
/// Get the platform-specific configuration directory
fn dirs_path() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        if let Some(config_dir) = std::env::var_os("XDG_CONFIG_HOME") {
            return PathBuf::from(config_dir).join("fushigi3d");
        }
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(".config/fushigi3d");
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join("Library/Application Support/fushigi3d");
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Some(appdata) = std::env::var_os("APPDATA") {
            return PathBuf::from(appdata).join("fushigi3d");
        }
    }

    PathBuf::from(".")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.audio.device, "default");
        assert_eq!(config.audio.sample_rate, 16000);
        assert!(!config.obs.enabled);
        assert!(config.http.enabled);
    }

    #[test]
    fn test_config_validation() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_parse_toml() {
        let toml = r#"
            [audio]
            device = "hw:1,0"
            sample_rate = 48000

            [vad]
            silero_threshold = 0.7
        "#;

        let config = Config::from_str(toml).unwrap();
        assert_eq!(config.audio.device, "hw:1,0");
        assert_eq!(config.audio.sample_rate, 48000);
        assert_eq!(config.vad.silero_threshold, 0.7);
    }
}
