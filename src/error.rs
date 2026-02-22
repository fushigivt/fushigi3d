//! Error types for Fushigi3D

use thiserror::Error;

/// Main error type for Fushigi3D
#[derive(Error, Debug)]
pub enum Fushigi3dError {
    #[error("Audio error: {0}")]
    Audio(#[from] AudioError),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Avatar error: {0}")]
    Avatar(#[from] AvatarError),

    #[error("Output error: {0}")]
    Output(#[from] OutputError),

    #[error("Web server error: {0}")]
    Web(#[from] WebError),

    #[error("Integration error: {0}")]
    Integration(#[from] IntegrationError),

    #[error("Tracking error: {0}")]
    Tracking(#[from] TrackingError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Audio-related errors
#[derive(Error, Debug)]
pub enum AudioError {
    #[error("No audio device found")]
    NoDeviceFound,

    #[error("Failed to enumerate audio devices: {0}")]
    DeviceEnumeration(String),

    #[error("Failed to get default input device")]
    NoDefaultInput,

    #[error("Failed to get supported config: {0}")]
    UnsupportedConfig(String),

    #[error("Failed to build input stream: {0}")]
    StreamBuild(String),

    #[error("Failed to start audio stream: {0}")]
    StreamStart(String),

    #[error("VAD initialization failed: {0}")]
    VadInit(String),

    #[error("VAD processing error: {0}")]
    VadProcess(String),
}

/// Configuration-related errors
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    ReadFile(String),

    #[error("Failed to parse config: {0}")]
    Parse(String),

    #[error("Invalid configuration value: {field} - {message}")]
    InvalidValue { field: String, message: String },

    #[error("Missing required field: {0}")]
    MissingField(String),
}

/// Avatar-related errors
#[derive(Error, Debug)]
pub enum AvatarError {
    #[error("Asset not found: {0}")]
    AssetNotFound(String),

    #[error("Failed to load image: {0}")]
    ImageLoad(String),

    #[error("Invalid expression: {0}")]
    InvalidExpression(String),

    #[error("State transition error: {from} -> {to}")]
    InvalidTransition { from: String, to: String },
}

/// Output-related errors
#[derive(Error, Debug)]
pub enum OutputError {
    #[error("OBS connection failed: {0}")]
    ObsConnection(String),

    #[error("OBS authentication failed")]
    ObsAuth,

    #[error("OBS scene not found: {0}")]
    ObsSceneNotFound(String),

    #[error("OBS source not found: {0}")]
    ObsSourceNotFound(String),

    #[error("VMC receiver error: {0}")]
    VmcReceiver(String),

    #[error("VMC parse error: {0}")]
    VmcParse(String),

    #[error("Browser source server error: {0}")]
    BrowserServer(String),

    #[error("SSE broadcast error: {0}")]
    SseBroadcast(String),
}

/// Web server errors
#[derive(Error, Debug)]
pub enum WebError {
    #[error("Failed to bind to address: {0}")]
    Bind(String),

    #[error("Server startup failed: {0}")]
    Startup(String),

    #[error("Template rendering failed: {0}")]
    TemplateRender(String),

    #[error("Asset upload failed: {0}")]
    AssetUpload(String),
}

/// Integration-related errors
#[derive(Error, Debug)]
pub enum IntegrationError {
    #[error("Media service connection failed: {0}")]
    MediaConnection(String),

    #[error("AI service connection failed: {0}")]
    AiConnection(String),

    #[error("TTS synthesis failed: {0}")]
    TtsSynthesis(String),

    #[error("LLM request failed: {0}")]
    LlmRequest(String),
}

/// Tracking-related errors (OSF + VMC + MediaPipe)
#[derive(Error, Debug)]
pub enum TrackingError {
    #[error("OSF receiver error: {0}")]
    OsfReceiver(String),

    #[error("OSF parse error: {0}")]
    OsfParse(String),

    #[error("OSF subprocess error: {0}")]
    OsfSubprocess(String),

    #[error("VMC receiver error: {0}")]
    VmcReceiver(String),

    #[error("VMC parse error: {0}")]
    VmcParse(String),

    #[error("MediaPipe receiver error: {0}")]
    MpReceiver(String),

    #[error("MediaPipe parse error: {0}")]
    MpParse(String),

    #[error("MediaPipe subprocess error: {0}")]
    MpSubprocess(String),
}

/// Result type alias for Fushigi3D operations
pub type Result<T> = std::result::Result<T, Fushigi3dError>;
