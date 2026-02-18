//! REST API endpoints

use axum::{
    extract::State,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::audio::capture;
use crate::output::sse;
use crate::AppState;

/// API response wrapper
#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl<T: Serialize> ApiResponse<T> {
    pub fn success(data: T) -> Json<Self> {
        Json(Self {
            success: true,
            data: Some(data),
            error: None,
        })
    }
}

impl ApiResponse<()> {
    pub fn error(message: &str) -> Json<Self> {
        Json(Self {
            success: false,
            data: None,
            error: Some(message.to_string()),
        })
    }

    pub fn ok() -> Json<Self> {
        Json(Self {
            success: true,
            data: None,
            error: None,
        })
    }
}

/// Status response
#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub state: String,
    pub is_speaking: bool,
    pub expression: Option<String>,
    pub asset_key: String,
    pub version: String,
    pub obs_connected: bool,
}

/// Get current status
pub async fn get_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let avatar = state.get_avatar_state().await;
    let obs_connected = state.obs_connected.load(std::sync::atomic::Ordering::Relaxed);

    ApiResponse::success(StatusResponse {
        state: avatar.state_type().to_string(),
        is_speaking: avatar.is_speaking(),
        expression: avatar.expression().map(|s| s.to_string()),
        asset_key: avatar.asset_key(),
        version: crate::VERSION.to_string(),
        obs_connected,
    })
}

/// Get current configuration
pub async fn get_config(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let config = state.config.read().await;
    Json(config.clone())
}

/// Update configuration
#[derive(Debug, Deserialize)]
pub struct ConfigUpdate {
    #[serde(default)]
    pub audio_device: Option<String>,
    #[serde(default)]
    pub vad_threshold: Option<f32>,
    #[serde(default)]
    pub obs_enabled: Option<bool>,
    #[serde(default)]
    pub obs_host: Option<String>,
    #[serde(default)]
    pub obs_port: Option<u16>,
    #[serde(default)]
    pub obs_password: Option<String>,
    #[serde(default)]
    pub obs_mode: Option<String>,
    #[serde(default)]
    pub obs_idle_scene: Option<String>,
    #[serde(default)]
    pub obs_speaking_scene: Option<String>,
    #[serde(default)]
    pub obs_scene: Option<String>,
    #[serde(default)]
    pub obs_idle_source: Option<String>,
    #[serde(default)]
    pub obs_speaking_source: Option<String>,
}

pub async fn update_config(
    State(state): State<Arc<AppState>>,
    Json(update): Json<ConfigUpdate>,
) -> impl IntoResponse {
    let mut config = state.config.write().await;

    if let Some(device) = update.audio_device {
        config.audio.device = device;
    }
    if let Some(threshold) = update.vad_threshold {
        config.vad.energy_threshold_db = threshold;
    }
    if let Some(enabled) = update.obs_enabled {
        config.obs.enabled = enabled;
    }
    if let Some(host) = update.obs_host {
        config.obs.host = host;
    }
    if let Some(port) = update.obs_port {
        config.obs.port = port;
    }
    if let Some(password) = update.obs_password {
        config.obs.password = if password.is_empty() { None } else { Some(password) };
    }
    if let Some(mode) = update.obs_mode {
        config.obs.mode = match mode.as_str() {
            "source" => crate::config::ObsMode::Source,
            _ => crate::config::ObsMode::Scene,
        };
    }
    if let Some(scene) = update.obs_idle_scene {
        config.obs.idle_scene = if scene.is_empty() { None } else { Some(scene) };
    }
    if let Some(scene) = update.obs_speaking_scene {
        config.obs.speaking_scene = if scene.is_empty() { None } else { Some(scene) };
    }
    if let Some(scene) = update.obs_scene {
        config.obs.scene = if scene.is_empty() { None } else { Some(scene) };
    }
    if let Some(source) = update.obs_idle_source {
        config.obs.idle_source = if source.is_empty() { None } else { Some(source) };
    }
    if let Some(source) = update.obs_speaking_source {
        config.obs.speaking_source = if source.is_empty() { None } else { Some(source) };
    }

    // Signal config change
    state.signal_config_changed();

    ApiResponse::<()>::ok()
}

/// Get current state
pub async fn get_state(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let avatar = state.get_avatar_state().await;

    Json(serde_json::json!({
        "state_type": avatar.state_type().to_string(),
        "is_speaking": avatar.is_speaking(),
        "expression": avatar.expression(),
        "mouth_open": avatar.mouth_open(),
        "blink": avatar.blink(),
        "head_position": avatar.head_position(),
        "head_rotation": avatar.head_rotation(),
        "asset_key": avatar.asset_key(),
    }))
}

/// Set state request
#[derive(Debug, Deserialize)]
pub struct SetStateRequest {
    #[serde(default)]
    pub speaking: Option<bool>,
    #[serde(default)]
    pub mouth_open: Option<f32>,
    #[serde(default)]
    pub blink: Option<f32>,
}

pub async fn set_state(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SetStateRequest>,
) -> impl IntoResponse {
    let mut current = state.get_avatar_state().await;

    if let Some(speaking) = request.speaking {
        current = current.with_speaking(speaking);
    }
    if let Some(mouth) = request.mouth_open {
        current = current.with_mouth_open(mouth);
    }
    if let Some(blink) = request.blink {
        current = current.with_blink(blink);
    }

    state.update_avatar_state(current).await;
    ApiResponse::<()>::ok()
}

/// Set expression request
#[derive(Debug, Deserialize)]
pub struct SetExpressionRequest {
    pub expression: String,
}

pub async fn set_expression(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SetExpressionRequest>,
) -> impl IntoResponse {
    let current = state.get_avatar_state().await;
    let new_state = current.with_expression(Some(request.expression));
    state.update_avatar_state(new_state).await;
    ApiResponse::<()>::ok()
}

/// Clear current expression
pub async fn clear_expression(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let current = state.get_avatar_state().await;
    let new_state = current.with_expression(None);
    state.update_avatar_state(new_state).await;
    ApiResponse::<()>::ok()
}

/// List available audio devices
pub async fn list_audio_devices() -> impl IntoResponse {
    let devices = capture::list_input_devices();
    let default = capture::default_input_device_name();

    ApiResponse::success(serde_json::json!({
        "devices": devices,
        "default": default,
    }))
}

/// OBS status response
#[derive(Debug, Serialize)]
pub struct ObsStatusResponse {
    pub connected: bool,
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub mode: String,
    pub scenes: Vec<String>,
    pub current_scene: Option<String>,
    pub error: Option<String>,
}

/// Get OBS connection status
pub async fn get_obs_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let config = state.config.read().await;
    let connected = state.obs_connected.load(std::sync::atomic::Ordering::Relaxed);
    let scenes = state.obs_scenes.read().await.clone();

    ApiResponse::success(ObsStatusResponse {
        connected,
        enabled: config.obs.enabled,
        host: config.obs.host.clone(),
        port: config.obs.port,
        mode: match config.obs.mode {
            crate::config::ObsMode::Scene => "scene".to_string(),
            crate::config::ObsMode::Source => "source".to_string(),
        },
        scenes,
        current_scene: None,
        error: None,
    })
}

/// Test OBS connection
#[derive(Debug, Deserialize)]
pub struct ObsTestRequest {
    pub host: String,
    pub port: u16,
    pub password: Option<String>,
}

pub async fn test_obs_connection(
    Json(request): Json<ObsTestRequest>,
) -> impl IntoResponse {
    use crate::output::obs::ObsClient;
    use crate::config::ObsConfig;

    let mut test_config = ObsConfig::default();
    test_config.host = request.host;
    test_config.port = request.port;
    test_config.password = request.password;

    let mut client = ObsClient::new(&test_config);

    match client.connect().await {
        Ok(()) => {
            // Try to list scenes to verify full connectivity
            match client.list_scenes().await {
                Ok(scenes) => ApiResponse::success(serde_json::json!({
                    "connected": true,
                    "scenes": scenes,
                })),
                Err(e) => ApiResponse::success(serde_json::json!({
                    "connected": true,
                    "scenes": [],
                    "warning": format!("Connected but failed to list scenes: {}", e),
                })),
            }
        }
        Err(e) => Json(ApiResponse {
            success: false,
            data: Some(serde_json::json!({
                "connected": false,
            })),
            error: Some(format!("Connection failed: {}", e)),
        }),
    }
}

/// Trigger OBS reconnect
pub async fn reconnect_obs(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.signal_obs_reconnect();
    ApiResponse::<()>::ok()
}

/// SSE stream endpoint
pub async fn state_stream(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    sse::create_state_stream(state)
}
