//! Route definitions for the web dashboard

use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;

use crate::config::HttpConfig;
use crate::AppState;

use super::api;
use super::htmx;

/// Create the main router with all routes
pub fn create_router(app_state: Arc<AppState>, config: &HttpConfig) -> Router {
    let cors = if config.cors_enabled {
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
    } else {
        CorsLayer::new()
    };

    Router::new()
        // Dashboard pages (HTMX)
        .route("/", get(htmx::index_page))
        .route("/settings", get(htmx::settings_page))
        .route("/settings/obs", get(htmx::obs_settings_page))
        .route("/preview", get(htmx::preview_page))
        // HTMX partials
        .route("/htmx/status", get(htmx::status_partial))
        .route("/htmx/preview", get(htmx::preview_partial))
        .route("/htmx/audio-devices", get(htmx::audio_devices_partial))
        .route("/htmx/expressions", get(htmx::expressions_partial))
        .route("/htmx/obs-status", get(htmx::obs_status_partial))
        .route("/htmx/obs-scenes", get(htmx::obs_scenes_partial))
        // HTMX actions
        .route("/htmx/set-expression", post(htmx::set_expression))
        .route("/htmx/clear-expression", post(htmx::clear_expression))
        .route("/htmx/update-settings", post(htmx::update_settings))
        .route("/htmx/update-obs-settings", post(htmx::update_obs_settings))
        .route("/htmx/test-obs", post(htmx::test_obs_connection))
        .route("/htmx/reconnect-obs", post(htmx::reconnect_obs))
        // API endpoints (JSON)
        .route("/api/status", get(api::get_status))
        .route("/api/config", get(api::get_config))
        .route("/api/config", post(api::update_config))
        .route("/api/state", get(api::get_state))
        .route("/api/state", post(api::set_state))
        .route("/api/expression", post(api::set_expression))
        .route("/api/expression", axum::routing::delete(api::clear_expression))
        .route("/api/audio/devices", get(api::list_audio_devices))
        // OBS API endpoints
        .route("/api/obs/status", get(api::get_obs_status))
        .route("/api/obs/test", post(api::test_obs_connection))
        .route("/api/obs/reconnect", post(api::reconnect_obs))
        // SSE stream for dashboard
        .route("/api/stream", get(api::state_stream))
        // Static files
        .nest_service("/static", ServeDir::new("static"))
        // Middleware
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(app_state)
}
