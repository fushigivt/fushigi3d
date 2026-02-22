//! Browser source HTTP server for OBS

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::get,
    Router,
};
use std::sync::Arc;
use tower_http::services::ServeDir;

use crate::avatar::AssetManager;
use crate::config::AvatarConfig;
use crate::output::sse;
use crate::AppState;

/// Browser source server state
pub struct BrowserServer {
    app_state: Arc<AppState>,
    asset_manager: AssetManager,
}

impl BrowserServer {
    /// Create a new browser server
    pub fn new(app_state: Arc<AppState>, avatar_config: &AvatarConfig) -> Self {
        let asset_manager = AssetManager::new(avatar_config).unwrap_or_else(|e| {
            tracing::warn!("Failed to load assets: {}", e);
            AssetManager::new(&AvatarConfig::default()).expect("Default asset manager")
        });

        Self {
            app_state,
            asset_manager,
        }
    }

    /// Create the router for browser source endpoints
    pub fn router(self) -> Router {
        let assets_dir = self.asset_manager.base_dir().to_path_buf();
        let shared_state = Arc::new(self);

        Router::new()
            .route("/avatar", get(avatar_page))
            .route("/avatar/stream", get(avatar_stream))
            .route("/avatar/state", get(avatar_state))
            .route("/avatar/current-image", get(current_image))
            .nest_service("/assets", ServeDir::new(assets_dir))
            .with_state(shared_state)
    }
}

/// Browser source state (shared)
type BrowserState = Arc<BrowserServer>;

/// Render the avatar page for browser source
async fn avatar_page(State(state): State<BrowserState>) -> Html<String> {
    let current_state = state.app_state.get_avatar_state().await;
    let asset_key = current_state.asset_key();

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fushigi3D Avatar</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            background: transparent;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }}
        .avatar-container {{
            position: relative;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        .avatar-image {{
            max-width: 100%;
            max-height: 100vh;
            object-fit: contain;
            transition: opacity 0.1s ease-in-out;
        }}
        .avatar-idle {{
            /* Idle state styles */
        }}
        .avatar-speaking {{
            /* Speaking state styles */
        }}
        .avatar-expression {{
            /* Expression state styles */
        }}
    </style>
</head>
<body>
    <div class="avatar-container">
        <img
            id="avatar-image"
            class="avatar-image avatar-{state_type}"
            src="/assets/{asset_key}.png"
            alt="Avatar"
            onerror="this.src='/assets/idle.png'"
        >
    </div>

    <script>
        // SSE connection for real-time updates
        const evtSource = new EventSource('/avatar/stream');

        evtSource.addEventListener('state', function(event) {{
            const data = JSON.parse(event.data);
            const img = document.getElementById('avatar-image');

            // Update image source
            const newSrc = '/assets/' + data.asset_key + '.png';
            if (img.src !== newSrc) {{
                img.src = newSrc;
            }}

            // Update classes
            img.className = 'avatar-image avatar-' + data.state;
        }});

        evtSource.onerror = function(err) {{
            console.error('SSE error:', err);
            // Attempt to reconnect after 5 seconds
            setTimeout(function() {{
                window.location.reload();
            }}, 5000);
        }};
    </script>
</body>
</html>"#,
        state_type = current_state.state_type(),
        asset_key = asset_key,
    );

    Html(html)
}

/// SSE endpoint for avatar state updates
async fn avatar_stream(State(state): State<BrowserState>) -> impl IntoResponse {
    sse::create_state_stream(Arc::clone(&state.app_state))
}

/// Get current avatar state as JSON
async fn avatar_state(State(state): State<BrowserState>) -> impl IntoResponse {
    let current = state.app_state.get_avatar_state().await;

    let response = serde_json::json!({
        "state": current.state_type().to_string(),
        "is_speaking": current.is_speaking(),
        "expression": current.expression(),
        "asset_key": current.asset_key(),
        "mouth_open": current.mouth_open(),
        "blink": current.blink(),
    });

    axum::Json(response)
}

/// Get the current avatar image directly
async fn current_image(State(state): State<BrowserState>) -> Response {
    let current = state.app_state.get_avatar_state().await;
    let asset_key = current.asset_key();

    // Try to get the asset data
    match state.asset_manager.get_data(&asset_key) {
        Ok(data) => {
            let mime = state
                .asset_manager
                .get_mime_type(&asset_key)
                .unwrap_or("image/png");

            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, mime)],
                data,
            )
                .into_response()
        }
        Err(_) => {
            // Try fallback to idle
            match state.asset_manager.get_data("idle") {
                Ok(data) => {
                    let mime = state
                        .asset_manager
                        .get_mime_type("idle")
                        .unwrap_or("image/png");

                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, mime)],
                        data,
                    )
                        .into_response()
                }
                Err(_) => StatusCode::NOT_FOUND.into_response(),
            }
        }
    }
}
