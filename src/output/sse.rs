//! Server-Sent Events for real-time state updates

use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::Stream;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

use crate::avatar::AvatarState;
use crate::AppState;

/// Create an SSE stream for avatar state updates
pub fn create_state_stream(
    app_state: Arc<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = app_state.subscribe_state();

    // Convert broadcast receiver to a stream
    let stream = BroadcastStream::new(rx).filter_map(|result| match result {
        Ok(state) => {
            let event = state_to_event(&state);
            Some(Ok(event))
        }
        Err(_) => None, // Skip lagged messages
    });

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

/// Convert avatar state to SSE event
fn state_to_event(state: &AvatarState) -> Event {
    let data = serde_json::json!({
        "state": state.state_type().to_string(),
        "is_speaking": state.is_speaking(),
        "expression": state.expression(),
        "mouth_open": state.mouth_open(),
        "blink": state.blink(),
        "asset_key": state.asset_key(),
    });

    Event::default()
        .event("state")
        .data(data.to_string())
}

/// State update for HTMX SSE
#[derive(Debug, Clone, serde::Serialize)]
pub struct StateUpdate {
    pub state: String,
    pub is_speaking: bool,
    pub expression: Option<String>,
    pub asset_key: String,
    pub image_url: Option<String>,
}

impl From<&AvatarState> for StateUpdate {
    fn from(state: &AvatarState) -> Self {
        Self {
            state: state.state_type().to_string(),
            is_speaking: state.is_speaking(),
            expression: state.expression().map(|s| s.to_string()),
            asset_key: state.asset_key(),
            image_url: None, // Set by handler based on asset manager
        }
    }
}

/// Create an HTMX-compatible SSE stream that sends HTML fragments
pub fn create_htmx_stream(
    app_state: Arc<AppState>,
    assets_url_prefix: &str,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = app_state.subscribe_state();
    let prefix = assets_url_prefix.to_string();

    let stream = BroadcastStream::new(rx).filter_map(move |result| {
        let prefix = prefix.clone();
        match result {
            Ok(state) => {
                // Generate HTMX-friendly HTML fragment
                let asset_key = state.asset_key();
                let image_url = format!("{}/{}.png", prefix, asset_key);

                let html = format!(
                    r#"<img id="avatar-image" src="{}" alt="Avatar" class="avatar-{}" hx-swap-oob="true">"#,
                    image_url,
                    state.state_type()
                );

                let event = Event::default()
                    .event("avatar-update")
                    .data(html);

                Some(Ok(event))
            }
            Err(_) => None,
        }
    });

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}
