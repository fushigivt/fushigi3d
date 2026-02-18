//! Web dashboard module
//!
//! HTMX-based dashboard for configuration and monitoring.

pub mod api;
pub mod htmx;
pub mod routes;

use axum::Router;
use std::sync::Arc;

use crate::config::HttpConfig;
use crate::AppState;

/// Web server for dashboard and API
pub struct WebServer {
    app_state: Arc<AppState>,
    config: HttpConfig,
}

impl WebServer {
    /// Create a new web server
    pub fn new(app_state: Arc<AppState>, config: &HttpConfig) -> Self {
        Self {
            app_state,
            config: config.clone(),
        }
    }

    /// Build the router
    pub fn router(&self) -> Router {
        routes::create_router(Arc::clone(&self.app_state), &self.config)
    }
}
