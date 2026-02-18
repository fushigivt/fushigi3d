//! Output module
//!
//! Handles various output methods for the avatar state:
//! - OBS WebSocket integration
//! - Browser source (HTTP/SSE)

pub mod browser;
pub mod obs;
pub mod sse;
