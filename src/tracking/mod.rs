//! Tracking module
//!
//! Face tracking backends for driving avatar state:
//! - OpenSeeFace native binary UDP protocol
//! - VMC/OSC protocol (VSeeFace, iFacialMocap, etc.)
//! - MediaPipe Face Landmarker (JSON over UDP)

pub mod mediapipe;
pub mod osf;
pub mod subprocess;
pub mod vmc;

use crate::avatar::AvatarState;
use crate::error::Fushigi3dError;

/// Unified interface for all tracking receivers.
///
/// Each backend (OSF, VMC, MediaPipe) implements this trait so the main
/// tracking loop can be written once instead of duplicated per-backend.
pub trait TrackingReceiver: Send {
    /// Bind the socket / start receiving.
    fn start(&mut self) -> Result<(), Fushigi3dError>;

    /// Stop the receiver (close socket).
    fn stop(&mut self);

    /// Receive one frame.
    ///
    /// Returns `Some(new_avatar_state)` when data was received and converted,
    /// or `None` on timeout / no data.
    fn process(
        &self,
        current: &AvatarState,
    ) -> impl std::future::Future<Output = Result<Option<AvatarState>, Fushigi3dError>> + Send;
}
