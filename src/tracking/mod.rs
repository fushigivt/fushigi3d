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
