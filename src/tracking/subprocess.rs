//! Tracking subprocess managers
//!
//! Launches and manages Python tracker subprocesses (OpenSeeFace, MediaPipe)
//! as child processes with automatic cleanup on drop.

use tokio::process::{Child, Command};

use crate::config::{MediaPipeConfig, OsfConfig};
use crate::error::{RustuberError, TrackingError};

/// Manages an OpenSeeFace facetracker subprocess
pub struct OsfSubprocess {
    child: Option<Child>,
    config: OsfConfig,
}

impl OsfSubprocess {
    /// Create a new subprocess manager (does not start the process)
    pub fn new(config: &OsfConfig) -> Self {
        Self {
            child: None,
            config: config.clone(),
        }
    }

    /// Launch the facetracker subprocess.
    ///
    /// Runs: `python3 <facetracker_path> -v 0 -s 1 --ip <listen_address> --port <port>
    ///        --capture <camera_device> --model-dir <model_quality>
    ///        --max-faces <max_faces> --width <capture_width> --height <capture_height>
    ///        --fps <capture_fps>`
    pub fn start(&mut self) -> Result<(), RustuberError> {
        if self.is_running() {
            return Ok(());
        }

        let child = Command::new("python3")
            .arg(&self.config.facetracker_path)
            .args(["-v", "0"])          // no visualization
            .args(["-s", "1"])          // headless / no preview
            .args(["--ip", &self.config.listen_address])
            .args(["--port", &self.config.port.to_string()])
            .args(["--capture", &self.config.camera_device.to_string()])
            .args(["--model-dir", &self.config.model_quality.to_string()])
            .args(["--max-faces", &self.config.max_faces.to_string()])
            .args(["--width", &self.config.capture_width.to_string()])
            .args(["--height", &self.config.capture_height.to_string()])
            .args(["--fps", &self.config.capture_fps.to_string()])
            .kill_on_drop(true)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                TrackingError::OsfSubprocess(format!(
                    "Failed to launch facetracker at '{}': {}",
                    self.config.facetracker_path, e
                ))
            })?;

        tracing::info!(
            "OpenSeeFace subprocess started (pid: {:?}, camera: {}, port: {})",
            child.id(),
            self.config.camera_device,
            self.config.port,
        );

        self.child = Some(child);
        Ok(())
    }

    /// Check if the subprocess is still running (non-blocking)
    pub fn is_running(&mut self) -> bool {
        match &mut self.child {
            Some(child) => match child.try_wait() {
                Ok(None) => true,              // still running
                Ok(Some(status)) => {
                    tracing::warn!("OpenSeeFace subprocess exited with: {}", status);
                    self.child = None;
                    false
                }
                Err(e) => {
                    tracing::error!("Failed to check subprocess status: {}", e);
                    false
                }
            },
            None => false,
        }
    }

    /// Stop the subprocess by killing it
    pub async fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            tracing::info!("Stopping OpenSeeFace subprocess (pid: {:?})", child.id());
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
    }
}

/// Manages a MediaPipe tracker subprocess (scripts/mp_tracker.py)
pub struct MpSubprocess {
    child: Option<Child>,
    config: MediaPipeConfig,
}

impl MpSubprocess {
    /// Create a new subprocess manager (does not start the process)
    pub fn new(config: &MediaPipeConfig) -> Self {
        Self {
            child: None,
            config: config.clone(),
        }
    }

    /// Launch the MediaPipe tracker subprocess.
    pub fn start(&mut self) -> Result<(), RustuberError> {
        if self.is_running() {
            return Ok(());
        }

        let child = Command::new("python3")
            .arg(&self.config.tracker_script)
            .args(["--ip", &self.config.listen_address])
            .args(["--port", &self.config.port.to_string()])
            .args(["--capture", &self.config.camera_device.to_string()])
            .args(["--width", &self.config.capture_width.to_string()])
            .args(["--height", &self.config.capture_height.to_string()])
            .args(["--fps", &self.config.capture_fps.to_string()])
            .args(["--model-dir", &self.config.model_dir])
            .kill_on_drop(true)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                TrackingError::MpSubprocess(format!(
                    "Failed to launch MediaPipe tracker at '{}': {}",
                    self.config.tracker_script, e
                ))
            })?;

        tracing::info!(
            "MediaPipe subprocess started (pid: {:?}, camera: {}, port: {})",
            child.id(),
            self.config.camera_device,
            self.config.port,
        );

        self.child = Some(child);
        Ok(())
    }

    /// Check if the subprocess is still running (non-blocking)
    pub fn is_running(&mut self) -> bool {
        match &mut self.child {
            Some(child) => match child.try_wait() {
                Ok(None) => true,
                Ok(Some(status)) => {
                    tracing::warn!("MediaPipe subprocess exited with: {}", status);
                    self.child = None;
                    false
                }
                Err(e) => {
                    tracing::error!("Failed to check MediaPipe subprocess status: {}", e);
                    false
                }
            },
            None => false,
        }
    }

    /// Stop the subprocess by killing it
    pub async fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            tracing::info!(
                "Stopping MediaPipe subprocess (pid: {:?})",
                child.id()
            );
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
    }
}

/// Check if the `mediapipe` Python package is available.
///
/// Runs `python3 -c "import mediapipe"` and returns true if it succeeds.
pub fn check_mediapipe_available() -> bool {
    match std::process::Command::new("python3")
        .args(["-c", "import mediapipe"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
    {
        Ok(status) => status.success(),
        Err(_) => false,
    }
}
