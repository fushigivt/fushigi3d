//! Tracking subprocess manager
//!
//! Launches and manages Python tracker subprocesses (OpenSeeFace, MediaPipe)
//! as child processes with automatic cleanup on drop.

use tokio::process::{Child, Command};

use crate::config::{MediaPipeConfig, OsfConfig};
use crate::error::{Fushigi3dError, TrackingError};

/// Generic tracker subprocess manager.
///
/// Wraps a `Child` process that is built from a command closure, providing
/// start / stop / health-check / auto-restart in one place.
pub struct TrackerSubprocess {
    name: &'static str,
    child: Option<Child>,
    build_cmd: Box<dyn Fn() -> Command + Send>,
    pub auto_restart: bool,
    pub restart_delay_secs: u64,
}

impl TrackerSubprocess {
    /// Launch the tracker subprocess.
    pub fn start(&mut self) -> Result<(), Fushigi3dError> {
        if self.is_running() {
            return Ok(());
        }

        let child = (self.build_cmd)()
            .kill_on_drop(true)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                TrackingError::Subprocess(format!(
                    "Failed to launch {} subprocess: {}",
                    self.name, e
                ))
            })?;

        tracing::info!(
            "{} subprocess started (pid: {:?})",
            self.name,
            child.id(),
        );

        self.child = Some(child);
        Ok(())
    }

    /// Check if the subprocess is still running (non-blocking).
    pub fn is_running(&mut self) -> bool {
        match &mut self.child {
            Some(child) => match child.try_wait() {
                Ok(None) => true,
                Ok(Some(status)) => {
                    tracing::warn!("{} subprocess exited with: {}", self.name, status);
                    self.child = None;
                    false
                }
                Err(e) => {
                    tracing::error!(
                        "Failed to check {} subprocess status: {}",
                        self.name,
                        e
                    );
                    false
                }
            },
            None => false,
        }
    }

    /// Stop the subprocess by killing it.
    pub async fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            tracing::info!("Stopping {} subprocess (pid: {:?})", self.name, child.id());
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
    }
}

/// Create a `TrackerSubprocess` for OpenSeeFace.
pub fn osf_subprocess(config: &OsfConfig) -> TrackerSubprocess {
    let config = config.clone();
    TrackerSubprocess {
        name: "OpenSeeFace",
        child: None,
        build_cmd: Box::new(move || {
            let mut cmd = Command::new("python3");
            cmd.arg(&config.facetracker_path)
                .args(["-v", "0"])
                .args(["-s", "1"])
                .args(["--ip", &config.listen_address])
                .args(["--port", &config.port.to_string()])
                .args(["--capture", &config.camera_device.to_string()])
                .args(["--model-dir", &config.model_quality.to_string()])
                .args(["--max-faces", &config.max_faces.to_string()])
                .args(["--width", &config.capture_width.to_string()])
                .args(["--height", &config.capture_height.to_string()])
                .args(["--fps", &config.capture_fps.to_string()]);
            cmd
        }),
        auto_restart: config.auto_restart,
        restart_delay_secs: config.restart_delay_secs,
    }
}

/// Create a `TrackerSubprocess` for the MediaPipe Python tracker.
pub fn mp_subprocess(config: &MediaPipeConfig) -> TrackerSubprocess {
    let config = config.clone();
    TrackerSubprocess {
        name: "MediaPipe",
        child: None,
        build_cmd: Box::new(move || {
            let mut cmd = Command::new("python3");
            cmd.arg(&config.tracker_script)
                .args(["--ip", &config.listen_address])
                .args(["--port", &config.port.to_string()])
                .args(["--capture", &config.camera_device.to_string()])
                .args(["--width", &config.capture_width.to_string()])
                .args(["--height", &config.capture_height.to_string()])
                .args(["--fps", &config.capture_fps.to_string()])
                .args(["--model-dir", &config.model_dir]);
            if !config.enable_body_tracking {
                cmd.arg("--no-pose");
            }
            cmd
        }),
        auto_restart: config.auto_restart,
        restart_delay_secs: config.restart_delay_secs,
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
