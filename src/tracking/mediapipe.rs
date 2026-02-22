//! MediaPipe face tracking receiver
//!
//! Receives JSON-over-UDP packets from the `scripts/mp_tracker.py` Python helper.
//! MediaPipe provides ARKit-compatible blendshape names, so we reuse the
//! `vmc::blendshapes` helpers for mapping to AvatarState.

use serde::Deserialize;
use std::collections::HashMap;
use std::net::UdpSocket;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use crate::avatar::AvatarState;
use crate::config::MediaPipeConfig;
use crate::error::{Fushigi3dError, TrackingError};
use crate::tracking::vmc;

/// A single JSON packet from the MediaPipe tracker
#[derive(Debug, Clone, Deserialize)]
pub struct MpPacket {
    /// Whether a face was detected this frame
    pub face_detected: bool,
    /// ARKit blendshape name → value (0.0–1.0)
    pub blendshapes: HashMap<String, f32>,
    /// Head translation [x, y, z]
    pub head_position: [f32; 3],
    /// Head rotation in degrees [pitch, yaw, roll]
    pub head_rotation: [f32; 3],
    /// Whether a body pose was detected this frame
    #[serde(default)]
    pub body_detected: bool,
    /// Body landmark name → [x, y, z] world position (glTF coords)
    #[serde(default)]
    pub body_landmarks: HashMap<String, [f32; 3]>,
}

/// Aggregated MediaPipe tracking data (mirrors VmcData/OsfData pattern)
#[derive(Debug, Clone)]
pub struct MpData {
    /// Most recently parsed packet
    pub packet: Option<MpPacket>,
    /// Whether any data has been received
    pub has_data: bool,
}

impl Default for MpData {
    fn default() -> Self {
        Self {
            packet: None,
            has_data: false,
        }
    }
}

impl MpData {
    /// Map MediaPipe data to AvatarState.
    ///
    /// Reuses `vmc::blendshapes` helpers since MediaPipe uses the same ARKit names.
    /// If `blend_with_vad` is true, the speaking state from VAD is preserved.
    pub fn to_avatar_state(&self, current: &AvatarState, blend_with_vad: bool) -> AvatarState {
        let pkt = match &self.packet {
            Some(p) if p.face_detected => p,
            _ => return current.clone(),
        };

        let mouth_open = vmc::blendshapes::mouth_open(&pkt.blendshapes);
        let blink = vmc::blendshapes::average_blink(&pkt.blendshapes);

        let mut state = current
            .clone()
            .with_mouth_open(mouth_open)
            .with_blink(blink)
            .with_head_position(pkt.head_position)
            .with_head_rotation(pkt.head_rotation)
            .with_blendshapes(pkt.blendshapes.clone());

        if !blend_with_vad {
            state = state.with_speaking(mouth_open > 0.15);
        }

        if pkt.body_detected && !pkt.body_landmarks.is_empty() {
            state = state.with_body_landmarks(pkt.body_landmarks.clone());
        } else {
            state = state.with_body_landmarks(HashMap::new());
        }

        state
    }
}

/// MediaPipe JSON-over-UDP receiver
pub struct MpReceiver {
    config: MediaPipeConfig,
    socket: Option<UdpSocket>,
    data: Arc<RwLock<MpData>>,
}

impl MpReceiver {
    /// Create a new MediaPipe receiver (does not bind yet)
    pub fn new(config: &MediaPipeConfig) -> Self {
        Self {
            config: config.clone(),
            socket: None,
            data: Arc::new(RwLock::new(MpData::default())),
        }
    }

    /// Bind the UDP socket and start receiving
    pub fn start(&mut self) -> Result<(), Fushigi3dError> {
        let addr = format!("{}:{}", self.config.listen_address, self.config.port);

        let socket = UdpSocket::bind(&addr).map_err(|e| {
            TrackingError::MpReceiver(format!("Failed to bind to {}: {}", addr, e))
        })?;

        socket.set_nonblocking(true).map_err(|e| {
            TrackingError::MpReceiver(format!("Failed to set non-blocking: {}", e))
        })?;

        socket
            .set_read_timeout(Some(Duration::from_millis(100)))
            .ok();

        tracing::info!("MediaPipe receiver listening on {}", addr);
        self.socket = Some(socket);

        Ok(())
    }

    /// Process incoming JSON packets (non-blocking)
    pub async fn process(&self) -> Result<Option<MpData>, Fushigi3dError> {
        let socket = match &self.socket {
            Some(s) => s,
            None => return Ok(None),
        };

        let mut buf = [0u8; 65536];

        match socket.recv(&mut buf) {
            Ok(size) if size > 0 => {
                let packet: MpPacket = serde_json::from_slice(&buf[..size]).map_err(|e| {
                    TrackingError::MpParse(format!("JSON parse error: {}", e))
                })?;

                let mut data = self.data.write().await;
                data.packet = Some(packet);
                data.has_data = true;
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // No data available
            }
            Err(e) => {
                return Err(
                    TrackingError::MpReceiver(format!("Receive error: {}", e)).into(),
                );
            }
        }

        Ok(Some(self.data.read().await.clone()))
    }

    /// Get the current MediaPipe data
    pub async fn get_data(&self) -> MpData {
        self.data.read().await.clone()
    }

    /// Check if any data has been received
    pub async fn has_data(&self) -> bool {
        self.data.read().await.has_data
    }

    /// Stop the receiver
    pub fn stop(&mut self) {
        self.socket = None;
        tracing::info!("MediaPipe receiver stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_json(face_detected: bool, jaw_open: f32, blink_l: f32, blink_r: f32) -> String {
        serde_json::json!({
            "face_detected": face_detected,
            "blendshapes": {
                "jawOpen": jaw_open,
                "eyeBlinkLeft": blink_l,
                "eyeBlinkRight": blink_r,
                "mouthSmileLeft": 0.2,
                "mouthSmileRight": 0.25
            },
            "head_position": [0.1, 0.2, 0.5],
            "head_rotation": [10.5, -3.2, 1.1]
        })
        .to_string()
    }

    #[test]
    fn test_parse_packet() {
        let json = sample_json(true, 0.45, 0.12, 0.15);
        let pkt: MpPacket = serde_json::from_str(&json).unwrap();

        assert!(pkt.face_detected);
        assert!((pkt.blendshapes["jawOpen"] - 0.45).abs() < 0.01);
        assert!((pkt.blendshapes["eyeBlinkLeft"] - 0.12).abs() < 0.01);
        assert!((pkt.blendshapes["eyeBlinkRight"] - 0.15).abs() < 0.01);
        assert!((pkt.head_position[0] - 0.1).abs() < 0.01);
        assert!((pkt.head_rotation[0] - 10.5).abs() < 0.1);
    }

    #[test]
    fn test_parse_no_face() {
        let json = r#"{"face_detected":false,"blendshapes":{},"head_position":[0,0,0],"head_rotation":[0,0,0]}"#;
        let pkt: MpPacket = serde_json::from_str(json).unwrap();
        assert!(!pkt.face_detected);
        assert!(pkt.blendshapes.is_empty());
    }

    #[test]
    fn test_to_avatar_state_blend() {
        let json = sample_json(true, 0.6, 0.3, 0.5);
        let pkt: MpPacket = serde_json::from_str(&json).unwrap();

        let data = MpData {
            packet: Some(pkt),
            has_data: true,
        };

        let current = AvatarState::default().with_speaking(true);
        let updated = data.to_avatar_state(&current, true);

        // With blend_with_vad=true, speaking state is preserved from VAD
        assert!(updated.is_speaking());
        assert!((updated.mouth_open() - 0.6).abs() < 0.01);
        // blink = avg(0.3, 0.5) = 0.4
        assert!((updated.blink() - 0.4).abs() < 0.01);
        assert!((updated.head_rotation()[0] - 10.5).abs() < 0.1);
        assert!((updated.head_position()[2] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_to_avatar_state_no_blend() {
        let json = sample_json(true, 0.6, 0.0, 0.0);
        let pkt: MpPacket = serde_json::from_str(&json).unwrap();

        let data = MpData {
            packet: Some(pkt),
            has_data: true,
        };

        let current = AvatarState::default();
        let updated = data.to_avatar_state(&current, false);

        // With blend_with_vad=false, mouth_open > 0.15 → speaking
        assert!(updated.is_speaking());
    }

    #[test]
    fn test_to_avatar_state_no_face() {
        let json = sample_json(false, 0.0, 0.0, 0.0);
        let pkt: MpPacket = serde_json::from_str(&json).unwrap();

        let data = MpData {
            packet: Some(pkt),
            has_data: true,
        };

        let current = AvatarState::default().with_speaking(true);
        let updated = data.to_avatar_state(&current, true);

        // No face detected → returns current state unchanged
        assert!(updated.is_speaking());
    }

    #[test]
    fn test_mp_data_default() {
        let data = MpData::default();
        assert!(!data.has_data);
        assert!(data.packet.is_none());
    }

    #[test]
    fn test_parse_packet_with_body() {
        let json = serde_json::json!({
            "face_detected": true,
            "blendshapes": {"jawOpen": 0.3},
            "head_position": [0.0, 0.0, 0.0],
            "head_rotation": [0.0, 0.0, 0.0],
            "body_detected": true,
            "body_landmarks": {
                "leftShoulder": [0.1, 0.5, -0.3],
                "rightShoulder": [-0.1, 0.5, -0.3],
                "leftElbow": [0.2, 0.3, -0.2],
                "rightElbow": [-0.2, 0.3, -0.2]
            }
        })
        .to_string();

        let pkt: MpPacket = serde_json::from_str(&json).unwrap();
        assert!(pkt.body_detected);
        assert_eq!(pkt.body_landmarks.len(), 4);
        assert!((pkt.body_landmarks["leftShoulder"][0] - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_parse_legacy_packet_without_body() {
        // Legacy packet without body_detected/body_landmarks fields
        let json = r#"{"face_detected":true,"blendshapes":{"jawOpen":0.5},"head_position":[0,0,0],"head_rotation":[0,0,0]}"#;
        let pkt: MpPacket = serde_json::from_str(json).unwrap();
        assert!(pkt.face_detected);
        assert!(!pkt.body_detected);
        assert!(pkt.body_landmarks.is_empty());
    }

    #[test]
    fn test_to_avatar_state_with_body() {
        let json = serde_json::json!({
            "face_detected": true,
            "blendshapes": {"jawOpen": 0.3},
            "head_position": [0.0, 0.0, 0.0],
            "head_rotation": [0.0, 0.0, 0.0],
            "body_detected": true,
            "body_landmarks": {
                "leftShoulder": [0.1, 0.5, -0.3],
                "rightShoulder": [-0.1, 0.5, -0.3]
            }
        })
        .to_string();

        let pkt: MpPacket = serde_json::from_str(&json).unwrap();
        let data = MpData {
            packet: Some(pkt),
            has_data: true,
        };

        let current = AvatarState::default();
        let updated = data.to_avatar_state(&current, true);
        assert!(updated.has_body_tracking());
        assert_eq!(updated.body_landmarks().len(), 2);
    }
}
