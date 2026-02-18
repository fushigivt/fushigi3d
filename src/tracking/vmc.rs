//! VMC protocol receiver for face tracking data
//!
//! VMC (Virtual Motion Capture) is a protocol for transmitting motion capture
//! data over OSC. This module receives data from applications like:
//! - VSeeFace
//! - iFacialMocap
//! - etc.

use rosc::{OscMessage, OscPacket, OscType};
use std::collections::HashMap;
use std::net::UdpSocket;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use crate::config::VmcConfig;
use crate::error::{TrackingError, RustuberError};

/// VMC data received from face tracking software
#[derive(Debug, Clone, Default)]
pub struct VmcData {
    /// Blendshape values (0.0 - 1.0)
    pub blendshapes: HashMap<String, f32>,
    /// Head position (x, y, z)
    pub head_position: [f32; 3],
    /// Head rotation quaternion (x, y, z, w)
    pub head_rotation: [f32; 4],
    /// Bone transforms by name
    pub bones: HashMap<String, BoneTransform>,
    /// Whether data has been received
    pub has_data: bool,
}

/// Bone transform data
#[derive(Debug, Clone, Default)]
pub struct BoneTransform {
    /// Position (x, y, z)
    pub position: [f32; 3],
    /// Rotation quaternion (x, y, z, w)
    pub rotation: [f32; 4],
}

/// VMC protocol receiver
pub struct VmcReceiver {
    config: VmcConfig,
    socket: Option<UdpSocket>,
    data: Arc<RwLock<VmcData>>,
}

impl VmcReceiver {
    /// Create a new VMC receiver
    pub fn new(config: &VmcConfig) -> Self {
        Self {
            config: config.clone(),
            socket: None,
            data: Arc::new(RwLock::new(VmcData::default())),
        }
    }

    /// Start receiving VMC data
    pub fn start(&mut self) -> Result<(), RustuberError> {
        let addr = format!("0.0.0.0:{}", self.config.receiver_port);

        let socket = UdpSocket::bind(&addr)
            .map_err(|e| TrackingError::VmcReceiver(format!("Failed to bind to {}: {}", addr, e)))?;

        // Set non-blocking for async operation
        socket
            .set_nonblocking(true)
            .map_err(|e| TrackingError::VmcReceiver(format!("Failed to set non-blocking: {}", e)))?;

        // Set receive timeout
        socket
            .set_read_timeout(Some(Duration::from_millis(100)))
            .ok();

        tracing::info!("VMC receiver listening on {}", addr);
        self.socket = Some(socket);

        Ok(())
    }

    /// Process incoming VMC packets (non-blocking)
    pub async fn process(&self) -> Result<Option<VmcData>, RustuberError> {
        let socket = match &self.socket {
            Some(s) => s,
            None => return Ok(None),
        };

        let mut buf = [0u8; 65536];

        match socket.recv(&mut buf) {
            Ok(size) => {
                if size > 0 {
                    if let Ok((_, packet)) = rosc::decoder::decode_udp(&buf[..size]) {
                        self.handle_packet(packet).await?;
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // No data available, that's fine
            }
            Err(e) => {
                return Err(TrackingError::VmcReceiver(format!("Receive error: {}", e)).into());
            }
        }

        Ok(Some(self.data.read().await.clone()))
    }

    /// Handle an OSC packet
    fn handle_packet<'a>(
        &'a self,
        packet: OscPacket,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), RustuberError>> + Send + 'a>> {
        Box::pin(async move {
            match packet {
                OscPacket::Message(msg) => {
                    self.handle_message(msg).await?;
                }
                OscPacket::Bundle(bundle) => {
                    for packet in bundle.content {
                        self.handle_packet(packet).await?;
                    }
                }
            }
            Ok(())
        })
    }

    /// Handle an OSC message
    async fn handle_message(&self, msg: OscMessage) -> Result<(), RustuberError> {
        let mut data = self.data.write().await;
        data.has_data = true;

        match msg.addr.as_str() {
            // VMC blendshape format: /VMC/Ext/Blend/Val <name> <value>
            "/VMC/Ext/Blend/Val" => {
                if msg.args.len() >= 2 {
                    if let (Some(OscType::String(name)), Some(value)) =
                        (msg.args.get(0), msg.args.get(1))
                    {
                        let value = match value {
                            OscType::Float(f) => *f,
                            OscType::Double(d) => *d as f32,
                            OscType::Int(i) => *i as f32,
                            _ => return Ok(()),
                        };
                        data.blendshapes.insert(name.clone(), value);
                    }
                }
            }

            // VMC blend apply (batch): /VMC/Ext/Blend/Apply
            "/VMC/Ext/Blend/Apply" => {
                // This is sent after all blend values, useful for batching
            }

            // VMC bone position: /VMC/Ext/Bone/Pos <name> <px> <py> <pz> <qx> <qy> <qz> <qw>
            "/VMC/Ext/Bone/Pos" => {
                if msg.args.len() >= 8 {
                    if let Some(OscType::String(name)) = msg.args.get(0) {
                        let floats: Vec<f32> = msg.args[1..8]
                            .iter()
                            .filter_map(|arg| match arg {
                                OscType::Float(f) => Some(*f),
                                OscType::Double(d) => Some(*d as f32),
                                _ => None,
                            })
                            .collect();

                        if floats.len() == 7 {
                            let transform = BoneTransform {
                                position: [floats[0], floats[1], floats[2]],
                                rotation: [floats[3], floats[4], floats[5], floats[6]],
                            };

                            // Special handling for head bone
                            if name == "Head" {
                                data.head_position = transform.position;
                                data.head_rotation = transform.rotation;
                            }

                            data.bones.insert(name.clone(), transform);
                        }
                    }
                }
            }

            // VMC root position: /VMC/Ext/Root/Pos <name> <px> <py> <pz> <qx> <qy> <qz> <qw>
            "/VMC/Ext/Root/Pos" => {
                // Handle root transform if needed
            }

            // VMC tracking state: /VMC/Ext/OK <loaded> <calibrated>
            "/VMC/Ext/OK" => {
                // Tracking status
            }

            _ => {
                // Unknown message, ignore
                tracing::trace!("Unknown VMC message: {}", msg.addr);
            }
        }

        Ok(())
    }

    /// Get the current VMC data
    pub async fn get_data(&self) -> VmcData {
        self.data.read().await.clone()
    }

    /// Get a specific blendshape value
    pub async fn get_blendshape(&self, name: &str) -> Option<f32> {
        self.data.read().await.blendshapes.get(name).copied()
    }

    /// Check if any data has been received
    pub async fn has_data(&self) -> bool {
        self.data.read().await.has_data
    }

    /// Reset the received data
    pub async fn reset(&self) {
        let mut data = self.data.write().await;
        *data = VmcData::default();
    }

    /// Stop the receiver
    pub fn stop(&mut self) {
        self.socket = None;
        tracing::info!("VMC receiver stopped");
    }
}

/// Quaternion to euler angles conversion (returns degrees: pitch, yaw, roll)
pub fn quaternion_to_euler(q: [f32; 4]) -> [f32; 3] {
    let [x, y, z, w] = q;

    // Pitch (x-axis rotation)
    let sinr_cosp = 2.0 * (w * x + y * z);
    let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    let pitch = sinr_cosp.atan2(cosr_cosp);

    // Yaw (y-axis rotation)
    let sinp = 2.0 * (w * y - z * x);
    let yaw = if sinp.abs() >= 1.0 {
        std::f32::consts::FRAC_PI_2.copysign(sinp)
    } else {
        sinp.asin()
    };

    // Roll (z-axis rotation)
    let siny_cosp = 2.0 * (w * z + x * y);
    let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    let roll = siny_cosp.atan2(cosy_cosp);

    [
        pitch.to_degrees(),
        yaw.to_degrees(),
        roll.to_degrees(),
    ]
}

/// Common ARKit blendshape names
pub mod blendshapes {
    pub const BROW_DOWN_LEFT: &str = "browDownLeft";
    pub const BROW_DOWN_RIGHT: &str = "browDownRight";
    pub const BROW_INNER_UP: &str = "browInnerUp";
    pub const BROW_OUTER_UP_LEFT: &str = "browOuterUpLeft";
    pub const BROW_OUTER_UP_RIGHT: &str = "browOuterUpRight";

    pub const EYE_BLINK_LEFT: &str = "eyeBlinkLeft";
    pub const EYE_BLINK_RIGHT: &str = "eyeBlinkRight";
    pub const EYE_LOOK_DOWN_LEFT: &str = "eyeLookDownLeft";
    pub const EYE_LOOK_DOWN_RIGHT: &str = "eyeLookDownRight";
    pub const EYE_LOOK_IN_LEFT: &str = "eyeLookInLeft";
    pub const EYE_LOOK_IN_RIGHT: &str = "eyeLookInRight";
    pub const EYE_LOOK_OUT_LEFT: &str = "eyeLookOutLeft";
    pub const EYE_LOOK_OUT_RIGHT: &str = "eyeLookOutRight";
    pub const EYE_LOOK_UP_LEFT: &str = "eyeLookUpLeft";
    pub const EYE_LOOK_UP_RIGHT: &str = "eyeLookUpRight";
    pub const EYE_SQUINT_LEFT: &str = "eyeSquintLeft";
    pub const EYE_SQUINT_RIGHT: &str = "eyeSquintRight";
    pub const EYE_WIDE_LEFT: &str = "eyeWideLeft";
    pub const EYE_WIDE_RIGHT: &str = "eyeWideRight";

    pub const JAW_FORWARD: &str = "jawForward";
    pub const JAW_LEFT: &str = "jawLeft";
    pub const JAW_OPEN: &str = "jawOpen";
    pub const JAW_RIGHT: &str = "jawRight";

    pub const MOUTH_CLOSE: &str = "mouthClose";
    pub const MOUTH_FUNNEL: &str = "mouthFunnel";
    pub const MOUTH_PUCKER: &str = "mouthPucker";
    pub const MOUTH_SMILE_LEFT: &str = "mouthSmileLeft";
    pub const MOUTH_SMILE_RIGHT: &str = "mouthSmileRight";

    pub const CHEEK_PUFF: &str = "cheekPuff";
    pub const CHEEK_SQUINT_LEFT: &str = "cheekSquintLeft";
    pub const CHEEK_SQUINT_RIGHT: &str = "cheekSquintRight";

    /// Get average blink value from left and right eyes
    pub fn average_blink(blendshapes: &std::collections::HashMap<String, f32>) -> f32 {
        let left = blendshapes.get(EYE_BLINK_LEFT).copied().unwrap_or(0.0);
        let right = blendshapes.get(EYE_BLINK_RIGHT).copied().unwrap_or(0.0);
        (left + right) / 2.0
    }

    /// Get mouth open value
    pub fn mouth_open(blendshapes: &std::collections::HashMap<String, f32>) -> f32 {
        blendshapes.get(JAW_OPEN).copied().unwrap_or(0.0)
    }

    /// Check if smiling
    pub fn is_smiling(blendshapes: &std::collections::HashMap<String, f32>, threshold: f32) -> bool {
        let left = blendshapes.get(MOUTH_SMILE_LEFT).copied().unwrap_or(0.0);
        let right = blendshapes.get(MOUTH_SMILE_RIGHT).copied().unwrap_or(0.0);
        (left + right) / 2.0 > threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vmc_data_default() {
        let data = VmcData::default();
        assert!(!data.has_data);
        assert!(data.blendshapes.is_empty());
    }

    #[test]
    fn test_blendshape_helpers() {
        let mut blendshapes = HashMap::new();
        blendshapes.insert("eyeBlinkLeft".to_string(), 0.5);
        blendshapes.insert("eyeBlinkRight".to_string(), 0.7);
        blendshapes.insert("jawOpen".to_string(), 0.3);

        assert!((blendshapes::average_blink(&blendshapes) - 0.6).abs() < 0.01);
        assert!((blendshapes::mouth_open(&blendshapes) - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_quaternion_to_euler_identity() {
        // Identity quaternion (0, 0, 0, 1) should give (0, 0, 0) euler angles
        let euler = quaternion_to_euler([0.0, 0.0, 0.0, 1.0]);
        assert!(euler[0].abs() < 0.01);
        assert!(euler[1].abs() < 0.01);
        assert!(euler[2].abs() < 0.01);
    }
}
