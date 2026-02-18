//! OpenSeeFace native binary UDP protocol receiver
//!
//! Parses the binary UDP protocol used by OpenSeeFace's facetracker.py
//! when sending face tracking data. Each face frame is exactly 1785 bytes.
//!
//! Protocol layout per face (1785 bytes), from OpenSee.cs readFromPacket:
//!   - f64 time (8 bytes)
//!   - i32 face_id (4 bytes)
//!   - 2×f32 camera_resolution (8 bytes) — width, height
//!   - f32 right_eye_open (4 bytes)
//!   - f32 left_eye_open (4 bytes)
//!   - u8 got_3d_points (1 byte)
//!   - f32 fit_3d_error (4 bytes)
//!   - 4×f32 quaternion (16 bytes) — x, y, z, w
//!   - 3×f32 euler (12 bytes) — pitch, yaw, roll
//!   - 3×f32 translation (12 bytes) — x, y, z
//!   - 68×f32 confidence (272 bytes) — per-landmark confidence
//!   - 68×(2×f32) 2D landmarks (544 bytes)
//!   - 70×(3×f32) 3D landmarks (840 bytes)
//!   - 14×f32 features (56 bytes)
//!   Total: 8+4+8+4+4+1+4+16+12+12+272+544+840+56 = 1785 bytes
//!
//! Multi-face packets contain N×1785 bytes.

use std::net::UdpSocket;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use crate::avatar::AvatarState;
use crate::config::OsfConfig;
use crate::error::{TrackingError, RustuberError};

/// Size of a single face frame in bytes
pub const FRAME_SIZE: usize = 1785;

/// Number of 2D landmarks
pub const NUM_LANDMARKS_2D: usize = 68;

/// Number of 3D landmarks
pub const NUM_LANDMARKS_3D: usize = 70;

/// Number of feature values
pub const NUM_FEATURES: usize = 14;

/// A single face frame from OpenSeeFace
#[derive(Debug, Clone)]
pub struct OsfFrame {
    /// Timestamp from tracker
    pub time: f64,
    /// Face ID for multi-face tracking
    pub face_id: i32,
    /// Camera resolution (width, height)
    pub camera_resolution: [f32; 2],
    /// Right eye open amount (0.0 - 1.0)
    pub right_eye_open: f32,
    /// Left eye open amount (0.0 - 1.0)
    pub left_eye_open: f32,
    /// Whether 3D points were computed
    pub got_3d_points: bool,
    /// 3D fit error
    pub fit_3d_error: f32,
    /// Quaternion rotation (x, y, z, w)
    pub quaternion: [f32; 4],
    /// Euler rotation in degrees (pitch, yaw, roll)
    pub euler: [f32; 3],
    /// Translation (x, y, z)
    pub translation: [f32; 3],
    /// Per-landmark confidence values (68 values)
    pub confidence: Vec<f32>,
    /// 2D landmarks (68 points, each [x, y])
    pub landmarks_2d: Vec<[f32; 2]>,
    /// 3D landmarks (70 points, each [x, y, z])
    pub landmarks_3d: Vec<[f32; 3]>,
    /// 14 named feature values
    pub features: OsfFeatures,
}

/// The 14 named feature values from OpenSeeFace
#[derive(Debug, Clone, Default)]
pub struct OsfFeatures {
    /// Left eye open/close (index 0)
    pub eye_left: f32,
    /// Right eye open/close (index 1)
    pub eye_right: f32,
    /// Left eyebrow steepness (index 2)
    pub eyebrow_steepness_left: f32,
    /// Right eyebrow steepness (index 3)
    pub eyebrow_steepness_right: f32,
    /// Left eyebrow up/down (index 4)
    pub eyebrow_up_down_left: f32,
    /// Right eyebrow up/down (index 5)
    pub eyebrow_up_down_right: f32,
    /// Left eyebrow quirk (index 6)
    pub eyebrow_quirk_left: f32,
    /// Right eyebrow quirk (index 7)
    pub eyebrow_quirk_right: f32,
    /// Mouth corner up/down left (index 8)
    pub mouth_corner_up_down_left: f32,
    /// Mouth corner up/down right (index 9)
    pub mouth_corner_up_down_right: f32,
    /// Mouth corner in/out left (index 10)
    pub mouth_corner_in_out_left: f32,
    /// Mouth corner in/out right (index 11)
    pub mouth_corner_in_out_right: f32,
    /// Mouth open (index 12)
    pub mouth_open: f32,
    /// Mouth wide (index 13)
    pub mouth_wide: f32,
}

/// Aggregated OSF tracking data (mirrors VmcData pattern)
#[derive(Debug, Clone)]
pub struct OsfData {
    /// Most recently parsed frame for the selected face
    pub frame: Option<OsfFrame>,
    /// Whether any data has been received
    pub has_data: bool,
}

impl Default for OsfData {
    fn default() -> Self {
        Self {
            frame: None,
            has_data: false,
        }
    }
}

impl OsfData {
    /// Map OSF data to AvatarState field updates.
    ///
    /// If `blend_with_vad` is true, the `is_speaking` state from VAD is preserved
    /// and only face-tracking fields (mouth_open, blink, head) are updated.
    pub fn to_avatar_state(&self, current: &AvatarState, blend_with_vad: bool) -> AvatarState {
        let frame = match &self.frame {
            Some(f) => f,
            None => return current.clone(),
        };

        let mouth_open = frame.features.mouth_open;
        let blink = 1.0 - (frame.right_eye_open + frame.left_eye_open) / 2.0;

        let mut state = current.clone()
            .with_mouth_open(mouth_open)
            .with_blink(blink)
            .with_head_position(frame.translation)
            .with_head_rotation(frame.euler);

        if !blend_with_vad {
            // If not blending, mouth_open drives speaking state
            state = state.with_speaking(mouth_open > 0.15);
        }

        state
    }
}

/// Parse a single face frame from the buffer at the given offset.
///
/// Returns the parsed frame. Consumes exactly FRAME_SIZE (1785) bytes on success.
pub fn parse_frame(buf: &[u8], offset: usize) -> Result<OsfFrame, TrackingError> {
    if buf.len() < offset + FRAME_SIZE {
        return Err(TrackingError::OsfParse(format!(
            "Buffer too short: need {} bytes at offset {}, have {}",
            FRAME_SIZE,
            offset,
            buf.len()
        )));
    }

    let b = &buf[offset..];
    let mut pos = 0;

    // Helper closures for reading little-endian values
    let read_f64 = |p: &mut usize| -> f64 {
        let val = f64::from_le_bytes(b[*p..*p + 8].try_into().unwrap());
        *p += 8;
        val
    };
    let read_f32 = |p: &mut usize| -> f32 {
        let val = f32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
        *p += 4;
        val
    };
    let read_i32 = |p: &mut usize| -> i32 {
        let val = i32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
        *p += 4;
        val
    };

    // 1. time (f64, 8 bytes)
    let time = read_f64(&mut pos);

    // 2. face_id (i32, 4 bytes)
    let face_id = read_i32(&mut pos);

    // 3. camera_resolution (2×f32, 8 bytes)
    let camera_resolution = [read_f32(&mut pos), read_f32(&mut pos)];

    // 4. right_eye_open (f32, 4 bytes)
    let right_eye_open = read_f32(&mut pos);

    // 5. left_eye_open (f32, 4 bytes)
    let left_eye_open = read_f32(&mut pos);

    // 6. got_3d_points (u8, 1 byte)
    let got_3d_points = b[pos] != 0;
    pos += 1;

    // 7. fit_3d_error (f32, 4 bytes)
    let fit_3d_error = read_f32(&mut pos);

    // 8. quaternion (4×f32, 16 bytes)
    let quaternion = [
        read_f32(&mut pos),
        read_f32(&mut pos),
        read_f32(&mut pos),
        read_f32(&mut pos),
    ];

    // 9. euler (3×f32, 12 bytes)
    let euler = [read_f32(&mut pos), read_f32(&mut pos), read_f32(&mut pos)];

    // 10. translation (3×f32, 12 bytes)
    let translation = [read_f32(&mut pos), read_f32(&mut pos), read_f32(&mut pos)];

    // 11. confidence (68×f32, 272 bytes)
    let confidence: Vec<f32> = (0..NUM_LANDMARKS_2D).map(|_| read_f32(&mut pos)).collect();

    // 12. 2D landmarks (68×2×f32, 544 bytes)
    let mut landmarks_2d = Vec::with_capacity(NUM_LANDMARKS_2D);
    for _ in 0..NUM_LANDMARKS_2D {
        let x = read_f32(&mut pos);
        let y = read_f32(&mut pos);
        landmarks_2d.push([x, y]);
    }

    // 13. 3D landmarks (70×3×f32, 840 bytes)
    let mut landmarks_3d = Vec::with_capacity(NUM_LANDMARKS_3D);
    for _ in 0..NUM_LANDMARKS_3D {
        let x = read_f32(&mut pos);
        let y = read_f32(&mut pos);
        let z = read_f32(&mut pos);
        landmarks_3d.push([x, y, z]);
    }

    // 14. features (14×f32, 56 bytes)
    let feature_vals: Vec<f32> = (0..NUM_FEATURES).map(|_| read_f32(&mut pos)).collect();
    let features = OsfFeatures {
        eye_left: feature_vals[0],
        eye_right: feature_vals[1],
        eyebrow_steepness_left: feature_vals[2],
        eyebrow_steepness_right: feature_vals[3],
        eyebrow_up_down_left: feature_vals[4],
        eyebrow_up_down_right: feature_vals[5],
        eyebrow_quirk_left: feature_vals[6],
        eyebrow_quirk_right: feature_vals[7],
        mouth_corner_up_down_left: feature_vals[8],
        mouth_corner_up_down_right: feature_vals[9],
        mouth_corner_in_out_left: feature_vals[10],
        mouth_corner_in_out_right: feature_vals[11],
        mouth_open: feature_vals[12],
        mouth_wide: feature_vals[13],
    };

    debug_assert_eq!(pos, FRAME_SIZE);

    Ok(OsfFrame {
        time,
        face_id,
        camera_resolution,
        right_eye_open,
        left_eye_open,
        got_3d_points,
        fit_3d_error,
        quaternion,
        euler,
        translation,
        confidence,
        landmarks_2d,
        landmarks_3d,
        features,
    })
}

/// OpenSeeFace UDP receiver
pub struct OsfReceiver {
    config: OsfConfig,
    socket: Option<UdpSocket>,
    data: Arc<RwLock<OsfData>>,
}

impl OsfReceiver {
    /// Create a new OSF receiver
    pub fn new(config: &OsfConfig) -> Self {
        Self {
            config: config.clone(),
            socket: None,
            data: Arc::new(RwLock::new(OsfData::default())),
        }
    }

    /// Start receiving OSF data
    pub fn start(&mut self) -> Result<(), RustuberError> {
        let addr = format!("{}:{}", self.config.listen_address, self.config.port);

        let socket = UdpSocket::bind(&addr)
            .map_err(|e| TrackingError::OsfReceiver(format!("Failed to bind to {}: {}", addr, e)))?;

        socket
            .set_nonblocking(true)
            .map_err(|e| TrackingError::OsfReceiver(format!("Failed to set non-blocking: {}", e)))?;

        socket
            .set_read_timeout(Some(Duration::from_millis(100)))
            .ok();

        tracing::info!("OSF receiver listening on {}", addr);
        self.socket = Some(socket);

        Ok(())
    }

    /// Process incoming OSF packets (non-blocking).
    ///
    /// Packets may contain multiple faces (N×1785 bytes).
    /// Selects the face matching `config.face_id`.
    pub async fn process(&self) -> Result<Option<OsfData>, RustuberError> {
        let socket = match &self.socket {
            Some(s) => s,
            None => return Ok(None),
        };

        // Max buffer: support up to ~36 faces per packet
        let mut buf = [0u8; 65536];

        match socket.recv(&mut buf) {
            Ok(size) => {
                if size >= FRAME_SIZE {
                    let num_faces = size / FRAME_SIZE;
                    let mut selected_frame = None;

                    for i in 0..num_faces {
                        match parse_frame(&buf, i * FRAME_SIZE) {
                            Ok(frame) => {
                                if frame.face_id == self.config.face_id {
                                    selected_frame = Some(frame);
                                    break;
                                }
                                // If no exact match yet, keep the first face as fallback
                                if selected_frame.is_none() && i == 0 {
                                    selected_frame = Some(frame);
                                }
                            }
                            Err(e) => {
                                tracing::trace!("OSF frame parse error: {}", e);
                            }
                        }
                    }

                    if let Some(frame) = selected_frame {
                        let mut data = self.data.write().await;
                        data.frame = Some(frame);
                        data.has_data = true;
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // No data available
            }
            Err(e) => {
                return Err(TrackingError::OsfReceiver(format!("Receive error: {}", e)).into());
            }
        }

        Ok(Some(self.data.read().await.clone()))
    }

    /// Get the current OSF data
    pub async fn get_data(&self) -> OsfData {
        self.data.read().await.clone()
    }

    /// Check if any data has been received
    pub async fn has_data(&self) -> bool {
        self.data.read().await.has_data
    }

    /// Stop the receiver
    pub fn stop(&mut self) {
        self.socket = None;
        tracing::info!("OSF receiver stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 1785-byte test frame matching the real protocol layout
    fn build_test_frame(
        time: f64,
        face_id: i32,
        euler: [f32; 3],
        translation: [f32; 3],
        mouth_open: f32,
        right_eye_open: f32,
        left_eye_open: f32,
    ) -> Vec<u8> {
        let mut buf = Vec::with_capacity(FRAME_SIZE);

        // 1. time (f64, 8 bytes)
        buf.extend_from_slice(&time.to_le_bytes());
        // 2. face_id (i32, 4 bytes)
        buf.extend_from_slice(&face_id.to_le_bytes());
        // 3. camera_resolution (2×f32, 8 bytes)
        buf.extend_from_slice(&640.0f32.to_le_bytes());
        buf.extend_from_slice(&480.0f32.to_le_bytes());
        // 4. right_eye_open (f32, 4 bytes)
        buf.extend_from_slice(&right_eye_open.to_le_bytes());
        // 5. left_eye_open (f32, 4 bytes)
        buf.extend_from_slice(&left_eye_open.to_le_bytes());
        // 6. got_3d_points (u8, 1 byte)
        buf.push(1);
        // 7. fit_3d_error (f32, 4 bytes)
        buf.extend_from_slice(&0.01f32.to_le_bytes());
        // 8. quaternion (4×f32, 16 bytes) — identity
        for &v in &[0.0f32, 0.0, 0.0, 1.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // 9. euler (3×f32, 12 bytes)
        for &v in &euler {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // 10. translation (3×f32, 12 bytes)
        for &v in &translation {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // 11. confidence (68×f32, 272 bytes)
        for i in 0..NUM_LANDMARKS_2D {
            buf.extend_from_slice(&(0.9 + i as f32 * 0.001).to_le_bytes());
        }
        // 12. 2D landmarks (68×2×f32, 544 bytes)
        for i in 0..NUM_LANDMARKS_2D {
            let x = i as f32 * 0.2;
            let y = i as f32 * 0.1;
            buf.extend_from_slice(&x.to_le_bytes());
            buf.extend_from_slice(&y.to_le_bytes());
        }
        // 13. 3D landmarks (70×3×f32, 840 bytes)
        for i in 0..NUM_LANDMARKS_3D {
            let x = i as f32 * 0.01;
            let y = i as f32 * 0.02;
            let z = i as f32 * 0.03;
            buf.extend_from_slice(&x.to_le_bytes());
            buf.extend_from_slice(&y.to_le_bytes());
            buf.extend_from_slice(&z.to_le_bytes());
        }
        // 14. features (14×f32, 56 bytes)
        let features = [
            0.8f32, 0.9, // eye_left, eye_right
            0.1, 0.1,    // eyebrow_steepness L/R
            0.2, 0.2,    // eyebrow_up_down L/R
            0.0, 0.0,    // eyebrow_quirk L/R
            0.3, 0.3,    // mouth_corner_up_down L/R
            0.0, 0.0,    // mouth_corner_in_out L/R
            mouth_open,  // mouth_open
            0.1,         // mouth_wide
        ];
        for &v in &features {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        assert_eq!(buf.len(), FRAME_SIZE);
        buf
    }

    #[test]
    fn test_parse_frame_basic() {
        let buf = build_test_frame(
            1.234,
            0,
            [10.0, 20.0, 5.0],
            [0.1, 0.2, 0.3],
            0.7,
            0.8,
            0.9,
        );

        let frame = parse_frame(&buf, 0).unwrap();

        assert!((frame.time - 1.234).abs() < 0.001);
        assert_eq!(frame.face_id, 0);
        assert!((frame.camera_resolution[0] - 640.0).abs() < 0.01);
        assert!((frame.camera_resolution[1] - 480.0).abs() < 0.01);
        assert!((frame.right_eye_open - 0.8).abs() < 0.01);
        assert!((frame.left_eye_open - 0.9).abs() < 0.01);
        assert!(frame.got_3d_points);
        assert!((frame.fit_3d_error - 0.01).abs() < 0.001);
        assert!((frame.euler[0] - 10.0).abs() < 0.01);
        assert!((frame.euler[1] - 20.0).abs() < 0.01);
        assert!((frame.euler[2] - 5.0).abs() < 0.01);
        assert!((frame.translation[0] - 0.1).abs() < 0.01);
        assert!((frame.translation[1] - 0.2).abs() < 0.01);
        assert!((frame.translation[2] - 0.3).abs() < 0.01);
        assert!((frame.features.mouth_open - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_parse_frame_confidence() {
        let buf = build_test_frame(0.0, 0, [0.0; 3], [0.0; 3], 0.0, 1.0, 1.0);
        let frame = parse_frame(&buf, 0).unwrap();

        assert_eq!(frame.confidence.len(), NUM_LANDMARKS_2D);
        assert!((frame.confidence[0] - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_parse_frame_landmarks_2d() {
        let buf = build_test_frame(0.0, 0, [0.0; 3], [0.0; 3], 0.0, 1.0, 1.0);
        let frame = parse_frame(&buf, 0).unwrap();

        assert_eq!(frame.landmarks_2d.len(), NUM_LANDMARKS_2D);
        // Landmark 5: x=5*0.2=1.0, y=5*0.1=0.5
        assert!((frame.landmarks_2d[5][0] - 1.0).abs() < 0.01);
        assert!((frame.landmarks_2d[5][1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_parse_frame_landmarks_3d() {
        let buf = build_test_frame(0.0, 0, [0.0; 3], [0.0; 3], 0.0, 1.0, 1.0);
        let frame = parse_frame(&buf, 0).unwrap();

        assert_eq!(frame.landmarks_3d.len(), NUM_LANDMARKS_3D);
        // Landmark 10: x=0.1, y=0.2, z=0.3
        assert!((frame.landmarks_3d[10][0] - 0.1).abs() < 0.01);
        assert!((frame.landmarks_3d[10][1] - 0.2).abs() < 0.01);
        assert!((frame.landmarks_3d[10][2] - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_parse_frame_buffer_too_short() {
        let buf = vec![0u8; 100];
        assert!(parse_frame(&buf, 0).is_err());
    }

    #[test]
    fn test_parse_multi_face() {
        let frame0 = build_test_frame(1.0, 0, [10.0, 0.0, 0.0], [0.0; 3], 0.5, 1.0, 1.0);
        let frame1 = build_test_frame(1.0, 1, [20.0, 0.0, 0.0], [0.0; 3], 0.8, 0.5, 0.5);

        let mut buf = Vec::new();
        buf.extend_from_slice(&frame0);
        buf.extend_from_slice(&frame1);

        let f0 = parse_frame(&buf, 0).unwrap();
        let f1 = parse_frame(&buf, FRAME_SIZE).unwrap();

        assert_eq!(f0.face_id, 0);
        assert!((f0.euler[0] - 10.0).abs() < 0.01);
        assert_eq!(f1.face_id, 1);
        assert!((f1.euler[0] - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_osf_data_to_avatar_state_blend() {
        let frame = parse_frame(
            &build_test_frame(
                0.0, 0,
                [15.0, -5.0, 2.0],
                [0.1, 0.2, 0.5],
                0.6,
                0.7,
                0.9,
            ),
            0,
        ).unwrap();

        let data = OsfData {
            frame: Some(frame),
            has_data: true,
        };

        // With blend_with_vad = true, speaking state should be preserved from current
        let current = AvatarState::default().with_speaking(true);
        let updated = data.to_avatar_state(&current, true);

        assert!(updated.is_speaking()); // preserved from VAD
        assert!((updated.mouth_open() - 0.6).abs() < 0.01);
        // blink = 1.0 - avg(0.7, 0.9) = 1.0 - 0.8 = 0.2
        assert!((updated.blink() - 0.2).abs() < 0.01);
        assert!((updated.head_rotation()[0] - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_osf_data_to_avatar_state_no_blend() {
        let frame = parse_frame(
            &build_test_frame(0.0, 0, [0.0; 3], [0.0; 3], 0.6, 1.0, 1.0),
            0,
        ).unwrap();

        let data = OsfData {
            frame: Some(frame),
            has_data: true,
        };

        // With blend_with_vad = false, mouth_open > 0.15 -> speaking
        let current = AvatarState::default();
        let updated = data.to_avatar_state(&current, false);
        assert!(updated.is_speaking());
    }

    #[test]
    fn test_feature_values() {
        let buf = build_test_frame(0.0, 0, [0.0; 3], [0.0; 3], 0.42, 1.0, 1.0);
        let frame = parse_frame(&buf, 0).unwrap();

        assert!((frame.features.eye_left - 0.8).abs() < 0.01);
        assert!((frame.features.eye_right - 0.9).abs() < 0.01);
        assert!((frame.features.mouth_open - 0.42).abs() < 0.01);
        assert!((frame.features.mouth_wide - 0.1).abs() < 0.01);
    }
}
