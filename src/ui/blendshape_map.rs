//! ARKit → VRM morph target mapper.
//!
//! Converts ARKit blendshape names (from MediaPipe/VMC) to VRM morph target
//! weights by name. The VRM model's `targetNames` extras provide the mapping
//! from name → morph target index.

#![cfg(feature = "native-ui")]

use std::collections::HashMap;

/// Compensate for eye occlusion when the head is turned.
///
/// When head yaw exceeds ~23 degrees, the far eye's landmarks become unreliable.
/// Smoothly blend the occluded eye toward the visible eye's value over a
/// transition zone (OCCLUDE_START..OCCLUDE_FULL).
///
/// Based on KalidoKit's `stabilizeBlink` head-rotation override.
///
/// Returns (corrected_left, corrected_right).
fn compensate_eye_occlusion(left: f32, right: f32, head_yaw_rad: f32) -> (f32, f32) {
    const OCCLUDE_START: f32 = 0.4; // ~23 deg — begin blending
    const OCCLUDE_FULL: f32 = 0.6; // ~34 deg — fully mirrored

    let abs_yaw = head_yaw_rad.abs();
    if abs_yaw <= OCCLUDE_START {
        return (left, right);
    }

    let t = ((abs_yaw - OCCLUDE_START) / (OCCLUDE_FULL - OCCLUDE_START)).clamp(0.0, 1.0);

    if head_yaw_rad > 0.0 {
        // Turned right → left eye occluded → blend left toward right
        (left * (1.0 - t) + right * t, right)
    } else {
        // Turned left → right eye occluded → blend right toward left
        (left, right * (1.0 - t) + left * t)
    }
}

/// Suppress blink crosstalk from MediaPipe.
///
/// When winking, MediaPipe leaks signal onto the opposite eye (typically 0.2-0.4).
/// If one eye is significantly more closed than the other, scale down the weaker
/// eye's value so winks don't bleed across.
///
/// Returns (corrected_left, corrected_right).
fn suppress_blink_crosstalk(left: f32, right: f32) -> (f32, f32) {
    // Both eyes roughly equal → no correction needed (natural symmetric blink)
    let max = left.max(right);
    if max < 0.05 {
        return (left, right);
    }

    let ratio = left.min(right) / max;

    // If the weaker eye is less than 60% of the stronger, it's likely crosstalk.
    // Scale the weaker eye down: at ratio=0.6 suppression is 0, at ratio=0 it's full.
    const THRESHOLD: f32 = 0.6;
    if ratio >= THRESHOLD {
        // Both eyes close to equal — genuine symmetric blink
        return (left, right);
    }

    // Linear suppression: weaker eye gets scaled toward zero as asymmetry grows
    let suppression = 1.0 - (THRESHOLD - ratio) / THRESHOLD;
    if left < right {
        (left * suppression, right)
    } else {
        (left, right * suppression)
    }
}

/// Maps ARKit blendshape names to VRM morph target weights.
pub struct BlendshapeMapper {
    /// VRM morph target name → index in the weights array
    name_to_index: HashMap<String, usize>,
    /// Total number of morph targets
    num_targets: usize,
}

impl BlendshapeMapper {
    /// Create a new mapper from the model's morph target name list.
    pub fn new(morph_target_names: &[String]) -> Self {
        let mut name_to_index = HashMap::new();
        for (i, name) in morph_target_names.iter().enumerate() {
            name_to_index.insert(name.clone(), i);
        }
        Self {
            num_targets: morph_target_names.len(),
            name_to_index,
        }
    }

    /// Convert ARKit blendshapes to a VRM morph weight array.
    ///
    /// `sensitivity` scales all raw ARKit values before mapping (e.g. 1.2 = 20% boost).
    /// `head_yaw_rad` is the current head yaw in radians (positive = turned right) for
    /// eye occlusion compensation.
    /// Returns a Vec of length `num_targets` with weights in [0.0, 1.0].
    pub fn map_blendshapes(
        &self,
        arkit: &HashMap<String, f32>,
        sensitivity: f32,
        head_yaw_rad: f32,
    ) -> Vec<f32> {
        let mut weights = vec![0.0f32; self.num_targets];

        // Apply sensitivity multiplier to raw values via helper
        let get = |key: &str| -> f32 { Self::get(arkit, key) * sensitivity };

        // Direct mappings: ARKit name → VRM morph target name
        self.set_weight(&mut weights, "Fcl_MTH_A", get("jawOpen"));
        self.set_weight(&mut weights, "Fcl_MTH_U", get("mouthPucker"));
        self.set_weight(&mut weights, "Fcl_MTH_O", get("mouthFunnel"));

        // mouthStretchLeft/Right → Fcl_MTH_I (average)
        let stretch_l = get("mouthStretchLeft");
        let stretch_r = get("mouthStretchRight");
        self.set_weight(&mut weights, "Fcl_MTH_I", (stretch_l + stretch_r) * 0.5);

        // mouthSmileLeft/Right → Fcl_MTH_E (×0.6)
        let smile_l = get("mouthSmileLeft");
        let smile_r = get("mouthSmileRight");
        self.set_weight(&mut weights, "Fcl_MTH_E", (smile_l + smile_r) * 0.6);

        // Blink: track each eye independently.
        // 1) Compensate for eye occlusion when head is turned (far eye unreliable)
        // 2) Suppress MediaPipe crosstalk leak (~0.2-0.4 on opposite eye when winking)
        let blink_l_raw = get("eyeBlinkLeft");
        let blink_r_raw = get("eyeBlinkRight");
        let (blink_l_occ, blink_r_occ) =
            compensate_eye_occlusion(blink_l_raw, blink_r_raw, head_yaw_rad);
        let (blink_l, blink_r) = suppress_blink_crosstalk(blink_l_occ, blink_r_occ);
        self.set_weight(&mut weights, "Fcl_EYE_Close_L", blink_l);
        self.set_weight(&mut weights, "Fcl_EYE_Close_R", blink_r);
        // Also set the combined target for models that only have Fcl_EYE_Close
        self.set_weight(&mut weights, "Fcl_EYE_Close", (blink_l + blink_r) * 0.5);

        // Derived emotions from blendshape combinations
        let avg_smile = (smile_l + smile_r) * 0.5;
        if avg_smile > 0.35 {
            self.set_weight(&mut weights, "Fcl_ALL_Joy", (avg_smile - 0.35) * 1.5);
        }

        let eye_wide_l = get("eyeWideLeft");
        let eye_wide_r = get("eyeWideRight");
        let jaw_open = get("jawOpen");
        let avg_eye_wide = (eye_wide_l + eye_wide_r) * 0.5;
        if avg_eye_wide > 0.2 && jaw_open > 0.2 {
            self.set_weight(
                &mut weights,
                "Fcl_ALL_Surprised",
                (avg_eye_wide + jaw_open) * 0.5,
            );
        }

        let brow_down_l = get("browDownLeft");
        let brow_down_r = get("browDownRight");
        let avg_brow_down = (brow_down_l + brow_down_r) * 0.5;
        if avg_brow_down > 0.3 {
            self.set_weight(
                &mut weights,
                "Fcl_ALL_Angry",
                (avg_brow_down - 0.3) * 1.4,
            );
        }

        weights
    }

    fn set_weight(&self, weights: &mut [f32], name: &str, value: f32) {
        if let Some(&idx) = self.name_to_index.get(name) {
            weights[idx] = value.clamp(0.0, 1.0);
        }
    }

    fn get(map: &HashMap<String, f32>, key: &str) -> f32 {
        map.get(key).copied().unwrap_or(0.0)
    }

    pub fn num_targets(&self) -> usize {
        self.num_targets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_names() -> Vec<String> {
        vec![
            "Fcl_ALL_Neutral".into(),
            "Fcl_ALL_Angry".into(),
            "Fcl_ALL_Fun".into(),
            "Fcl_ALL_Joy".into(),
            "Fcl_ALL_Sorrow".into(),
            "Fcl_ALL_Surprised".into(),
            // indices 6..11 placeholder
            "placeholder_6".into(),
            "placeholder_7".into(),
            "placeholder_8".into(),
            "placeholder_9".into(),
            "placeholder_10".into(),
            "placeholder_11".into(),
            "Fcl_EYE_Close".into(),   // index 12
            "Fcl_EYE_Close_L".into(), // index 13
            "Fcl_EYE_Close_R".into(), // index 14
            // 15..35 placeholder
            "p15".into(), "p16".into(),
            "p17".into(), "p18".into(), "p19".into(), "p20".into(),
            "p21".into(), "p22".into(), "p23".into(), "p24".into(),
            "p25".into(), "p26".into(), "p27".into(), "p28".into(),
            "p29".into(), "p30".into(), "p31".into(), "p32".into(),
            "p33".into(), "p34".into(), "p35".into(),
            "Fcl_MTH_A".into(), // index 36
            "Fcl_MTH_I".into(), // 37
            "Fcl_MTH_U".into(), // 38
            "Fcl_MTH_E".into(), // 39
            "Fcl_MTH_O".into(), // 40
        ]
    }

    #[test]
    fn test_jaw_open_maps_to_mouth_a() {
        let mapper = BlendshapeMapper::new(&sample_names());
        let mut arkit = HashMap::new();
        arkit.insert("jawOpen".to_string(), 0.8);

        let weights = mapper.map_blendshapes(&arkit, 1.0, 0.0);
        assert!((weights[36] - 0.8).abs() < 0.01, "Fcl_MTH_A should be 0.8");
    }

    #[test]
    fn test_blink_maps_separately() {
        let mapper = BlendshapeMapper::new(&sample_names());
        let mut arkit = HashMap::new();
        arkit.insert("eyeBlinkLeft".to_string(), 0.6);
        arkit.insert("eyeBlinkRight".to_string(), 0.8);

        let weights = mapper.map_blendshapes(&arkit, 1.0, 0.0);
        // Separate eye targets
        assert!(
            (weights[13] - 0.6).abs() < 0.01,
            "Fcl_EYE_Close_L should be 0.6, got {}",
            weights[13]
        );
        assert!(
            (weights[14] - 0.8).abs() < 0.01,
            "Fcl_EYE_Close_R should be 0.8, got {}",
            weights[14]
        );
        // Combined fallback still set
        assert!(
            (weights[12] - 0.7).abs() < 0.01,
            "Fcl_EYE_Close should be avg 0.7, got {}",
            weights[12]
        );
    }

    #[test]
    fn test_smile_triggers_joy() {
        let mapper = BlendshapeMapper::new(&sample_names());
        let mut arkit = HashMap::new();
        arkit.insert("mouthSmileLeft".to_string(), 0.9);
        arkit.insert("mouthSmileRight".to_string(), 0.9);

        let weights = mapper.map_blendshapes(&arkit, 1.0, 0.0);
        // avg smile = 0.9, Joy = (0.9 - 0.35) * 1.5 = 0.825
        assert!(weights[3] > 0.7, "Fcl_ALL_Joy should be triggered");
    }

    #[test]
    fn test_blink_crosstalk_symmetric() {
        // Both eyes roughly equal — no suppression
        let (l, r) = suppress_blink_crosstalk(0.8, 0.75);
        assert!((l - 0.8).abs() < 0.01, "Left should be unchanged: {}", l);
        assert!((r - 0.75).abs() < 0.01, "Right should be unchanged: {}", r);
    }

    #[test]
    fn test_blink_crosstalk_wink_left() {
        // Left eye closed (0.9), right leaked (0.25) — right should be suppressed
        let (l, r) = suppress_blink_crosstalk(0.9, 0.25);
        assert!((l - 0.9).abs() < 0.01, "Closed eye unchanged: {}", l);
        assert!(r < 0.15, "Leaked eye should be suppressed: {}", r);
    }

    #[test]
    fn test_blink_crosstalk_wink_right() {
        // Right eye closed (0.85), left leaked (0.3) — left should be suppressed
        let (l, r) = suppress_blink_crosstalk(0.3, 0.85);
        assert!(l < 0.2, "Leaked eye should be suppressed: {}", l);
        assert!((r - 0.85).abs() < 0.01, "Closed eye unchanged: {}", r);
    }

    #[test]
    fn test_blink_crosstalk_both_zero() {
        let (l, r) = suppress_blink_crosstalk(0.0, 0.0);
        assert!(l.abs() < 0.001);
        assert!(r.abs() < 0.001);
    }

    #[test]
    fn test_eye_occlusion_no_rotation() {
        // No head turn → values unchanged
        let (l, r) = compensate_eye_occlusion(0.8, 0.2, 0.0);
        assert!((l - 0.8).abs() < 0.001);
        assert!((r - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_eye_occlusion_turned_right_full() {
        // Head turned right past OCCLUDE_FULL (0.6 rad) → left eye mirrors right
        let (l, r) = compensate_eye_occlusion(0.8, 0.2, 0.7);
        assert!((l - 0.2).abs() < 0.01, "Left should mirror right: {}", l);
        assert!((r - 0.2).abs() < 0.01, "Right unchanged: {}", r);
    }

    #[test]
    fn test_eye_occlusion_turned_left_full() {
        // Head turned left past OCCLUDE_FULL → right eye mirrors left
        let (l, r) = compensate_eye_occlusion(0.2, 0.8, -0.7);
        assert!((l - 0.2).abs() < 0.01, "Left unchanged: {}", l);
        assert!((r - 0.2).abs() < 0.01, "Right should mirror left: {}", r);
    }

    #[test]
    fn test_eye_occlusion_partial_blend() {
        // Head at midpoint of blend zone (0.5 rad) → 50% blend
        let (l, r) = compensate_eye_occlusion(1.0, 0.0, 0.5);
        // t = (0.5 - 0.4) / (0.6 - 0.4) = 0.5
        // left = 1.0 * 0.5 + 0.0 * 0.5 = 0.5
        assert!((l - 0.5).abs() < 0.01, "Left should be half-blended: {}", l);
        assert!(r.abs() < 0.01, "Right unchanged: {}", r);
    }

    #[test]
    fn test_eye_occlusion_below_threshold() {
        // Head at 0.3 rad (below OCCLUDE_START) → no change
        let (l, r) = compensate_eye_occlusion(0.9, 0.1, 0.3);
        assert!((l - 0.9).abs() < 0.001);
        assert!((r - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_occlusion_with_mapper() {
        // End-to-end: head turned right, left eye should mirror right via mapper
        let mapper = BlendshapeMapper::new(&sample_names());
        let mut arkit = HashMap::new();
        arkit.insert("eyeBlinkLeft".to_string(), 0.9); // unreliable (occluded)
        arkit.insert("eyeBlinkRight".to_string(), 0.1); // visible eye open

        // At 0.7 rad yaw, left eye fully mirrors right
        let weights = mapper.map_blendshapes(&arkit, 1.0, 0.7);
        // Left eye should be close to right eye's value (0.1)
        assert!(
            weights[13] < 0.15,
            "Occluded left eye should mirror visible right: {}",
            weights[13]
        );
    }

    #[test]
    fn test_empty_blendshapes() {
        let mapper = BlendshapeMapper::new(&sample_names());
        let arkit = HashMap::new();
        let weights = mapper.map_blendshapes(&arkit, 1.0, 0.0);
        assert!(weights.iter().all(|&w| w == 0.0));
    }
}
