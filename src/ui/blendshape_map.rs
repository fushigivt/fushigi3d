//! ARKit → VRM morph target mapper.
//!
//! Converts ARKit blendshape names (from MediaPipe/VMC) to VRM morph target
//! weights by name. The VRM model's `targetNames` extras provide the mapping
//! from name → morph target index.

#![cfg(feature = "native-ui")]

use std::collections::HashMap;

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
    /// Returns a Vec of length `num_targets` with weights in [0.0, 1.0].
    pub fn map_blendshapes(&self, arkit: &HashMap<String, f32>, sensitivity: f32) -> Vec<f32> {
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

        // Blink: average of left/right → Fcl_EYE_Close
        let blink_l = get("eyeBlinkLeft");
        let blink_r = get("eyeBlinkRight");
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
            "Fcl_EYE_Close".into(), // index 12
            // 13..35 placeholder
            "p13".into(), "p14".into(), "p15".into(), "p16".into(),
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

        let weights = mapper.map_blendshapes(&arkit, 1.0);
        assert!((weights[36] - 0.8).abs() < 0.01, "Fcl_MTH_A should be 0.8");
    }

    #[test]
    fn test_blink_maps() {
        let mapper = BlendshapeMapper::new(&sample_names());
        let mut arkit = HashMap::new();
        arkit.insert("eyeBlinkLeft".to_string(), 0.6);
        arkit.insert("eyeBlinkRight".to_string(), 0.8);

        let weights = mapper.map_blendshapes(&arkit, 1.0);
        assert!(
            (weights[12] - 0.7).abs() < 0.01,
            "Fcl_EYE_Close should be avg 0.7"
        );
    }

    #[test]
    fn test_smile_triggers_joy() {
        let mapper = BlendshapeMapper::new(&sample_names());
        let mut arkit = HashMap::new();
        arkit.insert("mouthSmileLeft".to_string(), 0.9);
        arkit.insert("mouthSmileRight".to_string(), 0.9);

        let weights = mapper.map_blendshapes(&arkit, 1.0);
        // avg smile = 0.9, Joy = (0.9 - 0.35) * 1.5 = 0.825
        assert!(weights[3] > 0.7, "Fcl_ALL_Joy should be triggered");
    }

    #[test]
    fn test_empty_blendshapes() {
        let mapper = BlendshapeMapper::new(&sample_names());
        let arkit = HashMap::new();
        let weights = mapper.map_blendshapes(&arkit, 1.0);
        assert!(weights.iter().all(|&w| w == 0.0));
    }
}
