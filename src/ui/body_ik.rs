//! Limb-direction IK solver for body tracking.
//!
//! Converts MediaPipe Pose Landmarker world-space positions into VRM bone
//! rotations by comparing tracked limb directions against rest-pose directions.

#![cfg(feature = "native-ui")]

use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;

use super::skinning;
use super::vrm_loader::VrmModel;

/// A single limb segment: one bone driven by two landmarks.
struct LimbSegment {
    /// VRM bone name (for debugging)
    #[allow(dead_code)]
    bone_name: &'static str,
    /// Node index of the bone to rotate
    node: usize,
    /// Parent node index (rotation is computed in parent-local space)
    parent_node: usize,
    /// MediaPipe landmark key for the proximal joint
    from_landmark: &'static str,
    /// MediaPipe landmark key for the distal joint
    to_landmark: &'static str,
    /// Rest-pose limb direction in parent-local space (normalized)
    rest_dir_local: Vec3,
}

/// Precomputed IK setup for body tracking.
pub struct BodyIkSetup {
    limb_segments: Vec<LimbSegment>,
}

/// Limb segment definitions: (bone_name, from_landmark, to_landmark)
///
/// Landmark → VRM bone position mapping:
/// - "leftShoulder"  → world pos of leftUpperArm node
/// - "leftElbow"     → world pos of leftLowerArm node
/// - "leftWrist"     → world pos of leftHand node
/// - (mirror for right)
const LIMB_DEFS: &[(&str, &str, &str, &str, &str)] = &[
    // (bone_name, bone_vrm_name, parent_bone_name_hint, from_landmark, to_landmark)
    ("leftUpperArm", "leftUpperArm", "leftShoulder", "leftShoulder", "leftElbow"),
    ("leftLowerArm", "leftLowerArm", "leftUpperArm", "leftElbow", "leftWrist"),
    ("rightUpperArm", "rightUpperArm", "rightShoulder", "rightShoulder", "rightElbow"),
    ("rightLowerArm", "rightLowerArm", "rightUpperArm", "rightElbow", "rightWrist"),
];

/// Mapping from landmark name to the VRM bone whose world position represents it.
const LANDMARK_TO_BONE: &[(&str, &str)] = &[
    ("leftShoulder", "leftUpperArm"),
    ("leftElbow", "leftLowerArm"),
    ("leftWrist", "leftHand"),
    ("rightShoulder", "rightUpperArm"),
    ("rightElbow", "rightLowerArm"),
    ("rightWrist", "rightHand"),
];

impl BodyIkSetup {
    /// Build the IK setup from a loaded VRM model.
    ///
    /// Returns `None` if the model lacks required bones.
    pub fn from_model(model: &VrmModel) -> Option<Self> {
        // Compute rest-pose world transforms
        let rest_world = skinning::compute_world_transforms(model, &HashMap::new());

        // Build landmark name → world position map from rest pose
        let mut landmark_world: HashMap<&str, Vec3> = HashMap::new();
        for &(landmark, bone) in LANDMARK_TO_BONE {
            if let Some(&node) = model.bone_to_node.get(bone) {
                let world_pos = rest_world[node].col(3).truncate();
                landmark_world.insert(landmark, world_pos);
            }
        }

        let mut limb_segments = Vec::new();

        for &(bone_name, bone_vrm, _parent_hint, from_lm, to_lm) in LIMB_DEFS {
            let node = match model.bone_to_node.get(bone_vrm) {
                Some(&n) => n,
                None => continue,
            };

            let parent_node = match model.parents[node] {
                Some(p) => p,
                None => continue,
            };

            let from_world = match landmark_world.get(from_lm) {
                Some(&p) => p,
                None => continue,
            };

            let to_world = match landmark_world.get(to_lm) {
                Some(&p) => p,
                None => continue,
            };

            // Compute rest direction in parent-local space
            let parent_world_inv = rest_world[parent_node].inverse();
            let from_local = (parent_world_inv * from_world.extend(1.0)).truncate();
            let to_local = (parent_world_inv * to_world.extend(1.0)).truncate();

            let dir = to_local - from_local;
            let rest_dir_local = match dir.try_normalize() {
                Some(d) => d,
                None => continue,
            };

            limb_segments.push(LimbSegment {
                bone_name,
                node,
                parent_node,
                from_landmark: from_lm,
                to_landmark: to_lm,
                rest_dir_local,
            });
        }

        if limb_segments.is_empty() {
            return None;
        }

        Some(Self { limb_segments })
    }

    /// Solve IK for the given body landmarks.
    ///
    /// Returns a map of node_index → rotation quaternion for each limb bone.
    /// `rest_world` should be computed from `compute_world_transforms(model, &HashMap::new())`.
    pub fn solve(
        &self,
        model: &VrmModel,
        landmarks: &HashMap<String, [f32; 3]>,
        rest_world: &[Mat4],
    ) -> HashMap<usize, Quat> {
        let mut rotations = HashMap::new();

        for seg in &self.limb_segments {
            let from = match landmarks.get(seg.from_landmark) {
                Some(p) => Vec3::from_array(*p),
                None => continue,
            };
            let to = match landmarks.get(seg.to_landmark) {
                Some(p) => Vec3::from_array(*p),
                None => continue,
            };

            let tracked_dir_world = match (to - from).try_normalize() {
                Some(d) => d,
                None => continue,
            };

            // Transform tracked direction into parent-local space
            let parent_inv = rest_world[seg.parent_node].inverse();
            let tracked_dir_local =
                match parent_inv.transform_vector3(tracked_dir_world).try_normalize() {
                    Some(d) => d,
                    None => continue,
                };

            // Compute delta rotation from rest to tracked direction
            let delta = Quat::from_rotation_arc(seg.rest_dir_local, tracked_dir_local);

            // Apply delta on top of rest rotation
            let rest_rot = model.rest_rotations[seg.node];
            rotations.insert(seg.node, rest_rot * delta);
        }

        rotations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_model_produces_segments() {
        let model_path = "assets/default/model.glb";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let model = VrmModel::load(model_path).unwrap();
        let setup = BodyIkSetup::from_model(&model);

        assert!(setup.is_some(), "BodyIkSetup should be created from model");
        let setup = setup.unwrap();
        assert_eq!(
            setup.limb_segments.len(),
            4,
            "Expected 4 limb segments (2 per arm)"
        );

        // Verify rest directions are normalized
        for seg in &setup.limb_segments {
            let len = seg.rest_dir_local.length();
            assert!(
                (len - 1.0).abs() < 0.01,
                "Rest direction should be normalized, got length {}",
                len
            );
        }
    }

    #[test]
    fn test_solve_with_rest_pose_landmarks() {
        let model_path = "assets/default/model.glb";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let model = VrmModel::load(model_path).unwrap();
        let setup = BodyIkSetup::from_model(&model).unwrap();

        // Use rest-pose world positions as "landmarks"
        let rest_world = skinning::compute_world_transforms(&model, &HashMap::new());
        let mut landmarks = HashMap::new();

        let landmark_bones = [
            ("leftShoulder", "leftUpperArm"),
            ("leftElbow", "leftLowerArm"),
            ("leftWrist", "leftHand"),
            ("rightShoulder", "rightUpperArm"),
            ("rightElbow", "rightLowerArm"),
            ("rightWrist", "rightHand"),
        ];

        for (lm_name, bone_name) in &landmark_bones {
            if let Some(&node) = model.bone_to_node.get(*bone_name) {
                let pos = rest_world[node].col(3).truncate();
                landmarks.insert(lm_name.to_string(), pos.to_array());
            }
        }

        let rotations = setup.solve(&model, &landmarks, &rest_world);

        // With rest-pose landmarks, rotations should be close to rest rotations
        for (&node, &rot) in &rotations {
            let rest = model.rest_rotations[node];
            let angle = rest.angle_between(rot);
            assert!(
                angle < 0.1,
                "Rest-pose solve should produce near-rest rotation, got angle diff: {}",
                angle
            );
        }
    }

    #[test]
    fn test_solve_with_modified_landmarks() {
        let model_path = "assets/default/model.glb";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let model = VrmModel::load(model_path).unwrap();
        let setup = BodyIkSetup::from_model(&model).unwrap();

        let rest_world = skinning::compute_world_transforms(&model, &HashMap::new());

        // Create landmarks with left arm pointing up
        let mut landmarks = HashMap::new();
        landmarks.insert("leftShoulder".to_string(), [0.2, 1.0, 0.0]);
        landmarks.insert("leftElbow".to_string(), [0.2, 1.5, 0.0]);
        landmarks.insert("leftWrist".to_string(), [0.2, 2.0, 0.0]);
        landmarks.insert("rightShoulder".to_string(), [-0.2, 1.0, 0.0]);
        landmarks.insert("rightElbow".to_string(), [-0.3, 0.7, 0.0]);
        landmarks.insert("rightWrist".to_string(), [-0.4, 0.4, 0.0]);

        let rotations = setup.solve(&model, &landmarks, &rest_world);
        assert!(!rotations.is_empty(), "Should produce rotations");

        // Rotations should differ from rest pose when landmarks are modified
        let mut any_different = false;
        for (&node, &rot) in &rotations {
            let rest = model.rest_rotations[node];
            let angle = rest.angle_between(rot);
            if angle > 0.05 {
                any_different = true;
            }
        }
        assert!(
            any_different,
            "Modified landmarks should produce different rotations"
        );
    }
}
