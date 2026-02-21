//! Limb-direction IK solver for body tracking.
//!
//! Converts MediaPipe Pose Landmarker world-space positions into VRM bone
//! rotations by comparing tracked limb directions against rest-pose directions.
//! Uses hierarchical solve so that lower-arm IK sees the already-rotated upper
//! arm, applies elbow pole-vector correction, and computes torso lean/tilt from
//! shoulder + hip landmarks.

#![cfg(feature = "native-ui")]

use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;

use super::skinning;
use super::vrm_loader::VrmModel;

/// A single limb segment: one bone driven by two landmarks.
struct LimbSegment {
    /// VRM bone name (used for elbow pole-vector identification)
    bone_name: &'static str,
    /// Node index of the bone to rotate
    node: usize,
    /// Parent node index (rotation is computed in parent-local space)
    parent_node: usize,
    /// MediaPipe landmark key for the proximal joint
    from_landmark: &'static str,
    /// MediaPipe landmark key for the distal joint
    to_landmark: &'static str,
    /// Rest-pose limb direction in world space (normalized)
    rest_dir_world: Vec3,
}

/// Spine bone for torso tilt/lean distribution.
#[allow(dead_code)]
struct SpineBone {
    /// Node index
    node: usize,
    /// Fraction of total torso rotation this bone receives
    weight: f32,
}

/// Precomputed IK setup for body tracking.
pub struct BodyIkSetup {
    limb_segments: Vec<LimbSegment>,
    #[allow(dead_code)]
    spine_bones: Vec<SpineBone>,
}

/// Limb segment definitions: (bone_name, bone_vrm_name, parent_bone_name_hint, from_landmark, to_landmark)
///
/// Landmark → VRM bone position mapping:
/// - "leftShoulder"  → world pos of leftUpperArm node
/// - "leftElbow"     → world pos of leftLowerArm node
/// - "leftWrist"     → world pos of leftHand node
/// - (mirror for right)
const LIMB_DEFS: &[(&str, &str, &str, &str, &str)] = &[
    // Upper before lower so hierarchical propagation works
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

/// Spine bone definitions for torso tilt/lean: (bone_name, weight).
/// Weights sum to 1.0 — distributed across the chain.
const SPINE_DEFS: &[(&str, f32)] = &[
    ("spine", 0.40),
    ("chest", 0.35),
    ("upperChest", 0.25),
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

            // Rest-pose direction in world space
            let rest_dir_world = match (to_world - from_world).try_normalize() {
                Some(d) => d,
                None => continue,
            };

            limb_segments.push(LimbSegment {
                bone_name,
                node,
                parent_node,
                from_landmark: from_lm,
                to_landmark: to_lm,
                rest_dir_world,
            });
        }

        if limb_segments.is_empty() {
            return None;
        }

        // Look up spine bones for torso tilt/lean
        let mut spine_bones = Vec::new();
        for &(bone_name, weight) in SPINE_DEFS {
            if let Some(&node) = model.bone_to_node.get(bone_name) {
                spine_bones.push(SpineBone { node, weight });
            }
        }

        Some(Self {
            limb_segments,
            spine_bones,
        })
    }

    /// Solve IK for the given body landmarks.
    ///
    /// Returns a map of node_index → rotation quaternion for each affected bone
    /// (arms + spine/chest). `rest_world` should come from
    /// `compute_world_transforms(model, &HashMap::new())`.
    pub fn solve(
        &self,
        model: &VrmModel,
        landmarks: &HashMap<String, [f32; 3]>,
        rest_world: &[Mat4],
    ) -> HashMap<usize, Quat> {
        let mut rotations = HashMap::new();

        // Mutable world transforms — updated after each segment so child bones
        // see the parent's IK-rotated transform (hierarchical propagation).
        let mut current_world = rest_world.to_vec();

        // TODO: torso lean/tilt from shoulder + hip landmarks needs parent-space
        // conversion — currently disabled because world-space tilt angles applied
        // in bone-local space produce incorrect rotations on rigs where spine
        // parent orientation differs from world axes.
        // self.solve_torso(model, landmarks, &mut rotations);

        // --- Limb IK with hierarchical propagation ---
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

            // Use current_world (hierarchically updated) for parent transform
            let parent_inv = current_world[seg.parent_node].inverse();

            // Both rest and tracked directions in current parent-local space
            let rest_dir_local =
                match parent_inv.transform_vector3(seg.rest_dir_world).try_normalize() {
                    Some(d) => d,
                    None => continue,
                };
            let tracked_dir_local =
                match parent_inv.transform_vector3(tracked_dir_world).try_normalize() {
                    Some(d) => d,
                    None => continue,
                };

            // Compute delta rotation from rest to tracked direction
            let mut delta = Quat::from_rotation_arc(rest_dir_local, tracked_dir_local);

            // Elbow pole-vector correction for lower arm segments
            if seg.bone_name == "leftLowerArm" || seg.bone_name == "rightLowerArm" {
                delta = correct_elbow_pole(
                    seg,
                    &from,
                    &tracked_dir_world,
                    &tracked_dir_local,
                    delta,
                    landmarks,
                );
            }

            // Apply delta on top of rest rotation
            let rest_rot = model.rest_rotations[seg.node];
            let ik_rot = rest_rot * delta;
            rotations.insert(seg.node, ik_rot);

            // Update current_world so child segments see this rotation
            let ik_local = Mat4::from_scale_rotation_translation(
                model.rest_scales[seg.node],
                ik_rot,
                model.rest_translations[seg.node],
            );
            current_world[seg.node] = current_world[seg.parent_node] * ik_local;
        }

        rotations
    }

    /// Compute torso lean/tilt rotations from shoulder (and optionally hip) landmarks.
    #[allow(dead_code)]
    fn solve_torso(
        &self,
        model: &VrmModel,
        landmarks: &HashMap<String, [f32; 3]>,
        rotations: &mut HashMap<usize, Quat>,
    ) {
        if self.spine_bones.is_empty() {
            return;
        }

        let (ls, rs) = match (
            landmarks.get("leftShoulder"),
            landmarks.get("rightShoulder"),
        ) {
            (Some(l), Some(r)) => (Vec3::from_array(*l), Vec3::from_array(*r)),
            _ => return,
        };

        // Shoulder tilt (Z rotation): deviation from horizontal
        let shoulder_dy = ls.y - rs.y;
        let shoulder_width = (ls - rs).length().max(0.01);
        let tilt_z = (shoulder_dy / shoulder_width).clamp(-1.0, 1.0).asin();

        // Torso lean (X rotation) from hip-shoulder vertical line
        let mut lean_x = 0.0f32;
        if let (Some(lh), Some(rh)) = (landmarks.get("leftHip"), landmarks.get("rightHip")) {
            let left_h = Vec3::from_array(*lh);
            let right_h = Vec3::from_array(*rh);
            let shoulder_mid = (ls + rs) * 0.5;
            let hip_mid = (left_h + right_h) * 0.5;
            let delta = shoulder_mid - hip_mid;
            lean_x = delta.z.atan2(delta.y);
        }

        // Distribute across spine bones, composed on rest rotations
        for sb in &self.spine_bones {
            let tilt = Quat::from_rotation_z(tilt_z * sb.weight)
                * Quat::from_rotation_x(lean_x * sb.weight);
            let rest_rot = model.rest_rotations[sb.node];
            rotations.insert(sb.node, rest_rot * tilt);
        }
    }
}

/// If the arm-plane normal indicates the elbow is pointing forward, flip by 180°
/// around the limb axis to correct.
fn correct_elbow_pole(
    seg: &LimbSegment,
    elbow_pos: &Vec3,
    tracked_dir_world: &Vec3,
    tracked_dir_local: &Vec3,
    delta: Quat,
    landmarks: &HashMap<String, [f32; 3]>,
) -> Quat {
    let upper_from_lm = if seg.bone_name == "leftLowerArm" {
        "leftShoulder"
    } else {
        "rightShoulder"
    };

    if let Some(upper_from) = landmarks.get(upper_from_lm) {
        let upper_from = Vec3::from_array(*upper_from);
        if let Some(upper_dir) = (*elbow_pos - upper_from).try_normalize() {
            let bend_normal = upper_dir.cross(*tracked_dir_world);
            // Positive Z → elbow forward → flip
            if bend_normal.length_squared() > 0.0001 && bend_normal.z > 0.0 {
                let flip =
                    Quat::from_axis_angle(*tracked_dir_local, std::f32::consts::PI);
                return flip * delta;
            }
        }
    }

    delta
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
            let len = seg.rest_dir_world.length();
            assert!(
                (len - 1.0).abs() < 0.01,
                "Rest direction should be normalized, got length {}",
                len
            );
        }

        // Spine bones should be found
        assert!(
            !setup.spine_bones.is_empty(),
            "Should find spine bones for torso"
        );
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
