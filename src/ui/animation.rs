//! Procedural animation for VRM models.
//!
//! Generates per-frame bone rotations for breathing, idle sway, head tracking,
//! body follow, arm/hand animation, and finger curl.

#![cfg(feature = "native-ui")]

use glam::Quat;
use std::collections::HashMap;
use std::f32::consts::PI;

use super::vrm_loader::VrmModel;

/// Compute the idle rest pose (slight shoulder relaxation).
pub fn idle_pose(model: &VrmModel) -> HashMap<usize, Quat> {
    let mut pose = HashMap::new();

    if let Some(&l_shoulder) = model.bone_to_node.get("leftShoulder") {
        let rest = model.rest_rotations[l_shoulder];
        let relax = Quat::from_axis_angle(glam::Vec3::Z, -0.05);
        pose.insert(l_shoulder, rest * relax);
    }
    if let Some(&r_shoulder) = model.bone_to_node.get("rightShoulder") {
        let rest = model.rest_rotations[r_shoulder];
        let relax = Quat::from_axis_angle(glam::Vec3::Z, 0.05);
        pose.insert(r_shoulder, rest * relax);
    }

    pose
}

/// Apply breathing animation to spine and chest bones.
fn apply_breathing(pose: &mut HashMap<usize, Quat>, model: &VrmModel, time: f32) {
    let breath_phase = (time * 1.2 * PI).sin();
    let breath_angle = breath_phase * 0.008;

    if let Some(&spine) = model.bone_to_node.get("spine") {
        let rest = model.rest_rotations[spine];
        let base = *pose.get(&spine).unwrap_or(&rest);
        let breath = Quat::from_axis_angle(glam::Vec3::X, breath_angle);
        pose.insert(spine, base * breath);
    }
    if let Some(&chest) = model.bone_to_node.get("chest") {
        let rest = model.rest_rotations[chest];
        let base = *pose.get(&chest).unwrap_or(&rest);
        let breath = Quat::from_axis_angle(glam::Vec3::X, breath_angle * 0.5);
        pose.insert(chest, base * breath);
    }
}

/// Apply subtle lateral idle sway to hips.
fn apply_idle_sway(pose: &mut HashMap<usize, Quat>, model: &VrmModel, time: f32) {
    let sway_phase = (time * 0.4 * PI).sin();
    let sway_angle = sway_phase * 0.006;

    if let Some(&hips) = model.bone_to_node.get("hips") {
        let rest = model.rest_rotations[hips];
        let base = *pose.get(&hips).unwrap_or(&rest);
        let sway = Quat::from_axis_angle(glam::Vec3::Z, sway_angle);
        pose.insert(hips, base * sway);
    }
}

/// Compute head/neck tracking rotations. Returns (head_rx, head_ry, head_rz) in radians.
fn apply_head_tracking(
    pose: &mut HashMap<usize, Quat>,
    model: &VrmModel,
    time: f32,
    head_rotation_deg: Option<[f32; 3]>,
    is_speaking: bool,
    tuning: &crate::config::TrackingTuning,
) -> (f32, f32, f32) {
    let (mut head_rx, mut head_ry, mut head_rz) = (0.0f32, 0.0f32, 0.0f32);

    if let Some([pitch, yaw, roll]) = head_rotation_deg {
        head_rx = (pitch * tuning.head_sensitivity * tuning.pitch_scale).to_radians();
        head_ry = (yaw * tuning.head_sensitivity * tuning.yaw_scale).to_radians();
        head_rz = (roll * tuning.head_sensitivity * tuning.roll_scale).to_radians();
    } else {
        head_rx += (time * 0.7).sin() * 0.015;
        head_ry += (time * 0.5).sin() * 0.01;
    }

    if is_speaking {
        let speak_nod = (time * 6.0).sin() * 0.02;
        head_rx += speak_nod;
    }

    // 60% to head
    if let Some(&head) = model.bone_to_node.get("head") {
        let rest = model.rest_rotations[head];
        let hq = Quat::from_euler(
            glam::EulerRot::YXZ,
            head_ry * 0.6,
            head_rx * 0.6,
            head_rz * 0.6,
        );
        pose.insert(head, rest * hq);
    }

    // 40% to neck
    if let Some(&neck) = model.bone_to_node.get("neck") {
        let rest = model.rest_rotations[neck];
        let nq = Quat::from_euler(
            glam::EulerRot::YXZ,
            head_ry * 0.4,
            head_rx * 0.4,
            head_rz * 0.4,
        );
        pose.insert(neck, rest * nq);
    }

    (head_rx, head_ry, head_rz)
}

/// Apply subtle upper body coupling that follows head movement.
fn apply_body_follow(
    pose: &mut HashMap<usize, Quat>,
    model: &VrmModel,
    head_rx: f32,
    head_ry: f32,
) {
    // upperChest: 4% of head yaw/pitch
    if let Some(&upper_chest) = model.bone_to_node.get("upperChest") {
        let rest = model.rest_rotations[upper_chest];
        let base = *pose.get(&upper_chest).unwrap_or(&rest);
        let follow = Quat::from_euler(
            glam::EulerRot::YXZ,
            head_ry * 0.04,
            head_rx * 0.04,
            0.0,
        );
        pose.insert(upper_chest, base * follow);
    }

    // chest: 2.4% (composes on existing breathing)
    if let Some(&chest) = model.bone_to_node.get("chest") {
        let rest = model.rest_rotations[chest];
        let base = *pose.get(&chest).unwrap_or(&rest);
        let follow = Quat::from_euler(
            glam::EulerRot::YXZ,
            head_ry * 0.024,
            head_rx * 0.024,
            0.0,
        );
        pose.insert(chest, base * follow);
    }

    // spine: 1.2% (composes on existing breathing)
    if let Some(&spine) = model.bone_to_node.get("spine") {
        let rest = model.rest_rotations[spine];
        let base = *pose.get(&spine).unwrap_or(&rest);
        let follow = Quat::from_euler(
            glam::EulerRot::YXZ,
            head_ry * 0.012,
            head_rx * 0.012,
            0.0,
        );
        pose.insert(spine, base * follow);
    }

    // hips: 0.4% yaw only (composes on existing sway)
    if let Some(&hips) = model.bone_to_node.get("hips") {
        let rest = model.rest_rotations[hips];
        let base = *pose.get(&hips).unwrap_or(&rest);
        let follow = Quat::from_euler(glam::EulerRot::YXZ, head_ry * 0.004, 0.0, 0.0);
        pose.insert(hips, base * follow);
    }
}

/// Apply procedural arm idle sway and head counterbalance.
fn apply_arm_animation(
    pose: &mut HashMap<usize, Quat>,
    model: &VrmModel,
    time: f32,
    head_ry: f32,
    is_speaking: bool,
) {
    let speak_mult = if is_speaking { 1.5 } else { 1.0 };
    let arm_amp = 0.015 * speak_mult;

    // Upper arms: subtle X-axis pendulum, left/right out of phase
    let left_swing = (time * 0.8 * PI).sin() * arm_amp;
    let right_swing = (time * 0.8 * PI + PI * 0.5).sin() * arm_amp;

    // Head counterbalance: upper arms get tiny counter-yaw
    let counter_yaw = -head_ry * 0.05;

    if let Some(&l_upper) = model.bone_to_node.get("leftUpperArm") {
        let rest = model.rest_rotations[l_upper];
        let base = *pose.get(&l_upper).unwrap_or(&rest);
        let sway = Quat::from_euler(glam::EulerRot::YXZ, counter_yaw, left_swing, 0.0);
        pose.insert(l_upper, base * sway);
    }
    if let Some(&r_upper) = model.bone_to_node.get("rightUpperArm") {
        let rest = model.rest_rotations[r_upper];
        let base = *pose.get(&r_upper).unwrap_or(&rest);
        let sway = Quat::from_euler(glam::EulerRot::YXZ, counter_yaw, right_swing, 0.0);
        pose.insert(r_upper, base * sway);
    }

    // Lower arms: lag behind at half amplitude
    let left_lower = (time * 0.8 * PI - 0.3).sin() * arm_amp * 0.5;
    let right_lower = (time * 0.8 * PI + PI * 0.5 - 0.3).sin() * arm_amp * 0.5;

    if let Some(&l_lower) = model.bone_to_node.get("leftLowerArm") {
        let rest = model.rest_rotations[l_lower];
        let base = *pose.get(&l_lower).unwrap_or(&rest);
        let sway = Quat::from_axis_angle(glam::Vec3::X, left_lower);
        pose.insert(l_lower, base * sway);
    }
    if let Some(&r_lower) = model.bone_to_node.get("rightLowerArm") {
        let rest = model.rest_rotations[r_lower];
        let base = *pose.get(&r_lower).unwrap_or(&rest);
        let sway = Quat::from_axis_angle(glam::Vec3::X, right_lower);
        pose.insert(r_lower, base * sway);
    }

    // Subtle wrist micro-movement
    let wrist_angle = (time * 1.1 * PI).sin() * 0.005;
    for hand_name in &["leftHand", "rightHand"] {
        if let Some(&node) = model.bone_to_node.get(*hand_name) {
            let rest = model.rest_rotations[node];
            let base = *pose.get(&node).unwrap_or(&rest);
            let micro = Quat::from_axis_angle(glam::Vec3::X, wrist_angle);
            pose.insert(node, base * micro);
        }
    }
}

/// Apply natural finger rest curl to all 30 finger bones.
fn apply_finger_curl(pose: &mut HashMap<usize, Quat>, model: &VrmModel, time: f32) {
    // Finger bone definitions: (name, curl_angle, is_thumb)
    // Proximal ~15° (0.26 rad), Intermediate ~20° (0.35 rad), Distal ~10° (0.17 rad)
    struct FingerBone {
        name: &'static str,
        curl: f32,
        is_thumb: bool,
    }

    let finger_bones: &[FingerBone] = &[
        // Left thumb (VRM 1.0 naming)
        FingerBone { name: "leftThumbMetacarpal", curl: 0.26, is_thumb: true },
        FingerBone { name: "leftThumbProximal", curl: 0.35, is_thumb: true },
        FingerBone { name: "leftThumbDistal", curl: 0.17, is_thumb: true },
        // Left index
        FingerBone { name: "leftIndexProximal", curl: 0.26, is_thumb: false },
        FingerBone { name: "leftIndexIntermediate", curl: 0.35, is_thumb: false },
        FingerBone { name: "leftIndexDistal", curl: 0.17, is_thumb: false },
        // Left middle
        FingerBone { name: "leftMiddleProximal", curl: 0.26, is_thumb: false },
        FingerBone { name: "leftMiddleIntermediate", curl: 0.35, is_thumb: false },
        FingerBone { name: "leftMiddleDistal", curl: 0.17, is_thumb: false },
        // Left ring
        FingerBone { name: "leftRingProximal", curl: 0.26, is_thumb: false },
        FingerBone { name: "leftRingIntermediate", curl: 0.35, is_thumb: false },
        FingerBone { name: "leftRingDistal", curl: 0.17, is_thumb: false },
        // Left little
        FingerBone { name: "leftLittleProximal", curl: 0.26, is_thumb: false },
        FingerBone { name: "leftLittleIntermediate", curl: 0.35, is_thumb: false },
        FingerBone { name: "leftLittleDistal", curl: 0.17, is_thumb: false },
        // Right thumb (VRM 1.0 naming)
        FingerBone { name: "rightThumbMetacarpal", curl: 0.26, is_thumb: true },
        FingerBone { name: "rightThumbProximal", curl: 0.35, is_thumb: true },
        FingerBone { name: "rightThumbDistal", curl: 0.17, is_thumb: true },
        // Right index
        FingerBone { name: "rightIndexProximal", curl: 0.26, is_thumb: false },
        FingerBone { name: "rightIndexIntermediate", curl: 0.35, is_thumb: false },
        FingerBone { name: "rightIndexDistal", curl: 0.17, is_thumb: false },
        // Right middle
        FingerBone { name: "rightMiddleProximal", curl: 0.26, is_thumb: false },
        FingerBone { name: "rightMiddleIntermediate", curl: 0.35, is_thumb: false },
        FingerBone { name: "rightMiddleDistal", curl: 0.17, is_thumb: false },
        // Right ring
        FingerBone { name: "rightRingProximal", curl: 0.26, is_thumb: false },
        FingerBone { name: "rightRingIntermediate", curl: 0.35, is_thumb: false },
        FingerBone { name: "rightRingDistal", curl: 0.17, is_thumb: false },
        // Right little
        FingerBone { name: "rightLittleProximal", curl: 0.26, is_thumb: false },
        FingerBone { name: "rightLittleIntermediate", curl: 0.35, is_thumb: false },
        FingerBone { name: "rightLittleDistal", curl: 0.17, is_thumb: false },
    ];

    for (i, bone) in finger_bones.iter().enumerate() {
        // Per-finger micro-movement sine wave
        let micro = (time * 0.6 + i as f32 * 0.7).sin() * 0.008;
        let angle = bone.curl + micro;

        // Thumbs curl on Z axis, others on X
        let axis = if bone.is_thumb {
            glam::Vec3::Z
        } else {
            glam::Vec3::X
        };

        if let Some(&node) = model.bone_to_node.get(bone.name) {
            let rest = model.rest_rotations[node];
            let curl_q = Quat::from_axis_angle(axis, angle);
            pose.insert(node, rest * curl_q);
        }
    }
}

/// Compute animated bone rotations for the current frame.
///
/// `time`: elapsed time in seconds (monotonic)
/// `head_rotation_deg`: [pitch, yaw, roll] from tracking (degrees), or None for idle
/// `is_speaking`: whether the avatar is currently speaking
/// `tuning`: tracking sensitivity and scaling parameters
pub fn animated_pose(
    model: &VrmModel,
    time: f32,
    head_rotation_deg: Option<[f32; 3]>,
    is_speaking: bool,
    tuning: &crate::config::TrackingTuning,
) -> HashMap<usize, Quat> {
    let mut pose = idle_pose(model);

    apply_breathing(&mut pose, model, time);
    apply_idle_sway(&mut pose, model, time);

    let (head_rx, head_ry, _head_rz) =
        apply_head_tracking(&mut pose, model, time, head_rotation_deg, is_speaking, tuning);

    apply_body_follow(&mut pose, model, head_rx, head_ry);
    apply_arm_animation(&mut pose, model, time, head_ry, is_speaking);
    apply_finger_curl(&mut pose, model, time);

    pose
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_animated_pose_no_crash() {
        let model_path = "assets/default/model.glb";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let model = VrmModel::load(model_path).unwrap();

        let tuning = crate::config::TrackingTuning::default();

        // Should not panic with any input combination
        let _pose1 = animated_pose(&model, 0.0, None, false, &tuning);
        let _pose2 = animated_pose(&model, 1.5, Some([10.0, -5.0, 2.0]), true, &tuning);
        let _pose3 = animated_pose(&model, 100.0, None, true, &tuning);
    }

    #[test]
    fn test_animated_pose_bone_count() {
        let model_path = "assets/default/model.glb";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let model = VrmModel::load(model_path).unwrap();
        let tuning = crate::config::TrackingTuning::default();

        let pose = animated_pose(&model, 1.0, Some([5.0, -3.0, 1.0]), true, &tuning);

        // Should animate more than just head/neck/shoulders/spine/chest/hips
        assert!(
            pose.len() > 7,
            "Expected more than 7 animated bones, got {}",
            pose.len()
        );
    }

    #[test]
    fn test_body_follow_responds_to_head() {
        let model_path = "assets/default/model.glb";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let model = VrmModel::load(model_path).unwrap();
        let tuning = crate::config::TrackingTuning::default();

        let pose_neutral = animated_pose(&model, 1.0, Some([0.0, 0.0, 0.0]), false, &tuning);
        let pose_turned = animated_pose(&model, 1.0, Some([0.0, 30.0, 0.0]), false, &tuning);

        // Upper chest should differ when head is turned
        if let Some(&uc_node) = model.bone_to_node.get("upperChest") {
            if let (Some(&q_neutral), Some(&q_turned)) =
                (pose_neutral.get(&uc_node), pose_turned.get(&uc_node))
            {
                let diff = q_neutral.angle_between(q_turned);
                assert!(
                    diff > 0.001,
                    "upperChest should respond to head turn, angle diff: {}",
                    diff
                );
            }
        }
    }

    #[test]
    fn test_speaking_increases_arm_sway() {
        let model_path = "assets/default/model.glb";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let model = VrmModel::load(model_path).unwrap();
        let tuning = crate::config::TrackingTuning::default();

        // Test at a time where the sine is near peak to see amplitude difference
        let pose_quiet = animated_pose(&model, 0.625, None, false, &tuning);
        let pose_speak = animated_pose(&model, 0.625, None, true, &tuning);

        if let Some(&l_upper) = model.bone_to_node.get("leftUpperArm") {
            if let (Some(&q_quiet), Some(&q_speak)) =
                (pose_quiet.get(&l_upper), pose_speak.get(&l_upper))
            {
                // Speaking pose should differ from quiet (larger sway + micro-nod effect)
                let diff = q_quiet.angle_between(q_speak);
                assert!(
                    diff > 0.0001,
                    "Arm sway should differ when speaking, diff: {}",
                    diff
                );
            }
        }
    }
}
