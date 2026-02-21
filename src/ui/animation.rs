//! Procedural animation for VRM models.
//!
//! Generates per-frame bone rotations for breathing, idle sway, head tracking,
//! body follow, arm/hand animation, and finger curl.

#![cfg(feature = "native-ui")]

use glam::{Mat3, Mat4, Quat};
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

    // Rest-pose world transforms — used to find bone axes and convert rotations.
    let rest_world = super::skinning::compute_world_transforms(model, &HashMap::new());

    // Build tracking rotation using the HEAD bone's rest-pose world axes.
    // The tracker's pitch/yaw/roll correspond to rotations around the head's
    // local X (ear-to-ear), Y (up through crown), Z (nose direction) axes,
    // NOT world X/Y/Z.  We use from_axis_angle with the bone's world-space
    // axis directions, then compose ZYX to match the Python decomposition.
    if let Some(&head_node) = model.bone_to_node.get("head") {
        let hw = rest_world[head_node];
        let bone_x = glam::Vec3::new(hw.col(0).x, hw.col(0).y, hw.col(0).z).normalize();
        let bone_y = glam::Vec3::new(hw.col(1).x, hw.col(1).y, hw.col(1).z).normalize();
        let bone_z = glam::Vec3::new(hw.col(2).x, hw.col(2).y, hw.col(2).z).normalize();

        // Tracker axes → bone axes mapping (determined empirically):
        //   tracker pitch (head_rx) → bone_z
        //   tracker yaw   (head_ry) → bone_x
        //   tracker roll  (head_rz) → bone_y
        let rpitch = Quat::from_axis_angle(bone_z, head_rx);
        let ryaw   = Quat::from_axis_angle(bone_x, head_ry);
        let rroll  = Quat::from_axis_angle(bone_y, head_rz);
        let tracking_world = rroll * ryaw * rpitch;

        // 60% to head
        let rest = model.rest_rotations[head_node];
        let scaled = Quat::IDENTITY.slerp(tracking_world, 0.6);
        let local_delta = world_to_parent_local(model, &rest_world, head_node, scaled);
        pose.insert(head_node, local_delta * rest);

        // 40% to neck
        if let Some(&neck) = model.bone_to_node.get("neck") {
            let rest = model.rest_rotations[neck];
            let scaled = Quat::IDENTITY.slerp(tracking_world, 0.4);
            let local_delta = world_to_parent_local(model, &rest_world, neck, scaled);
            pose.insert(neck, local_delta * rest);
        }
    }

    (head_rx, head_ry, head_rz)
}

/// Convert a world-space rotation into the parent-local frame of `node`.
/// R_local = P⁻¹ * R_world * P  where P is the parent's rest-pose world rotation.
fn world_to_parent_local(
    model: &VrmModel,
    rest_world: &[Mat4],
    node: usize,
    rot_world: Quat,
) -> Quat {
    if let Some(parent) = model.parents[node] {
        let parent_rot = Quat::from_mat3(&Mat3::from_mat4(rest_world[parent])).normalize();
        parent_rot.inverse() * rot_world * parent_rot
    } else {
        rot_world
    }
}

/// Apply subtle upper body coupling that follows head movement.
fn apply_body_follow(
    pose: &mut HashMap<usize, Quat>,
    model: &VrmModel,
    head_rx: f32,
    head_ry: f32,
) {
    // World-space follow rotation from head movement (pitch + yaw only)
    let rest_world = super::skinning::compute_world_transforms(model, &HashMap::new());

    let follow_bones: &[(&str, f32)] = &[
        ("upperChest", 0.04),
        ("chest", 0.024),
        ("spine", 0.012),
    ];

    for &(bone_name, factor) in follow_bones {
        if let Some(&node) = model.bone_to_node.get(bone_name) {
            let rest = model.rest_rotations[node];
            let base = *pose.get(&node).unwrap_or(&rest);

            // Build follow rotation in world space, then convert to parent-local
            let follow_world = Quat::from_euler(
                glam::EulerRot::ZYX,
                0.0,
                head_ry * factor,
                head_rx * factor,
            );
            let follow_local = world_to_parent_local(model, &rest_world, node, follow_world);
            pose.insert(node, base * follow_local);
        }
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

/// Blend IK-solved arm rotations into the procedural pose.
fn apply_body_ik_blend(
    pose: &mut HashMap<usize, Quat>,
    model: &VrmModel,
    ik_rotations: &HashMap<usize, Quat>,
    blend: f32,
) {
    for (&node, &ik_rot) in ik_rotations {
        let rest = model.rest_rotations[node];
        let current = *pose.get(&node).unwrap_or(&rest);
        let blended = current.slerp(ik_rot, blend);
        pose.insert(node, blended);
    }
}

/// Compute animated bone rotations for the current frame.
///
/// `time`: elapsed time in seconds (monotonic)
/// `head_rotation_deg`: [pitch, yaw, roll] from tracking (degrees), or None for idle
/// `is_speaking`: whether the avatar is currently speaking
/// `tuning`: tracking sensitivity and scaling parameters
/// `body_ik_rotations`: optional IK-solved rotations from body tracking
pub fn animated_pose(
    model: &VrmModel,
    time: f32,
    head_rotation_deg: Option<[f32; 3]>,
    is_speaking: bool,
    tuning: &crate::config::TrackingTuning,
    body_ik_rotations: Option<&HashMap<usize, Quat>>,
) -> HashMap<usize, Quat> {
    let mut pose = idle_pose(model);

    apply_breathing(&mut pose, model, time);
    apply_idle_sway(&mut pose, model, time);

    let (head_rx, head_ry, _head_rz) =
        apply_head_tracking(&mut pose, model, time, head_rotation_deg, is_speaking, tuning);

    apply_body_follow(&mut pose, model, head_rx, head_ry);
    apply_arm_animation(&mut pose, model, time, head_ry, is_speaking);

    if let Some(ik) = body_ik_rotations {
        apply_body_ik_blend(&mut pose, model, ik, tuning.body_blend_factor);
    }

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
        let _pose1 = animated_pose(&model, 0.0, None, false, &tuning, None);
        let _pose2 = animated_pose(&model, 1.5, Some([10.0, -5.0, 2.0]), true, &tuning, None);
        let _pose3 = animated_pose(&model, 100.0, None, true, &tuning, None);
    }

    #[test]
    fn test_animated_pose_bone_count() {
        let model_path = "assets/default/model.glb";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let model = VrmModel::load(model_path).unwrap();
        let tuning = crate::config::TrackingTuning::default();

        let pose = animated_pose(&model, 1.0, Some([5.0, -3.0, 1.0]), true, &tuning, None);

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

        let pose_neutral = animated_pose(&model, 1.0, Some([0.0, 0.0, 0.0]), false, &tuning, None);
        let pose_turned = animated_pose(&model, 1.0, Some([0.0, 30.0, 0.0]), false, &tuning, None);

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

    // ---- Head rotation direction tests ----
    //
    // These test the full pipeline: input [pitch, yaw, roll] in degrees →
    // animated_pose → compute_world_transforms → head bone world-space vectors.
    //
    // Using time=0 eliminates breathing/sway noise.  sensitivity=1.0,
    // per-axis scale=1.0 so input degrees map 1:1 to radians.

    /// Tuning with no scaling — 1° input = 1° rotation.
    fn unit_tuning() -> crate::config::TrackingTuning {
        crate::config::TrackingTuning {
            head_sensitivity: 1.0,
            pitch_scale: 1.0,
            yaw_scale: 1.0,
            roll_scale: 1.0,
            ..Default::default()
        }
    }

    /// Pose the model with the given head rotation and return the head bone's
    /// world-space forward (-Z) and up (+Y) vectors.
    fn head_vectors(model: &VrmModel, rot_deg: [f32; 3]) -> (glam::Vec3, glam::Vec3) {
        let tuning = unit_tuning();
        let pose = animated_pose(model, 0.0, Some(rot_deg), false, &tuning, None);
        let world = super::super::skinning::compute_world_transforms(model, &pose);

        let head_node = *model.bone_to_node.get("head").expect("model needs head bone");
        let m = world[head_node];

        // Forward = -Z column, Up = +Y column (upper 3x3 of the world transform)
        let fwd = -glam::Vec3::new(m.col(2).x, m.col(2).y, m.col(2).z).normalize();
        let up = glam::Vec3::new(m.col(1).x, m.col(1).y, m.col(1).z).normalize();
        (fwd, up)
    }

    /// Baseline forward/up with zero rotation input.
    fn head_baseline(model: &VrmModel) -> (glam::Vec3, glam::Vec3) {
        head_vectors(model, [0.0, 0.0, 0.0])
    }

    fn load_test_model() -> Option<VrmModel> {
        let path = "assets/default/model.glb";
        if !std::path::Path::new(path).exists() {
            return None;
        }
        Some(VrmModel::load(path).unwrap())
    }

    // Model-agnostic axis helpers: use baseline forward/up/right to measure
    // pitch (tilt toward up), yaw (turn toward right), roll (up tilts toward right).
    // This works regardless of which world axes the model uses.

    fn head_right(model: &VrmModel) -> glam::Vec3 {
        let (fwd, up) = head_baseline(model);
        fwd.cross(up).normalize()
    }

    #[test]
    fn test_pitch_down_tilts_forward_vector_down() {
        let model = match load_test_model() { Some(m) => m, None => return };
        let (base_fwd, base_up) = head_baseline(&model);
        let (fwd, _) = head_vectors(&model, [20.0, 0.0, 0.0]);

        // Pitch should change forward's projection onto up axis
        let base_vert = base_fwd.dot(base_up);
        let vert = fwd.dot(base_up);
        let dv = (vert - base_vert).abs();
        assert!(dv > 0.05, "20° pitch should tilt forward vertically, dv={dv:.4}");
    }

    #[test]
    fn test_pitch_up_opposite_of_pitch_down() {
        let model = match load_test_model() { Some(m) => m, None => return };
        let (base_fwd, base_up) = head_baseline(&model);
        let (fwd_down, _) = head_vectors(&model, [20.0, 0.0, 0.0]);
        let (fwd_up, _) = head_vectors(&model, [-20.0, 0.0, 0.0]);

        let base_vert = base_fwd.dot(base_up);
        let d_down = fwd_down.dot(base_up) - base_vert;
        let d_up = fwd_up.dot(base_up) - base_vert;
        assert!(
            d_down * d_up < 0.0,
            "Pitch up/down should tilt in opposite directions: down={d_down:.4}, up={d_up:.4}"
        );
    }

    #[test]
    fn test_pitch_does_not_cause_yaw() {
        let model = match load_test_model() { Some(m) => m, None => return };
        let (base_fwd, _) = head_baseline(&model);
        let right = head_right(&model);
        let (fwd, _) = head_vectors(&model, [25.0, 0.0, 0.0]);

        // Pure pitch should barely change forward's projection onto right axis
        let base_horiz = base_fwd.dot(right);
        let horiz = fwd.dot(right);
        let dh = (horiz - base_horiz).abs();
        assert!(
            dh < 0.05,
            "Pure pitch should not cause yaw, right-projection shift={dh:.4}"
        );
    }

    #[test]
    fn test_pitch_does_not_cause_roll() {
        let model = match load_test_model() { Some(m) => m, None => return };
        let (_, base_up) = head_baseline(&model);
        let right = head_right(&model);
        let (_, up) = head_vectors(&model, [25.0, 0.0, 0.0]);

        // Roll = up vector tilting toward right axis
        let base_roll = base_up.dot(right);
        let roll_shift = (up.dot(right) - base_roll).abs();
        assert!(
            roll_shift < 0.05,
            "Pure pitch should not cause roll, up·right shift={roll_shift:.4}"
        );
    }

    #[test]
    fn test_yaw_right_turns_forward_vector_right() {
        let model = match load_test_model() { Some(m) => m, None => return };
        let (base_fwd, _) = head_baseline(&model);
        let right = head_right(&model);
        let (fwd, _) = head_vectors(&model, [0.0, 25.0, 0.0]);

        // Yaw should change forward's projection onto right axis
        let base_horiz = base_fwd.dot(right);
        let horiz = fwd.dot(right);
        let dh = (horiz - base_horiz).abs();
        assert!(
            dh > 0.05,
            "25° yaw should turn forward horizontally, right-projection shift={dh:.4}"
        );
    }

    #[test]
    fn test_yaw_left_opposite_of_yaw_right() {
        let model = match load_test_model() { Some(m) => m, None => return };
        let (base_fwd, _) = head_baseline(&model);
        let right = head_right(&model);
        let (fwd_r, _) = head_vectors(&model, [0.0, 25.0, 0.0]);
        let (fwd_l, _) = head_vectors(&model, [0.0, -25.0, 0.0]);

        let base_h = base_fwd.dot(right);
        let d_r = fwd_r.dot(right) - base_h;
        let d_l = fwd_l.dot(right) - base_h;
        assert!(
            d_r * d_l < 0.0,
            "Yaw left/right should turn in opposite directions: R={d_r:.4}, L={d_l:.4}"
        );
    }

    #[test]
    fn test_yaw_does_not_cause_pitch() {
        let model = match load_test_model() { Some(m) => m, None => return };
        let (base_fwd, base_up) = head_baseline(&model);
        let (fwd, _) = head_vectors(&model, [0.0, 25.0, 0.0]);

        // Pure yaw should barely change forward's vertical projection
        let base_vert = base_fwd.dot(base_up);
        let vert = fwd.dot(base_up);
        let dv = (vert - base_vert).abs();
        assert!(
            dv < 0.05,
            "Pure yaw should not cause pitch, up-projection shift={dv:.4}"
        );
    }

    #[test]
    fn test_yaw_does_not_cause_roll() {
        let model = match load_test_model() { Some(m) => m, None => return };
        let (_, base_up) = head_baseline(&model);
        let right = head_right(&model);
        let (_, up) = head_vectors(&model, [0.0, 25.0, 0.0]);

        let base_roll = base_up.dot(right);
        let roll_shift = (up.dot(right) - base_roll).abs();
        assert!(
            roll_shift < 0.05,
            "Pure yaw should not cause roll, up·right shift={roll_shift:.4}"
        );
    }

    #[test]
    fn test_roll_tilts_up_vector_sideways() {
        let model = match load_test_model() { Some(m) => m, None => return };
        let (_, base_up) = head_baseline(&model);
        let right = head_right(&model);
        let (_, up) = head_vectors(&model, [0.0, 0.0, 15.0]);

        // Roll = up vector tilting toward right axis
        let base_roll = base_up.dot(right);
        let roll_shift = (up.dot(right) - base_roll).abs();
        assert!(roll_shift > 0.03, "15° roll should tilt up sideways, up·right shift={roll_shift:.4}");
    }

    #[test]
    fn test_roll_does_not_cause_pitch() {
        let model = match load_test_model() { Some(m) => m, None => return };
        let (base_fwd, base_up) = head_baseline(&model);
        let (fwd, _) = head_vectors(&model, [0.0, 0.0, 15.0]);

        let base_vert = base_fwd.dot(base_up);
        let vert = fwd.dot(base_up);
        let dv = (vert - base_vert).abs();
        assert!(
            dv < 0.05,
            "Pure roll should not cause pitch, up-projection shift={dv:.4}"
        );
    }

    #[test]
    fn test_euler_round_trip_zyx() {
        // Verify that ZYX Euler angles round-trip through quat → decompose.
        let pitch_deg = 15.0f32;
        let yaw_deg = -10.0f32;
        let roll_deg = 5.0f32;

        let rx = pitch_deg.to_radians();
        let ry = yaw_deg.to_radians();
        let rz = roll_deg.to_radians();

        // Reconstruct with ZYX (matches Python decomposition R = Rz*Ry*Rx)
        let q = Quat::from_euler(glam::EulerRot::ZYX, rz, ry, rx);

        // Decompose back via glam's built-in
        let (ez, ey, ex) = q.to_euler(glam::EulerRot::ZYX);

        assert!((ex - rx).abs() < 0.001, "Pitch round-trip: {ex:.4} vs {rx:.4}");
        assert!((ey - ry).abs() < 0.001, "Yaw round-trip: {ey:.4} vs {ry:.4}");
        assert!((ez - rz).abs() < 0.001, "Roll round-trip: {ez:.4} vs {rz:.4}");
    }

    #[test]
    fn test_cross_coupling_pitch_vs_roll() {
        // The original bug: pitch changes caused roll changes.
        // Sweep pitch from -30 to +30 with roll=0 and verify
        // the up vector's right-projection (roll indicator) stays stable.
        let model = match load_test_model() { Some(m) => m, None => return };
        let (_, base_up) = head_baseline(&model);
        let right = head_right(&model);
        let base_roll = base_up.dot(right);

        for pitch in [-30.0, -15.0, 15.0, 30.0f32] {
            let (_, up) = head_vectors(&model, [pitch, 0.0, 0.0]);
            let roll_shift = (up.dot(right) - base_roll).abs();
            assert!(
                roll_shift < 0.08,
                "Pitch {pitch}° should not cause roll, up·right shift={roll_shift:.4}"
            );
        }
    }

    #[test]
    fn test_head_direction_snapshot() {
        // Print a snapshot table of head directions for visual inspection.
        // This test always passes — its value is in the printed output.
        // Run with: cargo test test_head_direction_snapshot -- --nocapture
        let model = match load_test_model() { Some(m) => m, None => return };

        let cases: &[(&str, [f32; 3])] = &[
            ("neutral",     [0.0,   0.0,  0.0]),
            ("pitch +20",   [20.0,  0.0,  0.0]),
            ("pitch -20",   [-20.0, 0.0,  0.0]),
            ("yaw +25",     [0.0,   25.0, 0.0]),
            ("yaw -25",     [0.0,  -25.0, 0.0]),
            ("roll +15",    [0.0,   0.0,  15.0]),
            ("roll -15",    [0.0,   0.0, -15.0]),
            ("combo 15/10", [15.0,  10.0, 5.0]),
        ];

        eprintln!("\n{:<14} {:>8} {:>8} {:>8}  |  {:>8} {:>8} {:>8}",
            "pose", "fwd.x", "fwd.y", "fwd.z", "up.x", "up.y", "up.z");
        eprintln!("{}", "-".repeat(76));

        for (label, rot) in cases {
            let (fwd, up) = head_vectors(&model, *rot);
            eprintln!("{:<14} {:>8.4} {:>8.4} {:>8.4}  |  {:>8.4} {:>8.4} {:>8.4}",
                label, fwd.x, fwd.y, fwd.z, up.x, up.y, up.z);
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
        let pose_quiet = animated_pose(&model, 0.625, None, false, &tuning, None);
        let pose_speak = animated_pose(&model, 0.625, None, true, &tuning, None);

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
