//! Procedural animation for VRM models.
//!
//! Generates per-frame bone rotations for breathing, idle sway, head tracking,
//! and speaking micro-nods. Matches the behavior from `skinning.py`.

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

/// Compute animated bone rotations for the current frame.
///
/// `time`: elapsed time in seconds (monotonic)
/// `head_rotation_deg`: [pitch, yaw, roll] from tracking (degrees), or None for idle
/// `is_speaking`: whether the avatar is currently speaking
pub fn animated_pose(
    model: &VrmModel,
    time: f32,
    head_rotation_deg: Option<[f32; 3]>,
    is_speaking: bool,
) -> HashMap<usize, Quat> {
    let mut pose = idle_pose(model);

    // Breathing: subtle chest expansion (~0.6 Hz)
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

    // Idle sway: very subtle lateral body movement
    let sway_phase = (time * 0.4 * PI).sin();
    let sway_angle = sway_phase * 0.006;

    if let Some(&hips) = model.bone_to_node.get("hips") {
        let rest = model.rest_rotations[hips];
        let base = *pose.get(&hips).unwrap_or(&rest);
        let sway = Quat::from_axis_angle(glam::Vec3::Z, sway_angle);
        pose.insert(hips, base * sway);
    }

    // Head/neck animation
    let (mut head_rx, mut head_ry, mut head_rz) = (0.0f32, 0.0f32, 0.0f32);

    if let Some([pitch, yaw, roll]) = head_rotation_deg {
        // Convert from degrees to radians, apply 60/40 head/neck split
        head_rx = pitch.to_radians();
        head_ry = yaw.to_radians();
        head_rz = roll.to_radians();
    } else {
        // Subtle idle head movement
        head_rx += (time * 0.7).sin() * 0.015;
        head_ry += (time * 0.5).sin() * 0.01;
    }

    // Speaking: subtle micro-nods
    if is_speaking {
        let speak_nod = (time * 6.0).sin() * 0.02;
        head_rx += speak_nod;
    }

    // Apply head rotation (60% to head)
    if let Some(&head) = model.bone_to_node.get("head") {
        let rest = model.rest_rotations[head];
        let hq = Quat::from_axis_angle(glam::Vec3::X, head_rx * 0.6)
            * Quat::from_axis_angle(glam::Vec3::Y, head_ry * 0.6)
            * Quat::from_axis_angle(glam::Vec3::Z, head_rz * 0.6);
        pose.insert(head, rest * hq);
    }

    // Apply neck rotation (40% to neck)
    if let Some(&neck) = model.bone_to_node.get("neck") {
        let rest = model.rest_rotations[neck];
        let nq = Quat::from_axis_angle(glam::Vec3::X, head_rx * 0.4)
            * Quat::from_axis_angle(glam::Vec3::Y, head_ry * 0.4);
        pose.insert(neck, rest * nq);
    }

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

        // Should not panic with any input combination
        let _pose1 = animated_pose(&model, 0.0, None, false);
        let _pose2 = animated_pose(&model, 1.5, Some([10.0, -5.0, 2.0]), true);
        let _pose3 = animated_pose(&model, 100.0, None, true);
    }
}
