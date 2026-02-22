//! Verlet-based spring bone simulator for VRM secondary motion (hair, cloth, etc.).
//!
//! Implements the VRMC_springBone-1.0 algorithm: verlet integration with stiffness,
//! gravity, drag, length constraints, and sphere/capsule collision.

#![cfg(feature = "native-ui")]

use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;

use super::skinning;
use super::vrm_loader::{ColliderShape, VrmModel};

/// Runtime state for a single spring joint.
struct JointState {
    /// Node index in the glTF skeleton
    node: usize,
    /// Previous tail position (world space)
    prev_tail: Vec3,
    /// Current tail position (world space)
    current_tail: Vec3,
    /// Rest-pose bone length (distance from this joint to child)
    bone_length: f32,
    /// Rest-pose bone axis in local space (normalized direction to child)
    bone_axis: Vec3,
    /// Physics parameters
    stiffness: f32,
    gravity_power: f32,
    gravity_dir: Vec3,
    drag_force: f32,
    hit_radius: f32,
}

/// Runtime state for a spring chain.
struct ChainState {
    joints: Vec<JointState>,
    /// Collider group indices this chain interacts with
    collider_group_indices: Vec<usize>,
}

/// Resolved collider in world space.
struct WorldCollider {
    shape: WorldColliderShape,
}

enum WorldColliderShape {
    Sphere { center: Vec3, radius: f32 },
    Capsule { start: Vec3, end: Vec3, radius: f32 },
}

/// Spring bone physics simulator.
pub struct SpringBoneSimulator {
    chains: Vec<ChainState>,
    /// Collider definitions (node + local shape)
    collider_nodes: Vec<(usize, ColliderShapeCopy)>,
    /// Collider groups: group index → list of collider indices
    collider_groups: Vec<Vec<usize>>,
}

/// Local copy of collider shape data (avoids borrowing VrmModel during simulation).
enum ColliderShapeCopy {
    Sphere { offset: Vec3, radius: f32 },
    Capsule { offset: Vec3, tail: Vec3, radius: f32 },
}

impl SpringBoneSimulator {
    /// Build the simulator from a loaded VRM model.
    /// Returns `None` if the model has no spring bones.
    pub fn new(model: &VrmModel) -> Option<Self> {
        if model.spring_chains.is_empty() {
            return None;
        }

        // Compute rest-pose world transforms
        let rest_world = skinning::compute_world_transforms(model, &HashMap::new());

        // Copy collider definitions
        let collider_nodes: Vec<(usize, ColliderShapeCopy)> = model
            .spring_colliders
            .iter()
            .map(|c| {
                let shape = match &c.shape {
                    ColliderShape::Sphere { offset, radius } => ColliderShapeCopy::Sphere {
                        offset: *offset,
                        radius: *radius,
                    },
                    ColliderShape::Capsule {
                        offset,
                        tail,
                        radius,
                    } => ColliderShapeCopy::Capsule {
                        offset: *offset,
                        tail: *tail,
                        radius: *radius,
                    },
                };
                (c.node, shape)
            })
            .collect();

        // Copy collider groups
        let collider_groups: Vec<Vec<usize>> = model
            .collider_groups
            .iter()
            .map(|g| g.collider_indices.clone())
            .collect();

        // Build chain states
        let mut chains = Vec::new();
        for chain in &model.spring_chains {
            let mut joint_states = Vec::new();

            for (i, joint) in chain.joints.iter().enumerate() {
                // Determine the child position to compute bone_length and bone_axis
                let joint_world_pos = rest_world[joint.node].col(3).truncate();

                let (bone_length, bone_axis) = if i + 1 < chain.joints.len() {
                    // Child is the next joint in the chain
                    let child_node = chain.joints[i + 1].node;
                    let child_world_pos = rest_world[child_node].col(3).truncate();
                    let world_dir = child_world_pos - joint_world_pos;
                    let length = world_dir.length();
                    if length < 1e-6 {
                        continue;
                    }
                    // Convert world direction to local space of this joint
                    let joint_rot = quat_from_mat4(&rest_world[joint.node]);
                    let local_dir = joint_rot.inverse() * world_dir.normalize();
                    (length, local_dir)
                } else {
                    // Last joint in chain — use the first child node from the skeleton,
                    // or synthesize a small tail along the parent direction.
                    let child_info = find_child_node(model, joint.node, &rest_world);
                    match child_info {
                        Some((length, local_axis)) => (length, local_axis),
                        None => {
                            // Synthesize: extend 7cm along parent-to-this direction
                            if let Some(parent) = model.parents[joint.node] {
                                let parent_pos = rest_world[parent].col(3).truncate();
                                let dir = (joint_world_pos - parent_pos).normalize_or_zero();
                                if dir.length_squared() < 0.5 {
                                    continue;
                                }
                                let joint_rot = quat_from_mat4(&rest_world[joint.node]);
                                let local_dir = joint_rot.inverse() * dir;
                                (0.07, local_dir)
                            } else {
                                continue;
                            }
                        }
                    }
                };

                // Tail position = joint world pos + direction * length
                let joint_rot = quat_from_mat4(&rest_world[joint.node]);
                let tail_world = joint_world_pos + joint_rot * bone_axis * bone_length;

                joint_states.push(JointState {
                    node: joint.node,
                    prev_tail: tail_world,
                    current_tail: tail_world,
                    bone_length,
                    bone_axis,
                    stiffness: joint.stiffness,
                    gravity_power: joint.gravity_power,
                    gravity_dir: joint.gravity_dir,
                    drag_force: joint.drag_force,
                    hit_radius: joint.hit_radius,
                });
            }

            if !joint_states.is_empty() {
                chains.push(ChainState {
                    joints: joint_states,
                    collider_group_indices: chain.collider_group_indices.clone(),
                });
            }
        }

        if chains.is_empty() {
            return None;
        }

        tracing::info!(
            "Spring bone simulator: {} chains, {} colliders",
            chains.len(),
            collider_nodes.len()
        );

        Some(Self {
            chains,
            collider_nodes,
            collider_groups,
        })
    }

    /// Step the simulation forward by `dt` seconds.
    ///
    /// `world_transforms` are the current frame's world transforms (after animation/IK).
    /// `gravity_scale` multiplies the per-joint gravity_power (1.0 = authored values).
    /// Returns a map of node_index → rotation quaternion for spring-affected bones.
    pub fn step(
        &mut self,
        model: &VrmModel,
        world_transforms: &[Mat4],
        dt: f32,
        gravity_scale: f32,
    ) -> HashMap<usize, Quat> {
        let dt = dt.min(0.05); // Clamp to avoid explosion on frame spikes
        if dt < 1e-6 {
            return HashMap::new();
        }

        // Resolve colliders to world space
        let world_colliders = self.resolve_colliders(world_transforms);

        let mut rotations = HashMap::new();

        for chain in &mut self.chains {
            // Collect collider indices for this chain
            let chain_colliders: Vec<&WorldCollider> = chain
                .collider_group_indices
                .iter()
                .filter_map(|&gi| self.collider_groups.get(gi))
                .flat_map(|indices| indices.iter())
                .filter_map(|&ci| world_colliders.get(ci))
                .collect();

            // Mutable world transforms for hierarchical propagation within the chain
            let mut current_world = world_transforms.to_vec();

            for joint in &mut chain.joints {
                // Propagate parent spring rotation to this joint's position.
                // Without this, child joints don't see their parent's spring motion
                // and compute tails/collisions against stale positions.
                if let Some(parent) = model.parents[joint.node] {
                    let rest_local = Mat4::from_scale_rotation_translation(
                        model.rest_scales[joint.node],
                        rotations
                            .get(&joint.node)
                            .copied()
                            .unwrap_or(model.rest_rotations[joint.node]),
                        model.rest_translations[joint.node],
                    );
                    current_world[joint.node] = current_world[parent] * rest_local;
                }

                let world_pos = current_world[joint.node].col(3).truncate();
                let parent_rot = if let Some(parent) = model.parents[joint.node] {
                    quat_from_mat4(&current_world[parent])
                } else {
                    quat_from_mat4(&current_world[joint.node])
                };

                // Rest rotation in world space
                let rest_local_rot = model.rest_rotations[joint.node];

                // 1. Inertia (verlet)
                let inertia = (joint.current_tail - joint.prev_tail) * (1.0 - joint.drag_force);

                // 2. Stiffness: pull toward rest pose direction
                let rest_tail_dir = parent_rot * rest_local_rot * joint.bone_axis;
                let stiffness_force = dt * rest_tail_dir * joint.stiffness;

                // 3. Gravity (scaled by user multiplier)
                let gravity = dt * joint.gravity_dir * joint.gravity_power * gravity_scale;

                // 4. Next tail position
                let mut next_tail = joint.current_tail + inertia + stiffness_force + gravity;

                // 5. Length constraint
                next_tail = apply_length_constraint(next_tail, world_pos, joint.bone_length, rest_tail_dir);

                // 6. Collision (multiple passes to resolve deep interpenetration)
                for _pass in 0..3 {
                    let mut moved = false;
                    for collider in &chain_colliders {
                        let resolved =
                            resolve_collision(next_tail, joint.hit_radius, collider, world_pos, joint.bone_length);
                        if resolved != next_tail {
                            next_tail = resolved;
                            moved = true;
                        }
                    }
                    if !moved {
                        break;
                    }
                }

                // Update state
                joint.prev_tail = joint.current_tail;
                joint.current_tail = next_tail;

                // 7. Compute rotation from rest direction to simulated direction
                let current_dir = (next_tail - world_pos).normalize_or_zero();
                let rest_dir = (parent_rot * rest_local_rot * joint.bone_axis).normalize_or_zero();

                if current_dir.length_squared() > 0.5 && rest_dir.length_squared() > 0.5 {
                    let world_delta = Quat::from_rotation_arc(rest_dir, current_dir);
                    let final_rot = world_delta * parent_rot * rest_local_rot;
                    // Convert back to parent-local space
                    let local_rot = parent_rot.inverse() * final_rot;
                    rotations.insert(joint.node, local_rot);

                    // Update world transform for hierarchical propagation
                    let local_mat = Mat4::from_scale_rotation_translation(
                        model.rest_scales[joint.node],
                        local_rot,
                        model.rest_translations[joint.node],
                    );
                    let parent_world = if let Some(parent) = model.parents[joint.node] {
                        current_world[parent]
                    } else {
                        Mat4::IDENTITY
                    };
                    current_world[joint.node] = parent_world * local_mat;
                }
            }
        }

        rotations
    }

    /// Resolve all colliders to world space using current world transforms.
    fn resolve_colliders(&self, world_transforms: &[Mat4]) -> Vec<WorldCollider> {
        self.collider_nodes
            .iter()
            .map(|(node, shape)| {
                let world_mat = world_transforms[*node];
                let world_pos = world_mat.col(3).truncate();
                let world_rot = quat_from_mat4(&world_mat);

                let shape = match shape {
                    ColliderShapeCopy::Sphere { offset, radius } => {
                        let center = world_pos + world_rot * *offset;
                        WorldColliderShape::Sphere {
                            center,
                            radius: *radius,
                        }
                    }
                    ColliderShapeCopy::Capsule {
                        offset,
                        tail,
                        radius,
                    } => {
                        let start = world_pos + world_rot * *offset;
                        let end = world_pos + world_rot * *tail;
                        WorldColliderShape::Capsule {
                            start,
                            end,
                            radius: *radius,
                        }
                    }
                };
                WorldCollider { shape }
            })
            .collect()
    }

    /// Reset all joint positions to rest pose.
    #[allow(dead_code)]
    pub fn reset(&mut self, model: &VrmModel) {
        let rest_world = skinning::compute_world_transforms(model, &HashMap::new());
        for chain in &mut self.chains {
            for joint in &mut chain.joints {
                let world_pos = rest_world[joint.node].col(3).truncate();
                let joint_rot = quat_from_mat4(&rest_world[joint.node]);
                let tail = world_pos + joint_rot * joint.bone_axis * joint.bone_length;
                joint.prev_tail = tail;
                joint.current_tail = tail;
            }
        }
    }
}

/// Constrain `tail` to be exactly `bone_length` from `world_pos`.
fn apply_length_constraint(tail: Vec3, world_pos: Vec3, bone_length: f32, fallback_dir: Vec3) -> Vec3 {
    let to_tail = tail - world_pos;
    let dist = to_tail.length();
    if dist > 1e-6 {
        world_pos + (to_tail / dist) * bone_length
    } else {
        world_pos + fallback_dir * bone_length
    }
}

/// Push a point out of a collider (sphere or capsule).
fn resolve_collision(
    tail: Vec3,
    hit_radius: f32,
    collider: &WorldCollider,
    world_pos: Vec3,
    bone_length: f32,
) -> Vec3 {
    match &collider.shape {
        WorldColliderShape::Sphere { center, radius } => {
            let diff = tail - *center;
            let dist = diff.length();
            let min_dist = *radius + hit_radius;
            if dist < min_dist && dist > 1e-6 {
                let push = *center + (diff / dist) * min_dist;
                // Re-apply length constraint after push
                let to_pushed = push - world_pos;
                let d = to_pushed.length();
                if d > 1e-6 {
                    world_pos + (to_pushed / d) * bone_length
                } else {
                    tail
                }
            } else {
                tail
            }
        }
        WorldColliderShape::Capsule {
            start,
            end,
            radius,
        } => {
            // Closest point on segment to tail
            let seg = *end - *start;
            let seg_len_sq = seg.length_squared();
            let t = if seg_len_sq < 1e-10 {
                0.0
            } else {
                ((tail - *start).dot(seg) / seg_len_sq).clamp(0.0, 1.0)
            };
            let closest = *start + seg * t;

            let diff = tail - closest;
            let dist = diff.length();
            let min_dist = *radius + hit_radius;
            if dist < min_dist && dist > 1e-6 {
                let push = closest + (diff / dist) * min_dist;
                let to_pushed = push - world_pos;
                let d = to_pushed.length();
                if d > 1e-6 {
                    world_pos + (to_pushed / d) * bone_length
                } else {
                    tail
                }
            } else {
                tail
            }
        }
    }
}

/// Extract rotation quaternion from a 4x4 transform matrix.
fn quat_from_mat4(m: &Mat4) -> Quat {
    let mat3 = glam::Mat3::from_mat4(*m);
    Quat::from_mat3(&mat3).normalize()
}

/// Find the first child of a node and return (bone_length, local_axis).
fn find_child_node(model: &VrmModel, node: usize, rest_world: &[Mat4]) -> Option<(f32, Vec3)> {
    let node_pos = rest_world[node].col(3).truncate();
    let node_rot = quat_from_mat4(&rest_world[node]);

    // Find children of this node
    for (child_idx, parent) in model.parents.iter().enumerate() {
        if *parent == Some(node) {
            let child_pos = rest_world[child_idx].col(3).truncate();
            let world_dir = child_pos - node_pos;
            let length = world_dir.length();
            if length < 1e-6 {
                continue;
            }
            let local_dir = node_rot.inverse() * world_dir.normalize();
            return Some((length, local_dir));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spring_simulator_creation() {
        let model_path = "assets/default/model.glb";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping test: model.glb not found");
            return;
        }

        let model = VrmModel::load(model_path).expect("Failed to load model");

        if model.spring_chains.is_empty() {
            eprintln!("Model has no spring bones, skipping");
            return;
        }

        let sim = SpringBoneSimulator::new(&model);
        assert!(sim.is_some(), "Should create simulator from model with spring bones");
    }

    #[test]
    fn test_spring_step_produces_rotations() {
        let model_path = "assets/default/model.glb";
        if !std::path::Path::new(model_path).exists() {
            return;
        }

        let model = VrmModel::load(model_path).unwrap();
        let mut sim = match SpringBoneSimulator::new(&model) {
            Some(s) => s,
            None => return,
        };

        let world = skinning::compute_world_transforms(&model, &HashMap::new());
        let rotations = sim.step(&model, &world, 1.0 / 60.0, 1.0);

        // With rest pose and gravity, should produce some rotations
        // (joints with gravity_power > 0 will move)
        eprintln!("Spring step produced {} rotations", rotations.len());
    }

    #[test]
    fn test_spring_reset() {
        let model_path = "assets/default/model.glb";
        if !std::path::Path::new(model_path).exists() {
            return;
        }

        let model = VrmModel::load(model_path).unwrap();
        let mut sim = match SpringBoneSimulator::new(&model) {
            Some(s) => s,
            None => return,
        };

        // Step a few times
        let world = skinning::compute_world_transforms(&model, &HashMap::new());
        for _ in 0..10 {
            sim.step(&model, &world, 1.0 / 60.0, 1.0);
        }

        // Reset should not panic
        sim.reset(&model);
    }

    #[test]
    fn test_sphere_collision() {
        let collider = WorldCollider {
            shape: WorldColliderShape::Sphere {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 0.1,
            },
        };

        // Tail inside the sphere, world_pos positioned so length constraint
        // doesn't pull back into the collider
        let tail = Vec3::new(0.05, 0.0, 0.0);
        let world_pos = Vec3::new(0.0, -0.3, 0.0);
        let result = resolve_collision(tail, 0.01, &collider, world_pos, 0.35);

        // Should be pushed further from center than original tail
        let orig_dist = (tail - Vec3::ZERO).length();
        let result_dist = (result - Vec3::ZERO).length();
        assert!(
            result_dist > orig_dist,
            "Should be pushed away from sphere center, orig={}, result={}",
            orig_dist,
            result_dist
        );
    }

    #[test]
    fn test_capsule_collision() {
        let collider = WorldCollider {
            shape: WorldColliderShape::Capsule {
                start: Vec3::new(0.0, -0.5, 0.0),
                end: Vec3::new(0.0, 0.5, 0.0),
                radius: 0.1,
            },
        };

        // Tail near the capsule axis
        let tail = Vec3::new(0.05, 0.0, 0.0);
        let world_pos = Vec3::new(0.0, -0.3, 0.0);
        let result = resolve_collision(tail, 0.01, &collider, world_pos, 0.35);

        // Should be pushed further from capsule axis than original
        let orig_dist_x = tail.x.abs();
        let result_dist_x = result.x.abs();
        assert!(
            result_dist_x > orig_dist_x,
            "Should be pushed away from capsule axis, orig_x={}, result_x={}",
            orig_dist_x,
            result_dist_x
        );
    }
}
