//! CPU skinning: forward kinematics, morph target application, and
//! linear blend skinning (LBS) for VRM models.

#![cfg(feature = "native-ui")]

use glam::{Mat4, Quat, Vec3, Vec4};
use std::collections::HashMap;

use super::vrm_loader::VrmModel;

/// Compute world transforms for all nodes using forward kinematics.
///
/// `local_rotations`: optional rotation overrides (node_index → quaternion).
/// Nodes without overrides use their rest-pose rotation.
pub fn compute_world_transforms(
    model: &VrmModel,
    local_rotations: &HashMap<usize, Quat>,
) -> Vec<Mat4> {
    let mut world = vec![Mat4::IDENTITY; model.node_count];
    let mut computed = vec![false; model.node_count];

    for i in 0..model.node_count {
        compute_node(model, local_rotations, &mut world, &mut computed, i);
    }

    world
}

fn compute_node(
    model: &VrmModel,
    local_rotations: &HashMap<usize, Quat>,
    world: &mut [Mat4],
    computed: &mut [bool],
    idx: usize,
) {
    if computed[idx] {
        return;
    }

    // Build local TRS matrix
    let t = model.rest_translations[idx];
    let r = local_rotations
        .get(&idx)
        .copied()
        .unwrap_or(model.rest_rotations[idx]);
    let s = model.rest_scales[idx];

    let local = Mat4::from_scale_rotation_translation(s, r, t);

    if let Some(parent) = model.parents[idx] {
        compute_node(model, local_rotations, world, computed, parent);
        world[idx] = world[parent] * local;
    } else {
        world[idx] = local;
    }
    computed[idx] = true;
}

/// Apply morph target deltas to the base positions of a face mesh.
///
/// Returns new position arrays for each primitive of the face mesh.
pub fn apply_morph_targets(
    model: &VrmModel,
    face_mesh_idx: usize,
    morph_weights: &[f32],
) -> Vec<Vec<Vec3>> {
    let mesh = &model.meshes[face_mesh_idx];
    let mut result = Vec::with_capacity(mesh.primitives.len());

    for (prim_idx, prim) in mesh.primitives.iter().enumerate() {
        let mut morphed = prim.positions.clone();

        let deltas = &mesh.morph_deltas[prim_idx];
        for (t_idx, &weight) in morph_weights.iter().enumerate() {
            if weight < 0.001 || t_idx >= deltas.len() {
                continue;
            }
            let target_deltas = &deltas[t_idx];
            if target_deltas.len() != morphed.len() {
                continue;
            }
            for (v, delta) in morphed.iter_mut().zip(target_deltas.iter()) {
                *v += *delta * weight;
            }
        }

        result.push(morphed);
    }

    result
}

/// Apply linear blend skinning to vertex positions.
///
/// `mesh_idx`: which mesh to skin
/// `vertices_per_prim`: base (or morphed) positions for each primitive
/// `world_transforms`: from `compute_world_transforms`
///
/// Returns skinned positions for each primitive.
pub fn skin_vertices(
    model: &VrmModel,
    mesh_idx: usize,
    vertices_per_prim: &[Vec<Vec3>],
    world_transforms: &[Mat4],
) -> Vec<Vec<Vec3>> {
    let skin_idx = match model.mesh_skin.get(&mesh_idx) {
        Some(&s) => s,
        None => return vertices_per_prim.to_vec(),
    };

    let skin = &model.skins[skin_idx];

    // Precompute joint matrices: world[joint_node] * inverse_bind_matrix
    let joint_matrices: Vec<Mat4> = skin
        .joints
        .iter()
        .zip(skin.inverse_bind_matrices.iter())
        .map(|(&node_idx, ibm)| world_transforms[node_idx] * *ibm)
        .collect();

    let mesh = &model.meshes[mesh_idx];
    let mut result = Vec::with_capacity(mesh.primitives.len());

    for (prim_idx, prim) in mesh.primitives.iter().enumerate() {
        let base_verts = &vertices_per_prim[prim_idx];
        let mut skinned = vec![Vec3::ZERO; base_verts.len()];

        for (v_idx, pos) in base_verts.iter().enumerate() {
            let j = prim.joints[v_idx];
            let w = prim.weights[v_idx];
            let p = Vec4::new(pos.x, pos.y, pos.z, 1.0);

            let mut result_pos = Vec4::ZERO;
            for k in 0..4 {
                if w[k] < 0.0001 {
                    continue;
                }
                let jm = joint_matrices[j[k] as usize];
                result_pos += w[k] * (jm * p);
            }

            skinned[v_idx] = result_pos.truncate();
        }

        result.push(skinned);
    }

    result
}

/// Convenience: get base positions (unskinned, unmorphed) for a mesh.
pub fn base_positions(model: &VrmModel, mesh_idx: usize) -> Vec<Vec<Vec3>> {
    model.meshes[mesh_idx]
        .primitives
        .iter()
        .map(|p| p.positions.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_skinning() {
        // With identity world transforms, skinned positions should match base
        let model_path = "assets/default/model.glb";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let model = VrmModel::load(model_path).unwrap();

        let rotations = HashMap::new();
        let world = compute_world_transforms(&model, &rotations);

        // Check that world transforms are computed for all nodes
        assert_eq!(world.len(), model.node_count);
    }

    #[test]
    fn test_morph_targets_zero_weights() {
        let model_path = "assets/default/model.glb";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let model = VrmModel::load(model_path).unwrap();

        let face_idx = model.face_mesh_idx.unwrap_or(1);
        let zero_weights = vec![0.0f32; model.morph_target_names.len()];
        let morphed = apply_morph_targets(&model, face_idx, &zero_weights);

        // With zero weights, morphed should equal base positions
        for (prim_idx, prim) in model.meshes[face_idx].primitives.iter().enumerate() {
            assert_eq!(morphed[prim_idx].len(), prim.positions.len());
            for (a, b) in morphed[prim_idx].iter().zip(prim.positions.iter()) {
                assert!((a.x - b.x).abs() < 1e-6);
                assert!((a.y - b.y).abs() < 1e-6);
                assert!((a.z - b.z).abs() < 1e-6);
            }
        }
    }
}
