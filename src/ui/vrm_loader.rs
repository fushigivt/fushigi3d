//! GLB/VRM model loader using the `gltf` crate.
//!
//! Extracts meshes, morph target deltas (including sparse), skeleton hierarchy,
//! inverse bind matrices, and per-vertex joint weights from a glTF binary.

#![cfg(feature = "native-ui")]

use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use std::path::Path;

/// A loaded VRM model ready for CPU skinning and GPU rendering.
pub struct VrmModel {
    /// Per-mesh data (Body=0, Face=1, Hair=2 in the sample model)
    pub meshes: Vec<MeshData>,
    /// Skeleton: node index → rest-pose local transform
    pub rest_translations: Vec<Vec3>,
    pub rest_rotations: Vec<Quat>,
    pub rest_scales: Vec<Vec3>,
    /// Parent map: child node → parent node (None if root)
    pub parents: Vec<Option<usize>>,
    /// Total number of nodes
    pub node_count: usize,
    /// Skin data (joint lists + inverse bind matrices)
    pub skins: Vec<SkinData>,
    /// Which skin each mesh uses: mesh_index → skin_index
    pub mesh_skin: HashMap<usize, usize>,
    /// VRM humanoid bone name → node index
    pub bone_to_node: HashMap<String, usize>,
    /// Face mesh morph target names (from mesh extras)
    pub morph_target_names: Vec<String>,
}

pub struct SkinData {
    pub joints: Vec<usize>,
    pub inverse_bind_matrices: Vec<Mat4>,
}

/// All geometry data for one mesh (potentially multiple primitives).
pub struct MeshData {
    pub primitives: Vec<PrimitiveData>,
    /// Morph target deltas per primitive (only for Face mesh).
    /// morph_deltas[prim_idx][target_idx] = Vec<Vec3> position deltas
    pub morph_deltas: Vec<Vec<Vec<Vec3>>>,
}

/// Geometry for a single primitive.
pub struct PrimitiveData {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub indices: Vec<u32>,
    /// Per-vertex joint indices (4 per vertex)
    pub joints: Vec<[u16; 4]>,
    /// Per-vertex joint weights (4 per vertex)
    pub weights: Vec<[f32; 4]>,
    /// Base color factor from material (RGBA)
    pub base_color: [f32; 4],
}

impl VrmModel {
    /// Load a GLB file and extract all data needed for rendering.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        let (document, buffers, _images) =
            gltf::import(path).map_err(|e| format!("Failed to load GLB: {}", e))?;

        let buf = &buffers;

        // Build parent map
        let node_count = document.nodes().count();
        let mut parents = vec![None; node_count];
        for node in document.nodes() {
            for child in node.children() {
                parents[child.index()] = Some(node.index());
            }
        }

        // Parse rest-pose transforms
        let mut rest_translations = Vec::with_capacity(node_count);
        let mut rest_rotations = Vec::with_capacity(node_count);
        let mut rest_scales = Vec::with_capacity(node_count);
        for node in document.nodes() {
            let (t, r, s) = node.transform().decomposed();
            rest_translations.push(Vec3::from(t));
            rest_rotations.push(Quat::from_array(r));
            rest_scales.push(Vec3::from(s));
        }

        // Parse VRM humanoid bone map by reading raw GLB JSON
        let bone_to_node = parse_vrm_bones_from_glb(path)?;

        // Parse skins
        let mut skins = Vec::new();
        for skin in document.skins() {
            let joints: Vec<usize> = skin.joints().map(|j| j.index()).collect();
            let reader = skin.reader(|buffer| Some(&buf[buffer.index()]));
            let ibms: Vec<Mat4> = reader
                .read_inverse_bind_matrices()
                .map(|iter| iter.map(|m| Mat4::from_cols_array_2d(&m)).collect())
                .unwrap_or_else(|| vec![Mat4::IDENTITY; joints.len()]);

            skins.push(SkinData {
                joints,
                inverse_bind_matrices: ibms,
            });
        }

        // Map mesh → skin
        let mut mesh_skin = HashMap::new();
        for node in document.nodes() {
            if let (Some(mesh), Some(skin)) = (node.mesh(), node.skin()) {
                mesh_skin.insert(mesh.index(), skin.index());
            }
        }

        // Parse meshes
        let mut meshes = Vec::new();
        let mut morph_target_names = Vec::new();

        for mesh in document.meshes() {
            let mesh_idx = mesh.index();

            // Get morph target names from extras (Face mesh)
            if mesh_idx == 1 {
                morph_target_names = parse_morph_target_names(&mesh);
            }

            let mut primitives = Vec::new();
            let mut prim_morph_deltas: Vec<Vec<Vec<Vec3>>> = Vec::new();

            for prim in mesh.primitives() {
                let reader = prim.reader(|buffer| Some(&buf[buffer.index()]));

                // Positions
                let positions: Vec<Vec3> = reader
                    .read_positions()
                    .map(|iter| iter.map(Vec3::from).collect())
                    .unwrap_or_default();

                // Normals
                let normals: Vec<Vec3> = reader
                    .read_normals()
                    .map(|iter| iter.map(Vec3::from).collect())
                    .unwrap_or_else(|| vec![Vec3::Y; positions.len()]);

                // Indices
                let indices: Vec<u32> = reader
                    .read_indices()
                    .map(|iter| iter.into_u32().collect())
                    .unwrap_or_default();

                // Joints
                let joints: Vec<[u16; 4]> = reader
                    .read_joints(0)
                    .map(|iter| iter.into_u16().collect())
                    .unwrap_or_else(|| vec![[0; 4]; positions.len()]);

                // Weights
                let weights: Vec<[f32; 4]> = reader
                    .read_weights(0)
                    .map(|iter| iter.into_f32().collect())
                    .unwrap_or_else(|| vec![[1.0, 0.0, 0.0, 0.0]; positions.len()]);

                // Base color from material
                let base_color = prim
                    .material()
                    .pbr_metallic_roughness()
                    .base_color_factor();

                primitives.push(PrimitiveData {
                    positions,
                    normals,
                    indices,
                    joints,
                    weights,
                    base_color,
                });

                // Morph target deltas (only for Face mesh)
                if mesh_idx == 1 {
                    let deltas = read_morph_deltas_for_primitive(&prim, buf);
                    prim_morph_deltas.push(deltas);
                } else {
                    prim_morph_deltas.push(Vec::new());
                }
            }

            meshes.push(MeshData {
                primitives,
                morph_deltas: prim_morph_deltas,
            });
        }

        Ok(VrmModel {
            meshes,
            rest_translations,
            rest_rotations,
            rest_scales,
            parents,
            node_count,
            skins,
            mesh_skin,
            bone_to_node,
            morph_target_names,
        })
    }
}

/// Parse VRM humanoid bone names by reading raw GLB JSON chunk.
fn parse_vrm_bones_from_glb(path: &Path) -> Result<HashMap<String, usize>, String> {
    let data = std::fs::read(path).map_err(|e| format!("Failed to read GLB: {}", e))?;

    // GLB format: 12-byte header + chunks
    // Header: magic(4) + version(4) + length(4)
    // Chunk: length(4) + type(4) + data(length)
    // First chunk is always JSON (type 0x4E4F534A)
    if data.len() < 20 {
        return Ok(HashMap::new());
    }

    let json_length =
        u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    if data.len() < 20 + json_length {
        return Ok(HashMap::new());
    }

    let json_data = &data[20..20 + json_length];
    let root: serde_json::Value =
        serde_json::from_slice(json_data).map_err(|e| format!("JSON parse error: {}", e))?;

    let mut map = HashMap::new();

    // Try VRMC_vrm (VRM 1.0)
    if let Some(vrmc) = root
        .get("extensions")
        .and_then(|e| e.get("VRMC_vrm"))
    {
        if let Some(bones) = vrmc
            .get("humanoid")
            .and_then(|h| h.get("humanBones"))
            .and_then(|b| b.as_object())
        {
            for (bone_name, data) in bones {
                if let Some(node_idx) = data.get("node").and_then(|n| n.as_u64()) {
                    map.insert(bone_name.clone(), node_idx as usize);
                }
            }
        }
    }

    // Fallback: try VRM 0.x format
    if map.is_empty() {
        if let Some(vrm_ext) = root
            .get("extensions")
            .and_then(|e| e.get("VRM"))
        {
            if let Some(bones) = vrm_ext
                .get("humanoid")
                .and_then(|h| h.get("humanBones"))
                .and_then(|b| b.as_array())
            {
                for bone in bones {
                    if let (Some(name), Some(node)) = (
                        bone.get("bone").and_then(|b| b.as_str()),
                        bone.get("node").and_then(|n| n.as_u64()),
                    ) {
                        // VRM 0.x uses camelCase bone names
                        map.insert(camel_to_lower(name), node as usize);
                    }
                }
            }
        }
    }

    Ok(map)
}

/// Convert VRM 0.x camelCase bone names to VRM 1.0 format.
fn camel_to_lower(s: &str) -> String {
    // VRM 0.x: "Head" → "head", "LeftUpperArm" → "leftUpperArm"
    let mut result = String::with_capacity(s.len());
    for (i, c) in s.chars().enumerate() {
        if i == 0 {
            result.extend(c.to_lowercase());
        } else {
            result.push(c);
        }
    }
    result
}

/// Parse morph target names from mesh extras JSON, stripping any shared prefix.
fn parse_morph_target_names(mesh: &gltf::Mesh) -> Vec<String> {
    if let Some(extras) = mesh.extras().as_ref() {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(extras.get()) {
            if let Some(names) = val.get("targetNames").and_then(|v| v.as_array()) {
                let raw: Vec<String> = names
                    .iter()
                    .filter_map(|n| n.as_str().map(String::from))
                    .collect();
                return strip_morph_prefixes(raw);
            }
        }
    }
    Vec::new()
}

/// Strip a shared dot-delimited prefix from morph target names.
///
/// Many VRM models store morph names as `"Face_Blendshape.Fcl_MTH_A"` where
/// `"Face_Blendshape."` is a mesh-level prefix. The `BlendshapeMapper` looks up
/// bare names like `"Fcl_MTH_A"`, so we strip the prefix here.
///
/// The prefix is only stripped when *all* names share the same `<something>.`
/// prefix. If names have no dot or mixed prefixes, they're returned as-is.
fn strip_morph_prefixes(names: Vec<String>) -> Vec<String> {
    if names.len() < 2 {
        return names;
    }

    // Find the first dot-prefix
    let first_dot = match names[0].find('.') {
        Some(pos) => pos,
        None => return names,
    };
    let prefix_len = first_dot + 1; // includes the dot

    // Check all names share this prefix
    let prefix = &names[0][..prefix_len];
    let all_share = names.iter().all(|n| n.starts_with(prefix));
    if !all_share {
        return names;
    }

    // Strip the prefix
    names
        .into_iter()
        .map(|n| n[prefix_len..].to_string())
        .collect()
}

/// Read morph target position deltas for a primitive.
fn read_morph_deltas_for_primitive(
    prim: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
) -> Vec<Vec<Vec3>> {
    let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

    let mut all_targets = Vec::new();
    // read_morph_targets returns tuples: (positions, normals, tangents)
    for (positions, _normals, _tangents) in reader.read_morph_targets() {
        let deltas: Vec<Vec3> = match positions {
            Some(iter) => iter.map(|p: [f32; 3]| Vec3::from(p)).collect(),
            None => Vec::new(),
        };
        all_targets.push(deltas);
    }
    all_targets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_model() {
        let model_path = "assets/default/model.glb";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping test: model.glb not found");
            return;
        }

        let model = VrmModel::load(model_path).expect("Failed to load model");

        // Should have 3 meshes: Body, Face, Hair
        assert_eq!(model.meshes.len(), 3, "Expected 3 meshes");

        // Face mesh (index 1) should have 57 morph targets
        assert!(
            !model.morph_target_names.is_empty(),
            "Expected morph target names"
        );

        // Morph names should not contain dot-prefixes after stripping
        for name in &model.morph_target_names {
            assert!(
                !name.contains('.'),
                "Morph name '{}' should not contain '.' after prefix stripping",
                name
            );
        }

        // Should have a skeleton
        assert!(!model.skins.is_empty(), "Expected at least one skin");

        // Should have humanoid bones
        assert!(
            model.bone_to_node.contains_key("head"),
            "Expected 'head' bone"
        );
        assert!(
            model.bone_to_node.contains_key("hips"),
            "Expected 'hips' bone"
        );

        // Body mesh should have primitives with vertices
        assert!(
            !model.meshes[0].primitives.is_empty(),
            "Body mesh should have primitives"
        );
        assert!(
            !model.meshes[0].primitives[0].positions.is_empty(),
            "Body primitives should have vertices"
        );
    }

    #[test]
    fn test_strip_morph_prefixes_shared() {
        let names = vec![
            "Face_Blendshape.Fcl_MTH_A".to_string(),
            "Face_Blendshape.Fcl_MTH_I".to_string(),
            "Face_Blendshape.Fcl_EYE_Close".to_string(),
        ];
        let stripped = strip_morph_prefixes(names);
        assert_eq!(stripped, vec!["Fcl_MTH_A", "Fcl_MTH_I", "Fcl_EYE_Close"]);
    }

    #[test]
    fn test_strip_morph_prefixes_no_dot() {
        let names = vec![
            "Fcl_MTH_A".to_string(),
            "Fcl_MTH_I".to_string(),
            "Fcl_EYE_Close".to_string(),
        ];
        let stripped = strip_morph_prefixes(names.clone());
        assert_eq!(stripped, names, "Names without dots should pass through unchanged");
    }

    #[test]
    fn test_strip_morph_prefixes_mixed() {
        let names = vec![
            "Face_Blendshape.Fcl_MTH_A".to_string(),
            "Other_Mesh.Fcl_MTH_I".to_string(),
            "Face_Blendshape.Fcl_EYE_Close".to_string(),
        ];
        let stripped = strip_morph_prefixes(names.clone());
        assert_eq!(stripped, names, "Mixed prefixes should not be stripped");
    }

    #[test]
    fn test_load_alicia_model() {
        let model_path = "assets/test/AliciaSolid.vrm";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping test: AliciaSolid.vrm not found (run scripts/download_test_model.sh)");
            return;
        }

        let model = VrmModel::load(model_path).expect("Failed to load AliciaSolid");

        // Should have humanoid bones including fingers
        assert!(model.bone_to_node.contains_key("head"), "Expected 'head' bone");
        assert!(model.bone_to_node.contains_key("hips"), "Expected 'hips' bone");
        assert!(
            model.bone_to_node.contains_key("leftIndexProximal"),
            "Expected finger bones"
        );

        // Morph names should be bare (no dot prefix)
        for name in &model.morph_target_names {
            assert!(
                !name.contains('.'),
                "Morph name '{}' should not contain '.'",
                name
            );
        }
    }
}
