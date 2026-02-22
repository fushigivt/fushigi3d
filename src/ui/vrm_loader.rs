//! GLB/VRM model loader using the `gltf` crate.
//!
//! Extracts meshes, morph target deltas (including sparse), skeleton hierarchy,
//! inverse bind matrices, and per-vertex joint weights from a glTF binary.

#![cfg(feature = "native-ui")]

use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use std::path::Path;

/// A VRM 1.0 expression bind: maps a preset expression to morph target indices.
#[derive(Clone, Debug)]
pub struct ExpressionBind {
    /// Morph target index within the Face mesh
    pub morph_index: usize,
    /// Weight to apply (typically 1.0)
    pub weight: f32,
}

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
    /// Index of the face mesh (the one with morph targets)
    pub face_mesh_idx: Option<usize>,
    /// Face mesh morph target names (from mesh extras)
    pub morph_target_names: Vec<String>,
    /// VRM 1.0 expression preset name → morph target binds
    pub expression_binds: HashMap<String, Vec<ExpressionBind>>,
    /// Expressions marked as `isBinary` (values snap to 0 or 1)
    pub binary_expressions: std::collections::HashSet<String>,
    /// Spring bone chains (hair, cloth, etc.)
    pub spring_chains: Vec<SpringChain>,
    /// Spring bone colliders
    pub spring_colliders: Vec<SpringCollider>,
    /// Collider groups
    pub collider_groups: Vec<ColliderGroup>,
}

pub struct SkinData {
    pub joints: Vec<usize>,
    pub inverse_bind_matrices: Vec<Mat4>,
}

/// A spring bone chain (hair, cloth, etc.) from VRMC_springBone.
pub struct SpringChain {
    #[allow(dead_code)]
    pub name: String,
    pub joints: Vec<SpringJoint>,
    pub collider_group_indices: Vec<usize>,
}

/// A single joint in a spring bone chain.
pub struct SpringJoint {
    pub node: usize,
    pub hit_radius: f32,
    pub stiffness: f32,
    pub gravity_power: f32,
    pub gravity_dir: Vec3,
    pub drag_force: f32,
}

/// A collider attached to a node.
pub struct SpringCollider {
    pub node: usize,
    pub shape: ColliderShape,
}

/// Collider geometry.
pub enum ColliderShape {
    Sphere { offset: Vec3, radius: f32 },
    Capsule { offset: Vec3, tail: Vec3, radius: f32 },
}

/// A named group of collider indices.
pub struct ColliderGroup {
    #[allow(dead_code)]
    pub name: String,
    pub collider_indices: Vec<usize>,
}

/// All geometry data for one mesh (potentially multiple primitives).
pub struct MeshData {
    pub primitives: Vec<PrimitiveData>,
    /// Morph target deltas per primitive (only for Face mesh).
    /// morph_deltas[prim_idx][target_idx] = Vec<Vec3> position deltas
    pub morph_deltas: Vec<Vec<Vec<Vec3>>>,
}

/// Decoded texture image (RGBA8).
pub struct TextureImage {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
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
    /// Per-vertex UV coordinates (from TEXCOORD_0)
    pub uvs: Vec<[f32; 2]>,
    /// Decoded texture image (if material has a baseColorTexture)
    pub texture: Option<TextureImage>,
}

impl VrmModel {
    /// Load a GLB file and extract all data needed for rendering.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        let (document, buffers, images) =
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

        // Parse spring bone data
        let (spring_chains, spring_colliders, collider_groups) =
            parse_spring_bones_from_glb(path)?;

        // Parse VRM 1.0 expression binds
        let (expression_binds, binary_expressions) = parse_expression_binds_from_glb(path)?;

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

        // Find face mesh index: the mesh with the most morph targets
        let face_mesh_idx = document
            .meshes()
            .filter_map(|m| {
                let count = m
                    .primitives()
                    .next()
                    .map(|p| p.morph_targets().count())
                    .unwrap_or(0);
                if count > 0 { Some((m.index(), count)) } else { None }
            })
            .max_by_key(|&(_, c)| c)
            .map(|(i, _)| i);

        // Parse meshes
        let mut meshes = Vec::new();
        let mut morph_target_names = Vec::new();

        for mesh in document.meshes() {
            let mesh_idx = mesh.index();
            let is_face = Some(mesh_idx) == face_mesh_idx;

            // Get morph target names from extras (Face mesh)
            if is_face {
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

                // UVs
                let uvs: Vec<[f32; 2]> = reader
                    .read_tex_coords(0)
                    .map(|iter| iter.into_f32().collect())
                    .unwrap_or_else(|| vec![[0.0; 2]; positions.len()]);

                // Base color from material
                let pbr = prim.material().pbr_metallic_roughness();
                let base_color = pbr.base_color_factor();

                // Texture image from material
                let texture = pbr.base_color_texture().and_then(|tex_info| {
                    let img_idx = tex_info.texture().source().index();
                    images.get(img_idx).map(|img_data| {
                        let pixels = convert_to_rgba8(img_data);
                        TextureImage {
                            pixels,
                            width: img_data.width,
                            height: img_data.height,
                        }
                    })
                });

                primitives.push(PrimitiveData {
                    positions,
                    normals,
                    indices,
                    joints,
                    weights,
                    base_color,
                    uvs,
                    texture,
                });

                // Morph target deltas (only for Face mesh)
                if is_face {
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
            face_mesh_idx,
            morph_target_names,
            expression_binds,
            binary_expressions,
            spring_chains,
            spring_colliders,
            collider_groups,
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

/// Parse spring bone data from the VRMC_springBone extension in a GLB file.
fn parse_spring_bones_from_glb(
    path: &Path,
) -> Result<(Vec<SpringChain>, Vec<SpringCollider>, Vec<ColliderGroup>), String> {
    let data = std::fs::read(path).map_err(|e| format!("Failed to read GLB: {}", e))?;

    if data.len() < 20 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    let json_length = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    if data.len() < 20 + json_length {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    let json_data = &data[20..20 + json_length];
    let root: serde_json::Value =
        serde_json::from_slice(json_data).map_err(|e| format!("JSON parse error: {}", e))?;

    // Try VRMC_springBone (VRM 1.0)
    if let Some(spring_ext) = root
        .get("extensions")
        .and_then(|e| e.get("VRMC_springBone"))
    {
        return parse_spring_bones_1_0(spring_ext);
    }

    // Fallback: VRM 0.x secondaryAnimation
    if let Some(vrm_ext) = root.get("extensions").and_then(|e| e.get("VRM")) {
        if let Some(secondary) = vrm_ext.get("secondaryAnimation") {
            return parse_spring_bones_0x(secondary);
        }
    }

    Ok((Vec::new(), Vec::new(), Vec::new()))
}

/// Parse VRM 1.0 VRMC_springBone extension.
fn parse_spring_bones_1_0(
    spring_ext: &serde_json::Value,
) -> Result<(Vec<SpringChain>, Vec<SpringCollider>, Vec<ColliderGroup>), String> {

    // Parse colliders
    let mut colliders = Vec::new();
    if let Some(collider_arr) = spring_ext.get("colliders").and_then(|c| c.as_array()) {
        for c in collider_arr {
            let node = c.get("node").and_then(|n| n.as_u64()).unwrap_or(0) as usize;
            let shape = if let Some(s) = c.get("shape") {
                if let Some(sphere) = s.get("sphere") {
                    let offset = parse_vec3(sphere.get("offset"));
                    let radius = sphere
                        .get("radius")
                        .and_then(|r| r.as_f64())
                        .unwrap_or(0.0) as f32;
                    ColliderShape::Sphere { offset, radius }
                } else if let Some(capsule) = s.get("capsule") {
                    let offset = parse_vec3(capsule.get("offset"));
                    let tail = parse_vec3(capsule.get("tail"));
                    let radius = capsule
                        .get("radius")
                        .and_then(|r| r.as_f64())
                        .unwrap_or(0.0) as f32;
                    ColliderShape::Capsule {
                        offset,
                        tail,
                        radius,
                    }
                } else {
                    ColliderShape::Sphere {
                        offset: Vec3::ZERO,
                        radius: 0.0,
                    }
                }
            } else {
                ColliderShape::Sphere {
                    offset: Vec3::ZERO,
                    radius: 0.0,
                }
            };
            colliders.push(SpringCollider { node, shape });
        }
    }

    // Parse collider groups
    let mut collider_groups = Vec::new();
    if let Some(group_arr) = spring_ext
        .get("colliderGroups")
        .and_then(|g| g.as_array())
    {
        for g in group_arr {
            let name = g
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();
            let collider_indices: Vec<usize> = g
                .get("colliders")
                .and_then(|c| c.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();
            collider_groups.push(ColliderGroup {
                name,
                collider_indices,
            });
        }
    }

    // Parse springs (chains)
    let mut chains = Vec::new();
    if let Some(spring_arr) = spring_ext.get("springs").and_then(|s| s.as_array()) {
        for s in spring_arr {
            let name = s
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();

            let mut joints = Vec::new();
            if let Some(joint_arr) = s.get("joints").and_then(|j| j.as_array()) {
                for j in joint_arr {
                    let node = j.get("node").and_then(|n| n.as_u64()).unwrap_or(0) as usize;
                    let hit_radius = j
                        .get("hitRadius")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0) as f32;
                    let stiffness = j
                        .get("stiffness")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1.0) as f32;
                    let gravity_power = j
                        .get("gravityPower")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0) as f32;
                    let gravity_dir = if let Some(gd) = j.get("gravityDir") {
                        parse_vec3(Some(gd))
                    } else {
                        Vec3::new(0.0, -1.0, 0.0)
                    };
                    let drag_force = j
                        .get("dragForce")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.5) as f32;

                    joints.push(SpringJoint {
                        node,
                        hit_radius,
                        stiffness,
                        gravity_power,
                        gravity_dir,
                        drag_force,
                    });
                }
            }

            let collider_group_indices: Vec<usize> = s
                .get("colliderGroups")
                .and_then(|c| c.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();

            chains.push(SpringChain {
                name,
                joints,
                collider_group_indices,
            });
        }
    }

    Ok((chains, colliders, collider_groups))
}

/// Parse VRM 0.x spring bones from `extensions.VRM.secondaryAnimation`.
///
/// VRM 0.x stores spring bones in `boneGroups` (each group is one chain) and
/// colliders in `colliderGroups` (each group has a node + array of colliders).
/// Notable: the stiffness field is misspelled as `stiffiness` in the VRM 0.x spec.
fn parse_spring_bones_0x(
    secondary: &serde_json::Value,
) -> Result<(Vec<SpringChain>, Vec<SpringCollider>, Vec<ColliderGroup>), String> {
    let mut colliders = Vec::new();
    let mut collider_groups_out = Vec::new();

    // VRM 0.x colliderGroups: each group has a `node` and a `colliders` array
    // of sphere colliders: `{ "offset": {"x":..,"y":..,"z":..}, "radius": f64 }`
    if let Some(groups) = secondary.get("colliderGroups").and_then(|g| g.as_array()) {
        for group in groups {
            let node = group.get("node").and_then(|n| n.as_u64()).unwrap_or(0) as usize;
            let mut indices = Vec::new();

            if let Some(cols) = group.get("colliders").and_then(|c| c.as_array()) {
                for c in cols {
                    let offset = parse_vec3(c.get("offset"));
                    let radius = c
                        .get("radius")
                        .and_then(|r| r.as_f64())
                        .unwrap_or(0.0) as f32;
                    indices.push(colliders.len());
                    colliders.push(SpringCollider {
                        node,
                        shape: ColliderShape::Sphere { offset, radius },
                    });
                }
            }

            collider_groups_out.push(ColliderGroup {
                name: format!("colliderGroup_{}", collider_groups_out.len()),
                collider_indices: indices,
            });
        }
    }

    // VRM 0.x boneGroups: each is a spring chain
    let mut chains = Vec::new();
    if let Some(bone_groups) = secondary.get("boneGroups").and_then(|g| g.as_array()) {
        for group in bone_groups {
            let comment = group
                .get("comment")
                .and_then(|c| c.as_str())
                .unwrap_or("");

            // VRM 0.x: `stiffiness` (note the typo) — also try `stiffness` as fallback
            let stiffness = group
                .get("stiffiness")
                .or_else(|| group.get("stiffness"))
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32;
            let gravity_power = group
                .get("gravityPower")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            let gravity_dir = parse_vec3(group.get("gravityDir"));
            let drag_force = group
                .get("dragForce")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5) as f32;
            let hit_radius = group
                .get("hitRadius")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;

            // VRM 0.x: `bones` is an array of node indices (the chain from root to leaves)
            let bone_nodes: Vec<usize> = group
                .get("bones")
                .and_then(|b| b.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();

            // VRM 0.x: `colliderGroups` is an array of group indices
            let collider_group_indices: Vec<usize> = group
                .get("colliderGroups")
                .and_then(|c| c.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();

            // In VRM 0.x, all bones in a group share the same physics params
            let joints: Vec<SpringJoint> = bone_nodes
                .iter()
                .map(|&node| SpringJoint {
                    node,
                    hit_radius,
                    stiffness,
                    gravity_power,
                    gravity_dir,
                    drag_force,
                })
                .collect();

            if !joints.is_empty() {
                chains.push(SpringChain {
                    name: comment.to_string(),
                    joints,
                    collider_group_indices,
                });
            }
        }
    }

    Ok((chains, colliders, collider_groups_out))
}

/// Parse VRM 1.0 expression preset binds from VRMC_vrm.expressions.preset.
///
/// Returns a map from expression name ("aa", "blink", "happy", ...) to a list
/// of morph target binds (index + weight).  Falls back to VRM 0.x
/// blendShapeMaster if VRMC_vrm is absent.
fn parse_expression_binds_from_glb(
    path: &Path,
) -> Result<(HashMap<String, Vec<ExpressionBind>>, std::collections::HashSet<String>), String> {
    use std::collections::HashSet;

    let data = std::fs::read(path).map_err(|e| format!("Failed to read GLB: {}", e))?;

    if data.len() < 20 {
        return Ok((HashMap::new(), HashSet::new()));
    }

    let json_length = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    if data.len() < 20 + json_length {
        return Ok((HashMap::new(), HashSet::new()));
    }

    let json_data = &data[20..20 + json_length];
    let root: serde_json::Value =
        serde_json::from_slice(json_data).map_err(|e| format!("JSON parse error: {}", e))?;

    let mut map = HashMap::new();
    let mut binary = HashSet::new();

    // Try VRMC_vrm 1.0 expressions
    if let Some(vrmc) = root.get("extensions").and_then(|e| e.get("VRMC_vrm")) {
        if let Some(preset) = vrmc
            .get("expressions")
            .and_then(|e| e.get("preset"))
            .and_then(|p| p.as_object())
        {
            for (name, expr) in preset {
                // isBinary: if true, expression snaps to 0 or 1
                if expr.get("isBinary").and_then(|b| b.as_bool()).unwrap_or(false) {
                    binary.insert(name.clone());
                }
                if let Some(binds) = expr.get("morphTargetBinds").and_then(|b| b.as_array()) {
                    let bind_list: Vec<ExpressionBind> = binds
                        .iter()
                        .filter_map(|b| {
                            let index = b.get("index").and_then(|i| i.as_u64())? as usize;
                            let weight = b
                                .get("weight")
                                .and_then(|w| w.as_f64())
                                .unwrap_or(1.0) as f32;
                            Some(ExpressionBind {
                                morph_index: index,
                                weight,
                            })
                        })
                        .collect();
                    if !bind_list.is_empty() {
                        map.insert(name.clone(), bind_list);
                    }
                }
            }
        }
    }

    // Fallback: VRM 0.x blendShapeMaster
    if map.is_empty() {
        if let Some(vrm_ext) = root.get("extensions").and_then(|e| e.get("VRM")) {
            if let Some(groups) = vrm_ext
                .get("blendShapeMaster")
                .and_then(|m| m.get("blendShapeGroups"))
                .and_then(|g| g.as_array())
            {
                for group in groups {
                    // VRM 0.x: isBinary
                    let is_binary = group
                        .get("isBinary")
                        .and_then(|b| b.as_bool())
                        .unwrap_or(false);

                    // Prefer presetName (standardized enum, lowercase) over name (freeform)
                    let raw_name = group
                        .get("presetName")
                        .and_then(|n| n.as_str())
                        .or_else(|| group.get("name").and_then(|n| n.as_str()));
                    let name = match raw_name {
                        Some(n) => n.to_lowercase(),
                        None => continue,
                    };
                    // VRM 0.x presetName values to VRM 1.0 preset names
                    let preset_name = match name.as_str() {
                        "a" => "aa",
                        "i" => "ih",
                        "u" => "ou",
                        "e" => "ee",
                        "o" => "oh",
                        "blink" => "blink",
                        "blink_l" => "blinkLeft",
                        "blink_r" => "blinkRight",
                        "joy" => "happy",
                        "angry" => "angry",
                        "sorrow" => "sad",
                        "fun" => "relaxed",
                        "neutral" => "neutral",
                        "lookup" => "lookUp",
                        "lookdown" => "lookDown",
                        "lookleft" => "lookLeft",
                        "lookright" => "lookRight",
                        other => other,
                    };

                    if is_binary {
                        binary.insert(preset_name.to_string());
                    }

                    if let Some(binds) = group.get("binds").and_then(|b| b.as_array()) {
                        let bind_list: Vec<ExpressionBind> = binds
                            .iter()
                            .filter_map(|b| {
                                let index =
                                    b.get("index").and_then(|i| i.as_u64())? as usize;
                                let weight = b
                                    .get("weight")
                                    .and_then(|w| w.as_f64())
                                    .unwrap_or(100.0)
                                    as f32
                                    / 100.0; // VRM 0.x uses 0-100 scale
                                Some(ExpressionBind {
                                    morph_index: index,
                                    weight,
                                })
                            })
                            .collect();
                        if !bind_list.is_empty() {
                            map.insert(preset_name.to_string(), bind_list);
                        }
                    }
                }
            }
        }
    }

    Ok((map, binary))
}

/// Parse a vec3 from JSON — handles both array `[x,y,z]` (VRMC 1.0) and
/// object `{"x":..,"y":..,"z":..}` (VRM 0.x) formats.
fn parse_vec3(val: Option<&serde_json::Value>) -> Vec3 {
    match val {
        Some(v) if v.is_array() => {
            let arr = v.as_array().unwrap();
            let x = arr.first().and_then(|n| n.as_f64()).unwrap_or(0.0) as f32;
            let y = arr.get(1).and_then(|n| n.as_f64()).unwrap_or(0.0) as f32;
            let z = arr.get(2).and_then(|n| n.as_f64()).unwrap_or(0.0) as f32;
            Vec3::new(x, y, z)
        }
        Some(v) => {
            let x = v.get("x").and_then(|n| n.as_f64()).unwrap_or(0.0) as f32;
            let y = v.get("y").and_then(|n| n.as_f64()).unwrap_or(0.0) as f32;
            let z = v.get("z").and_then(|n| n.as_f64()).unwrap_or(0.0) as f32;
            Vec3::new(x, y, z)
        }
        None => Vec3::ZERO,
    }
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

/// Convert a glTF image to RGBA8 pixel data.
fn convert_to_rgba8(img: &gltf::image::Data) -> Vec<u8> {
    match img.format {
        gltf::image::Format::R8G8B8A8 => img.pixels.clone(),
        gltf::image::Format::R8G8B8 => img
            .pixels
            .chunks(3)
            .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 255])
            .collect(),
        gltf::image::Format::R8 => img
            .pixels
            .iter()
            .flat_map(|&r| [r, r, r, 255])
            .collect(),
        gltf::image::Format::R8G8 => img
            .pixels
            .chunks(2)
            .flat_map(|rg| [rg[0], rg[1], 0, 255])
            .collect(),
        _ => {
            // Fallback: opaque white
            vec![255u8; (img.width * img.height * 4) as usize]
        }
    }
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

        // Should have meshes (count varies by model)
        assert!(!model.meshes.is_empty(), "Expected at least one mesh");

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
