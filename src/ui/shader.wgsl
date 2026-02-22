// ── 3D scene shader: vertex + fragment with 3-directional lighting ──

struct Uniforms {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    // Light directions (world space, normalized)
    light_dir_0: vec4<f32>,
    light_dir_1: vec4<f32>,
    light_dir_2: vec4<f32>,
    // Light colors * intensity
    light_col_0: vec4<f32>,
    light_col_1: vec4<f32>,
    light_col_2: vec4<f32>,
    // Ambient color
    ambient: vec4<f32>,
    // Material base color
    base_color: vec4<f32>,
    // x: 1.0 = sample texture, 0.0 = use base_color only
    use_texture: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var t_albedo: texture_2d<f32>;
@group(0) @binding(2) var s_albedo: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4<f32>(in.position, 1.0);
    // Transform normal by model matrix (ignoring non-uniform scale for now)
    out.world_normal = normalize((u.model * vec4<f32>(in.normal, 0.0)).xyz);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);

    // Directional lights (standard Lambert)
    // Light directions point FROM light TO surface, so negate for NdotL
    let ndl0 = max(dot(n, -u.light_dir_0.xyz), 0.0);
    let ndl1 = max(dot(n, -u.light_dir_1.xyz), 0.0);
    let ndl2 = max(dot(n, -u.light_dir_2.xyz), 0.0);

    var light = u.ambient.rgb;
    light += u.light_col_0.rgb * ndl0;
    light += u.light_col_1.rgb * ndl1;
    light += u.light_col_2.rgb * ndl2;

    // Material color: texture or flat base_color
    var albedo = u.base_color;
    if (u.use_texture.x > 0.5) {
        albedo = textureSample(t_albedo, s_albedo, in.uv);
    }

    var color = albedo.rgb * light;

    // Simple Reinhard tone mapping
    color = color / (color + vec3<f32>(1.0));

    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, albedo.a);
}

// ── Blit shader: fullscreen textured quad ──

struct BlitVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_blit(@builtin(vertex_index) vertex_index: u32) -> BlitVertexOutput {
    // Fullscreen triangle trick (3 vertices cover the screen)
    var out: BlitVertexOutput;
    let x = f32(i32(vertex_index & 1u) * 2 - 1);
    let y = f32(i32(vertex_index >> 1u) * 2 - 1);
    // Map to fullscreen quad using vertex_index 0,1,2 for a full-screen triangle
    let pos = vec2<f32>(
        f32(i32(vertex_index) / 2) * 4.0 - 1.0,
        f32(i32(vertex_index) % 2) * 4.0 - 1.0,
    );
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = vec2<f32>(
        (pos.x + 1.0) * 0.5,
        (1.0 - pos.y) * 0.5,  // flip Y for texture coordinates
    );
    return out;
}

@group(0) @binding(0) var t_color: texture_2d<f32>;
@group(0) @binding(1) var s_color: sampler;

@fragment
fn fs_blit(in: BlitVertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_color, s_color, in.uv);
}
