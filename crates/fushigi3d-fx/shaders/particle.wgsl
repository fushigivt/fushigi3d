// Billboard particle rendering with procedural SDF shapes.

struct CameraUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) shape: u32,
};

@vertex
fn vs_particle(
    @location(0) quad_offset: vec2<f32>,
    @location(1) inst_pos: vec3<f32>,
    @location(2) inst_size: f32,
    @location(3) inst_color: vec4<f32>,
    @location(4) inst_rotation: f32,
    @location(5) inst_shape: u32,
    @location(6) inst_alpha: f32,
) -> VertexOutput {
    // Extract camera right and up from view matrix
    let right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let up = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);

    // Rotate quad offset
    let c = cos(inst_rotation);
    let s = sin(inst_rotation);
    let rot = vec2<f32>(
        quad_offset.x * c - quad_offset.y * s,
        quad_offset.x * s + quad_offset.y * c,
    );

    // Billboard: expand quad in world space
    let scale = inst_size * 0.04;
    let world_pos = inst_pos + right * rot.x * scale + up * rot.y * scale;

    var out: VertexOutput;
    out.clip_position = camera.proj * camera.view * vec4<f32>(world_pos, 1.0);
    out.uv = quad_offset + 0.5;
    out.color = vec4<f32>(inst_color.rgb, inst_alpha);
    out.shape = inst_shape;
    return out;
}

// --- SDF shape functions ---
// Each returns 1.0 inside the shape, 0.0 outside, with anti-aliased edges.

// Signed distance to heart (Inigo Quilez). Negative inside, positive outside.
fn sdf_heart(p_in: vec2<f32>) -> f32 {
    var p = vec2<f32>(abs(p_in.x), p_in.y);
    if p.y + p.x > 1.0 {
        return length(p - vec2<f32>(0.25, 0.75)) - 0.3536;
    }
    let a = p - vec2<f32>(0.0, 1.0);
    let half = 0.5 * max(p.x + p.y, 0.0);
    let b = p - vec2<f32>(half, half);
    return sqrt(min(dot(a, a), dot(b, b))) * sign(p.x - p.y);
}

fn shape_heart(uv: vec2<f32>) -> f32 {
    var p = (uv - 0.5) * 2.2;
    p.y = -p.y + 0.3;
    let d = sdf_heart(p);
    let aa = max(fwidth(d), 0.001);
    return 1.0 - smoothstep(-aa, aa, d);
}

fn shape_star(uv: vec2<f32>) -> f32 {
    let p = (uv - 0.5) * 2.5;
    let r = length(p);
    let angle = atan2(p.y, p.x);
    // 4-pointed sparkle shape
    let star = pow(abs(cos(angle * 2.0)), 0.5);
    let d = r - mix(0.15, 0.9, star);
    let aa = max(fwidth(d), 0.001);
    return 1.0 - smoothstep(-aa, aa, d);
}

fn shape_rect(uv: vec2<f32>) -> f32 {
    let p = abs(uv - 0.5);
    let d = max(p.x - 0.35, p.y - 0.2);
    let aa = max(fwidth(d), 0.001);
    return 1.0 - smoothstep(-aa, aa, d);
}

fn shape_circle(uv: vec2<f32>) -> f32 {
    let d = length(uv - 0.5) - 0.4;
    let aa = max(fwidth(d), 0.001);
    return 1.0 - smoothstep(-aa, aa, d);
}

@fragment
fn fs_particle(in: VertexOutput) -> @location(0) vec4<f32> {
    var mask: f32;
    switch in.shape {
        case 0u: { mask = shape_heart(in.uv); }
        case 1u: { mask = shape_star(in.uv); }
        case 2u: { mask = shape_rect(in.uv); }
        default: { mask = shape_circle(in.uv); }
    }

    let final_alpha = mask * in.color.a;
    if final_alpha < 0.01 {
        discard;
    }
    return vec4<f32>(in.color.rgb, final_alpha);
}
