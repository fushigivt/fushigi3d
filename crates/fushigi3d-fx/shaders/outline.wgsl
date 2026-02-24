// Screen-space outline via Sobel edge detection on depth buffer.

struct Params {
    // x: thickness, y: threshold, z: 1/width, w: 1/height
    params: vec4<f32>,
    // outline color (rgb) + strength (a)
    color: vec4<f32>,
};

@group(0) @binding(0) var t_color: texture_2d<f32>;
@group(0) @binding(1) var s_color: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var t_depth: texture_2d<f32>;

fn sample_depth(uv: vec2<f32>) -> f32 {
    return textureSample(t_depth, s_color, uv).r;
}

fn linearize_depth(d: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - d * (far - near));
}

@fragment
fn fs_outline(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let color = textureSample(t_color, s_color, in.uv);
    let thickness = params.params.x;
    let threshold = params.params.y;
    let texel = vec2<f32>(params.params.z, params.params.w) * thickness;

    let near = 0.01;
    let far = 100.0;

    // Sample 3x3 neighborhood depths (linearized)
    let d00 = linearize_depth(sample_depth(in.uv + vec2(-texel.x, -texel.y)), near, far);
    let d10 = linearize_depth(sample_depth(in.uv + vec2(0.0, -texel.y)), near, far);
    let d20 = linearize_depth(sample_depth(in.uv + vec2(texel.x, -texel.y)), near, far);
    let d01 = linearize_depth(sample_depth(in.uv + vec2(-texel.x, 0.0)), near, far);
    let d21 = linearize_depth(sample_depth(in.uv + vec2(texel.x, 0.0)), near, far);
    let d02 = linearize_depth(sample_depth(in.uv + vec2(-texel.x, texel.y)), near, far);
    let d12 = linearize_depth(sample_depth(in.uv + vec2(0.0, texel.y)), near, far);
    let d22 = linearize_depth(sample_depth(in.uv + vec2(texel.x, texel.y)), near, far);

    // Sobel X: [-1 0 1; -2 0 2; -1 0 1]
    let gx = -d00 + d20 - 2.0 * d01 + 2.0 * d21 - d02 + d22;
    // Sobel Y: [-1 -2 -1; 0 0 0; 1 2 1]
    let gy = -d00 - 2.0 * d10 - d20 + d02 + 2.0 * d12 + d22;

    let edge = sqrt(gx * gx + gy * gy);
    let edge_strength = smoothstep(threshold, threshold + 0.02, edge) * params.color.a;

    let result = mix(color.rgb, params.color.rgb, edge_strength);
    return vec4<f32>(result, color.a);
}
