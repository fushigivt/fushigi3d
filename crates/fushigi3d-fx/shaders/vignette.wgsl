// Vignette: radial edge darkening.

struct Params {
    // x: intensity, y: radius, z: softness
    params: vec4<f32>,
};

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;
@group(0) @binding(2) var<uniform> params: Params;

@fragment
fn fs_vignette(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let color = textureSample(t_input, s_input, in.uv);
    let center = vec2<f32>(0.5, 0.5);
    let dist = distance(in.uv, center);
    let vignette = 1.0 - smoothstep(params.params.y, params.params.y + params.params.z, dist) * params.params.x;
    return vec4<f32>(color.rgb * vignette, color.a);
}
