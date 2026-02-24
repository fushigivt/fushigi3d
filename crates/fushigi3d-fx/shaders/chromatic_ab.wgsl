// Chromatic aberration: RGB channel UV offset.

struct Params {
    // x: intensity (pixel offset amount)
    params: vec4<f32>,
};

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;
@group(0) @binding(2) var<uniform> params: Params;

@fragment
fn fs_chromatic_ab(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let center = vec2<f32>(0.5, 0.5);
    let dir = in.uv - center;
    let offset = dir * params.params.x;

    let r = textureSample(t_input, s_input, in.uv + offset).r;
    let g = textureSample(t_input, s_input, in.uv).g;
    let b = textureSample(t_input, s_input, in.uv - offset).b;
    let a = textureSample(t_input, s_input, in.uv).a;

    return vec4<f32>(r, g, b, a);
}
