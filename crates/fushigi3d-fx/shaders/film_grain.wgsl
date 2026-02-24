// Film grain: procedural noise overlay.

struct Params {
    // x: intensity, y: time
    params: vec4<f32>,
};

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;
@group(0) @binding(2) var<uniform> params: Params;

fn hash(p: vec2<f32>) -> f32 {
    let p2 = fract(p * vec2<f32>(443.8975, 397.2973));
    let p3 = dot(p2, p2 + 19.19);
    return fract(p3 * p3);
}

@fragment
fn fs_film_grain(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let color = textureSample(t_input, s_input, in.uv);
    let noise = hash(in.uv * 1000.0 + params.params.y * 100.0) * 2.0 - 1.0;
    let grain = noise * params.params.x;
    return vec4<f32>(color.rgb + grain, color.a);
}
