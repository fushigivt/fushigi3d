// Bloom separable Gaussian blur (9-tap).

struct Params {
    // x: direction_x, y: direction_y (texel size in blur direction)
    params: vec4<f32>,
};

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;
@group(0) @binding(2) var<uniform> params: Params;

@fragment
fn fs_bloom_blur(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let dir = vec2<f32>(params.params.x, params.params.y);

    // 9-tap Gaussian weights (sigma ~= 4)
    let w0 = 0.227027;
    let w1 = 0.194596;
    let w2 = 0.121621;
    let w3 = 0.054054;
    let w4 = 0.016216;

    var color = textureSample(t_input, s_input, in.uv) * w0;
    color += textureSample(t_input, s_input, in.uv + dir * 1.0) * w1;
    color += textureSample(t_input, s_input, in.uv - dir * 1.0) * w1;
    color += textureSample(t_input, s_input, in.uv + dir * 2.0) * w2;
    color += textureSample(t_input, s_input, in.uv - dir * 2.0) * w2;
    color += textureSample(t_input, s_input, in.uv + dir * 3.0) * w3;
    color += textureSample(t_input, s_input, in.uv - dir * 3.0) * w3;
    color += textureSample(t_input, s_input, in.uv + dir * 4.0) * w4;
    color += textureSample(t_input, s_input, in.uv - dir * 4.0) * w4;

    return color;
}
