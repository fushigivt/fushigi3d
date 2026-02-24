// Bloom bright-pass: extract pixels above luminance threshold.

struct Params {
    // x: threshold, y: soft_knee
    params: vec4<f32>,
};

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;
@group(0) @binding(2) var<uniform> params: Params;

@fragment
fn fs_bloom_threshold(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let color = textureSample(t_input, s_input, in.uv);
    let luma = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let threshold = params.params.x;
    let knee = params.params.y;

    let soft = luma - threshold + knee;
    let soft_clamped = clamp(soft, 0.0, 2.0 * knee);
    let contribution = select(
        luma - threshold,
        soft_clamped * soft_clamped / (4.0 * knee + 0.00001),
        knee > 0.0
    );
    let factor = max(contribution, 0.0) / max(luma, 0.00001);

    return vec4<f32>(color.rgb * factor, color.a);
}
