// Bloom combine: additive blend of blurred bright pixels onto scene.

struct Params {
    // x: bloom intensity
    params: vec4<f32>,
};

@group(0) @binding(0) var t_scene: texture_2d<f32>;
@group(0) @binding(1) var s_scene: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var t_bloom: texture_2d<f32>;

@fragment
fn fs_bloom_combine(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(t_scene, s_scene, in.uv);
    let bloom = textureSample(t_bloom, s_scene, in.uv);
    return vec4<f32>(scene.rgb + bloom.rgb * params.params.x, scene.a);
}
