// Tonemapping post-process: HDR → LDR conversion.
// FullscreenOutput is provided by fullscreen.wgsl (prepended at build time).

struct Params {
    // x: mode (0 = Reinhard, 1 = ACES), y: exposure
    mode_exposure: vec4<f32>,
};

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;
@group(0) @binding(2) var<uniform> params: Params;

fn reinhard(color: vec3<f32>) -> vec3<f32> {
    return color / (color + vec3<f32>(1.0));
}

fn aces(color: vec3<f32>) -> vec3<f32> {
    // ACES filmic tone mapping (Narkowicz 2015)
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

@fragment
fn fs_tonemap(in: FullscreenOutput) -> @location(0) vec4<f32> {
    var color = textureSample(t_input, s_input, in.uv);
    let exposure = params.mode_exposure.y;
    var hdr = color.rgb * exposure;

    var ldr: vec3<f32>;
    if (params.mode_exposure.x < 0.5) {
        ldr = reinhard(hdr);
    } else {
        ldr = aces(hdr);
    }

    return vec4<f32>(ldr, color.a);
}
