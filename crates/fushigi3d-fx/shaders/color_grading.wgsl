// Color grading: brightness, contrast, saturation, hue shift.

struct Params {
    // x: brightness, y: contrast, z: saturation, w: hue_shift (radians)
    params: vec4<f32>,
};

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;
@group(0) @binding(2) var<uniform> params: Params;

fn rgb_to_hsv(c: vec3<f32>) -> vec3<f32> {
    let cmax = max(c.r, max(c.g, c.b));
    let cmin = min(c.r, min(c.g, c.b));
    let delta = cmax - cmin;
    var h = 0.0;
    if (delta > 0.001) {
        if (cmax == c.r) {
            h = ((c.g - c.b) / delta) % 6.0;
        } else if (cmax == c.g) {
            h = (c.b - c.r) / delta + 2.0;
        } else {
            h = (c.r - c.g) / delta + 4.0;
        }
        h /= 6.0;
        if (h < 0.0) { h += 1.0; }
    }
    let s = select(0.0, delta / cmax, cmax > 0.0);
    return vec3<f32>(h, s, cmax);
}

fn hsv_to_rgb(c: vec3<f32>) -> vec3<f32> {
    let h = c.x * 6.0;
    let s = c.y;
    let v = c.z;
    let hi = i32(floor(h)) % 6;
    let f = h - floor(h);
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    switch (hi) {
        case 0: { return vec3<f32>(v, t, p); }
        case 1: { return vec3<f32>(q, v, p); }
        case 2: { return vec3<f32>(p, v, t); }
        case 3: { return vec3<f32>(p, q, v); }
        case 4: { return vec3<f32>(t, p, v); }
        default: { return vec3<f32>(v, p, q); }
    }
}

@fragment
fn fs_color_grading(in: FullscreenOutput) -> @location(0) vec4<f32> {
    var color = textureSample(t_input, s_input, in.uv);

    // Brightness
    var c = color.rgb * params.params.x;

    // Contrast (around 0.5 mid-gray)
    c = (c - 0.5) * params.params.y + 0.5;

    // Saturation
    let luma = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    c = mix(vec3<f32>(luma), c, params.params.z);

    // Hue shift
    if (abs(params.params.w) > 0.001) {
        var hsv = rgb_to_hsv(max(c, vec3<f32>(0.0)));
        hsv.x = fract(hsv.x + params.params.w / 6.283185);
        c = hsv_to_rgb(hsv);
    }

    return vec4<f32>(max(c, vec3<f32>(0.0)), color.a);
}
