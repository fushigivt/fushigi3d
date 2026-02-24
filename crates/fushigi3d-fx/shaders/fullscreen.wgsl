// Shared fullscreen triangle vertex shader.
// Generates a single triangle that covers the entire screen.
// vertex_index: 0 → (-1, -1), 1 → (3, -1), 2 → (-1, 3)

struct FullscreenOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> FullscreenOutput {
    var out: FullscreenOutput;
    let x = f32(i32(vertex_index) / 2) * 4.0 - 1.0;
    let y = f32(i32(vertex_index) % 2) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(
        (x + 1.0) * 0.5,
        (1.0 - y) * 0.5,
    );
    return out;
}
