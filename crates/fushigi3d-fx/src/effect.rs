use std::any::Any;
use wgpu;

/// A single post-processing effect that reads one texture and writes another.
pub trait PostProcessEffect: Send + Sync {
    /// Upload any changed uniform data to the GPU.
    fn set_params(&mut self, queue: &wgpu::Queue);

    /// Recreate internal textures after a viewport resize.
    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32);

    /// Encode a fullscreen pass: read `input`, write `output`.
    fn apply(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
    );

    /// Whether this effect is currently active.
    fn enabled(&self) -> bool;

    /// Human-readable name for UI display.
    fn name(&self) -> &str;

    /// Downcast support for typed access from the application.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
