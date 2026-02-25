mod system;

pub use system::{ParticleGpuData, ParticleSystem, StickerPreset};

/// WGSL shader source for particle billboard rendering.
pub const PARTICLE_SHADER_WGSL: &str = include_str!("../../shaders/particle.wgsl");
