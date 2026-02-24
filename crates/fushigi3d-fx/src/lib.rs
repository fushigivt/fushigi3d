pub mod effect;
pub mod effects;
pub mod fullscreen;

use wgpu;

use effect::PostProcessEffect;

/// HDR texture format used internally by the post-processing chain.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Manages a sequence of post-processing effects with ping-pong textures.
pub struct PostProcessChain {
    effects: Vec<Box<dyn PostProcessEffect>>,
    ping: wgpu::Texture,
    pong: wgpu::Texture,
    width: u32,
    height: u32,
}

impl PostProcessChain {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let ping = create_pp_texture(device, width, height, "fx_ping");
        let pong = create_pp_texture(device, width, height, "fx_pong");

        Self {
            effects: Vec::new(),
            ping,
            pong,
            width,
            height,
        }
    }

    pub fn push(&mut self, effect: Box<dyn PostProcessEffect>) {
        self.effects.push(effect);
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        self.width = width;
        self.height = height;
        self.ping = create_pp_texture(device, width, height, "fx_ping");
        self.pong = create_pp_texture(device, width, height, "fx_pong");
        for effect in &mut self.effects {
            effect.resize(device, width, height);
        }
    }

    pub fn set_params(&mut self, queue: &wgpu::Queue) {
        for effect in &mut self.effects {
            effect.set_params(queue);
        }
    }

    /// Run all enabled effects. Reads from `scene_view`, writes final result to `output_view`.
    ///
    /// Caller should check `has_enabled_effects()` first. If false, use scene_view directly
    /// without calling this method.
    pub fn run(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        scene_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
    ) {
        let enabled: Vec<&dyn PostProcessEffect> = self
            .effects
            .iter()
            .filter(|e| e.enabled())
            .map(|e| e.as_ref())
            .collect();

        if enabled.is_empty() {
            return;
        }

        let ping_view = self.ping.create_view(&Default::default());
        let pong_view = self.pong.create_view(&Default::default());

        let mut current_input = scene_view;
        let mut use_ping_as_output = true;

        for (i, effect) in enabled.iter().enumerate() {
            let is_last = i == enabled.len() - 1;
            let current_output = if is_last {
                output_view
            } else if use_ping_as_output {
                &ping_view
            } else {
                &pong_view
            };

            effect.apply(device, encoder, current_input, current_output);

            if !is_last {
                current_input = if use_ping_as_output {
                    &ping_view
                } else {
                    &pong_view
                };
                use_ping_as_output = !use_ping_as_output;
            }
        }
    }

    pub fn has_enabled_effects(&self) -> bool {
        self.effects.iter().any(|e| e.enabled())
    }

    pub fn effects_mut(&mut self) -> &mut Vec<Box<dyn PostProcessEffect>> {
        &mut self.effects
    }
}

fn create_pp_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}
