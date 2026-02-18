//! egui-wgpu `CallbackTrait` implementation for the VRM viewport.
//!
//! Bridges between egui's paint callback system and our offscreen wgpu renderer.
//! `prepare()` runs the offscreen 3D render, `paint()` blits the result into
//! the egui render pass.

#![cfg(feature = "native-ui")]

use eframe::egui_wgpu;
use eframe::wgpu;
use std::sync::Arc;

use super::renderer::VrmRenderer;

/// Paint callback that blits the VRM offscreen texture into the egui render pass.
pub struct VrmViewportCallback {
    pub renderer: Arc<VrmRenderer>,
    pub viewport_width: u32,
    pub viewport_height: u32,
}

impl egui_wgpu::CallbackTrait for VrmViewportCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        _callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // Resize offscreen texture if viewport changed
        self.renderer
            .resize(device, self.viewport_width, self.viewport_height);
        self.renderer.render_offscreen(device, queue);
        Vec::new()
    }

    fn paint(
        &self,
        _info: eframe::egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        _callback_resources: &egui_wgpu::CallbackResources,
    ) {
        self.renderer.blit(render_pass);
    }
}
