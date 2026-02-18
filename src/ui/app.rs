//! Main egui application.

use std::sync::Arc;

use eframe::egui;

use crate::AppState;

/// The native egui application window.
pub struct RustuberApp {
    state: Arc<AppState>,
}

impl RustuberApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, state: Arc<AppState>) -> Self {
        Self { state }
    }

    /// Launch the native UI window. Blocks until the window is closed.
    pub fn run(state: Arc<AppState>) -> eframe::Result {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_title("rustuber")
                .with_inner_size([800.0, 600.0]),
            ..Default::default()
        };

        eframe::run_native(
            "rustuber",
            options,
            Box::new(move |cc| Ok(Box::new(Self::new(cc, state)))),
        )
    }
}

impl eframe::App for RustuberApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.label("rustuber");
                ui.separator();
                ui.label("headless VTuber engine");
            });
        });

        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.heading("Controls");
            ui.separator();

            // OBS status
            let obs_connected = self.state.obs_connected.load(std::sync::atomic::Ordering::Relaxed);
            ui.horizontal(|ui| {
                ui.label("OBS:");
                if obs_connected {
                    ui.colored_label(egui::Color32::GREEN, "connected");
                } else {
                    ui.colored_label(egui::Color32::RED, "disconnected");
                }
            });

            ui.separator();
            ui.label("Avatar, audio, and style controls will go here.");
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Avatar Preview");
            ui.label("Avatar viewport will render here.");
        });

        // Repaint continuously for real-time updates
        ctx.request_repaint();
    }
}
