//! Rustuber - Headless VTuber/PNGTuber Service
//!
//! Main entry point for the CLI application.

use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{error, info, warn, Level};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use rustuber::{
    audio::AudioPipeline,
    config::Config,
    output::{browser::BrowserServer, obs::ObsClient},
    tracking::{
        mediapipe::MpReceiver,
        osf::OsfReceiver,
        subprocess::{check_mediapipe_available, MpSubprocess, OsfSubprocess},
        vmc::{self, VmcReceiver},
    },
    web::WebServer,
    AppState,
};

/// Rustuber - Headless VTuber/PNGTuber Service for Linux
#[derive(Parser, Debug)]
#[command(name = "rustuber", version, about, long_about = None)]
struct Args {
    /// Configuration file path
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Audio input device (overrides config)
    #[arg(short, long)]
    device: Option<String>,

    /// List available audio devices and exit
    #[arg(long)]
    list_devices: bool,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Disable OBS integration
    #[arg(long)]
    no_obs: bool,

    /// Disable audio capture
    #[arg(long)]
    no_audio: bool,

    /// Disable HTTP server
    #[arg(long)]
    no_http: bool,

    /// HTTP server port (overrides config)
    #[arg(short, long)]
    port: Option<u16>,

    /// Launch native UI window
    #[cfg(feature = "native-ui")]
    #[arg(long)]
    ui: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(
            EnvFilter::builder()
                .with_default_directive(log_level.into())
                .from_env_lossy(),
        )
        .init();

    info!(
        "Starting {} v{}",
        rustuber::NAME,
        rustuber::VERSION
    );

    // Handle list-devices mode
    if args.list_devices {
        list_audio_devices();
        return Ok(());
    }

    // Build tokio runtime manually so the main thread stays free for the UI event loop
    let runtime = tokio::runtime::Runtime::new()?;

    // Do all async setup on the runtime
    let state = runtime.block_on(async {
        setup_and_spawn_services(&args).await
    })?;

    // If UI requested, run eframe on the main thread (blocks until window closes)
    #[cfg(feature = "native-ui")]
    if args.ui {
        info!("Launching native UI window");
        let ui_state = Arc::clone(&state);

        // Enter the tokio runtime context so try_current() works inside eframe
        // (needed for reading config via tokio::sync::RwLock in init_vrm)
        let _guard = runtime.enter();

        // eframe::run_native blocks the main thread (winit requirement)
        if let Err(e) = rustuber::ui::RustuberApp::run(ui_state) {
            error!("UI error: {}", e);
        }

        info!("UI window closed, shutting down");
        state.shutdown();

        // Give async tasks a moment to finish
        runtime.shutdown_timeout(std::time::Duration::from_secs(3));
        return Ok(());
    }

    // Headless mode: wait for Ctrl+C / SIGTERM
    runtime.block_on(async {
        shutdown_signal().await;
        info!("Shutdown signal received");
        state.shutdown();

        // Give tasks a moment to clean up
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    });

    info!("Rustuber stopped");
    Ok(())
}

/// Setup config, create AppState, and spawn all background services.
async fn setup_and_spawn_services(args: &Args) -> anyhow::Result<Arc<AppState>> {
    // Load configuration
    let mut config = if let Some(ref path) = args.config {
        Config::from_file(path)?
    } else {
        Config::load()?
    };

    // Apply CLI overrides
    if let Some(ref device) = args.device {
        config.audio.device = device.clone();
    }
    if args.no_obs {
        config.obs.enabled = false;
    }
    if args.no_audio {
        config.audio.enabled = false;
    }
    if args.no_http {
        config.http.enabled = false;
    }
    if let Some(port) = args.port {
        config.http.port = port;
    }

    // Validate configuration
    config.validate()?;

    info!("Audio device: {}", config.audio.device);
    info!("VAD provider: {:?}", config.vad.provider);
    info!("OBS integration: {}", config.obs.enabled);
    info!("HTTP server: {}", config.http.enabled);

    // Create shared application state
    let state = AppState::new(config.clone());

    // Start audio pipeline
    if config.audio.enabled {
        let audio_state = Arc::clone(&state);
        tokio::spawn(async move {
            if let Err(e) = run_audio_pipeline(audio_state).await {
                error!("Audio pipeline error: {}", e);
            }
        });
    } else {
        info!("Audio disabled");
    }

    // Start OBS client
    let obs_state = Arc::clone(&state);
    tokio::spawn(async move {
        if let Err(e) = run_obs_client(obs_state).await {
            error!("OBS client error: {}", e);
        }
    });

    // Start HTTP server if enabled
    if config.http.enabled {
        let http_state = Arc::clone(&state);
        tokio::spawn(async move {
            if let Err(e) = run_http_server(http_state).await {
                error!("HTTP server error: {}", e);
            }
        });
    }

    // Auto-detect MediaPipe if no tracker is explicitly enabled
    if !config.osf.enabled && !config.vmc.receiver_enabled && !config.mediapipe.enabled {
        if check_mediapipe_available() {
            info!("No tracker enabled — MediaPipe detected, auto-enabling");
            config.mediapipe.enabled = true;
            config.mediapipe.auto_launch = true;
            let state_inner = Arc::clone(&state);
            let mp_config = config.mediapipe.clone();
            {
                let mut cfg = state_inner.config.write().await;
                cfg.mediapipe = mp_config;
            }
        }
    }

    // Start OpenSeeFace tracking if enabled
    if config.osf.enabled {
        let osf_state = Arc::clone(&state);
        tokio::spawn(async move {
            if let Err(e) = run_osf_tracking(osf_state).await {
                error!("OSF tracking error: {}", e);
            }
        });
    }

    // Start VMC tracking if enabled
    if config.vmc.receiver_enabled {
        let vmc_state = Arc::clone(&state);
        tokio::spawn(async move {
            if let Err(e) = run_vmc_tracking(vmc_state).await {
                error!("VMC tracking error: {}", e);
            }
        });
    }

    // Start MediaPipe tracking if enabled
    if config.mediapipe.enabled {
        let mp_state = Arc::clone(&state);
        tokio::spawn(async move {
            if let Err(e) = run_mediapipe_tracking(mp_state).await {
                error!("MediaPipe tracking error: {}", e);
            }
        });
    }

    Ok(state)
}

fn list_audio_devices() {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();

    println!("Available audio input devices:\n");

    if let Some(device) = host.default_input_device() {
        if let Ok(name) = device.name() {
            println!("  * {} (default)", name);
        }
    }

    if let Ok(devices) = host.input_devices() {
        for device in devices {
            if let Ok(name) = device.name() {
                println!("    {}", name);
            }
        }
    }
}

async fn run_audio_pipeline(state: Arc<AppState>) -> anyhow::Result<()> {
    let mut shutdown_rx = state.subscribe_shutdown();

    loop {
        let config = state.config.read().await;
        let mut pipeline = match AudioPipeline::new(&config.audio, &config.vad) {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to create audio pipeline: {}", e);
                drop(config); // Release read lock before waiting
                tokio::select! {
                    _ = state.wait_audio_restart() => continue,
                    _ = shutdown_rx.recv() => return Ok(()),
                }
            }
        };
        drop(config);

        info!("Audio pipeline started");

        loop {
            tokio::select! {
                result = pipeline.process() => {
                    match result {
                        Ok(is_speaking) => {
                            // Publish audio levels for UI display
                            let activity = pipeline.get_activity();
                            state.set_audio_levels(activity.energy_db, activity.confidence);

                            let current = state.get_avatar_state().await;
                            // Only update if speaking state changed
                            if is_speaking != current.is_speaking() {
                                let old_state = current.state_type();
                                let new_state = current.with_speaking(is_speaking);
                                info!("State change: {:?} -> {:?}", old_state, new_state.state_type());
                                state.update_avatar_state(new_state).await;
                            }
                        }
                        Err(e) => {
                            let msg = e.to_string();
                            if msg.contains("Stream disconnected") {
                                warn!("Audio device unavailable, waiting for restart signal");
                                tokio::select! {
                                    _ = state.wait_audio_restart() => break,
                                    _ = shutdown_rx.recv() => return Ok(()),
                                }
                            }
                            error!("Audio processing error: {}", e);
                            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                        }
                    }
                }
                _ = state.wait_audio_restart() => {
                    info!("Audio device changed, restarting pipeline");
                    break; // breaks inner loop; pipeline is dropped at end of
                           // outer-loop body — AudioCapture::drop() joins the
                           // capture thread so the ALSA device is fully released
                           // before the next iteration reopens it.
                }
                _ = shutdown_rx.recv() => {
                    info!("Audio pipeline shutting down");
                    return Ok(());
                }
            }
        }
    }
}

async fn run_obs_client(state: Arc<AppState>) -> anyhow::Result<()> {
    let mut state_rx = state.subscribe_state();
    let mut shutdown_rx = state.subscribe_shutdown();

    loop {
        // Get current config
        let config = state.config.read().await;
        let obs_config = config.obs.clone();
        let obs_enabled = obs_config.enabled;
        drop(config);

        if !obs_enabled {
            info!("OBS integration disabled, waiting for config change or reconnect signal");
            state.set_obs_connected(false);
            state.set_obs_scenes(Vec::new()).await;

            tokio::select! {
                _ = state.wait_obs_reconnect() => {
                    info!("OBS reconnect signal received");
                    continue;
                }
                _ = state.wait_config_changed() => {
                    info!("Config changed, rechecking OBS settings");
                    continue;
                }
                _ = shutdown_rx.recv() => {
                    info!("OBS client shutting down");
                    return Ok(());
                }
            }
        }

        let mut client = ObsClient::new(&obs_config);

        // Connect to OBS
        match client.connect().await {
            Ok(()) => {
                info!("OBS client connected");
                state.set_obs_connected(true);

                // Fetch and cache scenes
                match client.list_scenes().await {
                    Ok(scenes) => {
                        info!("Loaded {} OBS scenes", scenes.len());
                        state.set_obs_scenes(scenes).await;
                    }
                    Err(e) => {
                        error!("Failed to list OBS scenes: {}", e);
                    }
                }
            }
            Err(e) => {
                error!("Failed to connect to OBS: {}", e);
                state.set_obs_connected(false);
                state.set_obs_scenes(Vec::new()).await;

                // Wait for reconnect signal, config change, or shutdown
                tokio::select! {
                    _ = state.wait_obs_reconnect() => {
                        info!("OBS reconnect signal received, retrying...");
                        continue;
                    }
                    _ = state.wait_config_changed() => {
                        info!("Config changed, retrying OBS connection...");
                        continue;
                    }
                    _ = shutdown_rx.recv() => {
                        info!("OBS client shutting down");
                        return Ok(());
                    }
                    _ = tokio::time::sleep(tokio::time::Duration::from_secs(10)) => {
                        info!("Retrying OBS connection...");
                        continue;
                    }
                }
            }
        }

        // Main event loop while connected
        loop {
            tokio::select! {
                result = state_rx.recv() => {
                    match result {
                        Ok(avatar_state) => {
                            if let Err(e) = client.update_state(&avatar_state).await {
                                error!("Failed to update OBS: {}", e);
                                state.set_obs_connected(false);
                                break; // Reconnect
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                            // Missed some messages, continue
                            continue;
                        }
                        Err(e) => {
                            error!("State receiver error: {}", e);
                            break;
                        }
                    }
                }
                _ = state.wait_obs_reconnect() => {
                    info!("OBS reconnect signal received");
                    state.set_obs_connected(false);
                    break; // Reconnect
                }
                _ = state.wait_config_changed() => {
                    info!("Config changed, reconnecting OBS...");
                    state.set_obs_connected(false);
                    break; // Reconnect with new config
                }
                _ = shutdown_rx.recv() => {
                    info!("OBS client shutting down");
                    state.set_obs_connected(false);
                    return Ok(());
                }
            }
        }
    }
}

async fn run_http_server(state: Arc<AppState>) -> anyhow::Result<()> {
    let config = state.config.read().await;
    let http_config = config.http.clone();
    let avatar_config = config.avatar.clone();
    drop(config);

    // Start browser source server (serves avatar for OBS browser source)
    let browser_server = BrowserServer::new(state.clone(), &avatar_config);

    // Start web dashboard server
    let web_server = WebServer::new(state.clone(), &http_config);

    let addr = format!("{}:{}", http_config.host, http_config.port);
    info!("HTTP server listening on {}", addr);

    // Combine both servers using axum's merge
    let app = web_server.router().merge(browser_server.router());

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    let mut shutdown_rx = state.subscribe_shutdown();

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = shutdown_rx.recv().await;
        })
        .await?;

    info!("HTTP server stopped");
    Ok(())
}

async fn run_osf_tracking(state: Arc<AppState>) -> anyhow::Result<()> {
    let config = state.config.read().await;
    let osf_config = config.osf.clone();
    drop(config);

    let mut shutdown_rx = state.subscribe_shutdown();

    // Optionally launch the subprocess
    let mut subprocess = if osf_config.auto_launch {
        let mut sp = OsfSubprocess::new(&osf_config);
        if let Err(e) = sp.start() {
            error!("Failed to auto-launch OpenSeeFace: {}", e);
            // Continue anyway — user may have it running externally
        }
        // Give the tracker a moment to start listening
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        Some(sp)
    } else {
        None
    };

    // Start the receiver
    let mut receiver = OsfReceiver::new(&osf_config);
    receiver.start()?;

    info!(
        "OSF tracking started (port: {}, blend_with_vad: {})",
        osf_config.port, osf_config.blend_with_vad
    );

    loop {
        tokio::select! {
            result = receiver.process() => {
                match result {
                    Ok(Some(data)) if data.has_data => {
                        let current = state.get_avatar_state().await;
                        let new_state = data.to_avatar_state(&current, osf_config.blend_with_vad);
                        if new_state != current {
                            state.update_avatar_state(new_state).await;
                        }
                    }
                    Ok(_) => {}
                    Err(e) => {
                        error!("OSF receive error: {}", e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    }
                }

                // Check subprocess health and auto-restart if needed
                if let Some(ref mut sp) = subprocess {
                    if !sp.is_running() && osf_config.auto_restart {
                        info!(
                            "OpenSeeFace subprocess crashed, restarting in {}s",
                            osf_config.restart_delay_secs
                        );
                        tokio::time::sleep(tokio::time::Duration::from_secs(
                            osf_config.restart_delay_secs,
                        ))
                        .await;
                        if let Err(e) = sp.start() {
                            error!("Failed to restart OpenSeeFace: {}", e);
                        }
                    }
                }
            }
            _ = shutdown_rx.recv() => {
                info!("OSF tracking shutting down");
                break;
            }
        }

        // Small yield to avoid busy-spinning when no data arrives
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
    }

    // Cleanup
    receiver.stop();
    if let Some(ref mut sp) = subprocess {
        sp.stop().await;
    }

    Ok(())
}

async fn run_vmc_tracking(state: Arc<AppState>) -> anyhow::Result<()> {
    let config = state.config.read().await;
    let vmc_config = config.vmc.clone();
    drop(config);

    let mut shutdown_rx = state.subscribe_shutdown();

    let mut receiver = VmcReceiver::new(&vmc_config);
    receiver.start()?;

    info!(
        "VMC tracking started (port: {}, blend_with_vad: {})",
        vmc_config.receiver_port, vmc_config.blend_with_vad
    );

    loop {
        tokio::select! {
            result = receiver.process() => {
                match result {
                    Ok(Some(data)) if data.has_data => {
                        let current = state.get_avatar_state().await;

                        // Map VMC data to avatar state
                        let mouth_open = vmc::blendshapes::mouth_open(&data.blendshapes);
                        let blink = vmc::blendshapes::average_blink(&data.blendshapes);
                        let head_euler = vmc::quaternion_to_euler(data.head_rotation);

                        let mut new_state = current.clone()
                            .with_mouth_open(mouth_open)
                            .with_blink(blink)
                            .with_head_position(data.head_position)
                            .with_head_rotation(head_euler)
                            .with_blendshapes(data.blendshapes.clone());

                        if !vmc_config.blend_with_vad {
                            new_state = new_state.with_speaking(mouth_open > 0.15);
                        }

                        // Check expression mappings
                        for (expr_name, mapping) in &vmc_config.expressions {
                            if let Some(&value) = data.blendshapes.get(&mapping.blendshape) {
                                if value > mapping.threshold {
                                    new_state = new_state.with_expression(Some(expr_name.clone()));
                                }
                            }
                        }

                        if new_state != current {
                            state.update_avatar_state(new_state).await;
                        }
                    }
                    Ok(_) => {}
                    Err(e) => {
                        error!("VMC receive error: {}", e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    }
                }
            }
            _ = shutdown_rx.recv() => {
                info!("VMC tracking shutting down");
                break;
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
    }

    receiver.stop();
    Ok(())
}

async fn run_mediapipe_tracking(state: Arc<AppState>) -> anyhow::Result<()> {
    let config = state.config.read().await;
    let mp_config = config.mediapipe.clone();
    drop(config);

    let mut shutdown_rx = state.subscribe_shutdown();

    // Optionally launch the subprocess
    let mut subprocess = if mp_config.auto_launch {
        let mut sp = MpSubprocess::new(&mp_config);
        if let Err(e) = sp.start() {
            error!("Failed to auto-launch MediaPipe tracker: {}", e);
        }
        // Give the tracker a moment to start
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        Some(sp)
    } else {
        None
    };

    // Start the receiver
    let mut receiver = MpReceiver::new(&mp_config);
    receiver.start()?;

    info!(
        "MediaPipe tracking started (port: {}, blend_with_vad: {})",
        mp_config.port, mp_config.blend_with_vad
    );

    loop {
        tokio::select! {
            result = receiver.process() => {
                match result {
                    Ok(Some(data)) if data.has_data => {
                        let current = state.get_avatar_state().await;
                        let new_state = data.to_avatar_state(&current, mp_config.blend_with_vad);
                        if new_state != current {
                            state.update_avatar_state(new_state).await;
                        }
                    }
                    Ok(_) => {}
                    Err(e) => {
                        error!("MediaPipe receive error: {}", e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    }
                }

                // Check subprocess health and auto-restart if needed
                if let Some(ref mut sp) = subprocess {
                    if !sp.is_running() && mp_config.auto_restart {
                        info!(
                            "MediaPipe subprocess crashed, restarting in {}s",
                            mp_config.restart_delay_secs
                        );
                        tokio::time::sleep(tokio::time::Duration::from_secs(
                            mp_config.restart_delay_secs,
                        ))
                        .await;
                        if let Err(e) = sp.start() {
                            error!("Failed to restart MediaPipe tracker: {}", e);
                        }
                    }
                }
            }
            _ = shutdown_rx.recv() => {
                info!("MediaPipe tracking shutting down");
                break;
            }
        }

        // Small yield to avoid busy-spinning when no data arrives
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
    }

    // Cleanup
    receiver.stop();
    if let Some(ref mut sp) = subprocess {
        sp.stop().await;
    }

    Ok(())
}

async fn shutdown_signal() {
    use tokio::signal;

    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}
