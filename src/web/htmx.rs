//! HTMX-specific handlers for the dashboard

use axum::{
    extract::{Form, State},
    http::StatusCode,
    response::{Html, IntoResponse},
};
use serde::Deserialize;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::audio::capture;
use crate::AppState;

/// Render the main dashboard page
pub async fn index_page(State(state): State<Arc<AppState>>) -> Html<String> {
    let avatar = state.get_avatar_state().await;
    let config = state.config.read().await;
    let obs_connected = state.obs_connected.load(Ordering::Relaxed);

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rustuber Dashboard</title>
    <script src="/static/htmx.min.js"></script>
    <script src="/static/sse.js"></script>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="dashboard">
        <header class="header">
            <h1>Rustuber</h1>
            <span class="version">v{version}</span>
        </header>

        <nav class="nav">
            <a href="/" class="nav-link active">Dashboard</a>
            <a href="/settings" class="nav-link">Settings</a>
            <a href="/settings/obs" class="nav-link">OBS</a>
            <a href="/avatar" class="nav-link" target="_blank">Browser Source</a>
        </nav>

        <main class="main">
            <section class="panel">
                <h2>Status</h2>
                <div id="status" hx-get="/htmx/status" hx-trigger="load, every 2s" hx-swap="innerHTML">
                    <p>State: <strong>{state}</strong></p>
                    <p>Speaking: <strong>{speaking}</strong></p>
                    <p>Expression: <strong>{expression}</strong></p>
                    <p>OBS: <strong class="{obs_class}">{obs_status}</strong></p>
                </div>
            </section>

            <section class="panel">
                <h2>Live Preview</h2>
                <div id="preview" class="preview-container" hx-ext="sse" sse-connect="/api/stream">
                    <img id="preview-image" src="/avatar/current-image" alt="Avatar Preview"
                         hx-get="/avatar/current-image" hx-trigger="sse:state" hx-swap="outerHTML">
                </div>
            </section>

            <section class="panel">
                <h2>Expressions</h2>
                <div id="expressions" hx-get="/htmx/expressions" hx-trigger="load" hx-swap="innerHTML">
                    Loading expressions...
                </div>
            </section>

            <section class="panel">
                <h2>Quick Actions</h2>
                <div class="actions">
                    <button hx-post="/api/state" hx-vals='{{"speaking": true}}' class="btn btn-primary">
                        Start Speaking
                    </button>
                    <button hx-post="/api/state" hx-vals='{{"speaking": false}}' class="btn btn-secondary">
                        Stop Speaking
                    </button>
                    <button hx-delete="/api/expression" class="btn btn-secondary">
                        Clear Expression
                    </button>
                </div>
            </section>
        </main>

        <footer class="footer">
            <p>Audio: {audio_device} | VAD: {vad_provider:?}</p>
        </footer>
    </div>
</body>
</html>"#,
        version = crate::VERSION,
        state = avatar.state_type(),
        speaking = avatar.is_speaking(),
        expression = avatar.expression().unwrap_or("none"),
        audio_device = config.audio.device,
        vad_provider = config.vad.provider,
        obs_status = if obs_connected { "Connected" } else if config.obs.enabled { "Disconnected" } else { "Disabled" },
        obs_class = if obs_connected { "status-connected" } else { "status-disconnected" },
    );

    Html(html)
}

/// Render the settings page
pub async fn settings_page(State(state): State<Arc<AppState>>) -> Html<String> {
    let config = state.config.read().await;

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Settings - Rustuber</title>
    <script src="/static/htmx.min.js"></script>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="dashboard">
        <header class="header">
            <h1>Rustuber Settings</h1>
        </header>

        <nav class="nav">
            <a href="/" class="nav-link">Dashboard</a>
            <a href="/settings" class="nav-link active">Settings</a>
            <a href="/settings/obs" class="nav-link">OBS</a>
            <a href="/avatar" class="nav-link" target="_blank">Browser Source</a>
        </nav>

        <main class="main">
            <form class="settings-form" hx-post="/htmx/update-settings" hx-swap="none">
                <section class="panel">
                    <h2>Audio Settings</h2>
                    <div class="form-group">
                        <label for="audio_device">Audio Device</label>
                        <select name="audio_device" id="audio_device" hx-get="/htmx/audio-devices" hx-trigger="load" hx-swap="innerHTML">
                            <option value="{audio_device}">{audio_device}</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="vad_threshold">VAD Threshold (dB)</label>
                        <input type="range" name="vad_threshold" id="vad_threshold"
                               min="-60" max="-20" value="{vad_threshold}"
                               oninput="document.getElementById('vad_value').textContent = this.value">
                        <span id="vad_value">{vad_threshold}</span>
                    </div>
                </section>

                <section class="panel">
                    <h2>VMC Settings</h2>
                    <div class="form-group">
                        <label>
                            <input type="checkbox" name="vmc_enabled" {vmc_checked}>
                            Enable VMC Receiver
                        </label>
                    </div>
                    <div class="form-group">
                        <label for="vmc_port">VMC Port</label>
                        <input type="number" name="vmc_port" id="vmc_port" value="{vmc_port}">
                    </div>
                </section>

                <div class="form-actions">
                    <button type="submit" class="btn btn-primary">Save Settings</button>
                </div>
            </form>
        </main>
    </div>

    <div id="notification" class="notification" style="display: none;"></div>
    <script>
        document.body.addEventListener('htmx:afterRequest', function(evt) {{
            if (evt.detail.successful) {{
                var notif = document.getElementById('notification');
                notif.textContent = 'Settings saved!';
                notif.style.display = 'block';
                setTimeout(function() {{ notif.style.display = 'none'; }}, 3000);
            }}
        }});
    </script>
</body>
</html>"#,
        audio_device = config.audio.device,
        vad_threshold = config.vad.energy_threshold_db as i32,
        vmc_checked = if config.vmc.receiver_enabled { "checked" } else { "" },
        vmc_port = config.vmc.receiver_port,
    );

    Html(html)
}

/// Render the OBS settings page
pub async fn obs_settings_page(State(state): State<Arc<AppState>>) -> Html<String> {
    let config = state.config.read().await;
    let obs_connected = state.obs_connected.load(Ordering::Relaxed);
    let scenes = state.obs_scenes.read().await.clone();

    let scene_options: String = scenes.iter()
        .map(|s| format!(r#"<option value="{}">{}</option>"#, s, s))
        .collect::<Vec<_>>()
        .join("\n");

    let html = format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OBS Settings - Rustuber</title>
    <script src="/static/htmx.min.js"></script>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="dashboard">
        <header class="header">
            <h1>OBS Settings</h1>
        </header>

        <nav class="nav">
            <a href="/" class="nav-link">Dashboard</a>
            <a href="/settings" class="nav-link">Settings</a>
            <a href="/settings/obs" class="nav-link active">OBS</a>
            <a href="/avatar" class="nav-link" target="_blank">Browser Source</a>
        </nav>

        <main class="main single-column">
            <section class="panel">
                <h2>Connection Status</h2>
                <div id="obs-status" hx-get="/htmx/obs-status" hx-trigger="load, every 5s" hx-swap="innerHTML">
                    <p class="status-indicator {status_class}">
                        <span class="status-dot"></span>
                        {status_text}
                    </p>
                </div>
                <div class="actions" style="margin-top: 1rem;">
                    <button hx-post="/htmx/reconnect-obs" class="btn btn-secondary" hx-swap="none">
                        Reconnect
                    </button>
                </div>
            </section>

            <form hx-post="/htmx/update-obs-settings" hx-swap="none">
                <section class="panel">
                    <h2>Connection Settings</h2>
                    <div class="form-group">
                        <label>
                            <input type="checkbox" name="obs_enabled" id="obs_enabled" {obs_enabled_checked}>
                            Enable OBS Integration
                        </label>
                    </div>
                    <div class="form-group">
                        <label for="obs_host">OBS Host</label>
                        <input type="text" name="obs_host" id="obs_host" value="{obs_host}" placeholder="127.0.0.1">
                    </div>
                    <div class="form-group">
                        <label for="obs_port">OBS Port</label>
                        <input type="number" name="obs_port" id="obs_port" value="{obs_port}" placeholder="4455">
                    </div>
                    <div class="form-group">
                        <label for="obs_password">OBS Password (optional)</label>
                        <input type="password" name="obs_password" id="obs_password" value="{obs_password}" placeholder="Leave empty if not set">
                    </div>
                    <div class="form-actions">
                        <button type="button" hx-post="/htmx/test-obs"
                                hx-include="#obs_host, #obs_port, #obs_password"
                                hx-target="#test-result"
                                class="btn btn-secondary">
                            Test Connection
                        </button>
                    </div>
                    <div id="test-result" class="test-result"></div>
                </section>

                <section class="panel">
                    <h2>Output Mode</h2>
                    <div class="form-group">
                        <label>
                            <input type="radio" name="obs_mode" value="scene" {mode_scene_checked}>
                            Scene Mode (switch between scenes)
                        </label>
                    </div>
                    <div class="form-group">
                        <label>
                            <input type="radio" name="obs_mode" value="source" {mode_source_checked}>
                            Source Mode (toggle source visibility)
                        </label>
                    </div>
                </section>

                <section class="panel" id="scene-mode-settings">
                    <h2>Scene Mode Settings</h2>
                    <p class="hint">Select which scenes to switch between based on avatar state.</p>
                    <div class="form-group">
                        <label for="obs_idle_scene">Idle Scene</label>
                        <input type="text" name="obs_idle_scene" id="obs_idle_scene"
                               value="{idle_scene}" placeholder="Idle"
                               list="scene-list">
                    </div>
                    <div class="form-group">
                        <label for="obs_speaking_scene">Speaking Scene</label>
                        <input type="text" name="obs_speaking_scene" id="obs_speaking_scene"
                               value="{speaking_scene}" placeholder="Speaking"
                               list="scene-list">
                    </div>
                    <datalist id="scene-list">
                        {scene_options}
                    </datalist>
                    <div hx-get="/htmx/obs-scenes" hx-trigger="load" hx-swap="innerHTML" hx-target="#scene-list"></div>
                </section>

                <section class="panel" id="source-mode-settings">
                    <h2>Source Mode Settings</h2>
                    <p class="hint">Toggle visibility of sources within a single scene.</p>
                    <div class="form-group">
                        <label for="obs_scene">Scene Name</label>
                        <input type="text" name="obs_scene" id="obs_scene"
                               value="{source_scene}" placeholder="Main"
                               list="scene-list">
                    </div>
                    <div class="form-group">
                        <label for="obs_idle_source">Idle Source</label>
                        <input type="text" name="obs_idle_source" id="obs_idle_source"
                               value="{idle_source}" placeholder="avatar_idle">
                    </div>
                    <div class="form-group">
                        <label for="obs_speaking_source">Speaking Source</label>
                        <input type="text" name="obs_speaking_source" id="obs_speaking_source"
                               value="{speaking_source}" placeholder="avatar_speaking">
                    </div>
                </section>

                <div class="form-actions">
                    <button type="submit" class="btn btn-primary">Save OBS Settings</button>
                </div>
            </form>
        </main>
    </div>

    <div id="notification" class="notification" style="display: none;"></div>
    <script>
        document.body.addEventListener('htmx:afterRequest', function(evt) {{
            if (evt.detail.successful && evt.detail.elt.tagName === 'FORM') {{
                var notif = document.getElementById('notification');
                notif.textContent = 'Settings saved!';
                notif.style.display = 'block';
                setTimeout(function() {{ notif.style.display = 'none'; }}, 3000);
            }}
        }});

        // Show/hide mode-specific settings
        function updateModeVisibility() {{
            var mode = document.querySelector('input[name="obs_mode"]:checked').value;
            document.getElementById('scene-mode-settings').style.display = mode === 'scene' ? 'block' : 'none';
            document.getElementById('source-mode-settings').style.display = mode === 'source' ? 'block' : 'none';
        }}
        document.querySelectorAll('input[name="obs_mode"]').forEach(function(el) {{
            el.addEventListener('change', updateModeVisibility);
        }});
        updateModeVisibility();
    </script>
</body>
</html>"##,
        status_class = if obs_connected { "status-connected" } else { "status-disconnected" },
        status_text = if obs_connected { "Connected to OBS" } else if config.obs.enabled { "Disconnected" } else { "Disabled" },
        obs_enabled_checked = if config.obs.enabled { "checked" } else { "" },
        obs_host = config.obs.host,
        obs_port = config.obs.port,
        obs_password = config.obs.password.as_deref().unwrap_or(""),
        mode_scene_checked = if matches!(config.obs.mode, crate::config::ObsMode::Scene) { "checked" } else { "" },
        mode_source_checked = if matches!(config.obs.mode, crate::config::ObsMode::Source) { "checked" } else { "" },
        idle_scene = config.obs.idle_scene.as_deref().unwrap_or(""),
        speaking_scene = config.obs.speaking_scene.as_deref().unwrap_or(""),
        source_scene = config.obs.scene.as_deref().unwrap_or(""),
        idle_source = config.obs.idle_source.as_deref().unwrap_or(""),
        speaking_source = config.obs.speaking_source.as_deref().unwrap_or(""),
        scene_options = scene_options,
    );

    Html(html)
}

/// Render the preview page (full-screen preview)
pub async fn preview_page(State(_state): State<Arc<AppState>>) -> Html<String> {
    let html = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Preview - Rustuber</title>
    <script src="/static/htmx.min.js"></script>
    <script src="/static/sse.js"></script>
    <style>
        body {
            margin: 0;
            padding: 0;
            background: #1a1a1a;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        img {
            max-width: 100%;
            max-height: 100vh;
        }
    </style>
</head>
<body hx-ext="sse" sse-connect="/api/stream">
    <img src="/avatar/current-image" alt="Avatar" hx-get="/avatar/current-image" hx-trigger="sse:state" hx-swap="outerHTML">
</body>
</html>"#;

    Html(html.to_string())
}

/// Status partial for HTMX updates
pub async fn status_partial(State(state): State<Arc<AppState>>) -> Html<String> {
    let avatar = state.get_avatar_state().await;
    let obs_connected = state.obs_connected.load(Ordering::Relaxed);
    let config = state.config.read().await;

    let obs_status = if obs_connected {
        "Connected"
    } else if config.obs.enabled {
        "Disconnected"
    } else {
        "Disabled"
    };

    let html = format!(
        r#"<p>State: <strong class="state-{state}">{state}</strong></p>
<p>Speaking: <strong>{speaking}</strong></p>
<p>Expression: <strong>{expression}</strong></p>
<p>OBS: <strong class="{obs_class}">{obs_status}</strong></p>"#,
        state = avatar.state_type(),
        speaking = if avatar.is_speaking() { "Yes" } else { "No" },
        expression = avatar.expression().unwrap_or("none"),
        obs_status = obs_status,
        obs_class = if obs_connected { "status-connected" } else { "status-disconnected" },
    );

    Html(html)
}

/// OBS status partial
pub async fn obs_status_partial(State(state): State<Arc<AppState>>) -> Html<String> {
    let obs_connected = state.obs_connected.load(Ordering::Relaxed);
    let config = state.config.read().await;

    let (status_class, status_text) = if obs_connected {
        ("status-connected", "Connected to OBS")
    } else if config.obs.enabled {
        ("status-disconnected", "Disconnected")
    } else {
        ("status-disabled", "Disabled")
    };

    let html = format!(
        r#"<p class="status-indicator {status_class}">
    <span class="status-dot"></span>
    {status_text}
</p>"#,
        status_class = status_class,
        status_text = status_text,
    );

    Html(html)
}

/// OBS scenes datalist options
pub async fn obs_scenes_partial(State(state): State<Arc<AppState>>) -> Html<String> {
    let scenes = state.obs_scenes.read().await.clone();

    let html: String = scenes.iter()
        .map(|s| format!(r#"<option value="{}">"#, s))
        .collect::<Vec<_>>()
        .join("\n");

    Html(html)
}

/// Preview partial
pub async fn preview_partial(State(_state): State<Arc<AppState>>) -> Html<String> {
    Html(r#"<img src="/avatar/current-image" alt="Avatar Preview" class="preview-image">"#.to_string())
}

/// Audio devices dropdown options
pub async fn audio_devices_partial(State(state): State<Arc<AppState>>) -> Html<String> {
    let devices = capture::list_input_devices();
    let config = state.config.read().await;
    let current = &config.audio.device;

    let mut html = String::new();
    html.push_str(&format!(
        r#"<option value="default" {}>default</option>"#,
        if current == "default" { "selected" } else { "" }
    ));

    for device in devices {
        let selected = if &device == current { "selected" } else { "" };
        html.push_str(&format!(
            r#"<option value="{}" {}>{}</option>"#,
            device, selected, device
        ));
    }

    Html(html)
}

/// Expressions list
pub async fn expressions_partial(State(state): State<Arc<AppState>>) -> Html<String> {
    let config = state.config.read().await;
    let avatar = state.get_avatar_state().await;
    let current_expr = avatar.expression();

    let mut html = String::from(r#"<div class="expression-grid">"#);

    for (name, _path) in &config.avatar.expressions {
        let active = if current_expr == Some(name.as_str()) {
            "active"
        } else {
            ""
        };

        html.push_str(&format!(
            r#"<button class="expression-btn {active}"
                       hx-post="/htmx/set-expression"
                       hx-vals='{{"expression": "{name}"}}'
                       hx-swap="none">
                {name}
            </button>"#,
            name = name,
            active = active,
        ));
    }

    html.push_str("</div>");
    Html(html)
}

/// Form data for setting expression
#[derive(Debug, Deserialize)]
pub struct ExpressionForm {
    pub expression: String,
}

/// Set expression from HTMX
pub async fn set_expression(
    State(state): State<Arc<AppState>>,
    Form(form): Form<ExpressionForm>,
) -> impl IntoResponse {
    let current = state.get_avatar_state().await;
    let new_state = current.with_expression(Some(form.expression));
    state.update_avatar_state(new_state).await;

    StatusCode::OK
}

/// Clear expression from HTMX
pub async fn clear_expression(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let current = state.get_avatar_state().await;
    let new_state = current.with_expression(None);
    state.update_avatar_state(new_state).await;

    StatusCode::OK
}

/// Form data for updating settings
#[derive(Debug, Deserialize)]
pub struct SettingsForm {
    #[serde(default)]
    pub audio_device: Option<String>,
    #[serde(default)]
    pub vad_threshold: Option<f32>,
    #[serde(default)]
    pub vmc_enabled: Option<String>,
    #[serde(default)]
    pub vmc_port: Option<u16>,
}

/// Update settings from HTMX form
pub async fn update_settings(
    State(state): State<Arc<AppState>>,
    Form(form): Form<SettingsForm>,
) -> impl IntoResponse {
    let mut config = state.config.write().await;

    if let Some(device) = form.audio_device {
        config.audio.device = device;
    }
    if let Some(threshold) = form.vad_threshold {
        config.vad.energy_threshold_db = threshold;
    }
    config.vmc.receiver_enabled = form.vmc_enabled.is_some();
    if let Some(port) = form.vmc_port {
        config.vmc.receiver_port = port;
    }

    drop(config);
    state.signal_config_changed();

    tracing::info!("Settings updated");

    (
        StatusCode::OK,
        [("HX-Trigger", "settings-saved")],
        "Settings saved",
    )
}

/// Form data for updating OBS settings
#[derive(Debug, Deserialize)]
pub struct ObsSettingsForm {
    #[serde(default)]
    pub obs_enabled: Option<String>,
    #[serde(default)]
    pub obs_host: Option<String>,
    #[serde(default)]
    pub obs_port: Option<u16>,
    #[serde(default)]
    pub obs_password: Option<String>,
    #[serde(default)]
    pub obs_mode: Option<String>,
    #[serde(default)]
    pub obs_idle_scene: Option<String>,
    #[serde(default)]
    pub obs_speaking_scene: Option<String>,
    #[serde(default)]
    pub obs_scene: Option<String>,
    #[serde(default)]
    pub obs_idle_source: Option<String>,
    #[serde(default)]
    pub obs_speaking_source: Option<String>,
}

/// Update OBS settings from HTMX form
pub async fn update_obs_settings(
    State(state): State<Arc<AppState>>,
    Form(form): Form<ObsSettingsForm>,
) -> impl IntoResponse {
    let mut config = state.config.write().await;

    config.obs.enabled = form.obs_enabled.is_some();

    if let Some(host) = form.obs_host {
        if !host.is_empty() {
            config.obs.host = host;
        }
    }
    if let Some(port) = form.obs_port {
        config.obs.port = port;
    }
    if let Some(password) = form.obs_password {
        config.obs.password = if password.is_empty() { None } else { Some(password) };
    }
    if let Some(mode) = form.obs_mode {
        config.obs.mode = match mode.as_str() {
            "source" => crate::config::ObsMode::Source,
            _ => crate::config::ObsMode::Scene,
        };
    }
    if let Some(scene) = form.obs_idle_scene {
        config.obs.idle_scene = if scene.is_empty() { None } else { Some(scene) };
    }
    if let Some(scene) = form.obs_speaking_scene {
        config.obs.speaking_scene = if scene.is_empty() { None } else { Some(scene) };
    }
    if let Some(scene) = form.obs_scene {
        config.obs.scene = if scene.is_empty() { None } else { Some(scene) };
    }
    if let Some(source) = form.obs_idle_source {
        config.obs.idle_source = if source.is_empty() { None } else { Some(source) };
    }
    if let Some(source) = form.obs_speaking_source {
        config.obs.speaking_source = if source.is_empty() { None } else { Some(source) };
    }

    drop(config);
    state.signal_config_changed();
    state.signal_obs_reconnect();

    tracing::info!("OBS settings updated");

    (
        StatusCode::OK,
        [("HX-Trigger", "settings-saved")],
        "OBS settings saved",
    )
}

/// Test OBS connection form
#[derive(Debug, Deserialize)]
pub struct TestObsForm {
    pub obs_host: String,
    pub obs_port: u16,
    #[serde(default)]
    pub obs_password: Option<String>,
}

/// Test OBS connection from HTMX
pub async fn test_obs_connection(
    Form(form): Form<TestObsForm>,
) -> Html<String> {
    use crate::output::obs::ObsClient;
    use crate::config::ObsConfig;

    let mut test_config = ObsConfig::default();
    test_config.host = form.obs_host;
    test_config.port = form.obs_port;
    test_config.password = form.obs_password.filter(|p| !p.is_empty());

    let mut client = ObsClient::new(&test_config);

    match client.connect().await {
        Ok(()) => {
            match client.list_scenes().await {
                Ok(scenes) => {
                    let scene_list = scenes.join(", ");
                    Html(format!(
                        r#"<div class="test-success">Connected successfully! Found {} scenes: {}</div>"#,
                        scenes.len(),
                        scene_list
                    ))
                }
                Err(e) => Html(format!(
                    r#"<div class="test-warning">Connected but failed to list scenes: {}</div>"#,
                    e
                )),
            }
        }
        Err(e) => Html(format!(
            r#"<div class="test-error">Connection failed: {}</div>"#,
            e
        )),
    }
}

/// Reconnect to OBS
pub async fn reconnect_obs(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.signal_obs_reconnect();
    StatusCode::OK
}
