//! OBS WebSocket integration

use obws::requests::scenes::SceneId;
use obws::Client;
use std::time::Duration;
use tokio::time::sleep;

use crate::avatar::AvatarState;
use crate::config::{ObsConfig, ObsMode};
use crate::error::{OutputError, Fushigi3dError};

/// OBS WebSocket client for controlling scenes/sources
pub struct ObsClient {
    config: ObsConfig,
    client: Option<Client>,
    last_state: Option<String>,
}

impl ObsClient {
    /// Create a new OBS client
    pub fn new(config: &ObsConfig) -> Self {
        Self {
            config: config.clone(),
            client: None,
            last_state: None,
        }
    }

    /// Connect to OBS WebSocket server
    pub async fn connect(&mut self) -> Result<(), Fushigi3dError> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        tracing::info!("Connecting to OBS at {}", addr);

        let client = Client::connect(&self.config.host, self.config.port, self.config.password.clone())
            .await
            .map_err(|e| OutputError::ObsConnection(e.to_string()))?;

        // Verify connection by getting version
        let version = client.general().version().await.map_err(|e| {
            OutputError::ObsConnection(format!("Failed to get OBS version: {}", e))
        })?;

        tracing::info!(
            "Connected to OBS {} (WebSocket {})",
            version.obs_version.to_string(),
            version.obs_web_socket_version.to_string()
        );

        self.client = Some(client);
        Ok(())
    }

    /// Disconnect from OBS
    pub async fn disconnect(&mut self) {
        self.client = None;
        tracing::info!("Disconnected from OBS");
    }

    /// Check if connected to OBS
    pub fn is_connected(&self) -> bool {
        self.client.is_some()
    }

    /// Update OBS based on avatar state
    pub async fn update_state(&mut self, state: &AvatarState) -> Result<(), Fushigi3dError> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| OutputError::ObsConnection("Not connected".to_string()))?;

        let scene_name = state.scene_name();

        // Skip if state hasn't changed
        if let Some(ref last) = self.last_state {
            if last == scene_name {
                return Ok(());
            }
        }

        tracing::debug!("Updating OBS state to: {}", scene_name);

        // Record the state now so we don't retry on every frame if it fails
        self.last_state = Some(scene_name.to_string());

        match self.config.mode {
            ObsMode::Scene => {
                self.switch_scene(client, scene_name).await?;
            }
            ObsMode::Source => {
                self.toggle_sources(client, scene_name).await?;
            }
        }

        Ok(())
    }

    /// Switch to a scene based on state
    async fn switch_scene(&self, client: &Client, state: &str) -> Result<(), Fushigi3dError> {
        let scene_name = match state {
            "speaking" => self
                .config
                .speaking_scene
                .as_ref()
                .ok_or_else(|| OutputError::ObsSceneNotFound("speaking_scene not configured".to_string()))?,
            "idle" | _ => self
                .config
                .idle_scene
                .as_ref()
                .ok_or_else(|| OutputError::ObsSceneNotFound("idle_scene not configured".to_string()))?,
        };

        tracing::debug!("Switching to scene: {}", scene_name);

        client
            .scenes()
            .set_current_program_scene(SceneId::Name(scene_name.as_str()))
            .await
            .map_err(|e| OutputError::ObsSceneNotFound(format!("{}: {}", scene_name, e)))?;

        Ok(())
    }

    /// Toggle source visibility based on state
    async fn toggle_sources(&self, client: &Client, state: &str) -> Result<(), Fushigi3dError> {
        let scene = self
            .config
            .scene
            .as_ref()
            .ok_or_else(|| OutputError::ObsSceneNotFound("scene not configured for source mode".to_string()))?;

        let idle_source = self.config.idle_source.as_ref();
        let speaking_source = self.config.speaking_source.as_ref();

        let (show_idle, show_speaking) = match state {
            "speaking" => (false, true),
            "idle" | _ => (true, false),
        };

        // Toggle idle source
        if let Some(source) = idle_source {
            tracing::debug!("Setting {} visibility to {}", source, show_idle);
            if let Err(e) = self.set_source_visibility(client, scene.as_str(), source.as_str(), show_idle).await {
                tracing::warn!("Failed to set idle source visibility: {}", e);
            }
        }

        // Toggle speaking source
        if let Some(source) = speaking_source {
            tracing::debug!("Setting {} visibility to {}", source, show_speaking);
            if let Err(e) = self.set_source_visibility(client, scene.as_str(), source.as_str(), show_speaking).await {
                tracing::warn!("Failed to set speaking source visibility: {}", e);
            }
        }

        Ok(())
    }

    /// Set visibility of a specific source
    async fn set_source_visibility(
        &self,
        client: &Client,
        scene: &str,
        source: &str,
        visible: bool,
    ) -> Result<(), Fushigi3dError> {
        // Get scene item ID
        let item_id = client
            .scene_items()
            .id(obws::requests::scene_items::Id {
                scene: SceneId::Name(scene),
                source,
                search_offset: None,
            })
            .await
            .map_err(|e| OutputError::ObsSourceNotFound(format!("{}: {}", source, e)))?;

        // Set visibility
        client
            .scene_items()
            .set_enabled(obws::requests::scene_items::SetEnabled {
                scene: SceneId::Name(scene),
                item_id,
                enabled: visible,
            })
            .await
            .map_err(|e| OutputError::ObsSourceNotFound(format!("Failed to set visibility: {}", e)))?;

        Ok(())
    }

    /// Get list of available scenes
    pub async fn list_scenes(&self) -> Result<Vec<String>, Fushigi3dError> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| OutputError::ObsConnection("Not connected".to_string()))?;

        let scenes = client
            .scenes()
            .list()
            .await
            .map_err(|e| OutputError::ObsConnection(format!("Failed to list scenes: {}", e)))?;

        // The Scene struct has an `id` field which contains the name
        Ok(scenes.scenes.into_iter().map(|s| s.id.name).collect())
    }

    /// Get list of sources in a scene
    pub async fn list_sources(&self, scene: &str) -> Result<Vec<String>, Fushigi3dError> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| OutputError::ObsConnection("Not connected".to_string()))?;

        let items = client
            .scene_items()
            .list(SceneId::Name(scene))
            .await
            .map_err(|e| OutputError::ObsConnection(format!("Failed to list sources: {}", e)))?;

        Ok(items.into_iter().map(|i| i.source_name).collect())
    }

    /// Attempt to reconnect to OBS with retry
    pub async fn reconnect_with_retry(&mut self) -> Result<(), Fushigi3dError> {
        let max_retries = 5;
        let mut retry_count = 0;

        loop {
            match self.connect().await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    retry_count += 1;
                    if retry_count >= max_retries {
                        return Err(e);
                    }
                    tracing::warn!(
                        "Failed to connect to OBS (attempt {}/{}): {}",
                        retry_count,
                        max_retries,
                        e
                    );
                    sleep(Duration::from_secs(self.config.reconnect_delay_secs)).await;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obs_client_creation() {
        let config = ObsConfig::default();
        let client = ObsClient::new(&config);
        assert!(!client.is_connected());
    }
}
