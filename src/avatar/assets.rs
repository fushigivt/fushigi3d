//! Asset loading and management

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::config::AvatarConfig;
use crate::error::{AvatarError, RustuberError};

/// Manages avatar assets (images)
#[derive(Debug)]
pub struct AssetManager {
    /// Base directory for assets
    base_dir: PathBuf,
    /// Cached asset paths (key -> absolute path)
    assets: HashMap<String, PathBuf>,
    /// State images mapping
    state_images: HashMap<String, String>,
    /// Expression images mapping
    expression_images: HashMap<String, String>,
}

impl AssetManager {
    /// Create a new asset manager from configuration
    pub fn new(config: &AvatarConfig) -> Result<Self, RustuberError> {
        let base_dir = if config.assets_dir.is_absolute() {
            config.assets_dir.clone()
        } else {
            std::env::current_dir()
                .unwrap_or_default()
                .join(&config.assets_dir)
        };

        let mut manager = Self {
            base_dir,
            assets: HashMap::new(),
            state_images: config.states.clone(),
            expression_images: config.expressions.clone(),
        };

        // Scan and cache available assets
        manager.scan_assets()?;

        Ok(manager)
    }

    /// Scan the assets directory and cache available assets
    fn scan_assets(&mut self) -> Result<(), RustuberError> {
        if !self.base_dir.exists() {
            tracing::warn!(
                "Assets directory does not exist: {}",
                self.base_dir.display()
            );
            return Ok(());
        }

        // Add state images
        for (state, filename) in &self.state_images {
            let path = self.base_dir.join(filename);
            if path.exists() {
                self.assets.insert(state.clone(), path);
                tracing::debug!("Loaded state asset: {} -> {}", state, filename);
            } else {
                tracing::warn!("State asset not found: {} ({})", state, path.display());
            }
        }

        // Add expression images
        for (expr, filename) in &self.expression_images {
            let path = self.base_dir.join(filename);
            if path.exists() {
                self.assets.insert(expr.clone(), path);
                tracing::debug!("Loaded expression asset: {} -> {}", expr, filename);
            } else {
                tracing::warn!("Expression asset not found: {} ({})", expr, path.display());
            }
        }

        Ok(())
    }

    /// Get the path for an asset by key
    pub fn get_path(&self, key: &str) -> Option<&Path> {
        self.assets.get(key).map(|p| p.as_path())
    }

    /// Get the URL path for serving this asset
    pub fn get_url_path(&self, key: &str) -> Option<String> {
        self.assets.get(key).map(|p| {
            // Return path relative to base_dir for URL construction
            if let Ok(relative) = p.strip_prefix(&self.base_dir) {
                format!("/assets/{}", relative.display())
            } else {
                format!("/assets/{}", p.file_name().unwrap_or_default().to_string_lossy())
            }
        })
    }

    /// Check if an asset exists
    pub fn has_asset(&self, key: &str) -> bool {
        self.assets.contains_key(key)
    }

    /// Get all available asset keys
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.assets.keys().map(|s| s.as_str())
    }

    /// Get all state keys
    pub fn state_keys(&self) -> impl Iterator<Item = &str> {
        self.state_images.keys().map(|s| s.as_str())
    }

    /// Get all expression keys
    pub fn expression_keys(&self) -> impl Iterator<Item = &str> {
        self.expression_images.keys().map(|s| s.as_str())
    }

    /// Get the base directory
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Add or update an asset
    pub fn add_asset(&mut self, key: &str, path: PathBuf) -> Result<(), RustuberError> {
        if !path.exists() {
            return Err(AvatarError::AssetNotFound(path.display().to_string()).into());
        }
        self.assets.insert(key.to_string(), path);
        Ok(())
    }

    /// Remove an asset
    pub fn remove_asset(&mut self, key: &str) -> bool {
        self.assets.remove(key).is_some()
    }

    /// Reload assets from disk
    pub fn reload(&mut self) -> Result<(), RustuberError> {
        self.assets.clear();
        self.scan_assets()
    }

    /// Get asset data (read file contents)
    pub fn get_data(&self, key: &str) -> Result<Vec<u8>, RustuberError> {
        let path = self
            .get_path(key)
            .ok_or_else(|| AvatarError::AssetNotFound(key.to_string()))?;

        std::fs::read(path).map_err(|e| {
            AvatarError::ImageLoad(format!("{}: {}", path.display(), e)).into()
        })
    }

    /// Get the MIME type for an asset based on extension
    pub fn get_mime_type(&self, key: &str) -> Option<&'static str> {
        let path = self.get_path(key)?;
        let extension = path.extension()?.to_str()?;

        Some(match extension.to_lowercase().as_str() {
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "svg" => "image/svg+xml",
            _ => "application/octet-stream",
        })
    }
}

/// Information about a loaded asset
#[derive(Debug, Clone)]
pub struct AssetInfo {
    /// Asset key
    pub key: String,
    /// File path
    pub path: PathBuf,
    /// File size in bytes
    pub size: u64,
    /// MIME type
    pub mime_type: String,
    /// Image dimensions (if available)
    pub dimensions: Option<(u32, u32)>,
}

impl AssetInfo {
    /// Create asset info from a path
    pub fn from_path(key: &str, path: &Path) -> Option<Self> {
        let metadata = std::fs::metadata(path).ok()?;
        let extension = path.extension()?.to_str()?;

        let mime_type = match extension.to_lowercase().as_str() {
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "svg" => "image/svg+xml",
            _ => "application/octet-stream",
        };

        Some(Self {
            key: key.to_string(),
            path: path.to_path_buf(),
            size: metadata.len(),
            mime_type: mime_type.to_string(),
            dimensions: None, // Could use image crate to get dimensions
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_assets() -> (TempDir, AvatarConfig) {
        let dir = TempDir::new().unwrap();

        // Create test image files
        std::fs::write(dir.path().join("idle.png"), b"fake png data").unwrap();
        std::fs::write(dir.path().join("speaking.png"), b"fake png data").unwrap();

        let expr_dir = dir.path().join("expressions");
        std::fs::create_dir(&expr_dir).unwrap();
        std::fs::write(expr_dir.join("happy.png"), b"fake png data").unwrap();

        let mut config = AvatarConfig::default();
        config.assets_dir = dir.path().to_path_buf();

        (dir, config)
    }

    #[test]
    fn test_asset_manager_creation() {
        let (_dir, config) = create_test_assets();
        let manager = AssetManager::new(&config).unwrap();

        assert!(manager.has_asset("idle"));
        assert!(manager.has_asset("speaking"));
        assert!(manager.has_asset("happy"));
    }

    #[test]
    fn test_get_path() {
        let (_dir, config) = create_test_assets();
        let manager = AssetManager::new(&config).unwrap();

        let idle_path = manager.get_path("idle").unwrap();
        assert!(idle_path.exists());
        assert!(idle_path.ends_with("idle.png"));
    }

    #[test]
    fn test_get_url_path() {
        let (_dir, config) = create_test_assets();
        let manager = AssetManager::new(&config).unwrap();

        let url = manager.get_url_path("idle").unwrap();
        assert!(url.contains("idle.png"));
    }
}
