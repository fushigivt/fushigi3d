//! Expression types and configuration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// An expression definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expression {
    /// Unique name for this expression
    pub name: String,
    /// Display name for UI
    pub display_name: Option<String>,
    /// Path to the expression image (relative to assets dir)
    pub image_path: String,
    /// Optional path to speaking variant of this expression
    pub speaking_image_path: Option<String>,
    /// Duration before auto-returning to idle (None = permanent until changed)
    pub duration: Option<Duration>,
    /// Priority level (higher overrides lower)
    pub priority: u8,
    /// Keyboard shortcut (if any)
    pub hotkey: Option<String>,
}

impl Expression {
    /// Create a new expression
    pub fn new(name: &str, image_path: &str) -> Self {
        Self {
            name: name.to_string(),
            display_name: None,
            image_path: image_path.to_string(),
            speaking_image_path: None,
            duration: None,
            priority: 0,
            hotkey: None,
        }
    }

    /// Set the display name
    pub fn with_display_name(mut self, name: &str) -> Self {
        self.display_name = Some(name.to_string());
        self
    }

    /// Set the speaking variant image
    pub fn with_speaking_image(mut self, path: &str) -> Self {
        self.speaking_image_path = Some(path.to_string());
        self
    }

    /// Set the auto-return duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Set the priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Set the hotkey
    pub fn with_hotkey(mut self, hotkey: &str) -> Self {
        self.hotkey = Some(hotkey.to_string());
        self
    }

    /// Get the effective image path for current state
    pub fn get_image_path(&self, is_speaking: bool) -> &str {
        if is_speaking {
            self.speaking_image_path
                .as_deref()
                .unwrap_or(&self.image_path)
        } else {
            &self.image_path
        }
    }

    /// Get the display name or fall back to name
    pub fn get_display_name(&self) -> &str {
        self.display_name.as_deref().unwrap_or(&self.name)
    }
}

/// Configuration for expressions loaded from config
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExpressionConfig {
    /// Map of expression name to expression definition
    pub expressions: HashMap<String, Expression>,
    /// Default expression on startup
    pub default_expression: Option<String>,
}

impl ExpressionConfig {
    /// Create a new expression config
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an expression
    pub fn add(&mut self, expression: Expression) {
        self.expressions.insert(expression.name.clone(), expression);
    }

    /// Get an expression by name
    pub fn get(&self, name: &str) -> Option<&Expression> {
        self.expressions.get(name)
    }

    /// Get all expression names
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.expressions.keys().map(|s| s.as_str())
    }

    /// Load expressions from avatar config hashmap
    pub fn from_hashmap(map: &HashMap<String, String>) -> Self {
        let mut config = Self::new();

        for (name, path) in map {
            config.add(Expression::new(name, path));
        }

        config
    }
}

/// VMC blendshape to expression mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlendshapeMapping {
    /// Expression to trigger
    pub expression: String,
    /// Blendshape name to monitor
    pub blendshape: String,
    /// Threshold to trigger (0.0 - 1.0)
    pub threshold: f32,
    /// Whether this is a toggle or momentary trigger
    pub toggle: bool,
}

impl BlendshapeMapping {
    /// Create a new blendshape mapping
    pub fn new(expression: &str, blendshape: &str, threshold: f32) -> Self {
        Self {
            expression: expression.to_string(),
            blendshape: blendshape.to_string(),
            threshold,
            toggle: false,
        }
    }

    /// Check if the blendshape value exceeds threshold
    pub fn is_triggered(&self, value: f32) -> bool {
        value >= self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression_creation() {
        let expr = Expression::new("happy", "expressions/happy.png")
            .with_display_name("Happy")
            .with_speaking_image("expressions/happy_speaking.png")
            .with_priority(1);

        assert_eq!(expr.name, "happy");
        assert_eq!(expr.get_display_name(), "Happy");
        assert_eq!(expr.get_image_path(false), "expressions/happy.png");
        assert_eq!(
            expr.get_image_path(true),
            "expressions/happy_speaking.png"
        );
    }

    #[test]
    fn test_expression_config() {
        let mut config = ExpressionConfig::new();
        config.add(Expression::new("happy", "happy.png"));
        config.add(Expression::new("sad", "sad.png"));

        assert!(config.get("happy").is_some());
        assert!(config.get("sad").is_some());
        assert!(config.get("angry").is_none());
    }

    #[test]
    fn test_blendshape_mapping() {
        let mapping = BlendshapeMapping::new("happy", "joy", 0.5);

        assert!(mapping.is_triggered(0.6));
        assert!(mapping.is_triggered(0.5));
        assert!(!mapping.is_triggered(0.4));
    }
}
