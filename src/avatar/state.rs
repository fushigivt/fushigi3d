//! Avatar state machine

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// The type of avatar state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StateType {
    /// Default idle state
    Idle,
    /// Speaking/talking state
    Speaking,
    /// Custom expression state
    Expression,
}

impl Default for StateType {
    fn default() -> Self {
        Self::Idle
    }
}

impl std::fmt::Display for StateType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StateType::Idle => write!(f, "idle"),
            StateType::Speaking => write!(f, "speaking"),
            StateType::Expression => write!(f, "expression"),
        }
    }
}

/// Full avatar state including expression and metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AvatarState {
    /// Current state type
    state_type: StateType,
    /// Whether currently speaking (can be combined with expression)
    is_speaking: bool,
    /// Current expression name (if any)
    expression: Option<String>,
    /// Mouth open amount (0.0 - 1.0) for lip sync
    mouth_open: f32,
    /// Eye blink amount (0.0 = open, 1.0 = closed)
    blink: f32,
    /// Head position (x, y, z)
    head_position: [f32; 3],
    /// Head rotation (pitch, yaw, roll) in degrees
    head_rotation: [f32; 3],
    /// Full ARKit blendshape map (used for VRM morph targets)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    blendshapes: HashMap<String, f32>,
    /// Timestamp of last state change (not serialized)
    #[serde(skip)]
    last_change: Option<Instant>,
}

impl Default for AvatarState {
    fn default() -> Self {
        Self {
            state_type: StateType::Idle,
            is_speaking: false,
            expression: None,
            mouth_open: 0.0,
            blink: 0.0,
            head_position: [0.0, 0.0, 0.0],
            head_rotation: [0.0, 0.0, 0.0],
            blendshapes: HashMap::new(),
            last_change: Some(Instant::now()),
        }
    }
}

impl AvatarState {
    /// Create a new avatar state with the given initial state name
    pub fn new(initial_state: &str) -> Self {
        let state_type = match initial_state.to_lowercase().as_str() {
            "idle" => StateType::Idle,
            "speaking" => StateType::Speaking,
            _ => StateType::Idle,
        };

        Self {
            state_type,
            ..Default::default()
        }
    }

    /// Get the current state type
    pub fn state_type(&self) -> StateType {
        self.state_type
    }

    /// Check if currently speaking
    pub fn is_speaking(&self) -> bool {
        self.is_speaking
    }

    /// Get the current expression name
    pub fn expression(&self) -> Option<&str> {
        self.expression.as_deref()
    }

    /// Get the mouth open amount
    pub fn mouth_open(&self) -> f32 {
        self.mouth_open
    }

    /// Get the blink amount
    pub fn blink(&self) -> f32 {
        self.blink
    }

    /// Get the head position
    pub fn head_position(&self) -> [f32; 3] {
        self.head_position
    }

    /// Get the head rotation
    pub fn head_rotation(&self) -> [f32; 3] {
        self.head_rotation
    }

    /// Get the full blendshape map
    pub fn blendshapes(&self) -> &HashMap<String, f32> {
        &self.blendshapes
    }

    /// Get duration since last state change
    pub fn time_in_state(&self) -> Duration {
        self.last_change
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    /// Create a new state with speaking flag changed
    pub fn with_speaking(mut self, speaking: bool) -> Self {
        if speaking != self.is_speaking {
            self.is_speaking = speaking;
            self.state_type = if speaking {
                StateType::Speaking
            } else if self.expression.is_some() {
                StateType::Expression
            } else {
                StateType::Idle
            };
            self.last_change = Some(Instant::now());
        }
        self
    }

    /// Create a new state with expression changed
    pub fn with_expression(mut self, expression: Option<String>) -> Self {
        if expression != self.expression {
            self.expression = expression;
            self.state_type = if self.is_speaking {
                StateType::Speaking
            } else if self.expression.is_some() {
                StateType::Expression
            } else {
                StateType::Idle
            };
            self.last_change = Some(Instant::now());
        }
        self
    }

    /// Create a new state with mouth open value
    pub fn with_mouth_open(mut self, value: f32) -> Self {
        self.mouth_open = value.clamp(0.0, 1.0);
        self
    }

    /// Create a new state with blink value
    pub fn with_blink(mut self, value: f32) -> Self {
        self.blink = value.clamp(0.0, 1.0);
        self
    }

    /// Create a new state with head position
    pub fn with_head_position(mut self, position: [f32; 3]) -> Self {
        self.head_position = position;
        self
    }

    /// Create a new state with head rotation
    pub fn with_head_rotation(mut self, rotation: [f32; 3]) -> Self {
        self.head_rotation = rotation;
        self
    }

    /// Create a new state with full blendshape map
    pub fn with_blendshapes(mut self, blendshapes: HashMap<String, f32>) -> Self {
        self.blendshapes = blendshapes;
        self
    }

    /// Get the asset key for current state (used for image lookup)
    pub fn asset_key(&self) -> String {
        if let Some(ref expr) = self.expression {
            if self.is_speaking {
                format!("{}_speaking", expr)
            } else {
                expr.clone()
            }
        } else if self.is_speaking {
            "speaking".to_string()
        } else {
            "idle".to_string()
        }
    }

    /// Get a simplified state name for OBS scene/source selection
    pub fn scene_name(&self) -> &str {
        if self.is_speaking {
            "speaking"
        } else {
            "idle"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_state() {
        let state = AvatarState::default();
        assert_eq!(state.state_type(), StateType::Idle);
        assert!(!state.is_speaking());
        assert!(state.expression().is_none());
    }

    #[test]
    fn test_state_transitions() {
        let state = AvatarState::default();

        // Idle -> Speaking
        let state = state.with_speaking(true);
        assert_eq!(state.state_type(), StateType::Speaking);
        assert!(state.is_speaking());

        // Speaking -> Idle
        let state = state.with_speaking(false);
        assert_eq!(state.state_type(), StateType::Idle);
        assert!(!state.is_speaking());

        // Idle -> Expression
        let state = state.with_expression(Some("happy".to_string()));
        assert_eq!(state.state_type(), StateType::Expression);
        assert_eq!(state.expression(), Some("happy"));

        // Expression + Speaking
        let state = state.with_speaking(true);
        assert_eq!(state.state_type(), StateType::Speaking);
        assert!(state.is_speaking());
        assert_eq!(state.expression(), Some("happy"));
    }

    #[test]
    fn test_asset_key() {
        let state = AvatarState::default();
        assert_eq!(state.asset_key(), "idle");

        let state = state.with_speaking(true);
        assert_eq!(state.asset_key(), "speaking");

        let state = AvatarState::default().with_expression(Some("happy".to_string()));
        assert_eq!(state.asset_key(), "happy");

        let state = state.with_speaking(true);
        assert_eq!(state.asset_key(), "happy_speaking");
    }
}
