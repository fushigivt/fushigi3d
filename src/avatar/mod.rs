//! Avatar state management module
//!
//! Handles avatar states, expressions, and asset loading.

pub mod assets;
pub mod expression;
pub mod state;

pub use assets::AssetManager;
pub use expression::{Expression, ExpressionConfig};
pub use state::{AvatarState, StateType};
