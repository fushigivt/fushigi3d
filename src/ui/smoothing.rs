//! Tracking smoothing orchestration layer.
//!
//! Wraps `signal_smooth` primitives into application-specific smoothers for
//! head rotation, blendshapes, and expression transitions.

#![cfg(feature = "native-ui")]

use std::collections::HashMap;

use signal_smooth::{apply_deadzone, DampedSpring, OneEuroFilter};

use crate::config::TrackingTuning;

/// Which smoothing algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoothingMode {
    Spring,
    OneEuro,
    None,
}

impl SmoothingMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "spring" | "damped_spring" => Self::Spring,
            "one_euro" | "oneeuro" | "1euro" => Self::OneEuro,
            "none" | "off" | "disabled" => Self::None,
            _ => Self::Spring,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Spring => "spring",
            Self::OneEuro => "one_euro",
            Self::None => "none",
        }
    }

    pub const ALL: [SmoothingMode; 3] = [Self::Spring, Self::OneEuro, Self::None];
}

/// Orchestrates per-channel smoothing for head rotation and blendshapes.
pub struct TrackingSmoother {
    /// [pitch, yaw, roll] springs
    head_springs: [DampedSpring; 3],
    /// [pitch, yaw, roll] 1-Euro filters
    head_euros: [OneEuroFilter; 3],
    /// Per-blendshape springs (lazily populated)
    bs_springs: HashMap<String, DampedSpring>,
    /// Per-blendshape 1-Euro filters (lazily populated)
    bs_euros: HashMap<String, OneEuroFilter>,
    /// Per-body-landmark-component springs (keyed "name.0"/"name.1"/"name.2")
    body_springs: HashMap<String, DampedSpring>,
    /// Per-body-landmark-component 1-Euro filters
    body_euros: HashMap<String, OneEuroFilter>,
    /// Active smoothing mode
    mode: SmoothingMode,
}

impl TrackingSmoother {
    pub fn new(mode: SmoothingMode, tuning: &TrackingTuning) -> Self {
        Self {
            head_springs: [
                DampedSpring::new(0.0),
                DampedSpring::new(0.0),
                DampedSpring::new(0.0),
            ],
            head_euros: [
                OneEuroFilter::new(tuning.head_min_cutoff, tuning.head_beta),
                OneEuroFilter::new(tuning.head_min_cutoff, tuning.head_beta),
                OneEuroFilter::new(tuning.head_min_cutoff, tuning.head_beta),
            ],
            bs_springs: HashMap::new(),
            bs_euros: HashMap::new(),
            body_springs: HashMap::new(),
            body_euros: HashMap::new(),
            mode,
        }
    }

    /// Smooth a [pitch, yaw, roll] triple. Applies deadzone first, then the active filter.
    pub fn smooth_head(
        &mut self,
        raw: [f32; 3],
        dt: f32,
        tuning: &TrackingTuning,
    ) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        for i in 0..3 {
            let deadzoned = apply_deadzone(raw[i], tuning.head_deadzone);
            out[i] = match self.mode {
                SmoothingMode::Spring => {
                    self.head_springs[i].update(deadzoned, tuning.head_halflife, dt)
                }
                SmoothingMode::OneEuro => self.head_euros[i].filter(deadzoned, dt),
                SmoothingMode::None => deadzoned,
            };
        }
        out
    }

    /// Smooth a set of blendshapes. Applies deadzone, then the active filter per key.
    pub fn smooth_blendshapes(
        &mut self,
        raw: &HashMap<String, f32>,
        dt: f32,
        tuning: &TrackingTuning,
    ) -> HashMap<String, f32> {
        let mut out = HashMap::with_capacity(raw.len());
        for (name, &value) in raw {
            let deadzoned = apply_deadzone(value, tuning.blendshape_deadzone);

            // Pick halflife: blinks get faster response
            let is_blink = name.contains("Blink") || name.contains("blink");
            let halflife = if is_blink {
                tuning.blink_halflife
            } else {
                tuning.blendshape_halflife
            };

            let smoothed = match self.mode {
                SmoothingMode::Spring => {
                    let spring = self
                        .bs_springs
                        .entry(name.clone())
                        .or_insert_with(|| DampedSpring::new(deadzoned));
                    spring.update(deadzoned, halflife, dt)
                }
                SmoothingMode::OneEuro => {
                    let (min_cutoff, beta) = if is_blink {
                        (tuning.blink_min_cutoff, tuning.blink_beta)
                    } else {
                        (tuning.blendshape_min_cutoff, tuning.blendshape_beta)
                    };
                    let euro = self
                        .bs_euros
                        .entry(name.clone())
                        .or_insert_with(|| OneEuroFilter::new(min_cutoff, beta));
                    euro.filter(deadzoned, dt)
                }
                SmoothingMode::None => deadzoned,
            };

            out.insert(name.clone(), smoothed);
        }
        out
    }

    /// Smooth body landmark positions. Each landmark's x/y/z is filtered independently.
    pub fn smooth_body_landmarks(
        &mut self,
        raw: &HashMap<String, [f32; 3]>,
        dt: f32,
        tuning: &TrackingTuning,
    ) -> HashMap<String, [f32; 3]> {
        let mut out = HashMap::with_capacity(raw.len());
        for (name, &pos) in raw {
            let mut smoothed = [0.0f32; 3];
            for i in 0..3 {
                let key = format!("{}.{}", name, i);
                let deadzoned = apply_deadzone(pos[i], 0.005);
                smoothed[i] = match self.mode {
                    SmoothingMode::Spring => {
                        let spring = self
                            .body_springs
                            .entry(key)
                            .or_insert_with(|| DampedSpring::new(deadzoned));
                        spring.update(deadzoned, tuning.body_halflife, dt)
                    }
                    SmoothingMode::OneEuro => {
                        let euro = self
                            .body_euros
                            .entry(key)
                            .or_insert_with(|| {
                                OneEuroFilter::new(tuning.head_min_cutoff, tuning.head_beta)
                            });
                        euro.filter(deadzoned, dt)
                    }
                    SmoothingMode::None => deadzoned,
                };
            }
            out.insert(name.clone(), smoothed);
        }
        out
    }

    /// Change the smoothing mode, resetting all filter state.
    pub fn set_mode(&mut self, mode: SmoothingMode, tuning: &TrackingTuning) {
        if mode == self.mode {
            return;
        }
        self.mode = mode;
        // Reset springs
        for s in &mut self.head_springs {
            s.set(0.0);
        }
        self.bs_springs.clear();
        self.body_springs.clear();
        // Reset euros
        self.head_euros = [
            OneEuroFilter::new(tuning.head_min_cutoff, tuning.head_beta),
            OneEuroFilter::new(tuning.head_min_cutoff, tuning.head_beta),
            OneEuroFilter::new(tuning.head_min_cutoff, tuning.head_beta),
        ];
        self.bs_euros.clear();
        self.body_euros.clear();
    }

    pub fn mode(&self) -> SmoothingMode {
        self.mode
    }
}

/// Easing function type for expression transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum EasingType {
    Linear,
    QuadInOut,
    CubicOut,
    CubicIn,
}

#[allow(dead_code)]
impl EasingType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "quad_in_out" | "quadinout" => Self::QuadInOut,
            "cubic_out" | "cubicout" => Self::CubicOut,
            "cubic_in" | "cubicin" => Self::CubicIn,
            "linear" => Self::Linear,
            _ => Self::QuadInOut,
        }
    }

    /// Evaluate the easing function at t in [0, 1].
    pub fn ease(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::QuadInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
                }
            }
            Self::CubicOut => 1.0 - (1.0 - t).powi(3),
            Self::CubicIn => t * t * t,
        }
    }
}

/// Smooth transition between two sets of morph weights.
#[cfg(feature = "native-ui")]
#[allow(dead_code)]
pub struct ExpressionTransition {
    from_weights: Vec<f32>,
    to_weights: Vec<f32>,
    duration: f32,
    elapsed: f32,
    easing: EasingType,
}

#[cfg(feature = "native-ui")]
#[allow(dead_code)]
impl ExpressionTransition {
    pub fn new(
        from_weights: Vec<f32>,
        to_weights: Vec<f32>,
        duration: f32,
        easing: EasingType,
    ) -> Self {
        Self {
            from_weights,
            to_weights,
            duration: duration.max(0.001),
            elapsed: 0.0,
            easing,
        }
    }

    /// Advance by `dt` seconds and return the interpolated weights.
    pub fn tick(&mut self, dt: f32) -> Vec<f32> {
        self.elapsed += dt;
        let t = (self.elapsed / self.duration).clamp(0.0, 1.0);
        let eased = self.easing.ease(t);

        self.from_weights
            .iter()
            .zip(self.to_weights.iter())
            .map(|(&from, &to)| from + (to - from) * eased)
            .collect()
    }

    pub fn is_done(&self) -> bool {
        self.elapsed >= self.duration
    }
}
