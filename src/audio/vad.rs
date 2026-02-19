//! Voice Activity Detection (VAD) processing

use crate::config::{VadConfig, VadProvider};
use crate::error::RustuberError;
use std::time::{Duration, Instant};

/// Voice activity detection result
#[derive(Debug, Clone, Copy)]
pub struct VoiceActivity {
    /// Whether speech is detected
    pub is_speech: bool,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f32,
    /// Energy level in dB
    pub energy_db: f32,
    /// Duration of current state
    pub duration: Duration,
}

impl Default for VoiceActivity {
    fn default() -> Self {
        Self {
            is_speech: false,
            confidence: 0.0,
            energy_db: -100.0,
            duration: Duration::ZERO,
        }
    }
}

/// VAD processor that supports multiple backends
pub struct VadProcessor {
    config: VadConfig,
    inner: VadInner,
    current_activity: VoiceActivity,
    state_start: Instant,
    speech_start: Option<Instant>,
    silence_start: Option<Instant>,
}

enum VadInner {
    Energy(EnergyVad),
    // Future: Silero, WebRTC
}

impl VadProcessor {
    /// Create a new VAD processor
    pub fn new(config: &VadConfig) -> Result<Self, RustuberError> {
        let inner = match config.provider {
            VadProvider::Energy => VadInner::Energy(EnergyVad::new(config.energy_threshold_db)),
            VadProvider::Silero => {
                tracing::warn!("Silero VAD not yet implemented, falling back to energy-based");
                VadInner::Energy(EnergyVad::new(config.energy_threshold_db))
            }
            VadProvider::WebRtc => {
                tracing::warn!("WebRTC VAD not yet implemented, falling back to energy-based");
                VadInner::Energy(EnergyVad::new(config.energy_threshold_db))
            }
        };

        Ok(Self {
            config: config.clone(),
            inner,
            current_activity: VoiceActivity::default(),
            state_start: Instant::now(),
            speech_start: None,
            silence_start: Some(Instant::now()),
        })
    }

    /// Process audio samples and return voice activity
    pub fn process(&mut self, samples: &[f32]) -> Result<VoiceActivity, RustuberError> {
        let (raw_is_speech, confidence, energy_db) = match &mut self.inner {
            VadInner::Energy(vad) => vad.process(samples),
        };

        // Apply attack/release timing
        let is_speech = self.apply_timing(raw_is_speech);

        // Update state tracking
        if is_speech != self.current_activity.is_speech {
            self.state_start = Instant::now();
        }

        self.current_activity = VoiceActivity {
            is_speech,
            confidence,
            energy_db,
            duration: self.state_start.elapsed(),
        };

        Ok(self.current_activity)
    }

    /// Apply attack and release timing
    fn apply_timing(&mut self, raw_is_speech: bool) -> bool {
        let attack_duration = Duration::from_millis(self.config.attack_ms as u64);
        let release_duration = Duration::from_millis(self.config.release_ms as u64);
        let min_speech = Duration::from_millis(self.config.min_speech_ms as u64);

        if raw_is_speech {
            self.silence_start = None;

            if self.speech_start.is_none() {
                self.speech_start = Some(Instant::now());
            }

            // Check if speech has been sustained long enough
            if let Some(start) = self.speech_start {
                if start.elapsed() >= attack_duration {
                    return true;
                }
            }
        } else {
            self.speech_start = None;

            if self.silence_start.is_none() {
                self.silence_start = Some(Instant::now());
            }

            // Keep speech state during release period
            if let Some(start) = self.silence_start {
                if start.elapsed() < release_duration && self.current_activity.is_speech {
                    return true;
                }
            }
        }

        // Return current state if we're in a transition period
        if self.current_activity.is_speech && self.current_activity.duration < min_speech {
            return true;
        }

        raw_is_speech
    }

    /// Get the current voice activity state
    pub fn current_activity(&self) -> VoiceActivity {
        self.current_activity
    }

    /// Reset the VAD state
    pub fn reset(&mut self) {
        self.current_activity = VoiceActivity::default();
        self.state_start = Instant::now();
        self.speech_start = None;
        self.silence_start = Some(Instant::now());
    }
}

/// Simple energy-based VAD
struct EnergyVad {
    threshold_db: f32,
    smoothed_energy: f32,
    smoothing_factor: f32,
}

impl EnergyVad {
    fn new(threshold_db: f32) -> Self {
        Self {
            threshold_db,
            smoothed_energy: -100.0,
            smoothing_factor: 0.3,
        }
    }

    fn process(&mut self, samples: &[f32]) -> (bool, f32, f32) {
        if samples.is_empty() {
            return (false, 0.0, self.smoothed_energy);
        }

        // Calculate RMS energy
        let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
        let rms = (sum_sq / samples.len() as f32).sqrt();

        // Convert to dB
        let energy_db = if rms > 0.0 {
            20.0 * rms.log10()
        } else {
            -100.0
        };

        // Apply smoothing
        self.smoothed_energy = self.smoothing_factor * energy_db
            + (1.0 - self.smoothing_factor) * self.smoothed_energy;

        // Calculate confidence based on how far above threshold
        let above_threshold = self.smoothed_energy - self.threshold_db;
        let confidence = if above_threshold > 0.0 {
            (above_threshold / 20.0).min(1.0) // Saturate at 20dB above threshold
        } else {
            0.0
        };

        let is_speech = self.smoothed_energy > self.threshold_db;

        (is_speech, confidence, self.smoothed_energy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_vad_silence() {
        let mut vad = EnergyVad::new(-40.0);
        let silence: Vec<f32> = vec![0.0; 512];
        let (is_speech, _, _) = vad.process(&silence);
        assert!(!is_speech);
    }

    #[test]
    fn test_energy_vad_speech() {
        let mut vad = EnergyVad::new(-40.0);
        // Generate loud signal
        let speech: Vec<f32> = (0..512)
            .map(|i| (i as f32 * 0.1).sin() * 0.5)
            .collect();
        // Feed multiple frames so the smoothed energy converges past the threshold
        // (smoothed_energy starts at -100 dB, needs several iterations to climb)
        let mut result = (false, 0.0f32, 0.0f32);
        for _ in 0..20 {
            result = vad.process(&speech);
        }
        assert!(result.0, "expected speech detected after convergence");
        assert!(result.1 > 0.0, "expected confidence > 0");
    }

    #[test]
    fn test_vad_processor() {
        let config = VadConfig::default();
        let mut processor = VadProcessor::new(&config).unwrap();

        let silence: Vec<f32> = vec![0.0; 512];
        let result = processor.process(&silence).unwrap();
        assert!(!result.is_speech);
    }
}
