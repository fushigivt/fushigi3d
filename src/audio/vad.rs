//! Voice Activity Detection (VAD) processing
//!
//! Supports Silero VAD (neural-network, default) with automatic fallback to
//! a simple energy-based detector.

use crate::config::{VadConfig, VadProvider};
use crate::error::Fushigi3dError;
#[cfg(feature = "silero-vad")]
use crate::error::AudioError;
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
    /// Which provider actually produced this result
    pub provider: VadProvider,
}

impl Default for VoiceActivity {
    fn default() -> Self {
        Self {
            is_speech: false,
            confidence: 0.0,
            energy_db: -100.0,
            duration: Duration::ZERO,
            provider: VadProvider::Energy,
        }
    }
}

// ---------------------------------------------------------------------------
// Inner VAD implementations
// ---------------------------------------------------------------------------

enum VadInner {
    Energy(EnergyVad),
    #[cfg(feature = "silero-vad")]
    Silero(SileroVad),
}

// ---------------------------------------------------------------------------
// Silero VAD wrapper
// ---------------------------------------------------------------------------

#[cfg(feature = "silero-vad")]
struct SileroVad {
    detector: voice_activity_detector::VoiceActivityDetector,
    /// Onset threshold — probability must exceed this to start speech.
    onset_threshold: f32,
    /// Offset threshold — probability must drop below this to begin ending speech.
    /// Lower than onset to prevent flickering at word boundaries (hysteresis).
    offset_threshold: f32,
    is_speech: bool,
    /// Last time probability was above the offset threshold.
    last_speech_frame: Instant,
    /// How long probability must stay below offset before speech ends.
    hangover: Duration,
    /// Accumulator so the model always receives exactly 512 samples.
    /// Feeding truncated/padded chunks corrupts the RNN's internal state.
    buffer: Vec<f32>,
    /// Most recent probability from the last complete 512-sample chunk.
    last_probability: f32,
}

#[cfg(feature = "silero-vad")]
impl SileroVad {
    fn new(config: &VadConfig) -> Result<Self, Fushigi3dError> {
        let detector = voice_activity_detector::VoiceActivityDetector::builder()
            .sample_rate(16000)
            .chunk_size(512usize)
            .build()
            .map_err(|e| AudioError::VadInit(format!("Silero VAD init failed: {}", e)))?;

        let onset = config.silero_threshold;
        // Offset is onset/3, rounded down — wide hysteresis band since
        // our mic environment has low noise.
        let offset = (onset / 3.0).max(0.01);

        Ok(Self {
            detector,
            onset_threshold: onset,
            offset_threshold: offset,
            is_speech: false,
            last_speech_frame: Instant::now(),
            hangover: Duration::from_millis(config.release_ms as u64),
            buffer: Vec::with_capacity(1024),
            last_probability: 0.0,
        })
    }

    fn process(&mut self, samples: &[f32]) -> (bool, f32, f32) {
        // Compute RMS energy for the UI meter (Silero doesn't expose this)
        let energy_db = compute_energy_db(samples);

        // Accumulate samples and process in exact 512-sample chunks
        self.buffer.extend_from_slice(samples);
        while self.buffer.len() >= 512 {
            let mut chunk: Vec<f32> = self.buffer.drain(..512).collect();
            // Normalize audio level for Silero — the model was trained on
            // normalized audio and outputs low probabilities for quiet input.
            // Target RMS of 0.1 (-20 dBFS) matches typical training levels.
            normalize_rms(&mut chunk);
            self.last_probability = self.detector.predict(chunk.iter().copied());
        }

        let probability = self.last_probability;
        let now = Instant::now();

        if self.is_speech {
            // Currently speaking — only start ending when probability drops
            // below the lower offset threshold (hysteresis band).
            if probability >= self.offset_threshold {
                self.last_speech_frame = now;
            } else if now.duration_since(self.last_speech_frame) >= self.hangover {
                self.is_speech = false;
            }
        } else {
            // Currently silent — need probability above onset to start speech.
            if probability >= self.onset_threshold {
                self.last_speech_frame = now;
                self.is_speech = true;
            }
        }

        (self.is_speech, probability, energy_db)
    }
}

// ---------------------------------------------------------------------------
// Simple energy-based VAD (original implementation)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// VadProcessor — public API
// ---------------------------------------------------------------------------

/// VAD processor that supports multiple backends
pub struct VadProcessor {
    config: VadConfig,
    inner: VadInner,
    current_activity: VoiceActivity,
    state_start: Instant,
    speech_start: Option<Instant>,
    silence_start: Option<Instant>,
    active_provider: VadProvider,
}

impl VadProcessor {
    /// Create a new VAD processor
    pub fn new(config: &VadConfig) -> Result<Self, Fushigi3dError> {
        let (inner, active_provider) = match config.provider {
            VadProvider::Silero => {
                #[cfg(feature = "silero-vad")]
                {
                    match SileroVad::new(config) {
                        Ok(s) => (VadInner::Silero(s), VadProvider::Silero),
                        Err(e) => {
                            tracing::warn!(
                                "Silero VAD init failed ({}), falling back to energy VAD",
                                e
                            );
                            (
                                VadInner::Energy(EnergyVad::new(config.energy_threshold_db)),
                                VadProvider::Energy,
                            )
                        }
                    }
                }
                #[cfg(not(feature = "silero-vad"))]
                {
                    tracing::warn!(
                        "Silero VAD requested but silero-vad feature not enabled, \
                         falling back to energy VAD"
                    );
                    (
                        VadInner::Energy(EnergyVad::new(config.energy_threshold_db)),
                        VadProvider::Energy,
                    )
                }
            }
            VadProvider::Energy => (
                VadInner::Energy(EnergyVad::new(config.energy_threshold_db)),
                VadProvider::Energy,
            ),
            VadProvider::WebRtc => {
                tracing::warn!("WebRTC VAD not yet implemented, falling back to energy VAD");
                (
                    VadInner::Energy(EnergyVad::new(config.energy_threshold_db)),
                    VadProvider::Energy,
                )
            }
        };

        Ok(Self {
            config: config.clone(),
            inner,
            current_activity: VoiceActivity::default(),
            state_start: Instant::now(),
            speech_start: None,
            silence_start: Some(Instant::now()),
            active_provider,
        })
    }

    /// Process audio samples and return voice activity
    pub fn process(&mut self, samples: &[f32]) -> Result<VoiceActivity, Fushigi3dError> {
        let (raw_is_speech, confidence, energy_db) = match &mut self.inner {
            VadInner::Energy(vad) => vad.process(samples),
            #[cfg(feature = "silero-vad")]
            VadInner::Silero(vad) => vad.process(samples),
        };

        // Silero handles its own timing via hysteresis + hangover.
        // Only apply the outer attack/release layer for energy VAD.
        let is_speech = match self.active_provider {
            VadProvider::Silero => raw_is_speech,
            _ => self.apply_timing(raw_is_speech),
        };

        // Update state tracking
        if is_speech != self.current_activity.is_speech {
            self.state_start = Instant::now();
        }

        self.current_activity = VoiceActivity {
            is_speech,
            confidence,
            energy_db,
            duration: self.state_start.elapsed(),
            provider: self.active_provider,
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

    /// Which VAD provider is actually running (may differ from config if fallback occurred)
    pub fn active_provider(&self) -> VadProvider {
        self.active_provider
    }

    /// Reset the VAD state
    pub fn reset(&mut self) {
        self.current_activity = VoiceActivity::default();
        self.state_start = Instant::now();
        self.speech_start = None;
        self.silence_start = Some(Instant::now());
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Normalize a chunk to a target RMS level for Silero VAD.
///
/// Silero V5 was trained on normalized audio and outputs low probabilities
/// for quiet input.  This scales each chunk so its RMS is approximately
/// 0.1 (-20 dBFS), which matches typical training levels.  Near-silent
/// chunks (RMS < 0.001) are left untouched to avoid amplifying noise.
#[cfg(feature = "silero-vad")]
fn normalize_rms(chunk: &mut [f32]) {
    const TARGET_RMS: f32 = 0.1;
    const MIN_RMS: f32 = 0.001;
    const MAX_GAIN: f32 = 20.0;

    let sum_sq: f32 = chunk.iter().map(|s| s * s).sum();
    let rms = (sum_sq / chunk.len() as f32).sqrt();
    if rms > MIN_RMS {
        let gain = (TARGET_RMS / rms).min(MAX_GAIN);
        for s in chunk.iter_mut() {
            *s *= gain;
        }
    }
}

/// Compute RMS energy in dB from f32 samples.
#[cfg(feature = "silero-vad")]
fn compute_energy_db(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return -100.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    let rms = (sum_sq / samples.len() as f32).sqrt();
    if rms > 0.0 {
        20.0 * rms.log10()
    } else {
        -100.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        let mut result = (false, 0.0f32, 0.0f32);
        for _ in 0..20 {
            result = vad.process(&speech);
        }
        assert!(result.0, "expected speech detected after convergence");
        assert!(result.1 > 0.0, "expected confidence > 0");
    }

    #[test]
    fn test_vad_processor_energy() {
        let config = VadConfig {
            provider: VadProvider::Energy,
            ..Default::default()
        };
        let mut processor = VadProcessor::new(&config).unwrap();
        assert_eq!(processor.active_provider(), VadProvider::Energy);

        let silence: Vec<f32> = vec![0.0; 512];
        let result = processor.process(&silence).unwrap();
        assert!(!result.is_speech);
    }

    #[cfg(feature = "silero-vad")]
    #[test]
    fn test_compute_energy_db() {
        assert_eq!(compute_energy_db(&[]), -100.0);
        assert_eq!(compute_energy_db(&[0.0; 512]), -100.0);

        // 1.0 amplitude → 0 dB
        let loud: Vec<f32> = vec![1.0; 512];
        let db = compute_energy_db(&loud);
        assert!((db - 0.0).abs() < 0.1, "expected ~0 dB, got {}", db);
    }

    #[cfg(feature = "silero-vad")]
    #[test]
    fn test_vad_processor_silero() {
        let config = VadConfig::default();
        let mut processor = VadProcessor::new(&config).unwrap();
        assert_eq!(processor.active_provider(), VadProvider::Silero);

        let silence: Vec<f32> = vec![0.0; 512];
        let result = processor.process(&silence).unwrap();
        assert!(!result.is_speech);
    }
}
