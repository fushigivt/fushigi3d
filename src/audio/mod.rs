//! Audio processing module
//!
//! Handles audio capture and voice activity detection.

pub mod capture;
pub mod vad;

use crate::config::{AudioConfig, VadConfig};
use crate::error::Fushigi3dError;

pub use capture::AudioCapture;
pub use vad::{VadProcessor, VoiceActivity};

/// Main audio pipeline combining capture and VAD
pub struct AudioPipeline {
    capture: AudioCapture,
    vad: VadProcessor,
}

impl AudioPipeline {
    /// Create a new audio pipeline
    pub fn new(audio_config: &AudioConfig, vad_config: &VadConfig) -> Result<Self, Fushigi3dError> {
        let capture = AudioCapture::new(audio_config)?;
        let vad = VadProcessor::new(vad_config)?;

        Ok(Self { capture, vad })
    }

    /// Process one frame of audio and return speaking state
    pub async fn process(&mut self) -> Result<bool, Fushigi3dError> {
        // Get audio samples from capture
        let samples = self.capture.get_samples().await?;

        if samples.is_empty() {
            return Ok(false);
        }

        // Process through VAD
        let activity = self.vad.process(&samples)?;

        Ok(activity.is_speech)
    }

    /// Get the current voice activity state
    pub fn get_activity(&self) -> VoiceActivity {
        self.vad.current_activity()
    }
}
