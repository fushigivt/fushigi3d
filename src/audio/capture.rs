//! Audio device capture using ALSA directly.
//!
//! cpal's callback-based ALSA backend stalls on PipeWire systems, so we use
//! the `alsa` crate directly with blocking reads (the same approach `arecord`
//! takes). Samples are read as S16_LE and converted to f32.

use alsa::pcm::{Access, Format, HwParams, PCM};
use alsa::{Direction, ValueOr};
use crossbeam_channel::{bounded, Receiver, Sender};
use std::thread;

use crate::config::AudioConfig;
use crate::error::{AudioError, RustuberError};

/// Audio capture from ALSA input device.
///
/// Runs a blocking-read loop on a dedicated thread and sends mono f32 sample
/// buffers through a crossbeam channel.
pub struct AudioCapture {
    sample_rx: Receiver<Vec<f32>>,
    stop_tx: Sender<()>,
    sample_rate: u32,
    _thread_handle: Option<thread::JoinHandle<()>>,
}

impl AudioCapture {
    /// Create a new audio capture from configuration.
    pub fn new(config: &AudioConfig) -> Result<Self, RustuberError> {
        let device_name = if config.device == "default" {
            "default".to_string()
        } else {
            config.device.clone()
        };

        tracing::info!("Using audio device: {}", device_name);

        // Open the ALSA capture device
        let pcm = PCM::new(&device_name, Direction::Capture, false)
            .map_err(|e| AudioError::StreamBuild(format!("Failed to open ALSA device '{}': {}", device_name, e)))?;

        // Configure hardware parameters
        let hw = HwParams::any(&pcm)
            .map_err(|e| AudioError::StreamBuild(format!("HwParams::any: {}", e)))?;

        hw.set_access(Access::RWInterleaved)
            .map_err(|e| AudioError::StreamBuild(format!("set_access: {}", e)))?;
        hw.set_format(Format::s16())
            .map_err(|e| AudioError::StreamBuild(format!("set_format S16_LE: {}", e)))?;
        hw.set_channels(config.channels as u32)
            .map_err(|e| AudioError::StreamBuild(format!("set_channels({}): {}", config.channels, e)))?;
        hw.set_rate_near(config.sample_rate, ValueOr::Nearest)
            .map_err(|e| AudioError::StreamBuild(format!("set_rate({}): {}", config.sample_rate, e)))?;

        // Use a reasonable buffer/period — let ALSA pick defaults
        hw.set_period_size_near(config.buffer_size as i64, ValueOr::Nearest)
            .map_err(|e| AudioError::StreamBuild(format!("set_period_size: {}", e)))?;
        hw.set_buffer_size_near((config.buffer_size * 4) as i64)
            .map_err(|e| AudioError::StreamBuild(format!("set_buffer_size: {}", e)))?;

        pcm.hw_params(&hw)
            .map_err(|e| AudioError::StreamBuild(format!("hw_params: {}", e)))?;

        let actual_rate = hw.get_rate()
            .map_err(|e| AudioError::StreamBuild(format!("get_rate: {}", e)))?;
        let actual_channels = hw.get_channels()
            .map_err(|e| AudioError::StreamBuild(format!("get_channels: {}", e)))?;
        let actual_period = hw.get_period_size()
            .map_err(|e| AudioError::StreamBuild(format!("get_period_size: {}", e)))?;

        // Drop HwParams before moving pcm into the thread
        drop(hw);

        tracing::info!(
            "ALSA stream: {} Hz, {} ch, period {} frames",
            actual_rate, actual_channels, actual_period,
        );

        // Create channels for samples and stop signal
        let (sample_tx, sample_rx) = bounded::<Vec<f32>>(32);
        let (stop_tx, stop_rx) = bounded::<()>(1);

        let channels = actual_channels as u16;
        let period_frames = actual_period as usize;

        // Spawn the blocking read thread
        let thread_handle = thread::Builder::new()
            .name("audio-capture".to_string())
            .spawn(move || {
                run_alsa_thread(pcm, channels, period_frames, sample_tx, stop_rx);
            })
            .map_err(|e| AudioError::StreamBuild(format!("Failed to spawn audio thread: {}", e)))?;

        Ok(Self {
            sample_rx,
            stop_tx,
            sample_rate: actual_rate,
            _thread_handle: Some(thread_handle),
        })
    }

    /// Get the next batch of audio samples (non-blocking).
    pub async fn get_samples(&self) -> Result<Vec<f32>, RustuberError> {
        match self.sample_rx.try_recv() {
            Ok(samples) => Ok(samples),
            Err(crossbeam_channel::TryRecvError::Empty) => {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                Ok(Vec::new())
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                Err(AudioError::StreamBuild("Audio stream disconnected".to_string()).into())
            }
        }
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        // Signal the capture thread to stop
        let _ = self.stop_tx.send(());
        // Wait for the thread to finish so the ALSA device is fully released
        // before anything tries to reopen it.
        if let Some(handle) = self._thread_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Downmix interleaved multi-channel f32 samples to mono.
fn downmix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    let ch = channels as usize;
    samples
        .chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

/// Blocking ALSA read loop.
fn run_alsa_thread(
    pcm: PCM,
    channels: u16,
    period_frames: usize,
    sample_tx: Sender<Vec<f32>>,
    stop_rx: Receiver<()>,
) {
    let ch = channels as usize;
    // Buffer for one period of interleaved i16 samples
    let mut buf = vec![0i16; period_frames * ch];

    // ALSA requires prepare before reading
    if let Err(e) = pcm.prepare() {
        tracing::error!("ALSA prepare failed: {}", e);
        return;
    }

    tracing::debug!("Audio capture thread started");

    loop {
        // Check stop signal (non-blocking)
        if stop_rx.try_recv().is_ok() {
            break;
        }

        let io = match pcm.io_i16() {
            Ok(io) => io,
            Err(e) => {
                tracing::error!("Failed to get ALSA I/O: {}", e);
                break;
            }
        };

        // Blocking read — returns number of frames read
        match io.readi(&mut buf) {
            Ok(frames) => {
                let sample_count = frames as usize * ch;
                // Convert i16 → f32
                let float_samples: Vec<f32> = buf[..sample_count]
                    .iter()
                    .map(|&s| s as f32 / i16::MAX as f32)
                    .collect();

                let mut mono = if channels > 1 {
                    downmix_to_mono(&float_samples, channels)
                } else {
                    float_samples
                };

                // Remove DC offset (digital mics often have a large constant bias)
                let mean = mono.iter().sum::<f32>() / mono.len() as f32;
                for s in mono.iter_mut() {
                    *s -= mean;
                }

                let _ = sample_tx.try_send(mono);
            }
            Err(e) => {
                // Try to recover from buffer overrun/underrun
                tracing::debug!("ALSA read error (recovering): {}", e);
                if let Err(e2) = pcm.try_recover(e, true) {
                    tracing::error!("ALSA recovery failed: {}", e2);
                    break;
                }
            }
        }
    }

    tracing::debug!("Audio capture thread stopping");
}

/// List all available ALSA input devices.
pub fn list_input_devices() -> Vec<String> {
    // Delegate to cpal for device enumeration (it handles PipeWire/JACK discovery)
    use cpal::traits::{DeviceTrait, HostTrait};
    let host = cpal::default_host();
    let mut devices = Vec::new();
    if let Ok(input_devices) = host.input_devices() {
        for device in input_devices {
            if let Ok(name) = device.name() {
                devices.push(name);
            }
        }
    }
    devices
}

/// Get the default input device name.
pub fn default_input_device_name() -> Option<String> {
    use cpal::traits::{DeviceTrait, HostTrait};
    let host = cpal::default_host();
    host.default_input_device()
        .and_then(|d| d.name().ok())
}
