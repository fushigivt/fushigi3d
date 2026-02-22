//! Audio device capture using ALSA directly.
//!
//! cpal's callback-based ALSA backend stalls on PipeWire systems, so we use
//! the `alsa` crate directly with blocking reads (the same approach `arecord`
//! takes). Samples are read as S16_LE and converted to f32.
//!
//! If the hardware sample rate differs from 16 kHz (the rate expected by the
//! VAD), we resample in software using the `rubato` crate rather than relying
//! on ALSA's `plughw:` plugin, whose default linear-interpolation resampler
//! introduces aliasing artifacts that degrade Silero VAD accuracy.

use alsa::pcm::{Access, Format, HwParams, PCM};
use alsa::{Direction, ValueOr};
use crossbeam_channel::{bounded, Receiver, Sender};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
use std::thread;

use crate::config::AudioConfig;
use crate::error::{AudioError, Fushigi3dError};

/// Target sample rate for the downstream VAD pipeline (Silero expects 16 kHz).
const TARGET_RATE: u32 = 16000;

/// Audio capture from ALSA input device.
///
/// Runs a blocking-read loop on a dedicated thread and sends mono f32 sample
/// buffers through a crossbeam channel.  Output is always 16 kHz mono
/// regardless of the hardware sample rate.
pub struct AudioCapture {
    sample_rx: Receiver<Vec<f32>>,
    stop_tx: Sender<()>,
    sample_rate: u32,
    _thread_handle: Option<thread::JoinHandle<()>>,
}

impl AudioCapture {
    /// Create a new audio capture from configuration.
    pub fn new(config: &AudioConfig) -> Result<Self, Fushigi3dError> {
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

        // Request the target rate; ALSA may give us something else (e.g. 48000)
        hw.set_rate_near(config.sample_rate, ValueOr::Nearest)
            .map_err(|e| AudioError::StreamBuild(format!("set_rate({}): {}", config.sample_rate, e)))?;
        let actual_rate = hw.get_rate()
            .map_err(|e| AudioError::StreamBuild(format!("get_rate: {}", e)))?;

        // Scale the period so that *after* resampling the output is close to
        // config.buffer_size frames (512 by default, matching Silero's chunk).
        let desired_period = if actual_rate != TARGET_RATE {
            ((config.buffer_size as u64 * actual_rate as u64) / TARGET_RATE as u64) as i64
        } else {
            config.buffer_size as i64
        };

        hw.set_period_size_near(desired_period, ValueOr::Nearest)
            .map_err(|e| AudioError::StreamBuild(format!("set_period_size: {}", e)))?;
        hw.set_buffer_size_near(desired_period * 4)
            .map_err(|e| AudioError::StreamBuild(format!("set_buffer_size: {}", e)))?;

        pcm.hw_params(&hw)
            .map_err(|e| AudioError::StreamBuild(format!("hw_params: {}", e)))?;

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

        // Build resampler if the hardware rate differs from the target
        let resampler = if actual_rate != TARGET_RATE {
            let ratio = TARGET_RATE as f64 / actual_rate as f64;
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            };
            let r = SincFixedIn::<f32>::new(
                ratio,
                2.0, // max relative ratio deviation
                params,
                actual_period as usize,
                1, // mono
            ).map_err(|e| AudioError::StreamBuild(format!("resampler init: {}", e)))?;

            tracing::info!(
                "Resampling {} Hz -> {} Hz (ratio {:.4}, input chunk {})",
                actual_rate, TARGET_RATE, ratio, actual_period,
            );
            Some(r)
        } else {
            tracing::info!("Hardware rate is {} Hz, no resampling needed", actual_rate);
            None
        };

        // Create channels for samples and stop signal
        let (sample_tx, sample_rx) = bounded::<Vec<f32>>(32);
        let (stop_tx, stop_rx) = bounded::<()>(1);

        let channels = actual_channels as u16;
        let period_frames = actual_period as usize;

        // Spawn the blocking read thread
        let thread_handle = thread::Builder::new()
            .name("audio-capture".to_string())
            .spawn(move || {
                run_alsa_thread(pcm, channels, period_frames, sample_tx, stop_rx, resampler);
            })
            .map_err(|e| AudioError::StreamBuild(format!("Failed to spawn audio thread: {}", e)))?;

        Ok(Self {
            sample_rx,
            stop_tx,
            sample_rate: TARGET_RATE,
            _thread_handle: Some(thread_handle),
        })
    }

    /// Get the next batch of audio samples (non-blocking).
    pub async fn get_samples(&self) -> Result<Vec<f32>, Fushigi3dError> {
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
    mut resampler: Option<SincFixedIn<f32>>,
) {
    let ch = channels as usize;
    // Buffer for one period of interleaved i16 samples
    let mut buf = vec![0i16; period_frames * ch];
    // Accumulator for resampling — handles the rare short-read from ALSA
    let mut resample_buf: Vec<f32> = Vec::new();

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

                // Resample to 16 kHz if needed, otherwise send directly
                if let Some(ref mut r) = resampler {
                    resample_buf.extend_from_slice(&mono);
                    while resample_buf.len() >= period_frames {
                        let chunk: Vec<f32> = resample_buf.drain(..period_frames).collect();
                        match r.process(&[&chunk], None) {
                            Ok(output) => {
                                if let Some(resampled) = output.into_iter().next() {
                                    let resampled: Vec<f32> = resampled;
                                    if !resampled.is_empty() {
                                        let _ = sample_tx.try_send(resampled);
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::error!("Resampling error: {}", e);
                            }
                        }
                    }
                } else {
                    let _ = sample_tx.try_send(mono);
                }
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
