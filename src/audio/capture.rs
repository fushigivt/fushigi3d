//! Audio device capture using cpal

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleRate, Stream, StreamConfig};
use crossbeam_channel::{bounded, Receiver, Sender};
use std::thread;

use crate::config::AudioConfig;
use crate::error::{AudioError, RustuberError};

/// Audio capture from input device
///
/// This struct uses a separate thread for audio capture since cpal::Stream
/// is not Send. Samples are communicated through crossbeam channels.
pub struct AudioCapture {
    sample_rx: Receiver<Vec<f32>>,
    stop_tx: Sender<()>,
    config: StreamConfig,
    _thread_handle: Option<thread::JoinHandle<()>>,
}

impl AudioCapture {
    /// Create a new audio capture from configuration
    pub fn new(config: &AudioConfig) -> Result<Self, RustuberError> {
        let host = cpal::default_host();

        // Find the requested device
        let device = if config.device == "default" {
            host.default_input_device()
                .ok_or(AudioError::NoDefaultInput)?
        } else {
            find_device_by_name(&host, &config.device)?
        };

        let device_name = device.name().unwrap_or_else(|_| "Unknown".to_string());
        tracing::info!("Using audio device: {}", device_name);

        // Get supported config
        let supported_config = device
            .supported_input_configs()
            .map_err(|e| AudioError::UnsupportedConfig(e.to_string()))?
            .filter(|c| c.channels() == config.channels)
            .find(|c| {
                c.min_sample_rate() <= SampleRate(config.sample_rate)
                    && c.max_sample_rate() >= SampleRate(config.sample_rate)
            })
            .or_else(|| {
                // Try any config if exact match not found
                device
                    .supported_input_configs()
                    .ok()?
                    .next()
            })
            .ok_or_else(|| AudioError::UnsupportedConfig("No suitable config found".to_string()))?;

        let stream_config = StreamConfig {
            channels: config.channels,
            sample_rate: SampleRate(config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(config.buffer_size),
        };

        tracing::debug!(
            "Stream config: {} Hz, {} channels, buffer size {}",
            stream_config.sample_rate.0,
            stream_config.channels,
            config.buffer_size
        );

        // Create channels for samples and stop signal
        let (sample_tx, sample_rx) = bounded::<Vec<f32>>(32);
        let (stop_tx, stop_rx) = bounded::<()>(1);

        let stream_config_clone = stream_config.clone();
        let sample_format = supported_config.sample_format();

        // Spawn the audio thread
        let thread_handle = thread::Builder::new()
            .name("audio-capture".to_string())
            .spawn(move || {
                run_audio_thread(device, stream_config_clone, sample_format, sample_tx, stop_rx);
            })
            .map_err(|e| AudioError::StreamBuild(format!("Failed to spawn audio thread: {}", e)))?;

        Ok(Self {
            sample_rx,
            stop_tx,
            config: stream_config,
            _thread_handle: Some(thread_handle),
        })
    }

    /// Get the next batch of audio samples (non-blocking)
    pub async fn get_samples(&self) -> Result<Vec<f32>, RustuberError> {
        // Use a small timeout to not block forever
        match self.sample_rx.try_recv() {
            Ok(samples) => Ok(samples),
            Err(crossbeam_channel::TryRecvError::Empty) => {
                // No samples ready, that's fine
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                Ok(Vec::new())
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                Err(AudioError::StreamBuild("Stream disconnected".to_string()).into())
            }
        }
    }

    /// Get the stream configuration
    pub fn config(&self) -> &StreamConfig {
        &self.config
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate.0
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        // Signal the audio thread to stop
        let _ = self.stop_tx.send(());
    }
}

/// Run the audio capture in a dedicated thread
fn run_audio_thread(
    device: Device,
    config: StreamConfig,
    sample_format: cpal::SampleFormat,
    sample_tx: Sender<Vec<f32>>,
    stop_rx: Receiver<()>,
) {
    let stream = match build_input_stream(&device, &config, sample_format, sample_tx) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("Failed to build audio stream: {}", e);
            return;
        }
    };

    if let Err(e) = stream.play() {
        tracing::error!("Failed to start audio stream: {}", e);
        return;
    }

    tracing::debug!("Audio capture thread started");

    // Wait for stop signal
    let _ = stop_rx.recv();

    tracing::debug!("Audio capture thread stopping");
    drop(stream);
}

/// Find an audio device by name
fn find_device_by_name(host: &cpal::Host, name: &str) -> Result<Device, RustuberError> {
    let devices = host
        .input_devices()
        .map_err(|e| AudioError::DeviceEnumeration(e.to_string()))?;

    for device in devices {
        if let Ok(device_name) = device.name() {
            if device_name.contains(name) || name.contains(&device_name) {
                return Ok(device);
            }
        }
    }

    Err(AudioError::NoDeviceFound.into())
}

/// Build input stream based on sample format
fn build_input_stream(
    device: &Device,
    config: &StreamConfig,
    sample_format: cpal::SampleFormat,
    tx: Sender<Vec<f32>>,
) -> Result<Stream, RustuberError> {
    let err_fn = |err| tracing::error!("Audio stream error: {}", err);

    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let _ = tx.try_send(data.to_vec());
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                let samples: Vec<f32> = data
                    .iter()
                    .map(|&s| s as f32 / i16::MAX as f32)
                    .collect();
                let _ = tx.try_send(samples);
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_input_stream(
            config,
            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                let samples: Vec<f32> = data
                    .iter()
                    .map(|&s| (s as f32 / u16::MAX as f32) * 2.0 - 1.0)
                    .collect();
                let _ = tx.try_send(samples);
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U8 => device.build_input_stream(
            config,
            move |data: &[u8], _: &cpal::InputCallbackInfo| {
                let samples: Vec<f32> = data
                    .iter()
                    .map(|&s| (s as f32 / 128.0) - 1.0)
                    .collect();
                let _ = tx.try_send(samples);
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I8 => device.build_input_stream(
            config,
            move |data: &[i8], _: &cpal::InputCallbackInfo| {
                let samples: Vec<f32> = data
                    .iter()
                    .map(|&s| s as f32 / i8::MAX as f32)
                    .collect();
                let _ = tx.try_send(samples);
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I32 => device.build_input_stream(
            config,
            move |data: &[i32], _: &cpal::InputCallbackInfo| {
                let samples: Vec<f32> = data
                    .iter()
                    .map(|&s| s as f32 / i32::MAX as f32)
                    .collect();
                let _ = tx.try_send(samples);
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::F64 => device.build_input_stream(
            config,
            move |data: &[f64], _: &cpal::InputCallbackInfo| {
                let samples: Vec<f32> = data.iter().map(|&s| s as f32).collect();
                let _ = tx.try_send(samples);
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U32 => device.build_input_stream(
            config,
            move |data: &[u32], _: &cpal::InputCallbackInfo| {
                let samples: Vec<f32> = data
                    .iter()
                    .map(|&s| (s as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32)
                    .collect();
                let _ = tx.try_send(samples);
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I64 => device.build_input_stream(
            config,
            move |data: &[i64], _: &cpal::InputCallbackInfo| {
                let samples: Vec<f32> = data
                    .iter()
                    .map(|&s| (s as f64 / i64::MAX as f64) as f32)
                    .collect();
                let _ = tx.try_send(samples);
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U64 => device.build_input_stream(
            config,
            move |data: &[u64], _: &cpal::InputCallbackInfo| {
                let samples: Vec<f32> = data
                    .iter()
                    .map(|&s| (s as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                    .collect();
                let _ = tx.try_send(samples);
            },
            err_fn,
            None,
        ),
        _ => {
            return Err(AudioError::UnsupportedConfig(format!(
                "Unsupported sample format: {:?}",
                sample_format
            ))
            .into());
        }
    }
    .map_err(|e| AudioError::StreamBuild(e.to_string()))?;

    Ok(stream)
}

/// List all available input devices
pub fn list_input_devices() -> Vec<String> {
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

/// Get the default input device name
pub fn default_input_device_name() -> Option<String> {
    let host = cpal::default_host();
    host.default_input_device()
        .and_then(|d| d.name().ok())
}
