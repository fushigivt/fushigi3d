# Fushigi3D

VTuber/PNGTuber UI or headless service for Linux.

## Features

- **Few Dependencies**: Standalone PNGTuber, MediaPipe for 3D model
- **Voice Activity Detection**: Multiple VAD backends (energy-based, Silero)
- **OBS Integration**: WebSocket-based scene/source switching

## Installation

```bash
# Clone the repository
git clone https://github.com/fushigivt/fushigi3d.git
cd fushigi3d

# Download free VRM models and placeholder sprites
./scripts/setup.sh

# install mediapipe and opencv 
pip install mediapipe opencv-python

# Build and run (opens 3D viewport by default)
cargo run --release

# Or run headless (no UI window)
cargo run --release -- --headless
```

## Usage

```bash
# List available audio devices
fushigi3d --list-devices

# Run with specific audio device
fushigi3d --device "hw:1,0"

# Run with custom config file
fushigi3d --config path/to/config.toml

# Run with verbose logging
fushigi3d --verbose

# Disable specific features
fushigi3d --no-obs --no-audio
```

## Configuration

Copy `config/default.toml` to `config.toml` and customize as needed.

### Audio Settings

```toml
[audio]
device = "default"
sample_rate = 16000
channels = 1

[vad]
provider = "energy"
energy_threshold_db = -40.0
attack_ms = 50
release_ms = 200
```

### OBS Integration

```toml
[obs]
enabled = true
host = "127.0.0.1"
port = 4455
mode = "scene"
idle_scene = "Idle"
speaking_scene = "Speaking"
```

## Development

```bash
# Run with debug logging
RUST_LOG=debug cargo run

# Run tests
cargo test

# Check formatting
cargo fmt --check

# Run linter
cargo clippy
```

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
