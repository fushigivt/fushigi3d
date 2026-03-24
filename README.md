# Fushigi3D

VTuber/PNGTuber UI for Linux (untested on Mac).

## Features

- **Few Dependencies**: MediaPipe for 3D model (pngtuber works standalone)
- **Voice Activity Detection**: Multiple VAD backends (energy-based, Silero)

## Installation

Install mediapipe and opencv first, in the application directory.

```bash
# Install mediapipe and opencv locally
python3 -m venv .venv # or virtualenv .venv
.venv/bin/pip install mediapipe opencv numpy 
```

### From github release
Download the tar or zip.

Once you have mediapipe installed run:

./fushigi3d


### From source (requires rust toolchain)

```bash
# Clone the repository
git clone https://github.com/fushigivt/fushigi3d.git
cd fushigi3d

# Build and run (opens 3D viewport by default)
cargo run --release
```

## Usage

```bash
# Run
fushigi3d

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

## Phone Face Tracking

1. Install a VMC-compatible app on your phone (e.g. MeowFace)
2. In the phone app, set the destination IP to your PC's local IP and port `39539`
 (or change it via interface).

```toml
[vmc]
receiver_enabled = true
receiver_port = 39539
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
