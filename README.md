# Fushigi3D

VTuber/PNGTuber UI or headless service for Linux.

## Features

- **Few Dependencies**: Standalone PNGTuber, MediaPipe for 3d model.
- **Voice Activity Detection**: Multiple VAD backends (energy-based, Silero)
- **OBS Integration**: WebSocket-based scene/source switching
- **Browser Source**: Self-contained avatar page for OBS browser sources
- **Web Dashboard**: Real-time configuration and preview

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

# Or run headless (browser source / OBS only)
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
fushigi3d --no-obs --no-http
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

### Browser Source

Add a browser source in OBS pointing to:
```
http://localhost:8080/avatar
```

The avatar will update in real-time based on voice activity.

## Avatar Assets

Place your avatar images in the `assets/default/` directory:

```
assets/default/
├── idle.png           # Default idle state
├── speaking.png       # Speaking state
└── expressions/
    ├── happy.png
    ├── sad.png
    └── surprised.png
```

## Web Dashboard (partially implemented)

Access the dashboard at `http://localhost:8080/`:

- **Dashboard**: Real-time status and preview
- **Settings**: Configure audio, OBS, VMC, and integrations
- **Browser Source**: Direct link for OBS (`/avatar`)

## API Endpoints (partially implemented)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Current status |
| `/api/state` | GET/POST | Avatar state |
| `/api/expression` | POST/DELETE | Set/clear expression |
| `/api/config` | GET/POST | Configuration |
| `/api/audio/devices` | GET | List audio devices |
| `/api/stream` | GET | SSE state updates |
| `/avatar` | GET | Browser source page |
| `/avatar/stream` | GET | SSE for avatar updates |

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
