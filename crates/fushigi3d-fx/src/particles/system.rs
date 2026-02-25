use bytemuck::{Pod, Zeroable};

const MAX_PARTICLES: usize = 2048;

/// Sticker preset types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StickerPreset {
    Hearts = 0,
    Sparkles = 1,
    Confetti = 2,
    Snow = 3,
}

impl StickerPreset {
    pub const ALL: [StickerPreset; 4] = [
        StickerPreset::Hearts,
        StickerPreset::Sparkles,
        StickerPreset::Confetti,
        StickerPreset::Snow,
    ];
}

const SHAPE_HEART: u32 = 0;
const SHAPE_STAR: u32 = 1;
const SHAPE_RECT: u32 = 2;
const SHAPE_CIRCLE: u32 = 3;

struct Particle {
    position: [f32; 3],
    velocity: [f32; 3],
    lifetime: f32,
    max_lifetime: f32,
    color: [f32; 4],
    size: f32,
    rotation: f32,
    rotation_speed: f32,
    shape: u32,
    phase: f32,
}

/// GPU instance data for one particle (48 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ParticleGpuData {
    pub position: [f32; 3],
    pub size: f32,
    pub color: [f32; 4],
    pub rotation: f32,
    pub shape: u32,
    pub alpha: f32,
    pub _pad: f32,
}

struct PresetState {
    enabled: bool,
    accumulator: f32,
}

/// CPU particle system: spawn, simulate, cull.
pub struct ParticleSystem {
    particles: Vec<Particle>,
    presets: [PresetState; 4],
    spawn_rate: f32,
    particle_size: f32,
    rng_state: u32,
}

const CONFETTI_COLORS: [[f32; 4]; 6] = [
    [1.0, 0.25, 0.35, 1.0],
    [0.2, 0.85, 0.35, 1.0],
    [0.3, 0.5, 1.0, 1.0],
    [1.0, 0.9, 0.15, 1.0],
    [1.0, 0.4, 0.8, 1.0],
    [0.55, 0.3, 1.0, 1.0],
];

impl ParticleSystem {
    pub fn new() -> Self {
        Self {
            particles: Vec::with_capacity(MAX_PARTICLES),
            presets: [
                PresetState { enabled: false, accumulator: 0.0 },
                PresetState { enabled: false, accumulator: 0.0 },
                PresetState { enabled: false, accumulator: 0.0 },
                PresetState { enabled: false, accumulator: 0.0 },
            ],
            spawn_rate: 30.0,
            particle_size: 1.0,
            rng_state: 0xDEAD_BEEF,
        }
    }

    fn xorshift32(&mut self) -> u32 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng_state = x;
        x
    }

    fn rand_f32(&mut self) -> f32 {
        (self.xorshift32() & 0x7F_FFFF) as f32 / 0x80_0000 as f32
    }

    fn rand_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.rand_f32() * (hi - lo)
    }

    pub fn set_preset_enabled(&mut self, preset: StickerPreset, enabled: bool) {
        self.presets[preset as usize].enabled = enabled;
    }

    pub fn set_spawn_rate(&mut self, rate: f32) {
        self.spawn_rate = rate;
    }

    pub fn set_particle_size(&mut self, size: f32) {
        self.particle_size = size;
    }

    pub fn active_count(&self) -> usize {
        self.particles.len()
    }

    fn spawn_particle(&mut self, preset: StickerPreset) {
        let particle = match preset {
            StickerPreset::Hearts => {
                let x = self.rand_range(-0.3, 0.3);
                let z = self.rand_range(-0.05, 0.1);
                let lifetime = self.rand_range(2.0, 3.0);
                let color = if self.rand_f32() < 0.5 {
                    [1.0, 0.3, 0.5, 1.0]
                } else {
                    [0.9, 0.15, 0.25, 1.0]
                };
                Particle {
                    position: [x, 1.0, z],
                    velocity: [
                        self.rand_range(-0.02, 0.02),
                        self.rand_range(0.08, 0.15),
                        0.0,
                    ],
                    lifetime,
                    max_lifetime: lifetime,
                    color,
                    size: self.rand_range(0.8, 1.2),
                    rotation: 2.0,
                    rotation_speed: self.rand_range(-0.3, 0.3),
                    shape: SHAPE_HEART,
                    phase: 0.0,
                }
            }
            StickerPreset::Sparkles => {
                let x = self.rand_range(-0.25, 0.25);
                let y = self.rand_range(1.2, 1.5);
                let z = self.rand_range(-0.05, 0.1);
                let lifetime = self.rand_range(0.5, 1.5);
                let brightness = self.rand_range(1.5, 3.0);
                let color = if self.rand_f32() < 0.5 {
                    [brightness, brightness * 0.9, brightness * 0.4, 1.0]
                } else {
                    [brightness, brightness, brightness, 1.0]
                };
                Particle {
                    position: [x, y, z],
                    velocity: [
                        self.rand_range(-0.01, 0.01),
                        self.rand_range(-0.01, 0.01),
                        0.0,
                    ],
                    lifetime,
                    max_lifetime: lifetime,
                    color,
                    size: self.rand_range(0.5, 0.9),
                    rotation: self.rand_range(0.0, std::f32::consts::TAU),
                    rotation_speed: self.rand_range(-1.0, 1.0),
                    shape: SHAPE_STAR,
                    phase: self.rand_range(0.0, std::f32::consts::TAU),
                }
            }
            StickerPreset::Confetti => {
                let x = self.rand_range(-0.4, 0.4);
                let z = self.rand_range(-0.05, 0.1);
                let lifetime = self.rand_range(3.0, 5.0);
                let ci = (self.xorshift32() % 6) as usize;
                let color = CONFETTI_COLORS[ci];
                Particle {
                    position: [x, 1.8, z],
                    velocity: [
                        self.rand_range(-0.05, 0.05),
                        self.rand_range(-0.3, -0.15),
                        0.0,
                    ],
                    lifetime,
                    max_lifetime: lifetime,
                    color,
                    size: self.rand_range(0.6, 1.0),
                    rotation: self.rand_range(0.0, std::f32::consts::TAU),
                    rotation_speed: self.rand_range(-3.0, 3.0),
                    shape: SHAPE_RECT,
                    phase: 0.0,
                }
            }
            StickerPreset::Snow => {
                let x = self.rand_range(-0.5, 0.5);
                let z = self.rand_range(-0.05, 0.1);
                let lifetime = self.rand_range(4.0, 7.0);
                Particle {
                    position: [x, 1.8, z],
                    velocity: [0.0, self.rand_range(-0.08, -0.04), 0.0],
                    lifetime,
                    max_lifetime: lifetime,
                    color: [1.0, 1.0, 1.0, 1.0],
                    size: self.rand_range(0.3, 0.7),
                    rotation: 0.0,
                    rotation_speed: 0.0,
                    shape: SHAPE_CIRCLE,
                    phase: self.rand_range(0.0, std::f32::consts::TAU),
                }
            }
        };
        self.particles.push(particle);
    }

    pub fn update(&mut self, dt: f32) {
        // Spawn new particles for each enabled preset
        for i in 0..4 {
            if !self.presets[i].enabled {
                self.presets[i].accumulator = 0.0;
                continue;
            }
            self.presets[i].accumulator += self.spawn_rate * dt;
            while self.presets[i].accumulator >= 1.0 && self.particles.len() < MAX_PARTICLES {
                self.presets[i].accumulator -= 1.0;
                self.spawn_particle(StickerPreset::ALL[i]);
            }
        }

        // Update physics
        for p in &mut self.particles {
            p.lifetime -= dt;
            p.position[0] += p.velocity[0] * dt;
            p.position[1] += p.velocity[1] * dt;
            p.position[2] += p.velocity[2] * dt;
            p.rotation += p.rotation_speed * dt;

            // Gravity for confetti
            if p.shape == SHAPE_RECT {
                p.velocity[1] -= 0.15 * dt;
            }

            // Sinusoidal X drift for snow
            if p.shape == SHAPE_CIRCLE {
                let elapsed = p.max_lifetime - p.lifetime;
                p.position[0] += (p.phase + elapsed * 2.0).sin() * 0.03 * dt;
            }
        }

        // Cull dead particles
        self.particles.retain(|p| p.lifetime > 0.0);
    }

    /// Produce GPU instance data for all active particles.
    pub fn gpu_data(&self) -> Vec<ParticleGpuData> {
        self.particles
            .iter()
            .map(|p| {
                let t = (p.lifetime / p.max_lifetime).clamp(0.0, 1.0);
                let alpha = if p.shape == SHAPE_STAR {
                    // Pulsing for sparkles
                    let elapsed = p.max_lifetime - p.lifetime;
                    let pulse = (elapsed * 8.0 + p.phase).sin() * 0.5 + 0.5;
                    pulse * t
                } else {
                    // Fade out in last 20% of lifetime
                    if t < 0.2 { t * 5.0 } else { 1.0 }
                };

                ParticleGpuData {
                    position: p.position,
                    size: p.size * self.particle_size,
                    color: p.color,
                    rotation: p.rotation,
                    shape: p.shape,
                    alpha,
                    _pad: 0.0,
                }
            })
            .collect()
    }
}
