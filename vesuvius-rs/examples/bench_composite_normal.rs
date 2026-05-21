//! Microbenchmark for the inner ray-march loop in `ObjVolume::paint`'s
//! compositing branch — the loop that, for each surface pixel, walks a
//! few dozen samples along the surface normal through the cache.
//!
//! Builds a `UnifiedVolume` backed by a `SyntheticBackfiller` over a
//! moderate-sized synthetic volume (random-ish bytes), pre-warms it so
//! every chunk is resident in mmaps, then times:
//!
//!  1. The baseline path that callers use today — per-sample
//!     `Volume::get_interpolated(...)`.
//!  2. The proposed fast path — `UnifiedVolume::composite_along_normal`,
//!     which amortizes the chunk lookup / interpolation lattice across
//!     the whole ray.
//!
//! Usage:
//!   cargo run -p vesuvius-rs --example bench_composite_normal --release
//!
//! Tunables via env:
//!   BENCH_RAYS     - number of rays (default 65536)
//!   BENCH_LAYERS   - samples per ray (default 25)
//!   BENCH_EXTENT   - cube side in voxels (default 512, must be /64)

use std::sync::Arc;
use std::time::{Duration, Instant};

use vesuvius_rs::cache::backfillers::synthetic::SyntheticBackfiller;
use vesuvius_rs::cache::{ChunkCache, ChunkKey, UnifiedVolume, CHUNK_SIDE};
use vesuvius_rs::volume::{Volume, VoxelVolume};

fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let extent_vox = env_usize("BENCH_EXTENT", 512);
    assert!(extent_vox % CHUNK_SIDE == 0, "BENCH_EXTENT must be a multiple of {}", CHUNK_SIDE);
    let chunks_per_axis = (extent_vox / CHUNK_SIDE) as u32;
    let num_rays = env_usize("BENCH_RAYS", 65_536);
    let layers = env_usize("BENCH_LAYERS", 25) as i32;

    println!(
        "config: extent={vox}³ ({chunks}³ chunks), rays={rays}, layers/ray={layers}",
        vox = extent_vox,
        chunks = chunks_per_axis,
        rays = num_rays,
        layers = layers,
    );

    // Random-ish but cheap-to-compute pattern. Real scan data is moderately
    // noisy; we want the AlphaCompositionState "speed through empty" branch
    // to fire some of the time but not always.
    let backfiller = Arc::new(SyntheticBackfiller::new(
        "bench-composite",
        [extent_vox as u32, extent_vox as u32, extent_vox as u32],
        0,
        |x, y, z, _lod| {
            let mut h: u64 = 0xcbf29ce484222325;
            h ^= x as u64;
            h = h.wrapping_mul(0x100000001b3);
            h ^= y as u64;
            h = h.wrapping_mul(0x100000001b3);
            h ^= z as u64;
            h = h.wrapping_mul(0x100000001b3);
            (h >> 24) as u8
        },
    ));

    let tmp = tempfile::tempdir().expect("tempdir");
    let cache = ChunkCache::new(tmp.path(), backfiller);

    // Pre-warm: dispatch all chunks first (so worker threads can fan out),
    // then wait_for each one. With max_lod=0 there's only one LOD.
    let t_warm = Instant::now();
    let mut keys = Vec::with_capacity((chunks_per_axis as usize).pow(3));
    for cz in 0..chunks_per_axis {
        for cy in 0..chunks_per_axis {
            for cx in 0..chunks_per_axis {
                let key = ChunkKey::new(0, cx, cy, cz);
                let _ = cache.state_or_fetch(key);
                keys.push(key);
            }
        }
    }
    for &key in &keys {
        let st = cache.wait_for(key, Duration::from_secs(60));
        debug_assert!(st.as_resident().is_some(), "chunk should be resident: {:?}", key);
    }
    println!(
        "pre-warm: {} chunks resident in {:?}",
        keys.len(),
        t_warm.elapsed()
    );

    // Build the wrapped volume used by the baseline path.
    let unified = UnifiedVolume::new(cache.clone());
    let vol = Volume::new(unified);

    // Generate ray data. Random base positions inside a margin, random
    // unit normals.
    let margin = (layers as f64 + 4.0).max(8.0);
    let lo = margin;
    let hi = extent_vox as f64 - margin;
    let mut rng = SimpleRng::new(0xC0FFEE);
    let mut bases = Vec::with_capacity(num_rays);
    let mut normals = Vec::with_capacity(num_rays);
    for _ in 0..num_rays {
        bases.push([rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi)]);
        let nx = rng.uniform(-1.0, 1.0);
        let ny = rng.uniform(-1.0, 1.0);
        let nz = rng.uniform(-1.0, 1.0);
        let l = (nx * nx + ny * ny + nz * nz).sqrt();
        normals.push([nx / l, ny / l, nz / l]);
    }

    let downsampling = 1i32;
    let ffactor = 1.0f64;
    let half = layers / 2;

    // --- Baseline ---
    let mut iters = 3;
    let mut best_baseline = Duration::from_secs(u64::MAX);
    let mut sink: u64 = 0;
    for it in 0..iters {
        let t0 = Instant::now();
        for i in 0..num_rays {
            let [x0, y0, z0] = bases[i];
            let [nx, ny, nz] = normals[i];
            // Max-composition equivalent — keep the inner loop simple to
            // reflect *the addressing/interpolation cost*, not the
            // composition update. Compare like-for-like below.
            let mut acc: u8 = 0;
            for w in -half..=half {
                let wf = w as f64;
                let px = (x0 + wf * nx) / ffactor;
                let py = (y0 + wf * ny) / ffactor;
                let pz = (z0 + wf * nz) / ffactor;
                let v = vol.get_interpolated([px, py, pz], downsampling);
                if v > acc {
                    acc = v;
                }
            }
            sink = sink.wrapping_add(acc as u64);
        }
        let dt = t0.elapsed();
        let samples = (num_rays as u64) * (layers as u64);
        let m = samples as f64 / dt.as_secs_f64() / 1e6;
        println!(
            "[baseline iter {}] {:?}  ({:.1} Msamples/s, {:.2} ns/sample)",
            it,
            dt,
            m,
            dt.as_secs_f64() * 1e9 / samples as f64
        );
        if dt < best_baseline {
            best_baseline = dt;
        }
    }
    let baseline_sink = sink;

    // --- Fast path (placeholder same as baseline until implemented) ---
    iters = 3;
    let mut best_fast = Duration::from_secs(u64::MAX);
    sink = 0;
    // Re-acquire the inner UnifiedVolume so we can call the specialized
    // method directly without going through the trait object indirection.
    let unified_fast = UnifiedVolume::new(cache.clone());
    for it in 0..iters {
        let t0 = Instant::now();
        for i in 0..num_rays {
            let [x0, y0, z0] = bases[i];
            let [nx, ny, nz] = normals[i];
            let base = [x0 / ffactor, y0 / ffactor, z0 / ffactor];
            let dir = [nx / ffactor, ny / ffactor, nz / ffactor];
            let acc = unified_fast.max_along_normal(base, dir, -half as f64, (half + 1) as f64, downsampling);
            sink = sink.wrapping_add(acc as u64);
        }
        let dt = t0.elapsed();
        let samples = (num_rays as u64) * (layers as u64);
        let m = samples as f64 / dt.as_secs_f64() / 1e6;
        println!(
            "[fast    iter {}] {:?}  ({:.1} Msamples/s, {:.2} ns/sample)",
            it,
            dt,
            m,
            dt.as_secs_f64() * 1e9 / samples as f64
        );
        if dt < best_fast {
            best_fast = dt;
        }
    }

    println!("baseline_sink={} fast_sink={}", baseline_sink, sink);
    if sink != baseline_sink {
        eprintln!("WARN: sinks differ — output not equivalent");
    }
    println!(
        "speedup (best/best): {:.2}× ({:?} -> {:?})",
        best_baseline.as_secs_f64() / best_fast.as_secs_f64(),
        best_baseline,
        best_fast
    );
}

// Tiny deterministic LCG so we don't have to wire `rand` everywhere. Quality
// doesn't matter — we just want reproducible coords.
struct SimpleRng {
    state: u64,
}
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_mul(0x9E3779B97F4A7C15) | 1 }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let t = (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        lo + (hi - lo) * t
    }
}
