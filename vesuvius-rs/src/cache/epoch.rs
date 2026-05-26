//! Cache-wide LRU bookkeeping for purge planning.
//!
//! A single global `EpochState` shared by every per-volume cache instance
//! lives at `<cache_root>/unified/epoch.idx`. It tracks:
//!
//! - `current` — the active u8 epoch. Wraps after 256 advances; purge keeps
//!   surviving chunks within 255 epochs of `current`, so wrap is safe.
//! - `bytes_since_advance` — cache-wide bytes written into shard files since
//!   the last advance. Advancing happens once `bytes_per_epoch` is crossed.
//! - `bytes_per_epoch` — derived from the configured cache cap so the full
//!   256-slot range covers ≈1.1× cap (10% headroom).
//! - `total_chunks` — current resident chunk count (cache-wide).
//! - `epoch_times[i]` — wall time at which slot `i` was last entered.
//! - `epoch_chunks[i]` — number of chunks whose `access_epoch == i`. Updated
//!   on fill, access transition (past the `!=` filter), and evict. Used
//!   for goal-based purge planning and as the wrap guard at advance time.
//!
//! Hot path (per chunk access) does at most: one `current()` load, one
//! comparison, and on transition two histogram updates. Reads that already
//! see `access_epoch == current` short-circuit before touching this struct.
//!
//! Persistence is a flat little-endian blob; serialization is intentionally
//! minimal because the file is small (a few KB) and rewritten alongside
//! the sidecar by the same sync thread.

use super::disk::{punch_hole_at, shard_filename, SHARD_CHUNKS_PER_AXIS};
use super::purge::{PurgePlan, PurgeTarget};
use super::sidecar::{self, Sidecar, STATE_MISSING, STATE_RESIDENT};
use super::CHUNK_VOXELS;
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Watchdog tick interval. Both the periodic save and the watermark /
/// statvfs checks run at this cadence.
const WATCHDOG_INTERVAL: Duration = Duration::from_secs(30);

/// High watermark expressed as a fraction of cap (×100). At or above
/// this fill the watchdog triggers a purge.
const HIGH_WATER_PCT: u64 = 95;
/// Low watermark expressed as a fraction of cap (×100). Each watermark
/// purge frees enough chunks to bring usage down to this level.
const LOW_WATER_PCT: u64 = 80;
/// Absolute minimum free space on the cache filesystem before the
/// watchdog issues a defensive purge regardless of cap fill.
const MIN_FREE_BYTES: u64 = 5 * 1024 * 1024 * 1024;

pub const EPOCH_SLOTS: usize = 256;

/// Default cache cap when neither the env var nor configuration sets
/// one explicitly. Sized for the canonical workstation deployment.
pub const DEFAULT_CAP_BYTES: u64 = 200 * 1024 * 1024 * 1024;

/// Env var overriding the cache cap, in GB. Parsed once per process on
/// first access. Decimal integers only; invalid values fall back to
/// `DEFAULT_CAP_BYTES` and emit a warning.
pub const CAP_ENV_VAR: &str = "VESUVIUS_CACHE_CAP_GB";

/// Read the cache cap from the env var, falling back to
/// `DEFAULT_CAP_BYTES`. Cached so the env is parsed exactly once even
/// across multiple `shared_for_unified_root` calls.
pub fn cap_bytes_from_env() -> u64 {
    static CACHED: OnceLock<u64> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var(CAP_ENV_VAR) {
        Ok(raw) => match raw.trim().parse::<u64>() {
            Ok(gb) if gb > 0 => gb.saturating_mul(1024 * 1024 * 1024),
            Ok(_) => {
                log::warn!("{} must be a positive integer; using default", CAP_ENV_VAR);
                DEFAULT_CAP_BYTES
            }
            Err(e) => {
                log::warn!("{} parse failed ({}); using default", CAP_ENV_VAR, e);
                DEFAULT_CAP_BYTES
            }
        },
        Err(_) => DEFAULT_CAP_BYTES,
    })
}

/// Headroom factor applied to the configured cache cap when deriving
/// `bytes_per_epoch`. With a 10% headroom the full 256 slots cover ≈1.1×
/// cap, so even a cache that runs slightly over its target before purge
/// fires still has wrap-safe budget.
const HEADROOM_NUM: u128 = 11;
const HEADROOM_DEN: u128 = 10;

pub struct EpochState {
    inner: Mutex<Inner>,
    targets: Mutex<Vec<Weak<dyn PurgeTarget>>>,
    /// volume_ids already accumulated into the global histogram. Makes
    /// `add_from_sidecar` idempotent so the registry-time scan and the
    /// per-volume `ChunkCache::new_with_downloader` seed don't
    /// double-count when both run for the same volume. Also used by
    /// the offline sweep to skip volumes currently represented by a
    /// live `PurgeTarget`.
    accumulated: Mutex<HashSet<String>>,
    /// The unified cache root this state belongs to. Set by
    /// `shared_for_unified_root`; absent for `EpochState`s constructed
    /// directly (tests). When present, `purge_all_to_target` also runs
    /// an offline sweep against subdirs not represented by a live
    /// target.
    unified_root: OnceLock<PathBuf>,
}

struct Inner {
    current: u8,
    bytes_since_advance: u64,
    bytes_per_epoch: u64,
    cap_bytes: u64,
    total_chunks: u64,
    epoch_times: [Option<SystemTime>; EPOCH_SLOTS],
    epoch_chunks: [u32; EPOCH_SLOTS],
}

impl EpochState {
    pub fn new(cap_bytes: u64) -> Self {
        let bytes_per_epoch = derive_bytes_per_epoch(cap_bytes);
        let mut epoch_times = [None; EPOCH_SLOTS];
        // Epoch 0 is reserved for "unknown / legacy": chunks loaded from
        // a sidecar that pre-dates the access-epoch column come in at 0
        // (the EOF-fallback in `Sidecar::load`). A fresh `EpochState`
        // therefore starts at `current = 1`, so freshly written chunks
        // are distinguishable from legacy ones and the modular wrap
        // guard fires before `current` would overwrite legacy entries.
        // Cost: one epoch slot is conceptually "reserved" — negligible.
        epoch_times[1] = Some(SystemTime::now());
        Self {
            inner: Mutex::new(Inner {
                current: 1,
                bytes_since_advance: 0,
                bytes_per_epoch,
                cap_bytes,
                total_chunks: 0,
                epoch_times,
                epoch_chunks: [0; EPOCH_SLOTS],
            }),
            targets: Mutex::new(Vec::new()),
            accumulated: Mutex::new(HashSet::new()),
            unified_root: OnceLock::new(),
        }
    }

    /// Register a purge target that the watchdog can call into. Weakly
    /// held; targets that have been dropped are skipped (and GC'd) on
    /// the next tick.
    pub fn register_target(&self, target: Weak<dyn PurgeTarget>) {
        let mut guard = self.targets.lock().unwrap();
        guard.push(target);
    }

    /// Build a purge plan against the current histogram and dispatch it
    /// to every live registered target. Returns the total chunks
    /// evicted. Used by the watchdog and by tests.
    ///
    /// After live targets, also runs an offline sweep against volume
    /// subdirs under `unified_root` that aren't represented by a live
    /// target — so volumes the user opened in a previous session but
    /// not this one still participate in eviction. The offline pass is
    /// a no-op for `EpochState`s built directly without going through
    /// the registry (tests, mainly).
    pub fn purge_all_to_target(&self, target_chunks: u64) -> u64 {
        let Some(plan) = PurgePlan::build(self, target_chunks) else {
            return 0;
        };
        let mut total: u64 = 0;
        let mut targets = self.targets.lock().unwrap();
        targets.retain(|w| w.upgrade().is_some());
        let live: Vec<Arc<dyn PurgeTarget>> = targets.iter().filter_map(Weak::upgrade).collect();
        drop(targets);
        for t in live {
            total = total.saturating_add(t.run_purge(plan));
        }
        if let Some(root) = self.unified_root.get() {
            total = total.saturating_add(self.purge_offline_volumes(plan, root));
        }
        total
    }

    /// Snapshot of volume_ids that have a live `PurgeTarget` registered.
    /// Used by the offline sweep to skip volumes the live sweep already
    /// covered. Distinct from `accumulated_volume_ids` (which also
    /// includes volumes loaded only via the startup `seed_from_disk`
    /// scan).
    pub fn live_target_volume_ids(&self) -> HashSet<String> {
        let mut targets = self.targets.lock().unwrap();
        targets.retain(|w| w.upgrade().is_some());
        targets
            .iter()
            .filter_map(|w| w.upgrade())
            .map(|t| t.volume_id())
            .collect()
    }

    /// Snapshot of every volume_id seeded into the global histogram.
    /// Useful for debugging / diagnostics.
    pub fn accumulated_volume_ids(&self) -> HashSet<String> {
        self.accumulated.lock().unwrap().clone()
    }

    /// Sweep every volume subdir under `unified_root` whose volume_id
    /// is *not* in `accumulated_volume_ids`. For each such volume,
    /// load its sidecar, evict every Resident chunk matching `plan`,
    /// punch holes in the shard files, and write the updated sidecar
    /// back atomically. Decrements the global histogram via
    /// `record_evict` for every chunk freed.
    ///
    /// This is the "no live cache" counterpart to live-target purging:
    /// without it, volumes the user opened in a previous session would
    /// inflate the trigger but never get touched, defeating the cap.
    ///
    /// Concurrency: snapshots the live set up-front and skips matches.
    /// If a `ChunkCache` opens one of these volumes between the
    /// snapshot and the sidecar write, the sidecar write (atomic
    /// rename) lands either before or after that cache's
    /// `Sidecar::load`. The post-load eviction race is the same
    /// transient-zero window the live sweep already tolerates.
    pub fn purge_offline_volumes(&self, plan: PurgePlan, unified_root: &Path) -> u64 {
        let live = self.live_target_volume_ids();
        let entries = match std::fs::read_dir(unified_root) {
            Ok(r) => r,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return 0,
            Err(e) => {
                log::warn!(
                    "offline purge: read_dir {} failed: {}",
                    unified_root.display(),
                    e
                );
                return 0;
            }
        };
        let current = self.current();
        let mut total_evicted: u64 = 0;
        let chunk_bytes = CHUNK_VOXELS as u64;
        let sca = SHARD_CHUNKS_PER_AXIS;
        let shard_total = (sca as u64).pow(3) * chunk_bytes;

        for entry in entries.flatten() {
            let vol_dir = entry.path();
            if !vol_dir.is_dir() {
                continue;
            }
            let sidecar_path = sidecar::sidecar_path(&vol_dir);
            let sidecar = match Sidecar::load(&sidecar_path) {
                Ok(Some(s)) => s,
                Ok(None) => continue,
                Err(e) => {
                    log::warn!(
                        "offline purge: sidecar {} failed: {}",
                        sidecar_path.display(),
                        e
                    );
                    continue;
                }
            };
            if live.contains(&sidecar.header.volume_id) {
                continue;
            }

            let mut evicted_here: u64 = 0;
            let mut open_shards: HashMap<(u8, super::disk::ShardCoord), std::fs::File> =
                HashMap::new();

            for (lod_idx, dims) in sidecar.header.lods.iter().enumerate() {
                let lod = lod_idx as u8;
                let nx = dims.nx as u64;
                let ny = dims.ny as u64;
                for idx in 0..dims.count() {
                    if sidecar.get_state(lod, idx) != STATE_RESIDENT {
                        continue;
                    }
                    let ae = sidecar.get_access_epoch(lod, idx);
                    if !plan.is_victim(ae, current) {
                        continue;
                    }
                    let x = (idx % nx) as u32;
                    let y = ((idx / nx) % ny) as u32;
                    let z = (idx / (nx * ny)) as u32;
                    let shard = (x / sca, y / sca, z / sca);
                    let wx = (x % sca) as u64;
                    let wy = (y % sca) as u64;
                    let wz = (z % sca) as u64;
                    let in_shard_idx = (wz * sca as u64 + wy) * sca as u64 + wx;

                    // Demote sidecar state before touching the file so a
                    // ChunkCache that races us reads Missing rather than
                    // Resident-but-zero. write_to below persists this.
                    sidecar.set_state(lod, idx, STATE_MISSING);

                    let file = match open_shards.entry((lod, shard)) {
                        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                        std::collections::hash_map::Entry::Vacant(e) => {
                            let path = vol_dir.join(shard_filename(lod, shard));
                            match OpenOptions::new().read(true).write(true).open(&path) {
                                Ok(f) => {
                                    // Defensive: refuse to punch if the
                                    // file has an unexpected length (a
                                    // cache made with a different shard
                                    // size, or truncated state).
                                    match f.metadata() {
                                        Ok(m) if m.len() == shard_total => e.insert(f),
                                        Ok(m) => {
                                            log::warn!(
                                                "offline purge: skip shard {} (len {} != expected {})",
                                                path.display(),
                                                m.len(),
                                                shard_total,
                                            );
                                            continue;
                                        }
                                        Err(err) => {
                                            log::warn!(
                                                "offline purge: stat {} failed: {}",
                                                path.display(),
                                                err
                                            );
                                            continue;
                                        }
                                    }
                                }
                                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                                    // No shard file means no allocated
                                    // blocks to reclaim. State demote
                                    // above is still useful.
                                    self.record_evict(ae);
                                    evicted_here += 1;
                                    continue;
                                }
                                Err(err) => {
                                    log::warn!(
                                        "offline purge: open {} failed: {}",
                                        path.display(),
                                        err
                                    );
                                    continue;
                                }
                            }
                        }
                    };
                    let off = in_shard_idx * chunk_bytes;
                    if let Err(err) = punch_hole_at(file, off, chunk_bytes) {
                        log::warn!(
                            "offline purge: punch_hole {:?}/{:?} failed: {}",
                            shard,
                            in_shard_idx,
                            err
                        );
                        continue;
                    }
                    self.record_evict(ae);
                    evicted_here += 1;
                }
            }

            if evicted_here == 0 {
                continue;
            }
            // Persist the demoted state column. Snapshot reads the
            // current in-memory bitmaps; write_to does atomic rename.
            let snap = sidecar.snapshot();
            if let Err(e) = snap.write_to(&sidecar.header, &sidecar_path) {
                log::warn!(
                    "offline purge: sidecar write {} failed: {}",
                    sidecar_path.display(),
                    e
                );
            }
            log::info!(
                "offline purge: volume={} evicted={}",
                sidecar.header.volume_id,
                evicted_here
            );
            total_evicted += evicted_here;
        }

        total_evicted
    }

    pub fn current(&self) -> u8 {
        self.inner.lock().unwrap().current
    }

    pub fn total_chunks(&self) -> u64 {
        self.inner.lock().unwrap().total_chunks
    }

    pub fn cap_bytes(&self) -> u64 {
        self.inner.lock().unwrap().cap_bytes
    }

    pub fn bytes_per_epoch(&self) -> u64 {
        self.inner.lock().unwrap().bytes_per_epoch
    }

    /// Snapshot the per-epoch histogram for purge planning / dashboards.
    pub fn epoch_chunks(&self) -> [u32; EPOCH_SLOTS] {
        self.inner.lock().unwrap().epoch_chunks
    }

    /// Snapshot of advance timestamps. `None` for slots that haven't been
    /// entered since the state was created.
    pub fn epoch_times(&self) -> [Option<SystemTime>; EPOCH_SLOTS] {
        self.inner.lock().unwrap().epoch_times
    }

    /// Account for a newly written chunk: bumps `bytes_since_advance`,
    /// advances the epoch if the threshold is crossed, then tags the
    /// chunk with the (post-advance) current epoch and bumps the
    /// histogram bucket for that epoch. Returns the epoch the chunk
    /// should be written into the sidecar's access-epoch column.
    pub fn record_fill(&self, bytes_written: u64) -> u8 {
        let mut s = self.inner.lock().unwrap();
        s.bytes_since_advance = s.bytes_since_advance.saturating_add(bytes_written);
        Self::maybe_advance(&mut s, SystemTime::now());
        s.total_chunks += 1;
        let cur = s.current;
        s.epoch_chunks[cur as usize] = s.epoch_chunks[cur as usize].saturating_add(1);
        cur
    }

    /// Account for an access to a chunk currently tagged with `old_epoch`.
    /// Returns the current epoch; if it differs from `old_epoch`, the
    /// caller should write it into the access-epoch column.
    pub fn record_access(&self, old_epoch: u8) -> u8 {
        let mut s = self.inner.lock().unwrap();
        let cur = s.current;
        if old_epoch != cur {
            let old = s.epoch_chunks[old_epoch as usize];
            if old > 0 {
                s.epoch_chunks[old_epoch as usize] = old - 1;
            }
            s.epoch_chunks[cur as usize] = s.epoch_chunks[cur as usize].saturating_add(1);
        }
        cur
    }

    /// Add one volume's residency contribution to the global histogram
    /// and `total_chunks`. Called once per `ChunkCache` at startup, after
    /// the sidecar is loaded. Multiple volumes sharing the same unified
    /// root each call this, so the global state accumulates the union.
    ///
    /// Resident chunks contribute `+1` to `epoch_chunks[their_access]`
    /// and `+1` to `total_chunks`. Missing / Empty / Dispatched slots
    /// don't count. Per-LOD layout comes from `sidecar.header.lods`.
    ///
    /// Idempotent per `sidecar.header.volume_id`: the registry's
    /// `seed_from_disk` scan adds every on-disk volume at startup, and
    /// the per-volume `ChunkCache::new_with_downloader` also calls in;
    /// the second call for the same volume is a no-op. New volumes
    /// (not present at startup) accumulate normally on first call.
    pub fn add_from_sidecar(&self, sidecar: &Sidecar) {
        {
            let mut acc = self.accumulated.lock().unwrap();
            if !acc.insert(sidecar.header.volume_id.clone()) {
                return;
            }
        }
        let mut s = self.inner.lock().unwrap();
        for (lod_idx, dims) in sidecar.header.lods.iter().enumerate() {
            let lod = lod_idx as u8;
            for idx in 0..dims.count() {
                if sidecar.get_state(lod, idx) == STATE_RESIDENT {
                    let ae = sidecar.get_access_epoch(lod, idx);
                    s.epoch_chunks[ae as usize] = s.epoch_chunks[ae as usize].saturating_add(1);
                    s.total_chunks += 1;
                }
            }
        }
    }

    /// Account for an evicted chunk whose last-known access epoch was
    /// `victim_epoch`. Decrements both the histogram bucket and the global
    /// chunk count.
    pub fn record_evict(&self, victim_epoch: u8) {
        let mut s = self.inner.lock().unwrap();
        let cnt = s.epoch_chunks[victim_epoch as usize];
        if cnt > 0 {
            s.epoch_chunks[victim_epoch as usize] = cnt - 1;
        }
        if s.total_chunks > 0 {
            s.total_chunks -= 1;
        }
    }

    fn maybe_advance(s: &mut Inner, now: SystemTime) {
        // Cap iterations at EPOCH_SLOTS: after that many advances we've
        // wrapped through every slot, so any leftover is just bookkeeping
        // noise. This also bounds the cost of pathological callers that
        // pass a huge `bytes_written` to `record_fill`.
        let mut advances = 0usize;
        while s.bytes_since_advance >= s.bytes_per_epoch && advances < EPOCH_SLOTS {
            s.bytes_since_advance -= s.bytes_per_epoch;
            advances += 1;
            let next = s.current.wrapping_add(1);
            if s.epoch_chunks[next as usize] != 0 {
                // TODO: trigger a synchronous purge here once purge.rs is
                // wired. In the stub we just shout — the watermark policy
                // should make this unreachable in practice.
                log::warn!(
                    "epoch advance into occupied slot {} ({} chunks); purge needed",
                    next,
                    s.epoch_chunks[next as usize]
                );
            }
            s.current = next;
            s.epoch_times[next as usize] = Some(now);
        }
        if advances == EPOCH_SLOTS {
            // We've already cycled the whole ring; drop the remainder so
            // the next fill starts from a clean state.
            s.bytes_since_advance = 0;
        }
    }

    /// Persist a snapshot to `path` (atomic temp+rename). Stub format:
    /// little-endian field-by-field. Versioned via the magic.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let s = self.inner.lock().unwrap();
        let mut buf = Vec::with_capacity(EPOCH_BYTES);
        buf.extend_from_slice(MAGIC);
        buf.push(s.current);
        buf.extend_from_slice(&s.bytes_since_advance.to_le_bytes());
        buf.extend_from_slice(&s.bytes_per_epoch.to_le_bytes());
        buf.extend_from_slice(&s.cap_bytes.to_le_bytes());
        buf.extend_from_slice(&s.total_chunks.to_le_bytes());
        for t in &s.epoch_times {
            let secs = t
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            buf.extend_from_slice(&secs.to_le_bytes());
        }
        for c in &s.epoch_chunks {
            buf.extend_from_slice(&c.to_le_bytes());
        }

        let parent = path.parent().expect("epoch path has parent");
        std::fs::create_dir_all(parent)?;
        let tmp = parent.join(format!(
            "{}.tmp",
            path.file_name().unwrap().to_string_lossy()
        ));
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&buf)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)
    }

    /// Load a snapshot from `path`. Returns `Ok(None)` if absent.
    pub fn load(path: &Path) -> std::io::Result<Option<Self>> {
        use std::io::Read;
        let mut f = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };
        let mut buf = Vec::with_capacity(EPOCH_BYTES);
        f.read_to_end(&mut buf)?;
        if buf.len() < EPOCH_BYTES || &buf[..MAGIC.len()] != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad epoch state magic / length",
            ));
        }
        let mut p = MAGIC.len();
        let current = buf[p];
        p += 1;
        let bytes_since_advance = read_u64(&buf, &mut p);
        let bytes_per_epoch = read_u64(&buf, &mut p);
        let cap_bytes = read_u64(&buf, &mut p);
        let total_chunks = read_u64(&buf, &mut p);

        let mut epoch_times = [None; EPOCH_SLOTS];
        for slot in &mut epoch_times {
            let secs = read_u64(&buf, &mut p);
            *slot = if secs == 0 {
                None
            } else {
                Some(UNIX_EPOCH + Duration::from_secs(secs))
            };
        }
        let mut epoch_chunks = [0u32; EPOCH_SLOTS];
        for slot in &mut epoch_chunks {
            *slot = read_u32(&buf, &mut p);
        }

        Ok(Some(Self {
            inner: Mutex::new(Inner {
                current,
                bytes_since_advance,
                bytes_per_epoch,
                cap_bytes,
                total_chunks,
                epoch_times,
                epoch_chunks,
            }),
            targets: Mutex::new(Vec::new()),
            accumulated: Mutex::new(HashSet::new()),
            unified_root: OnceLock::new(),
        }))
    }
}

fn derive_bytes_per_epoch(cap_bytes: u64) -> u64 {
    let n = (cap_bytes as u128 * HEADROOM_NUM) / (HEADROOM_DEN * EPOCH_SLOTS as u128);
    n.max(1) as u64
}

fn read_u64(buf: &[u8], p: &mut usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&buf[*p..*p + 8]);
    *p += 8;
    u64::from_le_bytes(a)
}

fn read_u32(buf: &[u8], p: &mut usize) -> u32 {
    let mut a = [0u8; 4];
    a.copy_from_slice(&buf[*p..*p + 4]);
    *p += 4;
    u32::from_le_bytes(a)
}

const MAGIC: &[u8; 8] = b"VCEPO001";
// magic (8) + current (1) + 4×u64 (32) + 256×u64 (2048) + 256×u32 (1024) = 3113
const EPOCH_BYTES: usize = 8 + 1 + 8 * 4 + 8 * EPOCH_SLOTS + 4 * EPOCH_SLOTS;

pub fn epoch_state_path(unified_root: &Path) -> PathBuf {
    unified_root.join("epoch.idx")
}

/// Process-wide registry of `EpochState` instances keyed by the unified
/// cache root (`<cache_dir>/unified/`). Every `ChunkCache` whose volume
/// lives under the same unified root sees the same in-memory state, so
/// epoch advances, the histogram, and the chunk count are cache-wide.
///
/// First call for a given root loads the persisted state if present, or
/// constructs a fresh one with `cap_bytes`. Subsequent calls return the
/// existing `Arc` and ignore `cap_bytes` (a running process keeps the
/// cap it was first given for this root — resize-while-running isn't
/// supported yet).
pub fn shared_for_unified_root(unified_root: &Path, cap_bytes: u64) -> Arc<EpochState> {
    static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, Arc<EpochState>>>> = OnceLock::new();
    let reg = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = reg.lock().unwrap();
    if let Some(s) = g.get(unified_root) {
        return s.clone();
    }
    let path = epoch_state_path(unified_root);
    let state = match EpochState::load(&path) {
        Ok(Some(s)) => Arc::new(s),
        Ok(None) => Arc::new(EpochState::new(cap_bytes)),
        Err(e) => {
            log::warn!(
                "epoch state load failed at {}: {}; starting fresh",
                path.display(),
                e
            );
            Arc::new(EpochState::new(cap_bytes))
        }
    };
    // The persisted histogram (epoch_chunks + total_chunks) is treated as
    // advisory only — we re-accumulate from authoritative on-disk
    // sidecars right below. Zero here so the scan starts clean and the
    // counts reflect every volume in the cache dir, not just the one
    // being opened right now.
    {
        let mut s = state.inner.lock().unwrap();
        s.epoch_chunks = [0; EPOCH_SLOTS];
        s.total_chunks = 0;
    }
    // Record the root so `purge_all_to_target` can run the offline
    // sweep against volume subdirs that don't have a live target.
    let _ = state.unified_root.set(unified_root.to_path_buf());
    seed_from_disk(&state, unified_root);
    g.insert(unified_root.to_path_buf(), state.clone());

    // Spawn the watchdog daemon thread for this unified root. The
    // thread holds `Weak<EpochState>` so the EpochState can be dropped
    // (when the registry entry is cleared in tests, say); the next tick
    // notices and exits. In production no one drops the registry entry,
    // so the thread lives until process exit.
    let weak = Arc::downgrade(&state);
    let watchdog_root = unified_root.to_path_buf();
    std::thread::Builder::new()
        .name("vesuvius-cache-epoch-watchdog".into())
        .spawn(move || watchdog_loop(weak, watchdog_root))
        .expect("spawn epoch watchdog");

    state
}

/// Scan `unified_root` for every per-volume cache subdirectory and
/// accumulate each one's sidecar into the global histogram. This makes
/// the global counter cover every volume on disk, not just the ones
/// opened by the current process — without it, a 5-volume cache that
/// only opens one volume this session would mis-report total residency
/// by 80% and let stale volumes hoard disk forever.
///
/// Each load is idempotent on `volume_id` (see `add_from_sidecar`), so
/// the per-volume `ChunkCache::new_with_downloader` call later finds
/// the volume already accumulated and skips its second pass. Volumes
/// created mid-session (not on disk at startup) accumulate at that
/// point.
fn seed_from_disk(state: &EpochState, unified_root: &Path) {
    let entries = match std::fs::read_dir(unified_root) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
        Err(e) => {
            log::warn!(
                "epoch seed: read_dir {} failed: {}",
                unified_root.display(),
                e
            );
            return;
        }
    };
    let mut seeded = 0u32;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let sidecar_path = sidecar::sidecar_path(&path);
        match Sidecar::load(&sidecar_path) {
            Ok(Some(s)) => {
                state.add_from_sidecar(&s);
                seeded += 1;
            }
            Ok(None) => {}
            Err(e) => log::warn!(
                "epoch seed: sidecar {} failed to load: {}",
                sidecar_path.display(),
                e
            ),
        }
    }
    if seeded > 0 {
        log::info!(
            "epoch seed: accumulated {} volume(s) from {}",
            seeded,
            unified_root.display()
        );
    }
}

fn watchdog_loop(weak: Weak<EpochState>, unified_root: PathBuf) {
    let path = epoch_state_path(&unified_root);
    loop {
        std::thread::sleep(WATCHDOG_INTERVAL);
        let Some(state) = weak.upgrade() else {
            return;
        };

        // Save state. Failures are logged but don't kill the loop.
        if let Err(e) = state.save(&path) {
            log::warn!("epoch state save failed at {}: {}", path.display(), e);
        }

        // Watermark + statvfs purge.
        let cap = state.cap_bytes();
        if cap == 0 {
            continue;
        }
        let chunk_bytes = CHUNK_VOXELS as u64;
        let chunks_capacity = cap / chunk_bytes;
        let high_water_chunks = chunks_capacity.saturating_mul(HIGH_WATER_PCT) / 100;
        let low_water_chunks = chunks_capacity.saturating_mul(LOW_WATER_PCT) / 100;
        let total = state.total_chunks();

        let mut target_to_free: u64 = 0;
        if total > high_water_chunks {
            target_to_free = total - low_water_chunks;
        }

        if let Some(free_bytes) = statvfs_free(&unified_root) {
            if free_bytes < MIN_FREE_BYTES {
                let need_bytes = MIN_FREE_BYTES - free_bytes;
                let extra_chunks = need_bytes.div_ceil(chunk_bytes);
                target_to_free = target_to_free.max(extra_chunks);
                log::info!(
                    "epoch watchdog: low disk free ({} MiB < {} MiB), targeting {} chunks",
                    free_bytes / (1024 * 1024),
                    MIN_FREE_BYTES / (1024 * 1024),
                    extra_chunks
                );
            }
        }

        if target_to_free > 0 {
            let evicted = state.purge_all_to_target(target_to_free);
            log::info!(
                "epoch watchdog: purge target={} evicted={} (was {} chunks resident)",
                target_to_free,
                evicted,
                total
            );
        }
    }
}

#[cfg(unix)]
fn statvfs_free(path: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut buf: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c.as_ptr(), &mut buf) };
    if rc != 0 {
        return None;
    }
    Some((buf.f_bavail as u64).saturating_mul(buf.f_frsize as u64))
}

#[cfg(not(unix))]
fn statvfs_free(_path: &Path) -> Option<u64> {
    None
}

#[cfg(test)]
impl EpochState {
    /// Test-only: advance `current` by `n` epochs without planting a
    /// chunk. Used to age existing chunks so a purge test can pick a
    /// non-trivial threshold. Does not adjust the histogram (no chunks
    /// move); only `current` and `epoch_times` change.
    pub fn force_advance(&self, n: u8) {
        let mut s = self.inner.lock().unwrap();
        let now = SystemTime::now();
        for _ in 0..n {
            let next = s.current.wrapping_add(1);
            s.current = next;
            s.epoch_times[next as usize] = Some(now);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_per_epoch_is_capx11_div_256() {
        let cap = 200u64 << 30; // 200 GiB
        let bpe = derive_bytes_per_epoch(cap);
        let expected = (cap as u128 * 11 / 10 / 256) as u64;
        assert_eq!(bpe, expected);
    }

    #[test]
    fn record_fill_advances_after_threshold() {
        let s = EpochState::new(1024); // bytes_per_epoch = 1024*11/10/256 = 44
        let initial = s.current();
        // Drop in chunks larger than one epoch's worth.
        for _ in 0..10 {
            s.record_fill(100);
        }
        assert_ne!(s.current(), initial, "expected epoch advance after fills");
        assert_eq!(s.total_chunks(), 10);
    }

    #[test]
    fn record_access_moves_histogram_bucket() {
        let s = EpochState::new(1 << 20);
        // Plant a chunk in epoch 0.
        let cur0 = s.record_fill(0);
        // Force exactly one advance, planting a second chunk in the new
        // epoch.
        let bpe = s.bytes_per_epoch();
        let cur1 = s.record_fill(bpe + 1);
        assert_ne!(cur0, cur1, "expected one epoch advance");
        // First chunk is tagged cur0, second is tagged cur1.
        let h_before = s.epoch_chunks();
        assert_eq!(h_before[cur0 as usize], 1);
        assert_eq!(h_before[cur1 as usize], 1);
        // Access the first chunk; it should move from cur0 to cur1.
        let observed_cur = s.record_access(cur0);
        assert_eq!(observed_cur, cur1);
        let h_after = s.epoch_chunks();
        assert_eq!(h_after[cur0 as usize], 0);
        assert_eq!(h_after[cur1 as usize], 2);
    }

    #[test]
    fn add_from_sidecar_counts_only_resident() {
        use super::super::sidecar::{Header, Sidecar, STATE_EMPTY, STATE_MISSING, STATE_RESIDENT};
        let h = Header::new("v".into(), [256, 256, 256], 0);
        let s = Sidecar::empty(h);
        // Plant 3 Resident at epochs 10, 10, 200; 1 Empty; 1 Missing.
        s.set_state(0, 0, STATE_RESIDENT);
        s.set_access_epoch(0, 0, 10);
        s.set_state(0, 1, STATE_RESIDENT);
        s.set_access_epoch(0, 1, 10);
        s.set_state(0, 2, STATE_RESIDENT);
        s.set_access_epoch(0, 2, 200);
        s.set_state(0, 3, STATE_EMPTY);
        s.set_state(0, 4, STATE_MISSING);

        let e = EpochState::new(1 << 30);
        e.add_from_sidecar(&s);
        assert_eq!(e.total_chunks(), 3);
        let h = e.epoch_chunks();
        assert_eq!(h[10], 2);
        assert_eq!(h[200], 1);
        for i in 0..EPOCH_SLOTS {
            if i != 10 && i != 200 {
                assert_eq!(h[i], 0, "expected slot {} to be 0", i);
            }
        }
    }

    #[test]
    fn add_from_sidecar_is_idempotent_on_volume_id() {
        use super::super::sidecar::{Header, Sidecar, STATE_RESIDENT};
        let h = Header::new("v-dedup".into(), [256, 256, 256], 0);
        let s = Sidecar::empty(h);
        s.set_state(0, 0, STATE_RESIDENT);
        s.set_access_epoch(0, 0, 7);

        let e = EpochState::new(1 << 30);
        e.add_from_sidecar(&s);
        e.add_from_sidecar(&s); // second call must not double-count
        assert_eq!(e.total_chunks(), 1);
        assert_eq!(e.epoch_chunks()[7], 1);
    }

    #[test]
    fn seed_from_disk_counts_every_volume_under_unified_root() {
        use super::super::sidecar::{Header, Sidecar, STATE_RESIDENT};
        let unified = std::env::temp_dir().join(format!(
            "vesuvius-epoch-seed-{}-{}",
            std::process::id(),
            // randomize within the process to keep parallel test runs disjoint
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&unified).unwrap();

        // Plant two volume directories with sidecars.
        for (vid, ae, n) in [("vol-a", 3u8, 2usize), ("vol-b", 9u8, 1usize)] {
            let vol_dir = unified.join(vid);
            std::fs::create_dir_all(&vol_dir).unwrap();
            let h = Header::new(vid.into(), [256, 256, 256], 0);
            let s = Sidecar::empty(h);
            for idx in 0..n as u64 {
                s.set_state(0, idx, STATE_RESIDENT);
                s.set_access_epoch(0, idx, ae);
            }
            let snap = s.snapshot();
            snap.write_to(&s.header, &sidecar::sidecar_path(&vol_dir))
                .unwrap();
        }

        // Build EpochState directly and run the scan (avoids the
        // process-wide registry / watchdog thread used by
        // shared_for_unified_root).
        let state = EpochState::new(1 << 30);
        seed_from_disk(&state, &unified);
        assert_eq!(state.total_chunks(), 3);
        let h = state.epoch_chunks();
        assert_eq!(h[3], 2);
        assert_eq!(h[9], 1);

        // Calling add_from_sidecar again for an already-seeded volume
        // (e.g., when its ChunkCache attaches) must not double-count.
        let h = Header::new("vol-a".into(), [256, 256, 256], 0);
        let s2 = Sidecar::empty(h);
        s2.set_state(0, 0, STATE_RESIDENT);
        s2.set_access_epoch(0, 0, 3);
        state.add_from_sidecar(&s2);
        assert_eq!(state.total_chunks(), 3);

        std::fs::remove_dir_all(&unified).ok();
    }

    #[test]
    fn offline_sweep_evicts_unopened_volumes_and_demotes_sidecar() {
        use super::super::purge::PurgePlan;
        use super::super::sidecar::{Header, Sidecar, STATE_MISSING, STATE_RESIDENT};
        let unified = std::env::temp_dir().join(format!(
            "vesuvius-epoch-offline-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&unified).unwrap();

        // Plant an unopened volume "vol-stale" with 3 Resident chunks at
        // epoch 10. No live PurgeTarget is registered, so the offline
        // sweep should pick it up.
        let stale_dir = unified.join("vol-stale");
        std::fs::create_dir_all(&stale_dir).unwrap();
        let stale_header = Header::new("vol-stale".into(), [256, 256, 256], 0);
        let stale = Sidecar::empty(stale_header);
        for idx in 0..3u64 {
            stale.set_state(0, idx, STATE_RESIDENT);
            stale.set_access_epoch(0, idx, 10);
        }
        stale
            .snapshot()
            .write_to(&stale.header, &sidecar::sidecar_path(&stale_dir))
            .unwrap();

        // Build EpochState by hand and seed from the disk scan.
        let state = EpochState::new(1 << 30);
        state.unified_root.set(unified.clone()).unwrap();
        seed_from_disk(&state, &unified);
        assert_eq!(state.total_chunks(), 3);

        // current = 1 → force-advance so chunks at epoch 10 are aged.
        state.force_advance(200);
        let cur = state.current();
        let plan = PurgePlan {
            age_threshold: 1,
            target_chunks: 3,
            freed_chunks: 3,
        };
        assert!(plan.is_victim(10, cur));

        let evicted = state.purge_offline_volumes(plan, &unified);
        assert_eq!(evicted, 3);
        assert_eq!(state.total_chunks(), 0);

        // Re-load from disk and confirm sidecar was persisted with
        // demoted states.
        let after = Sidecar::load(&sidecar::sidecar_path(&stale_dir))
            .unwrap()
            .unwrap();
        for idx in 0..3u64 {
            assert_eq!(
                after.get_state(0, idx),
                STATE_MISSING,
                "expected chunk {} demoted after offline sweep",
                idx
            );
        }

        std::fs::remove_dir_all(&unified).ok();
    }

    #[test]
    fn offline_sweep_skips_volumes_with_live_target() {
        use super::super::purge::{PurgePlan, PurgeTarget};
        use super::super::sidecar::{Header, Sidecar, STATE_RESIDENT};

        struct FakeTarget {
            vid: String,
        }
        impl PurgeTarget for FakeTarget {
            fn volume_id(&self) -> String {
                self.vid.clone()
            }
            fn run_purge(&self, _: PurgePlan) -> u64 {
                0
            }
        }

        let unified = std::env::temp_dir().join(format!(
            "vesuvius-epoch-offline-skip-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&unified).unwrap();
        let vol_dir = unified.join("vol-live");
        std::fs::create_dir_all(&vol_dir).unwrap();
        let h = Header::new("vol-live".into(), [256, 256, 256], 0);
        let s = Sidecar::empty(h);
        s.set_state(0, 0, STATE_RESIDENT);
        s.set_access_epoch(0, 0, 10);
        s.snapshot()
            .write_to(&s.header, &sidecar::sidecar_path(&vol_dir))
            .unwrap();

        let state = EpochState::new(1 << 30);
        state.unified_root.set(unified.clone()).unwrap();
        seed_from_disk(&state, &unified);

        let target: Arc<dyn PurgeTarget> = Arc::new(FakeTarget {
            vid: "vol-live".into(),
        });
        state.register_target(Arc::downgrade(&target));

        state.force_advance(200);
        let plan = PurgePlan {
            age_threshold: 1,
            target_chunks: 1,
            freed_chunks: 1,
        };
        // Live target → offline sweep must skip this volume.
        let evicted = state.purge_offline_volumes(plan, &unified);
        assert_eq!(evicted, 0);
        std::fs::remove_dir_all(&unified).ok();
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!("vesuvius-epoch-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = epoch_state_path(&dir);

        let s = EpochState::new(1 << 30);
        s.record_fill(0);
        s.save(&path).unwrap();

        let back = EpochState::load(&path).unwrap().unwrap();
        assert_eq!(back.current(), s.current());
        assert_eq!(back.total_chunks(), s.total_chunks());
        assert_eq!(back.epoch_chunks(), s.epoch_chunks());
    }
}
