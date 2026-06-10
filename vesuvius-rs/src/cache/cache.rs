//! In-memory chunk cache + plan-based async dispatch executor.
//!
//! Two-level scheduling:
//!   - **Chunks** are the unit the paint loop asks for (64³).
//!   - **Sources** are the unit backfillers actually fetch (one native zarr
//!     chunk, one HTTP GET, one decoded blob, …). A single source can be
//!     consumed by many chunks; the cache deduplicates so the fetch runs
//!     exactly once per source key.
//!
//! Flow for one chunk miss:
//!   1. `state_or_fetch` → `dispatch_chunk` → backfiller emits a
//!      `BackfillPlan`.
//!   2. The chunk is parked in `chunks` with a counter of unresolved sources.
//!   3. For each source: if first seen, queue a `FetchSource` task (Compute)
//!      or hand the URL to the downloader (Download); otherwise attach the
//!      chunk as a waiter on the existing source.
//!   4. When a source resolves, every waiter chunk's counter drops by one;
//!      when a chunk's counter hits zero, an `Extract` task is queued.
//!   5. Extract runs the backfiller's closure → writes to disk → mmaps →
//!      transitions chunk state to `Resident`.
//!
//! ### LIFO ordering + age pruning
//!
//! Tasks live in a `BTreeMap` keyed by `!seq`, i.e. plain LIFO — the most
//! recently submitted (or touched) entry pops first. Each paint frame
//! re-enters `state_or_fetch` for every chunk it wants and that re-touches
//! in-flight entries so they re-arm to the head of the queue. Older
//! un-touched entries slide toward the tail and either get processed in
//! LIFO order when workers catch up, or culled by `MAX_AGE`.
//!
//! The queue is **unbounded**: dedup happens upstream (cache's source map
//! ensures one source-key → one FetchSource enqueue; `satisfy` enqueues at
//! most one Extract per chunk). Workers prune at two points:
//!
//!   * **Age:** entries older than `MAX_AGE` are dropped + cancelled at pop.
//!   * **Already-met:** at pop, skip Extract for chunks that became
//!     Resident through another path, and FetchSource for sources that
//!     are already Done. Defensive against cooldown-retry races.

use super::backfiller::{
    BackfillError, BackfillPlan, ChunkBackfiller, ExtractedChunk, SourceOutcome, SourcePayload, SourceSpec,
};
use super::disk::{ChunkBitState, DiskStore, LoadOutcome, ShardCoord, ShardSnapshot};
use super::downloader::{DownloadError, DownloadResult, Downloader, OnDone, SubmitResult};
use super::epoch::{self, EpochState};
use super::purge::{PurgePlan, PurgeTarget};
use super::spill::SpillStore;
use super::state::{ChunkKey, ChunkState};
use super::{CHUNK_VOXELS, MAX_AGE};
use dashmap::DashMap;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant, SystemTime};

const COOLDOWN: Duration = Duration::from_secs(10);
const SHORT_COOLDOWN: Duration = Duration::from_millis(150);
const PERMANENT_COOLDOWN: Duration = Duration::from_secs(60 * 60 * 24 * 365);
/// Small worker pool — extract + decode is CPU-bound but lock-light. Keeping
/// the count low reduces the chance that a worker stalls behind a
/// DashMap shard another worker is holding.
const DEFAULT_WORKERS: usize = 4;

pub struct ChunkCache {
    inner: Arc<Inner>,
}

struct Inner {
    map: DashMap<ChunkKey, Arc<ChunkState>>,
    /// Chunks currently being dispatched. Acts as an atomic claim so two
    /// threads racing on the same key don't both run `dispatch_chunk`. We
    /// can't use `map.entry().or_insert_with` for this because the
    /// `or_insert_with` closure runs inside a shard-write lock, and
    /// `dispatch_chunk` synchronously triggers `complete_source` paths
    /// that may try to insert into the same shard — re-entrant DashMap
    /// access deadlocks.
    dispatching: DashMap<ChunkKey, ()>,
    disk: DiskStore,
    /// On-disk spill for downloaded source bytes. Sits between the
    /// downloader and Extract so the compressed payload doesn't live on
    /// the heap.
    spill: SpillStore,
    backfiller: Arc<dyn ChunkBackfiller>,
    /// `Mutex<HashMap>` rather than `DashMap` so claim-the-slot for a fresh
    /// source key is atomic. The lock is never held across `try_submit`
    /// (downloader/task queue) because submit can synchronously invoke
    /// callbacks that re-enter `complete_source` on the same thread.
    sources: Mutex<HashMap<String, Arc<Mutex<SourceState>>>>,
    chunks: DashMap<ChunkKey, Arc<Mutex<ChunkProgress>>>,
    /// Reverse index for `SourceSpec::Chunk` dependencies: when chunk `K`
    /// transitions to `Resident`, `publish_resident(K, …)` drains
    /// `pending_chunk_sources[K]` and completes each listed source key with
    /// `K`'s `Arc<ChunkState>` as the payload. Source-key entries are
    /// deduplicated by `register_source`'s Phase 1, so at most one entry per
    /// child chunk lands here — multiple parents waiting on the same child
    /// all attach as waiters on the same source state.
    pending_chunk_sources: DashMap<ChunkKey, Vec<String>>,
    task_queue: TaskQueue,
    downloader: Arc<Downloader>,
    /// Frame counter that gates per-Pending touch debouncing. Bumped by
    /// `ChunkCache::advance_frame` (called from `reset_for_painting` at
    /// the start of each pane paint). Initialized to 1 so a fresh
    /// `Pending` — whose `last_touched_frame` starts at 0 — always fires
    /// its first touch. Each `state_or_fetch` on a Pending chunk
    /// compares this against the chunk's stamp; equal means "already
    /// touched this frame, skip the queue mutexes."
    frame: AtomicU64,
    /// Cache-wide LRU bookkeeping shared across volumes under the same
    /// unified root. Bumped on chunk fill (write path) and on access
    /// transitions (read path). See `epoch.rs`.
    epoch: Arc<EpochState>,
}

enum SourceState {
    Pending {
        waiters: Vec<ChunkKey>,
    },
    /// Source completed. `remaining_consumers` counts chunks that have
    /// claimed this source but haven't yet finished `extract_chunk`. When
    /// it reaches zero the source entry is evicted from
    /// `Inner::sources`, dropping its `SourcePayload` (typically an
    /// `Arc<Mmap>`) so the kernel reclaims the spilled file's pages and
    /// disk space.
    Done {
        outcome: SourceOutcome,
        remaining_consumers: usize,
    },
}

struct ChunkProgress {
    /// Order matches the original `BackfillPlan.sources` ordering — the
    /// extract closure receives outcomes in this order.
    order: Vec<String>,
    remaining: usize,
    results: HashMap<String, SourceOutcome>,
    extract: Option<
        Box<dyn FnOnce(&[SourceOutcome]) -> Result<Vec<(ChunkKey, ExtractedChunk)>, BackfillError> + Send + 'static>,
    >,
    /// Sibling chunks pre-claimed as Pending at dispatch time. The
    /// extract task consults this list to (a) promote each sibling to its
    /// Resident/Empty state in-memory on success, and (b) clear the
    /// Pending claim on transient/permanent failure so retries can happen
    /// via a fresh dispatch.
    covered: Vec<ChunkKey>,
}

enum Task {
    FetchSource {
        key: String,
        fetch: Box<dyn FnOnce() -> SourceOutcome + Send + 'static>,
    },
    Extract,
}

/// Pure-LIFO queue for cache-side `Task`s. BTreeMap keyed by `!seq` so the
/// most recently submitted (or re-touched) entry pops first. Earlier
/// iterations split work across LOD-rank and viewport-distance tiers, but
/// in practice that let coarse-LOD work outrank current-viewport work and
/// stall painting. The simpler model: paint always asks for what it wants
/// right now, and "right now" wins. Older requests slide toward the tail
/// and either get processed in LIFO order when workers catch up, or culled
/// by `MAX_AGE`.
///
/// `chunk_index` mirrors `entries` keyed by `ChunkKey` for the O(1) lookup
/// `touch` needs to refresh seq + `added_at` on in-flight entries.
///
/// Unbounded; dedup happens at the cache layer (source-key + chunk-key
/// uniqueness). The only staleness check is age — see the module docs.
struct TaskQueue {
    inner: Mutex<TaskQueueInner>,
    not_empty: Condvar,
    max_age: Duration,
}

struct TaskQueueInner {
    entries: BTreeMap<u64, TaskEntry>,
    /// Reverse lookup: every queue key currently registered for a given
    /// chunk. A chunk can have several (its sources' FetchSource tasks,
    /// plus an Extract task once sources resolve). Maintained in lockstep
    /// with `entries`.
    chunk_index: HashMap<ChunkKey, Vec<u64>>,
    next_seq: u64,
    closed: bool,
}

/// Encode `seq` so that **larger** seq sorts BEFORE smaller (BTreeMap pops
/// smallest key first → LIFO). `seq` is monotonically increasing so `!seq`
/// is monotonically decreasing.
fn rev_seq(seq: u64) -> u64 {
    !seq
}

struct TaskEntry {
    chunk: ChunkKey,
    added_at: Instant,
    task: Task,
}

enum RegisterResult {
    /// First observer — fetch was queued (cache pool or downloader).
    Queued,
    /// An earlier observer's fetch is in-flight; we're now a waiter.
    AttachedPending,
    /// Source already resolved; outcome is returned for the caller to apply.
    AlreadyDone(SourceOutcome),
    /// Task / download queue is full; nothing was registered.
    QueueFull,
}

enum SatisfyResult {
    /// Either: progress updated and we're still waiting on other sources,
    /// or: all sources done and the Extract task was queued successfully,
    /// or: chunk progress was already evicted.
    Ok,
    /// All sources done but the Extract task couldn't be enqueued. Caller
    /// should put the chunk into a short cooldown so paint retries.
    QueueFullOnExtract,
}

/// Process-wide handle to one unified cache root (`<cache_dir>/unified/`).
///
/// This is the first-class entry point for the cache subsystem: app
/// startup gets-or-creates a `UnifiedCache` for the cache directory, and
/// every `ChunkCache` opened against that directory is reachable from
/// it. The same `UnifiedCache` owns:
///
///   * the cache-wide `EpochState` (shared across every volume under
///     this root — drives LRU bookkeeping, purge plans, watchdog),
///   * a registry of live per-volume `ChunkCache` `Inner`s so duplicate
///     opens for the same volume collapse to a single instance instead
///     of racing two sidecars / two purge targets / two sync threads
///     against one on-disk directory.
///
/// Singleton per unified root: `for_cache_dir(d)` and
/// `for_unified_root(d/unified)` always return clones of the same
/// `Arc<UnifiedCache>` within a process.
pub struct UnifiedCache {
    unified_root: PathBuf,
    epoch: Arc<EpochState>,
    /// Per-volume `Inner` registry, keyed by `backfiller.volume_id()`.
    /// `Weak` so external drop of every `ChunkCache` for a volume
    /// allows reclamation; in practice the per-cache worker threads +
    /// PurgeTarget registration keep the strong count > 0 for the
    /// process lifetime, which is the intended production behavior.
    volumes: Mutex<HashMap<String, Weak<Inner>>>,
}

fn unified_registry() -> &'static Mutex<HashMap<PathBuf, Arc<UnifiedCache>>> {
    static R: OnceLock<Mutex<HashMap<PathBuf, Arc<UnifiedCache>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

impl UnifiedCache {
    /// Resolve the cache directory passed by app config into a
    /// `UnifiedCache`. `cache_root` is the parent of the `unified/`
    /// subdir (the same argument that gets passed to
    /// `ChunkCache::new`).
    pub fn for_cache_dir(cache_root: impl AsRef<Path>) -> Arc<Self> {
        let unified_root = cache_root.as_ref().join("unified");
        let _ = std::fs::create_dir_all(&unified_root);
        Self::for_unified_root(unified_root)
    }

    /// Resolve a `unified/` directory directly. Most callers want
    /// `for_cache_dir`; this is the lower-level entry point for code
    /// that already holds the unified path (the epoch watchdog, the
    /// offline purge sweep).
    pub fn for_unified_root(unified_root: impl AsRef<Path>) -> Arc<Self> {
        let path = unified_root.as_ref();
        // Canonicalize so two callers that pass equivalent-but-differing
        // path strings (relative vs absolute, symlinks) still collapse.
        let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let mut reg = unified_registry().lock().unwrap();
        if let Some(c) = reg.get(&key) {
            return c.clone();
        }
        // EpochState construction has its own registry inside `epoch`;
        // its singleton is what we expose via `epoch_state()`. The two
        // registries collapse on the same canonicalized root, so a
        // caller routing through either entry point sees identical
        // state.
        let epoch = epoch::shared_for_unified_root(&key, epoch::cap_bytes_from_env());
        let unified = Arc::new(UnifiedCache {
            unified_root: key.clone(),
            epoch,
            volumes: Mutex::new(HashMap::new()),
        });
        reg.insert(key, unified.clone());
        unified
    }

    pub fn unified_root(&self) -> &Path {
        &self.unified_root
    }

    /// The cache-wide LRU bookkeeping (epoch counter, residency
    /// histogram, purge target registry). Use this for stats
    /// dashboards or to register custom purge logic.
    pub fn epoch_state(&self) -> &Arc<EpochState> {
        &self.epoch
    }

    /// Run the synchronous startup-time maintenance pass: if the
    /// on-disk residency observed by the construction-time survey is
    /// already above the high-water mark (or disk free space is below
    /// `MIN_FREE_BYTES`), evict down to the low-water mark before
    /// returning. Call this once at app startup, after building the
    /// `UnifiedCache` and before opening any volumes, so the app
    /// starts in a known good state rather than racing the
    /// background watchdog.
    ///
    /// Returns the number of chunks evicted (0 if the cache was
    /// already comfortably below water).
    pub fn run_startup_maintenance(&self) -> u64 {
        log::info!(
            target: super::purge::LOG_TARGET,
            "startup maintenance: surveying {} (cap={} GiB, resident={} chunks)",
            self.unified_root.display(),
            self.epoch.cap_bytes() / (1024 * 1024 * 1024),
            self.epoch.total_chunks(),
        );
        self.epoch.run_purge_pass("startup")
    }

    /// Synchronously flush every live volume's sidecar plus the
    /// cache-wide epoch state. Call this from the app's `on_exit` hook
    /// so a graceful shutdown durably records every chunk written
    /// during the session, even if the per-volume sync watchdog hasn't
    /// fired since the last batch of writes.
    ///
    /// Without this, the registry holds `Weak<Inner>` but workers and
    /// the purge target keep strong refs through process lifetime — so
    /// `DiskStore::Drop` never runs and the last 0–10 s of writes are
    /// lost when the process exits between watchdog ticks.
    ///
    /// Idempotent: re-flushing after no writes is cheap (the snapshot
    /// path early-returns when `pending == 0`). Safe to call from any
    /// thread.
    pub fn shutdown(&self) {
        // Snapshot the volume map under the lock, then drop the lock
        // before flushing — `disk.flush()` can block on fsync and we
        // don't want to hold the registry lock for that long.
        let live: Vec<Arc<Inner>> = {
            let mut volumes = self.volumes.lock().unwrap();
            volumes.retain(|_, w| w.strong_count() > 0);
            volumes.values().filter_map(|w| w.upgrade()).collect()
        };
        let n_vols = live.len();
        for inner in &live {
            inner.flush();
        }
        let epoch_path = epoch::epoch_state_path(&self.unified_root);
        if let Err(e) = self.epoch.save(&epoch_path) {
            log::warn!(
                target: super::purge::LOG_TARGET,
                "shutdown: epoch state save failed at {}: {}",
                epoch_path.display(),
                e
            );
        }
        log::info!(
            target: super::purge::LOG_TARGET,
            "shutdown: flushed {} volume(s) under {}",
            n_vols,
            self.unified_root.display()
        );
    }

    /// Flush every `UnifiedCache` registered in the process. Call this
    /// from the app's `on_exit` when you don't know which cache roots
    /// were touched. Order across roots is arbitrary; per-root work is
    /// synchronous.
    pub fn shutdown_all() {
        let caches: Vec<Arc<UnifiedCache>> = {
            let reg = unified_registry().lock().unwrap();
            reg.values().cloned().collect()
        };
        for c in caches {
            c.shutdown();
        }
    }

    /// Get-or-create the `ChunkCache` for one volume under this
    /// unified root. The second call for the same `volume_id` reuses
    /// the same in-memory `Inner` (and therefore the same workers,
    /// sidecar, and purge target) — duplicate opens during a volume
    /// switch are coalesced rather than racing.
    pub fn open_volume(&self, backfiller: Arc<dyn ChunkBackfiller>) -> ChunkCache {
        let volume_id = backfiller.volume_id();
        let mut volumes = self.volumes.lock().unwrap();
        volumes.retain(|_, w| w.strong_count() > 0);
        if let Some(w) = volumes.get(&volume_id) {
            if let Some(inner) = w.upgrade() {
                log::debug!(
                    target: super::purge::LOG_TARGET,
                    "UnifiedCache reuse: volume={} root={}",
                    volume_id,
                    self.unified_root.display(),
                );
                return ChunkCache { inner };
            }
        }
        let chunks_root = self.unified_root.join(&volume_id);
        let _ = std::fs::create_dir_all(&chunks_root);
        let inner = ChunkCache::build_inner(
            chunks_root,
            backfiller,
            DEFAULT_WORKERS,
            Arc::new(Downloader::new()),
            self.epoch.clone(),
        );
        volumes.insert(volume_id, Arc::downgrade(&inner));
        ChunkCache { inner }
    }
}

impl ChunkCache {
    fn build_inner(
        root: PathBuf,
        backfiller: Arc<dyn ChunkBackfiller>,
        workers: usize,
        downloader: Arc<Downloader>,
        epoch: Arc<EpochState>,
    ) -> Arc<Inner> {
        let task_queue = TaskQueue::new(MAX_AGE);
        let spill_root = root.join("spill");
        let chunks_root = root;
        let _ = std::fs::create_dir_all(&spill_root);

        let volume_id = backfiller.volume_id();
        let extent = backfiller.voxel_extent();
        let max_lod = backfiller.max_lod();

        let disk = DiskStore::new(chunks_root, volume_id, extent, max_lod);
        // Accumulate this volume's residency into the global histogram.
        // The registry already scanned the unified root at first init
        // and seeded every volume it found on disk, so for an existing
        // volume this is a no-op (add_from_sidecar is idempotent on
        // volume_id). For a volume that's brand-new to the cache dir
        // (no prior sidecar), the scan didn't see it and this call
        // does the initial accumulation — which for a fresh volume is
        // zero residency, but still records the volume_id so future
        // calls remain idempotent.
        epoch.add_from_sidecar(&disk.sidecar());

        let inner = Arc::new(Inner {
            map: DashMap::new(),
            dispatching: DashMap::new(),
            disk,
            spill: SpillStore::new(spill_root),
            backfiller,
            sources: Mutex::new(HashMap::new()),
            chunks: DashMap::new(),
            pending_chunk_sources: DashMap::new(),
            task_queue,
            downloader,
            frame: AtomicU64::new(1),
            epoch,
        });

        for i in 0..workers.max(1) {
            let inner = inner.clone();
            std::thread::Builder::new()
                .name(format!("vesuvius-cache-{}", i))
                .spawn(move || worker_loop(inner))
                .expect("spawn cache worker");
        }

        // Register this cache so the epoch watchdog can dispatch
        // purge plans to it. Coerce Arc<Inner> -> Arc<dyn PurgeTarget>
        // first so the Weak we register is type-erased and EpochState
        // doesn't need to know about Inner.
        let target: Arc<dyn PurgeTarget> = inner.clone();
        inner.epoch.register_target(Arc::downgrade(&target));

        inner
    }

    pub fn voxel(&self, x: u32, y: u32, z: u32, lod: u8) -> u8 {
        let key = ChunkKey::new(lod, x / 64, y / 64, z / 64);
        let state = self.state_or_fetch(key);
        if let Some(mmap) = state.as_resident() {
            let off = ((z & 63) as usize) * 64 * 64 + ((y & 63) as usize) * 64 + (x & 63) as usize;
            mmap[off]
        } else {
            0
        }
    }

    /// Return the cached state for `key`, dispatching a fetch if the slot
    /// is Missing/expired. Pending entries get touched on every call so
    /// the next worker pop sees the freshest in-flight chunks first.
    pub fn state_or_fetch(&self, key: ChunkKey) -> Arc<ChunkState> {
        let state = self.inner.state_or_fetch(key);
        if state.as_resident().is_some() {
            self.touch_access(key);
        }
        state
    }

    /// Bump `key`'s access-epoch tag to the current cache epoch. Cheap
    /// no-op when the tag already matches (two atomic loads + a compare,
    /// no histogram update). The per-voxel `get()` reads bypass this
    /// (they peek the mmap directly), but `touch_aabb` walks every
    /// chunk in the rendering region and calls `state_or_fetch` →
    /// `touch_access` before the inner loop runs, so every Resident
    /// chunk the renderer draws gets its access-epoch bumped per paint.
    ///
    /// Race protocol: read `old`, read `current`, attempt CAS(old →
    /// current) on the sidecar byte. Only the winning thread updates the
    /// histogram via `record_access_transition`. Losers either see the
    /// tag already at `current` (someone else won — done) or see it at
    /// some other value (epoch advanced again mid-flight — bail; the
    /// next paint frame will pick it up). Single-shot, no retry loop:
    /// `touch_access` is called per-chunk per-paint, so the retry
    /// budget is effectively the paint loop itself.
    fn touch_access(&self, key: ChunkKey) {
        let current = self.inner.epoch.current();
        let Some(old) = self.inner.disk.get_access_epoch(key) else {
            return;
        };
        if old == current {
            return;
        }
        if let Some(Ok(_)) = self.inner.disk.cas_access_epoch(key, old, current) {
            self.inner.epoch.record_access_transition(old, current);
        }
    }

    /// Evict the oldest chunks until at least `target_chunks` have been
    /// freed (or until no more victims are available). Returns the number
    /// of chunks actually evicted.
    ///
    /// Eviction order per victim, to keep readers safe:
    ///   1. Remove the in-memory `ChunkState::Resident` entry from the
    ///      DashMap, so any future lookup re-takes the slow path.
    ///   2. Demote the per-shard `ChunkStateBits` to Unknown (Release).
    ///   3. Demote the sidecar's persistent state byte to Missing
    ///      (Release; bumps the pending counter so the sync thread
    ///      eventually durably records the eviction).
    ///   4. Punch the chunk's slot in the shard file
    ///      (`fallocate(FALLOC_FL_PUNCH_HOLE)`), freeing physical
    ///      blocks.
    ///   5. Update the global `EpochState` histogram and total_chunks.
    ///
    /// Readers that already passed the bitmap check before step 2 may
    /// transiently see zeros from the punched mmap; the async pipeline
    /// tolerates that, and the next paint loop re-dispatches.
    pub fn purge_to_target(&self, target_chunks: u64) -> u64 {
        let Some(plan) = PurgePlan::build(&self.inner.epoch, target_chunks) else {
            return 0;
        };
        self.inner.run_purge(plan)
    }

    /// Access the cache-wide LRU bookkeeping (shared across volumes
    /// under the same unified root). Useful for stats dashboards and
    /// integration tests; the cache owns the lifetime.
    pub fn epoch_state(&self) -> Arc<EpochState> {
        self.inner.epoch.clone()
    }

    /// Cheap state lookup without dispatching a fetch. Returns `None` if no
    /// entry exists for `key` yet. Useful for LOD-fallback paths that only
    /// want to render whatever is already resident.
    pub fn peek(&self, key: ChunkKey) -> Option<Arc<ChunkState>> {
        self.inner.map.get(&key).map(|e| e.clone())
    }

    /// True iff at least one HTTP GET submitted on behalf of `key` is
    /// currently executing on a downloader worker. Used by the debug overlay
    /// to distinguish "in queue, not yet popped" from "bytes in flight".
    /// Note: a chunk can be Pending without being actively downloading — it
    /// could be waiting in the source queue, post-download in the extract
    /// queue, or fed by a Compute / Chunk source rather than a Download.
    pub fn is_downloading(&self, key: ChunkKey) -> bool {
        self.inner.downloader.is_active_chunk(key)
    }

    pub fn voxel_extent(&self) -> [u32; 3] {
        self.inner.backfiller.voxel_extent()
    }

    pub fn max_lod(&self) -> u8 {
        self.inner.backfiller.max_lod()
    }

    /// Chunks-per-axis of one shard cube. The volume's per-render hot slot
    /// derives shard coordinates from `ChunkKey`s using this value.
    pub fn shard_chunks_per_axis(&self) -> u32 {
        self.inner.disk.shard_chunks_per_axis()
    }

    /// Non-creating peek for a shard's mmap + per-chunk bitmap. Returns
    /// `Some` once at least one chunk in the shard has been materialized
    /// (the shard file is `set_len`'d + mmapped on first write); returns
    /// `None` otherwise. The bitmap lets the volume reader distinguish
    /// "resident" (fast read), "empty" (return 0), and "unknown" (slow path
    /// with LOD climb) without consulting the DashMap per voxel.
    pub fn peek_shard(&self, lod: u8, shard: ShardCoord) -> Option<ShardSnapshot> {
        self.inner.disk.peek_shard(lod, shard)
    }

    /// Open (sparse-mmap + seed bitmap) the shard at `(lod, shard)` if it
    /// isn't already, returning its snapshot. The volume's shard-based slow
    /// path calls this on its first miss for a shard so all subsequent
    /// per-voxel lookups in that shard are bitmap-only.
    pub fn ensure_shard_open(&self, lod: u8, shard: ShardCoord) -> Option<ShardSnapshot> {
        match self.inner.disk.ensure_shard_open(lod, shard) {
            Ok(snap) => snap,
            Err(e) => {
                log::warn!("ensure_shard_open failed for lod {} shard {:?}: {}", lod, shard, e);
                None
            }
        }
    }

    /// Look up the shard layout for `key`. Returns `(shard_coord,
    /// in_shard_chunk_idx)` for in-bounds chunks. Used by the volume's hot
    /// slot to avoid replicating shard-coord math from the disk layer.
    pub fn locate(&self, key: ChunkKey) -> Option<(ShardCoord, u64)> {
        self.inner.disk.locate(key)
    }

    pub fn wait_for(&self, key: ChunkKey, timeout: Duration) -> Arc<ChunkState> {
        let start = std::time::Instant::now();
        loop {
            let state = self.state_or_fetch(key);
            if !matches!(state.as_ref(), ChunkState::Pending { .. }) {
                return state;
            }
            if start.elapsed() >= timeout {
                return state;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Bump the per-cache frame counter that gates touch debouncing.
    /// Call once per render frame, before any per-voxel / per-tile
    /// sampling begins (host wires this into `reset_for_painting`).
    /// Pending chunks observed *after* this returns are eligible for a
    /// fresh queue-priority touch; subsequent observations of the same
    /// chunk within the frame are no-ops on the hot path.
    pub fn advance_frame(&self) {
        self.inner.frame.fetch_add(1, Ordering::Relaxed);
    }

    /// Dispatch every chunk in the AABB `[min, max]` (inclusive, in
    /// target-LOD voxel coords) at `target_lod`, and pre-dispatch a
    /// single coarse parent level so the upscale-from-parent preview
    /// path inside `dispatch_chunk` has a resident ancestor to fall
    /// back on. Used by `ObjVolume::paint` to pre-touch the chunks each
    /// triangle's ray will hit before the per-voxel composite loop runs
    /// — that loop reads the shard mmap unconditionally and relies on
    /// pre-dispatch (or the upscale fill) for the bytes to be there.
    pub fn touch_aabb(&self, min: [f64; 3], max: [f64; 3], target_lod: u8) {
        let max_lod = self.max_lod();
        if target_lod > max_lod {
            return;
        }
        let to_chunk = |v: f64| -> i64 { (v as i64).div_euclid(64) };
        // Clamp the upper corner to the chunk grid at `target_lod` —
        // surface AABBs routinely extend past the scroll, and every
        // out-of-bounds chunk dispatched here would just park a
        // permanent-cooldown entry in the map.
        let extent = self.voxel_extent();
        let chunk_world = 64u64 << target_lod;
        let max_chunk = |e: u32| -> i64 { ((e as u64).div_ceil(chunk_world) as i64 - 1).max(0) };
        let cx0 = to_chunk(min[0]).max(0);
        let cy0 = to_chunk(min[1]).max(0);
        let cz0 = to_chunk(min[2]).max(0);
        let cx1 = to_chunk(max[0]).max(0).min(max_chunk(extent[0]));
        let cy1 = to_chunk(max[1]).max(0).min(max_chunk(extent[1]));
        let cz1 = to_chunk(max[2]).max(0).min(max_chunk(extent[2]));
        if cx1 < cx0 || cy1 < cy0 || cz1 < cz0 {
            return;
        }
        // Pass 1: a SINGLE coarse parent level for the preview. We used
        // to walk every coarser LOD from `target+1` to `max_lod`, but
        // since the preview is now synthesized exactly once from the
        // finest *already-resident* ancestor (no per-frame
        // progressive-resharpen — see `dispatch_chunk` /
        // `try_upscale_from_parent`), fetching the full pyramid only
        // burned download bandwidth on intermediate levels that compete
        // with the target fetch and are never used for sharpening.
        //
        // We keep just the coarsest reachable level (`shift` capped at 6
        // — beyond that a target chunk maps to a sub-voxel region of the
        // parent). Coarsest = fewest chunks (often a single one covering
        // the whole AABB), fastest to land, most likely persisted from a
        // prior session — so it's the most reliable resident ancestor
        // for the one-shot preview. Incidentally-resident finer ancestors
        // (e.g. a level previously browsed as a target) are still picked
        // up by `try_upscale_from_parent`, which always prefers the
        // finest resident ancestor; we just no longer fetch them here.
        let preview_lod = max_lod.min(target_lod.saturating_add(6));
        if preview_lod > target_lod {
            let shift = preview_lod - target_lod;
            let px0 = (cx0 as u32) >> shift;
            let py0 = (cy0 as u32) >> shift;
            let pz0 = (cz0 as u32) >> shift;
            let px1 = (cx1 as u32) >> shift;
            let py1 = (cy1 as u32) >> shift;
            let pz1 = (cz1 as u32) >> shift;
            for pz in pz0..=pz1 {
                for py in py0..=py1 {
                    for px in px0..=px1 {
                        let _ = self.state_or_fetch(ChunkKey::new(preview_lod, px, py, pz));
                    }
                }
            }
        }
        // Pass 2: target chunks. Each first dispatch flips the bitmap
        // to Dispatched and (inside dispatch_chunk) tries upscale fill
        // from any already-Resident parent.
        for cz in cz0..=cz1 {
            for cy in cy0..=cy1 {
                for cx in cx0..=cx1 {
                    let _ = self.state_or_fetch(ChunkKey::new(target_lod, cx as u32, cy as u32, cz as u32));
                }
            }
        }
    }

    /// Synchronously persist the on-disk chunk-state sidecar. Call this
    /// before relying on a fresh `ChunkCache` opened against the same root
    /// to see chunks written by the current process — the background sync
    /// thread otherwise flushes only every ~10 s.
    pub fn flush(&self) {
        self.inner.disk.flush();
    }
}

impl Clone for ChunkCache {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

/// RAII guard that releases a `dispatching` claim no matter how
/// `state_or_fetch` returns — including panics.
struct DispatchGuard {
    inner: Arc<Inner>,
    key: ChunkKey,
}

impl Drop for DispatchGuard {
    fn drop(&mut self) {
        self.inner.dispatching.remove(&self.key);
    }
}

fn long_cooldown() -> Arc<ChunkState> {
    Arc::new(ChunkState::CooldownMiss { until: SystemTime::now() + PERMANENT_COOLDOWN })
}
fn cooldown() -> Arc<ChunkState> {
    Arc::new(ChunkState::CooldownMiss { until: SystemTime::now() + COOLDOWN })
}
fn short_cooldown() -> Arc<ChunkState> {
    Arc::new(ChunkState::CooldownMiss { until: SystemTime::now() + SHORT_COOLDOWN })
}
fn pending_state() -> Arc<ChunkState> {
    Arc::new(ChunkState::pending())
}
impl PurgeTarget for Inner {
    fn volume_id(&self) -> String {
        self.disk.sidecar().header.volume_id.clone()
    }
    fn summarize(&self, plan: PurgePlan, current: u8) -> super::purge::VolumeBreakdown {
        super::purge::VolumeBreakdown::from_sidecar(&self.disk.sidecar(), plan, current)
    }
    fn run_purge(&self, plan: PurgePlan) -> u64 {
        Inner::run_purge(self, plan)
    }
}

impl Inner {
    /// Synchronously flush this volume's sidecar to disk. Wraps
    /// `DiskStore::flush` so `UnifiedCache::shutdown` can drive it
    /// through a `Weak<Inner>` upgrade without exposing the disk store.
    fn flush(&self) {
        self.disk.flush();
    }

    /// Evict all Resident chunks whose access epoch falls into the
    /// `plan.is_victim` set, under per-slot CAS protection.
    ///
    /// Protocol per victim:
    ///   1. CAS sidecar `RESIDENT → LOCKED`. If the CAS fails, the slot
    ///      was concurrently demoted or claimed by another op — skip.
    ///   2. Drop the in-memory `ChunkState::Resident` entry so new
    ///      readers take the slow path.
    ///   3. Demote the per-shard bitmap to `Unknown` (Release).
    ///   4. punch_hole — physical reclaim.
    ///   5. Store `MISSING` (releases the lock; pairs with reader
    ///      Acquire loads).
    ///   6. record_evict.
    ///
    /// The CAS in step (1) serializes against any concurrent
    /// `write_atomic` on the same slot: a peer that started its CAS
    /// from `MISSING` would have failed (we're at `RESIDENT`), and a
    /// peer that wins our `LOCKED` claim will see LOCKED and spin
    /// until we release in step (5). The earlier race where a peer
    /// fetcher's pwrite could land between our `set MISSING` and
    /// `punch_hole` is now structurally impossible.
    fn run_purge(&self, plan: PurgePlan) -> u64 {
        let sidecar = self.disk.sidecar();
        let current = self.epoch.current();
        let volume_id = sidecar.header.volume_id.clone();
        log::info!(
            target: super::purge::LOG_TARGET,
            "cache purge starting: volume={} current_epoch={} age_threshold={} target={} expected_freed={}",
            volume_id,
            current,
            plan.age_threshold,
            plan.target_chunks,
            plan.freed_chunks,
        );
        let mut evicted: u64 = 0;
        let mut skipped: u64 = 0;

        for (lod_idx, dims) in sidecar.header.lods.iter().enumerate() {
            let lod = lod_idx as u8;
            let nx = dims.nx as u64;
            let ny = dims.ny as u64;
            for idx in 0..dims.count() {
                if sidecar.get_state(lod, idx) != super::sidecar::STATE_RESIDENT {
                    continue;
                }
                let ae = sidecar.get_access_epoch(lod, idx);
                if !plan.is_victim(ae, current) {
                    continue;
                }
                // (1) Claim the per-slot lock. CAS may fail if a peer
                // op transitioned the slot between our pre-screen
                // and here — skip and let the next purge cycle pick
                // it up if it ends back at RESIDENT.
                if sidecar
                    .compare_exchange_state(lod, idx, super::sidecar::STATE_RESIDENT, super::sidecar::STATE_LOCKED)
                    .is_err()
                {
                    skipped += 1;
                    continue;
                }

                // Un-flatten linear idx to (x, y, z); raster order is
                // `(z * ny + y) * nx + x`.
                let x = (idx % nx) as u32;
                let y = ((idx / nx) % ny) as u32;
                let z = (idx / (nx * ny)) as u32;
                let key = ChunkKey::new(lod, x, y, z);

                // (2) Drop the in-memory Resident entry. New lookups
                // take the slow path; in-flight Arc<ChunkState>
                // holders may transiently read zeros — that's the
                // documented mmap glitch readers tolerate.
                self.map.remove(&key);

                // (3) Per-shard bitmap demote (Release).
                if let Some((shard, in_shard_idx)) = self.disk.locate(key) {
                    if let Some(snap) = self.disk.peek_shard(lod, shard) {
                        snap.state_bits.store(in_shard_idx, ChunkBitState::Unknown);
                    }
                    let _ = (shard, in_shard_idx);
                }

                // (4) Physical reclaim under the lock — no concurrent
                // pwrite can interleave because peer write_atomic is
                // either spinning on LOCKED or already failed its CAS.
                if let Err(e) = self.disk.punch_hole(key) {
                    log::warn!(target: super::purge::LOG_TARGET, "punch_hole failed for {}: {}", key, e);
                }

                // (5) Release the lock by publishing MISSING.
                sidecar.set_state(lod, idx, super::sidecar::STATE_MISSING);

                // (6) Bookkeeping.
                self.epoch.record_evict(ae);
                evicted += 1;
            }
        }

        log::info!(
            target: super::purge::LOG_TARGET,
            "cache purge finished: volume={} evicted={} skipped_cas={} total_remaining={}",
            volume_id,
            evicted,
            skipped,
            self.epoch.total_chunks(),
        );
        evicted
    }

    /// Same semantics as `ChunkCache::state_or_fetch` but callable from
    /// inside cache internals (notably `register_source`'s Chunk-variant
    /// handler, which dispatches a child chunk synchronously without ever
    /// blocking a worker on its completion).
    fn state_or_fetch(self: &Arc<Self>, key: ChunkKey) -> Arc<ChunkState> {
        if let Some(entry) = self.map.get(&key) {
            let state = entry.clone();
            drop(entry);
            // Pending chunks: touch their entries in both queues so the
            // current frame's chunks bubble back to the LIFO head and
            // re-arm against MAX_AGE. Out-of-frame chunks stop getting
            // touched and age out. The check is debounced per chunk —
            // surface rendering re-enters here per voxel, and the queue
            // mutexes inside the touch calls would otherwise serialize
            // every CPU thread on the same futex.
            if let ChunkState::Pending { last_touched_frame, .. } = state.as_ref() {
                if self.claim_touch(last_touched_frame) {
                    self.task_queue.touch(key);
                    self.downloader.touch(key);
                    // NB: the upscale-from-parent preview is synthesized
                    // exactly once, at dispatch (see `dispatch_chunk`). We
                    // deliberately do NOT re-run it here per frame to
                    // "improve" the preview from a finer ancestor: the
                    // composite reads the target-LOD shard mmap directly,
                    // so once the real bytes land `write_atomic` overwrites
                    // the preview in place and the next paint picks them up
                    // with no paint-path work. Re-synthesizing every frame
                    // was a 262k-voxel trilinear pass + 256 KB alloc per
                    // visible Pending chunk per touching tile — pure churn.
                }
            }
            if let ChunkState::CooldownMiss { until } = state.as_ref() {
                if SystemTime::now() < *until {
                    return state;
                }
            } else {
                return state;
            }
        }

        let claimed = self.dispatching.insert(key, ()).is_none();
        if !claimed {
            return self
                .map
                .get(&key)
                .map(|e| e.clone())
                .unwrap_or_else(pending_state);
        }

        let _guard = DispatchGuard { inner: self.clone(), key };
        let state = self.dispatch_chunk(key);
        self.map.insert(key, state.clone());
        self.publish_terminal(key, &state);
        state
    }

    /// Per-Pending touch debouncer. Returns true at most once per
    /// `advance_frame` tick per chunk across all calling threads: the
    /// CAS wins exclusive permission to bump the queues, everybody else
    /// short-circuits and avoids the global mutex inside the touch.
    /// Two relaxed atomic loads on the hot path — no clock read.
    fn claim_touch(&self, last_touched_frame: &AtomicU64) -> bool {
        let now = self.frame.load(Ordering::Relaxed);
        let prev = last_touched_frame.load(Ordering::Relaxed);
        if prev == now {
            return false;
        }
        last_touched_frame
            .compare_exchange(prev, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    }

    /// Called right after a chunk is written into `self.map`. If the new
    /// state is terminal (`Resident` or `Empty`), drain
    /// `pending_chunk_sources[key]` and complete each waiting source:
    ///   - `Resident` → payload is the chunk's `Arc<ChunkState>`.
    ///   - `Empty`    → `Ok(None)` (parent sees the child as absent).
    /// Source completion is idempotent (`complete_source` no-ops on a Done
    /// source), so double-fires from racing publishers are safe.
    fn publish_terminal(self: &Arc<Self>, key: ChunkKey, state: &Arc<ChunkState>) {
        let outcome: SourceOutcome = match state.as_ref() {
            ChunkState::Resident { .. } => Ok(Some(state.clone() as SourcePayload)),
            ChunkState::Empty => Ok(None),
            _ => return,
        };
        let waiters: Vec<String> = match self.pending_chunk_sources.remove(&key) {
            Some((_, v)) => v,
            None => return,
        };
        for source_key in waiters {
            self.complete_source(source_key, outcome.clone());
        }
    }

    /// Drive one chunk from Missing to Pending. Plans + registers sources;
    /// queues either FetchSource / a download or (if all sources already
    /// Done) Extract.
    ///
    /// Returns the state to insert in `self.map`. Caller is the
    /// `or_insert_with` closure on the entry guard — we must NEVER call
    /// `self.map.insert(key, …)` for this same key from here.
    fn dispatch_chunk(self: &Arc<Self>, key: ChunkKey) -> Arc<ChunkState> {
        match self.disk.load(key) {
            LoadOutcome::Resident { mmap, offset } => {
                log::trace!("[{}] disk hit", key);
                return Arc::new(ChunkState::Resident { mmap, offset });
            }
            LoadOutcome::Empty => {
                log::trace!("[{}] disk hit (empty)", key);
                return Arc::new(ChunkState::Empty);
            }
            LoadOutcome::Missing => {}
        }
        if self.is_out_of_bounds(key) {
            log::trace!("[{}] out of bounds", key);
            return long_cooldown();
        }
        // Claim the chunk's shard bitmap cell as `Dispatched`. This is what
        // lets the volume's per-voxel slow path tell "fetch in flight" from
        // "never tried" without ever consulting the DashMap once we've
        // primed it here — the bitmap is the single source of truth for
        // per-chunk presence going forward.
        // `mark_dispatched` flips the bitmap cell `Unknown→Dispatched` and
        // returns `true` only on that first transition. The in-memory shard
        // bitmap outlives this chunk's DashMap entry: purge demotes a
        // *Resident* cell back to `Unknown` when it reclaims the bytes, but a
        // preview-only cell (sidecar still MISSING, never Resident) keeps its
        // `Dispatched` bit. So `first_dispatch` means precisely "no preview
        // has been synthesized for this slot yet this session (or the real
        // data it held was purged)" — which is exactly when we want to fill
        // the preview.
        let first_dispatch = match self.disk.mark_dispatched(key) {
            Ok(transitioned) => transitioned,
            Err(e) => {
                log::trace!("[{}] mark_dispatched failed: {}", key, e);
                false
            }
        };

        // Upscale-from-parent preview. Walks the LOD pyramid from
        // `key.lod + 1` toward `max_lod`, and at the first resident
        // ancestor synthesizes a `2^shift`× upsampled fill for the
        // target slot's shard mmap so the composite reads a sensible
        // preview while the real bytes stream in. Bytes get overwritten
        // by `write_atomic` when the real fetch lands. Serialized via the
        // outer `dispatching` claim — only the thread that won the claim
        // is here, so we can't tear with another upscale on the same key
        // (and the real download is queued *after* this point, ordering
        // pwrites by submission).
        //
        // Gated on `first_dispatch` so the preview is synthesized at most
        // once per slot per session — never re-run per frame (Step 1) nor
        // on re-dispatch after DashMap eviction. Overlapping/adjacent tiles
        // that re-enter dispatch for the same chunk reuse the existing
        // preview bytes instead of re-running the 262k-voxel trilinear pass.
        if first_dispatch {
            self.try_upscale_from_parent(key);
        }

        let plan = match self.backfiller.plan(key) {
            Ok(p) => p,
            Err(BackfillError::OutOfBounds) => return long_cooldown(),
            Err(BackfillError::Permanent(reason)) => {
                log::warn!("[{}] permanent (plan): {}", key, reason);
                return long_cooldown();
            }
            Err(BackfillError::Transient(reason)) => {
                log::debug!("[{}] transient (plan): {}", key, reason);
                return cooldown();
            }
        };

        let BackfillPlan { covered, sources, extract } = plan;
        let order: Vec<String> = sources.iter().map(|s| s.key().to_string()).collect();
        log::debug!(
            "[{}] miss → fetching {} source(s), covers {} chunk(s)",
            key,
            order.len(),
            covered.len()
        );
        // Pre-claim sibling cache chunks as Pending. Subsequent
        // `state_or_fetch` calls for any of them return Pending immediately
        // instead of running another plan() + extract that would redo the
        // same decode the primary extract is about to do.
        //
        // Only insert into the map if the slot is genuinely free (no entry,
        // or a stale CooldownMiss that we'd retry anyway). If a sibling is
        // already Resident / Empty / Pending we leave it alone. The
        // `dispatching` DashMap is the per-key claim guard, but here we're
        // claiming *other* keys' map slots — so we use map.entry with a
        // CAS-style check on what's already there.
        for sib in &covered {
            if *sib == key {
                continue;
            }
            let mut entry = self.map.entry(*sib).or_insert_with(pending_state);
            if let ChunkState::CooldownMiss { until } = entry.as_ref() {
                if SystemTime::now() >= *until {
                    *entry = pending_state();
                }
            }
        }
        let progress_arc = Arc::new(Mutex::new(ChunkProgress {
            order: order.clone(),
            remaining: order.len(),
            results: HashMap::new(),
            extract: Some(extract),
            covered,
        }));
        self.chunks.insert(key, progress_arc.clone());

        if order.is_empty() {
            // 0-source plan: queue Extract immediately.
            return match self.task_queue.try_submit(key, Task::Extract) {
                Ok(()) => pending_state(),
                Err(dropped) => {
                    log::debug!("[{}] dropped: extract queue full (dispatch)", key);
                    self.chunks.remove(&key);
                    drop(dropped);
                    short_cooldown()
                }
            };
        }

        // Register sources. Track immediate Dones so we can satisfy them
        // after the loop (avoids deep recursion through self.satisfy during
        // dispatch, and lets us bail on queue-full cleanly).
        let mut immediates: Vec<(String, SourceOutcome)> = Vec::new();
        for spec in sources {
            let source_key = spec.key().to_string();
            match self.register_source(key, spec) {
                RegisterResult::Queued | RegisterResult::AttachedPending => {}
                RegisterResult::AlreadyDone(outcome) => immediates.push((source_key, outcome)),
                RegisterResult::QueueFull => {
                    log::debug!("[{}] dropped: source queue full (dispatch)", key);
                    self.chunks.remove(&key);
                    return short_cooldown();
                }
            }
        }

        // Apply immediates. If they push the chunk to Extract-ready, queue
        // Extract here.
        for (sk, outcome) in immediates {
            match self.satisfy(key, &sk, outcome) {
                SatisfyResult::Ok => {}
                SatisfyResult::QueueFullOnExtract => {
                    log::debug!("[{}] dropped: extract queue full (immediate)", key);
                    return short_cooldown();
                }
            }
        }
        pending_state()
    }

    fn register_source(self: &Arc<Self>, chunk_key: ChunkKey, spec: SourceSpec) -> RegisterResult {
        let source_key = spec.key().to_string();

        // Phase 1: under self.sources lock, either attach as a waiter on
        // an existing source or install a fresh `Pending` placeholder for
        // this source key.
        //
        // We MUST NOT call `downloader.try_submit` or `task_queue.try_submit`
        // while holding this lock. Both can fire callbacks synchronously
        // (downloader on queue-full / eviction). Those callbacks re-enter
        // `complete_source`, which locks `self.sources` again on the same
        // thread — a self-deadlock that wedged the cache after a few frames.
        enum Slot {
            Existing(Arc<Mutex<SourceState>>),
            Fresh,
        }
        let slot = {
            let mut sources = self.sources.lock().unwrap();
            if let Some(arc) = sources.get(&source_key) {
                Slot::Existing(arc.clone())
            } else {
                let arc = Arc::new(Mutex::new(SourceState::Pending {
                    waiters: vec![chunk_key],
                }));
                sources.insert(source_key.clone(), arc);
                Slot::Fresh
            }
        };

        if let Slot::Existing(arc) = slot {
            drop(spec); // first observer's fetch/url is authoritative
            let mut s = arc.lock().unwrap();
            return match &mut *s {
                SourceState::Pending { waiters } => {
                    waiters.push(chunk_key);
                    log::trace!("[{}] attach (chunk {})", source_key, chunk_key);
                    RegisterResult::AttachedPending
                }
                SourceState::Done {
                    outcome,
                    remaining_consumers,
                } => {
                    // New consumer for an already-completed source — bump
                    // the refcount so `extract_chunk`'s eviction logic
                    // waits for this chunk too.
                    *remaining_consumers += 1;
                    log::trace!(
                        "[{}] reuse (chunk {}, refcount {})",
                        source_key,
                        chunk_key,
                        remaining_consumers
                    );
                    RegisterResult::AlreadyDone(outcome.clone())
                }
            };
        }

        // Phase 2: we're the first observer and we own the placeholder slot.
        // Dispatch the fetch with no locks held.
        match spec {
            SourceSpec::Compute { key: _, fetch } => {
                match self.task_queue.try_submit(
                    chunk_key,
                    Task::FetchSource {
                        key: source_key.clone(),
                        fetch,
                    },
                ) {
                    Ok(()) => {
                        log::trace!("[{}] queued (chunk {})", source_key, chunk_key);
                        RegisterResult::Queued
                    }
                    Err(_dropped) => {
                        // Roll the placeholder forward to Done(Err) so any
                        // concurrent waiters (the only chunk that could have
                        // attached after we released the lock) get notified.
                        log::trace!("[{}] dropped: cache queue full", source_key);
                        self.complete_source(
                            source_key,
                            Err(BackfillError::Transient("cache queue full".into())),
                        );
                        RegisterResult::QueueFull
                    }
                }
            }
            SourceSpec::Chunk { key: _, chunk_key: child_key } => {
                // Register our interest BEFORE dispatching the child. If
                // dispatch happens to disk-hit and synchronously transition
                // the child to Resident, the publish_resident call inside
                // state_or_fetch drains our entry and completes the source
                // right then. complete_source is idempotent, so the
                // post-dispatch check below double-firing is harmless —
                // the second call no-ops on the already-Done source.
                self.pending_chunk_sources
                    .entry(child_key)
                    .or_insert_with(Vec::new)
                    .push(source_key.clone());

                let child_state = self.state_or_fetch(child_key);
                match child_state.as_ref() {
                    ChunkState::Resident { .. } => {
                        // publish_terminal either fired during the dispatch
                        // call above or is about to via the disk path; in
                        // both cases the source is (or will be) Done.
                        log::trace!(
                            "[{}] chunk dep on {} resident (chunk {})",
                            source_key,
                            child_key,
                            chunk_key
                        );
                        RegisterResult::Queued
                    }
                    ChunkState::Empty | ChunkState::CooldownMiss { .. } => {
                        // Child is definitively absent or won't load — resolve
                        // our source as absent. (Empty also gets handled by
                        // publish_terminal, but we may have raced ahead of
                        // that; complete_source is idempotent.)
                        log::trace!(
                            "[{}] chunk dep on {} unresolvable (chunk {})",
                            source_key,
                            child_key,
                            chunk_key
                        );
                        self.complete_source(source_key.clone(), Ok(None));
                        RegisterResult::Queued
                    }
                    _ => {
                        // Pending / Missing — publish_terminal will satisfy
                        // us when the child reaches a terminal state.
                        log::trace!(
                            "[{}] chunk dep on {} pending (chunk {})",
                            source_key,
                            child_key,
                            chunk_key
                        );
                        RegisterResult::Queued
                    }
                }
            }
            SourceSpec::Download { key: _, url, range } => {
                let inner = self.clone();
                let key_for_done = source_key.clone();
                let on_done: OnDone = Box::new(move |result: DownloadResult| {
                    // Spill the bytes to disk as soon as they arrive so the
                    // payload is mmap-backed by the time Extract picks it
                    // up. This keeps the heap bounded even when many
                    // downloads complete faster than the (limited) cache
                    // workers can extract them.
                    let outcome: SourceOutcome = match result {
                        Ok(Some(bytes)) => match inner.spill.write_and_mmap(&bytes) {
                            Ok(mmap) => Ok(Some(Arc::new(mmap) as SourcePayload)),
                            Err(e) => {
                                log::warn!("[{}] spill failed ({}); falling back to in-memory", key_for_done, e);
                                Ok(Some(Arc::new(bytes) as SourcePayload))
                            }
                        },
                        Ok(None) => Ok(None),
                        Err(DownloadError::Transient(s)) => {
                            Err(BackfillError::Transient(format!("download: {}", s)))
                        }
                    };
                    inner.complete_source(key_for_done, outcome);
                });

                // Submit without holding self.sources. The downloader may
                // synchronously fire `on_done` (queue full / eviction), which
                // calls complete_source — that path now safely re-locks
                // self.sources.
                match self.downloader.try_submit(&url, range, chunk_key, on_done) {
                    SubmitResult::Submitted => {
                        log::trace!("[{}] submitted (chunk {})", source_key, chunk_key);
                        RegisterResult::Queued
                    }
                    SubmitResult::QueueFull => {
                        // The downloader already invoked our on_done with
                        // Err synchronously — complete_source has run and
                        // cleared the placeholder. Nothing more to do.
                        log::trace!("[{}] dropped: downloader queue full", source_key);
                        RegisterResult::QueueFull
                    }
                }
            }
        }
    }

    fn fetch_source(
        self: &Arc<Self>,
        source_key: String,
        fetch: Box<dyn FnOnce() -> SourceOutcome + Send + 'static>,
    ) {
        let outcome = fetch();
        self.complete_source(source_key, outcome);
    }

    /// Mark a source as `Done(outcome)` and notify every chunk currently
    /// waiting on it. Called from both the synchronous `FetchSource` worker
    /// path and the download callback path.
    fn complete_source(self: &Arc<Self>, source_key: String, outcome: SourceOutcome) {
        let arc = {
            let sources = self.sources.lock().unwrap();
            match sources.get(&source_key) {
                Some(a) => a.clone(),
                None => {
                    log::trace!("[{}] evicted before completion", source_key);
                    return;
                }
            }
        };

        let waiters = {
            let mut s = arc.lock().unwrap();
            let n = match &*s {
                SourceState::Pending { waiters } => waiters.len(),
                _ => 0,
            };
            let prev = std::mem::replace(
                &mut *s,
                SourceState::Done {
                    outcome: outcome.clone(),
                    remaining_consumers: n,
                },
            );
            match prev {
                SourceState::Pending { waiters } => waiters,
                SourceState::Done { .. } => return,
            }
        };

        let outcome_label = match &outcome {
            Ok(Some(_)) => "ok",
            Ok(None) => "absent",
            Err(e) => match e {
                BackfillError::Transient(_) => "transient",
                BackfillError::Permanent(_) => "permanent",
                BackfillError::OutOfBounds => "oob",
            },
        };
        log::trace!(
            "[{}] resolved {} → {} waiter(s)",
            source_key,
            outcome_label,
            waiters.len()
        );

        // Evict errored entries so future plans retry instead of permanently
        // inheriting the failure.
        if outcome.is_err() {
            self.sources.lock().unwrap().remove(&source_key);
        }

        for w in waiters {
            if matches!(
                self.satisfy(w, &source_key, outcome.clone()),
                SatisfyResult::QueueFullOnExtract
            ) {
                self.map.insert(w, short_cooldown());
            }
        }
    }

    /// Apply one source's outcome to the chunk's progress. Returns
    /// `QueueFullOnExtract` only when all sources are now resolved but
    /// the resulting Extract task couldn't be enqueued; caller must then
    /// move the chunk to a short cooldown.
    fn satisfy(self: &Arc<Self>, chunk_key: ChunkKey, source_key: &str, outcome: SourceOutcome) -> SatisfyResult {
        let arc = match self.chunks.get(&chunk_key).map(|e| e.clone()) {
            Some(a) => a,
            None => return SatisfyResult::Ok,
        };
        let queue_extract = {
            let mut p = arc.lock().unwrap();
            if p.results.contains_key(source_key) {
                false
            } else {
                p.results.insert(source_key.to_string(), outcome);
                p.remaining = p.remaining.saturating_sub(1);
                p.remaining == 0
            }
        };
        if queue_extract {
            match self.task_queue.try_submit(chunk_key, Task::Extract) {
                Ok(()) => SatisfyResult::Ok,
                Err(_dropped) => {
                    log::debug!("[{}] dropped: extract queue full", chunk_key);
                    self.chunks.remove(&chunk_key);
                    SatisfyResult::QueueFullOnExtract
                }
            }
        } else {
            SatisfyResult::Ok
        }
    }

    fn extract_chunk(self: &Arc<Self>, key: ChunkKey) {
        let t0 = std::time::Instant::now();
        log::trace!("[{}] extract start", key);
        let (_, arc) = match self.chunks.remove(&key) {
            Some(v) => v,
            None => return,
        };
        let (order, results, extract, covered) = {
            let mut p = arc.lock().unwrap();
            let extract = match p.extract.take() {
                Some(e) => e,
                None => return,
            };
            let order = std::mem::take(&mut p.order);
            let results = std::mem::take(&mut p.results);
            let covered = std::mem::take(&mut p.covered);
            (order, results, extract, covered)
        };
        let mut inputs: Vec<SourceOutcome> = Vec::with_capacity(order.len());
        let mut results = results;
        for k in &order {
            inputs.push(results.remove(k).unwrap_or_else(|| Ok(None)));
        }

        let mut primary_state: Option<Arc<ChunkState>> = None;
        let mut sibling_states: Vec<(ChunkKey, Arc<ChunkState>)> = Vec::new();
        let mut failure_state: Option<Arc<ChunkState>> = None;
        match extract(&inputs) {
            Ok(fills) => {
                // Each fill is materialized to its terminal state (disk
                // write + mmap → Resident, or .empty sentinel → Empty).
                // Promoting *every* fill — not just the primary — means
                // sibling chunks claimed Pending in dispatch_chunk
                // transition directly to Resident here without needing a
                // follow-up disk-load round trip.
                let mut seen_primary = false;
                for (k, data) in fills {
                    let state = if k == key {
                        let s = self.primary_state(k, data, t0);
                        seen_primary = true;
                        primary_state = Some(s.clone());
                        s
                    } else {
                        self.primary_state(k, data, t0)
                    };
                    if k != key {
                        sibling_states.push((k, state));
                    }
                }
                if !seen_primary {
                    log::warn!("[{}] extract produced no entry for the dispatched key", key);
                    failure_state = Some(cooldown());
                }
            }
            Err(BackfillError::OutOfBounds) => failure_state = Some(long_cooldown()),
            Err(BackfillError::Permanent(reason)) => {
                log::warn!("[{}] permanent: {}", key, reason);
                failure_state = Some(long_cooldown());
            }
            Err(BackfillError::Transient(reason)) => {
                // Aged-out / cancelled fetches aren't a chunk failure — they
                // just mean the viewport moved on before the source landed.
                // Surface them as cancellations so they don't look like errors.
                if reason.contains("aged out") || reason.contains("queue closed") {
                    log::trace!("[{}] cancelled: {}", key, reason);
                } else {
                    log::debug!("[{}] transient: {}", key, reason);
                }
                failure_state = Some(cooldown());
            }
        }
        // Drop our inputs so the per-source payloads (mmaps in the
        // download path) only stay alive while the refcount-eviction loop
        // can see them — Arc clones inside the source entries remain.
        drop(inputs);
        self.release_sources(&order);

        let new_state = primary_state.unwrap_or_else(|| failure_state.clone().unwrap_or_else(cooldown));
        self.map.insert(key, new_state.clone());
        self.publish_terminal(key, &new_state);

        // Promote siblings to their terminal states. The set of keys we
        // touched is `{primary} ∪ {fills}`; any covered key not in that
        // set didn't get a fill — clear its Pending claim with the same
        // failure state we'd use for the primary in that case (so a
        // future dispatch can retry it without confusion).
        let mut touched: std::collections::HashSet<ChunkKey> = std::collections::HashSet::new();
        touched.insert(key);
        for (k, state) in &sibling_states {
            touched.insert(*k);
            self.map.insert(*k, state.clone());
            self.publish_terminal(*k, state);
        }
        for c in &covered {
            if touched.contains(c) {
                continue;
            }
            // A covered slot that the extract didn't fill — leave it as
            // a short cooldown so the next dispatch (post-cooldown) will
            // re-plan instead of being stuck on a stale Pending. On
            // extract failure we use the same fallback state as the
            // primary; on success this is an unexpected gap we surface
            // with a short cooldown.
            let s = failure_state.clone().unwrap_or_else(short_cooldown);
            self.map.insert(*c, s);
        }
    }

    /// Translate a primary `ExtractedChunk` into the in-memory state to
    /// publish. `Bytes` → write to disk, mmap, `Resident`. `Empty` → persist
    /// the `.empty` sentinel, `Empty`. IO failures fall back to a cooldown
    /// so paint retries.
    fn primary_state(&self, key: ChunkKey, primary: ExtractedChunk, t0: std::time::Instant) -> Arc<ChunkState> {
        use super::disk::WriteOutcome;
        match primary {
            ExtractedChunk::Bytes(bytes) => match self.disk.write_atomic(key, &bytes) {
                Ok(outcome) => {
                    // Only the thread that actually performed the
                    // MISSING → RESIDENT transition should record_fill —
                    // sibling writes that lose the CAS race (slot was
                    // already filled by a peer) must skip, otherwise
                    // the histogram and `total_chunks` over-count by
                    // one per redundant sibling.
                    if matches!(outcome, WriteOutcome::Wrote) {
                        let ep = self.epoch.record_fill(CHUNK_VOXELS as u64);
                        self.disk.set_access_epoch(key, ep);
                    }
                    match outcome {
                        WriteOutcome::Wrote | WriteOutcome::AlreadyResident => match self.disk.try_load(key) {
                            Some((mmap, offset)) => {
                                log::debug!("[{}] ready ({:?})", key, t0.elapsed());
                                Arc::new(ChunkState::Resident { mmap, offset })
                            }
                            None => {
                                log::warn!("[{}] write ok but mmap reload failed", key);
                                cooldown()
                            }
                        },
                        WriteOutcome::AlreadyEmpty => {
                            log::debug!("[{}] empty (peer-marked, {:?})", key, t0.elapsed());
                            Arc::new(ChunkState::Empty)
                        }
                    }
                }
                Err(e) => {
                    log::warn!("[{}] disk write failed: {}", key, e);
                    cooldown()
                }
            },
            ExtractedChunk::Empty => {
                if let Err(e) = self.disk.mark_empty(key) {
                    log::warn!("[{}] mark_empty failed ({}); empty still cached in-memory only", key, e);
                }
                log::debug!("[{}] empty ({:?})", key, t0.elapsed());
                Arc::new(ChunkState::Empty)
            }
        }
    }

    /// Decrement consumer refcounts for each source this chunk used. When a
    /// count hits zero we evict the source entry from `self.sources`; that
    /// drops its `Arc<Mmap>`, the kernel reclaims the spill file's pages,
    /// and (because the spill file was already unlinked) the disk space is
    /// freed too.
    fn release_sources(&self, order: &[String]) {
        if order.is_empty() {
            return;
        }
        let mut sources = self.sources.lock().unwrap();
        for source_key in order {
            let should_evict = if let Some(arc) = sources.get(source_key) {
                let mut s = arc.lock().unwrap();
                if let SourceState::Done { remaining_consumers, .. } = &mut *s {
                    *remaining_consumers = remaining_consumers.saturating_sub(1);
                    *remaining_consumers == 0
                } else {
                    false
                }
            } else {
                false
            };
            if should_evict {
                log::trace!("[{}] evict (last consumer)", source_key);
                sources.remove(source_key);
            }
        }
    }

    /// Handle a `TaskEntry` that the queue dropped (age / shutdown). Acts
    /// as if the task ran and failed transiently:
    /// `FetchSource` resolves with a Transient error so waiters back off;
    /// `Extract` just cleans up progress and reverts the chunk to cooldown.
    fn cancel_dropped_task(self: &Arc<Self>, entry: TaskEntry, reason: &str) {
        match entry.task {
            Task::FetchSource { key, fetch: _ } => {
                log::trace!("[{}] cancel: {}", key, reason);
                self.complete_source(key, Err(BackfillError::Transient(reason.into())));
            }
            Task::Extract => {
                let chunk_key = entry.chunk;
                log::debug!("[{}] dropped: {}", chunk_key, reason);
                self.chunks.remove(&chunk_key);
                self.map.insert(chunk_key, short_cooldown());
            }
        }
    }

    fn is_chunk_done(&self, key: ChunkKey) -> bool {
        self.map
            .get(&key)
            .map(|s| s.as_ref().is_terminal())
            .unwrap_or(false)
    }

    fn is_source_done(&self, source_key: &str) -> bool {
        let sources = self.sources.lock().unwrap();
        match sources.get(source_key) {
            Some(arc) => matches!(*arc.lock().unwrap(), SourceState::Done { .. }),
            None => false,
        }
    }

    /// Walk the LOD pyramid from `key.lod + 1` toward `max_lod` and, at
    /// the first resident ancestor, synthesize a `2^shift`× trilinear
    /// upsampled fill and pwrite it into the target shard's mmap region
    /// so the composite reads a sensible preview while the real bytes
    /// stream in. The eventual `write_atomic` overwrites these bytes.
    ///
    /// Called exactly once per slot per session, from `dispatch_chunk`
    /// gated on the bitmap's first `Unknown→Dispatched` transition — the
    /// preview is never re-synthesized per frame or on re-dispatch (see
    /// the call site). The walk takes the *finest* resident ancestor
    /// (shifts ascend, so the first hit is finest) and stops there.
    fn try_upscale_from_parent(&self, key: ChunkKey) {
        let max_lod = self.backfiller.max_lod();
        if key.lod >= max_lod {
            return;
        }
        // `shift > 6` would map a target chunk to a sub-voxel position
        // in the parent which we can't address, so cap the walk there
        // even if `max_lod - key.lod` would otherwise extend further.
        let max_shift: u8 = (max_lod - key.lod).min(6);
        for shift in 1..=max_shift {
            let parent_lod = key.lod + shift;
            let parent_key = ChunkKey::new(parent_lod, key.x >> shift, key.y >> shift, key.z >> shift);
            let (mmap, offset) = match self.disk.load(parent_key) {
                LoadOutcome::Resident { mmap, offset } => (mmap, offset),
                _ => continue,
            };
            let parent_slice = &mmap[offset..offset + CHUNK_VOXELS];
            let bytes = upsample_from_parent(parent_slice, key.x, key.y, key.z, shift);
            match self.disk.write_unconfirmed(key, &bytes) {
                Ok(true) => {
                    // The source parent is now serving as a live preview
                    // backing for the composite — LRU-bump it (CAS +
                    // histogram) so the purge sweep keeps it warm. The
                    // target chunk's sidecar slot is still MISSING
                    // (write_unconfirmed reset it), so its access_epoch
                    // isn't visible to the purger, but stamp the byte
                    // anyway so any later code that inspects it sees this
                    // frame's epoch.
                    self.touch_access_inline(parent_key);
                    self.disk.set_access_epoch(key, self.epoch.current());
                }
                Ok(false) => {
                    // CAS lost — target was already non-MISSING (real
                    // fetch beat us, or a peer raced us through). Still
                    // bump the parent we read.
                    self.touch_access_inline(parent_key);
                }
                Err(e) => {
                    log::debug!("[{}] upscale-from-parent write failed: {}", key, e);
                }
            }
            return;
        }
    }

    /// Bump `key`'s access-epoch tag to the current epoch with histogram
    /// bookkeeping. Inline copy of `ChunkCache::touch_access`'s body for
    /// use from `Inner` paths that can't go through `ChunkCache`.
    fn touch_access_inline(&self, key: ChunkKey) {
        let current = self.epoch.current();
        let Some(old) = self.disk.get_access_epoch(key) else {
            return;
        };
        if old == current {
            return;
        }
        if let Some(Ok(_)) = self.disk.cas_access_epoch(key, old, current) {
            self.epoch.record_access_transition(old, current);
        }
    }

    fn is_out_of_bounds(&self, key: ChunkKey) -> bool {
        let extent = self.backfiller.voxel_extent();
        let scale = 1u64 << key.lod;
        let chunk_voxels = 64u64 * scale;
        let start = [
            key.x as u64 * chunk_voxels,
            key.y as u64 * chunk_voxels,
            key.z as u64 * chunk_voxels,
        ];
        start[0] >= extent[0] as u64
            || start[1] >= extent[1] as u64
            || start[2] >= extent[2] as u64
            || key.lod > self.backfiller.max_lod()
    }
}

impl TaskQueue {
    fn new(max_age: Duration) -> Self {
        Self {
            inner: Mutex::new(TaskQueueInner {
                entries: BTreeMap::new(),
                chunk_index: HashMap::new(),
                next_seq: 0,
                closed: false,
            }),
            not_empty: Condvar::new(),
            max_age,
        }
    }

    /// Submit a task. Only fails if the queue is closed (shutdown). The
    /// queue is otherwise unbounded; cache-layer dedup ensures we don't
    /// queue duplicate Fetch/Extract tasks for the same key.
    fn try_submit(&self, chunk: ChunkKey, task: Task) -> Result<(), Task> {
        let mut q = self.inner.lock().unwrap();
        if q.closed {
            return Err(task);
        }
        q.next_seq += 1;
        let key = rev_seq(q.next_seq);
        q.entries.insert(
            key,
            TaskEntry {
                chunk,
                added_at: Instant::now(),
                task,
            },
        );
        q.chunk_index.entry(chunk).or_default().push(key);
        self.not_empty.notify_one();
        Ok(())
    }

    /// Refresh every queued entry for `chunk`: bump its seq (moving it
    /// to the head of the LIFO order) and reset `added_at` so MAX_AGE
    /// re-counts from now. No-op when the chunk has no queued entries.
    fn touch(&self, chunk: ChunkKey) {
        let mut q = self.inner.lock().unwrap();
        let old_keys = match q.chunk_index.remove(&chunk) {
            Some(v) if !v.is_empty() => v,
            _ => return,
        };
        let mut new_keys = Vec::with_capacity(old_keys.len());
        let now = Instant::now();
        for old_key in old_keys {
            let Some(mut entry) = q.entries.remove(&old_key) else {
                continue;
            };
            entry.added_at = now;
            q.next_seq += 1;
            let new_key = rev_seq(q.next_seq);
            q.entries.insert(new_key, entry);
            new_keys.push(new_key);
        }
        if !new_keys.is_empty() {
            q.chunk_index.insert(chunk, new_keys);
            self.not_empty.notify_one();
        }
    }

    /// Block until either a non-stale entry is available (returned) or the
    /// queue is closed. Entries dropped along the way (older than
    /// `max_age`) are surfaced as `dropped` so the caller can run their
    /// cancellation paths.
    fn pop(&self) -> PopResult {
        let mut q = self.inner.lock().unwrap();
        let mut dropped: Vec<TaskEntry> = Vec::new();
        loop {
            if q.closed && q.entries.is_empty() {
                return PopResult::Closed { dropped };
            }
            let Some((key, entry)) = q.entries.pop_first() else {
                q = self.not_empty.wait(q).unwrap();
                continue;
            };
            forget_chunk_key(&mut q.chunk_index, entry.chunk, key);
            if entry.added_at.elapsed() > self.max_age {
                dropped.push(entry);
                continue;
            }
            return PopResult::Ready { entry, dropped };
        }
    }
}

fn forget_chunk_key(index: &mut HashMap<ChunkKey, Vec<u64>>, chunk: ChunkKey, key: u64) {
    if let Some(keys) = index.get_mut(&chunk) {
        keys.retain(|k| *k != key);
        if keys.is_empty() {
            index.remove(&chunk);
        }
    }
}

enum PopResult {
    Ready {
        entry: TaskEntry,
        dropped: Vec<TaskEntry>,
    },
    Closed {
        dropped: Vec<TaskEntry>,
    },
}

fn worker_loop(inner: Arc<Inner>) {
    loop {
        match inner.task_queue.pop() {
            PopResult::Ready { entry, dropped } => {
                for d in dropped {
                    inner.cancel_dropped_task(d, "stale on pop");
                }
                // Skip-met: cooldown-retry races or duplicate enqueues can
                // leave stale work in the queue. Drop it instead of doing
                // redundant disk + decode work.
                let chunk = entry.chunk;
                match &entry.task {
                    Task::Extract => {
                        if inner.is_chunk_done(chunk) {
                            log::trace!("[{}] skip extract (already terminal)", chunk);
                            inner.chunks.remove(&chunk);
                            continue;
                        }
                    }
                    Task::FetchSource { key, .. } => {
                        if inner.is_source_done(key) {
                            log::trace!("[{}] skip fetch (already done)", key);
                            continue;
                        }
                    }
                }
                match entry.task {
                    Task::FetchSource { key, fetch } => inner.fetch_source(key, fetch),
                    Task::Extract => inner.extract_chunk(chunk),
                }
            }
            PopResult::Closed { dropped } => {
                for d in dropped {
                    inner.cancel_dropped_task(d, "queue closed");
                }
                return;
            }
        }
    }
}

/// Trilinear `2^shift`× upsample of a `(64 >> shift)³` subregion of a
/// 64³ parent chunk into a fresh 64³ buffer. `(target_cx, target_cy,
/// target_cz)` are the *target chunk's* coordinates at its LOD; the low
/// `shift` bits select which `(64 >> shift)³` block of the parent the
/// target chunk corresponds to.
///
/// `shift` must be in `1..=6`. `shift == 6` is the degenerate case where
/// the target chunk maps to a single parent voxel — trilinear blending
/// with the parent's immediate neighbors (clamped at index 63) still
/// produces a smooth gradient across the target chunk in that case,
/// which is the main reason to prefer trilinear over nearest-neighbor
/// here: the preview looks like a low-pass image rather than a grid of
/// 64³ flat-color blocks.
///
/// Edges of the parent are clamped (we have no access to neighboring
/// parent chunks here). Sampling uses Q0.8 fixed point with parent
/// voxels treated as point samples at integer coordinates; target voxel
/// `t` on an axis with offset `o` samples parent at the continuous
/// coordinate `o + (t + 0.5)/scale - 0.5`.
fn upsample_from_parent(parent: &[u8], target_cx: u32, target_cy: u32, target_cz: u32, shift: u8) -> Vec<u8> {
    debug_assert_eq!(parent.len(), CHUNK_VOXELS);
    debug_assert!((1..=6).contains(&shift));
    let mask = (1u32 << shift) - 1;
    let region = (64usize) >> shift;
    let scale = 1usize << shift;
    let ox = ((target_cx & mask) as usize) * region;
    let oy = ((target_cy & mask) as usize) * region;
    let oz = ((target_cz & mask) as usize) * region;

    // Per-axis precompute: for each of 64 target positions, derive
    // (p0, p1, wf) where p0/p1 are the bracketing parent indices
    // (clamped to [0, 63]) and `wf ∈ [0, 256)` is the Q0.8 weight on
    // p1. wf is forced to 0 when pos lands outside [0, 63] so the
    // sample collapses to the clamped boundary instead of leaking
    // weight from a phantom neighbor.
    fn axis(o: usize, scale: usize) -> [(u8, u8, i32); 64] {
        let mut out = [(0u8, 0u8, 0i32); 64];
        for t in 0..64usize {
            // pos_q = (o + (t + 0.5)/scale - 0.5) * 256
            //       = o*256 + ((2t+1) * 128) / scale - 128
            let pos_q = (o as i32) * 256 + (((2 * t + 1) as i32) * 128) / (scale as i32) - 128;
            let p0_raw = pos_q >> 8;
            let p0 = p0_raw.clamp(0, 63);
            let p1 = (p0 + 1).min(63);
            let wf = if pos_q < 0 || p0_raw >= 63 { 0 } else { pos_q - (p0 << 8) };
            out[t] = (p0 as u8, p1 as u8, wf);
        }
        out
    }

    let ax = axis(ox, scale);
    let ay = axis(oy, scale);
    let az = axis(oz, scale);

    let mut out = vec![0u8; CHUNK_VOXELS];
    for tz in 0..64usize {
        let (z0, z1, wfz) = (az[tz].0 as usize, az[tz].1 as usize, az[tz].2);
        let wcz = 256 - wfz;
        for ty in 0..64usize {
            let (y0, y1, wfy) = (ay[ty].0 as usize, ay[ty].1 as usize, ay[ty].2);
            let wcy = 256 - wfy;
            let r00 = z0 * 64 * 64 + y0 * 64;
            let r01 = z0 * 64 * 64 + y1 * 64;
            let r10 = z1 * 64 * 64 + y0 * 64;
            let r11 = z1 * 64 * 64 + y1 * 64;
            let t_row = tz * 64 * 64 + ty * 64;
            for tx in 0..64usize {
                let (x0, x1, wfx) = (ax[tx].0 as usize, ax[tx].1 as usize, ax[tx].2);
                let wcx = 256 - wfx;

                let v000 = parent[r00 + x0] as i32;
                let v001 = parent[r00 + x1] as i32;
                let v010 = parent[r01 + x0] as i32;
                let v011 = parent[r01 + x1] as i32;
                let v100 = parent[r10 + x0] as i32;
                let v101 = parent[r10 + x1] as i32;
                let v110 = parent[r11 + x0] as i32;
                let v111 = parent[r11 + x1] as i32;

                let v00 = (v000 * wcx + v001 * wfx) >> 8;
                let v01 = (v010 * wcx + v011 * wfx) >> 8;
                let v10 = (v100 * wcx + v101 * wfx) >> 8;
                let v11 = (v110 * wcx + v111 * wfx) >> 8;
                let v0 = (v00 * wcy + v01 * wfy) >> 8;
                let v1 = (v10 * wcy + v11 * wfy) >> 8;
                let v = (v0 * wcz + v1 * wfz) >> 8;
                out[t_row + tx] = v as u8;
            }
        }
    }
    out
}
