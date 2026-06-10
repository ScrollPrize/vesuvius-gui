//! On-disk persistence for cache chunks.
//!
//! Layout: a 3D grid of sparse "shard" data files per LOD,
//! `{root}/chunks-L{lod:02}-X{sx:03}-Y{sy:03}-Z{sz:03}.dat`. A shard owns
//! every cache chunk falling inside a `SHARD_CHUNKS_PER_AXIS³` cube
//! (defaults to 128³ chunks = 8192³ voxels). At 256 KiB per chunk that
//! gives a constant logical file size of `128³ · 256 KiB = 2³⁹ bytes`
//! (512 GiB) — well below the per-file sparse-allocation ceilings imposed
//! by common file systems on very large single files. Each shard is
//! `set_len(SHARD_BYTES)` on first open; only chunks actually written
//! occupy physical blocks (ext4/xfs/btrfs/APFS).
//!
//! Chunk state (Missing / Resident / Empty) still lives in a single
//! `chunks.idx` sidecar (`super::sidecar::Sidecar`) — one byte per slot per
//! LOD, addressed by the global linear index `((z·ny + y)·nx + x)`
//! regardless of which shard the chunk lives in. Writers update the bitmap
//! with `Release` after pwriting bytes; readers observing `Resident` with
//! `Acquire` are guaranteed to see the full 256 KiB through the mmap of
//! the matching shard (same inode → same page cache).
//!
//! A background sync thread snapshots the bitmap, fsyncs every shard file
//! currently open for any LOD that saw transitions, and atomically renames
//! the sidecar into place every `SYNC_INTERVAL` or after
//! `SYNC_COUNT_THRESHOLD` transitions — whichever comes first. The sidecar
//! is always a strict subset of durable bytes, so a crash loses at most
//! the last sync interval of work (chunks are re-downloaded).

use super::sidecar::{self, LodDims, Sidecar, STATE_EMPTY, STATE_LOCKED, STATE_MISSING, STATE_RESIDENT};
use super::state::ChunkKey;
use super::CHUNK_VOXELS;
use memmap::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub enum LoadOutcome {
    Resident { mmap: Arc<Mmap>, offset: usize },
    Empty,
    Missing,
}

/// Result of `DiskStore::write_atomic`. The variant tells the caller
/// whether the slot transitioned (so it should `record_fill` for LRU
/// accounting) or whether another thread had already published a
/// definitive state. See `write_atomic` for the full protocol.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WriteOutcome {
    /// We held the per-slot lock and wrote the bytes; sidecar
    /// transitioned `MISSING → RESIDENT`. Caller must call
    /// `record_fill` exactly once.
    Wrote,
    /// CAS observed `RESIDENT` before we could claim — another writer
    /// got there first. Skip the write; the existing bytes are
    /// authoritative.
    AlreadyResident,
    /// CAS observed `EMPTY` — another path marked this slot
    /// definitively absent. Skip the write; publish `Empty`.
    AlreadyEmpty,
}

/// Read-only view returned by `DiskStore::peek_shard`. Bundles the shard's
/// mmap (for sparse-hole-aware voxel reads) with its per-chunk state
/// bitmap (so the reader can distinguish "resident" from "sparse hole").
pub struct ShardSnapshot {
    pub mmap: Arc<Mmap>,
    pub state_bits: Arc<ChunkStateBits>,
}

#[derive(Clone)]
struct LodFile {
    file: Arc<File>,
    mmap: Arc<Mmap>,
    /// Per-chunk-in-shard 2-bit state map. Mirrors the sidecar's per-chunk
    /// state, but indexed by `in_shard_chunk_idx` (so volume readers can
    /// probe it from a cached shard base without consulting the global
    /// sidecar bitmap or the cache's DashMap). Populated when the shard is
    /// first opened (`ensure_open`) and updated on every `write_atomic` /
    /// `mark_empty` transition.
    state_bits: Arc<ChunkStateBits>,
    /// Set by `write_atomic` when chunk bytes land in this shard; consumed
    /// by `do_sync`, which fsyncs only dirty shards. Without it, any
    /// sidecar transition (including pure-read access-epoch bumps) forced
    /// an fsync of every open shard of the LOD every sync tick.
    dirty: Arc<AtomicBool>,
}

pub type ShardCoord = (u32, u32, u32);

/// Per-shard chunk-state bitmap. Two bits per chunk, raster order matching
/// `in_shard_chunk_idx`. Production size: 128³ × 2 bits = 524 288 bytes
/// (512 KiB) per (lod, shard).
///
/// Encoding:
/// - `00` Unknown    — never observed; reader takes the slow path (LOD climb).
/// - `01` Dispatched — fetch in flight; reader still climbs, but bulk
///   dispatchers can read this to skip re-issuing the same fetch.
/// - `10` Resident   — chunk bytes are present in the mmap at the matching
///   offset; reader observing this with `Acquire` is guaranteed to see the
///   full 256 KiB (paired with the `Release` store in `write_atomic`).
/// - `11` Empty      — chunk is definitively absent; reader returns 0 and
///   does *not* climb (Empty at a fine LOD overrides coarser data).
pub struct ChunkStateBits {
    words: Box<[AtomicU64]>,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ChunkBitState {
    Unknown = 0b00,
    Dispatched = 0b01,
    Resident = 0b10,
    Empty = 0b11,
}

const BITS_PER_CHUNK: u32 = 2;

impl ChunkStateBits {
    fn new(num_chunks: u64) -> Self {
        let total_bits = num_chunks * BITS_PER_CHUNK as u64;
        let num_words = ((total_bits + 63) / 64) as usize;
        let mut words = Vec::with_capacity(num_words);
        for _ in 0..num_words {
            words.push(AtomicU64::new(0));
        }
        Self { words: words.into_boxed_slice() }
    }

    #[inline]
    pub fn load(&self, in_shard_idx: u64) -> ChunkBitState {
        let bit_pos = in_shard_idx * BITS_PER_CHUNK as u64;
        let word_idx = (bit_pos / 64) as usize;
        let shift = (bit_pos % 64) as u32;
        // SAFETY: word_idx is checked via the `Box<[AtomicU64]>` indexing.
        // Acquire pairs with the Release store in `store(Resident)` to
        // guarantee the mmap bytes are visible.
        let bits = (self.words[word_idx].load(Ordering::Acquire) >> shift) & 0b11;
        match bits {
            0b00 => ChunkBitState::Unknown,
            0b01 => ChunkBitState::Dispatched,
            0b10 => ChunkBitState::Resident,
            _ => ChunkBitState::Empty,
        }
    }

    /// CAS-loop store of a 2-bit field. Concurrent stores to *different*
    /// chunks within the same word race on the CAS but each will succeed
    /// within a few attempts; same-chunk concurrent writes are guarded by
    /// the cache's per-chunk dispatch claim, so the only contention here is
    /// across neighboring chunks of the same shard.
    pub fn store(&self, in_shard_idx: u64, state: ChunkBitState) {
        let bit_pos = in_shard_idx * BITS_PER_CHUNK as u64;
        let word_idx = (bit_pos / 64) as usize;
        let shift = (bit_pos % 64) as u32;
        let new_bits = (state as u64) << shift;
        let mask = 0b11u64 << shift;
        let word = &self.words[word_idx];
        let mut cur = word.load(Ordering::Relaxed);
        loop {
            let next = (cur & !mask) | new_bits;
            if next == cur {
                return;
            }
            match word.compare_exchange_weak(cur, next, Ordering::Release, Ordering::Relaxed) {
                Ok(_) => return,
                Err(actual) => cur = actual,
            }
        }
    }

    /// Conditional store: write `state` only if the current 2-bit field is
    /// `Unknown`. Returns true if we transitioned, false otherwise. Used to
    /// claim "needs dispatch" without ever clobbering a Resident/Empty/
    /// Dispatched cell.
    pub fn store_if_unknown(&self, in_shard_idx: u64, state: ChunkBitState) -> bool {
        let bit_pos = in_shard_idx * BITS_PER_CHUNK as u64;
        let word_idx = (bit_pos / 64) as usize;
        let shift = (bit_pos % 64) as u32;
        let new_bits = (state as u64) << shift;
        let mask = 0b11u64 << shift;
        let word = &self.words[word_idx];
        let mut cur = word.load(Ordering::Relaxed);
        loop {
            if (cur >> shift) & 0b11 != 0 {
                return false;
            }
            let next = (cur & !mask) | new_bits;
            match word.compare_exchange_weak(cur, next, Ordering::Release, Ordering::Relaxed) {
                Ok(_) => return true,
                Err(actual) => cur = actual,
            }
        }
    }
}

struct LodSlot {
    dims: LodDims,
    /// Lazily populated per shard. The HashMap lookup runs once per chunk
    /// state transition (inside `load` / `write_atomic`), not per voxel
    /// read — voxel access goes through the cached `Arc<Mmap>` returned
    /// from `load` and never re-enters this map.
    opened: Mutex<HashMap<ShardCoord, LodFile>>,
}

struct SyncInner {
    shutdown: bool,
    wake_now: bool,
    last_sync: Instant,
}

struct SyncState {
    inner: Mutex<SyncInner>,
    cv: Condvar,
}

struct DiskStoreInner {
    root: PathBuf,
    shard_chunks_per_axis: u32,
    sidecar: Arc<Sidecar>,
    lods: Vec<LodSlot>,
    sync_state: SyncState,
}

pub struct DiskStore {
    inner: Arc<DiskStoreInner>,
    sync_thread: Mutex<Option<JoinHandle<()>>>,
}

/// Default shard side length in chunks. `128³ · 256 KiB = 2³⁹` bytes per
/// shard — comfortably under per-file sparse ceilings while keeping shard
/// counts tiny for typical volumes (worst case 80 TiB across ~160 shards
/// for our largest current volume).
pub(crate) const SHARD_CHUNKS_PER_AXIS: u32 = 128;

const SYNC_INTERVAL: Duration = Duration::from_secs(10);
const SYNC_COUNT_THRESHOLD: u64 = 256;

struct ResolvedKey {
    sidecar_idx: u64,
    shard: ShardCoord,
    in_shard_idx: u64,
}

impl DiskStore {
    /// Open (or create) the cache directory for a volume.
    ///
    /// `extent` and `max_lod` come from the backfiller and define the sparse
    /// shard layout. If an existing sidecar describes a different layout,
    /// all data files plus the sidecar are renamed aside
    /// (`*.stale.<unix_ts>`) and a fresh cache is created — silent
    /// corruption is worse than the re-download cost.
    pub fn new(root: impl Into<PathBuf>, volume_id: String, extent: [u32; 3], max_lod: u8) -> Self {
        Self::new_with_shard_chunks_per_axis(root, volume_id, extent, max_lod, SHARD_CHUNKS_PER_AXIS)
    }

    /// Test-only constructor letting callers pick a smaller shard side so
    /// multi-shard layouts can be exercised without inflating extents.
    fn new_with_shard_chunks_per_axis(
        root: impl Into<PathBuf>,
        volume_id: String,
        extent: [u32; 3],
        max_lod: u8,
        shard_chunks_per_axis: u32,
    ) -> Self {
        assert!(shard_chunks_per_axis > 0, "shard side must be > 0");
        let root = root.into();
        std::fs::create_dir_all(&root).expect("create cache root");

        let expected = sidecar::Header::new(volume_id, extent, max_lod);
        let sidecar_path = sidecar::sidecar_path(&root);

        // open_or_create maps the file RW: existing data is preserved
        // (and the file extended if shorter than the new layout); a
        // brand-new file is sized and seeded with the header prefix.
        // It returns Ok(None) when an existing file's header does NOT
        // match `expected`, and Err on real I/O failure — both fall
        // back to "rename aside + rebuild fresh".
        let sidecar = match Sidecar::open_or_create(&sidecar_path, expected.clone()) {
            Ok(Some(s)) => Arc::new(s),
            Ok(None) => {
                log::warn!(
                    "[cache] sidecar header mismatch at {}; renaming aside and rebuilding",
                    sidecar_path.display(),
                );
                stale_rename_everything(&root);
                Arc::new(
                    Sidecar::open_or_create(&sidecar_path, expected)
                        .expect("rebuilt sidecar create succeeded")
                        .expect("rebuilt sidecar present on fresh file"),
                )
            }
            Err(e) => {
                log::warn!(
                    "[cache] sidecar open failed at {} ({}); renaming aside and rebuilding",
                    sidecar_path.display(),
                    e
                );
                stale_rename_everything(&root);
                Arc::new(
                    Sidecar::open_or_create(&sidecar_path, expected)
                        .expect("rebuilt sidecar create succeeded")
                        .expect("rebuilt sidecar present on fresh file"),
                )
            }
        };

        let lods: Vec<LodSlot> = sidecar
            .header
            .lods
            .iter()
            .map(|d| LodSlot {
                dims: *d,
                opened: Mutex::new(HashMap::new()),
            })
            .collect();

        let inner = Arc::new(DiskStoreInner {
            root,
            shard_chunks_per_axis,
            sidecar,
            lods,
            sync_state: SyncState {
                inner: Mutex::new(SyncInner {
                    shutdown: false,
                    wake_now: false,
                    last_sync: Instant::now(),
                }),
                cv: Condvar::new(),
            },
        });

        let recovered = inner.recover_locked_slots();
        if recovered > 0 {
            log::info!(
                "[cache] recovered {} LOCKED slot(s) at {} from prior session",
                recovered,
                inner.root.display(),
            );
        }

        let (reclaimed_bytes, shards_punched) = inner.reclaim_orphan_bytes();
        if reclaimed_bytes > 0 {
            log::info!(
                "[cache] reclaimed {} MiB of orphan bytes across {} shard(s) at {}",
                reclaimed_bytes / (1024 * 1024),
                shards_punched,
                inner.root.display(),
            );
        }

        let sync_inner = inner.clone();
        let sync_thread = std::thread::Builder::new()
            .name("vesuvius-cache-sync".into())
            .spawn(move || sync_loop(sync_inner))
            .expect("spawn sync thread");

        Self {
            inner,
            sync_thread: Mutex::new(Some(sync_thread)),
        }
    }

    pub fn load(&self, key: ChunkKey) -> LoadOutcome {
        let r = match self.inner.resolve(key) {
            Some(r) => r,
            None => return LoadOutcome::Missing,
        };
        match self.inner.sidecar.get_state(key.lod, r.sidecar_idx) {
            STATE_MISSING => LoadOutcome::Missing,
            STATE_EMPTY => LoadOutcome::Empty,
            STATE_RESIDENT => match self.inner.ensure_open(key.lod, r.shard) {
                Ok(lf) => {
                    let off = (r.in_shard_idx as usize) * CHUNK_VOXELS;
                    debug_assert!(off + CHUNK_VOXELS <= lf.mmap.len());
                    LoadOutcome::Resident {
                        mmap: lf.mmap,
                        offset: off,
                    }
                }
                Err(e) => {
                    log::warn!("[{}] ensure_open failed during load: {}", key, e);
                    LoadOutcome::Missing
                }
            },
            other => {
                log::warn!("[{}] unknown sidecar state {}", key, other);
                LoadOutcome::Missing
            }
        }
    }

    /// Convenience used after a `write_atomic` to obtain the freshly-written
    /// slice for in-memory residency. Returns `None` if the chunk isn't
    /// marked Resident in the sidecar.
    pub fn try_load(&self, key: ChunkKey) -> Option<(Arc<Mmap>, usize)> {
        match self.load(key) {
            LoadOutcome::Resident { mmap, offset } => Some((mmap, offset)),
            _ => None,
        }
    }

    /// Write a 64³ chunk into its slot in the matching shard file under
    /// per-slot CAS protection.
    ///
    /// Protocol:
    ///   1. CAS sidecar `MISSING → LOCKED` to claim exclusive access.
    ///   2. pwrite the bytes into the shard.
    ///   3. Store `RESIDENT` (releases the lock, publishes visibility).
    ///   4. Store `Resident` into the per-shard bitmap with `Release`,
    ///      pairing with the reader fast path's `Acquire`.
    ///
    /// Concurrency outcomes (see `WriteOutcome` for the return type):
    ///   - CAS sees `RESIDENT` → another writer already filled the slot.
    ///     Skip — return `AlreadyResident` so the caller can publish from
    ///     the existing bytes without double-counting in `record_fill`.
    ///   - CAS sees `EMPTY` → another path marked the slot definitively
    ///     absent. Return `AlreadyEmpty`; caller publishes `Empty`.
    ///   - CAS sees `LOCKED` → a peer write or punch_hole is in flight.
    ///     Spin (µs-scale) until the lock releases, then retry the CAS.
    pub fn write_atomic(&self, key: ChunkKey, bytes: &[u8]) -> std::io::Result<WriteOutcome> {
        assert_eq!(
            bytes.len(),
            CHUNK_VOXELS,
            "unified-cache: backfiller returned {} bytes, expected {}",
            bytes.len(),
            CHUNK_VOXELS
        );
        let r = self.inner.resolve(key).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("chunk {} out of bounds for this LOD", key),
            )
        })?;
        let lf = self.inner.ensure_open(key.lod, r.shard)?;

        // Acquire the per-slot lock. The sidecar byte itself is the
        // mutex cell — see `STATE_LOCKED`.
        let sidecar = &self.inner.sidecar;
        loop {
            match sidecar.compare_exchange_state(key.lod, r.sidecar_idx, STATE_MISSING, STATE_LOCKED) {
                Ok(_) => break,
                // Yield rather than spin: the peer's critical section
                // includes a 256 KiB pwrite (or a batched purge punch),
                // which can take milliseconds on a saturated disk —
                // spin_loop would burn a full core for the duration.
                Err(STATE_LOCKED) => std::thread::yield_now(),
                Err(STATE_RESIDENT) => return Ok(WriteOutcome::AlreadyResident),
                Err(STATE_EMPTY) => return Ok(WriteOutcome::AlreadyEmpty),
                Err(other) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("unexpected sidecar state {} for {}", other, key),
                    ));
                }
            }
        }

        let off = r.in_shard_idx * CHUNK_VOXELS as u64;
        if let Err(e) = pwrite_all(&lf.file, off, bytes) {
            // Release the lock back to MISSING so spinning peers can
            // make progress. The slot stays logically empty; any
            // orphan bytes already written will be overwritten on
            // retry or punched by a later eviction.
            sidecar.set_state(key.lod, r.sidecar_idx, STATE_MISSING);
            return Err(e);
        }
        // Mark before publishing RESIDENT so a sync racing this write
        // either fsyncs the bytes or hasn't seen the RESIDENT bit yet.
        lf.dirty.store(true, Ordering::Release);
        sidecar.set_state(key.lod, r.sidecar_idx, STATE_RESIDENT);
        lf.state_bits.store(r.in_shard_idx, ChunkBitState::Resident);
        self.inner.maybe_wake_sync();
        Ok(WriteOutcome::Wrote)
    }

    /// Synchronously snapshot the sidecar, fsync any shard files that
    /// had transitions since the last sync, and atomically write the
    /// sidecar to disk. Used on graceful shutdown and by tests that need
    /// durability before the periodic sync would fire.
    pub fn flush(&self) {
        do_sync(&self.inner);
    }

    /// Chunks-per-axis of one shard cube. Volume readers use this to derive
    /// `(shard_coord, in_shard_chunk_idx)` from a `ChunkKey` so they can
    /// cache the shard's mmap base in their hot slot. Production value is
    /// `SHARD_CHUNKS_PER_AXIS = 128`; tests can construct stores with a
    /// smaller value.
    pub fn shard_chunks_per_axis(&self) -> u32 {
        self.inner.shard_chunks_per_axis
    }

    /// Return the shard's mmap + per-chunk state bitmap if the shard is
    /// currently open, without creating or mapping it. Used by the volume's
    /// per-render hot slot to fast-path reads once any chunk in the shard
    /// has been materialized.
    pub fn peek_shard(&self, lod: u8, shard: ShardCoord) -> Option<ShardSnapshot> {
        let slot = self.inner.lods.get(lod as usize)?;
        slot.opened.lock().unwrap().get(&shard).map(|lf| ShardSnapshot {
            mmap: lf.mmap.clone(),
            state_bits: lf.state_bits.clone(),
        })
    }

    /// Ensure the shard at `(lod, shard)` is open (sparse mmap + seeded
    /// bitmap), then return its snapshot. The shard-based volume slow path
    /// calls this on its first miss for a shard so subsequent per-voxel
    /// lookups can drive entirely off the bitmap without re-entering the
    /// DashMap. Returns `Ok(None)` when the LOD index is out of range.
    pub fn ensure_shard_open(&self, lod: u8, shard: ShardCoord) -> std::io::Result<Option<ShardSnapshot>> {
        let Some(slot) = self.inner.lods.get(lod as usize) else {
            return Ok(None);
        };
        // Refuse shards wholly outside the LOD's chunk grid — a stray
        // probe past the volume extent would otherwise create a phantom
        // sparse shard file and pay the bitmap-seed walk for slots that
        // can never hold data.
        let sca = self.inner.shard_chunks_per_axis as u64;
        let dims = &slot.dims;
        if (shard.0 as u64) * sca >= dims.nx as u64
            || (shard.1 as u64) * sca >= dims.ny as u64
            || (shard.2 as u64) * sca >= dims.nz as u64
        {
            return Ok(None);
        }
        let lf = self.inner.ensure_open(lod, shard)?;
        Ok(Some(ShardSnapshot {
            mmap: lf.mmap.clone(),
            state_bits: lf.state_bits.clone(),
        }))
    }

    /// Mark `key` as `Dispatched` on its shard bitmap if the cell is still
    /// `Unknown`. No sidecar write — Dispatched is in-memory only (it's a
    /// per-process "fetch in flight" claim, not durable state). Returns
    /// `Ok(false)` when the bit was already non-Unknown (Resident / Empty /
    /// Dispatched), so callers can skip redundant dispatch work.
    pub fn mark_dispatched(&self, key: ChunkKey) -> std::io::Result<bool> {
        let r = self.inner.resolve(key).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("chunk {} out of bounds for this LOD", key),
            )
        })?;
        let lf = self.inner.ensure_open(key.lod, r.shard)?;
        Ok(lf.state_bits.store_if_unknown(r.in_shard_idx, ChunkBitState::Dispatched))
    }

    /// Write `bytes` into the slot for `key` as a best-effort preview.
    /// Used by the upscale-from-parent path: it stashes interpolated
    /// bytes into the target shard's mmap region so subsequent reads
    /// see something while the real fetch streams in. The eventual
    /// `write_atomic` overwrites these bytes with the downloaded data.
    ///
    /// Concurrency: takes the per-slot CAS lock to serialize against
    /// `write_atomic` and `punch_hole`. Returns `Ok(false)` (silently
    /// skipped) if the CAS finds the slot in any non-MISSING state —
    /// the preview is best-effort and must never compete with, or
    /// clobber, a definitive transition. Always resets the sidecar
    /// byte back to `MISSING` after writing, so the preview bytes are
    /// effectively orphaned from a sidecar-state perspective (kept
    /// only as a read-through-mmap hint until the real fetch lands).
    pub fn write_unconfirmed(&self, key: ChunkKey, bytes: &[u8]) -> std::io::Result<bool> {
        assert_eq!(
            bytes.len(),
            CHUNK_VOXELS,
            "unified-cache: write_unconfirmed got {} bytes, expected {}",
            bytes.len(),
            CHUNK_VOXELS
        );
        let r = self.inner.resolve(key).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("chunk {} out of bounds for this LOD", key),
            )
        })?;
        let lf = self.inner.ensure_open(key.lod, r.shard)?;

        // Try to claim the per-slot lock. Don't spin on LOCKED —
        // upscale is best-effort; if a real op is in flight we skip.
        let sidecar = &self.inner.sidecar;
        if sidecar
            .compare_exchange_state(key.lod, r.sidecar_idx, STATE_MISSING, STATE_LOCKED)
            .is_err()
        {
            return Ok(false);
        }

        let off = r.in_shard_idx * CHUNK_VOXELS as u64;
        let result = pwrite_all(&lf.file, off, bytes);
        // Release the lock back to MISSING regardless of write result:
        // the slot has no durable readable state from this path, and
        // a future write_atomic must be free to claim it.
        sidecar.set_state(key.lod, r.sidecar_idx, STATE_MISSING);
        result.map(|_| true)
    }

    /// Mark `key` as definitively absent in the sidecar. No bytes are written
    /// to a data file.
    pub fn mark_empty(&self, key: ChunkKey) -> std::io::Result<()> {
        let r = self.inner.resolve(key).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("chunk {} out of bounds for this LOD", key),
            )
        })?;
        // Open the shard so subsequent reads can see the Empty bit through
        // the per-shard bitmap fast path. The shard file is created (sparse)
        // by `ensure_open`; no bytes are written.
        let lf = self.inner.ensure_open(key.lod, r.shard)?;
        self.inner.sidecar.set_state(key.lod, r.sidecar_idx, STATE_EMPTY);
        lf.state_bits.store(r.in_shard_idx, ChunkBitState::Empty);
        self.inner.maybe_wake_sync();
        Ok(())
    }

    /// Locate `key`'s shard layout: `(shard_coord, in_shard_chunk_idx)`. The
    /// volume reader uses this to address the shard's mmap and bitmap. Returns
    /// `None` if the chunk is out of bounds for this LOD.
    pub fn locate(&self, key: ChunkKey) -> Option<(ShardCoord, u64)> {
        let r = self.inner.resolve(key)?;
        Some((r.shard, r.in_shard_idx))
    }

    /// Read the access epoch tag for `key` from the sidecar. Returns
    /// `None` only when the key is out of bounds for its LOD.
    pub fn get_access_epoch(&self, key: ChunkKey) -> Option<u8> {
        let r = self.inner.resolve(key)?;
        Some(self.inner.sidecar.get_access_epoch(key.lod, r.sidecar_idx))
    }

    /// Expose the underlying `Sidecar` for purge / epoch-seed paths that
    /// need to iterate per-chunk state. Returns the existing `Arc` clone
    /// (cheap). The DiskStore retains ownership; callers don't keep
    /// long-lived references past their seed/purge passes.
    pub fn sidecar(&self) -> Arc<Sidecar> {
        self.inner.sidecar.clone()
    }

    /// Stamp `key` with `epoch`. Silently no-ops if the key is out of
    /// bounds. Does not mark the sidecar pending — access-epoch updates
    /// are LRU bookkeeping, not residency state; the sync thread picks
    /// them up alongside the next bitmap flush.
    pub fn set_access_epoch(&self, key: ChunkKey, epoch: u8) {
        if let Some(r) = self.inner.resolve(key) {
            self.inner.sidecar.set_access_epoch(key.lod, r.sidecar_idx, epoch);
        }
    }

    /// CAS the access-epoch tag for `key` from `current` to `new`. Returns
    /// `None` if the key is out of bounds; otherwise the CAS result.
    ///
    /// `touch_access` uses this to arbitrate concurrent LRU bumps: the
    /// winning CAS is the only thread that should adjust the histogram
    /// (otherwise N racing touches inflate the destination bucket by N
    /// and decrement the source bucket by N, instead of by 1).
    pub fn cas_access_epoch(&self, key: ChunkKey, current: u8, new: u8) -> Option<Result<u8, u8>> {
        let r = self.inner.resolve(key)?;
        Some(
            self.inner
                .sidecar
                .compare_exchange_access_epoch(key.lod, r.sidecar_idx, current, new),
        )
    }

    /// Punch a hole in the matching shard file at `key`'s slot, freeing
    /// the underlying physical blocks. The shard file's logical size is
    /// unchanged (sparse file). Returns `Ok(false)` if the shard isn't
    /// open (nothing to punch — the slot was never written), `Ok(true)`
    /// if a hole was successfully punched.
    ///
    /// IMPORTANT ordering: callers must demote the chunk's state to
    /// MISSING (sidecar + per-shard ChunkStateBits, with Release) BEFORE
    /// calling this. Otherwise a concurrent reader observing Resident
    /// may pass through and read zeros from a chunk it believes is
    /// valid. Readers that already passed the bitmap check before our
    /// demote may transiently see zeros — the async pipeline tolerates
    /// that (next frame re-reads from the now-Missing slot).
    /// Test-only single-chunk wrapper; production eviction goes through
    /// `punch_hole_run` with batched, run-coalesced victims.
    #[cfg(test)]
    pub fn punch_hole(&self, key: ChunkKey) -> std::io::Result<bool> {
        let r = self.inner.resolve(key).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("chunk {} out of bounds for this LOD", key),
            )
        })?;
        self.punch_hole_run(key.lod, r.shard, r.in_shard_idx, 1)
    }

    /// Punch a contiguous run of `n_chunks` slots starting at
    /// `start_in_shard_idx` in one shard — one `fallocate` for the whole
    /// run, which is what makes batch eviction cheap (purge victims walk
    /// in raster order, so runs of hundreds of chunks are common).
    ///
    /// Uses the open shard's fd when available; otherwise opens the file
    /// ad hoc — a RESIDENT chunk from a previous session can be purged
    /// without the shard ever having been opened this session. Returns
    /// `Ok(false)` if the shard file doesn't exist (no bytes to reclaim)
    /// or has an unexpected length (defensive — never punch a file we
    /// don't understand).
    ///
    /// Same ordering contract as `punch_hole`: callers must hold the
    /// per-slot LOCKED claim (or have demoted the slots) for every chunk
    /// in the run before punching.
    pub fn punch_hole_run(
        &self,
        lod: u8,
        shard: ShardCoord,
        start_in_shard_idx: u64,
        n_chunks: u64,
    ) -> std::io::Result<bool> {
        let off = start_in_shard_idx * CHUNK_VOXELS as u64;
        let len = n_chunks * CHUNK_VOXELS as u64;
        if let Some(slot) = self.inner.lods.get(lod as usize) {
            let lf = slot.opened.lock().unwrap().get(&shard).cloned();
            if let Some(lf) = lf {
                punch_hole_at(&lf.file, off, len)?;
                return Ok(true);
            }
        }
        let path = self.inner.root.join(shard_filename(lod, shard));
        let file = match OpenOptions::new().read(true).write(true).open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(e),
        };
        let expected = self.inner.shard_bytes();
        let actual = file.metadata()?.len();
        if actual != expected {
            log::warn!(
                "[cache] punch_hole_run: skip shard {} (len {} != expected {})",
                path.display(),
                actual,
                expected,
            );
            return Ok(false);
        }
        punch_hole_at(&file, off, len)?;
        Ok(true)
    }
}

// Hole punching is OS-specific. Linux uses `fallocate(FALLOC_FL_PUNCH_HOLE)`,
// macOS uses `fcntl(F_PUNCHHOLE, ...)`. Other targets get a no-op fallback:
// eviction still demotes the bitmap (so reads return Missing) but physical
// disk space isn't reclaimed.

#[cfg(target_os = "linux")]
pub(crate) fn punch_hole_at(file: &std::fs::File, offset: u64, len: u64) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    // FALLOC_FL_PUNCH_HOLE requires FALLOC_FL_KEEP_SIZE on Linux.
    let mode = libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE;
    let rc = unsafe {
        libc::fallocate(file.as_raw_fd(), mode, offset as libc::off_t, len as libc::off_t)
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(crate) fn punch_hole_at(file: &std::fs::File, offset: u64, len: u64) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    // Apple platforms expose hole punching via fcntl(F_PUNCHHOLE) with a
    // `fpunchhole_t` struct. Same semantics as Linux's PUNCH_HOLE +
    // KEEP_SIZE: bytes in the range become zeros, file size is unchanged.
    let arg = libc::fpunchhole_t {
        fp_flags: 0,
        reserved: 0,
        fp_offset: offset as libc::off_t,
        fp_length: len as libc::off_t,
    };
    let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_PUNCHHOLE, &arg) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "ios")))]
pub(crate) fn punch_hole_at(_file: &std::fs::File, _offset: u64, _len: u64) -> std::io::Result<()> {
    Ok(())
}

impl DiskStoreInner {
    fn resolve(&self, key: ChunkKey) -> Option<ResolvedKey> {
        let slot = self.lods.get(key.lod as usize)?;
        let sidecar_idx = slot.dims.linear_index(key.x, key.y, key.z)?;
        let sca = self.shard_chunks_per_axis;
        let shard = (key.x / sca, key.y / sca, key.z / sca);
        let wx = (key.x % sca) as u64;
        let wy = (key.y % sca) as u64;
        let wz = (key.z % sca) as u64;
        let s = sca as u64;
        let in_shard_idx = (wz * s + wy) * s + wx;
        Some(ResolvedKey {
            sidecar_idx,
            shard,
            in_shard_idx,
        })
    }

    fn ensure_open(&self, lod: u8, shard: ShardCoord) -> std::io::Result<LodFile> {
        let slot = &self.lods[lod as usize];
        // Fast path: shard already open. Drop the lock immediately so the
        // happy case never serializes with concurrent first-opens of other
        // shards.
        if let Some(lf) = slot.opened.lock().unwrap().get(&shard) {
            return Ok(lf.clone());
        }
        // Cold path: build the file + mmap + seeded bitmap *outside* the
        // per-LOD `opened` mutex. Seeding walks 128³ sidecar entries per
        // shard — holding the lock through that serializes every worker
        // and per-voxel `peek_shard` against the cold opener. Two threads
        // can race here; the cheaper-loser handles it with the
        // double-checked insert below.
        let total = self.shard_bytes();
        let path = self.root.join(shard_filename(lod, shard));
        let file = OpenOptions::new().read(true).write(true).create(true).open(&path)?;
        let cur_len = file.metadata()?.len();
        if cur_len == 0 {
            file.set_len(total)?;
        } else if cur_len != total {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "lod {} shard {:?} at {} has length {} (expected {}); refusing to use",
                    lod,
                    shard,
                    path.display(),
                    cur_len,
                    total,
                ),
            ));
        }
        let mmap = unsafe { MmapOptions::new().len(total as usize).map(&file)? };
        let sca = self.shard_chunks_per_axis as u64;
        let state_bits = Arc::new(ChunkStateBits::new(sca * sca * sca));
        self.seed_shard_state_bits(lod, shard, &slot.dims, &state_bits);
        let lf = LodFile {
            file: Arc::new(file),
            mmap: Arc::new(mmap),
            state_bits,
            dirty: Arc::new(AtomicBool::new(false)),
        };
        // Re-check under the lock: if another thread opened the same shard
        // concurrently, drop our work and use theirs to keep the
        // mmap/bitmap unique per shard.
        let mut guard = slot.opened.lock().unwrap();
        if let Some(existing) = guard.get(&shard) {
            return Ok(existing.clone());
        }
        guard.insert(shard, lf.clone());
        Ok(lf)
    }

    /// Populate `bits` for the chunks owned by `(lod, shard)` from the
    /// sidecar's persisted state. Skips chunks outside the LOD extent
    /// (`linear_index` returns `None`) — they stay as Unknown.
    fn seed_shard_state_bits(&self, lod: u8, shard: ShardCoord, dims: &LodDims, bits: &ChunkStateBits) {
        let sca = self.shard_chunks_per_axis;
        let s = sca as u64;
        let base_cx = shard.0 * sca;
        let base_cy = shard.1 * sca;
        let base_cz = shard.2 * sca;
        // Clamp the iteration to the LOD's chunk extent so we don't touch
        // sidecar slots that don't exist for this LOD.
        let hi_cx = (base_cx + sca).min(dims.nx);
        let hi_cy = (base_cy + sca).min(dims.ny);
        let hi_cz = (base_cz + sca).min(dims.nz);
        if base_cx >= dims.nx || base_cy >= dims.ny || base_cz >= dims.nz {
            return;
        }
        for cz in base_cz..hi_cz {
            for cy in base_cy..hi_cy {
                for cx in base_cx..hi_cx {
                    let sidecar_idx = match dims.linear_index(cx, cy, cz) {
                        Some(i) => i,
                        None => continue,
                    };
                    let raw = self.sidecar.get_state(lod, sidecar_idx);
                    let s_bit = match raw {
                        STATE_RESIDENT => ChunkBitState::Resident,
                        STATE_EMPTY => ChunkBitState::Empty,
                        _ => continue,
                    };
                    let wx = (cx - base_cx) as u64;
                    let wy = (cy - base_cy) as u64;
                    let wz = (cz - base_cz) as u64;
                    let in_shard_idx = (wz * s + wy) * s + wx;
                    bits.store(in_shard_idx, s_bit);
                }
            }
        }
    }

    fn shard_bytes(&self) -> u64 {
        let n = self.shard_chunks_per_axis as u64;
        n * n * n * CHUNK_VOXELS as u64
    }

    fn maybe_wake_sync(&self) {
        if self.sidecar.total_pending() >= SYNC_COUNT_THRESHOLD {
            let mut g = self.sync_state.inner.lock().unwrap();
            g.wake_now = true;
            self.sync_state.cv.notify_all();
        }
    }

    /// Walk every LOD's sidecar bitmap, find slots stuck in `STATE_LOCKED`
    /// (a process died mid-`write_atomic`), punch their byte ranges, and
    /// demote them to `STATE_MISSING`. Returns the number of slots
    /// recovered. Must run before any new writers can race against the
    /// bitmap, so callers invoke it from `DiskStore::new` before the sync
    /// thread starts.
    ///
    /// Leaving slots LOCKED would deadlock future `write_atomic` peers
    /// (which spin on LOCKED) and orphan the partially-written bytes.
    fn recover_locked_slots(&self) -> u64 {
        let mut recovered: u64 = 0;
        for (lod_idx, slot) in self.lods.iter().enumerate() {
            let lod = lod_idx as u8;
            let dims = slot.dims;
            let nx = dims.nx;
            let ny = dims.ny;
            let count = dims.count();
            for idx in 0..count {
                if self.sidecar.get_state(lod, idx) != STATE_LOCKED {
                    continue;
                }
                let x = (idx % nx as u64) as u32;
                let y = ((idx / nx as u64) % ny as u64) as u32;
                let z = (idx / (nx as u64 * ny as u64)) as u32;
                let key = ChunkKey { lod, x, y, z };
                let r = match self.resolve(key) {
                    Some(r) => r,
                    None => {
                        log::warn!("[cache] LOCKED slot at {} but resolve failed; clearing", key);
                        self.sidecar.set_state(lod, idx, STATE_MISSING);
                        recovered += 1;
                        continue;
                    }
                };
                match self.ensure_open(lod, r.shard) {
                    Ok(lf) => {
                        let off = r.in_shard_idx * CHUNK_VOXELS as u64;
                        if let Err(e) = punch_hole_at(&lf.file, off, CHUNK_VOXELS as u64) {
                            log::warn!("[cache] punch_hole failed for {}: {}", key, e);
                        }
                    }
                    Err(e) => {
                        log::warn!("[cache] ensure_open failed during LOCKED recovery for {}: {}", key, e);
                    }
                }
                self.sidecar.set_state(lod, idx, STATE_MISSING);
                recovered += 1;
            }
        }
        recovered
    }

    /// Walk every shard file on disk and reclaim physical blocks that no
    /// longer back a `STATE_RESIDENT` slot. Sources of orphan bytes:
    ///   - `write_unconfirmed` previews (writes bytes, then resets the
    ///     sidecar back to MISSING — the bytes linger until reclaimed).
    ///   - Crashed `write_atomic` that wrote bytes but never published
    ///     RESIDENT (the LOCKED slot was swept just above to MISSING).
    ///   - Failed `punch_hole` calls from a prior purge pass.
    ///
    /// Cheap when clean: one `fstat` per shard, no syscalls when
    /// physical ≤ expected + slop. Efficient when dirty: walks the
    /// shard's chunk grid raster-ordered and emits one
    /// `fallocate(PUNCH_HOLE)` per contiguous run of non-RESIDENT slots
    /// — best case one syscall per shard if all chunks are evictable.
    ///
    /// Returns `(bytes_reclaimed, shards_punched)`.
    fn reclaim_orphan_bytes(&self) -> (u64, u64) {
        let chunk_bytes = CHUNK_VOXELS as u64;
        // Slop absorbs one chunk's worth of physical noise (a single
        // failed punch from a prior session, a single in-flight preview
        // write) so we don't enter the per-shard bitmap walk for
        // trivially-small discrepancies. Two or more orphan chunks
        // (≥ 512 KiB reclaimable) triggers the sweep.
        let threshold_bytes = chunk_bytes;
        let sca = self.shard_chunks_per_axis;
        let s = sca as u64;

        let entries = match std::fs::read_dir(&self.root) {
            Ok(e) => e,
            Err(e) => {
                log::warn!(
                    "[cache] orphan sweep: read_dir({}) failed: {}",
                    self.root.display(),
                    e
                );
                return (0, 0);
            }
        };

        let mut total_reclaimed: u64 = 0;
        let mut shards_punched: u64 = 0;

        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            let Some((lod, shard)) = parse_shard_filename(&name_str) else {
                continue;
            };
            let Some(slot) = self.lods.get(lod as usize) else {
                continue;
            };
            let dims = slot.dims;
            let path = entry.path();

            let physical_bytes: u64 = {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::MetadataExt;
                    match std::fs::metadata(&path) {
                        Ok(m) => m.blocks() * 512,
                        Err(e) => {
                            log::warn!(
                                "[cache] orphan sweep: stat {} failed: {}",
                                path.display(),
                                e
                            );
                            continue;
                        }
                    }
                }
                #[cfg(not(unix))]
                {
                    let _ = path;
                    continue;
                }
            };

            let base_cx = shard.0 * sca;
            let base_cy = shard.1 * sca;
            let base_cz = shard.2 * sca;
            if base_cx >= dims.nx || base_cy >= dims.ny || base_cz >= dims.nz {
                continue;
            }

            // Count RESIDENT slots in this shard. Expected physical
            // bytes = count × CHUNK_VOXELS, since CHUNK_VOXELS (256 KiB)
            // is a multiple of every common FS block size.
            let mut resident_count: u64 = 0;
            let hi_cx = (base_cx + sca).min(dims.nx);
            let hi_cy = (base_cy + sca).min(dims.ny);
            let hi_cz = (base_cz + sca).min(dims.nz);
            for cz in base_cz..hi_cz {
                for cy in base_cy..hi_cy {
                    for cx in base_cx..hi_cx {
                        if let Some(idx) = dims.linear_index(cx, cy, cz) {
                            if self.sidecar.get_state(lod, idx) == STATE_RESIDENT {
                                resident_count += 1;
                            }
                        }
                    }
                }
            }
            let expected_bytes = resident_count * chunk_bytes;
            if physical_bytes <= expected_bytes + threshold_bytes {
                continue;
            }

            let file = match OpenOptions::new().read(true).write(true).open(&path) {
                Ok(f) => f,
                Err(e) => {
                    log::warn!(
                        "[cache] orphan sweep: open {} for punching failed: {}",
                        path.display(),
                        e
                    );
                    continue;
                }
            };

            // Walk in raster order matching `in_shard_idx = (wz·s + wy)·s + wx`
            // — consecutive iterations have consecutive file offsets so
            // non-RESIDENT runs are contiguous on disk.
            let mut shard_reclaimed: u64 = 0;
            let mut run_start: Option<u64> = None;
            let mut run_end: u64 = 0;
            for wz in 0..s {
                for wy in 0..s {
                    for wx in 0..s {
                        let in_shard_idx = (wz * s + wy) * s + wx;
                        let cx = base_cx as u64 + wx;
                        let cy = base_cy as u64 + wy;
                        let cz = base_cz as u64 + wz;
                        let is_resident = match dims.linear_index(cx as u32, cy as u32, cz as u32) {
                            Some(idx) => self.sidecar.get_state(lod, idx) == STATE_RESIDENT,
                            None => false,
                        };
                        if !is_resident {
                            if run_start.is_none() {
                                run_start = Some(in_shard_idx);
                            }
                            run_end = in_shard_idx + 1;
                        } else if let Some(start) = run_start.take() {
                            let off = start * chunk_bytes;
                            let len = (run_end - start) * chunk_bytes;
                            match punch_hole_at(&file, off, len) {
                                Ok(()) => shard_reclaimed += len,
                                Err(e) => log::warn!(
                                    "[cache] orphan sweep punch lod={} shard={:?} off={} len={}: {}",
                                    lod, shard, off, len, e
                                ),
                            }
                        }
                    }
                }
            }
            if let Some(start) = run_start.take() {
                let off = start * chunk_bytes;
                let len = (run_end - start) * chunk_bytes;
                match punch_hole_at(&file, off, len) {
                    Ok(()) => shard_reclaimed += len,
                    Err(e) => log::warn!(
                        "[cache] orphan sweep punch lod={} shard={:?} off={} len={}: {}",
                        lod, shard, off, len, e
                    ),
                }
            }

            if shard_reclaimed > 0 {
                total_reclaimed += shard_reclaimed;
                shards_punched += 1;
                log::debug!(
                    "[cache] orphan sweep: reclaimed {} bytes in lod {} shard ({},{},{}) (was {}, expected ≤ {})",
                    shard_reclaimed, lod, shard.0, shard.1, shard.2,
                    physical_bytes, expected_bytes
                );
            }
        }

        (total_reclaimed, shards_punched)
    }
}

impl Drop for DiskStore {
    fn drop(&mut self) {
        {
            let mut g = self.inner.sync_state.inner.lock().unwrap();
            g.shutdown = true;
            self.inner.sync_state.cv.notify_all();
        }
        if let Some(h) = self.sync_thread.lock().unwrap().take() {
            let _ = h.join();
        }
    }
}

fn sync_loop(inner: Arc<DiskStoreInner>) {
    loop {
        let should_shutdown = {
            let g = inner.sync_state.inner.lock().unwrap();
            let timeout = SYNC_INTERVAL
                .checked_sub(g.last_sync.elapsed())
                .unwrap_or(Duration::from_secs(0));
            let (mut g, _) = inner
                .sync_state
                .cv
                .wait_timeout_while(g, timeout, |inner| !inner.shutdown && !inner.wake_now)
                .unwrap();
            let shutdown = g.shutdown;
            g.wake_now = false;
            g.last_sync = Instant::now();
            shutdown
        };
        do_sync(&inner);
        if should_shutdown {
            return;
        }
    }
}

fn do_sync(inner: &DiskStoreInner) {
    let pending = inner.sidecar.take_pending();
    if pending.iter().all(|&p| p == 0) {
        // Nothing changed since the last sync; nothing to flush.
        return;
    }
    // fsync only shards that took data writes since the last sync
    // (`write_atomic` sets the per-shard dirty flag before publishing
    // RESIDENT). Sidecar-only transitions — access-epoch LRU bumps,
    // mark_empty, evictions — need just the msync below, so a pure-read
    // session never fsyncs a data file at all.
    for (lod_idx, slot) in inner.lods.iter().enumerate() {
        let opened: Vec<(ShardCoord, LodFile)> = slot
            .opened
            .lock()
            .unwrap()
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        for (shard, lf) in opened {
            if !lf.dirty.swap(false, Ordering::AcqRel) {
                continue;
            }
            if let Err(e) = lf.file.sync_data() {
                log::warn!(
                    "[cache] sync: fsync LOD {} shard {:?} failed: {}",
                    lod_idx,
                    shard,
                    e
                );
                // Leave the shard dirty so the next sync retries before
                // the sidecar can persist its RESIDENT bits durably.
                lf.dirty.store(true, Ordering::Release);
            }
        }
    }
    // Data files are durable above; now flush the sidecar mmap so the
    // RESIDENT/EMPTY bits and access-epoch tags reach the disk too.
    // Ordering matters: any RESIDENT bit visible on disk must point at
    // already-durable chunk bytes. msync(MS_SYNC) here pairs with the
    // dirty-shard `sync_data` above to honor that invariant.
    if let Err(e) = inner.sidecar.flush() {
        log::warn!("[cache] sync: sidecar msync failed: {}", e);
    }
}

pub(crate) fn shard_filename(lod: u8, shard: ShardCoord) -> String {
    format!(
        "chunks-L{:02}-X{:03}-Y{:03}-Z{:03}.dat",
        lod, shard.0, shard.1, shard.2
    )
}

/// Inverse of `shard_filename`. Returns `(lod, (sx, sy, sz))` for a
/// recognized shard filename, `None` for anything else (including
/// `.stale.*` siblings and the `chunks.idx` sidecar). Used by the
/// startup orphan-byte sweep to enumerate shard files without
/// pre-loading them.
fn parse_shard_filename(name: &str) -> Option<(u8, ShardCoord)> {
    let stripped = name.strip_prefix("chunks-L")?.strip_suffix(".dat")?;
    let mut parts = stripped.split('-');
    let lod: u8 = parts.next()?.parse().ok()?;
    let sx: u32 = parts.next()?.strip_prefix('X')?.parse().ok()?;
    let sy: u32 = parts.next()?.strip_prefix('Y')?.parse().ok()?;
    let sz: u32 = parts.next()?.strip_prefix('Z')?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((lod, (sx, sy, sz)))
}

fn stale_rename_everything(root: &Path) {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        let is_data = name_str.starts_with("chunks-L") && name_str.ends_with(".dat");
        let is_sidecar = name_str == "chunks.idx";
        if !is_data && !is_sidecar {
            continue;
        }
        let from = entry.path();
        let to = root.join(format!("{}.stale.{}", name_str, ts));
        if let Err(e) = std::fs::rename(&from, &to) {
            log::warn!(
                "[cache] failed to rename {} → {}: {}",
                from.display(),
                to.display(),
                e
            );
        }
    }
}

fn pwrite_all(file: &File, off: u64, mut buf: &[u8]) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        let mut off = off;
        while !buf.is_empty() {
            let n = file.write_at(buf, off)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "write_at returned 0",
                ));
            }
            buf = &buf[n..];
            off += n as u64;
        }
        Ok(())
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut off = off;
        while !buf.is_empty() {
            let n = file.seek_write(buf, off)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "seek_write returned 0",
                ));
            }
            buf = &buf[n..];
            off += n as u64;
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, off, buf);
        compile_error!("sparse cache store requires unix or windows");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "vesuvius-cache-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn make_chunk(seed: u8) -> Vec<u8> {
        let mut v = vec![0u8; CHUNK_VOXELS];
        for (i, b) in v.iter_mut().enumerate() {
            *b = ((i as u8) ^ seed).wrapping_mul(31);
        }
        v
    }

    #[test]
    fn write_then_load_roundtrip() {
        let tmp = tempdir();
        let store = DiskStore::new(&tmp, "v".into(), [128, 128, 128], 0);
        let key = ChunkKey::new(0, 1, 0, 1);
        assert!(matches!(store.load(key), LoadOutcome::Missing));

        let bytes = make_chunk(0x42);
        store.write_atomic(key, &bytes).unwrap();
        let (mmap, off) = store.try_load(key).expect("resident after write");
        assert_eq!(&mmap[off..off + 16], &bytes[..16]);
        assert_eq!(mmap[off + CHUNK_VOXELS - 1], bytes[CHUNK_VOXELS - 1]);
    }

    #[test]
    fn empty_sentinel_persisted_across_reopen() {
        let tmp = tempdir();
        let key = ChunkKey::new(0, 0, 0, 0);
        {
            let store = DiskStore::new(&tmp, "v".into(), [64, 64, 64], 0);
            store.mark_empty(key).unwrap();
            // Drop runs final sync via background thread.
        }
        let store = DiskStore::new(&tmp, "v".into(), [64, 64, 64], 0);
        assert!(matches!(store.load(key), LoadOutcome::Empty));
    }

    #[test]
    fn multi_lod_offsets_disjoint() {
        let tmp = tempdir();
        let store = DiskStore::new(&tmp, "v".into(), [256, 256, 256], 2);
        // Write distinguishable chunks at every LOD, at (0,0,0).
        let mut written = Vec::new();
        for lod in 0..=2u8 {
            let bytes = make_chunk(lod ^ 0xa5);
            store.write_atomic(ChunkKey::new(lod, 0, 0, 0), &bytes).unwrap();
            written.push(bytes);
        }
        for lod in 0..=2u8 {
            let (mmap, off) = store.try_load(ChunkKey::new(lod, 0, 0, 0)).unwrap();
            assert_eq!(&mmap[off..off + 32], &written[lod as usize][..32]);
        }
        assert!(tmp.join("chunks-L00-X000-Y000-Z000.dat").exists());
        assert!(tmp.join("chunks-L01-X000-Y000-Z000.dat").exists());
        assert!(tmp.join("chunks-L02-X000-Y000-Z000.dat").exists());
    }

    #[test]
    fn unused_lod_creates_no_file() {
        let tmp = tempdir();
        let store = DiskStore::new(&tmp, "v".into(), [256, 256, 256], 2);
        store.write_atomic(ChunkKey::new(0, 0, 0, 0), &make_chunk(1)).unwrap();
        assert!(tmp.join("chunks-L00-X000-Y000-Z000.dat").exists());
        assert!(!tmp.join("chunks-L01-X000-Y000-Z000.dat").exists());
        assert!(!tmp.join("chunks-L02-X000-Y000-Z000.dat").exists());
    }

    #[test]
    fn extent_mismatch_rebuilds_files() {
        let tmp = tempdir();
        {
            let store = DiskStore::new(&tmp, "v".into(), [128, 128, 128], 0);
            store.write_atomic(ChunkKey::new(0, 0, 0, 0), &make_chunk(7)).unwrap();
        }
        let store = DiskStore::new(&tmp, "v".into(), [256, 128, 128], 0);
        assert!(matches!(store.load(ChunkKey::new(0, 0, 0, 0)), LoadOutcome::Missing));
        let any_stale = std::fs::read_dir(&tmp)
            .unwrap()
            .flatten()
            .any(|e| e.file_name().to_string_lossy().contains(".stale."));
        assert!(any_stale, "expected a .stale.* file in {}", tmp.display());
    }

    #[test]
    fn rejects_truncated_data_file() {
        let tmp = tempdir();
        {
            let store = DiskStore::new(&tmp, "v".into(), [128, 128, 128], 0);
            store.write_atomic(ChunkKey::new(0, 0, 0, 0), &make_chunk(1)).unwrap();
        }
        let data = tmp.join("chunks-L00-X000-Y000-Z000.dat");
        std::fs::OpenOptions::new()
            .write(true)
            .open(&data)
            .unwrap()
            .set_len(128)
            .unwrap();
        let store = DiskStore::new(&tmp, "v".into(), [128, 128, 128], 0);
        assert!(matches!(store.load(ChunkKey::new(0, 0, 0, 0)), LoadOutcome::Missing));
    }

    #[test]
    fn concurrent_writers_distinct_chunks() {
        // At LOD 0 a [2048, 64, 1024] extent gives 32 × 1 × 16 chunks —
        // enough room for the (t, 0, i) keys below.
        let tmp = tempdir();
        let store = Arc::new(DiskStore::new(&tmp, "v".into(), [2048, 64, 1024], 0));
        let mut handles = Vec::new();
        for t in 0..4u8 {
            let store = store.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..16u32 {
                    let key = ChunkKey::new(0, t as u32, 0, i);
                    let mut bytes = make_chunk(t);
                    bytes[0] = i as u8;
                    store.write_atomic(key, &bytes).unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        for t in 0..4u8 {
            for i in 0..16u32 {
                let key = ChunkKey::new(0, t as u32, 0, i);
                let (mmap, off) = store.try_load(key).unwrap();
                assert_eq!(mmap[off], i as u8, "{}", key);
                assert_eq!(mmap[off + 1], (1u8 ^ t).wrapping_mul(31), "{}", key);
            }
        }
    }

    /// Stress test for the per-slot CAS protocol: a writer pool races a
    /// purger over the same key set. Invariant: any slot whose sidecar
    /// reports RESIDENT at end-of-test has its sentinel bytes intact —
    /// no permanent "sidecar RESIDENT + shard punched-to-zero" state.
    ///
    /// Without the CAS path the purger could punch a slot between a
    /// writer's pwrite and the writer's `set_state(RESIDENT)`, leaving
    /// the sidecar claiming Resident while the bytes were zeroed. With
    /// the CAS path that interleaving is structurally impossible.
    ///
    /// `#[ignore]`'d by default because the busy writers + purger
    /// briefly saturate the cores and can starve other timing-sensitive
    /// tests in the same `cargo test` parallel run. Run with:
    ///   cargo test -p vesuvius-rs --lib cache::disk::tests::write_atomic_vs_purge -- --ignored
    #[test]
    #[ignore]
    fn write_atomic_vs_purge_no_resident_zero_drift() {
        use std::sync::atomic::AtomicBool;
        use std::time::{Duration, Instant};

        const NUM_SLOTS: u32 = 32;
        const SENTINEL_HEAD: u8 = 0xa5;
        const SENTINEL_TAIL: u8 = 0x5a;

        let tmp = tempdir();
        // Extent picked so the slot range [0..NUM_SLOTS] all fit at LOD 0
        // in a single shard (avoids the small-shard test path here).
        let store = Arc::new(DiskStore::new(&tmp, "v".into(), [4096, 64, 64], 0));

        // Compute the LOD-0 dims once so the purger can map x → sidecar
        // idx without going through the private resolve().
        let lod0 = store.sidecar().header.lods[0];
        let nx = lod0.nx as u64;
        let ny = lod0.ny as u64;
        let _ = ny; // (z=0, y=0): idx is just x

        let stop = Arc::new(AtomicBool::new(false));
        let mut handles: Vec<std::thread::JoinHandle<()>> = Vec::new();

        // Writer threads: re-fill the slot range with sentinel bytes.
        // `thread::yield_now` between iterations keeps us from starving
        // concurrently-running unit tests of CPU.
        for _ in 0..2 {
            let store = store.clone();
            let stop = stop.clone();
            handles.push(std::thread::spawn(move || {
                let mut bytes = vec![0u8; CHUNK_VOXELS];
                bytes[0] = SENTINEL_HEAD;
                bytes[CHUNK_VOXELS - 1] = SENTINEL_TAIL;
                while !stop.load(Ordering::Relaxed) {
                    for x in 0..NUM_SLOTS {
                        let _ = store.write_atomic(ChunkKey::new(0, x, 0, 0), &bytes);
                    }
                    std::thread::yield_now();
                }
            }));
        }

        // Purger thread: imitate cache.rs::run_purge per-slot — CAS
        // RESIDENT→LOCKED, punch, set MISSING. Failures (slot wasn't
        // RESIDENT when we tried) are expected and just skipped.
        {
            let store = store.clone();
            let stop = stop.clone();
            handles.push(std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    for x in 0..NUM_SLOTS {
                        let key = ChunkKey::new(0, x, 0, 0);
                        let idx: u64 = x as u64; // (z=0, y=0) at LOD 0
                        let sidecar = store.sidecar();
                        if sidecar
                            .compare_exchange_state(0, idx, STATE_RESIDENT, STATE_LOCKED)
                            .is_ok()
                        {
                            // Replicate the cache-side ordering: punch
                            // under the lock, then release with
                            // MISSING.
                            let _ = store.punch_hole(key);
                            sidecar.set_state(0, idx, STATE_MISSING);
                            let _ = nx;
                        }
                    }
                    std::thread::yield_now();
                }
            }));
        }

        let deadline = Instant::now() + Duration::from_millis(150);
        while Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(25));
        }
        stop.store(true, Ordering::Relaxed);
        for h in handles {
            h.join().unwrap();
        }

        // Invariant: any RESIDENT slot must read back our sentinel.
        let sidecar = store.sidecar();
        let mut checked = 0usize;
        for x in 0..NUM_SLOTS {
            let key = ChunkKey::new(0, x, 0, 0);
            let idx: u64 = x as u64;
            if sidecar.get_state(0, idx) != STATE_RESIDENT {
                continue;
            }
            let (mmap, off) = store
                .try_load(key)
                .expect("RESIDENT slot must be loadable");
            assert_eq!(
                mmap[off], SENTINEL_HEAD,
                "slot {} sidecar=RESIDENT but head byte = {:#x} (race left punched bytes)",
                x, mmap[off]
            );
            assert_eq!(
                mmap[off + CHUNK_VOXELS - 1],
                SENTINEL_TAIL,
                "slot {} sidecar=RESIDENT but tail byte = {:#x}",
                x,
                mmap[off + CHUNK_VOXELS - 1]
            );
            checked += 1;
        }
        // Sanity: the test is meaningful only if some slots ended up
        // RESIDENT. With 4 writers vs 1 purger the writers should
        // almost always win some races.
        assert!(checked > 0, "no RESIDENT slots at end — test inconclusive");
    }

    /// Plant STATE_LOCKED bytes into the sidecar (simulating a crash
    /// mid-`write_atomic`), reopen the store, and confirm the startup
    /// sweep demoted them back to MISSING — and that a fresh writer
    /// can claim the slot without spinning on the stale lock.
    #[test]
    fn startup_sweep_clears_locked_slots() {
        let tmp = tempdir();
        {
            let store = DiskStore::new(&tmp, "v".into(), [128, 128, 128], 0);
            // Write one chunk so the shard file gets created — recovery
            // for that LOCKED slot will hit an already-open shard.
            store.write_atomic(ChunkKey::new(0, 0, 0, 0), &make_chunk(1)).unwrap();
            // Plant LOCKED bytes at two slots: one whose shard exists,
            // one in the same shard that's never been written.
            let sidecar = store.sidecar();
            let dims = sidecar.header.lods[0];
            let idx_a = dims.linear_index(1, 0, 0).unwrap();
            let idx_b = dims.linear_index(0, 1, 0).unwrap();
            sidecar.set_state(0, idx_a, STATE_LOCKED);
            sidecar.set_state(0, idx_b, STATE_LOCKED);
            store.flush();
        }
        let store = DiskStore::new(&tmp, "v".into(), [128, 128, 128], 0);
        let sidecar = store.sidecar();
        let dims = sidecar.header.lods[0];
        let idx_a = dims.linear_index(1, 0, 0).unwrap();
        let idx_b = dims.linear_index(0, 1, 0).unwrap();
        assert_eq!(
            sidecar.get_state(0, idx_a),
            STATE_MISSING,
            "slot A should have been swept LOCKED→MISSING"
        );
        assert_eq!(sidecar.get_state(0, idx_b), STATE_MISSING);
        // The pre-existing RESIDENT slot must be untouched.
        let idx_resident = dims.linear_index(0, 0, 0).unwrap();
        assert_eq!(sidecar.get_state(0, idx_resident), STATE_RESIDENT);
        // A fresh write to the recovered slot must succeed (no deadlock
        // spinning on the cleared LOCKED).
        let bytes = make_chunk(0x77);
        let outcome = store.write_atomic(ChunkKey::new(0, 1, 0, 0), &bytes).unwrap();
        assert_eq!(outcome, WriteOutcome::Wrote);
        let (mmap, off) = store.try_load(ChunkKey::new(0, 1, 0, 0)).unwrap();
        assert_eq!(mmap[off], bytes[0]);
    }

    /// Plant orphan bytes (pwrite into the shard outside the sidecar's
    /// RESIDENT slots), reopen the store, and confirm the startup
    /// orphan sweep punched them while leaving the RESIDENT slot intact.
    #[cfg(unix)]
    #[test]
    fn startup_sweep_reclaims_orphan_bytes() {
        use std::os::unix::fs::MetadataExt;
        let tmp = tempdir();
        let shard_file = tmp.join("chunks-L00-X000-Y000-Z000.dat");
        {
            // sca=4 → 4³ = 64 slots per shard. Extent [256, 64, 64] at
            // LOD 0 gives 4×1×1 chunks, all in shard (0,0,0).
            let store = DiskStore::new_with_shard_chunks_per_axis(
                &tmp, "v".into(), [256, 64, 64], 0, 4,
            );
            store.write_atomic(ChunkKey::new(0, 0, 0, 0), &make_chunk(1)).unwrap();
            // Pwrite junk into slot 1 and slot 2 without touching the
            // sidecar — those slots stay MISSING but burn physical blocks
            // (the orphan-bytes case `write_unconfirmed` and crashed
            // writers produce in real workloads).
            let f = OpenOptions::new().read(true).write(true).open(&shard_file).unwrap();
            let junk = vec![0xee_u8; CHUNK_VOXELS];
            pwrite_all(&f, CHUNK_VOXELS as u64, &junk).unwrap();
            pwrite_all(&f, 2 * CHUNK_VOXELS as u64, &junk).unwrap();
            drop(f);
            store.flush();
        }

        let physical_before = std::fs::metadata(&shard_file).unwrap().blocks() * 512;
        assert!(
            physical_before >= 3 * CHUNK_VOXELS as u64,
            "expected ≥ 3 chunks of physical blocks before sweep, got {}",
            physical_before,
        );

        let store = DiskStore::new_with_shard_chunks_per_axis(
            &tmp, "v".into(), [256, 64, 64], 0, 4,
        );

        // RESIDENT slot is untouched.
        let bytes = make_chunk(1);
        let (mmap, off) = store.try_load(ChunkKey::new(0, 0, 0, 0)).unwrap();
        assert_eq!(mmap[off], bytes[0]);
        assert_eq!(mmap[off + CHUNK_VOXELS - 1], bytes[CHUNK_VOXELS - 1]);

        // Sidecar state of the orphan slots is unchanged (still MISSING) —
        // the sweep only releases physical blocks.
        let sidecar = store.sidecar();
        let dims = sidecar.header.lods[0];
        assert_eq!(
            sidecar.get_state(0, dims.linear_index(1, 0, 0).unwrap()),
            STATE_MISSING,
        );
        assert_eq!(
            sidecar.get_state(0, dims.linear_index(2, 0, 0).unwrap()),
            STATE_MISSING,
        );

        let physical_after = std::fs::metadata(&shard_file).unwrap().blocks() * 512;
        assert!(
            physical_after < physical_before,
            "physical bytes did not shrink: {} → {}",
            physical_before, physical_after,
        );
        // The RESIDENT slot's chunk still occupies real blocks.
        assert!(
            physical_after >= CHUNK_VOXELS as u64,
            "RESIDENT chunk's bytes were also punched: {} bytes left",
            physical_after,
        );
    }

    /// A clean cache (no orphans, no LOCKED slots) should not trigger
    /// any punch_hole syscalls on startup — the sweep is a no-op below
    /// its slop threshold. We can't directly observe syscalls in a unit
    /// test, but we can verify physical size is unchanged.
    #[cfg(unix)]
    #[test]
    fn startup_sweep_leaves_clean_shards_alone() {
        use std::os::unix::fs::MetadataExt;
        let tmp = tempdir();
        let shard_file = tmp.join("chunks-L00-X000-Y000-Z000.dat");
        {
            let store = DiskStore::new_with_shard_chunks_per_axis(
                &tmp, "v".into(), [256, 64, 64], 0, 4,
            );
            for cx in 0..4u32 {
                store.write_atomic(ChunkKey::new(0, cx, 0, 0), &make_chunk(cx as u8)).unwrap();
            }
            store.flush();
        }
        let physical_before = std::fs::metadata(&shard_file).unwrap().blocks() * 512;
        let _store = DiskStore::new_with_shard_chunks_per_axis(
            &tmp, "v".into(), [256, 64, 64], 0, 4,
        );
        let physical_after = std::fs::metadata(&shard_file).unwrap().blocks() * 512;
        assert_eq!(
            physical_before, physical_after,
            "clean shard's physical size changed across reopen",
        );
    }

    #[test]
    fn writes_across_shard_boundary() {
        // Shard side of 4 chunks → 4³ × 256 KiB = 16 MiB per shard file.
        // Extent [1024, 64, 64] gives 16 × 1 × 1 chunks at LOD 0, which
        // spans 4 shards along X.
        let tmp = tempdir();
        let store = DiskStore::new_with_shard_chunks_per_axis(&tmp, "v".into(), [1024, 64, 64], 0, 4);
        for cx in 0..16u32 {
            let mut bytes = make_chunk(cx as u8);
            bytes[0] = cx as u8;
            bytes[1] = (cx as u8).wrapping_add(0x80);
            store.write_atomic(ChunkKey::new(0, cx, 0, 0), &bytes).unwrap();
        }
        for cx in 0..16u32 {
            let (mmap, off) = store.try_load(ChunkKey::new(0, cx, 0, 0)).unwrap();
            assert_eq!(mmap[off], cx as u8, "chunk {}", cx);
            assert_eq!(mmap[off + 1], (cx as u8).wrapping_add(0x80), "chunk {}", cx);
        }
        for sx in 0..4u32 {
            let p = tmp.join(format!("chunks-L00-X{:03}-Y000-Z000.dat", sx));
            assert!(p.exists(), "missing shard {}: {}", sx, p.display());
        }
        assert!(!tmp.join("chunks-L00-X004-Y000-Z000.dat").exists());
    }

}
