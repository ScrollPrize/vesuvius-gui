//! Per-LOD chunk-state bitmap, persisted to a small `chunks.idx` sidecar.
//!
//! For each LOD level the volume has a fixed number of 64³ chunk slots
//! (`nx*ny*nz`). The sidecar stores one byte per slot — `0` Missing, `1`
//! Resident, `2` Empty — alongside a header that records the extent, LOD
//! count, and per-LOD dimensions so the offset math can be reconstructed on
//! reopen.
//!
//! The sidecar file is `mmap(MAP_SHARED, PROT_READ|PROT_WRITE)`'d once at
//! open. State transitions write directly into the mapping via atomic
//! byte ops; the kernel's writeback flushes dirty pages on its own
//! schedule (typically every few seconds), so even an externally killed
//! process (`SIGKILL`) leaves a mostly-up-to-date sidecar on disk. The
//! periodic `do_sync` watchdog still fsyncs the data files for crash
//! ordering and calls `msync` to bound the dirty window, and the on-exit
//! `UnifiedCache::shutdown` path calls `flush()` to force a synchronous
//! msync before the process leaves.
//!
//! On-disk layout (unchanged from the previous Vec-backed format, so
//! existing caches load without migration): `header_bytes | bitmap LOD 0
//! | bitmap LOD 1 | … | access_epoch LOD 0 | access_epoch LOD 1 | …`.
//! Each per-LOD column is `nx * ny * nz` contiguous bytes.

use memmap::{MmapMut, MmapOptions};
use std::fs::OpenOptions;
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

pub const MAGIC: &[u8; 8] = b"VCSPRS01";
pub const STATE_MISSING: u8 = 0;
pub const STATE_RESIDENT: u8 = 1;
pub const STATE_EMPTY: u8 = 2;
/// Transient "operation in flight on this slot" marker. Acts as a
/// per-slot mutex: every state transition is a `compare_exchange` from a
/// specific predecessor to `STATE_LOCKED`, the disk op runs while we
/// hold the lock, then a plain `store` of the destination value
/// releases it. Other contenders observing LOCKED spin until released
/// (the critical sections — `pwrite_all` of one 256 KiB chunk,
/// `fallocate(PUNCH_HOLE)` of one chunk — are µs-scale on local disk).
///
/// LOCKED on a persisted sidecar (process died mid-op) signals
/// "presumed compromised": the startup sweep punches the slot and
/// demotes it to MISSING. The reader fast path treats LOCKED as
/// not-readable (same as MISSING).
pub const STATE_LOCKED: u8 = 3;

#[derive(Clone, Copy, Debug)]
pub struct LodDims {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
}

impl LodDims {
    pub fn for_lod(extent: [u32; 3], lod: u8) -> Self {
        let span = 64u32 << lod;
        Self {
            nx: div_ceil_u32(extent[0], span),
            ny: div_ceil_u32(extent[1], span),
            nz: div_ceil_u32(extent[2], span),
        }
    }

    pub fn count(&self) -> u64 {
        self.nx as u64 * self.ny as u64 * self.nz as u64
    }

    pub fn linear_index(&self, x: u32, y: u32, z: u32) -> Option<u64> {
        if x >= self.nx || y >= self.ny || z >= self.nz {
            return None;
        }
        Some((z as u64 * self.ny as u64 + y as u64) * self.nx as u64 + x as u64)
    }
}

#[derive(Debug, Clone)]
pub struct Header {
    pub volume_id: String,
    pub chunk_side: u8,
    pub max_lod: u8,
    pub extent: [u32; 3],
    pub lods: Vec<LodDims>,
}

impl Header {
    pub fn new(volume_id: String, extent: [u32; 3], max_lod: u8) -> Self {
        let lods = (0..=max_lod).map(|l| LodDims::for_lod(extent, l)).collect();
        Self {
            volume_id,
            chunk_side: 64,
            max_lod,
            extent,
            lods,
        }
    }

    /// True if this header describes the same volume layout as `other`.
    /// Mismatch triggers a hard rebuild of all data files.
    pub fn matches(&self, other: &Header) -> bool {
        self.volume_id == other.volume_id
            && self.chunk_side == other.chunk_side
            && self.max_lod == other.max_lod
            && self.extent == other.extent
    }

    fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(MAGIC);
        let vid = self.volume_id.as_bytes();
        out.extend_from_slice(&(vid.len() as u16).to_le_bytes());
        out.extend_from_slice(vid);
        out.push(self.chunk_side);
        out.push(self.max_lod);
        for v in self.extent {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for lod in &self.lods {
            out.extend_from_slice(&lod.nx.to_le_bytes());
            out.extend_from_slice(&lod.ny.to_le_bytes());
            out.extend_from_slice(&lod.nz.to_le_bytes());
        }
    }

    /// Byte length of `write_to`'s output. Used by `Sidecar` to compute
    /// the offset where bitmap data starts in the mmap.
    fn serialized_size(&self) -> usize {
        8 + 2 + self.volume_id.len() + 1 + 1 + 12 + 12 * self.lods.len()
    }

    fn read_from<R: Read>(r: &mut R) -> std::io::Result<Self> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad sidecar magic: {:?}", magic),
            ));
        }
        let mut len_buf = [0u8; 2];
        r.read_exact(&mut len_buf)?;
        let vid_len = u16::from_le_bytes(len_buf) as usize;
        let mut vid = vec![0u8; vid_len];
        r.read_exact(&mut vid)?;
        let volume_id = String::from_utf8(vid)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut byte = [0u8; 1];
        r.read_exact(&mut byte)?;
        let chunk_side = byte[0];
        r.read_exact(&mut byte)?;
        let max_lod = byte[0];
        let mut u32_buf = [0u8; 4];
        let mut extent = [0u32; 3];
        for slot in &mut extent {
            r.read_exact(&mut u32_buf)?;
            *slot = u32::from_le_bytes(u32_buf);
        }
        let mut lods = Vec::with_capacity(max_lod as usize + 1);
        for _ in 0..=max_lod {
            r.read_exact(&mut u32_buf)?;
            let nx = u32::from_le_bytes(u32_buf);
            r.read_exact(&mut u32_buf)?;
            let ny = u32::from_le_bytes(u32_buf);
            r.read_exact(&mut u32_buf)?;
            let nz = u32::from_le_bytes(u32_buf);
            lods.push(LodDims { nx, ny, nz });
        }
        Ok(Self {
            volume_id,
            chunk_side,
            max_lod,
            extent,
            lods,
        })
    }
}

/// Mmap-backed chunk-state index. The bitmap + access-epoch bytes live
/// in a `MmapMut` (anonymous for `Sidecar::empty`, file-backed for
/// `Sidecar::open_or_create`). State transitions are atomic byte ops on
/// `&AtomicU8` slices computed by `bytes_at` from per-LOD byte ranges
/// into the mapping.
///
/// Durability model: writes hit the mmap (and therefore the kernel page
/// cache) immediately. The kernel flushes dirty pages on its own
/// writeback schedule, so even an external `SIGKILL` typically leaves
/// most recent transitions on disk. The watchdog `do_sync` calls
/// `flush()` periodically (msync) to bound the dirty window and to
/// fsync data files first for the "RESIDENT implies durable bytes"
/// ordering invariant. `UnifiedCache::shutdown` calls `flush()` for a
/// synchronous final msync on graceful exit.
///
/// `pending` is the per-LOD count of transitions since the last
/// watchdog tick. It is in-memory only (never persisted) and used to
/// (a) drive wake-up of the sync watchdog and (b) decide which LODs'
/// data files actually need fsync before the next msync.
pub struct Sidecar {
    pub header: Header,
    mmap: MmapMut,
    /// `bitmap_ranges[lod]` is the half-open byte range of LOD `lod`'s
    /// state bitmap inside `mmap`.
    bitmap_ranges: Vec<Range<usize>>,
    /// `ae_ranges[lod]` is the half-open byte range of LOD `lod`'s
    /// access-epoch column inside `mmap`.
    ae_ranges: Vec<Range<usize>>,
    pending: Vec<AtomicU64>,
}

impl Sidecar {
    /// In-memory (anonymous mmap) sidecar with every chunk Missing.
    /// Used by tests that don't want to touch disk.
    pub fn empty(header: Header) -> Self {
        let layout = SidecarLayout::for_header(&header);
        let mmap = MmapOptions::new()
            .len(layout.total_size)
            .map_anon()
            .expect("anon mmap for sidecar");
        let mut s = Self::from_mapped(header, mmap, layout);
        // Anon mmaps are zero-filled by the kernel; STATE_MISSING == 0 and
        // access_epoch default == 0 so we don't need to initialize bytes.
        // We do still write the header into the prefix so any future
        // `flush_header` / debugging dump sees a coherent file.
        s.write_header_prefix();
        s
    }

    /// Open or create the on-disk sidecar at `path` with the given
    /// expected layout. If the file exists and its header matches, the
    /// existing state is preserved (we just remap it). If it's shorter
    /// than the expected layout (old format with no access-epoch
    /// column, or a fresh `set_len(0)`), the file is extended; new
    /// bytes are zero-filled by the OS (STATE_MISSING / epoch 0). If
    /// the file's header does NOT match, returns `Ok(None)` so the
    /// caller can rename-aside and retry; returns `Err` on real I/O
    /// failures.
    pub fn open_or_create(path: &Path, expected: Header) -> std::io::Result<Option<Self>> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        let on_disk_len = file.metadata()?.len();
        let layout = SidecarLayout::for_header(&expected);
        let needed = layout.total_size as u64;

        if on_disk_len == 0 {
            // Brand-new (or zero-length) file: size it, write the
            // header prefix, mmap it.
            file.set_len(needed)?;
            let mmap = unsafe { MmapOptions::new().len(layout.total_size).map_mut(&file)? };
            let mut s = Self::from_mapped(expected, mmap, layout);
            s.write_header_prefix();
            return Ok(Some(s));
        }

        // File exists. Validate the header against `expected` before we
        // do anything destructive. Header read is short — we don't mmap
        // for this; we use a fresh File handle to get a Read cursor at
        // the start.
        let mut hdr_file = std::fs::File::open(path)?;
        let on_disk_header = Header::read_from(&mut hdr_file)?;
        if !on_disk_header.matches(&expected) {
            return Ok(None);
        }

        // Header matches. Make sure the file is big enough to hold both
        // bitmaps and the access-epoch column; if not, extend (zero-fill).
        if on_disk_len < needed {
            file.set_len(needed)?;
        }
        let mmap = unsafe { MmapOptions::new().len(layout.total_size).map_mut(&file)? };
        let s = Self::from_mapped(expected, mmap, layout);
        Ok(Some(s))
    }

    /// Load an existing sidecar from `path`. Returns `Ok(None)` if the
    /// file doesn't exist (lets callers treat absent and present
    /// uniformly). Returns the loaded sidecar without doing the
    /// header-match check — callers (`DiskStore::new`) compare
    /// `loaded.header.matches(expected)` and rebuild on mismatch.
    ///
    /// On old-format files (no access-epoch column), the file is
    /// extended (zero-filled) to the full layout size, so on-disk
    /// access epochs default to 0 — the LRU treats those chunks as
    /// "oldest possible" and evicts them first, which is the desired
    /// migration behavior.
    pub fn load(path: &Path) -> std::io::Result<Option<Self>> {
        let file = match OpenOptions::new().read(true).write(true).open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };

        let mut hdr_file = std::fs::File::open(path)?;
        let header = Header::read_from(&mut hdr_file)?;
        let layout = SidecarLayout::for_header(&header);
        let needed = layout.total_size as u64;
        let on_disk_len = file.metadata()?.len();
        if on_disk_len < needed {
            file.set_len(needed)?;
        }
        let mmap = unsafe { MmapOptions::new().len(layout.total_size).map_mut(&file)? };
        Ok(Some(Self::from_mapped(header, mmap, layout)))
    }

    fn from_mapped(header: Header, mmap: MmapMut, layout: SidecarLayout) -> Self {
        let pending = (0..header.lods.len()).map(|_| AtomicU64::new(0)).collect();
        Self {
            header,
            mmap,
            bitmap_ranges: layout.bitmap_ranges,
            ae_ranges: layout.ae_ranges,
            pending,
        }
    }

    fn write_header_prefix(&mut self) {
        let mut buf = Vec::with_capacity(self.header.serialized_size());
        self.header.write_to(&mut buf);
        debug_assert_eq!(buf.len(), self.header.serialized_size());
        self.mmap[..buf.len()].copy_from_slice(&buf);
    }

    /// View `range` as a slice of `&AtomicU8`. Sound because `AtomicU8`
    /// has the same in-memory representation as `u8` (per std docs).
    /// Concurrent shared `&AtomicU8` slices into the same mmap region
    /// are the expected access pattern.
    #[inline]
    fn atoms_at(&self, range: &Range<usize>) -> &[AtomicU8] {
        let bytes = &self.mmap[range.start..range.end];
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<AtomicU8>(), bytes.len()) }
    }

    pub fn get_state(&self, lod: u8, idx: u64) -> u8 {
        self.atoms_at(&self.bitmap_ranges[lod as usize])[idx as usize].load(Ordering::Acquire)
    }

    /// Publish a new state for `idx` and bump the per-LOD pending counter.
    /// Returns the previous state (for callers that want to detect duplicate
    /// writes; current callers ignore).
    pub fn set_state(&self, lod: u8, idx: u64, state: u8) -> u8 {
        let prev = self.atoms_at(&self.bitmap_ranges[lod as usize])[idx as usize]
            .swap(state, Ordering::Release);
        if prev != state {
            self.pending[lod as usize].fetch_add(1, Ordering::Relaxed);
        }
        prev
    }

    /// CAS the state byte at `(lod, idx)` from `current` to `new`. Returns
    /// `Ok(current)` on success or `Err(observed)` on failure (the caller
    /// gets the actual current value to drive its retry / skip logic).
    /// AcqRel on success pairs with `Acquire` loads on the reader fast
    /// path; Acquire on failure ensures the failed CAS doesn't reorder
    /// later loads. Bumps the pending counter only on success.
    pub fn compare_exchange_state(&self, lod: u8, idx: u64, current: u8, new: u8) -> Result<u8, u8> {
        let res = self.atoms_at(&self.bitmap_ranges[lod as usize])[idx as usize].compare_exchange(
            current,
            new,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        if res.is_ok() && current != new {
            self.pending[lod as usize].fetch_add(1, Ordering::Relaxed);
        }
        res
    }

    /// Reset the per-LOD pending counters and return their prior
    /// values. Used by the sync watchdog to decide which LODs' data
    /// files need fsync before the next msync.
    pub fn take_pending(&self) -> Vec<u64> {
        self.pending
            .iter()
            .map(|c| c.swap(0, Ordering::AcqRel))
            .collect()
    }

    /// Total transitions across all LODs since the last `take_pending`.
    pub fn total_pending(&self) -> u64 {
        self.pending.iter().map(|c| c.load(Ordering::Relaxed)).sum()
    }

    /// Synchronously flush dirty mmap pages to disk (msync(MS_SYNC)).
    /// No-op for anonymous mmaps (memmap returns Ok). Callers should
    /// `fsync` the underlying data files BEFORE invoking this so the
    /// invariant "any RESIDENT slot in the persisted sidecar has its
    /// bytes already durable" holds across a crash.
    pub fn flush(&self) -> std::io::Result<()> {
        self.mmap.flush()
    }

    /// Asynchronously hint to the kernel to schedule writeback of dirty
    /// pages (msync(MS_ASYNC)). Returns immediately; durability is not
    /// guaranteed on return. Used as a low-cost periodic nudge.
    #[allow(dead_code)]
    pub fn flush_async(&self) -> std::io::Result<()> {
        self.mmap.flush_async()
    }

    /// Read the access epoch tagged on `(lod, idx)`. Returns 0 for slots
    /// that have never been touched (or were reloaded from an older
    /// sidecar format).
    pub fn get_access_epoch(&self, lod: u8, idx: u64) -> u8 {
        self.atoms_at(&self.ae_ranges[lod as usize])[idx as usize].load(Ordering::Relaxed)
    }

    /// Stamp `(lod, idx)` with `epoch`. Called by the cache on transitions
    /// (the read fast path's `!=` filter ensures this is only called when
    /// the value would actually change).
    pub fn set_access_epoch(&self, lod: u8, idx: u64, epoch: u8) {
        self.atoms_at(&self.ae_ranges[lod as usize])[idx as usize].store(epoch, Ordering::Relaxed);
    }

    /// CAS the access-epoch byte at `(lod, idx)` from `current` to `new`.
    /// Returns `Ok(current)` on success or `Err(observed)` on failure so
    /// the caller can decide whether to retry or bail.
    ///
    /// Used by `touch_access` to arbitrate concurrent LRU bumps: only the
    /// thread whose CAS wins is allowed to adjust the epoch histogram, so
    /// N concurrent touchers on the same slot produce exactly one bucket
    /// transition instead of N. Relaxed ordering matches the rest of the
    /// access-epoch column — it's pure LRU bookkeeping, no read of other
    /// state piggybacks on this. Bumps the pending counter on success so
    /// the watchdog persists the new tag.
    pub fn compare_exchange_access_epoch(
        &self,
        lod: u8,
        idx: u64,
        current: u8,
        new: u8,
    ) -> Result<u8, u8> {
        let res = self.atoms_at(&self.ae_ranges[lod as usize])[idx as usize].compare_exchange(
            current,
            new,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        if res.is_ok() && current != new {
            self.pending[lod as usize].fetch_add(1, Ordering::Relaxed);
        }
        res
    }
}

/// Precomputed byte offsets for the body of a sidecar mmap, given a
/// header. `header_size` is the byte length of `Header::write_to`;
/// bitmap and access-epoch columns follow contiguously per-LOD.
struct SidecarLayout {
    total_size: usize,
    bitmap_ranges: Vec<Range<usize>>,
    ae_ranges: Vec<Range<usize>>,
}

impl SidecarLayout {
    fn for_header(header: &Header) -> Self {
        let header_size = header.serialized_size();
        let mut cursor = header_size;
        let mut bitmap_ranges = Vec::with_capacity(header.lods.len());
        for d in &header.lods {
            let n = d.count() as usize;
            bitmap_ranges.push(cursor..cursor + n);
            cursor += n;
        }
        let mut ae_ranges = Vec::with_capacity(header.lods.len());
        for d in &header.lods {
            let n = d.count() as usize;
            ae_ranges.push(cursor..cursor + n);
            cursor += n;
        }
        Self {
            total_size: cursor,
            bitmap_ranges,
            ae_ranges,
        }
    }
}

fn div_ceil_u32(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

/// Helper: derive the sidecar file path from the cache root.
pub fn sidecar_path(root: &Path) -> PathBuf {
    root.join("chunks.idx")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lod_dims_match_extent() {
        let d = LodDims::for_lod([128, 128, 128], 0);
        assert_eq!((d.nx, d.ny, d.nz), (2, 2, 2));
        let d1 = LodDims::for_lod([128, 128, 128], 1);
        assert_eq!((d1.nx, d1.ny, d1.nz), (1, 1, 1));
        let d_odd = LodDims::for_lod([65, 130, 200], 0);
        assert_eq!((d_odd.nx, d_odd.ny, d_odd.nz), (2, 3, 4));
    }

    #[test]
    fn header_roundtrip() {
        let h = Header::new("vol-x".into(), [256, 384, 512], 3);
        let mut buf = Vec::new();
        h.write_to(&mut buf);
        let mut r = std::io::Cursor::new(&buf);
        let back = Header::read_from(&mut r).unwrap();
        assert!(h.matches(&back));
        assert_eq!(back.lods.len(), 4);
    }

    #[test]
    fn take_pending_resets_counter() {
        let h = Header::new("v".into(), [64, 64, 64], 0);
        let s = Sidecar::empty(h);
        s.set_state(0, 0, STATE_RESIDENT);
        assert_eq!(s.total_pending(), 1);
        let pending = s.take_pending();
        assert_eq!(pending[0], 1);
        assert_eq!(s.total_pending(), 0);
    }

    #[test]
    fn sidecar_file_roundtrip() {
        let dir = tempdir();
        let h = Header::new("vol".into(), [256, 128, 64], 1);
        let path = sidecar_path(&dir);
        {
            let s = Sidecar::open_or_create(&path, h)
                .unwrap()
                .expect("fresh file accepts header");
            s.set_state(0, 0, STATE_RESIDENT);
            s.set_state(0, 3, STATE_EMPTY);
            // msync so the next process-equivalent open via load() sees
            // the bytes (same-process opens already share the page cache,
            // but flush exercises the production path).
            s.flush().unwrap();
        }

        let expected = Header::new("vol".into(), [256, 128, 64], 1);
        let loaded = Sidecar::load(&path).unwrap().expect("file should exist");
        assert!(loaded.header.matches(&expected));
        assert_eq!(loaded.get_state(0, 0), STATE_RESIDENT);
        assert_eq!(loaded.get_state(0, 1), STATE_MISSING);
        assert_eq!(loaded.get_state(0, 3), STATE_EMPTY);
    }

    #[test]
    fn sidecar_access_epoch_roundtrip() {
        // 256³ at LOD 0 = 4*4*4 = 64 slots; plenty of room for several
        // tagged slots without going out of bounds.
        let dir = tempdir();
        let path = sidecar_path(&dir);
        let h = Header::new("vol".into(), [256, 256, 256], 1);
        {
            let s = Sidecar::open_or_create(&path, h)
                .unwrap()
                .expect("fresh file accepts header");
            s.set_state(0, 0, STATE_RESIDENT);
            s.set_access_epoch(0, 0, 42);
            s.set_state(0, 17, STATE_RESIDENT);
            s.set_access_epoch(0, 17, 199);
            s.set_state(1, 0, STATE_RESIDENT);
            s.set_access_epoch(1, 0, 7);
            s.flush().unwrap();
        }

        let loaded = Sidecar::load(&path).unwrap().expect("file should exist");
        assert_eq!(loaded.get_access_epoch(0, 0), 42);
        assert_eq!(loaded.get_access_epoch(0, 17), 199);
        assert_eq!(loaded.get_access_epoch(0, 1), 0);
        assert_eq!(loaded.get_access_epoch(1, 0), 7);
    }

    #[test]
    fn sidecar_old_format_loads_with_zero_access_epochs() {
        // Simulate an old sidecar file: header + state bitmaps only,
        // no trailing access-epoch column. load() must extend the file
        // (zero-fill) and present access epochs as all-zero.
        let dir = tempdir();
        let h = Header::new("vol".into(), [128, 128, 64], 0);
        let mut buf = Vec::new();
        h.write_to(&mut buf);
        // One LOD, dims 2*2*1 = 4 slots.
        buf.extend_from_slice(&[STATE_RESIDENT, STATE_MISSING, STATE_RESIDENT, STATE_EMPTY]);
        let path = sidecar_path(&dir);
        std::fs::write(&path, &buf).unwrap();

        let loaded = Sidecar::load(&path).unwrap().expect("file should exist");
        assert_eq!(loaded.get_state(0, 0), STATE_RESIDENT);
        assert_eq!(loaded.get_state(0, 2), STATE_RESIDENT);
        // Access epochs all default to 0 — old format had no column,
        // file was extended (zero-fill) on load.
        for i in 0..4 {
            assert_eq!(loaded.get_access_epoch(0, i), 0);
        }
    }

    fn tempdir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "vesuvius-sidecar-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
