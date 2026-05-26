//! `UnifiedVolume` — single `VoxelVolume` + `PaintVolume` impl over a
//! `ChunkCache`. The only place every backfiller-backed volume needs.
//!
//! ## LOD fallback strategy
//!
//! `paint()` and `get()` both walk the LOD pyramid: if the requested
//! ("target") LOD chunk isn't resident yet, we sample from the first
//! coarser-LOD parent chunk that is. The target tile's screen region is
//! painted **once** at whichever LOD has data — we never overdraw multiple
//! LODs into the same region. Pre-dispatch in `paint()` happens coarse-first
//! so the wider, viewport-covering parent chunks land in the LIFO queue
//! before their finer-LOD children pile on top: a low-res preview shows
//! up promptly while detail streams in.
//!
//! `get()` is the per-voxel sampler used by surface (ObjVolume) and PPM
//! renderers that don't go through `UnifiedVolume::paint`. It can't rely on
//! a prior pre-dispatch pass, so on a target-LOD miss it dispatches every
//! coarser LOD along the same column and picks the first resident one. The
//! per-volume hot slot caches that chosen `(lod, chunk_state)` keyed by the
//! target-LOD chunk so subsequent pixels in the same target chunk skip the
//! pyramid walk entirely.
//!
//! ## Coordinate conventions
//!
//! * `paint()` takes `xyz: [i32; 3]` in **world voxel coords** (LOD 0). The
//!   per-LOD chunk lookup divides by `64 * (1 << lod)` to land on the right
//!   chunk and the rendering loop knows how to step samples per pixel.
//! * `get()` follows the `VoxelVolume::get` convention used everywhere else
//!   in the codebase (see `VolumeGrid64x4Mapped`, `ZarrContext`, and
//!   `PPMVolume`/`ObjVolume` callers): `xyz` is in **voxel coords at the
//!   requested downsampling** — the caller has already divided world coords
//!   by `downsampling`. This is the convention surface painting depends on,
//!   so do NOT redivide here.

use super::cache::ChunkCache;
use super::disk::{ChunkBitState, ChunkStateBits, ShardCoord};
use super::state::{ChunkKey, ChunkState};
use super::CHUNK_VOXELS;
use crate::volume::composition::{CompositionState, CompositorRef, MaxCompositionState};
use crate::volume::{DrawingConfig, Image, PaintVolume, VolumeCons, VoxelPaintVolume, VoxelVolume};
use ecolor::Color32;
use libm::modf;
use memmap::Mmap;
use std::cell::RefCell;
use std::sync::Arc;

/// Overlay alpha for `debug_chunk_overlay`. Low enough that voxel detail is
/// still readable underneath.
const OVERLAY_ALPHA: f32 = 0.35;

/// Number of shard hot slots, one per LOD level. 16 covers any conceivable
/// pyramid depth — at 64³ chunks the deepest is ≈ 8 (2⁸ · 64 = 16384 base
/// voxels per coarsest chunk). Indexing by `lod as usize` skips a slot
/// match key compare in the per-voxel fast path.
const SHARD_SLOTS_PER_LOD: usize = 16;

pub struct UnifiedVolume {
    cache: ChunkCache,
    /// Side length (in chunks) of one shard cube — cached from
    /// `ChunkCache::shard_chunks_per_axis` at construction so the per-voxel
    /// shard-coord derivation is a couple of integer ops with no method
    /// call. Production value is 128.
    shard_chunks_per_axis: u32,
    // Per-volume hot slot for the last target chunk touched by `get`. The
    // cached `chosen` may be a coarser-LOD parent when the target chunk
    // wasn't resident on first hit. `paint()` and external callers clear it
    // between frames via `reset_for_painting`. `Volume` clones produce fresh
    // `UnifiedVolume` instances with their own slot via the `shared()`
    // constructor, so the `RefCell` is never shared across threads.
    local: RefCell<LocalSlot>,
}

struct LocalSlot {
    /// One shard-grain fast path *per LOD*. Indexed by `lod as usize`.
    /// The interpolation lattice path in `interpolate_u8` recurses into
    /// `get` at the *chosen* LOD (which may be coarser than the original
    /// target LOD), so a single shared slot would thrash between e.g.
    /// LOD 0 and a coarser-parent LOD as the +1 corner crosses a chunk
    /// boundary. Per-LOD slots keep both alive simultaneously without
    /// cross-eviction. The bitmap on `ShardSlot` gates reads so unwritten
    /// chunks fall back to the slow path instead of returning the kernel's
    /// zero page as if it were data.
    shards: [Option<ShardSlot>; SHARD_SLOTS_PER_LOD],
    /// Chunk-grain slot for the slow path's LOD-fallback result. Lets
    /// repeat samples in the same target chunk skip the pyramid walk
    /// while a resolved coarser parent is the best we have.
    target_key: Option<ChunkKey>,
    chosen: Option<(u8, Arc<ChunkState>)>,
}

impl Default for LocalSlot {
    fn default() -> Self {
        Self {
            shards: [const { None }; SHARD_SLOTS_PER_LOD],
            target_key: None,
            chosen: None,
        }
    }
}

struct ShardSlot {
    shard: ShardCoord,
    base: *const u8,
    _mmap: Arc<Mmap>,
    /// Per-chunk 2-bit state map for this shard. Probed before every
    /// shard-slot read so we can distinguish Resident (fast mmap deref)
    /// from Unknown / Dispatched (fall back to LOD climb) and Empty
    /// (return 0 without climbing).
    state_bits: Arc<ChunkStateBits>,
}

/// Outcome of a per-voxel probe against the shard hot slot.
enum ShardSlotProbe {
    /// Chunk is resident in the cached shard — fast-path the read.
    Resident { base: *const u8, in_shard_idx: u64 },
    /// Chunk is definitively empty at this LOD — return 0, do not climb.
    Empty,
    /// Slot matched the shard but the chunk is Unknown or Dispatched —
    /// caller must fall through to the slow path so LOD climb can serve a
    /// coarser parent while the target chunk loads.
    NotPresent,
    /// Slot was empty or addressed a different shard.
    SlotMiss,
}

impl UnifiedVolume {
    pub fn new(cache: ChunkCache) -> Self {
        let shard_chunks_per_axis = cache.shard_chunks_per_axis();
        Self {
            cache,
            shard_chunks_per_axis,
            local: RefCell::new(LocalSlot::default()),
        }
    }

    pub fn cache(&self) -> &ChunkCache {
        &self.cache
    }

    fn drop_hot_slot(&self) {
        let mut b = self.local.borrow_mut();
        for s in b.shards.iter_mut() {
            *s = None;
        }
        b.target_key = None;
        b.chosen = None;
    }

    /// Decompose a target-LOD chunk coord into `(shard_coord,
    /// in_shard_chunk_idx)`. The byte offset of voxel `(vx, vy, vz)` inside
    /// the shard mmap is then
    /// `in_shard_chunk_idx * CHUNK_VOXELS + ((vz & 63) * 64 * 64 + (vy & 63) * 64 + (vx & 63))`.
    #[inline]
    fn shard_decompose(&self, cx: u32, cy: u32, cz: u32) -> (ShardCoord, u64) {
        let sca = self.shard_chunks_per_axis;
        let shard = (cx / sca, cy / sca, cz / sca);
        let wx = (cx % sca) as u64;
        let wy = (cy % sca) as u64;
        let wz = (cz % sca) as u64;
        let s = sca as u64;
        let in_shard_idx = (wz * s + wy) * s + wx;
        (shard, in_shard_idx)
    }

    /// Outcome of a shard-slot lookup. `Resident` is the fast-path return;
    /// `Empty` says "no data here, don't climb"; `NotPresent` means the
    /// caller must fall through to the slow path (LOD climb).
    #[inline]
    fn shard_slot_chunk_state(
        &self,
        target_lod: u8,
        cx: u32,
        cy: u32,
        cz: u32,
    ) -> ShardSlotProbe {
        let lod_ix = target_lod as usize;
        if lod_ix >= SHARD_SLOTS_PER_LOD {
            return ShardSlotProbe::SlotMiss;
        }
        let (shard, in_shard_idx) = self.shard_decompose(cx, cy, cz);
        let b = self.local.borrow();
        let Some(slot) = b.shards[lod_ix].as_ref() else {
            return ShardSlotProbe::SlotMiss;
        };
        if slot.shard != shard {
            return ShardSlotProbe::SlotMiss;
        }
        match slot.state_bits.load(in_shard_idx) {
            ChunkBitState::Resident => ShardSlotProbe::Resident { base: slot.base, in_shard_idx },
            ChunkBitState::Empty => ShardSlotProbe::Empty,
            ChunkBitState::Unknown | ChunkBitState::Dispatched => ShardSlotProbe::NotPresent,
        }
    }

    /// Borrow the resident chunk's 64³ slice from the shard hot slot, if
    /// the slot addresses the same `(lod, shard)` that contains
    /// `(target_cx, target_cy, target_cz)` AND the chunk is marked
    /// Resident. Returns the slice base pointer; `None` for any other
    /// state (caller must slow-path).
    #[inline]
    fn shard_slot_chunk_slice(&self, target_lod: u8, target_cx: u32, target_cy: u32, target_cz: u32) -> Option<*const u8> {
        match self.shard_slot_chunk_state(target_lod, target_cx, target_cy, target_cz) {
            ShardSlotProbe::Resident { base, in_shard_idx } => {
                // SAFETY: in_shard_idx * CHUNK_VOXELS + CHUNK_VOXELS ≤
                // shard mmap length.
                Some(unsafe { base.add((in_shard_idx as usize) * CHUNK_VOXELS) })
            }
            _ => None,
        }
    }

    /// Populate the shard hot slot for `(target_lod, shard)`, opening the
    /// shard file (sparse mmap + seeded bitmap) if it isn't already. After
    /// this call, the slot's bitmap is the single source of truth for
    /// every chunk inside this shard — the per-voxel slow path never
    /// needs to re-enter the DashMap or the per-LOD `opened` mutex.
    ///
    /// No-op if the slot already addresses this shard.
    fn populate_shard_slot(&self, target_lod: u8, shard: ShardCoord) {
        let lod_ix = target_lod as usize;
        if lod_ix >= SHARD_SLOTS_PER_LOD {
            return;
        }
        {
            let b = self.local.borrow();
            if let Some(slot) = &b.shards[lod_ix] {
                if slot.shard == shard {
                    return;
                }
            }
        }
        if let Some(snap) = self.cache.ensure_shard_open(target_lod, shard) {
            let base = snap.mmap.as_ptr();
            let mut b = self.local.borrow_mut();
            b.shards[lod_ix] = Some(ShardSlot {
                shard,
                base,
                _mmap: snap.mmap,
                state_bits: snap.state_bits,
            });
            drop(b);
            // The chunks we're about to read through this shard slot
            // bypass `state_or_fetch`, so without an explicit signal
            // their access epochs would stay stale and purge would
            // consider them eligible for eviction. Stamp the shard
            // instead.
            self.cache.touch_shard_access(target_lod, shard);
        }
    }

}

fn lod_for(sfactor: u8) -> u8 {
    // sfactor is expected to be a power of two: 1, 2, 4, 8, …
    (sfactor as u32).max(1).trailing_zeros() as u8
}

impl VoxelVolume for UnifiedVolume {
    fn reset_for_painting(&self) {
        self.drop_hot_slot();
        // Tick the cache's frame counter so the per-Pending touch
        // debounce in `state_or_fetch` lets each chunk through once
        // during this paint instead of once per ~16 ms wall-clock.
        self.cache.advance_frame();
    }

    fn touch_aabb(&self, min: [f64; 3], max: [f64; 3], downsampling: i32) {
        let target_lod = lod_for(downsampling.max(1) as u8);
        self.cache.touch_aabb(min, max, target_lod);
    }

    fn get(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        // Per the `VoxelVolume::get` convention shared with VolumeGrid64x4Mapped,
        // ZarrContext, and PPMVolume callers (notably `ObjVolume::paint`, which
        // passes `[x / sfactor, y / sfactor, z / sfactor]`): `xyz` is in
        // voxel-coords at the requested downsampling. We do NOT re-divide by
        // scale here.
        let target_lod = lod_for(downsampling.max(1) as u8);
        let max_lod = self.cache.max_lod();

        let target_sx = (xyz[0] as i64).max(0) as u64;
        let target_sy = (xyz[1] as i64).max(0) as u64;
        let target_sz = (xyz[2] as i64).max(0) as u64;
        let target_cx = (target_sx / 64) as u32;
        let target_cy = (target_sy / 64) as u32;
        let target_cz = (target_sz / 64) as u32;

        match self.resolve_chunk(target_lod, max_lod, target_cx, target_cy, target_cz) {
            Some(bound) if bound.shift == 0 => {
                let mmap = unsafe { bound.mmap_slice() };
                let off = ((target_sz & 63) as usize) * 64 * 64
                    + ((target_sy & 63) as usize) * 64
                    + (target_sx & 63) as usize;
                mmap[off]
            }
            Some(bound) => {
                // Sampled via a coarser parent — re-derive the in-chunk
                // offset in the parent's coord space.
                let lod_use = target_lod + bound.shift;
                let mmap = unsafe { bound.mmap_slice() };
                sample_at(target_sx, target_sy, target_sz, target_lod, lod_use, mmap)
            }
            None => 0,
        }
    }

    fn get_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        self.interpolate_u8(xyz, downsampling)
    }

    fn get_color_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> Color32 {
        Color32::from_gray(self.interpolate_u8(xyz, downsampling))
    }

    /// Fast-path override of the trait default. Amortizes the per-sample
    /// chunk lookup that `get_interpolated` redoes from scratch each call,
    /// then walks the ray as Q32.32 fixed-point with Q0.8 trilerp inside
    /// each chunk. Boundary samples (4.7% at random directions) use an
    /// inline 2-chunk path; 2- and 3-axis chunk corners (~0.07% / 0.0004%)
    /// fall back to `interpolate_u8`.
    ///
    /// The unswitch on `CompositorRef` is what buys per-sample
    /// monomorphization: each arm calls a separate instantiation of the
    /// generic `composite_along_normal_inner` so `state.update(v)` folds
    /// into the inner loop with zero virtual dispatch.
    fn composite_along_normal(
        &self,
        base: [f64; 3],
        dir: [f64; 3],
        w_lo: f64,
        w_hi: f64,
        downsampling: i32,
        compositor: &mut CompositorRef<'_>,
        climb_lod: bool,
    ) {
        match compositor {
            CompositorRef::Max(s) => {
                self.composite_along_normal_inner(base, dir, w_lo, w_hi, downsampling, climb_lod, |v| s.update(v))
            }
            CompositorRef::Alpha(s) => {
                self.composite_along_normal_inner(base, dir, w_lo, w_hi, downsampling, climb_lod, |v| s.update(v))
            }
            CompositorRef::HeightMap(s) => {
                self.composite_along_normal_inner(base, dir, w_lo, w_hi, downsampling, climb_lod, |v| s.update(v))
            }
            CompositorRef::None(s) => {
                self.composite_along_normal_inner(base, dir, w_lo, w_hi, downsampling, climb_lod, |v| s.update(v))
            }
        }
    }
}

impl UnifiedVolume {
    /// Shard-based Q32.32 + Q0.8 ray walker. Generic over the per-sample
    /// sink so each `CompositorRef` arm gets its own monomorphization —
    /// `sink(v)` inlines to the concrete `CompositionState::update` body,
    /// no virtual dispatch in the hot loop.
    ///
    /// The inner loop is intentionally chunk-state-unaware: it reads
    /// straight off the target-LOD shard mmap with no atomic probes, no
    /// LOD climb, no slow-mode fallback. Pre-dispatch (see
    /// `touch_aabb`, called per-triangle by `ObjVolume::paint`) lands the
    /// chunks the ray crosses before this is called; un-arrived bytes
    /// read as zero from the sparse mmap, and the optional one-level
    /// upscale-from-parent path in `touch_aabb` fills target chunks with
    /// a preview while the real bytes stream in. `_climb_lod` is kept on
    /// the trait method for compatibility and ignored here.
    fn composite_along_normal_inner<F: FnMut(u8) -> bool>(
        &self,
        base: [f64; 3],
        dir: [f64; 3],
        w_lo: f64,
        w_hi: f64,
        downsampling: i32,
        _climb_lod: bool,
        mut sink: F,
    ) {
        let target_lod = lod_for(downsampling.max(1) as u8);
        let n_total = (w_hi - w_lo) as i32;
        if n_total <= 0 {
            return;
        }

        let dx = dir[0];
        let dy = dir[1];
        let dz = dir[2];

        let mut px = base[0] + w_lo * dx;
        let mut py = base[1] + w_lo * dy;
        let mut pz = base[2] + w_lo * dz;

        let mut remaining = n_total as usize;
        let sca = self.shard_chunks_per_axis;

        while remaining > 0 {
            if px < 0.0 || py < 0.0 || pz < 0.0 {
                if !sink(0) {
                    return;
                }
                px += dx;
                py += dy;
                pz += dz;
                remaining -= 1;
                continue;
            }
            let txu = px as u64;
            let tyu = py as u64;
            let tzu = pz as u64;
            let target_cx = (txu / 64) as u32;
            let target_cy = (tyu / 64) as u32;
            let target_cz = (tzu / 64) as u32;
            let (shard, in_shard_idx) = self.shard_decompose(target_cx, target_cy, target_cz);
            self.populate_shard_slot(target_lod, shard);

            // Borrow the slot just long enough to clone pointers + the
            // mmap Arc. After this block we never re-enter `self.local`
            // until the next outer iteration, so the raw pointers remain
            // valid.
            let (slot_base, _mmap_keepalive, slot_shard) = {
                let b = self.local.borrow();
                match b.shards[target_lod as usize].as_ref() {
                    Some(slot) => (slot.base, slot._mmap.clone(), slot.shard),
                    None => {
                        // Shard couldn't be opened (I/O error); feed 0
                        // and advance.
                        if !sink(0) {
                            return;
                        }
                        px += dx;
                        py += dy;
                        pz += dz;
                        remaining -= 1;
                        continue;
                    }
                }
            };
            if slot_shard != shard {
                if !sink(0) {
                    return;
                }
                px += dx;
                py += dy;
                pz += dz;
                remaining -= 1;
                continue;
            }
            let chunk_base = unsafe { slot_base.add((in_shard_idx as usize) * CHUNK_VOXELS) };

            let bx = (txu & 63) == 63;
            let by = (tyu & 63) == 63;
            let bz = (tzu & 63) == 63;
            let n_boundary = bx as u8 + by as u8 + bz as u8;

            if n_boundary >= 2 {
                // 2/3-axis +1 corner crossings need samples from up to 7
                // neighbor chunks across possibly multiple shards. Rare
                // (~0.07% / 0.0004%); defer to the legacy trilerp path.
                let v = self.interpolate_u8([px, py, pz], downsampling);
                if !sink(v) {
                    return;
                }
                px += dx;
                py += dy;
                pz += dz;
                remaining -= 1;
                continue;
            }
            if n_boundary == 1 {
                let (nx_c, ny_c, nz_c) = if bx {
                    (target_cx + 1, target_cy, target_cz)
                } else if by {
                    (target_cx, target_cy + 1, target_cz)
                } else {
                    (target_cx, target_cy, target_cz + 1)
                };
                // Same-shard neighbor: read direct from the shard mmap
                // (kernel zero if unwritten, real bytes / upscale fill
                // if dispatched). Cross-shard +1 corner: we can't
                // address the neighbor without evicting our home slot,
                // so treat the neighbor as 0 — at most ~4.7% of samples
                // × 1/sca chance per axis ≈ 0.04% pixels affected.
                let same_shard = nx_c / sca == shard.0 && ny_c / sca == shard.1 && nz_c / sca == shard.2;
                let home_slice = unsafe { std::slice::from_raw_parts(chunk_base, CHUNK_VOXELS) };
                let v = if same_shard {
                    let nwx = (nx_c % sca) as u64;
                    let nwy = (ny_c % sca) as u64;
                    let nwz = (nz_c % sca) as u64;
                    let s = sca as u64;
                    let neighbor_idx = (nwz * s + nwy) * s + nwx;
                    let neighbor_base =
                        unsafe { slot_base.add((neighbor_idx as usize) * CHUNK_VOXELS) };
                    let neighbor_slice = unsafe { std::slice::from_raw_parts(neighbor_base, CHUNK_VOXELS) };
                    sample_boundary_1axis(home_slice, Some(neighbor_slice), bx, by, bz, txu, tyu, tzu, px, py, pz)
                } else {
                    sample_boundary_1axis(home_slice, None, bx, by, bz, txu, tyu, tzu, px, py, pz)
                };
                if !sink(v) {
                    return;
                }
                px += dx;
                py += dy;
                pz += dz;
                remaining -= 1;
                continue;
            }

            // Pure in-chunk run-length trilerp from the shard mmap.
            let lo_x = (target_cx as f64) * 64.0;
            let lo_y = (target_cy as f64) * 64.0;
            let lo_z = (target_cz as f64) * 64.0;
            let hi_x = lo_x + 63.0;
            let hi_y = lo_y + 63.0;
            let hi_z = lo_z + 63.0;
            let kx = run_length_1d(px, dx, lo_x, hi_x);
            let ky = run_length_1d(py, dy, lo_y, hi_y);
            let kz = run_length_1d(pz, dz, lo_z, hi_z);
            let run = kx.min(ky).min(kz).min(remaining);
            debug_assert!(run >= 1);

            let mmap = unsafe { std::slice::from_raw_parts(chunk_base, CHUNK_VOXELS) };
            const Q32_F: f64 = 4294967296.0; // 2^32
            let mut posx = unsafe { (px * Q32_F).to_int_unchecked::<i64>() };
            let mut posy = unsafe { (py * Q32_F).to_int_unchecked::<i64>() };
            let mut posz = unsafe { (pz * Q32_F).to_int_unchecked::<i64>() };
            let dx_q = unsafe { (dx * Q32_F).to_int_unchecked::<i64>() };
            let dy_q = unsafe { (dy * Q32_F).to_int_unchecked::<i64>() };
            let dz_q = unsafe { (dz * Q32_F).to_int_unchecked::<i64>() };

            let mut stopped = false;
            let mut consumed = 0usize;
            unsafe {
                for _ in 0..run {
                    let cx = (posx >> 32) as u64;
                    let cy = (posy >> 32) as u64;
                    let cz = (posz >> 32) as u64;
                    let tx = (cx & 63) as usize;
                    let ty = (cy & 63) as usize;
                    let tz = (cz & 63) as usize;
                    let idx = tz * 64 * 64 + ty * 64 + tx;
                    let p000 = *mmap.get_unchecked(idx);
                    let p100 = *mmap.get_unchecked(idx + 1);
                    let p010 = *mmap.get_unchecked(idx + 64);
                    let p110 = *mmap.get_unchecked(idx + 65);
                    let p001 = *mmap.get_unchecked(idx + 64 * 64);
                    let p101 = *mmap.get_unchecked(idx + 64 * 64 + 1);
                    let p011 = *mmap.get_unchecked(idx + 64 * 64 + 64);
                    let p111 = *mmap.get_unchecked(idx + 64 * 64 + 65);
                    let fx_q8 = ((posx >> 24) & 0xFF) as u32;
                    let fy_q8 = ((posy >> 24) & 0xFF) as u32;
                    let fz_q8 = ((posz >> 24) & 0xFF) as u32;
                    let v = trilerp_q8(
                        fx_q8,
                        fy_q8,
                        fz_q8,
                        [p000, p100, p010, p110, p001, p101, p011, p111],
                    );
                    if !sink(v) {
                        stopped = true;
                        break;
                    }
                    consumed += 1;
                    posx = posx.wrapping_add(dx_q);
                    posy = posy.wrapping_add(dy_q);
                    posz = posz.wrapping_add(dz_q);
                }
            }
            const INV_Q32: f64 = 1.0 / 4294967296.0;
            px = posx as f64 * INV_Q32;
            py = posy as f64 * INV_Q32;
            pz = posz as f64 * INV_Q32;
            remaining -= consumed;
            if stopped {
                return;
            }
        }
    }
}

impl UnifiedVolume {
    /// Trilinear interpolation specialized over the cache's 64³ chunks.
    ///
    /// Mirrors the fast/slow split in
    /// `volume64x4::VolumeGrid64x4Mapped::get_interpolated` and
    /// `vesuvius_zarr::ZarrContext::get_interpolated`, with one extra
    /// wrinkle for the cache's LOD pyramid: when the target chunk isn't
    /// resident and a coarser parent is used instead, the **interpolation
    /// lattice shifts to the chosen LOD's coordinate space**. Otherwise
    /// `target_sx` and `target_sx + 1` both map onto the same coarse voxel
    /// (since `floor((target_sx + 1) / 2^shift) == floor(target_sx /
    /// 2^shift)` for most `target_sx`), the 8 corners collapse to one
    /// value, and the output bands instead of smoothly interpolating.
    fn interpolate_u8(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        let target_lod = lod_for(downsampling.max(1) as u8);
        let max_lod = self.cache.max_lod();

        let target_sx = (xyz[0] as i64).max(0) as u64;
        let target_sy = (xyz[1] as i64).max(0) as u64;
        let target_sz = (xyz[2] as i64).max(0) as u64;
        let target_cx = (target_sx / 64) as u32;
        let target_cy = (target_sy / 64) as u32;
        let target_cz = (target_sz / 64) as u32;
        let key_t = ChunkKey::new(target_lod, target_cx, target_cy, target_cz);

        // Shard-slot fast path. If the target shard is mmapped *and* this
        // chunk is marked Resident, do the trilerp straight against the
        // shard's bytes. Only the in-chunk fast case runs here — the
        // +1-crosses-chunk boundary path uses `get`, which itself
        // fast-paths through the shard slot.
        if let Some(chunk_ptr) = self.shard_slot_chunk_slice(target_lod, target_cx, target_cy, target_cz) {
            // `lod_use == target_lod` for the shard slot path.
            let (fx_frac, cx0_f) = modf(xyz[0]);
            let (fy_frac, cy0_f) = modf(xyz[1]);
            let (fz_frac, cz0_f) = modf(xyz[2]);
            let cx0 = (cx0_f as i64).max(0) as u64;
            let cy0 = (cy0_f as i64).max(0) as u64;
            let cz0 = (cz0_f as i64).max(0) as u64;
            let fast = (cx0 & 63) != 63 && (cy0 & 63) != 63 && (cz0 & 63) != 63;
            if fast {
                let tx = (cx0 & 63) as usize;
                let ty = (cy0 & 63) as usize;
                let tz = (cz0 & 63) as usize;
                let idx = tz * 64 * 64 + ty * 64 + tx;
                // SAFETY: chunk_ptr addresses a 64³ region inside the shard
                // mmap (see shard_slot_chunk_slice). idx + 65 + 64*65 ≤
                // CHUNK_VOXELS for any (tx,ty,tz) with each < 63.
                let ps: [u8; 8] = unsafe {
                    [
                        *chunk_ptr.add(idx),
                        *chunk_ptr.add(idx + 1),
                        *chunk_ptr.add(idx + 64),
                        *chunk_ptr.add(idx + 65),
                        *chunk_ptr.add(idx + 64 * 64),
                        *chunk_ptr.add(idx + 64 * 64 + 1),
                        *chunk_ptr.add(idx + 64 * 64 + 64),
                        *chunk_ptr.add(idx + 64 * 64 + 65),
                    ]
                };
                return trilerp_q8(frac_q8(fx_frac), frac_q8(fy_frac), frac_q8(fz_frac), ps);
            }
            // Fall through to the legacy path which handles the +1-corner
            // crossing into a neighbor chunk via recursive `get` calls.
        }

        // Resolve the LOD to interpolate at: reuse the chunk slot if it
        // still points at our target chunk, otherwise walk the pyramid.
        let chosen: Option<(u8, Arc<ChunkState>)> = {
            let b = self.local.borrow();
            if b.target_key == Some(key_t) {
                b.chosen.clone()
            } else {
                None
            }
        };
        let chosen = match chosen {
            Some(c) => c,
            None => {
                let walk_lo = target_lod.min(max_lod);
                let walk_hi = max_lod;
                let mut found: Option<(u8, Arc<ChunkState>)> = None;
                for lod_try in walk_lo..=walk_hi {
                    let (lx, ly, lz) = coord_at_lod(target_sx, target_sy, target_sz, target_lod, lod_try);
                    let key = ChunkKey::new(lod_try, (lx / 64) as u32, (ly / 64) as u32, (lz / 64) as u32);
                    let s = self.cache.state_or_fetch(key);
                    if s.is_terminal() {
                        found = Some((lod_try, s));
                        break;
                    }
                }
                // Populate the shard slot only when we landed on a
                // Resident chunk at target_lod — that's when we know the
                // shard is mmapped. Calling peek_shard on a full miss
                // would acquire the per-LOD mutex per voxel during
                // streaming, serializing the worker pool.
                if let Some((lt, s)) = &found {
                    if *lt == target_lod && matches!(s.as_ref(), ChunkState::Resident { .. }) {
                        let (target_shard, _) = self.shard_decompose(key_t.x, key_t.y, key_t.z);
                        self.populate_shard_slot(target_lod, target_shard);
                    }
                }
                match found {
                    Some(c) => c,
                    None => return 0,
                }
            }
        };
        let (lod_use, state) = chosen;

        // Compute the interpolation lattice in `lod_use` coordinates: dividing
        // the raw f64 inputs by `2^shift` gives fractional coarse-voxel
        // positions whose 8 surrounding corners are genuine neighbors in the
        // coarse mip, not duplicates of the same coarse voxel.
        let shift = lod_use - target_lod;
        let scale = (1u64 << shift) as f64;
        let (cdx, cx0_f) = modf(xyz[0] / scale);
        let (cdy, cy0_f) = modf(xyz[1] / scale);
        let (cdz, cz0_f) = modf(xyz[2] / scale);
        let cx0 = (cx0_f as i64).max(0) as u64;
        let cy0 = (cy0_f as i64).max(0) as u64;
        let cz0 = (cz0_f as i64).max(0) as u64;

        let result = match state.as_resident() {
            Some(mmap) => {
                // Fast path: the 2×2×2 block at the chosen LOD lives inside
                // the chunk we just resolved. `target_sx / (64 << shift) ==
                // cx0 / 64` always (and likewise for y/z), so the boundary
                // check is purely about `cx0 + 1` crossing into the next
                // coarse chunk.
                let fast = (cx0 & 63) != 63 && (cy0 & 63) != 63 && (cz0 & 63) != 63;
                let ps: [u8; 8] = if fast {
                    let tx = (cx0 & 63) as usize;
                    let ty = (cy0 & 63) as usize;
                    let tz = (cz0 & 63) as usize;
                    let idx = tz * 64 * 64 + ty * 64 + tx;
                    [
                        mmap[idx],
                        mmap[idx + 1],
                        mmap[idx + 64],
                        mmap[idx + 65],
                        mmap[idx + 64 * 64],
                        mmap[idx + 64 * 64 + 1],
                        mmap[idx + 64 * 64 + 64],
                        mmap[idx + 64 * 64 + 65],
                    ]
                } else {
                    // +1 corner crosses into a neighboring coarse chunk.
                    // Sample each corner via `get` *at the chosen LOD*
                    // (downsampling = 1 << lod_use), so the per-corner
                    // pyramid walks start at `lod_use` instead of redoing
                    // target-LOD lookups that would just collapse onto
                    // duplicate coarse voxels again.
                    let ds = 1i32 << lod_use;
                    let cx0f = cx0 as f64;
                    let cy0f = cy0 as f64;
                    let cz0f = cz0 as f64;
                    [
                        self.get([cx0f, cy0f, cz0f], ds),
                        self.get([cx0f + 1.0, cy0f, cz0f], ds),
                        self.get([cx0f, cy0f + 1.0, cz0f], ds),
                        self.get([cx0f + 1.0, cy0f + 1.0, cz0f], ds),
                        self.get([cx0f, cy0f, cz0f + 1.0], ds),
                        self.get([cx0f + 1.0, cy0f, cz0f + 1.0], ds),
                        self.get([cx0f, cy0f + 1.0, cz0f + 1.0], ds),
                        self.get([cx0f + 1.0, cy0f + 1.0, cz0f + 1.0], ds),
                    ]
                };
                trilerp_q8(frac_q8(cdx), frac_q8(cdy), frac_q8(cdz), ps)
            }
            None => 0, // Empty
        };

        // Re-anchor the hot slot to our (target chunk → chosen LOD) binding.
        // Slow-path `get` calls may have rewritten it for their own corner
        // chunks at `lod_use`, but the next sample in this target chunk
        // wants our mapping back.
        let mut b = self.local.borrow_mut();
        b.target_key = Some(key_t);
        b.chosen = Some((lod_use, state));
        result
    }

    /// Convenience wrapper around `composite_along_normal` that returns
    /// the per-sample max along the ray. Kept so the microbench can
    /// measure the realistic monomorphized fast-path cost of what
    /// `ObjVolume::paint` runs in practice.
    pub fn max_along_normal(&self, base: [f64; 3], dir: [f64; 3], w_lo: f64, w_hi: f64, downsampling: i32) -> u8 {
        let mut state = MaxCompositionState::new();
        let mut compositor = CompositorRef::Max(&mut state);
        // Drive the trait override so we exercise the same unswitch +
        // monomorphized inner-loop path callers reach via Volume. Climb
        // enabled so the bench measures the full pyramid-aware behavior.
        <Self as VoxelVolume>::composite_along_normal(
            self,
            base,
            dir,
            w_lo,
            w_hi,
            downsampling,
            &mut compositor,
            true,
        );
        state.result(0)
    }
}

/// How many integer-step advances `p, p+d, p+2d, …` stay strictly inside
/// `[lo, hi)`, counting the initial sample (which the caller has confirmed
/// is inside the range — i.e. `lo <= p < hi`).
///
/// For the in-chunk fast path: `lo = cx*64`, `hi = cx*64 + 63` (NOT 64 — the
/// `+1` corner of the trilerp must stay inside the chunk).
#[inline]
fn run_length_1d(p: f64, d: f64, lo: f64, hi: f64) -> usize {
    if d > 0.0 {
        // Stop one step before crossing `hi`. The largest k with
        // `p + k*d < hi` is `floor((hi - p - eps) / d)`. We use the
        // half-open `< hi` semantics directly with `floor`, taking care
        // that an exact landing on `hi` does NOT count.
        let q = (hi - p) / d;
        // `q` ≥ 0 since p < hi and d > 0 (caller guarantees boundary safety).
        let k = if q.fract() == 0.0 { q - 1.0 } else { q.floor() };
        if k < 0.0 {
            1
        } else {
            (k as usize).saturating_add(1)
        }
    } else if d < 0.0 {
        // Symmetric: stop one step before crossing `lo`. Largest k with
        // `p + k*d >= lo` is `floor((lo - p) / d)` (d<0 flips inequality).
        let q = (lo - p) / d;
        let k = q.floor();
        if k < 0.0 {
            1
        } else {
            (k as usize).saturating_add(1)
        }
    } else {
        usize::MAX
    }
}

/// One target-chunk → LOD-use binding cached across multiple samples on the
/// same ray. Holds an `Arc<Mmap>` so the underlying region stays alive; the
/// raw pointer + length pair lets the inner loop materialize `&[u8]`
/// without re-deref'ing the `Arc` per sample.
struct BoundChunk {
    /// `lod_use - target_lod`. Zero in the common case (target chunk
    /// resident at the requested LOD); positive when a coarser parent was
    /// used as a fallback. The run-length optimization only runs when
    /// `shift == 0` — coarser-LOD parents would require lattice math in
    /// `lod_use` coordinates, which the outer loop just defers to
    /// `interpolate_u8` one sample at a time.
    shift: u8,
    /// Owns the mmap that `mmap_ptr` indexes into.
    _mmap: Arc<Mmap>,
    mmap_ptr: *const u8,
    mmap_len: usize,
}

impl BoundChunk {
    /// Materialize the chunk's bytes as a slice. The borrow is tied to
    /// `&self`, and `_mmap` keeps the underlying mapping alive.
    unsafe fn mmap_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.mmap_ptr, self.mmap_len)
    }
}

impl UnifiedVolume {
    /// Shard-based LOD climb. After populating each LOD's hot shard slot,
    /// every per-voxel decision (resident / dispatched / empty / unknown)
    /// comes from the shard bitmap — a lock-free atomic read. The DashMap
    /// is only consulted on the first encounter with an `Unknown` chunk,
    /// to kick off its dispatch; subsequent voxels see `Dispatched` on the
    /// bitmap and skip the cache layer entirely.
    fn resolve_chunk(&self, target_lod: u8, max_lod: u8, cx: u32, cy: u32, cz: u32) -> Option<BoundChunk> {
        let walk_lo = target_lod.min(max_lod);
        for lod_try in walk_lo..=max_lod {
            let shift = lod_try - target_lod;
            let cx_try = cx >> shift;
            let cy_try = cy >> shift;
            let cz_try = cz >> shift;
            let (shard, in_shard_idx) = self.shard_decompose(cx_try, cy_try, cz_try);
            self.populate_shard_slot(lod_try, shard);

            let lod_ix = lod_try as usize;
            let probe = {
                let b = self.local.borrow();
                let Some(slot) = b.shards[lod_ix].as_ref() else {
                    continue;
                };
                if slot.shard != shard {
                    continue;
                }
                let state = slot.state_bits.load(in_shard_idx);
                let mmap_arc = slot._mmap.clone();
                let base = slot.base;
                (state, base, mmap_arc)
            };
            let (state, base, mmap_arc) = probe;
            match state {
                ChunkBitState::Resident => {
                    let mmap_ptr = unsafe { base.add((in_shard_idx as usize) * CHUNK_VOXELS) };
                    return Some(BoundChunk {
                        shift,
                        _mmap: mmap_arc,
                        mmap_ptr,
                        mmap_len: CHUNK_VOXELS,
                    });
                }
                ChunkBitState::Empty => return None,
                ChunkBitState::Dispatched => continue,
                ChunkBitState::Unknown => {
                    // First voxel to find this chunk Unknown kicks off the
                    // dispatch. `dispatch_chunk` flips the bitmap to
                    // Dispatched (or Resident / Empty if disk had it), so
                    // siblings of this voxel never re-enter the DashMap.
                    let key = ChunkKey::new(lod_try, cx_try, cy_try, cz_try);
                    let _ = self.cache.state_or_fetch(key);
                    continue;
                }
            }
        }
        None
    }
}

/// Sample at a position whose `floor()` lands on the +63 row of exactly
/// **one** chunk axis — the +1 trilerp corner along that axis crosses into
/// a neighbor chunk. 4 of the 8 corners come from `home`, the other 4 from
/// `neigh` (or are 0 if `neigh` is None, i.e. an Empty neighbor).
///
/// Exactly one of `bx`, `by`, `bz` must be true; the caller asserts this.
/// The 4 non-crossed corners are safe to read from `home` because the
/// other two axes' `+1` stay below 64 (their boundary bits are false).
#[inline]
fn sample_boundary_1axis(
    home: &[u8],
    neigh: Option<&[u8]>,
    bx: bool,
    by: bool,
    bz: bool,
    txu: u64,
    tyu: u64,
    tzu: u64,
    px: f64,
    py: f64,
    pz: f64,
) -> u8 {
    debug_assert!((bx as u8 + by as u8 + bz as u8) == 1);
    let tx = (txu & 63) as usize;
    let ty = (tyu & 63) as usize;
    let tz = (tzu & 63) as usize;

    // SAFETY: `tx + (1-bx)`, `ty + (1-by)`, `tz + (1-bz)` are all ≤ 63
    // for the home reads; the neighbor reads use 0 on the crossed axis.
    // All indices stay inside their respective 64³ mmap.
    let (h0, h1, h2, h3, n0, n1, n2, n3) = unsafe {
        if bx {
            // Crossed: +x. Home rows at tx=63; neighbor rows at tx=0.
            let h = tz * 64 * 64 + ty * 64 + 63;
            let n = tz * 64 * 64 + ty * 64;
            (
                *home.get_unchecked(h),                  // p000 home[63, ty, tz]
                *home.get_unchecked(h + 64),             // p010 home[63, ty+1, tz]
                *home.get_unchecked(h + 64 * 64),        // p001 home[63, ty, tz+1]
                *home.get_unchecked(h + 64 * 64 + 64),   // p011 home[63, ty+1, tz+1]
                neigh.map(|s| *s.get_unchecked(n)).unwrap_or(0),                  // p100
                neigh.map(|s| *s.get_unchecked(n + 64)).unwrap_or(0),             // p110
                neigh.map(|s| *s.get_unchecked(n + 64 * 64)).unwrap_or(0),        // p101
                neigh.map(|s| *s.get_unchecked(n + 64 * 64 + 64)).unwrap_or(0),   // p111
            )
        } else if by {
            // Crossed: +y. Home rows at ty=63; neighbor rows at ty=0.
            let h = tz * 64 * 64 + 63 * 64 + tx;
            let n = tz * 64 * 64 + tx;
            (
                *home.get_unchecked(h),                  // p000 home[tx, 63, tz]
                *home.get_unchecked(h + 1),              // p100 home[tx+1, 63, tz]
                *home.get_unchecked(h + 64 * 64),        // p001 home[tx, 63, tz+1]
                *home.get_unchecked(h + 64 * 64 + 1),    // p101 home[tx+1, 63, tz+1]
                neigh.map(|s| *s.get_unchecked(n)).unwrap_or(0),              // p010
                neigh.map(|s| *s.get_unchecked(n + 1)).unwrap_or(0),          // p110
                neigh.map(|s| *s.get_unchecked(n + 64 * 64)).unwrap_or(0),    // p011
                neigh.map(|s| *s.get_unchecked(n + 64 * 64 + 1)).unwrap_or(0),// p111
            )
        } else {
            // Crossed: +z. Home rows at tz=63; neighbor rows at tz=0.
            let h = 63 * 64 * 64 + ty * 64 + tx;
            let n = ty * 64 + tx;
            (
                *home.get_unchecked(h),                  // p000 home[tx, ty, 63]
                *home.get_unchecked(h + 1),              // p100 home[tx+1, ty, 63]
                *home.get_unchecked(h + 64),             // p010 home[tx, ty+1, 63]
                *home.get_unchecked(h + 65),             // p110 home[tx+1, ty+1, 63]
                neigh.map(|s| *s.get_unchecked(n)).unwrap_or(0),         // p001
                neigh.map(|s| *s.get_unchecked(n + 1)).unwrap_or(0),     // p101
                neigh.map(|s| *s.get_unchecked(n + 64)).unwrap_or(0),    // p011
                neigh.map(|s| *s.get_unchecked(n + 65)).unwrap_or(0),    // p111
            )
        }
    };

    // Re-order into the canonical (p000, p100, p010, p110, p001, p101, p011, p111).
    // Above we built ordered groups per axis; map back here.
    let p: [u8; 8] = if bx {
        [h0, n0, h1, n1, h2, n2, h3, n3]
    } else if by {
        [h0, h1, n0, n1, h2, h3, n2, n3]
    } else {
        [h0, h1, h2, h3, n0, n1, n2, n3]
    };

    let fx = px - txu as f64;
    let fy = py - tyu as f64;
    let fz = pz - tzu as f64;
    trilerp_q8(frac_q8(fx), frac_q8(fy), frac_q8(fz), p)
}

/// Trilinear blend of 8 u8 corner samples ordered (p000, p100, p010, p110,
/// p001, p101, p011, p111), where bits encode (z, y, x) offsets of 0/1.
///
/// Weights are Q0.8 (`0..=256` represents `0..=1.0`). Output is u8 with
/// truncate-toward-zero semantics — matches the previous f64 `c as u8` for
/// non-negative blends, which is everything we get from a u8 corner field.
///
/// All intermediates fit in u32: the deepest nest is
/// `c0 * (256 - fz) + c1 * fz ≤ 65280 * 256 * 256 = 0xFF000000`, which is
/// just under `2^32`.
#[inline]
fn trilerp_q8(fx: u32, fy: u32, fz: u32, p: [u8; 8]) -> u8 {
    let nfx = 256 - fx;
    let nfy = 256 - fy;
    let nfz = 256 - fz;
    let c00 = (p[0] as u32) * nfx + (p[1] as u32) * fx;
    let c10 = (p[2] as u32) * nfx + (p[3] as u32) * fx;
    let c01 = (p[4] as u32) * nfx + (p[5] as u32) * fx;
    let c11 = (p[6] as u32) * nfx + (p[7] as u32) * fx;
    let c0 = c00 * nfy + c10 * fy;
    let c1 = c01 * nfy + c11 * fy;
    let c = c0 * nfz + c1 * fz;
    (c >> 24) as u8
}

/// Convert a fractional `[0, 1)` weight into Q0.8 (`0..=255`). Negative
/// inputs saturate to 0; inputs ≥ 1.0 saturate to 256 (still valid for
/// `trilerp_q8`).
#[inline]
fn frac_q8(f: f64) -> u32 {
    (f * 256.0) as u32
}

/// Map a target-LOD voxel coord to its corresponding voxel coord at `lod_use`.
/// Shifts right when going coarser, left when going finer.
fn coord_at_lod(sx: u64, sy: u64, sz: u64, from_lod: u8, to_lod: u8) -> (u64, u64, u64) {
    if to_lod >= from_lod {
        let shift = to_lod - from_lod;
        (sx >> shift, sy >> shift, sz >> shift)
    } else {
        let shift = from_lod - to_lod;
        (sx << shift, sy << shift, sz << shift)
    }
}

/// Map the cache state of one target-LOD chunk (plus which LOD actually
/// rendered and whether an HTTP GET is in flight) to an overlay tint.
/// `None` means "ready at target LOD; no overlay needed".
///
/// Color legend:
/// - Green   — Empty (definitively absent, cached as a sentinel)
/// - Blue    — served from coarser LOD, waiting in the work queue
/// - Cyan    — served from coarser LOD, actively downloading right now
/// - Yellow  — Pending, no coarser fallback yet, waiting in queue
/// - Orange  — Pending, no coarser fallback yet, actively downloading
/// - Red     — CooldownMiss (recent fetch failed)
/// - Magenta — Missing / never dispatched
fn overlay_color_for(
    target_lod: u8,
    chosen_lod: Option<u8>,
    target_state: Option<&ChunkState>,
    is_downloading: bool,
) -> Option<Color32> {
    // Empty target wins over any LOD fallback — the chunk is definitively
    // absent. Use a vivid green so it doesn't blend with mid-gray voxel data
    // the way the previous neutral tint did.
    if matches!(target_state, Some(ChunkState::Empty)) {
        return Some(Color32::from_rgb(60, 200, 110)); // green
    }
    match (chosen_lod, target_state) {
        // Rendered at target LOD with real data — happy path, no tint.
        (Some(l), _) if l == target_lod => None,
        // Rendered via a coarser parent (target not ready). Cyan if the
        // target's bytes are actually moving over the wire right now, blue
        // if it's just sitting in the queue waiting for a worker.
        (Some(_), _) => {
            if is_downloading {
                Some(Color32::from_rgb(40, 200, 220)) // cyan
            } else {
                Some(Color32::from_rgb(60, 130, 230)) // blue
            }
        }
        // Nothing resident yet — inspect target state for finer signal.
        (None, Some(ChunkState::Pending { .. })) => {
            if is_downloading {
                Some(Color32::from_rgb(230, 140, 40)) // orange
            } else {
                Some(Color32::from_rgb(230, 200, 40)) // yellow
            }
        }
        (None, Some(ChunkState::CooldownMiss { .. })) => Some(Color32::from_rgb(220, 60, 60)), // red
        (None, Some(ChunkState::Missing)) | (None, None) => Some(Color32::from_rgb(220, 60, 220)), // magenta
        // Defensive: Resident / Empty here mean the LOD-walk produced no
        // chosen — shouldn't happen, but if it does, no overlay.
        (None, Some(ChunkState::Resident { .. } | ChunkState::Empty)) => None,
    }
}

fn sample_at(sx: u64, sy: u64, sz: u64, from_lod: u8, to_lod: u8, mmap: &[u8]) -> u8 {
    let (lx, ly, lz) = coord_at_lod(sx, sy, sz, from_lod, to_lod);
    let off = ((lz & 63) as usize) * 64 * 64 + ((ly & 63) as usize) * 64 + ((lx & 63) as usize);
    mmap[off]
}

/// Geometry describing the chunk grid at one LOD inside one viewport.
struct ViewportTiles {
    chunk_world: i32,
    tile_u_lo: i32,
    tile_u_hi: i32,
    tile_v_lo: i32,
    tile_v_hi: i32,
    tile_pc: i32,
}

fn viewport_tiles(min_uc: i32, max_uc: i32, min_vc: i32, max_vc: i32, pc: i32, lod: u8) -> Option<ViewportTiles> {
    let scale = 1i32 << lod;
    let chunk_world = 64 * scale;
    let tile_pc = pc.div_euclid(chunk_world);
    if tile_pc < 0 {
        return None;
    }
    Some(ViewportTiles {
        chunk_world,
        tile_u_lo: min_uc.div_euclid(chunk_world).max(0),
        tile_u_hi: max_uc.div_euclid(chunk_world),
        tile_v_lo: min_vc.div_euclid(chunk_world).max(0),
        tile_v_hi: max_vc.div_euclid(chunk_world),
        tile_pc,
    })
}

impl PaintVolume for UnifiedVolume {
    fn paint(
        &self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        canvas_width: usize,
        canvas_height: usize,
        sfactor: u8,
        paint_zoom: u8,
        config: &DrawingConfig,
        buffer: &mut Image,
    ) {
        // The hot slot is sized for one chunk and doesn't help the per-tile
        // walk; drop it so we don't accidentally serve stale data after a
        // pan.
        self.drop_hot_slot();

        let target_lod = lod_for(sfactor);
        let max_lod = self.cache.max_lod();

        let pzoom = paint_zoom as i32;
        let width_world = pzoom * canvas_width as i32;
        let height_world = pzoom * canvas_height as i32;
        let min_uc = xyz[u_coord] - width_world / 2;
        let max_uc = xyz[u_coord] + width_world / 2;
        let min_vc = xyz[v_coord] - height_world / 2;
        let max_vc = xyz[v_coord] + height_world / 2;
        let pc = xyz[plane_coord];
        if pc < 0 {
            return;
        }

        // -------- Pass 1: dispatch coarse → fine --------
        // Walking coarse → fine means the first submissions of a cold
        // viewport are the low-LOD preview chunks. The cache + downloader
        // queues are pure LIFO, so the most recently submitted (finest)
        // work pops first — but coarse-first submission still gets the
        // low-LOD chunks into flight before the worker pool can drain
        // them, so a quick preview shows up promptly while detail
        // streams in behind it.
        for lod in (target_lod..=max_lod).rev() {
            let Some(tiles) = viewport_tiles(min_uc, max_uc, min_vc, max_vc, pc, lod) else {
                continue;
            };
            for tu in tiles.tile_u_lo..=tiles.tile_u_hi {
                for tv in tiles.tile_v_lo..=tiles.tile_v_hi {
                    let mut chunk = [0i32; 3];
                    chunk[u_coord] = tu;
                    chunk[v_coord] = tv;
                    chunk[plane_coord] = tiles.tile_pc;
                    let key = ChunkKey::new(lod, chunk[0] as u32, chunk[1] as u32, chunk[2] as u32);
                    let _ = self.cache.state_or_fetch(key);
                }
            }
        }

        // -------- Pass 2: render per target tile, picking best resident LOD --------
        let Some(t) = viewport_tiles(min_uc, max_uc, min_vc, max_vc, pc, target_lod) else {
            return;
        };
        let ceil_div = |x: i32, d: i32| (x + d - 1).div_euclid(d);

        for tu in t.tile_u_lo..=t.tile_u_hi {
            for tv in t.tile_v_lo..=t.tile_v_hi {
                // Screen rect this target tile owns.
                let chunk_u_lo_t = tu * t.chunk_world;
                let chunk_u_hi_t = chunk_u_lo_t + t.chunk_world;
                let chunk_v_lo_t = tv * t.chunk_world;
                let chunk_v_hi_t = chunk_v_lo_t + t.chunk_world;
                let u_px_lo = ceil_div(chunk_u_lo_t - min_uc, pzoom).max(0);
                let u_px_hi = ceil_div(chunk_u_hi_t - min_uc, pzoom).min(canvas_width as i32);
                let v_px_lo = ceil_div(chunk_v_lo_t - min_vc, pzoom).max(0);
                let v_px_hi = ceil_div(chunk_v_hi_t - min_vc, pzoom).min(canvas_height as i32);
                if u_px_lo >= u_px_hi || v_px_lo >= v_px_hi {
                    continue;
                }

                // Walk target → coarsest, stopping at the first *terminal*
                // state (Resident or Empty). An `Empty` at a finer LOD wins
                // over coarser data: the fine-grained structure is what tells
                // us "no data here", so we shouldn't paint coarser-LOD values
                // through it.
                let mut chosen: Option<(u8, Arc<ChunkState>)> = None;
                for lod_try in target_lod..=max_lod {
                    let shift = lod_try - target_lod;
                    let mut chunk = [0u32; 3];
                    chunk[u_coord] = (tu as u32) >> shift;
                    chunk[v_coord] = (tv as u32) >> shift;
                    chunk[plane_coord] = (t.tile_pc as u32) >> shift;
                    let key = ChunkKey::new(lod_try, chunk[0], chunk[1], chunk[2]);
                    let s = if lod_try == target_lod {
                        self.cache.state_or_fetch(key)
                    } else {
                        match self.cache.peek(key) {
                            Some(s) => s,
                            None => continue,
                        }
                    };
                    if s.is_terminal() {
                        chosen = Some((lod_try, s));
                        break;
                    }
                }

                let painted = if let Some((lod_use, state)) = chosen.as_ref() {
                    if let Some(mmap) = state.as_resident() {
                        let scale_use = 1i32 << *lod_use;
                        let chunk_world_use = 64 * scale_use;
                        let shift = *lod_use - target_lod;
                        let parent_tu = tu >> shift;
                        let parent_tv = tv >> shift;
                        let parent_tpc = t.tile_pc >> shift;
                        let chunk_u_lo = parent_tu * chunk_world_use;
                        let chunk_v_lo = parent_tv * chunk_world_use;
                        let chunk_pc_lo = parent_tpc * chunk_world_use;
                        let plane_sample = ((pc - chunk_pc_lo) / scale_use) as usize;

                        for v_px in v_px_lo..v_px_hi {
                            let world_v = min_vc + v_px * pzoom;
                            let sample_v = ((world_v - chunk_v_lo) / scale_use) as usize;
                            for u_px in u_px_lo..u_px_hi {
                                let world_u = min_uc + u_px * pzoom;
                                let sample_u = ((world_u - chunk_u_lo) / scale_use) as usize;
                                let mut s = [0usize; 3];
                                s[u_coord] = sample_u;
                                s[v_coord] = sample_v;
                                s[plane_coord] = plane_sample;
                                let off = s[2] * 64 * 64 + s[1] * 64 + s[0];
                                let value = config.filter(mmap[off]);
                                buffer.set_gray(u_px as usize, v_px as usize, value);
                            }
                        }
                        true
                    } else {
                        // Empty at this LOD — fill the rect with the filtered
                        // zero value so the user sees a clean "no data" cell
                        // instead of whatever the buffer happened to contain.
                        let zero = config.filter(0);
                        for v_px in v_px_lo..v_px_hi {
                            for u_px in u_px_lo..u_px_hi {
                                buffer.set_gray(u_px as usize, v_px as usize, zero);
                            }
                        }
                        true
                    }
                } else {
                    false
                };

                if !painted && !config.debug_chunk_overlay {
                    // Nothing terminal yet, no overlay requested — leave the
                    // rect alone (caller cleared the buffer).
                    continue;
                }

                if config.debug_chunk_overlay {
                    let mut chunk = [0u32; 3];
                    chunk[u_coord] = tu as u32;
                    chunk[v_coord] = tv as u32;
                    chunk[plane_coord] = t.tile_pc as u32;
                    let target_key = ChunkKey::new(target_lod, chunk[0], chunk[1], chunk[2]);
                    let target_state = self.cache.peek(target_key);
                    let chosen_lod = chosen.as_ref().map(|(l, _)| *l);
                    let is_downloading = self.cache.is_downloading(target_key);
                    let color =
                        overlay_color_for(target_lod, chosen_lod, target_state.as_deref(), is_downloading);
                    if let Some(c) = color {
                        for v_px in v_px_lo..v_px_hi {
                            for u_px in u_px_lo..u_px_hi {
                                buffer.blend(u_px as usize, v_px as usize, c, OVERLAY_ALPHA);
                            }
                        }
                    }
                }
            }
        }
    }

    fn shared(&self) -> VolumeCons {
        let cache = self.cache.clone();
        Box::new(move || VoxelPaintVolume::into_volume(UnifiedVolume::new(cache)))
    }
}
