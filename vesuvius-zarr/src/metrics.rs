//! Process-global counters for local zarr chunk I/O.
//!
//! The render's local-fetch path reads a compressed chunk file off the backing
//! store (e.g. mountpoint-s3) and then blosc-decodes it, both on the same cache
//! worker thread. Splitting the two — `read_ns` (waiting on the store) vs
//! `decode_ns` (CPU) — is the decisive signal when tuning concurrency against a
//! high-latency filesystem: a high read fraction means we're store-latency
//! bound and more workers / fewer round-trips help; a high decode fraction
//! means we're CPU bound and more workers past `num_cpus` won't.
//!
//! Counters are plain relaxed atomics (monotonic since process start). They are
//! only written on the decode-from-file paths, so they reflect source reads,
//! not already-decoded cache hits. Reading is via [`snapshot`].

use std::sync::atomic::{AtomicU64, Ordering};

static READ_NS: AtomicU64 = AtomicU64::new(0);
static DECODE_NS: AtomicU64 = AtomicU64::new(0);
static STORE_BYTES: AtomicU64 = AtomicU64::new(0);
static CHUNK_COUNT: AtomicU64 = AtomicU64::new(0);

/// Record one chunk file read + decode. `read_ns` is the time spent pulling the
/// (compressed) bytes off the store, `decode_ns` the blosc decode (0 for the
/// uncompressed/raw path), `store_bytes` the number of bytes read from disk.
pub fn record_chunk_io(read_ns: u64, decode_ns: u64, store_bytes: u64) {
    READ_NS.fetch_add(read_ns, Ordering::Relaxed);
    DECODE_NS.fetch_add(decode_ns, Ordering::Relaxed);
    STORE_BYTES.fetch_add(store_bytes, Ordering::Relaxed);
    CHUNK_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Immutable snapshot of the cumulative local-fetch counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct ZarrIoStats {
    /// Cumulative wall time spent reading chunk bytes off the store.
    pub read_ns: u64,
    /// Cumulative wall time spent blosc-decoding chunk bytes.
    pub decode_ns: u64,
    /// Cumulative (compressed) bytes read from the store.
    pub store_bytes: u64,
    /// Number of chunk files read.
    pub chunk_count: u64,
}

impl ZarrIoStats {
    /// Fraction of (read + decode) time spent waiting on the store, in `0..=1`.
    /// Returns 0 when no time has been recorded yet.
    pub fn read_fraction(&self) -> f64 {
        let busy = self.read_ns + self.decode_ns;
        if busy == 0 {
            0.0
        } else {
            self.read_ns as f64 / busy as f64
        }
    }

    /// Total worker-busy time (read + decode) in nanoseconds.
    pub fn busy_ns(&self) -> u64 {
        self.read_ns + self.decode_ns
    }
}

/// Read the current cumulative counters.
pub fn snapshot() -> ZarrIoStats {
    ZarrIoStats {
        read_ns: READ_NS.load(Ordering::Relaxed),
        decode_ns: DECODE_NS.load(Ordering::Relaxed),
        store_bytes: STORE_BYTES.load(Ordering::Relaxed),
        chunk_count: CHUNK_COUNT.load(Ordering::Relaxed),
    }
}
