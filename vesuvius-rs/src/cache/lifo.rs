//! Shared pure-LIFO work queue, used by both the cache's task queue and
//! the downloader's job queue.
//!
//! Entries live in a `BTreeMap` keyed by `!seq`, so the most recently
//! submitted (or re-touched) entry pops first. Earlier iterations split
//! work across LOD-rank and viewport-distance tiers, but in practice that
//! let coarse-LOD work outrank current-viewport work and stall painting.
//! The simpler model: paint always asks for what it wants right now, and
//! "right now" wins. Older requests slide toward the tail and either get
//! processed in LIFO order when workers catch up, or culled by `max_age`
//! at pop.
//!
//! `chunk_index` mirrors `entries` keyed by `ChunkKey` for the O(1)
//! lookup `touch` needs to refresh seq + `added_at` on in-flight entries
//! — the paint loop re-touches what it wants every frame so the queue
//! head tracks the current viewport.
//!
//! Unbounded; dedup happens at the cache layer (source-key + chunk-key
//! uniqueness), so submission always succeeds. The only staleness check
//! is age.

use super::state::ChunkKey;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

pub(super) struct LifoQueue<T> {
    inner: Mutex<Inner<T>>,
    not_empty: Condvar,
    max_age: Duration,
}

struct Inner<T> {
    entries: BTreeMap<u64, QueueEntry<T>>,
    /// Reverse lookup: every queue key currently registered for a given
    /// chunk. A chunk can have several entries (e.g. its sources'
    /// FetchSource tasks plus an Extract task once sources resolve).
    /// Maintained in lockstep with `entries`.
    chunk_index: HashMap<ChunkKey, Vec<u64>>,
    next_seq: u64,
}

pub(super) struct QueueEntry<T> {
    /// The cache chunk this work is on behalf of — the `touch` handle.
    pub chunk: ChunkKey,
    pub added_at: Instant,
    pub item: T,
}

/// Encode `seq` so that **larger** seq sorts BEFORE smaller (BTreeMap pops
/// smallest key first → LIFO). `seq` is monotonically increasing so `!seq`
/// is monotonically decreasing.
fn rev_seq(seq: u64) -> u64 {
    !seq
}

impl<T> LifoQueue<T> {
    pub fn new(max_age: Duration) -> Self {
        Self {
            inner: Mutex::new(Inner {
                entries: BTreeMap::new(),
                chunk_index: HashMap::new(),
                next_seq: 0,
            }),
            not_empty: Condvar::new(),
            max_age,
        }
    }

    /// Submit one work item on behalf of `chunk`. Always succeeds (the
    /// queue is unbounded); the only way queued work dies unprocessed is
    /// the `max_age` cull at pop.
    pub fn submit(&self, chunk: ChunkKey, item: T) {
        let mut q = self.inner.lock().unwrap();
        q.next_seq += 1;
        let key = rev_seq(q.next_seq);
        q.entries.insert(
            key,
            QueueEntry {
                chunk,
                added_at: Instant::now(),
                item,
            },
        );
        q.chunk_index.entry(chunk).or_default().push(key);
        self.not_empty.notify_one();
    }

    /// Refresh every queued entry for `chunk`: bump its seq (moving it
    /// to the head of the LIFO order) and reset `added_at` so `max_age`
    /// re-counts from now. No-op when the chunk has no queued entries.
    pub fn touch(&self, chunk: ChunkKey) {
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

    /// Block until a non-stale entry is available. Entries culled along
    /// the way (older than `max_age`) are returned as the second tuple
    /// element so the caller can run their cancellation paths outside
    /// the queue lock.
    pub fn pop(&self) -> (QueueEntry<T>, Vec<QueueEntry<T>>) {
        let mut q = self.inner.lock().unwrap();
        let mut dropped: Vec<QueueEntry<T>> = Vec::new();
        loop {
            let Some((key, entry)) = q.entries.pop_first() else {
                q = self.not_empty.wait(q).unwrap();
                continue;
            };
            if let Some(keys) = q.chunk_index.get_mut(&entry.chunk) {
                keys.retain(|k| *k != key);
                if keys.is_empty() {
                    q.chunk_index.remove(&entry.chunk);
                }
            }
            if entry.added_at.elapsed() > self.max_age {
                dropped.push(entry);
                continue;
            }
            return (entry, dropped);
        }
    }
}
