//! Optional per-request download telemetry, written as JSONL.
//!
//! Enabled by setting `VESUVIUS_NET_LOG=/path/to/file.jsonl` before launch;
//! when unset, every call short-circuits on a `OnceLock` read. One JSON
//! object per line, flushed per event so a hard kill loses nothing.
//!
//! Consumed offline (e.g. `scripts/analyze-netlog.py`) to break download time
//! into queue wait vs TTFB vs body transfer, per host, and to reconstruct
//! the in-flight concurrency / link-utilization timeline.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

static SINK: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();

fn sink() -> Option<&'static Mutex<BufWriter<File>>> {
    SINK.get_or_init(|| {
        let path = std::env::var("VESUVIUS_NET_LOG").ok()?;
        let file = OpenOptions::new().create(true).append(true).open(&path).ok()?;
        Some(Mutex::new(BufWriter::new(file)))
    })
    .as_ref()
}

pub(super) fn enabled() -> bool {
    sink().is_some()
}

/// Milliseconds since the unix epoch — event timestamps in the log, so
/// runs can be correlated with external captures.
pub(super) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub(super) fn emit(value: serde_json::Value) {
    if let Some(m) = sink() {
        if let Ok(mut w) = m.lock() {
            let _ = serde_json::to_writer(&mut *w, &value);
            let _ = w.write_all(b"\n");
            let _ = w.flush();
        }
    }
}
