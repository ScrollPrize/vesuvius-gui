//! Cache-path resolution and downloading for atlas segments.
//!
//! Atlas segments are fetched as **tifxyz** by default — a directory of
//! `meta.json` + `x.tif` + `y.tif` + `z.tif` (the layout
//! `TifXyzVolume::load_from_directory` expects) — falling back to a legacy
//! single `.obj` file for segments that only publish a mesh. This module owns
//! the cache layout and the (network) download; the GUI keeps only the
//! in-progress/blinking state and the load-on-ready handling.

use std::path::PathBuf;
use std::sync::mpsc::Sender;

use directories::BaseDirs;

use super::app::UINotification;

/// Files that make up a tifxyz segment directory, in download order. A cached
/// tifxyz segment must have all of them; a partial set is treated as not cached.
pub const TIFXYZ_FILES: [&str; 4] = ["meta.json", "x.tif", "y.tif", "z.tif"];

fn atlas_segments_root() -> PathBuf {
    BaseDirs::new().unwrap().cache_dir().join("vesuvius-gui").join("atlas-segments")
}

/// Legacy single-file obj cache path: `…/atlas-segments/<sample>/<segment>.obj`.
pub fn obj_cache_path(sample_id: &str, segment_id: &str) -> PathBuf {
    atlas_segments_root().join(format!("{}/{}.obj", sample_id, segment_id))
}

/// Cache directory for a downloaded tifxyz segment. Holds `meta.json` +
/// `x.tif` + `y.tif` + `z.tif`, the layout `TifXyzVolume::load_from_directory`
/// (and `setup_segment`'s directory branch) expects.
pub fn tifxyz_cache_path(sample_id: &str, segment_id: &str) -> PathBuf {
    atlas_segments_root().join(format!("{}/{}", sample_id, segment_id))
}

/// Local path of an already-downloaded atlas segment, if any. Prefers the
/// tifxyz directory (the default), falling back to a legacy `.obj` file for
/// segments that only publish an obj.
pub fn cached_path(sample_id: &str, segment_id: &str) -> Option<PathBuf> {
    let tifxyz_dir = tifxyz_cache_path(sample_id, segment_id);
    // Require the full set — a partial download (e.g. meta.json + z.tif but
    // no x/y.tif) must not be mistaken for a usable cache, or loading panics.
    if TIFXYZ_FILES.iter().all(|f| tifxyz_dir.join(f).is_file()) {
        return Some(tifxyz_dir);
    }
    let obj = obj_cache_path(sample_id, segment_id);
    if obj.exists() {
        return Some(obj);
    }
    None
}

/// Download a tifxyz segment directory (`meta.json` + `x/y/z.tif`) into the
/// cache, then notify (via `sender`) once the full set has arrived. `tifxyz_url`
/// points at the directory; the individual file names are appended.
///
/// Files are fetched one at a time: a handful of parallel S3/CDN requests
/// proved flaky (some would drop, leaving a half-written dir), and a few MB
/// sequentially is cheap. On any failure the partial directory is removed so it
/// isn't mistaken for a cache and the user can retry by clicking again.
pub fn download_tifxyz(
    sender: Sender<UINotification>,
    sample_id: String,
    segment_id: String,
    tifxyz_url: String,
    volume_id: String,
) {
    let dir = tifxyz_cache_path(&sample_id, &segment_id);
    // Start from a clean slate so a prior partial download can't shadow this one.
    let _ = std::fs::remove_dir_all(&dir);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log::error!("Failed to create tifxyz cache dir {}: {}", dir.display(), e);
        let _ = sender.send(UINotification::AtlasSegmentDownloadFailed(sample_id, segment_id));
        return;
    }
    let base_url = tifxyz_url.trim_end_matches('/').to_string();
    fetch_tifxyz_file(sender, base_url, dir, sample_id, segment_id, volume_id, 0);
}

/// Fetch tifxyz file `idx`, then chain to the next on success; on failure
/// remove the partial directory and report it so the blinking state clears.
fn fetch_tifxyz_file(
    sender: Sender<UINotification>,
    base_url: String,
    dir: PathBuf,
    sample_id: String,
    segment_id: String,
    volume_id: String,
    idx: usize,
) {
    if idx >= TIFXYZ_FILES.len() {
        let _ = sender.send(UINotification::AtlasSegmentDownloadReady(sample_id, segment_id, volume_id));
        return;
    }
    let fname = TIFXYZ_FILES[idx];
    let url = format!("{}/{}", base_url, fname);
    let path = dir.join(fname);
    log::info!("Downloading atlas tifxyz file {}", url);
    ehttp::fetch(ehttp::Request::get(&url), move |response| {
        let ok = match response {
            Ok(response) if response.ok => match std::fs::File::create(&path) {
                Ok(mut file) => {
                    let res = std::io::copy(&mut std::io::Cursor::new(response.bytes), &mut file);
                    if let Err(e) = &res {
                        log::error!("Failed to write {}: {}", path.display(), e);
                    }
                    res.is_ok()
                }
                Err(e) => {
                    log::error!("Failed to create {}: {}", path.display(), e);
                    false
                }
            },
            Ok(response) => {
                log::error!("Failed to download {}: HTTP {} {}", url, response.status, response.status_text);
                false
            }
            Err(e) => {
                log::error!("Failed to download {}: {}", url, e);
                false
            }
        };
        if ok {
            fetch_tifxyz_file(sender, base_url, dir, sample_id, segment_id, volume_id, idx + 1);
        } else {
            let _ = std::fs::remove_dir_all(&dir);
            log::error!("tifxyz download for {}/{} failed", sample_id, segment_id);
            let _ = sender.send(UINotification::AtlasSegmentDownloadFailed(sample_id, segment_id));
        }
    });
}
