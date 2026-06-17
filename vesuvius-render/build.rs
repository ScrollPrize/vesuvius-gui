//! Bakes version provenance into the binary at compile time so every rendered
//! artifact can record exactly which build produced it. The values are exposed
//! to the crate via `env!()`:
//!   - `VESUVIUS_GIT_REVISION` — `git rev-parse HEAD`, suffixed `-dirty` when
//!     tracked files differ from HEAD (untracked files are ignored).
//!   - `VESUVIUS_BUILD_TIME`   — UTC build timestamp (ISO-8601).
//!
//! Best-effort: if git (or `date`) is unavailable the value is `"unknown"`.

use std::path::Path;
use std::process::Command;

fn git(args: &[&str]) -> Option<std::process::Output> {
    Command::new("git").args(args).output().ok().filter(|o| o.status.success())
}

fn main() {
    let revision = git(&["rev-parse", "HEAD"])
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .map(|rev| {
            // `--quiet` exits non-zero when tracked files differ from HEAD.
            let dirty = Command::new("git")
                .args(["diff", "--quiet", "HEAD"])
                .status()
                .map(|s| !s.success())
                .unwrap_or(false);
            if dirty {
                format!("{}-dirty", rev)
            } else {
                rev
            }
        })
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=VESUVIUS_GIT_REVISION={}", revision);

    let build_time = Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=VESUVIUS_BUILD_TIME={}", build_time);

    // Re-run when the checked-out commit or the index (staged/tracked changes)
    // may have moved, so the baked revision stays in sync. Best-effort: missing
    // paths are simply skipped.
    for p in ["../.git/HEAD", "../.git/index", "../.git/logs/HEAD"] {
        if Path::new(p).exists() {
            println!("cargo:rerun-if-changed={}", p);
        }
    }
}
