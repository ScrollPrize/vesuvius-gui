//! Optional path / URL remap configuration.
//!
//! A user-provided JSON file pointed to by the `VESUVIUS_REMAP_CONFIG` env var. When the
//! variable is unset or the file is missing, no remappings are applied (the default).
//!
//! Two kinds of remappings are supported:
//!
//! 1. `atlas_url_rewrites` — atlas access roots are normally filtered by `usage`. An
//!    entry here whitelists access roots whose full URL (access root + origin path)
//!    starts with `match_url_prefix`, optionally restricted to a specific `usage`,
//!    and optionally rewrites the prefix before use. The remap is checked before the
//!    `standard` short-circuit, so a matching rule wins over the default URL.
//! 2. `volume_overrides` — maps a volume id (as returned by `VolumeReference::id()`) to a
//!    direct OME-zarr URL. When the GUI loads a volume whose id matches, it uses the
//!    overridden URL instead of the default tile layout.
//!
//! Example config:
//!
//! ```json
//! {
//!   "atlas_url_rewrites": [
//!     {
//!       "match_url_prefix": "https://example.invalid",
//!       "rewrite_to": "https://example.test/volumes"
//!     }
//!   ],
//!   "volume_overrides": {
//!     "00000000000000": "https://example.test/volumes/some.zarr/"
//!   }
//! }
//! ```

use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

pub const ENV_VAR: &str = "VESUVIUS_REMAP_CONFIG";

#[derive(Debug, Clone, Default, Deserialize)]
pub struct RemapConfig {
    #[serde(default)]
    pub atlas_url_rewrites: Vec<AtlasUrlRewrite>,
    #[serde(default)]
    pub volume_overrides: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AtlasUrlRewrite {
    /// Atlas access root `usage` to match. If omitted, the rule matches any usage.
    #[serde(default)]
    pub usage: Option<String>,
    pub match_url_prefix: String,
    /// If present, the matched prefix is replaced with this string. If omitted, the URL is
    /// allowed through unchanged.
    #[serde(default)]
    pub rewrite_to: Option<String>,
}

impl RemapConfig {
    /// Return the loaded config, reading it from `VESUVIUS_REMAP_CONFIG` on first access.
    /// Subsequent calls return the cached value. If the env var is unset or the file
    /// cannot be read, an empty default config is cached.
    pub fn get() -> &'static RemapConfig {
        static CONFIG: OnceLock<RemapConfig> = OnceLock::new();
        CONFIG.get_or_init(Self::load_from_env)
    }

    fn load_from_env() -> RemapConfig {
        let Some(path) = std::env::var_os(ENV_VAR).map(PathBuf::from) else {
            return RemapConfig::default();
        };
        match std::fs::read_to_string(&path) {
            Ok(content) => match serde_json::from_str::<RemapConfig>(&content) {
                Ok(cfg) => {
                    log::info!("Loaded remap config from {}", path.display());
                    cfg
                }
                Err(e) => {
                    log::warn!("Failed to parse remap config {}: {}", path.display(), e);
                    RemapConfig::default()
                }
            },
            Err(e) => {
                log::warn!("Failed to read remap config {}: {}", path.display(), e);
                RemapConfig::default()
            }
        }
    }

    /// Apply atlas URL rewrite rules. `url` is the full URL (access root joined with
    /// origin path) the caller would otherwise use. If `usage` and the URL prefix
    /// match a configured rule, return the rewritten URL (or the original if the rule
    /// has no `rewrite_to`). If no rule matches, return `None` — the caller falls back
    /// to its default URL resolution.
    pub fn rewrite_atlas_url(&self, usage: &str, url: &str) -> Option<String> {
        for rule in &self.atlas_url_rewrites {
            let usage_matches = rule.usage.as_deref().is_none_or(|u| u == usage);
            if usage_matches && url.starts_with(&rule.match_url_prefix) {
                let rewritten = match &rule.rewrite_to {
                    Some(to) => url.replacen(&rule.match_url_prefix, to, 1),
                    None => url.to_string(),
                };
                log::info!("Atlas URL remap (usage={}): {} -> {}", usage, url, rewritten);
                return Some(rewritten);
            }
        }
        None
    }

    /// Direct URL override for a volume id, e.g. mapping `"20230205180739"` to a specific
    /// OME-zarr URL.
    pub fn volume_override_url(&self, volume_id: &str) -> Option<&str> {
        self.volume_overrides.get(volume_id).map(String::as_str)
    }
}
