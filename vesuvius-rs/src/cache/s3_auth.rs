//! Optional AWS SigV4 signing for cache HTTP downloads that target S3.
//!
//! When the renderer fetches zarr chunks directly from a private S3 bucket over
//! HTTPS (instead of reading them through a mountpoint-s3 FUSE mount), every GET
//! must be SigV4-signed. Credentials come from the standard AWS provider chain
//! (env vars, shared profile, IRSA web-identity token → STS, IMDS), resolved and
//! refreshed on a dedicated background thread so the synchronous download workers
//! can sign each request without blocking on async credential I/O.
//!
//! Signing is gated to AWS S3 hosts ([`is_s3_host`]) so any non-S3 download (a
//! public tile server, a presigned CDN URL) passes through untouched.

use std::sync::mpsc;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

use aws_credential_types::provider::ProvideCredentials;
use aws_credential_types::Credentials;
use aws_sigv4::http_request::{sign, PayloadChecksumKind, SignableBody, SignableRequest, SigningSettings};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;

/// How long to wait for the first credential resolution before giving up and
/// falling back to unsigned downloads.
const FIRST_RESOLVE_TIMEOUT: Duration = Duration::from_secs(20);
/// Refresh this far ahead of credential expiry.
const REFRESH_SLACK: Duration = Duration::from_secs(300);

#[derive(Clone)]
struct CachedCreds {
    access_key_id: String,
    secret_access_key: String,
    session_token: Option<String>,
}

pub struct S3Signer {
    creds: Arc<RwLock<Option<CachedCreds>>>,
    fallback_region: String,
}

impl S3Signer {
    /// Resolve AWS credentials via the default provider chain and start a
    /// background refresher. Returns `None` if no credentials resolve within
    /// [`FIRST_RESOLVE_TIMEOUT`], so environments without AWS auth transparently
    /// fall back to unsigned requests.
    pub fn try_new() -> Option<Arc<S3Signer>> {
        let creds: Arc<RwLock<Option<CachedCreds>>> = Arc::new(RwLock::new(None));
        let fallback_region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|_| "us-east-1".to_string());

        let (ready_tx, ready_rx) = mpsc::channel::<bool>();
        let creds_for_thread = creds.clone();
        std::thread::Builder::new()
            .name("vesuvius-s3-creds".into())
            .spawn(move || refresher_loop(creds_for_thread, ready_tx))
            .ok()?;

        match ready_rx.recv_timeout(FIRST_RESOLVE_TIMEOUT) {
            Ok(true) => {
                log::info!("S3 SigV4 signing enabled (region fallback: {})", fallback_region);
                Some(Arc::new(S3Signer { creds, fallback_region }))
            }
            _ => {
                log::warn!("S3 signing requested but no AWS credentials resolved; downloads will be unsigned");
                None
            }
        }
    }

    /// Compute the SigV4 headers (`Authorization`, `x-amz-date`,
    /// `x-amz-content-sha256`, and `x-amz-security-token` for temporary creds)
    /// to attach to a GET for `url`.
    pub fn sign_get(&self, url: &str) -> Result<Vec<(String, String)>, String> {
        let creds = self
            .creds
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| "no credentials available".to_string())?;
        let region = region_from_s3_url(url).unwrap_or_else(|| self.fallback_region.clone());

        let identity: Identity = Credentials::new(
            creds.access_key_id,
            creds.secret_access_key,
            creds.session_token,
            None,
            "vesuvius-s3",
        )
        .into();

        let mut settings = SigningSettings::default();
        // S3 requires the x-amz-content-sha256 header; UNSIGNED-PAYLOAD is fine
        // over HTTPS and avoids hashing every multi-MB body.
        settings.payload_checksum_kind = PayloadChecksumKind::XAmzSha256;

        let params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&region)
            .name("s3")
            .time(SystemTime::now())
            .settings(settings)
            .build()
            .map_err(|e| format!("sigv4 params: {e}"))?
            .into();

        // Host is derived from the URI and signed automatically; the Range
        // header we also send is left unsigned (S3 honours it regardless).
        let signable = SignableRequest::new("GET", url, std::iter::empty(), SignableBody::UnsignedPayload)
            .map_err(|e| format!("signable request: {e}"))?;

        let (instructions, _signature) = sign(signable, &params).map_err(|e| format!("sign: {e}"))?.into_parts();

        Ok(instructions
            .headers()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect())
    }
}

fn refresher_loop(creds: Arc<RwLock<Option<CachedCreds>>>, ready_tx: mpsc::Sender<bool>) {
    let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            log::error!("S3 creds: failed to build runtime: {e}");
            let _ = ready_tx.send(false);
            return;
        }
    };
    rt.block_on(async move {
        let config = aws_config::defaults(aws_config::BehaviorVersion::latest()).load().await;
        let provider = match config.credentials_provider() {
            Some(p) => p,
            None => {
                log::error!("S3 creds: no credentials provider resolved in default chain");
                let _ = ready_tx.send(false);
                return;
            }
        };
        let mut first = true;
        loop {
            match provider.provide_credentials().await {
                Ok(c) => {
                    // Schedule the next refresh a little before expiry (or a
                    // fixed interval if the creds are non-expiring), clamped so
                    // a near-expired credential doesn't busy-loop.
                    let next_refresh = c
                        .expiry()
                        .and_then(|exp| exp.duration_since(SystemTime::now()).ok())
                        .map(|d| d.saturating_sub(REFRESH_SLACK))
                        .unwrap_or(Duration::from_secs(900))
                        .max(Duration::from_secs(60));
                    *creds.write().unwrap() = Some(CachedCreds {
                        access_key_id: c.access_key_id().to_string(),
                        secret_access_key: c.secret_access_key().to_string(),
                        session_token: c.session_token().map(|s| s.to_string()),
                    });
                    if first {
                        let _ = ready_tx.send(true);
                        first = false;
                    }
                    log::debug!("S3 creds refreshed; next refresh in {:?}", next_refresh);
                    tokio::time::sleep(next_refresh).await;
                }
                Err(e) => {
                    log::error!("S3 creds: provide_credentials failed: {e}");
                    if first {
                        let _ = ready_tx.send(false);
                        return;
                    }
                    tokio::time::sleep(Duration::from_secs(30)).await;
                }
            }
        }
    });
}

/// Only AWS S3 endpoints are signed, so unsigned hosts pass through untouched.
pub fn is_s3_host(url: &str) -> bool {
    url_host(url).map(|h| h.ends_with("amazonaws.com")).unwrap_or(false)
}

fn url_host(url: &str) -> Option<String> {
    let rest = url.split("://").nth(1)?;
    let authority = rest.split('/').next()?;
    let host = authority.rsplit('@').next()?; // strip any userinfo
    let host = host.split(':').next()?; // strip port
    if host.is_empty() {
        None
    } else {
        Some(host.to_string())
    }
}

/// Extract the AWS region from an S3 endpoint host, e.g.
/// `bucket.s3.us-east-1.amazonaws.com`, `s3.eu-west-1.amazonaws.com`, or
/// `bucket.s3-us-west-2.amazonaws.com`. Returns `None` for region-less hosts
/// (`s3.amazonaws.com` == us-east-1) so the caller's fallback applies.
fn region_from_s3_url(url: &str) -> Option<String> {
    let host = url_host(url)?;
    let parts: Vec<&str> = host.split('.').collect();
    for (i, p) in parts.iter().enumerate() {
        if let Some(rest) = p.strip_prefix("s3-") {
            return Some(rest.to_string());
        }
        if *p == "s3" {
            if let Some(next) = parts.get(i + 1) {
                if *next != "amazonaws" {
                    return Some((*next).to_string());
                }
            }
            return None;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_parsing() {
        assert_eq!(
            region_from_s3_url("https://bkt.s3.us-east-2.amazonaws.com/a/b/0.0.0"),
            Some("us-east-2".to_string())
        );
        assert_eq!(
            region_from_s3_url("https://s3.eu-west-1.amazonaws.com/bkt/a/0"),
            Some("eu-west-1".to_string())
        );
        assert_eq!(
            region_from_s3_url("https://bkt.s3-us-west-2.amazonaws.com/a/0"),
            Some("us-west-2".to_string())
        );
        assert_eq!(region_from_s3_url("https://bkt.s3.amazonaws.com/a/0"), None);
    }

    #[test]
    fn host_gating() {
        assert!(is_s3_host("https://bkt.s3.us-east-1.amazonaws.com/x"));
        assert!(!is_s3_host("https://dl.ash2txt.org/x"));
        assert!(!is_s3_host("not a url"));
    }
}
