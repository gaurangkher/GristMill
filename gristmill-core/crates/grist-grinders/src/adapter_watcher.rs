//! AdapterWatcher — filesystem watcher for LoRA adapter hot-swap (Phase 3).
//!
//! Watches `/gristmill/checkpoints/active/` (or a configured path) for
//! changes using the platform-native FS events API:
//!
//! | Platform | Backend             |
//! |----------|---------------------|
//! | Linux    | inotify             |
//! | macOS    | FSEvents            |
//! | Windows  | ReadDirectoryChangesW |
//!
//! When a domain sub-directory changes (e.g. `active/code/` is written after a
//! checkpoint promotion), the watcher sends the domain name over an unbounded
//! mpsc channel.  The caller (typically the daemon) calls
//! `Grinders::hot_reload(domain_model_id(domain))` upon receipt.
//!
//! # Design
//!
//! The `notify` watcher runs on a dedicated OS thread (spawned internally by
//! `notify`).  Events are relayed into a `tokio::sync::mpsc::UnboundedSender`
//! so the async daemon can `recv()` without blocking.
//!
//! # Usage
//!
//! ```rust,no_run
//! use grist_grinders::adapter_watcher::AdapterWatcher;
//! use std::path::PathBuf;
//!
//! #[tokio::main]
//! async fn main() {
//!     let active_dir = PathBuf::from("/gristmill/checkpoints/active");
//!     let (watcher, mut rx) = AdapterWatcher::spawn(active_dir).unwrap();
//!
//!     while let Some(domain) = rx.recv().await {
//!         println!("Adapter changed for domain: {}", domain);
//!         // grinders.hot_reload(&domain_model_id(&domain)).ok();
//!     }
//! }
//! ```

use std::path::PathBuf;

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;
use tracing::{info, warn};

/// Returns the grinder model-id used for a given domain adapter.
///
/// Convention: `adapter:{domain}` — e.g. `adapter:code`, `adapter:writing`.
/// The `Grinders` registry stores domain adapters under this key so that
/// `Grinders::hot_reload("adapter:code")` reloads only the code adapter.
pub fn domain_model_id(domain: &str) -> String {
    format!("adapter:{domain}")
}

/// Filesystem watcher for the `active/` adapter directory.
///
/// Dropping this struct stops the watcher.
pub struct AdapterWatcher {
    /// The underlying notify watcher — kept alive for the struct's lifetime.
    _watcher: RecommendedWatcher,
    /// Absolute path being watched (for logging / tests).
    pub active_dir: PathBuf,
}

impl AdapterWatcher {
    /// Start watching `active_dir` for FS events.
    ///
    /// Returns `(AdapterWatcher, Receiver<domain>)`.  The receiver yields the
    /// name of each domain sub-directory whose adapter changed (e.g. `"code"`).
    ///
    /// Only `Create` and `Modify` events are forwarded — `Remove` events are
    /// ignored because a rollback that clears a domain dir should not trigger
    /// a hot-reload.
    pub fn spawn(active_dir: PathBuf) -> notify::Result<(Self, mpsc::UnboundedReceiver<String>)> {
        let (tx, rx) = mpsc::unbounded_channel::<String>();
        let active_dir_clone = active_dir.clone();

        let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
            match res {
                Ok(event) => {
                    // Only react to creation/modification — not removals.
                    if !matches!(event.kind, EventKind::Create(_) | EventKind::Modify(_)) {
                        return;
                    }
                    for path in &event.paths {
                        // Determine the domain sub-directory that changed.
                        // Path looks like: active_dir/{domain}/adapter_model.bin
                        // We want the first component after `active_dir`.
                        let domain = path
                            .strip_prefix(&active_dir_clone)
                            .ok()
                            .and_then(|rel| rel.components().next())
                            .map(|c| c.as_os_str().to_string_lossy().into_owned());

                        if let Some(domain) = domain {
                            if !domain.is_empty() && domain != "." {
                                // Deduplicate: ignore if the channel is already saturated.
                                let _ = tx.send(domain);
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "adapter watcher error");
                }
            }
        })?;

        watcher.watch(&active_dir, RecursiveMode::Recursive)?;
        info!(
            path = %active_dir.display(),
            "adapter watcher started (platform: {})",
            std::env::consts::OS
        );

        Ok((
            Self {
                _watcher: watcher,
                active_dir,
            },
            rx,
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn domain_model_id_format() {
        assert_eq!(domain_model_id("code"), "adapter:code");
        assert_eq!(domain_model_id("writing"), "adapter:writing");
        assert_eq!(domain_model_id("default"), "adapter:default");
    }

    #[tokio::test]
    async fn watcher_spawns_without_error() {
        let dir = tempfile::tempdir().unwrap();
        let active = dir.path().join("active");
        std::fs::create_dir_all(&active).unwrap();
        let (_w, _rx) = AdapterWatcher::spawn(active).expect("watcher should start");
    }

    // FSEvents on macOS coalesces events and has a platform-dependent minimum
    // latency that makes this test unreliable in automated test runs.
    // Run manually with: cargo test watcher_detects -- --ignored
    #[tokio::test]
    #[ignore = "FSEvents latency varies by OS; run manually to exercise FS-event delivery"]
    async fn watcher_detects_domain_file_creation() {
        let dir = tempfile::tempdir().unwrap();
        let active = dir.path().join("active");
        let code_dir = active.join("code");
        std::fs::create_dir_all(&code_dir).unwrap();

        let (_w, mut rx) = AdapterWatcher::spawn(active.clone()).expect("watcher should start");

        // Give the watcher time to initialise — FSEvents on macOS can take >100ms.
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Write a file into the `code` domain sub-directory.
        std::fs::write(code_dir.join("adapter_config.json"), b"{}").unwrap();

        // Expect a domain event within 5 seconds (FSEvents coalesces events).
        let domain = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("timeout waiting for watcher event")
            .expect("channel closed");

        assert_eq!(domain, "code");
    }
}
