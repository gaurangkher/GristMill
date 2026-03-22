//! TrainerSocketClient — subscriber for the gristmill-trainer IPC socket.
//!
//! Connects to `trainer.sock` (newline-delimited JSON, server → client only)
//! and dispatches incoming messages to the appropriate handlers:
//!
//! | Message type             | Action                                             |
//! |--------------------------|-----------------------------------------------------|
//! | `checkpoint_promoted`    | Log + signal hot-reload of `/gristmill/checkpoints/active/` |
//! | `checkpoint_rolled_back` | Log — Inference Stack continues with current adapter |
//! | `training_started`       | Log — UI status indicator updated via status file   |
//! | `training_progress`      | Log progress percentage                             |
//! | `trainer_paused`         | Log reason                                         |
//!
//! Reconnects with exponential back-off (1 s → 64 s cap) when the trainer
//! is not yet running or the connection drops.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::Notify;
use tracing::{debug, info, warn};

/// Notified whenever a `checkpoint_promoted` message is received.
/// The Inference Stack watches this to trigger an adapter hot-reload.
pub type ReloadNotify = Arc<Notify>;

const BACKOFF_MIN_SECS: u64 = 1;
const BACKOFF_MAX_SECS: u64 = 64;

/// Resolve the trainer socket path.
///
/// Priority:
///   1. `GRISTMILL_TRAINER_SOCK` env var
///   2. `/gristmill/run/trainer.sock`
///   3. `~/.gristmill/run/trainer.sock` (dev fallback)
pub fn resolve_sock_path() -> PathBuf {
    if let Ok(p) = std::env::var("GRISTMILL_TRAINER_SOCK") {
        return PathBuf::from(p);
    }
    let default = PathBuf::from("/gristmill/run/trainer.sock");
    if default.parent().map(|p| p.exists()).unwrap_or(false) {
        return default;
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home)
        .join(".gristmill")
        .join("run")
        .join("trainer.sock")
}

/// Spawn a background task that subscribes to the trainer socket and
/// notifies `reload` on every `checkpoint_promoted` message.
///
/// Returns the task handle — abort it on daemon shutdown.
pub fn spawn(sock_path: PathBuf, reload: ReloadNotify) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut backoff = BACKOFF_MIN_SECS;
        loop {
            match connect_and_read(&sock_path, Arc::clone(&reload)).await {
                Ok(()) => {
                    // Server closed connection cleanly — reconnect immediately.
                    debug!("trainer.sock connection closed cleanly, reconnecting");
                    backoff = BACKOFF_MIN_SECS;
                }
                Err(e) => {
                    debug!(
                        sock = %sock_path.display(),
                        error = %e,
                        backoff_secs = backoff,
                        "trainer.sock unavailable, retrying"
                    );
                    tokio::time::sleep(Duration::from_secs(backoff)).await;
                    backoff = (backoff * 2).min(BACKOFF_MAX_SECS);
                }
            }
        }
    })
}

async fn connect_and_read(sock_path: &PathBuf, reload: ReloadNotify) -> std::io::Result<()> {
    let stream = UnixStream::connect(sock_path).await?;
    info!(sock = %sock_path.display(), "Connected to trainer.sock");
    let reader = BufReader::new(stream);
    let mut lines = reader.lines();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim().to_owned();
        if line.is_empty() {
            continue;
        }
        handle_message(&line, Arc::clone(&reload));
    }
    Ok(())
}

fn handle_message(json: &str, reload: ReloadNotify) {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(json) else {
        warn!("trainer.sock: unparseable message: {json}");
        return;
    };

    let msg_type = value
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("<unknown>");

    match msg_type {
        "checkpoint_promoted" => {
            let version = value.get("version").and_then(|v| v.as_u64()).unwrap_or(0);
            let score = value
                .get("validation_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let records = value
                .get("record_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            info!(
                version = version,
                validation_score = score,
                record_count = records,
                "Checkpoint promoted — triggering adapter hot-reload"
            );
            reload.notify_waiters();
        }
        "checkpoint_rolled_back" => {
            let version = value.get("version").and_then(|v| v.as_u64()).unwrap_or(0);
            let reason = value
                .get("reason")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            warn!(
                version = version,
                reason = reason,
                "Checkpoint rolled back — keeping current adapter"
            );
        }
        "training_started" => {
            let est_mins = value
                .get("estimated_duration_minutes")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let count = value
                .get("record_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            info!(
                estimated_minutes = est_mins,
                record_count = count,
                "Trainer: distillation cycle started"
            );
        }
        "training_progress" => {
            let pct = value
                .get("pct_complete")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let elapsed = value
                .get("elapsed_minutes")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            debug!(
                pct = pct,
                elapsed_minutes = elapsed,
                "Trainer: cycle progress"
            );
        }
        "trainer_paused" => {
            let reason = value
                .get("reason")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            info!(reason = reason, "Trainer: paused");
        }
        other => {
            debug!("trainer.sock: unhandled message type '{other}'");
        }
    }
}
