//! GristMill daemon — Phase 2.
//!
//! Initialises [`GristMillCore`] (Sieve + Ledger + Hammer + Millwright + Bus)
//! and exposes a Unix-socket IPC server so the TypeScript / Python shells can
//! call into the Rust core without an in-process FFI build.
//!
//! Socket path (in priority order):
//!   1. `GRISTMILL_SOCK` environment variable
//!   2. `~/.gristmill/gristmill.sock`
//!
//! Config path:
//!   1. `GRISTMILL_CONFIG` environment variable
//!   2. Built-in defaults (no file needed for dev / tests)

mod ipc;
mod trainer_socket;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use grist_core::GristMillCore;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ── Observability ─────────────────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    info!("GristMill daemon starting (Phase 2)");

    // ── Core ──────────────────────────────────────────────────────────────────
    let config_path = std::env::var("GRISTMILL_CONFIG").ok().map(PathBuf::from);
    let core = Arc::new(GristMillCore::new(config_path).await?);
    info!("GristMillCore ready");

    // ── IPC socket path ───────────────────────────────────────────────────────
    let socket_path = std::env::var("GRISTMILL_SOCK")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            PathBuf::from(home)
                .join(".gristmill")
                .join("gristmill.sock")
        });

    // ── IPC server (background task) ─────────────────────────────────────────
    let server = ipc::IpcServer::new(&socket_path, Arc::clone(&core));
    let server_handle = tokio::spawn(async move {
        if let Err(e) = server.run().await {
            tracing::error!("IPC server exited with error: {e}");
        }
    });

    // ── inference.lock heartbeat (background task) ────────────────────────────
    // Writes the current Unix epoch (seconds) to /gristmill/run/inference.lock
    // every 10 seconds while the daemon is alive.  gristmill-trainer reads this
    // file before claiming GPU memory — a stale heartbeat (>30s) means the
    // Inference Stack is idle and the trainer may proceed safely.
    let lock_path = {
        let run_dir = PathBuf::from("/gristmill/run");
        if run_dir.exists() || std::fs::create_dir_all(&run_dir).is_ok() {
            run_dir.join("inference.lock")
        } else {
            // Local development fallback: ~/.gristmill/run/inference.lock
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            let fallback = PathBuf::from(home).join(".gristmill").join("run");
            let _ = std::fs::create_dir_all(&fallback);
            fallback.join("inference.lock")
        }
    };
    info!(lock = %lock_path.display(), "starting inference.lock heartbeat (10s interval)");
    let heartbeat_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
        loop {
            interval.tick().await;
            let epoch = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            if let Err(e) = tokio::fs::write(&lock_path, epoch.to_string()).await {
                warn!(
                    lock = %lock_path.display(),
                    error = %e,
                    "failed to write inference.lock heartbeat"
                );
            }
        }
    });

    // ── Trainer socket subscriber (background task) ───────────────────────────
    // Subscribes to gristmill-trainer's Unix socket and triggers adapter
    // hot-reload whenever a checkpoint_promoted message is received.
    let reload_notify = Arc::new(tokio::sync::Notify::new());
    let trainer_sock_path = trainer_socket::resolve_sock_path();
    info!(sock = %trainer_sock_path.display(), "subscribing to trainer.sock (exponential backoff reconnect)");
    let trainer_handle = trainer_socket::spawn(trainer_sock_path, Arc::clone(&reload_notify));

    info!("GristMill daemon ready — press Ctrl+C to stop");
    info!(socket = %socket_path.display(), "IPC socket");

    // ── Shutdown on SIGINT / SIGTERM ──────────────────────────────────────────
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            info!("received SIGINT, shutting down");
        }
        _ = async {
            #[cfg(unix)]
            {
                use tokio::signal::unix::{signal, SignalKind};
                if let Ok(mut sigterm) = signal(SignalKind::terminate()) {
                    sigterm.recv().await;
                } else {
                    std::future::pending::<()>().await;
                }
            }
            #[cfg(not(unix))]
            std::future::pending::<()>().await;
        } => {
            info!("received SIGTERM, shutting down");
        }
    }

    server_handle.abort();
    heartbeat_handle.abort();
    trainer_handle.abort();
    Ok(())
}
