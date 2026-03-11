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

use std::path::PathBuf;
use std::sync::Arc;

use grist_core::GristMillCore;
use tracing::info;
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
    Ok(())
}
