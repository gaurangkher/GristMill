//! `gristmill` — GristMill command-line interface.

use anyhow::Result;
use clap::{Parser, Subcommand};

mod client;
mod commands;

use commands::{
    memory::MemoryCmd, metrics::MetricsCmd, models::ModelsCmd, pipeline::PipelineCmd,
    train::TrainCmd, watch::WatchCmd,
};

#[derive(Parser)]
#[command(
    name = "gristmill",
    about = "GristMill v2 — local-first AI orchestration",
    version,
    propagate_version = true
)]
struct Cli {
    /// Path to the daemon socket (overrides GRISTMILL_SOCK env var).
    #[arg(long, global = true, env = "GRISTMILL_SOCK")]
    sock: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the GristMill daemon.
    Start,
    /// Stop the running daemon.
    Stop,
    /// Show daemon health / system status.
    Status,
    /// Diagnose configuration issues.
    Doctor,
    /// Model management.
    #[command(subcommand)]
    Models(ModelsCmd),
    /// Pipeline management.
    #[command(subcommand)]
    Pipeline(PipelineCmd),
    /// Memory management.
    #[command(subcommand)]
    Memory(MemoryCmd),
    /// Watch / notification management.
    #[command(subcommand)]
    Watch(WatchCmd),
    /// Trigger model (re)training via the Python shell.
    #[command(subcommand)]
    Train(TrainCmd),
    /// Show aggregate system metrics.
    Metrics(MetricsCmd),
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();

    let cli = Cli::parse();
    let sock = cli.sock.unwrap_or_else(default_socket_path);

    match cli.command {
        Command::Start => commands::daemon::start().await,
        Command::Stop => commands::daemon::stop(&sock).await,
        Command::Status => commands::daemon::status(&sock).await,
        Command::Doctor => commands::daemon::doctor(&sock).await,
        Command::Models(cmd) => commands::models::run(cmd, &sock).await,
        Command::Pipeline(cmd) => commands::pipeline::run(cmd, &sock).await,
        Command::Memory(cmd) => commands::memory::run(cmd, &sock).await,
        Command::Watch(cmd) => commands::watch::run(cmd, &sock).await,
        Command::Train(cmd) => commands::train::run(cmd).await,
        Command::Metrics(cmd) => commands::metrics::run(cmd, &sock).await,
    }
}

fn default_socket_path() -> String {
    dirs::home_dir()
        .map(|h| h.join(".gristmill").join("gristmill.sock"))
        .and_then(|p| p.to_str().map(String::from))
        .unwrap_or_else(|| "/tmp/gristmill.sock".to_string())
}
