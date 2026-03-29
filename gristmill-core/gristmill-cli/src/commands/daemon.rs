//! start / stop / status / doctor

use anyhow::{Context, Result};
use colored::Colorize;

use crate::client::IpcClient;

// Unused imports from the parent; kept for clippy hygiene.
#[allow(unused_imports)]
pub use clap::Subcommand;

/// Placeholder enum — daemon sub-commands are handled at the top level.
#[derive(clap::Subcommand)]
pub enum DaemonCmd {}

pub async fn start() -> Result<()> {
    // Locate the daemon binary alongside the CLI binary.
    let exe = std::env::current_exe().context("cannot locate current executable")?;
    let bin_dir = exe.parent().unwrap_or(std::path::Path::new("."));
    let daemon_bin = bin_dir.join("gristmill-daemon");

    if !daemon_bin.exists() {
        anyhow::bail!(
            "daemon binary not found at {}\n  → run `cargo build -p gristmill-daemon` first",
            daemon_bin.display()
        );
    }

    // Create workspace dir so the socket path exists.
    let sock_dir = dirs::home_dir()
        .map(|h| h.join(".gristmill"))
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"));
    std::fs::create_dir_all(&sock_dir).ok();

    let pid_path = sock_dir.join("daemon.pid");

    let child = std::process::Command::new(&daemon_bin)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .with_context(|| format!("failed to spawn {}", daemon_bin.display()))?;

    let pid = child.id();
    std::fs::write(&pid_path, pid.to_string())
        .with_context(|| format!("cannot write PID file {}", pid_path.display()))?;

    println!("{} daemon started (pid {})", "✓".green().bold(), pid);
    Ok(())
}

pub async fn stop(sock: &str) -> Result<()> {
    let pid_path = dirs::home_dir()
        .map(|h| h.join(".gristmill").join("daemon.pid"))
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp/daemon.pid"));

    if pid_path.exists() {
        let pid_str = std::fs::read_to_string(&pid_path).context("reading daemon.pid")?;
        let pid: u32 = pid_str
            .trim()
            .parse()
            .context("invalid PID in daemon.pid")?;
        // SIGTERM
        unsafe { libc_kill(pid as i32, 15) };
        std::fs::remove_file(&pid_path).ok();
        println!("{} daemon stopped (pid {})", "✓".green().bold(), pid);
    } else {
        // Try a graceful IPC stop first, fall back to ENOENT warning.
        match IpcClient::connect(sock).await {
            Ok(_) => {
                println!(
                    "{} no pid file found, but daemon socket exists",
                    "⚠".yellow()
                );
                println!("  send SIGTERM manually or use: kill $(lsof -t {sock})");
            }
            Err(_) => {
                println!("{} daemon does not appear to be running", "⚠".yellow());
            }
        }
    }
    Ok(())
}

pub async fn status(sock: &str) -> Result<()> {
    let mut client = IpcClient::connect(sock).await?;
    let health = client.health().await?;
    let status = health["status"].as_str().unwrap_or("unknown");
    if status == "ok" {
        println!("{} daemon is running", "✓".green().bold());
    } else {
        println!("{} daemon status: {status}", "⚠".yellow().bold());
    }
    Ok(())
}

pub async fn doctor(sock: &str) -> Result<()> {
    println!("{}", "GristMill Doctor".bold());
    println!("{}", "─".repeat(40));

    // 1. Daemon reachability
    match IpcClient::connect(sock).await {
        Ok(mut c) => match c.health().await {
            Ok(h) if h["status"] == "ok" => {
                println!("{} daemon reachable at {sock}", "✓".green());
            }
            Ok(h) => {
                println!(
                    "{} daemon responded but status = {}",
                    "⚠".yellow(),
                    h["status"]
                );
            }
            Err(e) => println!("{} health check failed: {e}", "✗".red()),
        },
        Err(e) => println!("{} cannot reach daemon: {e}", "✗".red()),
    }

    // 2. Config file
    let cfg_path = dirs::home_dir()
        .map(|h| h.join(".gristmill").join("config.yaml"))
        .unwrap_or_default();
    if cfg_path.exists() {
        println!("{} config file found: {}", "✓".green(), cfg_path.display());
    } else {
        println!(
            "{} config file missing: {} (using defaults)",
            "⚠".yellow(),
            cfg_path.display()
        );
    }

    // 3. Model files
    let models_dir = dirs::home_dir()
        .map(|h| h.join(".gristmill").join("models"))
        .unwrap_or_default();
    for model in ["minilm-l6-v2.onnx", "intent-classifier-v1.onnx"] {
        let p = models_dir.join(model);
        if p.exists() {
            println!("{} model file: {model}", "✓".green());
        } else {
            println!(
                "{} model missing: {model}  → run: python gristmill-ml/scripts/bootstrap_models.py",
                "⚠".yellow()
            );
        }
    }

    // 4. Feedback dir
    let feedback_dir = dirs::home_dir()
        .map(|h| h.join(".gristmill").join("feedback"))
        .unwrap_or_default();
    if feedback_dir.exists() {
        let n = std::fs::read_dir(&feedback_dir)
            .map(|d| d.count())
            .unwrap_or(0);
        println!("{} feedback directory: {n} file(s)", "✓".green());
    } else {
        println!(
            "{} feedback directory missing (will be created on first routing decision)",
            "⚠".yellow()
        );
    }

    println!("{}", "─".repeat(40));
    Ok(())
}

/// Safe wrapper around libc kill — avoids pulling in the full `libc` crate.
#[allow(non_snake_case)]
unsafe fn libc_kill(pid: i32, sig: i32) {
    extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    kill(pid, sig);
}
