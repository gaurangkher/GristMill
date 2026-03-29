//! gristmill watch <list|create|test>

use anyhow::Result;
use clap::Subcommand;
use colored::Colorize;

#[derive(Subcommand)]
pub enum WatchCmd {
    /// List active watches.
    List,
    /// Create a new watch rule.
    Create {
        /// Human-readable name.
        #[arg(long)]
        name: String,
        /// Condition expression (e.g. `confidence < 0.5`).
        #[arg(long)]
        condition: String,
        /// Notification channel (`slack`, `email`, `webhook`, …).
        #[arg(long)]
        channel: String,
    },
    /// Send a test notification for a watch.
    Test {
        /// Watch id.
        id: String,
    },
}

pub async fn run(cmd: WatchCmd, _sock: &str) -> Result<()> {
    match cmd {
        WatchCmd::List => list(),
        WatchCmd::Create {
            name,
            condition,
            channel,
        } => create(&name, &condition, &channel),
        WatchCmd::Test { id } => test_watch(&id),
    }
}

fn list() -> Result<()> {
    println!(
        "{} watch management is handled by the TypeScript bell-tower.",
        "ℹ".cyan()
    );
    println!("  → Use the dashboard at http://localhost:3000 or the TS SDK.");
    println!("  → gristmill-integrations bell-tower manages watch rules.");
    Ok(())
}

fn create(name: &str, condition: &str, channel: &str) -> Result<()> {
    println!(
        "{} watch creation delegated to TypeScript bell-tower.",
        "ℹ".cyan()
    );
    println!("  name      : {name}");
    println!("  condition : {condition}");
    println!("  channel   : {channel}");
    println!("  → POST http://localhost:3000/api/watches  with the above payload.");
    Ok(())
}

fn test_watch(id: &str) -> Result<()> {
    println!(
        "{} watch test delegated to TypeScript bell-tower.",
        "ℹ".cyan()
    );
    println!("  → POST http://localhost:3000/api/watches/{id}/test");
    Ok(())
}
