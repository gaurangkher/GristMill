//! gristmill memory <search|stats|compact|export>

use anyhow::Result;
use clap::Subcommand;
use colored::Colorize;

use crate::client::IpcClient;

#[derive(Subcommand)]
pub enum MemoryCmd {
    /// Search memories by semantic query.
    Search {
        /// Query string.
        query: String,
        /// Maximum number of results.
        #[arg(long, short = 'n', default_value = "10")]
        limit: usize,
    },
    /// Show memory-tier sizes and routing-cache statistics.
    Stats,
    /// Trigger a manual warm→cold compaction cycle.
    Compact {
        /// Skip confirmation prompt.
        #[arg(long)]
        force: bool,
    },
    /// Export all memories to stdout as JSON-lines.
    Export {
        /// Output format (`json` or `jsonl`).
        #[arg(long, default_value = "json")]
        format: String,
    },
}

pub async fn run(cmd: MemoryCmd, sock: &str) -> Result<()> {
    match cmd {
        MemoryCmd::Search { query, limit } => search(&query, limit, sock).await,
        MemoryCmd::Stats => stats(sock).await,
        MemoryCmd::Compact { force } => compact(force, sock).await,
        MemoryCmd::Export { format } => export(&format, sock).await,
    }
}

async fn search(query: &str, limit: usize, sock: &str) -> Result<()> {
    let mut client = IpcClient::connect(sock).await?;
    let v = client.recall(query, limit).await?;

    let results = v.as_array().cloned().unwrap_or_default();
    if results.is_empty() {
        println!("no memories found for {:?}", query);
        return Ok(());
    }

    println!("{} result(s) for {:?}", results.len(), query);
    println!("{}", "─".repeat(60));
    for (i, m) in results.iter().enumerate() {
        let id = m["id"].as_str().unwrap_or("?");
        let score = m["score"].as_f64().unwrap_or(0.0);
        let content = m["content"].as_str().unwrap_or("?");
        let tags: Vec<&str> = m["tags"]
            .as_array()
            .map(|a| a.iter().filter_map(|t| t.as_str()).collect())
            .unwrap_or_default();
        println!(
            "  {}. [{}]  score={:.3}  tags=[{}]",
            i + 1,
            id.dimmed(),
            score,
            tags.join(", ")
        );
        let preview = if content.len() > 120 {
            &content[..120]
        } else {
            content
        };
        println!("     {}", preview);
        println!();
    }
    Ok(())
}

async fn stats(sock: &str) -> Result<()> {
    let mut client = IpcClient::connect(sock).await?;
    let v = client.memory_stats().await?;

    println!("{}", "Memory Stats".bold());
    println!("{}", "─".repeat(40));

    if let Some(rc) = v.get("routing_cache") {
        println!("  Routing cache");
        println!("    exact_hits    : {}", rc["exact_hits"]);
        println!("    semantic_hits : {}", rc["semantic_hits"]);
        println!("    misses        : {}", rc["misses"]);
        println!(
            "    hit_rate      : {:.1}%",
            rc["hit_rate"].as_f64().unwrap_or(0.0) * 100.0
        );
        println!("    exact_size    : {}", rc["exact_size"]);
        println!("    semantic_size : {}", rc["semantic_size"]);
    }

    if let Some(b) = v.get("hammer_budget") {
        println!("\n  LLM budget");
        println!("    daily_used      : {}", b["daily_used"]);
        println!("    daily_limit     : {}", b["daily_limit"]);
        println!("    daily_remaining : {}", b["daily_remaining"]);
    }

    println!("\n  Feedback records sent: {}", v["feedback_records_sent"]);
    Ok(())
}

async fn compact(force: bool, sock: &str) -> Result<()> {
    if !force {
        println!("This will flush the warm tier to cold storage.");
        println!("Re-run with --force to proceed.");
        return Ok(());
    }
    let mut client = IpcClient::connect(sock).await?;
    let v = client.compact().await?;
    println!("{} compaction triggered", "✓".green());
    println!("  {}", v["note"].as_str().unwrap_or(""));
    Ok(())
}

async fn export(format: &str, sock: &str) -> Result<()> {
    // Recall a large batch and dump to stdout.
    let mut client = IpcClient::connect(sock).await?;
    let v = client.recall("", 10_000).await?;
    let results = v.as_array().cloned().unwrap_or_default();

    match format {
        "jsonl" => {
            for m in &results {
                println!("{}", serde_json::to_string(m)?);
            }
        }
        _ => {
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
    }
    Ok(())
}
