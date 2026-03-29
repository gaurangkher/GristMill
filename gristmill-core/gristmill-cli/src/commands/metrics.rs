//! gristmill metrics [--period <duration>]

use anyhow::Result;
use clap::Args;
use colored::Colorize;

use crate::client::IpcClient;

#[derive(Args)]
pub struct MetricsCmd {
    /// Time window (e.g. `1h`, `7d`). Currently informational only.
    #[arg(long, short = 'p')]
    period: Option<String>,
}

pub async fn run(cmd: MetricsCmd, sock: &str) -> Result<()> {
    let mut client = IpcClient::connect(sock).await?;
    let v = client.metrics().await?;

    if let Some(p) = &cmd.period {
        println!("{} (period: {})", "System Metrics".bold(), p.cyan());
    } else {
        println!("{}", "System Metrics".bold());
    }
    println!("{}", "─".repeat(50));

    if let Some(sieve) = v.get("sieve") {
        println!("\n  {}", "Sieve (routing)".underline());
        println!(
            "    confidence threshold : {:.2}",
            sieve["confidence_threshold"].as_f64().unwrap_or(0.0)
        );
        if let Some(rc) = sieve.get("routing_cache") {
            let exact = rc["exact_hits"].as_u64().unwrap_or(0);
            let sem = rc["semantic_hits"].as_u64().unwrap_or(0);
            let misses = rc["misses"].as_u64().unwrap_or(0);
            let total = exact + sem + misses;
            let hit_pct = if total > 0 {
                (exact + sem) as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            println!("    routing cache hits   : {exact} exact  +  {sem} semantic");
            println!("    routing cache misses : {misses}");
            println!("    overall hit rate     : {hit_pct:.1}%");
        }
        println!(
            "    feedback records sent: {}",
            sieve["feedback_records_sent"]
        );
    }

    if let Some(hammer) = v.get("hammer") {
        println!("\n  {}", "Hammer (LLM gateway)".underline());
        println!("    semantic cache size  : {}", hammer["cache_size"]);
        let used = hammer["daily_tokens_used"].as_u64().unwrap_or(0);
        let limit = hammer["daily_token_limit"].as_u64().unwrap_or(0);
        let rem = hammer["daily_tokens_remaining"].as_u64().unwrap_or(0);
        println!("    daily tokens used    : {used}");
        println!("    daily token limit    : {limit}");
        println!("    daily tokens left    : {rem}");
        let monthly_used = hammer["monthly_tokens_used"].as_u64().unwrap_or(0);
        let monthly_limit = hammer["monthly_token_limit"].as_u64().unwrap_or(0);
        println!("    monthly tokens used  : {monthly_used} / {monthly_limit}");
    }

    if let Some(n) = v.get("pipelines_registered") {
        println!("\n  {}", "Pipelines".underline());
        println!("    registered: {n}");
    }

    println!();
    Ok(())
}
