//! gristmill models <list|pull|reload|benchmark>

use anyhow::Result;
use clap::Subcommand;
use colored::Colorize;

use crate::client::IpcClient;

#[derive(Subcommand)]
pub enum ModelsCmd {
    /// List registered models and their status.
    List,
    /// Download a model pack (delegates to bootstrap_models.py).
    Pull {
        /// Pack name (e.g. `starter-pack`).
        pack: String,
    },
    /// Hot-reload a model without restarting the daemon.
    Reload {
        /// Model identifier (`sieve` or a grinder model id).
        model: String,
    },
    /// Run inference benchmarks and report p50/p99 latencies.
    Benchmark,
}

pub async fn run(cmd: ModelsCmd, sock: &str) -> Result<()> {
    match cmd {
        ModelsCmd::List => list(sock).await,
        ModelsCmd::Pull { pack } => pull(&pack).await,
        ModelsCmd::Reload { model } => reload(&model, sock).await,
        ModelsCmd::Benchmark => benchmark(sock).await,
    }
}

async fn list(sock: &str) -> Result<()> {
    let mut client = IpcClient::connect(sock).await?;
    let v = client.models_list().await?;

    println!("{}", "Models".bold());
    println!("{}", "─".repeat(50));

    if let Some(models) = v["models"].as_array() {
        for m in models {
            let id = m["id"].as_str().unwrap_or("?");
            let mtype = m["type"].as_str().unwrap_or("?");
            let status = m["status"].as_str().unwrap_or("?");
            let thresh = m["confidence_threshold"].as_f64().unwrap_or(0.0);
            println!(
                "  {:<28} {:>6}  {}  threshold={:.2}",
                id.cyan(),
                mtype,
                if status == "loaded" {
                    "●".green().to_string()
                } else {
                    "○".red().to_string()
                },
                thresh
            );
        }
    }

    if let Some(hammer) = v.get("hammer") {
        println!("\n{}", "Hammer (LLM gateway)".bold());
        let cache_size = hammer["cache_size"].as_u64().unwrap_or(0);
        println!("  semantic cache entries: {cache_size}");
    }
    Ok(())
}

async fn pull(pack: &str) -> Result<()> {
    println!("{} pulling model pack: {pack}", "→".cyan());
    println!("  delegating to: python gristmill-ml/scripts/bootstrap_models.py");
    let status = std::process::Command::new("python")
        .args(["gristmill-ml/scripts/bootstrap_models.py"])
        .status();
    match status {
        Ok(s) if s.success() => println!("{} pack pulled", "✓".green()),
        Ok(s) => anyhow::bail!("bootstrap script exited with status {s}"),
        Err(e) => anyhow::bail!("failed to run bootstrap script: {e}"),
    }
    Ok(())
}

async fn reload(model: &str, sock: &str) -> Result<()> {
    let mut client = IpcClient::connect(sock).await?;
    let v = client.models_reload(model).await?;
    println!("{} model '{}' reloaded: {}", "✓".green(), model, v);
    Ok(())
}

async fn benchmark(sock: &str) -> Result<()> {
    println!("{}", "Inference Benchmark".bold());
    println!("{}", "─".repeat(50));

    let mut client = IpcClient::connect(sock).await?;

    // Use a simple triage loop as a proxy latency benchmark.
    let sample_event = serde_json::json!({
        "id": "01HZ000000000000000000000",
        "channel": "benchmark",
        "source": "cli",
        "payload": { "text": "benchmark latency probe" },
        "priority": 3,
        "timestamp": "2026-01-01T00:00:00Z"
    })
    .to_string();

    let n = 20u32;
    let mut latencies: Vec<u128> = Vec::with_capacity(n as usize);

    for _ in 0..n {
        let t0 = std::time::Instant::now();
        client.triage(&sample_event).await?;
        latencies.push(t0.elapsed().as_micros());
    }

    latencies.sort_unstable();
    let p50 = latencies[latencies.len() / 2];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
    let mean: u128 = latencies.iter().sum::<u128>() / latencies.len() as u128;

    println!("  iterations : {n}");
    println!("  mean       : {:.2} ms", mean as f64 / 1000.0);
    println!("  p50        : {:.2} ms", p50 as f64 / 1000.0);
    println!("  p99        : {:.2} ms", p99 as f64 / 1000.0);

    let target_ms = 5.0f64;
    if (p99 as f64 / 1000.0) <= target_ms {
        println!("{} p99 within 5 ms target", "✓".green());
    } else {
        println!("{} p99 exceeds 5 ms target", "⚠".yellow());
    }
    Ok(())
}
