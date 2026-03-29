//! gristmill pipeline <list|create|run|inspect>

use anyhow::Result;
use clap::Subcommand;
use colored::Colorize;

use crate::client::IpcClient;

#[derive(Subcommand)]
pub enum PipelineCmd {
    /// List registered pipelines.
    List,
    /// Create a pipeline from a JSON file or inline JSON.
    Create {
        /// Path to a pipeline JSON file, or inline JSON string.
        #[arg(long)]
        from: String,
    },
    /// Run a pipeline with an event payload.
    Run {
        /// Pipeline id.
        id: String,
        /// Event payload as a JSON string.
        #[arg(long)]
        input: String,
    },
    /// Inspect the result of a pipeline run (by result id).
    Inspect {
        /// Run result id returned by `pipeline run`.
        run_id: String,
    },
}

pub async fn run(cmd: PipelineCmd, sock: &str) -> Result<()> {
    match cmd {
        PipelineCmd::List => list(sock).await,
        PipelineCmd::Create { from } => create(&from, sock).await,
        PipelineCmd::Run { id, input } => run_pipeline(&id, &input, sock).await,
        PipelineCmd::Inspect { run_id } => inspect(&run_id),
    }
}

async fn list(sock: &str) -> Result<()> {
    let mut client = IpcClient::connect(sock).await?;
    let v = client.pipeline_ids().await?;
    let ids = v.as_array().cloned().unwrap_or_default();
    if ids.is_empty() {
        println!("no pipelines registered");
    } else {
        println!("{}", "Pipelines".bold());
        println!("{}", "─".repeat(40));
        for id in &ids {
            println!("  {}", id.as_str().unwrap_or("?").cyan());
        }
        println!("\n{} pipeline(s) total", ids.len());
    }
    Ok(())
}

async fn create(from: &str, sock: &str) -> Result<()> {
    // `from` can be a file path or a raw JSON string.
    let pipeline_json = if std::path::Path::new(from).exists() {
        std::fs::read_to_string(from)?
    } else {
        from.to_string()
    };

    // Validate it parses as JSON before sending.
    let _: serde_json::Value = serde_json::from_str(&pipeline_json)
        .map_err(|e| anyhow::anyhow!("invalid pipeline JSON: {e}"))?;

    let mut client = IpcClient::connect(sock).await?;
    client
        .call(
            "register_pipeline",
            Some(serde_json::json!({ "pipeline_json": pipeline_json })),
        )
        .await?;
    println!("{} pipeline registered", "✓".green());
    Ok(())
}

async fn run_pipeline(id: &str, input: &str, sock: &str) -> Result<()> {
    // `input` may be a raw payload JSON — wrap it in a GristEvent if needed.
    let event_json = if input.contains("\"id\"") && input.contains("\"channel\"") {
        input.to_string()
    } else {
        // Build a minimal GristEvent around the payload.
        serde_json::json!({
            "id": ulid_now(),
            "channel": "cli",
            "source": "cli",
            "payload": serde_json::from_str::<serde_json::Value>(input)
                .unwrap_or(serde_json::json!({"text": input})),
            "priority": 3,
            "timestamp": chrono_now(),
        })
        .to_string()
    };

    let mut client = IpcClient::connect(sock).await?;
    let result = client.run_pipeline(id, &event_json).await?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn inspect(run_id: &str) -> Result<()> {
    println!(
        "{} run inspection is not yet persisted by the daemon",
        "⚠".yellow()
    );
    println!("  run_id: {run_id}");
    println!("  → Coming in Phase 3: DAG trace storage in grist-ledger warm tier.");
    Ok(())
}

fn ulid_now() -> String {
    // Simple timestamp-based placeholder; real ULIDs need the `ulid` crate.
    format!(
        "{:026X}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    )
}

fn chrono_now() -> String {
    // RFC-3339 timestamp without pulling in chrono.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{secs}")
}
