//! gristmill train <sieve|export|evaluate>
//! Delegates to the Python shell (gristmill-ml).

use anyhow::Result;
use clap::Subcommand;
use colored::Colorize;

#[derive(Subcommand)]
pub enum TrainCmd {
    /// Retrain the Sieve classifier from feedback logs.
    Sieve {
        /// Number of training epochs.
        #[arg(long, default_value = "10")]
        epochs: u32,
    },
    /// Export a trained model to ONNX.
    Export {
        /// Model name to export (`sieve`).
        #[arg(long)]
        model: String,
        /// Quantization mode (`int8` or `fp32`).
        #[arg(long, default_value = "int8")]
        quantize: String,
    },
    /// Compare two model versions on a held-out evaluation set.
    Evaluate {
        /// Candidate model path / id.
        #[arg(long)]
        model: String,
        /// Baseline model path / id.
        #[arg(long)]
        against: String,
    },
}

pub async fn run(cmd: TrainCmd) -> Result<()> {
    match cmd {
        TrainCmd::Sieve { epochs } => train_sieve(epochs).await,
        TrainCmd::Export { model, quantize } => export_model(&model, &quantize).await,
        TrainCmd::Evaluate { model, against } => evaluate(&model, &against).await,
    }
}

async fn train_sieve(epochs: u32) -> Result<()> {
    println!("{} delegating to Python shell …", "→".cyan());
    let script = locate_script("bootstrap_models.py")?;
    let status = std::process::Command::new("python")
        .arg(&script)
        .arg("--classifier-epochs")
        .arg(epochs.to_string())
        .status()
        .map_err(|e| anyhow::anyhow!("failed to run python: {e}"))?;

    if status.success() {
        println!("{} sieve training complete ({epochs} epochs)", "✓".green());
    } else {
        anyhow::bail!("training script exited with {status}");
    }
    Ok(())
}

async fn export_model(model: &str, quantize: &str) -> Result<()> {
    println!(
        "{} exporting model '{model}' (quantize={quantize}) …",
        "→".cyan()
    );
    let script = locate_script("bootstrap_models.py")?;
    let mut cmd = std::process::Command::new("python");
    cmd.arg(&script);
    if quantize == "fp32" {
        cmd.arg("--no-quantize");
    }
    if model == "sieve" {
        cmd.arg("--classifier-only");
    } else if model == "embedder" {
        cmd.arg("--embedder-only");
    }
    let status = cmd
        .status()
        .map_err(|e| anyhow::anyhow!("failed to run python: {e}"))?;
    if status.success() {
        println!("{} export complete", "✓".green());
    } else {
        anyhow::bail!("export script exited with {status}");
    }
    Ok(())
}

async fn evaluate(model: &str, against: &str) -> Result<()> {
    println!("{} model evaluation …", "→".cyan());
    println!("  candidate : {model}");
    println!("  baseline  : {against}");
    println!("  → validate.py in gristmill-ml/src/gristmill_ml/export/");
    let validate = locate_validate_script();
    if let Ok(script) = validate {
        let status = std::process::Command::new("python")
            .arg(&script)
            .arg("--candidate")
            .arg(model)
            .arg("--baseline")
            .arg(against)
            .status()
            .map_err(|e| anyhow::anyhow!("failed to run python: {e}"))?;
        if !status.success() {
            anyhow::bail!("evaluate script exited with {status}");
        }
    } else {
        println!("{} validate.py not found — skipping", "⚠".yellow());
    }
    Ok(())
}

fn locate_script(name: &str) -> Result<std::path::PathBuf> {
    // Try relative to cwd (running from workspace root).
    let candidates = [
        format!("gristmill-ml/scripts/{name}"),
        format!("../gristmill-ml/scripts/{name}"),
    ];
    for c in &candidates {
        let p = std::path::Path::new(c);
        if p.exists() {
            return Ok(p.to_path_buf());
        }
    }
    anyhow::bail!("cannot find {name} — run from GristMill workspace root")
}

fn locate_validate_script() -> Result<std::path::PathBuf> {
    let candidates = [
        "gristmill-ml/src/gristmill_ml/export/validate.py",
        "../gristmill-ml/src/gristmill_ml/export/validate.py",
    ];
    for c in &candidates {
        let p = std::path::Path::new(c);
        if p.exists() {
            return Ok(p.to_path_buf());
        }
    }
    anyhow::bail!("validate.py not found")
}
