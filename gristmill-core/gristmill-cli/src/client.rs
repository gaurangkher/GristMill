//! Minimal IPC client — same 4-byte-LE-u32 + MessagePack frame codec as the
//! daemon's `IpcServer`.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::Path;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

const MAX_FRAME: usize = 16 * 1024 * 1024;

pub struct IpcClient {
    stream: UnixStream,
    next_id: u32,
}

impl IpcClient {
    pub async fn connect(socket_path: &str) -> Result<Self> {
        let stream = UnixStream::connect(Path::new(socket_path))
            .await
            .with_context(|| format!("cannot connect to daemon socket at {socket_path}\n  → is the daemon running? try: gristmill start"))?;
        Ok(Self { stream, next_id: 1 })
    }

    /// Send a method call and return the `ok` payload on success.
    pub async fn call(&mut self, method: &str, params: Option<Value>) -> Result<Value> {
        let request = match params {
            Some(p) => serde_json::json!({ "method": method, "params": p }),
            None => serde_json::json!({ "method": method }),
        };
        let envelope = serde_json::json!({ "id": self.next_id, "request": request });
        self.next_id += 1;

        let encoded = rmp_serde::to_vec_named(&envelope).context("failed to encode IPC request")?;
        write_frame(&mut self.stream, &encoded).await?;

        let frame = read_frame(&mut self.stream).await?;
        let response: Value =
            rmp_serde::from_slice(&frame).context("failed to decode IPC response")?;

        if let Some(err) = response.get("error").and_then(|e| e.as_str()) {
            bail!("daemon error: {err}");
        }
        Ok(response["ok"].clone())
    }

    // ── Convenience wrappers ────────────────────────────────────────────────

    pub async fn health(&mut self) -> Result<Value> {
        self.call("health", None).await
    }

    pub async fn triage(&mut self, event_json: &str) -> Result<Value> {
        self.call(
            "triage",
            Some(serde_json::json!({ "event_json": event_json })),
        )
        .await
    }

    #[allow(dead_code)]
    pub async fn remember(&mut self, content: &str, tags: Vec<String>) -> Result<Value> {
        self.call(
            "remember",
            Some(serde_json::json!({ "content": content, "tags": tags })),
        )
        .await
    }

    pub async fn recall(&mut self, query: &str, limit: usize) -> Result<Value> {
        self.call(
            "recall",
            Some(serde_json::json!({ "query": query, "limit": limit })),
        )
        .await
    }

    pub async fn models_list(&mut self) -> Result<Value> {
        self.call("models_list", None).await
    }

    pub async fn models_reload(&mut self, model_id: &str) -> Result<Value> {
        self.call(
            "models_reload",
            Some(serde_json::json!({ "model_id": model_id })),
        )
        .await
    }

    pub async fn memory_stats(&mut self) -> Result<Value> {
        self.call("memory_stats", None).await
    }

    pub async fn compact(&mut self) -> Result<Value> {
        self.call("compact", None).await
    }

    pub async fn metrics(&mut self) -> Result<Value> {
        self.call("metrics", None).await
    }

    pub async fn pipeline_ids(&mut self) -> Result<Value> {
        self.call("pipeline_ids", None).await
    }

    pub async fn run_pipeline(&mut self, pipeline_id: &str, event_json: &str) -> Result<Value> {
        self.call(
            "run_pipeline",
            Some(serde_json::json!({
                "pipeline_id": pipeline_id,
                "event_json": event_json,
            })),
        )
        .await
    }
}

async fn read_frame(stream: &mut UnixStream) -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream
        .read_exact(&mut len_buf)
        .await
        .context("reading frame length")?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_FRAME {
        bail!("IPC frame too large: {len} bytes");
    }
    let mut body = vec![0u8; len];
    stream
        .read_exact(&mut body)
        .await
        .context("reading frame body")?;
    Ok(body)
}

async fn write_frame(stream: &mut UnixStream, data: &[u8]) -> Result<()> {
    let len = (data.len() as u32).to_le_bytes();
    stream
        .write_all(&len)
        .await
        .context("writing frame length")?;
    stream.write_all(data).await.context("writing frame body")?;
    Ok(())
}
