//! Unix-socket IPC server — MessagePack-framed request/response protocol.
//!
//! # Frame format
//!
//! Each frame is a **4-byte little-endian u32 length prefix** followed by a
//! **MessagePack-encoded payload**.  One request / one response per frame
//! exchange on the same connection; connections are kept alive for multiple
//! round-trips.
//!
//! # Request envelope
//!
//! ```json
//! { "id": 1, "request": { "method": "triage", ... } }
//! ```
//!
//! # Response envelope
//!
//! Success:  `{ "id": 1, "ok": <value> }`
//! Failure:  `{ "id": 1, "error": "reason string" }`
//!
//! # Socket path
//!
//! Default: `~/.gristmill/gristmill.sock`
//! Override: set the `GRISTMILL_SOCK` environment variable.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use grist_core::GristMillCore;
use grist_event::GristEvent;
use grist_millwright::Pipeline;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tracing::{debug, error, info, warn};

// ─────────────────────────────────────────────────────────────────────────────
// Protocol types
// ─────────────────────────────────────────────────────────────────────────────

/// All operations the daemon supports over the socket.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "method", content = "params", rename_all = "snake_case")]
pub enum IpcRequest {
    /// Ping — returns `{ "status": "ok" }`.
    Health,
    /// Triage a pre-built GristEvent JSON string.
    Triage { event_json: String },
    /// Store a memory and return its ULID id.
    Remember { content: String, tags: Vec<String> },
    /// Recall memories matching a query.
    Recall { query: String, limit: usize },
    /// Retrieve a single memory by ULID.
    GetMemory { id: String },
    /// Escalate a prompt to an LLM provider.
    Escalate { prompt: String, max_tokens: u32 },
    /// Register a pipeline from its JSON serialisation.
    RegisterPipeline { pipeline_json: String },
    /// Run a registered pipeline with the given event.
    RunPipeline { pipeline_id: String, event_json: String },
    /// Return the ids of all registered pipelines.
    PipelineIds,
    /// Build a GristEvent from a channel string + payload JSON.
    BuildEvent { channel: String, payload_json: String },
}

/// Wrapper that pairs a caller-chosen correlation id with a request.
#[derive(Debug, Serialize, Deserialize)]
pub struct IpcEnvelope {
    pub id: u32,
    pub request: IpcRequest,
}

/// Response sent back to the caller.
#[derive(Debug, Serialize, Deserialize)]
pub struct IpcResponse {
    pub id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ok: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl IpcResponse {
    fn ok(id: u32, value: impl Serialize) -> Self {
        let ok = serde_json::to_value(value).unwrap_or(Value::Null);
        Self { id, ok: Some(ok), error: None }
    }

    fn err(id: u32, msg: impl Into<String>) -> Self {
        Self { id, ok: None, error: Some(msg.into()) }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame codec
// ─────────────────────────────────────────────────────────────────────────────

const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024; // 16 MiB safety cap

async fn read_frame(stream: &mut UnixStream) -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_FRAME_BYTES {
        anyhow::bail!("IPC frame too large: {len} bytes (limit {MAX_FRAME_BYTES})");
    }
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body).await?;
    Ok(body)
}

async fn write_frame(stream: &mut UnixStream, data: &[u8]) -> Result<()> {
    let len = (data.len() as u32).to_le_bytes();
    stream.write_all(&len).await?;
    stream.write_all(data).await?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Request dispatch
// ─────────────────────────────────────────────────────────────────────────────

async fn dispatch(core: &Arc<GristMillCore>, envelope: IpcEnvelope) -> IpcResponse {
    let id = envelope.id;
    match envelope.request {
        IpcRequest::Health => IpcResponse::ok(id, serde_json::json!({ "status": "ok" })),

        IpcRequest::Triage { event_json } => {
            match serde_json::from_str::<GristEvent>(&event_json) {
                Err(e) => IpcResponse::err(id, format!("invalid event JSON: {e}")),
                Ok(event) => match core.triage(&event).await {
                    Ok(decision) => IpcResponse::ok(id, decision),
                    Err(e) => IpcResponse::err(id, e.to_string()),
                },
            }
        }

        IpcRequest::Remember { content, tags } => match core.remember(content, tags).await {
            Ok(mem_id) => IpcResponse::ok(id, mem_id),
            Err(e) => IpcResponse::err(id, e.to_string()),
        },

        IpcRequest::Recall { query, limit } => match core.recall(&query, limit).await {
            Ok(results) => IpcResponse::ok(id, results),
            Err(e) => IpcResponse::err(id, e.to_string()),
        },

        IpcRequest::GetMemory { id: mem_id } => match core.get_memory(&mem_id).await {
            Ok(opt) => IpcResponse::ok(id, opt),
            Err(e) => IpcResponse::err(id, e.to_string()),
        },

        IpcRequest::Escalate { prompt, max_tokens } => {
            match core.escalate(prompt, max_tokens).await {
                Ok(result) => IpcResponse::ok(id, result),
                Err(e) => IpcResponse::err(id, e.to_string()),
            }
        }

        IpcRequest::RegisterPipeline { pipeline_json } => {
            match serde_json::from_str::<Pipeline>(&pipeline_json) {
                Err(e) => IpcResponse::err(id, format!("invalid pipeline JSON: {e}")),
                Ok(pipeline) => {
                    core.register_pipeline(pipeline);
                    IpcResponse::ok(id, serde_json::json!({ "registered": true }))
                }
            }
        }

        IpcRequest::RunPipeline { pipeline_id, event_json } => {
            match serde_json::from_str::<GristEvent>(&event_json) {
                Err(e) => IpcResponse::err(id, format!("invalid event JSON: {e}")),
                Ok(event) => match core.run_pipeline(&pipeline_id, &event).await {
                    Ok(result) => IpcResponse::ok(id, result),
                    Err(e) => IpcResponse::err(id, e.to_string()),
                },
            }
        }

        IpcRequest::PipelineIds => IpcResponse::ok(id, core.pipeline_ids()),

        IpcRequest::BuildEvent { channel, payload_json } => {
            match serde_json::from_str::<Value>(&payload_json) {
                Err(e) => IpcResponse::err(id, format!("invalid payload JSON: {e}")),
                Ok(payload) => {
                    let event = GristMillCore::build_event(&channel, payload);
                    match serde_json::to_value(&event) {
                        Ok(v) => IpcResponse::ok(id, v),
                        Err(e) => IpcResponse::err(id, e.to_string()),
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection handler
// ─────────────────────────────────────────────────────────────────────────────

async fn handle_connection(mut stream: UnixStream, core: Arc<GristMillCore>) {
    debug!("IPC: client connected");
    loop {
        let frame = match read_frame(&mut stream).await {
            Ok(f) => f,
            Err(e) => {
                // EOF = clean client disconnect; anything else is a real error.
                let is_eof = e
                    .downcast_ref::<std::io::Error>()
                    .map(|io| io.kind() == std::io::ErrorKind::UnexpectedEof)
                    .unwrap_or(false);
                if !is_eof {
                    warn!("IPC: read error: {e}");
                }
                return;
            }
        };

        let envelope: IpcEnvelope = match rmp_serde::from_slice(&frame) {
            Ok(e) => e,
            Err(e) => {
                warn!("IPC: deserialise error: {e}");
                return;
            }
        };

        let response = dispatch(&core, envelope).await;

        let encoded = match rmp_serde::to_vec_named(&response) {
            Ok(b) => b,
            Err(e) => {
                error!("IPC: serialise response error: {e}");
                return;
            }
        };

        if let Err(e) = write_frame(&mut stream, &encoded).await {
            warn!("IPC: write error: {e}");
            return;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Server
// ─────────────────────────────────────────────────────────────────────────────

pub struct IpcServer {
    socket_path: std::path::PathBuf,
    core: Arc<GristMillCore>,
}

impl IpcServer {
    pub fn new(socket_path: impl AsRef<Path>, core: Arc<GristMillCore>) -> Self {
        Self {
            socket_path: socket_path.as_ref().to_path_buf(),
            core,
        }
    }

    /// Start accepting connections.  Runs forever; cancel via the returned
    /// task handle or by calling `abort()` on the spawned task.
    pub async fn run(self) -> Result<()> {
        // Remove a stale socket left by a previous crash.
        let _ = std::fs::remove_file(&self.socket_path);

        if let Some(parent) = self.socket_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let listener = UnixListener::bind(&self.socket_path)?;
        info!(socket = %self.socket_path.display(), "IPC server listening");

        loop {
            match listener.accept().await {
                Ok((stream, _addr)) => {
                    let core = self.core.clone();
                    tokio::spawn(async move {
                        handle_connection(stream, core).await;
                    });
                }
                Err(e) => {
                    error!("IPC: accept error: {e}");
                }
            }
        }
    }
}
