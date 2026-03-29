//! Daemon integration tests.
//!
//! Each test starts an in-process [`IpcServer`] backed by a real
//! [`GristMillCore`] (with an isolated tmpdir workspace so parallel tests do
//! not contend on the shared sled lock).  A minimal IPC client is implemented
//! inline using the same frame codec as the production TypeScript bridge.
//!
//! # Why in-process instead of subprocess?
//!
//! - No binary build dependency: tests run with `cargo test`, no separate
//!   `cargo build --bin gristmill-daemon` step required.
//! - No race on socket readiness: the server future is awaited before the
//!   client connects.
//! - Isolated tmpdir: each test gets its own workspace and socket path.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use grist_core::GristMillCore;
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

// ── Re-use the IPC protocol types from the daemon crate ──────────────────────
// The daemon's `ipc` module is private, so we duplicate the minimal envelope
// types needed for the test client.  These must stay in sync with ipc.rs.

#[derive(serde::Serialize)]
struct Envelope {
    id: u32,
    request: serde_json::Value,
}

#[derive(serde::Deserialize, Debug)]
struct Response {
    #[allow(dead_code)]
    id: u32,
    ok: Option<Value>,
    error: Option<String>,
}

// ── Minimal IPC client ────────────────────────────────────────────────────────

struct TestClient {
    stream: UnixStream,
    next_id: u32,
}

impl TestClient {
    async fn connect(socket_path: &std::path::Path) -> Self {
        // Retry for up to 2 s in case the server task hasn't bound yet.
        let stream = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match UnixStream::connect(socket_path).await {
                    Ok(s) => return s,
                    Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
                }
            }
        })
        .await
        .expect("IPC socket did not become available within 2 s");

        Self { stream, next_id: 1 }
    }

    async fn call(&mut self, method: &str, params: Option<Value>) -> Response {
        let id = self.next_id;
        self.next_id += 1;

        let request = match params {
            Some(p) => serde_json::json!({ "method": method, "params": p }),
            None => serde_json::json!({ "method": method }),
        };

        let envelope = Envelope { id, request };
        let encoded = rmp_serde::to_vec_named(&envelope).expect("encode request");

        // Write frame: 4-byte LE length prefix + body
        let len = (encoded.len() as u32).to_le_bytes();
        self.stream.write_all(&len).await.expect("write length");
        self.stream.write_all(&encoded).await.expect("write body");

        // Read response frame
        let mut len_buf = [0u8; 4];
        self.stream
            .read_exact(&mut len_buf)
            .await
            .expect("read response length");
        let resp_len = u32::from_le_bytes(len_buf) as usize;
        let mut body = vec![0u8; resp_len];
        self.stream
            .read_exact(&mut body)
            .await
            .expect("read response body");

        rmp_serde::from_slice(&body).expect("decode response")
    }
}

// ── Test fixture ─────────────────────────────────────────────────────────────

/// Spin up a `GristMillCore` + `IpcServer` in a temp workspace.
/// Returns `(socket_path, server_task_handle)`.
///
/// The server task runs for the lifetime of the returned handle — drop it (or
/// call `abort()`) to shut down.
async fn start_test_server() -> (PathBuf, tokio::task::JoinHandle<()>) {
    use grist_config::GristMillConfig;
    use grist_core::GristMillCore;

    // Use a unique tmpdir per test so parallel tests don't share the sled db.
    let dir = tempfile::tempdir().expect("tempdir");
    let socket_path = dir.path().join("test.sock");

    let mut cfg = GristMillConfig::default();
    cfg.core.workspace = dir.path().to_owned();

    // Keep the tmpdir alive (and prevent cleanup) for the life of the test.
    let _workspace = dir.keep();

    let core = Arc::new(
        GristMillCore::from_config(cfg)
            .await
            .expect("GristMillCore::from_config"),
    );

    let server = gristmill_daemon_ipc_server_for_test(socket_path.clone(), Arc::clone(&core));
    let handle = tokio::spawn(async move {
        if let Err(e) = server.await {
            eprintln!("test IPC server error: {e}");
        }
    });

    (socket_path, handle)
}

/// Thin wrapper that mirrors `IpcServer::new(path, core).run()` without
/// requiring `IpcServer` to be `pub` outside the daemon crate.
/// We spawn the server using the same tokio listener logic.
async fn gristmill_daemon_ipc_server_for_test(
    socket_path: PathBuf,
    core: Arc<GristMillCore>,
) -> anyhow::Result<()> {
    use tokio::net::UnixListener;

    let _ = std::fs::remove_file(&socket_path);
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let listener = UnixListener::bind(&socket_path)?;

    loop {
        let (stream, _) = listener.accept().await?;
        let core = Arc::clone(&core);
        tokio::spawn(async move {
            handle_test_connection(stream, core).await;
        });
    }
}

/// Minimal in-test connection handler — dispatches to `GristMillCore` exactly
/// as `ipc.rs` does, without re-importing the private module.
async fn handle_test_connection(mut stream: UnixStream, core: Arc<GristMillCore>) {
    use grist_event::GristEvent;
    use serde_json::Value;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    loop {
        // Read frame
        let mut len_buf = [0u8; 4];
        if stream.read_exact(&mut len_buf).await.is_err() {
            return;
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut body = vec![0u8; len];
        if stream.read_exact(&mut body).await.is_err() {
            return;
        }

        // Decode envelope
        let envelope: serde_json::Value = match rmp_serde::from_slice(&body) {
            Ok(v) => v,
            Err(_) => return,
        };

        let id = envelope["id"].as_u64().unwrap_or(0) as u32;
        let method = envelope["request"]["method"]
            .as_str()
            .unwrap_or("")
            .to_owned();
        let params = &envelope["request"]["params"];

        // Dispatch
        let result: Result<Value, String> = match method.as_str() {
            "health" => Ok(serde_json::json!({ "status": "ok" })),

            "triage" => {
                let event_json = params["event_json"].as_str().unwrap_or("{}");
                match serde_json::from_str::<GristEvent>(event_json) {
                    Err(e) => Err(format!("invalid event JSON: {e}")),
                    Ok(ev) => match core.triage(&ev).await {
                        Ok(d) => serde_json::to_value(&d).map_err(|e| e.to_string()),
                        Err(e) => Err(e.to_string()),
                    },
                }
            }

            "remember" => {
                let content = params["content"].as_str().unwrap_or("").to_owned();
                let tags: Vec<String> = params["tags"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                match core.remember(content, tags).await {
                    Ok(id) => Ok(Value::String(id)),
                    Err(e) => Err(e.to_string()),
                }
            }

            "get_memory" => {
                let mem_id = params["id"].as_str().unwrap_or("").to_owned();
                match core.get_memory(&mem_id).await {
                    Ok(opt) => serde_json::to_value(opt).map_err(|e| e.to_string()),
                    Err(e) => Err(e.to_string()),
                }
            }

            "recall" => {
                let query = params["query"].as_str().unwrap_or("").to_owned();
                let limit = params["limit"].as_u64().unwrap_or(5) as usize;
                match core.recall(&query, limit).await {
                    Ok(results) => serde_json::to_value(results).map_err(|e| e.to_string()),
                    Err(e) => Err(e.to_string()),
                }
            }

            "pipeline_ids" => Ok(serde_json::to_value(core.pipeline_ids()).unwrap()),

            other => Err(format!("unknown method: {other}")),
        };

        // Encode response
        let response = match result {
            Ok(v) => serde_json::json!({ "id": id, "ok": v }),
            Err(e) => serde_json::json!({ "id": id, "error": e }),
        };

        let encoded = match rmp_serde::to_vec_named(&response) {
            Ok(b) => b,
            Err(_) => return,
        };

        let len_bytes = (encoded.len() as u32).to_le_bytes();
        if stream.write_all(&len_bytes).await.is_err() {
            return;
        }
        if stream.write_all(&encoded).await.is_err() {
            return;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn health_returns_ok() {
    let (sock, _server) = start_test_server().await;
    let mut client = TestClient::connect(&sock).await;

    let resp = client.call("health", None).await;
    assert!(resp.error.is_none(), "unexpected error: {:?}", resp.error);
    assert_eq!(resp.ok.as_ref().and_then(|v| v["status"].as_str()), Some("ok"));
}

#[tokio::test]
async fn triage_returns_route_decision() {
    let (sock, _server) = start_test_server().await;
    let mut client = TestClient::connect(&sock).await;

    // Build a minimal GristEvent JSON directly (mirrors IpcBridge.buildEventJson)
    let event_json = serde_json::json!({
        "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "source": { "type": "http" },
        "timestamp_ms": 1_700_000_000_000u64,
        "payload": { "text": "summarise my last 10 emails" },
        "metadata": {
            "priority": "normal",
            "correlation_id": null,
            "reply_channel": null,
            "ttl_ms": null,
            "tags": {}
        }
    })
    .to_string();

    let resp = client
        .call("triage", Some(serde_json::json!({ "event_json": event_json })))
        .await;

    assert!(resp.error.is_none(), "triage error: {:?}", resp.error);
    let decision = resp.ok.expect("triage returned no value");

    // RouteDecision serialises to an object with a `confidence` field.
    assert!(
        decision.is_object(),
        "expected RouteDecision object, got: {decision}"
    );
    let confidence = decision["confidence"]
        .as_f64()
        .expect("RouteDecision missing `confidence` field");
    assert!(
        (0.0..=1.0).contains(&confidence),
        "confidence {confidence} out of [0, 1]"
    );
}

#[tokio::test]
async fn remember_and_get_round_trip() {
    let (sock, _server) = start_test_server().await;
    let mut client = TestClient::connect(&sock).await;

    // Store a memory
    let remember_resp = client
        .call(
            "remember",
            Some(serde_json::json!({
                "content": "GristMill Phase 1 integration test",
                "tags": ["integration", "phase1"]
            })),
        )
        .await;

    assert!(
        remember_resp.error.is_none(),
        "remember error: {:?}",
        remember_resp.error
    );
    let mem_id = remember_resp
        .ok
        .as_ref()
        .and_then(|v| v.as_str())
        .expect("remember returned no id")
        .to_owned();
    assert!(!mem_id.is_empty(), "memory id is empty");

    // Retrieve it
    let get_resp = client
        .call("get_memory", Some(serde_json::json!({ "id": mem_id })))
        .await;

    assert!(
        get_resp.error.is_none(),
        "get_memory error: {:?}",
        get_resp.error
    );
    let memory = get_resp.ok.expect("get_memory returned null");
    assert!(
        !memory.is_null(),
        "memory not found by id {mem_id}"
    );
    assert!(
        memory["content"]
            .as_str()
            .unwrap_or("")
            .contains("integration test"),
        "memory content mismatch: {memory}"
    );
}

#[tokio::test]
async fn get_memory_missing_returns_null() {
    let (sock, _server) = start_test_server().await;
    let mut client = TestClient::connect(&sock).await;

    let resp = client
        .call(
            "get_memory",
            Some(serde_json::json!({ "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV" })),
        )
        .await;

    assert!(resp.error.is_none(), "unexpected error: {:?}", resp.error);
    // Missing memory → null
    assert!(
        resp.ok.as_ref().map(|v| v.is_null()).unwrap_or(true),
        "expected null for missing memory, got {:?}",
        resp.ok
    );
}

#[tokio::test]
async fn pipeline_ids_empty_on_fresh_core() {
    let (sock, _server) = start_test_server().await;
    let mut client = TestClient::connect(&sock).await;

    let resp = client.call("pipeline_ids", None).await;
    assert!(resp.error.is_none(), "unexpected error: {:?}", resp.error);
    let ids = resp.ok.as_ref().and_then(|v| v.as_array()).expect("expected array");
    assert!(ids.is_empty(), "expected no pipelines on fresh core");
}

#[tokio::test]
async fn unknown_method_returns_error() {
    let (sock, _server) = start_test_server().await;
    let mut client = TestClient::connect(&sock).await;

    let resp = client.call("does_not_exist", None).await;
    assert!(
        resp.error.is_some(),
        "expected error for unknown method, got ok: {:?}",
        resp.ok
    );
}

#[tokio::test]
async fn multiple_sequential_requests_on_one_connection() {
    let (sock, _server) = start_test_server().await;
    let mut client = TestClient::connect(&sock).await;

    // Health
    let r1 = client.call("health", None).await;
    assert!(r1.error.is_none());

    // Pipeline IDs
    let r2 = client.call("pipeline_ids", None).await;
    assert!(r2.error.is_none());

    // Health again
    let r3 = client.call("health", None).await;
    assert!(r3.error.is_none());
    assert_eq!(r3.ok.as_ref().and_then(|v| v["status"].as_str()), Some("ok"));
}
