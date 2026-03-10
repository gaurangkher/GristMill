//! napi-rs bridge — exposes [`GristMillCore`] to Node.js.
//!
//! All async methods return JavaScript `Promise<string>` where the string is a
//! JSON-serialised Rust type.  JSON transport is used throughout to avoid
//! complex napi-rs type-mapping for deeply nested Rust enums.
//!
//! The bridge keeps a dedicated multi-thread Tokio runtime alive for the
//! lifetime of the [`GristMillBridge`] instance so that background tasks
//! spawned during `GristMillCore::new()` (Ledger compactor, Sieve feedback
//! writer, etc.) continue running.
//!
//! Enabled by `--features node`.

use std::sync::{Arc, Mutex};

use napi::bindgen_prelude::*;
use napi_derive::napi;

use grist_event::GristEvent;
use grist_millwright::dag::Pipeline;

use crate::core::GristMillCore;

// ─────────────────────────────────────────────────────────────────────────────
// GristMillBridge
// ─────────────────────────────────────────────────────────────────────────────

/// Node.js-visible wrapper around [`GristMillCore`].
///
/// ```js
/// const { GristMillBridge } = require('@gristmill/core');
/// const bridge = new GristMillBridge(null);
/// const decision = JSON.parse(await bridge.triage(eventJson));
/// ```
#[napi]
pub struct GristMillBridge {
    inner: Arc<GristMillCore>,
    /// Keep the runtime alive so background Tokio tasks remain running.
    _rt: Arc<tokio::runtime::Runtime>,
}

#[napi]
impl GristMillBridge {
    /// Create and initialise the GristMill core.
    ///
    /// `configPath` — optional path to `~/.gristmill/config.yaml`.
    /// Pass `null` / `undefined` to use built-in defaults.
    #[napi(constructor)]
    pub fn new(config_path: Option<String>) -> napi::Result<Self> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;

        let path = config_path.map(std::path::PathBuf::from);
        let inner = rt
            .block_on(GristMillCore::new(path))
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(inner),
            _rt: Arc::new(rt),
        })
    }

    // ── Sieve ────────────────────────────────────────────────────────────────

    /// Triage an event.
    ///
    /// `eventJson` — a [`GristEvent`] serialised as JSON
    /// (use `buildEvent()` to construct one).
    ///
    /// Returns a JSON string: `{"route":"LOCAL_ML","confidence":0.93,...}`.
    #[napi]
    pub async fn triage(&self, event_json: String) -> napi::Result<String> {
        let inner = self.inner.clone();
        let event: GristEvent = serde_json::from_str(&event_json)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let result = inner
            .triage(&event)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_string(&result)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    // ── Ledger ───────────────────────────────────────────────────────────────

    /// Store a text memory and return its ULID string id.
    #[napi]
    pub async fn remember(
        &self,
        content: String,
        tags: Vec<String>,
    ) -> napi::Result<String> {
        let inner = self.inner.clone();
        inner
            .remember(content, tags)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Semantic + keyword recall.
    ///
    /// Returns a JSON array of ranked memories:
    /// `[{"memory":{...},"score":0.87,"sources":["keyword"]},...]`.
    #[napi]
    pub async fn recall(
        &self,
        query: String,
        limit: u32,
    ) -> napi::Result<String> {
        let inner = self.inner.clone();
        let results = inner
            .recall(&query, limit as usize)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_string(&results)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Retrieve a single memory by ULID id.
    ///
    /// Returns the memory JSON string or `null` if not found.
    #[napi]
    pub async fn get_memory(
        &self,
        id: String,
    ) -> napi::Result<Option<String>> {
        let inner = self.inner.clone();
        let result = inner
            .get_memory(&id)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        match result {
            Some(m) => Ok(Some(
                serde_json::to_string(&m)
                    .map_err(|e| napi::Error::from_reason(e.to_string()))?,
            )),
            None => Ok(None),
        }
    }

    // ── Hammer ───────────────────────────────────────────────────────────────

    /// Escalate a prompt to an LLM provider.
    ///
    /// Returns a JSON string: `{"content":"...","provider":"...","cache_hit":false,...}`.
    #[napi]
    pub async fn escalate(
        &self,
        prompt: String,
        max_tokens: u32,
    ) -> napi::Result<String> {
        let inner = self.inner.clone();
        let result = inner
            .escalate(prompt, max_tokens)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_string(&result)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    // ── Millwright ───────────────────────────────────────────────────────────

    /// Register a pipeline from its JSON serialisation.
    #[napi]
    pub fn register_pipeline(&self, pipeline_json: String) -> napi::Result<()> {
        let pipeline: Pipeline = serde_json::from_str(&pipeline_json)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        self.inner.register_pipeline(pipeline);
        Ok(())
    }

    /// Run a registered pipeline with the given event.
    ///
    /// Returns the `PipelineResult` as a JSON string.
    #[napi]
    pub async fn run_pipeline(
        &self,
        pipeline_id: String,
        event_json: String,
    ) -> napi::Result<String> {
        let inner = self.inner.clone();
        let event: GristEvent = serde_json::from_str(&event_json)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let result = inner
            .run_pipeline(&pipeline_id, &event)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_string(&result)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Return the ids of all registered pipelines.
    #[napi]
    pub fn pipeline_ids(&self) -> Vec<String> {
        self.inner.pipeline_ids()
    }

    // ── Bus ──────────────────────────────────────────────────────────────────

    /// Subscribe to a bus topic.  Returns a [`BusSubscription`] whose
    /// `nextJson()` method resolves with the next event as a JSON string.
    #[napi]
    pub fn subscribe(&self, topic: String) -> BusSubscription {
        let rx = self.inner.subscribe(&topic);
        BusSubscription {
            rx: Arc::new(Mutex::new(rx)),
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Build a `GristEvent` and return it as a JSON string.
    ///
    /// `channel` — one of `http`, `websocket`, `cli`, `cron`, `webhook`,
    /// `mq`, `fs`, `python`, `typescript`, `internal`.
    #[napi]
    pub fn build_event(
        &self,
        channel: String,
        payload_json: String,
    ) -> napi::Result<String> {
        let payload: serde_json::Value = serde_json::from_str(&payload_json)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let event = GristMillCore::build_event(&channel, payload);
        serde_json::to_string(&event)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BusSubscription
// ─────────────────────────────────────────────────────────────────────────────

/// A handle to a bus topic subscription.
///
/// Call `nextJson()` in an async loop to receive events.
#[napi]
pub struct BusSubscription {
    rx: Arc<Mutex<grist_bus::Subscription>>,
}

#[napi]
impl BusSubscription {
    /// Wait for the next event and return it as a JSON string.
    ///
    /// Returns `null` if the bus has been shut down.
    #[napi]
    pub async fn next_json(&self) -> napi::Result<Option<String>> {
        let rx = self.rx.clone();
        // Await the next message; lagged errors just return None so the
        // caller can loop and try again.
        let result = {
            let mut guard = rx
                .lock()
                .map_err(|_| napi::Error::from_reason("mutex poisoned"))?;
            guard.recv().await
        };
        match result {
            Ok(val) => Ok(Some(
                serde_json::to_string(&val)
                    .map_err(|e| napi::Error::from_reason(e.to_string()))?,
            )),
            Err(tokio::sync::broadcast::error::RecvError::Closed) => Ok(None),
            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                tracing::warn!(lagged = n, "BusSubscription: lagged, skipping events");
                Ok(None)
            }
        }
    }
}
