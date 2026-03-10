//! PyO3 bridge — exposes [`GristMillCore`] to Python.
//!
//! All public methods use blocking `rt.block_on()` with `py.allow_threads()`
//! so the GIL is released during I/O-bound Rust work.  Types cross the
//! boundary as JSON strings to avoid complex pyo3 type-mapping for deeply
//! nested Rust enums.
//!
//! Enabled by `--features python`.

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use grist_event::{GristEvent, Priority};
use grist_ledger::Memory;
use grist_millwright::dag::Pipeline;
use grist_sieve::RouteDecision;

use crate::core::GristMillCore;
use crate::error::FfiError;

// ─────────────────────────────────────────────────────────────────────────────
// Error helpers
// ─────────────────────────────────────────────────────────────────────────────

fn ffi_err(e: FfiError) -> PyErr {
    PyErr::new::<PyRuntimeError, _>(e.to_string())
}

fn json_err(e: serde_json::Error) -> PyErr {
    PyErr::new::<PyValueError, _>(e.to_string())
}

// ─────────────────────────────────────────────────────────────────────────────
// PyGristMill
// ─────────────────────────────────────────────────────────────────────────────

/// Python-visible wrapper around [`GristMillCore`].
///
/// ```python
/// import gristmill_core
/// mill = gristmill_core.PyGristMill(None)
/// decision_json = mill.triage(event.to_json())
/// ```
#[pyclass(name = "PyGristMill")]
pub struct PyGristMill {
    inner: Arc<GristMillCore>,
    rt: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyGristMill {
    /// Create and initialise the GristMill core.
    ///
    /// Args:
    ///     config_path: Optional path to `~/.gristmill/config.yaml`.
    ///                  Pass `None` to use built-in defaults.
    #[new]
    fn new(config_path: Option<String>) -> PyResult<Self> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

        let path = config_path.map(std::path::PathBuf::from);
        let inner = rt
            .block_on(GristMillCore::new(path))
            .map_err(ffi_err)?;

        Ok(Self {
            inner: Arc::new(inner),
            rt: Arc::new(rt),
        })
    }

    /// Triage a [`PyGristEvent`] and return the routing decision as a JSON
    /// string (`{"route": "LOCAL_ML", "confidence": 0.93, ...}`).
    fn triage(&self, py: Python<'_>, event_json: String) -> PyResult<String> {
        let event: GristEvent =
            serde_json::from_str(&event_json).map_err(json_err)?;
        let inner = Arc::clone(&self.inner);
        let rt = Arc::clone(&self.rt);
        let result = py
            .allow_threads(move || rt.block_on(inner.triage(&event)))
            .map_err(ffi_err)?;
        serde_json::to_string(&result).map_err(json_err)
    }

    /// Store a text memory and return its ULID string id.
    fn remember(
        &self,
        py: Python<'_>,
        content: String,
        tags: Vec<String>,
    ) -> PyResult<String> {
        let inner = Arc::clone(&self.inner);
        let rt = Arc::clone(&self.rt);
        py.allow_threads(move || rt.block_on(inner.remember(content, tags)))
            .map_err(ffi_err)
    }

    /// Semantic + keyword recall — returns a JSON array of ranked memories.
    fn recall(
        &self,
        py: Python<'_>,
        query: String,
        limit: usize,
    ) -> PyResult<String> {
        let inner = Arc::clone(&self.inner);
        let rt = Arc::clone(&self.rt);
        let results = py
            .allow_threads(move || rt.block_on(inner.recall(&query, limit)))
            .map_err(ffi_err)?;
        serde_json::to_string(&results).map_err(json_err)
    }

    /// Retrieve a single memory by ULID id, or `None` if not found.
    /// Returns the memory as a JSON string.
    fn get_memory(
        &self,
        py: Python<'_>,
        id: String,
    ) -> PyResult<Option<String>> {
        let inner = Arc::clone(&self.inner);
        let rt = Arc::clone(&self.rt);
        let result = py
            .allow_threads(move || rt.block_on(inner.get_memory(&id)))
            .map_err(ffi_err)?;
        match result {
            Some(m) => Ok(Some(serde_json::to_string(&m).map_err(json_err)?)),
            None => Ok(None),
        }
    }

    /// Escalate a prompt to an LLM provider.  Returns a JSON string with
    /// `content`, `provider`, `cache_hit`, `tokens_used`, `elapsed_ms`.
    fn escalate(
        &self,
        py: Python<'_>,
        prompt: String,
        max_tokens: u32,
    ) -> PyResult<String> {
        let inner = Arc::clone(&self.inner);
        let rt = Arc::clone(&self.rt);
        let result = py
            .allow_threads(move || rt.block_on(inner.escalate(prompt, max_tokens)))
            .map_err(ffi_err)?;
        serde_json::to_string(&result).map_err(json_err)
    }

    /// Register a pipeline from its JSON serialisation.
    fn register_pipeline(&self, pipeline_json: String) -> PyResult<()> {
        let pipeline: Pipeline =
            serde_json::from_str(&pipeline_json).map_err(json_err)?;
        self.inner.register_pipeline(pipeline);
        Ok(())
    }

    /// Run a registered pipeline.  Returns the `PipelineResult` as JSON.
    fn run_pipeline(
        &self,
        py: Python<'_>,
        pipeline_id: String,
        event_json: String,
    ) -> PyResult<String> {
        let event: GristEvent =
            serde_json::from_str(&event_json).map_err(json_err)?;
        let inner = Arc::clone(&self.inner);
        let rt = Arc::clone(&self.rt);
        let result = py
            .allow_threads(move || {
                rt.block_on(inner.run_pipeline(&pipeline_id, &event))
            })
            .map_err(ffi_err)?;
        serde_json::to_string(&result).map_err(json_err)
    }

    /// Return the ids of all registered pipelines.
    fn pipeline_ids(&self) -> Vec<String> {
        self.inner.pipeline_ids()
    }

    /// Build a `GristEvent` JSON string from a channel label and JSON payload.
    ///
    /// `channel` must be one of: `http`, `websocket`, `cli`, `cron`,
    /// `webhook`, `mq`, `fs`, `python`, `typescript`, `internal`.
    #[staticmethod]
    fn build_event(channel: String, payload_json: String) -> PyResult<String> {
        let payload: serde_json::Value =
            serde_json::from_str(&payload_json).map_err(json_err)?;
        let event = GristMillCore::build_event(&channel, payload);
        serde_json::to_string(&event).map_err(json_err)
    }

    /// Subscribe to a bus topic.  Returns a [`PyBusSubscription`] whose
    /// `recv_json()` method blocks until the next event arrives.
    fn subscribe(&self, topic: String) -> PyBusSubscription {
        let rx = self.inner.subscribe(&topic);
        PyBusSubscription {
            rx: Arc::new(Mutex::new(rx)),
            rt: Arc::clone(&self.rt),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyGristEvent
// ─────────────────────────────────────────────────────────────────────────────

/// Python-visible wrapper around a [`GristEvent`].
#[pyclass(name = "PyGristEvent")]
pub struct PyGristEvent {
    pub(crate) inner: GristEvent,
}

#[pymethods]
impl PyGristEvent {
    /// Create a new event.
    ///
    /// Args:
    ///     source: Channel label (e.g. ``"http"``, ``"cron"``).
    ///     payload_json: JSON string to use as the event payload.
    #[new]
    fn new(source: String, payload_json: String) -> PyResult<Self> {
        let payload: serde_json::Value =
            serde_json::from_str(&payload_json).map_err(json_err)?;
        let event = GristMillCore::build_event(&source, payload);
        Ok(Self { inner: event })
    }

    /// Return a new event with the given priority applied.
    ///
    /// `priority` must be one of ``"low"``, ``"normal"``, ``"high"``,
    /// ``"critical"``.
    fn with_priority(&self, priority: String) -> Self {
        let p = match priority.to_lowercase().as_str() {
            "low" => Priority::Low,
            "high" => Priority::High,
            "critical" => Priority::Critical,
            _ => Priority::Normal,
        };
        Self {
            inner: self.inner.clone().with_priority(p),
        }
    }

    /// Serialise this event to a JSON string (for passing to `triage` etc.).
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(json_err)
    }

    /// ULID string identifier.
    #[getter]
    fn id(&self) -> String {
        self.inner.id.to_string()
    }

    /// Wall-clock timestamp in milliseconds since the UNIX epoch.
    #[getter]
    fn timestamp_ms(&self) -> u64 {
        self.inner.timestamp_ms
    }

    /// Approximate token count of the payload.
    #[getter]
    fn estimated_token_count(&self) -> u32 {
        self.inner.estimated_token_count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyRouteDecision
// ─────────────────────────────────────────────────────────────────────────────

/// Routing decision returned by `PyGristMill.triage()`.
///
/// All fields are readable as Python attributes.
#[pyclass(name = "PyRouteDecision")]
pub struct PyRouteDecision {
    /// ``"LOCAL_ML"``, ``"RULES"``, ``"HYBRID"``, or ``"LLM_NEEDED"``.
    #[pyo3(get)]
    pub route: String,
    /// Classifier confidence in [0, 1].
    #[pyo3(get)]
    pub confidence: f32,
    /// Model or rule id (present for LOCAL_ML, RULES, HYBRID).
    #[pyo3(get)]
    pub model_id: Option<String>,
    /// Human-readable reason (present for LLM_NEEDED).
    #[pyo3(get)]
    pub reason: Option<String>,
    /// Estimated LLM token cost (present for HYBRID and LLM_NEEDED).
    #[pyo3(get)]
    pub estimated_tokens: Option<u32>,
}

#[pymethods]
impl PyRouteDecision {
    /// Serialise to JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&serde_json::json!({
            "route": self.route,
            "confidence": self.confidence,
            "model_id": self.model_id,
            "reason": self.reason,
            "estimated_tokens": self.estimated_tokens,
        }))
        .map_err(json_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyRouteDecision(route={:?}, confidence={:.3})",
            self.route, self.confidence
        )
    }
}

impl PyRouteDecision {
    pub fn from_route_decision(rd: &RouteDecision) -> Self {
        match rd {
            RouteDecision::LocalMl { model_id, confidence } => Self {
                route: "LOCAL_ML".to_string(),
                confidence: *confidence,
                model_id: Some(model_id.clone()),
                reason: None,
                estimated_tokens: None,
            },
            RouteDecision::Rules { rule_id, confidence } => Self {
                route: "RULES".to_string(),
                confidence: *confidence,
                model_id: Some(rule_id.clone()),
                reason: None,
                estimated_tokens: None,
            },
            RouteDecision::Hybrid {
                local_model,
                estimated_tokens,
                confidence,
                ..
            } => Self {
                route: "HYBRID".to_string(),
                confidence: *confidence,
                model_id: Some(local_model.clone()),
                reason: None,
                estimated_tokens: Some(*estimated_tokens),
            },
            RouteDecision::LlmNeeded {
                reason,
                estimated_tokens,
                confidence,
                ..
            } => Self {
                route: "LLM_NEEDED".to_string(),
                confidence: *confidence,
                model_id: None,
                reason: Some(reason.clone()),
                estimated_tokens: Some(*estimated_tokens),
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyMemory
// ─────────────────────────────────────────────────────────────────────────────

/// A memory retrieved from the ledger.
#[pyclass(name = "PyMemory")]
pub struct PyMemory {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub tags: Vec<String>,
    #[pyo3(get)]
    pub created_at_ms: u64,
    /// ``"hot"``, ``"warm"``, or ``"cold"``.
    #[pyo3(get)]
    pub tier: String,
}

#[pymethods]
impl PyMemory {
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&serde_json::json!({
            "id": self.id,
            "content": self.content,
            "tags": self.tags,
            "created_at_ms": self.created_at_ms,
            "tier": self.tier,
        }))
        .map_err(json_err)
    }

    fn __repr__(&self) -> String {
        format!("PyMemory(id={:?}, tier={:?})", self.id, self.tier)
    }
}

impl PyMemory {
    pub fn from_memory(m: &Memory) -> Self {
        Self {
            id: m.id.clone(),
            content: m.content.clone(),
            tags: m.tags.clone(),
            created_at_ms: m.created_at_ms,
            tier: format!("{:?}", m.tier).to_lowercase(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyBusSubscription
// ─────────────────────────────────────────────────────────────────────────────

/// A handle to a bus topic subscription.
///
/// Call `recv_json()` in a loop to receive events.
#[pyclass(name = "PyBusSubscription")]
pub struct PyBusSubscription {
    rx: Arc<Mutex<grist_bus::Subscription>>,
    rt: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyBusSubscription {
    /// Block until the next event arrives and return it as a JSON string.
    ///
    /// Returns ``None`` if the bus has been shut down.
    fn recv_json(&self, py: Python<'_>) -> PyResult<Option<String>> {
        let rx = Arc::clone(&self.rx);
        let rt = Arc::clone(&self.rt);
        let result = py.allow_threads(move || {
            let mut guard = rx
                .lock()
                .map_err(|_| FfiError::runtime("mutex poisoned"))?;
            rt.block_on(guard.recv())
                .map_err(|e| FfiError::runtime(e.to_string()))
        });
        match result {
            Ok(val) => {
                Ok(Some(serde_json::to_string(&val).map_err(json_err)?))
            }
            Err(e) if e.to_string().contains("closed") => Ok(None),
            Err(e) => Err(ffi_err(e)),
        }
    }
}
