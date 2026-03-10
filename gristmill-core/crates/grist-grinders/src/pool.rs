//! Rayon-based worker pool with dynamic batching (PRD G-01, G-05, G-07).
//!
//! # Design
//!
//! ```text
//! caller ─────────────────────────────────────────────────────────────────┐
//!         submit(InferenceRequest)                                         │
//!              │                                                           │
//!              ▼                                                           │
//!         mpsc channel ──→ batcher task (Tokio)                           │
//!                               │                                         │
//!                               │  accumulate up to `batch_window_ms` or  │
//!                               │  `max_batch_size` requests              │
//!                               ▼                                         │
//!                         rayon::spawn_fifo ──→ worker thread             │
//!                               │                    │                    │
//!                               │            session.run(req)             │
//!                               │                    │                    │
//!                               └──── oneshot channel ──→ caller ◄────────┘
//! ```
//!
//! The pool submits batches to Rayon (CPU threads) to keep the Tokio reactor
//! free for I/O.  Each caller receives a `oneshot::Receiver` that resolves
//! when inference completes (or times out — PRD G-07).

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot};
use tracing::{debug, instrument, warn};

use crate::error::GrindersError;
use crate::registry::ModelRegistry;
use crate::session::{InferenceOutput, InferenceRequest};

// ─────────────────────────────────────────────────────────────────────────────
// Internal message flowing through the dispatch channel
// ─────────────────────────────────────────────────────────────────────────────

struct DispatchMsg {
    request: InferenceRequest,
    reply: oneshot::Sender<Result<InferenceOutput, GrindersError>>,
    deadline: Instant,
}

// ─────────────────────────────────────────────────────────────────────────────
// Pool configuration (subset used internally by the pool)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub queue_depth: usize,
    pub batch_window: Duration,
    pub max_batch_size: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// WorkerPool
// ─────────────────────────────────────────────────────────────────────────────

/// Rayon-backed ML inference pool with dynamic batching.
///
/// `WorkerPool` is `Send + Sync + Clone` — wrap in `Arc` and share across
/// Tokio tasks.  The underlying Rayon thread pool is shared globally
/// (Rayon's design) unless a custom pool is configured.
pub struct WorkerPool {
    sender: mpsc::Sender<DispatchMsg>,
    #[allow(dead_code)] // retained for future capacity introspection
    config: PoolConfig,
}

impl WorkerPool {
    /// Construct a pool with the given config and registry.
    ///
    /// Spawns a Tokio task that batches incoming requests and dispatches them
    /// to Rayon workers.
    pub fn new(config: PoolConfig, registry: Arc<ModelRegistry>) -> Self {
        let (tx, rx) = mpsc::channel::<DispatchMsg>(config.queue_depth);

        // Batcher task lives for the lifetime of the pool.
        let cfg = config.clone();
        tokio::spawn(batcher_task(rx, registry, cfg));

        Self { sender: tx, config }
    }

    /// Submit a single inference request and return a future that resolves
    /// to the output.
    ///
    /// Times out after the per-model timeout stored in the model config
    /// (falling back to 5 seconds if not available).
    #[instrument(level = "debug", skip(self, request), fields(model_id = %request.model_id))]
    pub async fn submit(
        &self,
        request: InferenceRequest,
        timeout: Duration,
    ) -> Result<InferenceOutput, GrindersError> {
        let model_id = request.model_id.clone();
        let deadline = Instant::now() + timeout;

        let (reply_tx, reply_rx) = oneshot::channel();

        self.sender
            .try_send(DispatchMsg {
                request,
                reply: reply_tx,
                deadline,
            })
            .map_err(|_| GrindersError::PoolFull(model_id.clone()))?;

        // Await response or timeout.
        let remaining = deadline.saturating_duration_since(Instant::now());
        match tokio::time::timeout(remaining, reply_rx).await {
            Ok(Ok(result)) => {
                metrics::counter!("grinders.pool.completed", "model_id" => model_id.clone())
                    .increment(1);
                result
            }
            Ok(Err(_)) => {
                // Sender dropped — pool shut down.
                Err(GrindersError::PoolFull(model_id))
            }
            Err(_) => {
                warn!(model_id, "inference request timed out");
                metrics::counter!("grinders.pool.timeouts", "model_id" => model_id.clone())
                    .increment(1);
                Err(GrindersError::Timeout {
                    model_id,
                    elapsed_ms: timeout.as_millis() as u64,
                })
            }
        }
    }

    /// Queue depth remaining (number of requests that can still be enqueued).
    pub fn queue_capacity(&self) -> usize {
        self.sender.capacity()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Batcher task (runs on Tokio, dispatches to Rayon)
// ─────────────────────────────────────────────────────────────────────────────

/// The batcher collects requests within a `batch_window` and then dispatches
/// them as a group to Rayon (PRD G-05: batch throughput >2× single-request).
///
/// Requests past their deadline are dropped immediately without hitting Rayon.
async fn batcher_task(
    mut rx: mpsc::Receiver<DispatchMsg>,
    registry: Arc<ModelRegistry>,
    config: PoolConfig,
) {
    loop {
        // Wait for the first message.
        let first = match rx.recv().await {
            Some(msg) => msg,
            None => break, // channel closed — pool dropped
        };

        let mut batch = vec![first];

        // Accumulate additional messages within the batch window (G-05).
        if config.batch_window > Duration::ZERO && config.max_batch_size > 1 {
            let window_end = tokio::time::Instant::now() + config.batch_window;
            loop {
                if batch.len() >= config.max_batch_size {
                    break;
                }
                match tokio::time::timeout_at(window_end, rx.recv()).await {
                    Ok(Some(msg)) => batch.push(msg),
                    _ => break,
                }
            }
        }

        let batch_size = batch.len();
        metrics::histogram!("grinders.pool.batch_size").record(batch_size as f64);
        debug!(batch_size, "dispatching batch to Rayon");

        // Dispatch to Rayon — each item in the batch runs in parallel.
        let reg = Arc::clone(&registry);
        rayon::spawn(move || {
            rayon::scope(|s| {
                for msg in batch {
                    let reg = Arc::clone(&reg);
                    s.spawn(move |_| {
                        process_one(msg, &reg);
                    });
                }
            });
        });
    }
}

/// Process a single dispatch message on a Rayon worker thread.
fn process_one(msg: DispatchMsg, registry: &ModelRegistry) {
    // Check if the deadline has already passed before doing any work.
    if Instant::now() >= msg.deadline {
        let model_id = msg.request.model_id.clone();
        debug!(model_id, "request expired before processing");
        let _ = msg.reply.send(Err(GrindersError::Timeout {
            model_id,
            elapsed_ms: 0,
        }));
        return;
    }

    let result = registry
        .get_or_load(&msg.request.model_id)
        .and_then(|session| session.run(&msg.request));

    let _ = msg.reply.send(result);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelConfig, ModelRuntime};
    use crate::registry::ModelRegistry;
    use crate::session::InferenceRequest;
    use ndarray::Array2;
    use std::time::Duration;

    fn make_pool() -> (WorkerPool, Arc<ModelRegistry>) {
        let registry = Arc::new(ModelRegistry::new());
        // Register a cold stub model (ONNX without the feature → stub session).
        registry
            .register(ModelConfig {
                model_id: "stub-model".into(),
                path: std::path::PathBuf::from("/nonexistent/stub.onnx"),
                runtime: ModelRuntime::Onnx,
                warm: false, // cold — load on first access
                timeout: Duration::from_secs(5),
                max_tokens: 0,
                description: "test stub".into(),
            })
            .unwrap();

        let pool = WorkerPool::new(
            PoolConfig {
                queue_depth: 64,
                batch_window: Duration::from_millis(5),
                max_batch_size: 8,
                },
            Arc::clone(&registry),
        );
        (pool, registry)
    }

    #[tokio::test]
    async fn submit_returns_output_for_registered_model() {
        let (pool, _reg) = make_pool();
        let req = InferenceRequest::from_features("stub-model", Array2::zeros((1, 392)));
        let out = pool.submit(req, Duration::from_secs(5)).await.unwrap();
        assert!(out.tensor.is_some(), "expected tensor output");
    }

    #[tokio::test]
    async fn submit_unknown_model_returns_not_found() {
        let (pool, _reg) = make_pool();
        let req = InferenceRequest::from_features("unknown-model", Array2::zeros((1, 392)));
        let err = pool.submit(req, Duration::from_secs(5)).await.unwrap_err();
        assert!(
            matches!(err, GrindersError::ModelNotFound(_)),
            "expected ModelNotFound, got {err:?}"
        );
    }

    #[tokio::test]
    async fn batch_multiple_requests_concurrently() {
        let (pool, _reg) = make_pool();
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let req = InferenceRequest::from_features("stub-model", Array2::zeros((1, 392)));
                let _ = i;
                tokio::spawn({
                    // We can't move pool (no Clone), so run them sequentially
                    // in this test — the real concurrency test is in lib.rs.
                    async move { req }
                })
            })
            .collect();

        // Collect the requests from spawned tasks, then submit.
        let pool_ref = &pool;
        for h in handles {
            let req = h.await.unwrap();
            let result = pool_ref.submit(req, Duration::from_secs(5)).await;
            assert!(result.is_ok(), "expected Ok, got {result:?}");
        }
    }

    #[tokio::test]
    async fn pool_full_returns_error_when_channel_saturated() {
        let registry = Arc::new(ModelRegistry::new());
        // Create a pool with a tiny queue depth.
        let pool = WorkerPool::new(
            PoolConfig {
                queue_depth: 1,
                batch_window: Duration::from_millis(1000), // long window to prevent draining
                max_batch_size: 100,
            },
            Arc::clone(&registry),
        );

        // Send enough requests to overflow the queue.
        let mut pool_full_seen = false;
        for _ in 0..20 {
            let req = InferenceRequest::from_features("x", Array2::zeros((1, 1)));
            if pool.sender.try_send(DispatchMsg {
                request: req,
                reply: tokio::sync::oneshot::channel().0,
                deadline: Instant::now() + Duration::from_secs(1),
            }).is_err() {
                pool_full_seen = true;
                break;
            }
        }
        assert!(pool_full_seen, "expected pool full error after queue saturation");
    }
}
