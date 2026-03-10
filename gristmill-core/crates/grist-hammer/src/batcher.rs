//! Request batcher for grist-hammer.
//!
//! [`RequestBatcher`] accumulates incoming [`EscalationRequest`]s over a
//! configurable time window (default 5 s) or until `max_batch_size` (default 10)
//! is reached, then dispatches each item as a concurrent Tokio task.
//!
//! Pattern mirrors `grist-grinders/pool.rs`: the constructor is *synchronous*
//! but calls `tokio::spawn` internally, so an active Tokio runtime is required.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};
use tracing::{debug, warn};

use crate::config::BatchConfig;
use crate::error::HammerError;
use crate::router::RequestRouter;
use crate::types::{EscalationRequest, EscalationResponse};

// ─────────────────────────────────────────────────────────────────────────────
// BatchMsg
// ─────────────────────────────────────────────────────────────────────────────

/// An individual message sent to the batcher task.
pub(crate) struct BatchMsg {
    pub request: EscalationRequest,
    pub reply: oneshot::Sender<Result<EscalationResponse, HammerError>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// RequestBatcher
// ─────────────────────────────────────────────────────────────────────────────

/// Handle to the background batcher task.
pub struct RequestBatcher {
    tx: mpsc::Sender<BatchMsg>,
    config: BatchConfig,
}

impl RequestBatcher {
    /// Create a new batcher and spawn the background dispatch task.
    ///
    /// **Requires an active Tokio runtime.**
    pub fn new(config: BatchConfig, router: Arc<RequestRouter>) -> Self {
        let (tx, rx) = mpsc::channel::<BatchMsg>(1024);
        let cfg = config.clone();
        tokio::spawn(batcher_task(rx, router, cfg));
        Self { tx, config }
    }

    /// Submit a request to the batcher and wait for the response.
    pub async fn submit(
        &self,
        request: EscalationRequest,
    ) -> Result<EscalationResponse, HammerError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(BatchMsg { request, reply: reply_tx })
            .await
            .map_err(|_| HammerError::Config("batcher channel closed".into()))?;

        reply_rx
            .await
            .map_err(|_| HammerError::Config("batcher reply channel dropped".into()))?
    }

    pub fn config(&self) -> &BatchConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// batcher_task
// ─────────────────────────────────────────────────────────────────────────────

async fn batcher_task(
    mut rx: mpsc::Receiver<BatchMsg>,
    router: Arc<RequestRouter>,
    config: BatchConfig,
) {
    let window = Duration::from_millis(config.window_ms);
    let max = config.max_batch_size;

    loop {
        // Wait for the first message.
        let first = match rx.recv().await {
            Some(msg) => msg,
            None => {
                debug!("batcher channel closed, task exiting");
                return;
            }
        };

        let mut batch = vec![first];

        // Accumulate more messages within the window or until max batch size.
        let deadline = tokio::time::sleep(window);
        tokio::pin!(deadline);

        while batch.len() < max {
            tokio::select! {
                biased;
                msg = rx.recv() => {
                    match msg {
                        Some(m) => batch.push(m),
                        None => break,
                    }
                }
                _ = &mut deadline => break,
            }
        }

        debug!(batch_size = batch.len(), "dispatching batch");

        // Dispatch each item as a concurrent Tokio task.
        for msg in batch {
            let r = Arc::clone(&router);
            tokio::spawn(async move {
                let result = r.route(&msg.request).await;
                if msg.reply.send(result).is_err() {
                    warn!("batcher: caller dropped reply receiver");
                }
            });
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use crate::config::HammerConfig;
    use crate::router::RequestRouter;
    use crate::types::{EscalationRequest, Provider};

    struct CountingProvider {
        count: Arc<AtomicUsize>,
    }

    impl crate::router::ProviderFn for CountingProvider {
        fn call(
            &self,
            req: &EscalationRequest,
        ) -> Result<(String, u32, Provider), HammerError> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok((format!("response to: {}", req.prompt), 5, Provider::AnthropicPrimary))
        }
    }

    fn make_router(count: Arc<AtomicUsize>) -> Arc<RequestRouter> {
        let providers: Vec<Arc<dyn crate::router::ProviderFn>> =
            vec![Arc::new(CountingProvider { count })];
        Arc::new(RequestRouter::with_mock_providers(HammerConfig::default(), providers))
    }

    fn batch_config(window_ms: u64, max_batch_size: usize) -> BatchConfig {
        BatchConfig { enabled: true, window_ms, max_batch_size }
    }

    #[tokio::test]
    async fn batcher_dispatches_single_request() {
        let count = Arc::new(AtomicUsize::new(0));
        let router = make_router(Arc::clone(&count));
        let batcher = RequestBatcher::new(batch_config(100, 10), router);

        let req = EscalationRequest::new("hello", 50);
        let resp = batcher.submit(req).await.unwrap();
        assert_eq!(resp.content, "response to: hello");
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn batcher_accumulates_within_window() {
        let count = Arc::new(AtomicUsize::new(0));
        let router = make_router(Arc::clone(&count));
        // Long window, small max so multiple items fit.
        let batcher = Arc::new(RequestBatcher::new(batch_config(200, 5), router));

        let mut handles = Vec::new();
        for i in 0..3 {
            let b = Arc::clone(&batcher);
            handles.push(tokio::spawn(async move {
                b.submit(EscalationRequest::new(format!("q{i}"), 50)).await
            }));
        }

        for h in handles {
            let resp = h.await.unwrap().unwrap();
            assert!(resp.content.starts_with("response to:"));
        }
        assert_eq!(count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn batcher_dispatches_on_max_batch_size() {
        let count = Arc::new(AtomicUsize::new(0));
        let router = make_router(Arc::clone(&count));
        // max_batch_size = 2, long window — first batch of 2 should dispatch immediately.
        let batcher = Arc::new(RequestBatcher::new(batch_config(5000, 2), router));

        let b1 = Arc::clone(&batcher);
        let h1 = tokio::spawn(async move {
            b1.submit(EscalationRequest::new("q1", 50)).await
        });
        let b2 = Arc::clone(&batcher);
        let h2 = tokio::spawn(async move {
            b2.submit(EscalationRequest::new("q2", 50)).await
        });

        h1.await.unwrap().unwrap();
        h2.await.unwrap().unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }
}
