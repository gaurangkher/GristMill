//! Feedback log — writes one JSONL record per routing decision for the
//! closed learning loop.
//!
//! PRD requirement S-06:
//! "Log all routing decisions with confidence scores for feedback loop.
//!  JSONL feedback log passes schema validation."
//!
//! Schema (stable — Python SieveTrainer depends on it):
//! ```json
//! {
//!   "event_id":           "01HXY…",       // ULID
//!   "timestamp_ms":       1700000000000,
//!   "route_decision":     "LOCAL_ML",
//!   "confidence":         0.937,
//!   "estimated_tokens":   0,
//!   "actual_tokens":      null,            // filled retrospectively
//!   "could_have_been_local": null,         // filled after LLM response
//!   "event_source":       "http",
//!   "token_count":        12
//! }
//! ```
//!
//! The `FeedbackLog` is designed to be write-heavy and rarely read.  It
//! uses a background Tokio channel to avoid blocking the triage hot path.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use grist_event::GristEvent;
#[allow(unused_imports)]
use metrics;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;
use tracing::{error, info, instrument, warn};
use ulid::Ulid;

use crate::cost_oracle::RouteDecision;
use crate::error::SieveError;

// ─────────────────────────────────────────────────────────────────────────────
// Feedback record (stable schema)
// ─────────────────────────────────────────────────────────────────────────────

/// A single record in the feedback JSONL log.
///
/// Fields marked `Option` are filled in retrospectively:
/// - `actual_tokens` — set when Hammer reports the real token usage.
/// - `could_have_been_local` — set after analysing whether the LLM was needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackRecord {
    /// ULID of the originating [`GristEvent`].
    pub event_id: String,
    /// Wall-clock time at the moment of triage (ms since UNIX epoch).
    pub timestamp_ms: u64,
    /// Route chosen by the Sieve.
    pub route_decision: String,
    /// Classifier confidence [0, 1].
    pub confidence: f32,
    /// Estimated LLM tokens (0 for LOCAL_ML / RULES routes).
    pub estimated_tokens: u32,
    /// Actual LLM tokens used (filled retrospectively by Hammer).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_tokens: Option<u32>,
    /// Whether a LOCAL_ML route would have been sufficient (retrospective).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub could_have_been_local: Option<bool>,
    /// Source channel label (e.g. "http", "cron").
    pub event_source: String,
    /// Approximate token count of the event payload.
    pub token_count: u32,
}

impl FeedbackRecord {
    /// Construct from an event and its routing decision.
    pub fn new(event: &GristEvent, decision: &RouteDecision) -> Self {
        let (estimated_tokens, route_decision_str) = match decision {
            RouteDecision::LocalMl { confidence: _, .. } => (0, "LOCAL_ML".to_string()),
            RouteDecision::Rules { confidence: _, .. } => (0, "RULES".to_string()),
            RouteDecision::Hybrid {
                estimated_tokens, ..
            } => (*estimated_tokens, "HYBRID".to_string()),
            RouteDecision::LlmNeeded {
                estimated_tokens, ..
            } => (*estimated_tokens, "LLM_NEEDED".to_string()),
        };

        Self {
            event_id: event.id.to_string(),
            timestamp_ms: event.timestamp_ms,
            route_decision: route_decision_str,
            confidence: decision.confidence(),
            estimated_tokens,
            actual_tokens: None,
            could_have_been_local: None,
            event_source: event.source.label().to_string(),
            token_count: event.estimated_token_count(),
        }
    }

    /// Serialise to a single JSONL line (no trailing newline).
    pub fn to_jsonl(&self) -> Result<String, SieveError> {
        serde_json::to_string(self)
            .map_err(|e| SieveError::Feedback(format!("serialisation failed: {e}")))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FeedbackLog — async, non-blocking writer
// ─────────────────────────────────────────────────────────────────────────────

/// Writes feedback records to a rotating daily JSONL log.
///
/// The writer runs in a background Tokio task.  The triage hot path only
/// sends a record over a bounded mpsc channel — it never touches disk directly.
#[derive(Debug, Clone)]
pub struct FeedbackLog {
    tx: mpsc::Sender<FeedbackRecord>,
    records_sent: Arc<AtomicU64>,
}

impl FeedbackLog {
    /// Create a new feedback log that writes to `dir/feedback-YYYY-MM-DD.jsonl`.
    ///
    /// The returned `FeedbackLog` is a cheap handle.  The background writer
    /// task is spawned on the current Tokio runtime and continues until the
    /// last sender is dropped.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, SieveError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|e| {
            SieveError::Feedback(format!("cannot create feedback dir {}: {e}", dir.display()))
        })?;

        let (tx, rx) = mpsc::channel::<FeedbackRecord>(4096);
        let records_sent = Arc::new(AtomicU64::new(0));

        let records_sent_bg = Arc::clone(&records_sent);
        tokio::spawn(writer_task(dir, rx, records_sent_bg));

        Ok(Self { tx, records_sent })
    }

    /// No-op feedback log (for tests / CLI mode with no feedback dir).
    pub fn noop() -> Self {
        let (tx, mut rx) = mpsc::channel::<FeedbackRecord>(1);
        // Drain the channel to avoid blocking senders.
        tokio::spawn(async move { while rx.recv().await.is_some() {} });
        Self {
            tx,
            records_sent: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Record a routing decision.  Non-blocking — sends over the channel.
    ///
    /// If the channel is full (back-pressure), the record is dropped with a
    /// warning.  We must never block the hot path.
    #[instrument(level = "trace", skip(self, event, decision),
                 fields(event_id = %event.id, route = ?decision.label()))]
    pub fn record(&self, event: &GristEvent, decision: &RouteDecision) {
        let record = FeedbackRecord::new(event, decision);
        match self.tx.try_send(record) {
            Ok(_) => {
                self.records_sent.fetch_add(1, Ordering::Relaxed);
                metrics::counter!("sieve.feedback.sent").increment(1);
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                warn!("feedback log channel full — dropping record");
                metrics::counter!("sieve.feedback.dropped").increment(1);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                // Writer task exited — this is a programming error.
                error!("feedback log writer task has exited unexpectedly");
            }
        }
    }

    /// Total records sent since creation (for observability).
    pub fn records_sent(&self) -> u64 {
        self.records_sent.load(Ordering::Relaxed)
    }

    /// Update actual token usage and local-sufficiency flag for a past event.
    ///
    /// In a full implementation this would seek to the record in the JSONL file
    /// and patch it.  For now we log a supplemental record with the same
    /// `event_id` so the Python training pipeline can JOIN on it.
    pub fn update_actual_tokens(
        &self,
        event_id: Ulid,
        actual_tokens: u32,
        could_have_been_local: bool,
    ) {
        // Emit a correction record with the same event_id.
        let correction = FeedbackRecord {
            event_id: event_id.to_string(),
            timestamp_ms: grist_event::current_timestamp_ms(),
            route_decision: "CORRECTION".to_string(),
            confidence: 0.0,
            estimated_tokens: 0,
            actual_tokens: Some(actual_tokens),
            could_have_been_local: Some(could_have_been_local),
            event_source: "internal".to_string(),
            token_count: actual_tokens,
        };
        let _ = self.tx.try_send(correction);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Background writer task
// ─────────────────────────────────────────────────────────────────────────────

/// Receives records from the channel and appends them to a daily log file.
async fn writer_task(
    dir: PathBuf,
    mut rx: mpsc::Receiver<FeedbackRecord>,
    records_written: Arc<AtomicU64>,
) {
    info!(dir = %dir.display(), "feedback log writer started");

    let mut current_date = today_string();
    let mut file = open_log_file(&dir, &current_date).await;

    while let Some(record) = rx.recv().await {
        // Rotate file on date change.
        let today = today_string();
        if today != current_date {
            current_date = today.clone();
            file = open_log_file(&dir, &today).await;
        }

        match file {
            Ok(ref mut f) => match record.to_jsonl() {
                Ok(line) => {
                    let bytes = format!("{line}\n");
                    if let Err(e) = f.write_all(bytes.as_bytes()).await {
                        warn!(error = %e, "failed to write feedback record");
                    } else {
                        records_written.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(e) => {
                    warn!(error = %e, "failed to serialise feedback record");
                }
            },
            Err(ref e) => {
                warn!(error = %e, "feedback log file not open — dropping record");
            }
        }
    }

    info!("feedback log writer shutting down");
}

async fn open_log_file(dir: &Path, date: &str) -> Result<tokio::fs::File, std::io::Error> {
    let path: PathBuf = dir.join(format!("feedback-{date}.jsonl"));
    tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await
}

fn today_string() -> String {
    chrono::Utc::now().format("%Y-%m-%d").to_string()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost_oracle::RouteDecision;
    use grist_event::{ChannelType, GristEvent};

    fn make_event() -> GristEvent {
        GristEvent::new(
            ChannelType::Http,
            serde_json::json!({ "text": "schedule a meeting" }),
        )
    }

    fn local_decision() -> RouteDecision {
        RouteDecision::LocalMl {
            model_id: "test".into(),
            confidence: 0.95,
        }
    }

    fn llm_decision() -> RouteDecision {
        RouteDecision::LlmNeeded {
            reason: "test".into(),
            estimated_tokens: 500,
            estimated_cost_usd: 0.0075,
            confidence: 0.6,
        }
    }

    #[test]
    fn feedback_record_local_ml_zero_tokens() {
        let record = FeedbackRecord::new(&make_event(), &local_decision());
        assert_eq!(record.route_decision, "LOCAL_ML");
        assert_eq!(record.estimated_tokens, 0);
        assert!(record.actual_tokens.is_none());
        assert!(record.could_have_been_local.is_none());
    }

    #[test]
    fn feedback_record_llm_has_token_estimate() {
        let record = FeedbackRecord::new(&make_event(), &llm_decision());
        assert_eq!(record.route_decision, "LLM_NEEDED");
        assert!(record.estimated_tokens > 0);
    }

    #[test]
    fn feedback_record_jsonl_is_valid_json() {
        let record = FeedbackRecord::new(&make_event(), &local_decision());
        let line = record.to_jsonl().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed["route_decision"], "LOCAL_ML");
    }

    #[test]
    fn feedback_record_has_event_id() {
        let event = make_event();
        let record = FeedbackRecord::new(&event, &local_decision());
        assert_eq!(record.event_id, event.id.to_string());
    }

    #[test]
    fn feedback_record_schema_fields_present() {
        let record = FeedbackRecord::new(&make_event(), &local_decision());
        let json: serde_json::Value = serde_json::from_str(&record.to_jsonl().unwrap()).unwrap();

        for field in &[
            "event_id",
            "timestamp_ms",
            "route_decision",
            "confidence",
            "estimated_tokens",
            "event_source",
            "token_count",
        ] {
            assert!(json.get(field).is_some(), "missing field: {field}");
        }
    }

    #[tokio::test]
    async fn noop_log_does_not_panic() {
        let log = FeedbackLog::noop();
        log.record(&make_event(), &local_decision());
        // Give the task a moment to drain.
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        assert_eq!(log.records_sent(), 1);
    }

    #[tokio::test]
    async fn file_log_writes_to_disk() {
        let dir = tempfile::tempdir().unwrap();
        let log = FeedbackLog::open(dir.path()).unwrap();
        log.record(&make_event(), &local_decision());
        log.record(&make_event(), &llm_decision());
        // Allow background writer to flush.
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Find the log file.
        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert!(!entries.is_empty(), "expected at least one log file");

        let content = std::fs::read_to_string(entries[0].path()).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2, "expected 2 JSONL lines");

        // Validate both lines are valid JSON with required fields.
        for line in &lines {
            let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(parsed.get("event_id").is_some());
        }
    }
}
