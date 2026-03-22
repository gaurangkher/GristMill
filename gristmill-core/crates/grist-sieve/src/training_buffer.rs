//! Distillation training buffer — SQLite WAL-mode persistent store.
//!
//! Every teacher-model response that arrives from a `LocalOpenSource` provider
//! (i.e. Ollama / llama.cpp) is written here as a training record.  The
//! `gristmill-trainer` Python service reads `PENDING` records, marks them
//! `IN_TRAINING` during a cycle, and `CONSUMED` on completion.
//!
//! ## Design
//!
//! - SQLite in WAL mode allows the Inference Stack (Rust) to write while
//!   `gristmill-trainer` (Python) reads concurrently without blocking.
//! - Writes are non-blocking: the hot path sends a [`TrainingRecord`] over a
//!   bounded `mpsc` channel; a background Tokio task performs the actual
//!   `INSERT` via `spawn_blocking`.
//! - PII scrubbing is applied to `query_text` **before** the record is enqueued
//!   (see [`crate::pii::scrub`]).
//!
//! ## Schema (stable — gristmill-trainer depends on it)
//!
//! ```sql
//! CREATE TABLE training_records (
//!     record_id        TEXT    PRIMARY KEY,
//!     timestamp        TEXT    NOT NULL,
//!     query_text       TEXT    NOT NULL,
//!     teacher_response TEXT    NOT NULL,
//!     grinder_response TEXT,
//!     confidence_score REAL    NOT NULL,
//!     domain_tag       TEXT    NOT NULL,
//!     teacher_logits   BLOB,
//!     status           TEXT    NOT NULL DEFAULT 'PENDING',
//!     in_retention     INTEGER NOT NULL DEFAULT 0,
//!     provider_type    TEXT    NOT NULL
//! );
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{error, info, instrument, warn};
use ulid::Ulid;

use crate::error::SieveError;
use crate::pii;

// ─────────────────────────────────────────────────────────────────────────────
// TrainingRecord
// ─────────────────────────────────────────────────────────────────────────────

/// A single distillation training record.
///
/// Written after every successful teacher (open-source local) escalation.
/// The `provider_type` field is the runtime gate: only `"local_open_source"`
/// records are written by design; this field is stored for auditability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRecord {
    /// Unique record identifier (ULID).
    pub record_id: String,
    /// ISO 8601 timestamp of when the escalation was processed.
    pub timestamp: String,
    /// PII-scrubbed user query text.
    pub query_text: String,
    /// Teacher model response — the distillation learning target.
    pub teacher_response: String,
    /// Grinder's held response (if confidence was MED). `None` for LOW.
    pub grinder_response: Option<String>,
    /// Calibrated confidence score that triggered escalation [0, 1].
    pub confidence_score: f32,
    /// Auto-classified domain tag.
    pub domain_tag: DomainTag,
    /// Top-k teacher token logits (serialised as JSON bytes), if available.
    pub teacher_logits: Option<Vec<u8>>,
    /// Provider type — must be `"local_open_source"` for training use.
    pub provider_type: String,
}

/// Distillation training record domain classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DomainTag {
    Code,
    Writing,
    Reasoning,
    Qa,
    Creative,
    #[default]
    Other,
}

impl DomainTag {
    pub fn as_str(self) -> &'static str {
        match self {
            DomainTag::Code => "code",
            DomainTag::Writing => "writing",
            DomainTag::Reasoning => "reasoning",
            DomainTag::Qa => "qa",
            DomainTag::Creative => "creative",
            DomainTag::Other => "other",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RecordStatus
// ─────────────────────────────────────────────────────────────────────────────

/// Lifecycle status of a training record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordStatus {
    /// Written by Inference Stack. Awaiting trainer pickup.
    Pending,
    /// Claimed by gristmill-trainer for the current cycle.
    InTraining,
    /// Successfully used in a completed training cycle.
    Consumed,
}

impl RecordStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            RecordStatus::Pending => "PENDING",
            RecordStatus::InTraining => "IN_TRAINING",
            RecordStatus::Consumed => "CONSUMED",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal write command
// ─────────────────────────────────────────────────────────────────────────────

struct WriteCmd(TrainingRecord);

// ─────────────────────────────────────────────────────────────────────────────
// TrainingBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// Non-blocking handle to the SQLite WAL training buffer.
///
/// Construct once and clone cheaply — internally backed by a shared channel.
/// The background writer task is spawned on the current Tokio runtime and runs
/// until the last `TrainingBuffer` handle is dropped.
#[derive(Debug, Clone)]
pub struct TrainingBuffer {
    tx: mpsc::Sender<WriteCmd>,
    records_queued: Arc<AtomicU64>,
}

impl TrainingBuffer {
    /// Open (or create) the training buffer at `db_path`.
    ///
    /// Creates the database and its parent directories if they do not exist.
    /// Enables WAL journal mode and creates the schema on first run.
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self, SieveError> {
        let db_path = db_path.as_ref().to_path_buf();

        // Ensure parent directory exists.
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                SieveError::Feedback(format!(
                    "cannot create training buffer dir {}: {e}",
                    parent.display()
                ))
            })?;
        }

        // Open connection synchronously to run schema migrations before
        // handing control to the background task.
        {
            let conn = Connection::open(&db_path)
                .map_err(|e| SieveError::Feedback(format!("cannot open training buffer: {e}")))?;
            init_schema(&conn).map_err(|e| {
                SieveError::Feedback(format!("training buffer schema init failed: {e}"))
            })?;
        }

        let (tx, rx) = mpsc::channel::<WriteCmd>(4096);
        let records_queued = Arc::new(AtomicU64::new(0));

        let records_queued_bg = Arc::clone(&records_queued);
        tokio::spawn(writer_task(db_path, rx, records_queued_bg));

        info!("training buffer opened");
        Ok(Self { tx, records_queued })
    }

    /// No-op buffer (for tests / environments without a training path configured).
    pub fn noop() -> Self {
        let (tx, mut rx) = mpsc::channel::<WriteCmd>(1);
        tokio::spawn(async move { while rx.recv().await.is_some() {} });
        Self {
            tx,
            records_queued: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Enqueue a training record for writing.  Non-blocking — sends over the
    /// mpsc channel.  If the channel is full, the record is dropped with a
    /// warning rather than blocking the hot path.
    ///
    /// **PII scrubbing is applied here** to `query_text` before the record is
    /// enqueued.  The caller does not need to scrub the text first.
    ///
    /// **Training buffer gate**: if `provider_type` is not `"local_open_source"`,
    /// the record is silently dropped.  This is the enforcement point of the
    /// spec requirement that commercial API responses never enter the training
    /// buffer.
    #[instrument(level = "trace", skip(self, record), fields(record_id = %record.record_id))]
    pub fn insert(&self, mut record: TrainingRecord) {
        // ── Training buffer gate ───────────────────────────────────────────
        if record.provider_type != "local_open_source" {
            // Commercial API response — do NOT insert. This is the hard gate
            // required by the distillation spec (ToS compliance).
            metrics::counter!("training_buffer.commercial_blocked").increment(1);
            return;
        }

        // ── PII scrubbing ─────────────────────────────────────────────────
        record.query_text = pii::scrub(&record.query_text);

        // ── Enqueue ───────────────────────────────────────────────────────
        match self.tx.try_send(WriteCmd(record)) {
            Ok(_) => {
                self.records_queued.fetch_add(1, Ordering::Relaxed);
                metrics::counter!("training_buffer.records_queued").increment(1);
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                warn!("training buffer channel full — dropping record");
                metrics::counter!("training_buffer.records_dropped").increment(1);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                error!("training buffer writer task has exited unexpectedly");
            }
        }
    }

    /// Total records enqueued since this handle was created.
    pub fn records_queued(&self) -> u64 {
        self.records_queued.load(Ordering::Relaxed)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Background writer task
// ─────────────────────────────────────────────────────────────────────────────

async fn writer_task(
    db_path: PathBuf,
    mut rx: mpsc::Receiver<WriteCmd>,
    records_written: Arc<AtomicU64>,
) {
    info!(db = %db_path.display(), "training buffer writer started");

    while let Some(WriteCmd(record)) = rx.recv().await {
        let path = db_path.clone();
        let written = Arc::clone(&records_written);

        // rusqlite is synchronous — run on a blocking thread.
        let result = tokio::task::spawn_blocking(move || {
            let conn = Connection::open(&path)?;
            // Ensure WAL is still active (in case the file was re-created).
            conn.pragma_update_and_check(None, "journal_mode", "WAL", |_row| Ok(()))?;
            insert_record(&conn, &record)
        })
        .await;

        match result {
            Ok(Ok(())) => {
                written.fetch_add(1, Ordering::Relaxed);
                metrics::counter!("training_buffer.records_written").increment(1);
            }
            Ok(Err(e)) => {
                warn!(error = %e, "failed to write training record");
                metrics::counter!("training_buffer.write_errors").increment(1);
            }
            Err(e) => {
                error!(error = %e, "training buffer spawn_blocking panicked");
            }
        }
    }

    info!("training buffer writer shutting down");
}

// ─────────────────────────────────────────────────────────────────────────────
// SQLite helpers
// ─────────────────────────────────────────────────────────────────────────────

fn init_schema(conn: &Connection) -> rusqlite::Result<()> {
    // PRAGMA journal_mode returns the resulting mode as a result row.
    // Use pragma_update_and_check to consume the row (ignore the value).
    conn.pragma_update_and_check(None, "journal_mode", "WAL", |_row| Ok(()))?;
    // PRAGMA synchronous does not return a row; pragma_update is safe.
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS training_records (
             record_id        TEXT    PRIMARY KEY,
             timestamp        TEXT    NOT NULL,
             query_text       TEXT    NOT NULL,
             teacher_response TEXT    NOT NULL,
             grinder_response TEXT,
             confidence_score REAL    NOT NULL,
             domain_tag       TEXT    NOT NULL,
             teacher_logits   BLOB,
             status           TEXT    NOT NULL DEFAULT 'PENDING',
             in_retention     INTEGER NOT NULL DEFAULT 0,
             provider_type    TEXT    NOT NULL
         );

         CREATE INDEX IF NOT EXISTS idx_status ON training_records(status);
         CREATE INDEX IF NOT EXISTS idx_domain  ON training_records(domain_tag);
        ",
    )
}

fn insert_record(conn: &Connection, r: &TrainingRecord) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO training_records
            (record_id, timestamp, query_text, teacher_response, grinder_response,
             confidence_score, domain_tag, teacher_logits, status, in_retention, provider_type)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'PENDING', 0, ?9)",
        params![
            r.record_id,
            r.timestamp,
            r.query_text,
            r.teacher_response,
            r.grinder_response,
            r.confidence_score,
            r.domain_tag.as_str(),
            r.teacher_logits,
            r.provider_type,
        ],
    )?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Builder helper
// ─────────────────────────────────────────────────────────────────────────────

/// Construct a [`TrainingRecord`] from the fields available at escalation time.
pub fn build_record(
    query_text: impl Into<String>,
    teacher_response: impl Into<String>,
    grinder_response: Option<String>,
    confidence_score: f32,
    domain_tag: DomainTag,
    provider_type: impl Into<String>,
) -> TrainingRecord {
    TrainingRecord {
        record_id: Ulid::new().to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        query_text: query_text.into(),
        teacher_response: teacher_response.into(),
        grinder_response,
        confidence_score,
        domain_tag,
        teacher_logits: None,
        provider_type: provider_type.into(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn local_record() -> TrainingRecord {
        build_record(
            "What is the capital of France?",
            "Paris.",
            None,
            0.55,
            DomainTag::Qa,
            "local_open_source",
        )
    }

    fn commercial_record() -> TrainingRecord {
        build_record(
            "Explain quantum entanglement.",
            "Quantum entanglement is...",
            None,
            0.40,
            DomainTag::Reasoning,
            "commercial_api",
        )
    }

    #[tokio::test]
    async fn noop_buffer_does_not_panic() {
        let buf = TrainingBuffer::noop();
        buf.insert(local_record());
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    #[tokio::test]
    async fn commercial_records_are_blocked() {
        let buf = TrainingBuffer::noop();
        buf.insert(commercial_record());
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        // Commercial records must not increment the queued counter.
        assert_eq!(
            buf.records_queued(),
            0,
            "commercial record should be blocked"
        );
    }

    #[tokio::test]
    async fn local_records_are_queued() {
        let buf = TrainingBuffer::noop();
        buf.insert(local_record());
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        assert_eq!(buf.records_queued(), 1);
    }

    #[tokio::test]
    async fn pii_is_scrubbed_before_insert() {
        let buf = TrainingBuffer::noop();
        let mut rec = local_record();
        rec.query_text = "Email me at secret@corp.com".into();
        buf.insert(rec.clone());
        // The scrubbed text is applied in insert(); we test the scrubber
        // directly here since noop() discards the record.
        let scrubbed = pii::scrub(&rec.query_text);
        assert!(scrubbed.contains("[EMAIL]"));
        assert!(!scrubbed.contains("secret@corp.com"));
    }

    #[tokio::test]
    async fn writes_to_sqlite() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("training_buffer.sqlite");
        let buf = TrainingBuffer::open(&db_path).unwrap();

        buf.insert(local_record());
        buf.insert(local_record());
        buf.insert(commercial_record()); // should be blocked

        // Allow background writer to flush.
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // Verify via direct SQLite read.
        let conn = Connection::open(&db_path).unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM training_records WHERE status = 'PENDING'",
                [],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(count, 2, "expected 2 PENDING records (commercial blocked)");
    }

    #[tokio::test]
    async fn schema_has_all_columns() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("schema_test.sqlite");
        let _buf = TrainingBuffer::open(&db_path).unwrap();

        let conn = Connection::open(&db_path).unwrap();
        let cols: Vec<String> = {
            let mut stmt = conn.prepare("PRAGMA table_info(training_records)").unwrap();
            stmt.query_map([], |row| row.get::<_, String>(1))
                .unwrap()
                .filter_map(|r| r.ok())
                .collect()
        };

        for expected in &[
            "record_id",
            "timestamp",
            "query_text",
            "teacher_response",
            "grinder_response",
            "confidence_score",
            "domain_tag",
            "teacher_logits",
            "status",
            "in_retention",
            "provider_type",
        ] {
            assert!(
                cols.iter().any(|c| c == expected),
                "missing column: {expected}"
            );
        }
    }
}
