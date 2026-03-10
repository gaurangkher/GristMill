//! Error types for grist-ledger.

/// All errors produced by the ledger subsystem.
#[derive(Debug, thiserror::Error)]
pub enum LedgerError {
    #[error("memory not found: {0}")]
    NotFound(String),

    #[error("hot tier error: {0}")]
    HotTier(String),

    #[error("warm tier error: {0}")]
    WarmTier(String),

    #[error("cold tier error: {0}")]
    ColdTier(String),

    #[error("embedding failed: {0}")]
    Embedding(String),

    #[error("compactor error: {0}")]
    Compactor(String),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
