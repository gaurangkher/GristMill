//! Error types for grist-hammer.

/// All errors produced by the LLM escalation gateway.
#[derive(Debug, thiserror::Error)]
pub enum HammerError {
    #[error("budget exceeded: daily used {daily_used}/{daily_limit} tokens")]
    BudgetExceeded { daily_used: u64, daily_limit: u64 },

    #[error("all providers failed: {0}")]
    AllProvidersFailed(String),

    #[error("provider error ({provider}): {reason}")]
    ProviderError { provider: String, reason: String },

    #[error("request timeout after {elapsed_ms}ms")]
    Timeout { elapsed_ms: u64 },

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
