//! Error types for the `grist-sieve` crate.

#[derive(Debug, thiserror::Error)]
pub enum SieveError {
    #[error("feature extraction failed: {0}")]
    FeatureExtraction(String),

    #[error("model load failed: {0}")]
    ModelLoad(String),

    #[error("inference failed: {0}")]
    Inference(String),

    #[error("feedback log error: {0}")]
    Feedback(String),

    #[error("cache error: {0}")]
    Cache(String),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("event expired (TTL exceeded)")]
    EventExpired,

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
