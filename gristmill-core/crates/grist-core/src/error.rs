//! Core error type aggregating failures from all GristMill subsystems.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("config error: {0}")]
    Config(String),

    #[error("sieve error: {0}")]
    Sieve(#[from] grist_sieve::SieveError),

    #[error("ledger error: {0}")]
    Ledger(#[from] grist_ledger::LedgerError),

    #[error("hammer error: {0}")]
    Hammer(#[from] grist_hammer::HammerError),

    #[error("millwright error: {0}")]
    Millwright(#[from] grist_millwright::MillwrightError),

    #[error("event error: {0}")]
    Event(#[from] grist_event::EventError),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("runtime error: {0}")]
    Runtime(String),
}

impl CoreError {
    pub fn config(msg: impl Into<String>) -> Self {
        CoreError::Config(msg.into())
    }

    pub fn runtime(msg: impl Into<String>) -> Self {
        CoreError::Runtime(msg.into())
    }
}
