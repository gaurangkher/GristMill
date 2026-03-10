//! Re-export the core error type under the legacy `FfiError` alias.
//!
//! Bridge modules and external callers may use either name; both resolve to
//! [`grist_core::CoreError`].
pub use grist_core::CoreError;
pub use grist_core::CoreError as FfiError;
