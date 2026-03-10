//! grist-ffi — FFI bridges for GristMill core.
//!
//! This crate provides thin, zero-business-logic bindings that expose the
//! GristMill Rust core to:
//!
//! - **Python** via [PyO3] (feature `python`) — consumed by `gristmill-ml`.
//! - **Node.js** via [napi-rs] (feature `node`) — consumed by
//!   `gristmill-integrations`.
//!
//! Both bridges surface a single [`GristMillCore`] aggregate that holds all
//! subsystems (Sieve, Ledger, Hammer, Millwright, Bus).  Construction is
//! async; the bridges each provide language-appropriate async wrappers.
//!
//! # Architecture rule
//!
//! Per `CLAUDE.md`: *"Keep pyo3_bridge.rs and napi_bridge.rs thin — no
//! business logic."*  All routing decisions, memory operations, and LLM calls
//! are delegated to the underlying crates.
//!
//! [PyO3]: https://pyo3.rs
//! [napi-rs]: https://napi.rs

pub mod core;
pub mod error;

#[cfg(feature = "python")]
pub mod pyo3_bridge;

#[cfg(feature = "node")]
pub mod napi_bridge;

/// Re-export the aggregate core so bridge modules (and tests) can import it
/// from one place.
pub use core::GristMillCore;
pub use error::FfiError;

// ── PyO3 module registration ──────────────────────────────────────────────────

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python module entry-point (called by the CPython import machinery).
///
/// Registers all Python-visible classes and functions under the
/// `gristmill_core` module name.
#[cfg(feature = "python")]
#[pymodule]
fn gristmill_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    use pyo3_bridge::{PyGristEvent, PyGristMill, PyMemory, PyRouteDecision};
    m.add_class::<PyGristMill>()?;
    m.add_class::<PyGristEvent>()?;
    m.add_class::<PyRouteDecision>()?;
    m.add_class::<PyMemory>()?;
    Ok(())
}

// ── napi-rs module registration ───────────────────────────────────────────────
// napi-rs generates its own `napi_register_module_v1` via the proc-macro;
// no manual registration is required here.
