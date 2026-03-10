//! Re-export [`GristMillCore`] from `grist-core`.
//!
//! Business logic lives in `grist-core`; this module keeps the import path
//! `crate::core::GristMillCore` stable for the bridge modules.
pub use grist_core::GristMillCore;
