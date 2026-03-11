//! Token budget manager for grist-hammer.
//!
//! Tracks daily and monthly token consumption.  Resets automatically when
//! the UTC day or month rolls over.  All state is guarded by a
//! [`parking_lot::Mutex`] so check + update is atomic.

use chrono::{Datelike, NaiveDate, Utc};
use parking_lot::Mutex;
use tracing::{debug, warn};

use crate::config::BudgetConfig;
use crate::error::HammerError;
use crate::types::BudgetInfo;

// ─────────────────────────────────────────────────────────────────────────────
// BudgetState
// ─────────────────────────────────────────────────────────────────────────────

struct BudgetState {
    daily_used: u64,
    monthly_used: u64,
    last_daily_reset: NaiveDate,
    last_monthly_reset: (i32, u32), // (year, month)
}

impl BudgetState {
    fn new() -> Self {
        let today = Utc::now().date_naive();
        Self {
            daily_used: 0,
            monthly_used: 0,
            last_daily_reset: today,
            last_monthly_reset: (today.year(), today.month()),
        }
    }

    /// Apply automatic resets if the UTC day or month has rolled over.
    fn maybe_reset(&mut self) {
        let today = Utc::now().date_naive();
        let (year, month) = (today.year(), today.month());

        if today != self.last_daily_reset {
            debug!("daily budget reset");
            self.daily_used = 0;
            self.last_daily_reset = today;
        }
        if (year, month) != self.last_monthly_reset {
            debug!("monthly budget reset");
            self.monthly_used = 0;
            self.last_monthly_reset = (year, month);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BudgetManager
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe token budget manager.
pub struct BudgetManager {
    state: Mutex<BudgetState>,
    config: BudgetConfig,
}

impl BudgetManager {
    pub fn new(config: BudgetConfig) -> Self {
        Self {
            state: Mutex::new(BudgetState::new()),
            config,
        }
    }

    /// Check whether `estimated_tokens` can be consumed without exceeding limits.
    ///
    /// Returns [`HammerError::BudgetExceeded`] if either the daily or monthly
    /// limit would be breached.  Does NOT modify the counters.
    pub fn check(&self, estimated_tokens: u32) -> Result<(), HammerError> {
        let mut state = self.state.lock();
        state.maybe_reset();

        let tokens = estimated_tokens as u64;

        if state.daily_used + tokens > self.config.daily_tokens {
            warn!(
                daily_used = state.daily_used,
                daily_limit = self.config.daily_tokens,
                requested = tokens,
                "daily budget exceeded",
            );
            return Err(HammerError::BudgetExceeded {
                daily_used: state.daily_used,
                daily_limit: self.config.daily_tokens,
            });
        }
        if state.monthly_used + tokens > self.config.monthly_tokens {
            warn!(
                monthly_used = state.monthly_used,
                monthly_limit = self.config.monthly_tokens,
                "monthly budget exceeded",
            );
            return Err(HammerError::BudgetExceeded {
                daily_used: state.monthly_used,
                daily_limit: self.config.monthly_tokens,
            });
        }
        Ok(())
    }

    /// Record actual token usage after a successful request.
    pub fn record_usage(&self, tokens: u32) {
        let mut state = self.state.lock();
        state.maybe_reset();
        let t = tokens as u64;
        state.daily_used += t;
        state.monthly_used += t;
        metrics::counter!("hammer.budget.tokens_used").increment(t);
        debug!(tokens, daily_used = state.daily_used, "tokens recorded");
    }

    /// Return a snapshot of current budget state.
    pub fn info(&self) -> BudgetInfo {
        let mut state = self.state.lock();
        state.maybe_reset();
        let daily_remaining = self.config.daily_tokens.saturating_sub(state.daily_used);
        let monthly_remaining = self
            .config
            .monthly_tokens
            .saturating_sub(state.monthly_used);
        BudgetInfo {
            daily_used: state.daily_used,
            daily_limit: self.config.daily_tokens,
            monthly_used: state.monthly_used,
            monthly_limit: self.config.monthly_tokens,
            daily_remaining,
            monthly_remaining,
        }
    }

    /// Force-reset all counters (test helper).
    #[cfg(test)]
    pub fn reset(&self) {
        let mut state = self.state.lock();
        state.daily_used = 0;
        state.monthly_used = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_budget(daily: u64, monthly: u64) -> BudgetManager {
        BudgetManager::new(BudgetConfig {
            daily_tokens: daily,
            monthly_tokens: monthly,
        })
    }

    #[test]
    fn budget_allows_within_limit() {
        let b = make_budget(1000, 10_000);
        assert!(b.check(500).is_ok());
    }

    #[test]
    fn budget_rejects_over_daily_limit() {
        let b = make_budget(100, 10_000);
        b.record_usage(90);
        let err = b.check(20).unwrap_err();
        assert!(matches!(err, HammerError::BudgetExceeded { .. }));
    }

    #[test]
    fn budget_rejects_over_monthly_limit() {
        let b = make_budget(10_000, 100);
        b.record_usage(90);
        let err = b.check(20).unwrap_err();
        assert!(matches!(err, HammerError::BudgetExceeded { .. }));
    }

    #[test]
    fn budget_records_usage() {
        let b = make_budget(10_000, 100_000);
        b.record_usage(300);
        let info = b.info();
        assert_eq!(info.daily_used, 300);
        assert_eq!(info.monthly_used, 300);
    }

    #[test]
    fn budget_info_remaining_is_correct() {
        let b = make_budget(1000, 5000);
        b.record_usage(200);
        let info = b.info();
        assert_eq!(info.daily_remaining, 800);
        assert_eq!(info.monthly_remaining, 4800);
    }

    #[test]
    fn budget_reset_clears_counters() {
        let b = make_budget(1000, 5000);
        b.record_usage(500);
        b.reset();
        assert_eq!(b.info().daily_used, 0);
    }

    #[test]
    fn budget_check_at_exact_limit_is_ok() {
        let b = make_budget(100, 1000);
        // Exactly at the limit — should pass.
        assert!(b.check(100).is_ok());
    }

    #[test]
    fn budget_check_one_over_limit_fails() {
        let b = make_budget(100, 1000);
        assert!(b.check(101).is_err());
    }
}
