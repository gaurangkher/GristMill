//! Retry and timeout policies for pipeline steps.
//!
//! Each step may declare a [`RetryPolicy`] that controls:
//! - How many times to re-attempt after a failure.
//! - How long to wait between attempts (exponential back-off with jitter).
//! - Maximum total time to spend retrying (global retry budget).
//!
//! The [`RetryExecutor`] is a helper that wraps an async closure and applies
//! the policy, returning `Ok(T)` on success or the last error on exhaustion.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

// ─────────────────────────────────────────────────────────────────────────────
// RetryPolicy
// ─────────────────────────────────────────────────────────────────────────────

/// Retry behaviour for a single pipeline step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of additional attempts after the first failure.
    /// Total attempts = `max_retries + 1`.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// Initial delay before the first retry (ms).
    #[serde(default = "default_initial_delay_ms")]
    pub initial_delay_ms: u64,

    /// Multiplicative factor applied to the delay after each attempt.
    /// 1.0 = constant delay, 2.0 = doubling (standard exponential back-off).
    #[serde(default = "default_backoff_factor")]
    pub backoff_factor: f64,

    /// Maximum delay between retries (ms) — caps exponential growth.
    #[serde(default = "default_max_delay_ms")]
    pub max_delay_ms: u64,

    /// Add random jitter up to ±20 % of the computed delay to avoid
    /// thundering-herd on shared resources.
    #[serde(default = "default_true")]
    pub jitter: bool,
}

fn default_max_retries() -> u32 {
    3
}
fn default_initial_delay_ms() -> u64 {
    500
}
fn default_backoff_factor() -> f64 {
    2.0
}
fn default_max_delay_ms() -> u64 {
    30_000
}
fn default_true() -> bool {
    true
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: default_max_retries(),
            initial_delay_ms: default_initial_delay_ms(),
            backoff_factor: default_backoff_factor(),
            max_delay_ms: default_max_delay_ms(),
            jitter: true,
        }
    }
}

impl RetryPolicy {
    /// Compute the delay before attempt number `attempt` (0-indexed, so
    /// `attempt = 0` is the *first* retry, i.e. after the initial failure).
    pub fn delay(&self, attempt: u32) -> Duration {
        let base = self.initial_delay_ms as f64 * self.backoff_factor.powi(attempt as i32);
        let capped = base.min(self.max_delay_ms as f64);
        let millis = if self.jitter {
            // ±20 % jitter using a fast LCG-derived pseudo-random value
            // (no dependency on `rand` crate).
            let jitter_factor = 1.0 + (pseudo_rand_jitter() - 0.5) * 0.4;
            (capped * jitter_factor).max(0.0)
        } else {
            capped
        };
        Duration::from_millis(millis as u64)
    }
}

/// Simple deterministic pseudo-random value in [0, 1) for jitter.
/// Uses a time-seeded bit mix — good enough for back-off jitter.
fn pseudo_rand_jitter() -> f64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(12345);
    // Avalanche mix
    let mut x = nanos as u64;
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    // Map to [0, 1)
    (x >> 11) as f64 / (1u64 << 53) as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// RetryExecutor
// ─────────────────────────────────────────────────────────────────────────────

/// Execute an async closure with the given retry policy.
///
/// ```text
/// let result = run_with_retry(&policy, "my-step", || async {
///     do_work().await
/// }).await;
/// ```
pub async fn run_with_retry<F, Fut, T, E>(
    policy: &RetryPolicy,
    step_id: &str,
    mut f: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Debug,
{
    let max_attempts = policy.max_retries + 1;

    for attempt in 0..max_attempts {
        match f().await {
            Ok(v) => {
                if attempt > 0 {
                    debug!(step_id, attempt, "step succeeded after retry");
                    metrics::counter!("millwright.step.retry_success", "step_id" => step_id.to_owned())
                        .increment(1);
                }
                return Ok(v);
            }
            Err(e) => {
                let is_last = attempt + 1 >= max_attempts;
                if is_last {
                    warn!(step_id, attempt, ?e, "step failed, retry budget exhausted");
                    return Err(e);
                }
                let delay = policy.delay(attempt);
                warn!(
                    step_id,
                    attempt,
                    ?e,
                    delay_ms = delay.as_millis(),
                    "step failed, will retry",
                );
                metrics::counter!("millwright.step.retries", "step_id" => step_id.to_owned())
                    .increment(1);
                tokio::time::sleep(delay).await;
            }
        }
    }

    unreachable!("loop always returns")
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_has_three_retries() {
        let p = RetryPolicy::default();
        assert_eq!(p.max_retries, 3);
    }

    #[test]
    fn delay_grows_exponentially() {
        let p = RetryPolicy {
            initial_delay_ms: 100,
            backoff_factor: 2.0,
            max_delay_ms: 10_000,
            jitter: false, // disable jitter for determinism
            ..Default::default()
        };
        assert_eq!(p.delay(0), Duration::from_millis(100));
        assert_eq!(p.delay(1), Duration::from_millis(200));
        assert_eq!(p.delay(2), Duration::from_millis(400));
    }

    #[test]
    fn delay_is_capped_at_max() {
        let p = RetryPolicy {
            initial_delay_ms: 1_000,
            backoff_factor: 100.0,
            max_delay_ms: 5_000,
            jitter: false,
            ..Default::default()
        };
        // 1000 * 100^5 >> 5000
        assert_eq!(p.delay(5), Duration::from_millis(5_000));
    }

    #[test]
    fn delay_with_jitter_is_within_bounds() {
        let p = RetryPolicy {
            initial_delay_ms: 1_000,
            backoff_factor: 1.0,
            max_delay_ms: 10_000,
            jitter: true,
            ..Default::default()
        };
        for _ in 0..20 {
            let d = p.delay(0).as_millis();
            // ±20% of 1000 → [600, 1400]
            assert!(
                (600..=1_400).contains(&d),
                "jitter delay {d} out of expected range"
            );
        }
    }

    #[tokio::test]
    async fn run_with_retry_succeeds_on_first_try() {
        let policy = RetryPolicy {
            max_retries: 3,
            ..Default::default()
        };
        let mut calls = 0u32;
        let result: Result<i32, String> = run_with_retry(&policy, "test", || {
            calls += 1;
            async { Ok(42) }
        })
        .await;
        assert_eq!(result.unwrap(), 42);
        assert_eq!(calls, 1);
    }

    #[tokio::test]
    async fn run_with_retry_retries_and_succeeds() {
        let policy = RetryPolicy {
            max_retries: 3,
            initial_delay_ms: 1, // tiny delay for test speed
            jitter: false,
            ..Default::default()
        };
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let calls2 = calls.clone();
        let result: Result<i32, String> = run_with_retry(&policy, "test", || {
            let c = calls2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async move {
                if c < 2 {
                    Err("not yet".to_string())
                } else {
                    Ok(99)
                }
            }
        })
        .await;
        assert_eq!(result.unwrap(), 99);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn run_with_retry_exhausts_and_returns_last_error() {
        let policy = RetryPolicy {
            max_retries: 2,
            initial_delay_ms: 1,
            jitter: false,
            ..Default::default()
        };
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let calls2 = calls.clone();
        let result: Result<i32, String> = run_with_retry(&policy, "test", || {
            calls2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async { Err("always fails".to_string()) }
        })
        .await;
        assert!(result.is_err());
        // 1 initial + 2 retries = 3 total attempts
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 3);
    }
}
