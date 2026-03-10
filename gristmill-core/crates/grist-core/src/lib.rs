//! `grist-core` — thread-safe aggregate of all GristMill subsystems.
//!
//! Both the daemon (via the Unix-socket IPC server) and the FFI layer
//! (`grist-ffi`) wrap [`GristMillCore`] so all business logic lives in one
//! place.
//!
//! Construction is **async** because [`grist_ledger::Ledger::new`] is async.

pub mod embedder;
pub mod error;
pub use error::CoreError;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use grist_bus::EventBus;
use grist_config::GristMillConfig;
use grist_event::{ChannelType, GristEvent};
use grist_grinders::GrindersConfig;
use grist_hammer::{EscalationRequest, EscalationResponse, Hammer, HammerConfig};
use grist_ledger::{Ledger, LedgerConfig, Memory, RankedMemory};
use grist_millwright::{Millwright, MillwrightConfig, Pipeline, PipelineResult};
use grist_sieve::{RouteDecision, Sieve, SieveConfig};
use serde_json::Value;
use tracing::info;

// ─────────────────────────────────────────────────────────────────────────────
// GristMillCore
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe aggregate of all GristMill subsystems.
///
/// Wrap in [`Arc`] and share across async tasks / language bindings.
pub struct GristMillCore {
    pub(crate) sieve: Sieve,
    pub(crate) ledger: Ledger,
    pub(crate) hammer: Hammer,
    pub(crate) millwright: Millwright,
    pub bus: Arc<EventBus>,
}

impl GristMillCore {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Build a core from an optional config file path.
    ///
    /// When `config_path` is `None` (or the path doesn't exist), defaults are
    /// used — useful for tests and local development.  Environment variable
    /// overrides (e.g. `ANTHROPIC_API_KEY`, `SLACK_WEBHOOK_URL`) are always
    /// applied on top of the loaded or default config.
    pub async fn new(config_path: Option<PathBuf>) -> Result<Self, CoreError> {
        let cfg = match config_path {
            Some(p) => GristMillConfig::load_from(p)
                .map_err(|e| CoreError::config(e.to_string()))?,
            None => GristMillConfig::default(),
        }
        .apply_env();

        let workspace = cfg.core.workspace.clone();
        info!(workspace = %workspace.display(), "GristMillCore initialising subsystems");

        // ── Sieve ────────────────────────────────────────────────────────────
        let sieve = Sieve::new(build_sieve_config(&cfg)).map_err(CoreError::Sieve)?;

        // ── Bus ──────────────────────────────────────────────────────────────
        let bus = Arc::new(EventBus::default());

        // ── Grinders config (supplies the MiniLM embedder to the Ledger) ─────
        let grinders_cfg = build_grinders_config(&cfg, &workspace);

        // ── Ledger ───────────────────────────────────────────────────────────
        let ledger_embedder = embedder::build_ledger_embedder(&grinders_cfg);
        let ledger = Ledger::new(build_ledger_config(&cfg, &workspace), ledger_embedder)
            .await
            .map_err(CoreError::Ledger)?;

        // ── Hammer ───────────────────────────────────────────────────────────
        let hammer = Hammer::new(build_hammer_config(&cfg)).map_err(CoreError::Hammer)?;

        // ── Millwright ───────────────────────────────────────────────────────
        let millwright =
            Millwright::new(build_millwright_config(&cfg, &workspace), Some(Arc::clone(&bus)))
                .map_err(CoreError::Millwright)?;

        info!("GristMillCore ready");
        Ok(Self { sieve, ledger, hammer, millwright, bus })
    }

    // ── Sieve ─────────────────────────────────────────────────────────────────

    pub async fn triage(&self, event: &GristEvent) -> Result<RouteDecision, CoreError> {
        self.sieve.triage(event).await.map_err(CoreError::Sieve)
    }

    // ── Ledger ────────────────────────────────────────────────────────────────

    pub async fn remember(
        &self,
        content: impl Into<String> + Send,
        tags: Vec<String>,
    ) -> Result<String, CoreError> {
        self.ledger
            .remember(content, tags)
            .await
            .map_err(CoreError::Ledger)
    }

    pub async fn recall(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RankedMemory>, CoreError> {
        self.ledger.recall(query, limit).await.map_err(CoreError::Ledger)
    }

    pub async fn get_memory(&self, id: &str) -> Result<Option<Memory>, CoreError> {
        self.ledger.get(id).await.map_err(CoreError::Ledger)
    }

    // ── Hammer ────────────────────────────────────────────────────────────────

    pub async fn escalate(
        &self,
        prompt: impl Into<String> + Send,
        max_tokens: u32,
    ) -> Result<EscalationResponse, CoreError> {
        let req = EscalationRequest::new(prompt, max_tokens);
        self.hammer.escalate(req).await.map_err(CoreError::Hammer)
    }

    // ── Millwright ────────────────────────────────────────────────────────────

    pub fn register_pipeline(&self, pipeline: Pipeline) {
        self.millwright.register_pipeline(pipeline);
    }

    pub async fn run_pipeline(
        &self,
        pipeline_id: &str,
        event: &GristEvent,
    ) -> Result<PipelineResult, CoreError> {
        self.millwright
            .run(pipeline_id, event)
            .await
            .map_err(CoreError::Millwright)
    }

    pub fn pipeline_ids(&self) -> Vec<String> {
        self.millwright.pipeline_ids()
    }

    // ── Bus ───────────────────────────────────────────────────────────────────

    pub fn subscribe(&self, topic: &str) -> grist_bus::Subscription {
        self.bus.subscribe(topic)
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a [`GristEvent`] from a channel string and raw JSON payload.
    ///
    /// `channel` must be one of: `http`, `websocket`, `cli`, `cron`,
    /// `webhook`, `mq`, `fs`, `python`, `typescript`, `internal`.
    /// Unknown values fall back to `cli`.
    pub fn build_event(channel: &str, payload: Value) -> GristEvent {
        GristEvent::new(parse_channel(channel), payload)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Config builders
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve `path` relative to `base` when it is not already absolute.
fn resolve(path: &Path, base: &Path) -> PathBuf {
    if path.is_absolute() { path.to_owned() } else { base.join(path) }
}

fn build_sieve_config(cfg: &GristMillConfig) -> SieveConfig {
    SieveConfig {
        model_path: cfg.sieve.model.clone(),
        confidence_threshold: cfg.sieve.confidence_threshold,
        feedback_dir: cfg.sieve.feedback_dir.clone(),
        exact_cache_size: cfg.sieve.cache_size,
        ..SieveConfig::default()
    }
}

fn build_grinders_config(cfg: &GristMillConfig, workspace: &Path) -> GrindersConfig {
    use grist_grinders::config::{ModelConfig as GrindersModelConfig, ModelRuntime};
    use std::time::Duration;

    let models: Vec<GrindersModelConfig> = cfg
        .grinders
        .models
        .iter()
        .map(|(model_id, mc)| GrindersModelConfig {
            model_id: model_id.clone(),
            path: resolve(&mc.path, workspace),
            runtime: match mc.runtime.as_str() {
                "gguf" => ModelRuntime::Gguf,
                _ => ModelRuntime::Onnx,
            },
            warm: mc.warm,
            timeout: Duration::from_millis(mc.timeout_ms),
            max_tokens: mc.max_tokens.unwrap_or(128) as usize,
            description: String::new(),
        })
        .collect();

    let defaults = GrindersConfig::default();
    GrindersConfig {
        worker_threads: if cfg.grinders.workers > 0 {
            cfg.grinders.workers
        } else {
            defaults.worker_threads
        },
        models,
        ..defaults
    }
}

fn build_hammer_config(cfg: &GristMillConfig) -> HammerConfig {
    use grist_hammer::config::{
        AnthropicConfig, BatchConfig, BudgetConfig, CacheConfig, OllamaConfig, ProvidersConfig,
    };

    HammerConfig {
        providers: ProvidersConfig {
            anthropic: AnthropicConfig {
                api_key: cfg.hammer.providers.anthropic.api_key.clone(),
                default_model: cfg.hammer.providers.anthropic.default_model.clone(),
                fallback_model: cfg.hammer.providers.anthropic.fallback_model.clone(),
                base_url: cfg.hammer.providers.anthropic.base_url.clone(),
            },
            ollama: OllamaConfig {
                base_url: cfg.hammer.providers.ollama.base_url.clone(),
                model: cfg.hammer.providers.ollama.model.clone(),
            },
        },
        budget: BudgetConfig {
            daily_tokens: cfg.hammer.budget.daily_tokens,
            monthly_tokens: cfg.hammer.budget.monthly_tokens,
        },
        cache: CacheConfig {
            enabled: cfg.hammer.cache.enabled,
            similarity_threshold: cfg.hammer.cache.similarity_threshold,
            max_entries: cfg.hammer.cache.max_entries,
            ..CacheConfig::default()
        },
        batch: BatchConfig {
            enabled: cfg.hammer.batch.enabled,
            window_ms: cfg.hammer.batch.window_ms,
            max_batch_size: cfg.hammer.batch.max_batch_size,
        },
    }
}

fn build_millwright_config(cfg: &GristMillConfig, workspace: &Path) -> MillwrightConfig {
    MillwrightConfig {
        max_concurrency: cfg.millwright.max_concurrency,
        default_timeout_ms: cfg.millwright.default_timeout_ms,
        checkpoint_dir: resolve(&cfg.millwright.checkpoint_dir, workspace),
    }
}

fn build_ledger_config(cfg: &GristMillConfig, workspace: &Path) -> LedgerConfig {
    use grist_ledger::config::{ColdConfig, CompactorConfig, HotConfig, WarmConfig};

    LedgerConfig {
        hot: HotConfig {
            lru_capacity: HotConfig::default().lru_capacity,
            sled_path: workspace.join("ledger").join("hot"),
        },
        warm: WarmConfig {
            db_path: resolve(&cfg.ledger.warm.db_path, workspace),
            vector_index_path: resolve(&cfg.ledger.warm.vector_index_path, workspace),
            ..WarmConfig::default()
        },
        cold: ColdConfig {
            archive_dir: resolve(&cfg.ledger.cold.archive_dir, workspace),
            compress_level: cfg.ledger.cold.compress_level,
        },
        compactor: CompactorConfig {
            interval_secs: cfg.ledger.compaction.interval_hours * 3600,
            similarity_threshold: cfg.ledger.compaction.similarity_threshold,
            verbose_threshold_tokens: cfg.ledger.compaction.verbose_threshold_tokens,
            stale_days: cfg.ledger.compaction.stale_days,
        },
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

pub fn parse_channel(channel: &str) -> ChannelType {
    match channel.to_lowercase().as_str() {
        "http"       => ChannelType::Http,
        "websocket"  => ChannelType::WebSocket,
        "cli"        => ChannelType::Cli,
        "cron"       => ChannelType::Cron,
        "webhook"    => ChannelType::Webhook { provider: "generic".into() },
        "mq"         => ChannelType::MessageQueue { topic: "default".into() },
        "fs"         => ChannelType::FileSystem { path: "/".into() },
        "python"     => ChannelType::Python { callback_id: "default".into() },
        "typescript" => ChannelType::TypeScript { adapter_id: "default".into() },
        "internal"   => ChannelType::Internal { subsystem: "core".into() },
        other        => {
            tracing::warn!(channel = other, "unknown channel type, falling back to Cli");
            ChannelType::Cli
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `GristMillCore` using an isolated tmpdir so parallel tests don't
    /// fight over the shared sled lock at `~/.gristmill/ledger/hot`.
    async fn core_for_test() -> GristMillCore {
        let dir = tempfile::tempdir().expect("tmpdir");
        let mut cfg = GristMillConfig::default();
        cfg.core.workspace = dir.path().to_owned();
        // Re-derive ledger paths from the temporary workspace.
        let workspace = dir.path();
        let ledger_embedder = embedder::build_ledger_embedder(
            &build_grinders_config(&cfg, workspace),
        );
        let ledger = Ledger::new(build_ledger_config(&cfg, workspace), ledger_embedder)
            .await
            .expect("ledger");
        let sieve = Sieve::new(build_sieve_config(&cfg)).expect("sieve");
        let bus = Arc::new(EventBus::default());
        let hammer = Hammer::new(build_hammer_config(&cfg)).expect("hammer");
        let millwright =
            Millwright::new(build_millwright_config(&cfg, workspace), Some(Arc::clone(&bus)))
                .expect("millwright");
        // Keep `dir` alive for the duration of the test by leaking it.
        std::mem::forget(dir);
        GristMillCore { sieve, ledger, hammer, millwright, bus }
    }

    #[test]
    fn parse_channel_http() {
        assert!(matches!(parse_channel("http"), ChannelType::Http));
    }

    #[test]
    fn parse_channel_case_insensitive() {
        assert!(matches!(parse_channel("HTTP"), ChannelType::Http));
        assert!(matches!(parse_channel("WebSocket"), ChannelType::WebSocket));
    }

    #[test]
    fn parse_channel_unknown_falls_back_to_cli() {
        assert!(matches!(parse_channel("bogus"), ChannelType::Cli));
    }

    #[test]
    fn build_event_produces_valid_event() {
        let ev = GristMillCore::build_event("cli", serde_json::json!({"text": "hello"}));
        assert!(matches!(ev.source, ChannelType::Cli));
        assert_eq!(ev.payload["text"], "hello");
    }

    #[tokio::test]
    async fn core_new_with_defaults() {
        let core = core_for_test().await;
        assert!(core.pipeline_ids().is_empty());
    }

    #[tokio::test]
    async fn core_triage_returns_decision() {
        let core = core_for_test().await;
        let ev = GristMillCore::build_event("cli", serde_json::json!({"text": "hello"}));
        let decision = core.triage(&ev).await.unwrap();
        assert!((0.0..=1.0).contains(&decision.confidence()));
    }

    #[tokio::test]
    async fn core_remember_and_get() {
        // `recall()` searches the warm tier (SQLite), which is populated on hot-tier
        // eviction — not synchronously after `remember()`. Use `get()` to verify
        // immediate retrieval from the hot tier.
        let core = core_for_test().await;
        let id = core
            .remember("GristMill is a Rust-first AI engine", vec!["rust".into()])
            .await
            .unwrap();
        assert!(!id.is_empty());
        let mem = core.get_memory(&id).await.unwrap();
        assert!(mem.is_some());
        assert!(mem.unwrap().content.contains("GristMill"));
    }

    #[tokio::test]
    async fn core_get_memory_missing_returns_none() {
        let core = core_for_test().await;
        let mem = core.get_memory("01ARZ3NDEKTSV4RRFFQ69G5FAV").await.unwrap();
        assert!(mem.is_none());
    }

    #[tokio::test]
    async fn core_register_and_list_pipeline() {
        use grist_millwright::{Pipeline, Step, StepType};
        let core = core_for_test().await;
        assert!(core.pipeline_ids().is_empty());
        let step = Step::new("step-1", StepType::LocalMl { model_id: "noop".into() });
        let pipeline = Pipeline::new("pipe-1").with_step(step);
        core.register_pipeline(pipeline);
        assert_eq!(core.pipeline_ids(), vec!["pipe-1"]);
    }

    #[tokio::test]
    async fn core_subscribe_and_publish() {
        let core = core_for_test().await;
        let mut sub = core.subscribe("test.topic");
        core.bus.publish(
            "test.topic",
            serde_json::json!({"msg": "ping"}),
        );
        let evt = sub.try_recv();
        assert!(evt.is_ok());
    }

    #[test]
    fn config_wiring_sieve_maps_confidence_threshold() {
        use grist_config::{GristMillConfig, SieveConfig as CfgSieve};
        let mut cfg = GristMillConfig::default();
        cfg.sieve = CfgSieve { confidence_threshold: 0.70, cache_size: 500, ..CfgSieve::default() };
        let sc = build_sieve_config(&cfg);
        assert!((sc.confidence_threshold - 0.70).abs() < 1e-6);
        assert_eq!(sc.exact_cache_size, 500);
    }

    #[test]
    fn config_wiring_hammer_maps_budget() {
        use grist_config::{GristMillConfig, HammerBudgetConfig};
        let mut cfg = GristMillConfig::default();
        cfg.hammer.budget = HammerBudgetConfig { daily_tokens: 123_456, monthly_tokens: 9_999_999 };
        let hc = build_hammer_config(&cfg);
        assert_eq!(hc.budget.daily_tokens, 123_456);
        assert_eq!(hc.budget.monthly_tokens, 9_999_999);
    }

    #[test]
    fn config_wiring_millwright_resolves_checkpoint_dir() {
        use grist_config::GristMillConfig;
        let cfg = GristMillConfig::default();
        let workspace = PathBuf::from("/tmp/test-workspace");
        let mc = build_millwright_config(&cfg, &workspace);
        assert!(mc.checkpoint_dir.starts_with("/tmp/test-workspace"));
    }

    #[test]
    fn config_wiring_ledger_maps_compactor_interval() {
        use grist_config::GristMillConfig;
        let mut cfg = GristMillConfig::default();
        cfg.ledger.compaction.interval_hours = 12;
        let workspace = PathBuf::from("/tmp/test-workspace");
        let lc = build_ledger_config(&cfg, &workspace);
        assert_eq!(lc.compactor.interval_secs, 12 * 3600);
    }
}
