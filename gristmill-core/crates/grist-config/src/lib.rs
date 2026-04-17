//! `grist-config` — Configuration management for GristMill.
//!
//! Loads configuration from `~/.gristmill/config.yaml` with optional
//! environment-variable overlay.  Each field can be overridden via an env var
//! whose name mirrors the YAML path with dots replaced by underscores and
//! uppercased (e.g. `GRISTMILL_CORE_LOG_LEVEL`).
//!
//! # Example
//! ```rust
//! use grist_config::GristMillConfig;
//! let config = GristMillConfig::default();
//! assert_eq!(config.core.log_level, "info");
//! ```

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tracing::warn;

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("I/O error reading config file: {0}")]
    Io(#[from] std::io::Error),

    #[error("YAML parse error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("configuration error: {0}")]
    Other(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level config
// ─────────────────────────────────────────────────────────────────────────────

/// Top-level GristMill configuration (`~/.gristmill/config.yaml`).
///
/// All fields have sensible defaults via [`Default`].  Any sub-section omitted
/// from the YAML file will fall back to its defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct GristMillConfig {
    pub core: CoreConfig,
    pub sieve: SieveConfig,
    pub grinders: GrindersConfig,
    pub hammer: HammerConfig,
    pub millwright: MillwrightConfig,
    pub ledger: LedgerConfig,
    pub bell_tower: BellTowerConfig,
    pub integrations: IntegrationsConfig,
}

impl GristMillConfig {
    /// Load config from the default location (`~/.gristmill/config.yaml`).
    ///
    /// Falls back to [`GristMillConfig::default()`] if the file is missing.
    pub fn load() -> Result<Self, ConfigError> {
        Self::load_from(default_config_path())
    }

    /// Load config from an explicit path.
    ///
    /// Falls back to [`GristMillConfig::default()`] if the file does not exist.
    pub fn load_from(path: PathBuf) -> Result<Self, ConfigError> {
        if !path.exists() {
            warn!(path = %path.display(), "config file not found, using defaults");
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(&path)?;
        let cfg: GristMillConfig = serde_yaml::from_str(&content)?;
        Ok(cfg)
    }

    /// Apply environment-variable overrides.
    ///
    /// Reads `GRISTMILL_*` env vars and applies them over the parsed values.
    pub fn apply_env(mut self) -> Self {
        macro_rules! env_override {
            ($env:literal => $field:expr) => {
                if let Ok(v) = std::env::var($env) {
                    $field = v;
                }
            };
            ($env:literal => $field:expr, parse) => {
                if let Ok(v) = std::env::var($env) {
                    if let Ok(parsed) = v.parse() {
                        $field = parsed;
                    }
                }
            };
        }

        env_override!("GRISTMILL_CORE_LOG_LEVEL"              => self.core.log_level);
        env_override!("GRISTMILL_CORE_MODE"                    => self.core.mode);
        env_override!("GRISTMILL_SIEVE_CONFIDENCE_THRESHOLD"   => self.sieve.confidence_threshold, parse);
        env_override!("ANTHROPIC_API_KEY"                      => self.hammer.providers.anthropic.api_key);
        env_override!("GRISTMILL_HAMMER_ANTHROPIC_API_KEY"     => self.hammer.providers.anthropic.api_key);
        env_override!("GRISTMILL_HAMMER_ANTHROPIC_MODEL"       => self.hammer.providers.anthropic.default_model);
        env_override!("GRISTMILL_HAMMER_OLLAMA_BASE_URL"       => self.hammer.providers.ollama.base_url);
        env_override!("GRISTMILL_HAMMER_OLLAMA_MODEL"          => self.hammer.providers.ollama.model);
        env_override!("GRISTMILL_HAMMER_BUDGET_DAILY"          => self.hammer.budget.daily_tokens, parse);
        env_override!("GRISTMILL_HAMMER_BUDGET_MONTHLY"        => self.hammer.budget.monthly_tokens, parse);
        env_override!("SLACK_WEBHOOK_URL"                      => self.bell_tower.channels.slack.webhook_url);
        env_override!("TELEGRAM_BOT_TOKEN"                     => self.bell_tower.channels.telegram.bot_token);
        env_override!("TELEGRAM_CHAT_ID"                       => self.bell_tower.channels.telegram.chat_id);
        env_override!("EMAIL_USER"                             => self.bell_tower.channels.email.username);
        env_override!("EMAIL_PASS"                             => self.bell_tower.channels.email.password);
        env_override!("SLACK_APP_TOKEN"                        => self.integrations.slack.app_token);
        env_override!("SLACK_BOT_TOKEN"                        => self.integrations.slack.bot_token);
        env_override!("SLACK_SIGNING_SECRET"                   => self.integrations.slack.signing_secret);
        env_override!("SLACK_REPLY_MODE"                       => self.integrations.slack.reply_mode);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CoreConfig
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CoreConfig {
    /// GristMill workspace directory.
    pub workspace: PathBuf,
    /// Log level (trace | debug | info | warn | error).
    pub log_level: String,
    /// Runtime mode (daemon | cli | embedded).
    pub mode: String,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            workspace: home_dir().join(".gristmill"),
            log_level: "info".to_string(),
            mode: "daemon".to_string(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SieveConfig
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SieveConfig {
    /// Path to the ONNX sieve model file.
    pub model: Option<PathBuf>,
    /// Confidence below which events are escalated (spec: 0.85).
    pub confidence_threshold: f32,
    /// Directory where routing feedback is written (JSONL).
    pub feedback_dir: Option<PathBuf>,
    /// In-memory result cache size (LRU entries).
    pub cache_size: usize,
    /// Path to the SQLite WAL distillation training buffer.
    /// Defaults to `~/.gristmill/db/training_buffer.sqlite`.
    pub training_buffer_path: Option<PathBuf>,
}

impl Default for SieveConfig {
    fn default() -> Self {
        Self {
            model: None,
            confidence_threshold: 0.85,
            feedback_dir: None,
            cache_size: 10_000,
            training_buffer_path: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GrindersConfig
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct GrindersConfig {
    /// Number of inference workers (0 = auto = CPU cores − 1).
    pub workers: usize,
    /// Per-model configuration map.
    pub models: std::collections::HashMap<String, ModelConfig>,
}

/// Configuration for a single ML model entry under `grinders.models.*`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Inference runtime (onnx | gguf | tflite).
    pub runtime: String,
    /// Path to the model file.
    pub path: PathBuf,
    /// Whether to pre-load the model at startup.
    pub warm: bool,
    /// Maximum batch size.
    pub max_batch: usize,
    /// Inference timeout in milliseconds.
    pub timeout_ms: u64,
    /// Maximum tokens to generate (GGUF/LLM models).
    pub max_tokens: Option<u32>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            runtime: "onnx".to_string(),
            path: PathBuf::new(),
            warm: false,
            max_batch: 32,
            timeout_ms: 500,
            max_tokens: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HammerConfig (mirrors grist-hammer but avoids cross-crate dep)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct HammerConfig {
    pub providers: HammerProvidersConfig,
    pub budget: HammerBudgetConfig,
    pub cache: HammerCacheConfig,
    pub batch: HammerBatchConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct HammerProvidersConfig {
    pub anthropic: AnthropicProviderConfig,
    pub ollama: OllamaProviderConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AnthropicProviderConfig {
    pub api_key: String,
    pub default_model: String,
    pub fallback_model: String,
    pub base_url: String,
}

impl Default for AnthropicProviderConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
            default_model: "claude-sonnet-4-20250514".to_string(),
            fallback_model: "claude-haiku-4-5-20251001".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OllamaProviderConfig {
    /// Set to `false` to skip Ollama entirely (no warning spam when Ollama is
    /// not installed).  Defaults to `true` so existing deployments are unaffected.
    pub enabled: bool,
    pub base_url: String,
    pub model: String,
}

impl Default for OllamaProviderConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_url: "http://localhost:11434".to_string(),
            model: "llama3.1:8b".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HammerBudgetConfig {
    pub daily_tokens: u64,
    pub monthly_tokens: u64,
}

impl Default for HammerBudgetConfig {
    fn default() -> Self {
        Self {
            daily_tokens: 500_000,
            monthly_tokens: 10_000_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HammerCacheConfig {
    pub enabled: bool,
    pub similarity_threshold: f32,
    pub max_entries: usize,
}

impl Default for HammerCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            similarity_threshold: 0.92,
            max_entries: 50_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HammerBatchConfig {
    pub enabled: bool,
    pub window_ms: u64,
    pub max_batch_size: usize,
}

impl Default for HammerBatchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            window_ms: 5_000,
            max_batch_size: 10,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MillwrightConfig
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MillwrightConfig {
    pub max_concurrency: usize,
    pub default_timeout_ms: u64,
    pub checkpoint_dir: PathBuf,
}

impl Default for MillwrightConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 8,
            default_timeout_ms: 30_000,
            checkpoint_dir: PathBuf::from("./checkpoints"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LedgerConfig
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct LedgerConfig {
    pub hot: LedgerHotConfig,
    pub warm: LedgerWarmConfig,
    pub cold: LedgerColdConfig,
    pub compaction: LedgerCompactionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LedgerHotConfig {
    pub max_size_mb: u64,
}

impl Default for LedgerHotConfig {
    fn default() -> Self {
        Self { max_size_mb: 512 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LedgerWarmConfig {
    pub db_path: PathBuf,
    pub vector_index_path: PathBuf,
    pub fts_enabled: bool,
}

impl Default for LedgerWarmConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("./memory/warm.db"),
            vector_index_path: PathBuf::from("./memory/vectors.usearch"),
            fts_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LedgerColdConfig {
    pub archive_dir: PathBuf,
    pub compression: String,
    pub compress_level: i32,
}

impl Default for LedgerColdConfig {
    fn default() -> Self {
        Self {
            archive_dir: PathBuf::from("./memory/cold"),
            compression: "zstd".to_string(),
            compress_level: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LedgerCompactionConfig {
    pub interval_hours: u64,
    pub similarity_threshold: f32,
    pub verbose_threshold_tokens: usize,
    pub stale_days: u64,
}

impl Default for LedgerCompactionConfig {
    fn default() -> Self {
        Self {
            interval_hours: 6,
            similarity_threshold: 0.90,
            verbose_threshold_tokens: 512,
            stale_days: 90,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BellTowerConfig
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct BellTowerConfig {
    pub channels: BellTowerChannels,
    pub quiet_hours: QuietHoursConfig,
    pub digest: DigestConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct BellTowerChannels {
    pub slack: SlackConfig,
    pub telegram: TelegramConfig,
    pub email: EmailConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SlackConfig {
    pub webhook_url: String,
}

impl Default for SlackConfig {
    fn default() -> Self {
        Self {
            webhook_url: std::env::var("SLACK_WEBHOOK_URL").unwrap_or_default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TelegramConfig {
    pub bot_token: String,
    pub chat_id: String,
}

impl Default for TelegramConfig {
    fn default() -> Self {
        Self {
            bot_token: std::env::var("TELEGRAM_BOT_TOKEN").unwrap_or_default(),
            chat_id: std::env::var("TELEGRAM_CHAT_ID").unwrap_or_default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmailConfig {
    pub smtp_host: String,
    pub smtp_port: u16,
    pub username: String,
    pub password: String,
}

impl Default for EmailConfig {
    fn default() -> Self {
        Self {
            smtp_host: "smtp.gmail.com".to_string(),
            smtp_port: 587,
            username: std::env::var("EMAIL_USER").unwrap_or_default(),
            password: std::env::var("EMAIL_PASS").unwrap_or_default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct QuietHoursConfig {
    pub start: String,
    pub end: String,
    pub timezone: String,
    pub override_for: Vec<String>,
}

impl Default for QuietHoursConfig {
    fn default() -> Self {
        Self {
            start: "22:00".to_string(),
            end: "07:00".to_string(),
            timezone: "UTC".to_string(),
            override_for: vec!["critical".to_string()],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DigestConfig {
    pub enabled: bool,
    pub interval_minutes: u64,
    pub max_items: usize,
}

impl Default for DigestConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_minutes: 60,
            max_items: 50,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IntegrationsConfig
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IntegrationsConfig {
    pub dashboard: DashboardConfig,
    pub plugins_dir: PathBuf,
    /// Slack inbound integration (Socket Mode + HTTP Events API).
    pub slack: SlackIntegrationConfig,
}

impl Default for IntegrationsConfig {
    fn default() -> Self {
        Self {
            dashboard: DashboardConfig::default(),
            plugins_dir: PathBuf::from("./plugins"),
            slack: SlackIntegrationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DashboardConfig {
    pub enabled: bool,
    pub port: u16,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 8420,
        }
    }
}

/// Configuration for Slack inbound integration.
///
/// ```yaml
/// integrations:
///   slack:
///     app_token: "xapp-..."        # Socket Mode (connections:write scope)
///     bot_token: "xoxb-..."        # Bot OAuth token (for replies)
///     signing_secret: "..."        # HTTP Events API request verification
///     reply_mode: "thread"         # "thread" | "off"
/// ```
///
/// Tokens can also be supplied via env vars (`SLACK_APP_TOKEN`, etc.) which
/// take precedence over the YAML values when `apply_env()` is called.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SlackIntegrationConfig {
    /// App-level token (`xapp-…`) — required for Socket Mode.
    pub app_token: String,
    /// Bot OAuth token (`xoxb-…`) — required for posting replies.
    pub bot_token: String,
    /// Signing secret — used to verify HTTP Events API requests.
    pub signing_secret: String,
    /// Reply mode: `"thread"` (default) or `"off"`.
    pub reply_mode: String,
}

impl Default for SlackIntegrationConfig {
    fn default() -> Self {
        Self {
            app_token: std::env::var("SLACK_APP_TOKEN").unwrap_or_default(),
            bot_token: std::env::var("SLACK_BOT_TOKEN").unwrap_or_default(),
            signing_secret: std::env::var("SLACK_SIGNING_SECRET").unwrap_or_default(),
            reply_mode: "thread".to_string(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Default config file path: `~/.gristmill/config.yaml`.
pub fn default_config_path() -> PathBuf {
    home_dir().join(".gristmill").join("config.yaml")
}

fn home_dir() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Default values ────────────────────────────────────────────────────

    #[test]
    fn default_core_config() {
        let c = CoreConfig::default();
        assert_eq!(c.log_level, "info");
        assert_eq!(c.mode, "daemon");
        assert!(c.workspace.to_string_lossy().contains(".gristmill"));
    }

    #[test]
    fn default_sieve_config() {
        let c = SieveConfig::default();
        assert!((c.confidence_threshold - 0.85).abs() < 1e-6);
        assert_eq!(c.cache_size, 10_000);
        assert!(c.model.is_none());
    }

    #[test]
    fn default_hammer_config() {
        let c = HammerConfig::default();
        assert_eq!(c.budget.daily_tokens, 500_000);
        assert_eq!(c.budget.monthly_tokens, 10_000_000);
        assert!((c.cache.similarity_threshold - 0.92).abs() < 1e-6);
        assert_eq!(c.cache.max_entries, 50_000);
        assert_eq!(c.batch.window_ms, 5_000);
        assert_eq!(c.batch.max_batch_size, 10);
        assert_eq!(
            c.providers.anthropic.default_model,
            "claude-sonnet-4-20250514"
        );
        assert_eq!(c.providers.ollama.model, "llama3.1:8b");
    }

    #[test]
    fn default_millwright_config() {
        let c = MillwrightConfig::default();
        assert_eq!(c.max_concurrency, 8);
        assert_eq!(c.default_timeout_ms, 30_000);
    }

    #[test]
    fn default_ledger_config() {
        let c = LedgerConfig::default();
        assert_eq!(c.hot.max_size_mb, 512);
        assert!(c.warm.fts_enabled);
        assert_eq!(c.cold.compress_level, 3);
        assert_eq!(c.compaction.interval_hours, 6);
        assert!((c.compaction.similarity_threshold - 0.90).abs() < 1e-6);
        assert_eq!(c.compaction.stale_days, 90);
        assert_eq!(c.compaction.verbose_threshold_tokens, 512);
    }

    #[test]
    fn default_bell_tower_config() {
        let c = BellTowerConfig::default();
        assert_eq!(c.quiet_hours.start, "22:00");
        assert_eq!(c.quiet_hours.end, "07:00");
        assert_eq!(c.quiet_hours.timezone, "UTC");
        assert!(c.quiet_hours.override_for.contains(&"critical".to_string()));
        assert!(c.digest.enabled);
        assert_eq!(c.digest.interval_minutes, 60);
        assert_eq!(c.digest.max_items, 50);
    }

    #[test]
    fn default_integrations_config() {
        let c = IntegrationsConfig::default();
        assert!(c.dashboard.enabled);
        assert_eq!(c.dashboard.port, 8420);
    }

    #[test]
    fn default_grinders_config_has_no_models() {
        let c = GrindersConfig::default();
        assert!(c.models.is_empty());
    }

    // ── Serialization round-trips ─────────────────────────────────────────

    #[test]
    fn config_json_roundtrip() {
        let c = GristMillConfig::default();
        let json = serde_json::to_string(&c).unwrap();
        let c2: GristMillConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(c.core.log_level, c2.core.log_level);
        assert_eq!(c.hammer.budget.daily_tokens, c2.hammer.budget.daily_tokens);
        assert_eq!(
            c.ledger.compaction.stale_days,
            c2.ledger.compaction.stale_days
        );
    }

    #[test]
    fn config_yaml_roundtrip() {
        let c = GristMillConfig::default();
        let yaml = serde_yaml::to_string(&c).unwrap();
        let c2: GristMillConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(c.core.mode, c2.core.mode);
        assert_eq!(c.sieve.cache_size, c2.sieve.cache_size);
        assert_eq!(c.millwright.max_concurrency, c2.millwright.max_concurrency);
    }

    // ── YAML file loading ─────────────────────────────────────────────────

    #[test]
    fn load_from_missing_file_returns_defaults() {
        let path = PathBuf::from("/tmp/gristmill_nonexistent_config_abc123.yaml");
        let c = GristMillConfig::load_from(path).unwrap();
        assert_eq!(c.core.log_level, "info");
    }

    #[test]
    fn load_from_partial_yaml_fills_remaining_with_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.yaml");
        std::fs::write(&path, "core:\n  log_level: debug\n").unwrap();
        let c = GristMillConfig::load_from(path).unwrap();
        assert_eq!(c.core.log_level, "debug");
        assert_eq!(c.hammer.budget.daily_tokens, 500_000);
        assert_eq!(c.sieve.cache_size, 10_000);
    }

    #[test]
    fn load_full_yaml_spec() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.yaml");
        let yaml = r#"
core:
  log_level: warn
  mode: cli

sieve:
  confidence_threshold: 0.90
  cache_size: 5000

hammer:
  budget:
    daily_tokens: 100000
    monthly_tokens: 2000000
  cache:
    enabled: false
    similarity_threshold: 0.95
    max_entries: 1000
  batch:
    window_ms: 2000
    max_batch_size: 5

millwright:
  max_concurrency: 4
  default_timeout_ms: 10000

ledger:
  hot:
    max_size_mb: 256
  compaction:
    interval_hours: 12
    stale_days: 30

bell_tower:
  digest:
    enabled: false
    interval_minutes: 30

integrations:
  dashboard:
    port: 9000
"#;
        std::fs::write(&path, yaml).unwrap();
        let c = GristMillConfig::load_from(path).unwrap();
        assert_eq!(c.core.log_level, "warn");
        assert_eq!(c.core.mode, "cli");
        assert!((c.sieve.confidence_threshold - 0.90).abs() < 1e-6);
        assert_eq!(c.sieve.cache_size, 5000);
        assert_eq!(c.hammer.budget.daily_tokens, 100_000);
        assert!(!c.hammer.cache.enabled);
        assert!((c.hammer.cache.similarity_threshold - 0.95).abs() < 1e-6);
        assert_eq!(c.hammer.batch.window_ms, 2000);
        assert_eq!(c.millwright.max_concurrency, 4);
        assert_eq!(c.ledger.hot.max_size_mb, 256);
        assert_eq!(c.ledger.compaction.interval_hours, 12);
        assert_eq!(c.ledger.compaction.stale_days, 30);
        assert!(!c.bell_tower.digest.enabled);
        assert_eq!(c.integrations.dashboard.port, 9000);
    }

    #[test]
    fn load_grinders_models_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.yaml");
        let yaml = r#"
grinders:
  workers: 4
  models:
    intent-classifier:
      runtime: onnx
      path: ./models/intent-v3.onnx
      warm: true
      max_batch: 32
      timeout_ms: 500
    summarizer:
      runtime: gguf
      path: ./models/phi3-mini.gguf
      warm: false
      max_tokens: 512
"#;
        std::fs::write(&path, yaml).unwrap();
        let c = GristMillConfig::load_from(path).unwrap();
        assert_eq!(c.grinders.workers, 4);
        assert_eq!(c.grinders.models.len(), 2);
        let intent = &c.grinders.models["intent-classifier"];
        assert_eq!(intent.runtime, "onnx");
        assert!(intent.warm);
        let summ = &c.grinders.models["summarizer"];
        assert_eq!(summ.runtime, "gguf");
        assert_eq!(summ.max_tokens, Some(512));
    }

    // ── training_buffer_path ──────────────────────────────────────────────

    #[test]
    fn training_buffer_path_parsed_from_yaml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.yaml");
        let yaml = r#"
sieve:
  training_buffer_path: /data/gristmill/db/training_buffer.sqlite
"#;
        std::fs::write(&path, yaml).unwrap();
        let c = GristMillConfig::load_from(path).unwrap();
        let buf_path = c
            .sieve
            .training_buffer_path
            .as_ref()
            .expect("training_buffer_path should be Some after parsing YAML");
        assert_eq!(
            buf_path.to_string_lossy(),
            "/data/gristmill/db/training_buffer.sqlite",
            "training_buffer_path should match the value in YAML"
        );
    }

    #[test]
    fn training_buffer_path_defaults_to_none() {
        let c = SieveConfig::default();
        assert!(
            c.training_buffer_path.is_none(),
            "training_buffer_path should default to None when not set in YAML"
        );
    }

    // ── apply_env ─────────────────────────────────────────────────────────

    #[test]
    fn apply_env_does_not_crash_without_vars() {
        // When no GRISTMILL_* vars are set, apply_env should be a no-op.
        let c = GristMillConfig::default().apply_env();
        assert!(!c.core.log_level.is_empty());
        assert_eq!(c.hammer.batch.max_batch_size, 10);
    }

    // ── default_config_path ───────────────────────────────────────────────

    #[test]
    fn default_config_path_ends_with_expected_suffix() {
        let p = default_config_path();
        let s = p.to_string_lossy();
        assert!(s.ends_with("config.yaml"), "got: {s}");
        assert!(s.contains(".gristmill"), "got: {s}");
    }
}
