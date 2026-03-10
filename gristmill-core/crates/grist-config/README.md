# grist-config

Configuration management for GristMill. Loads YAML config from disk and overlays environment variable overrides.

## Purpose

Provides a single `GristMillConfig` struct that all subsystems receive at startup. Config is loaded once, env vars applied, then distributed to subsystem-specific sub-configs via `grist-core`.

## Key Type

```rust
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
```

### `CoreConfig`

```rust
pub struct CoreConfig {
    pub workspace: PathBuf,    // Default: ~/.gristmill
    pub log_level: String,     // Default: "info"
    pub mode: String,          // "daemon" | "embedded"
}
```

### `SieveConfig` (subset)

```rust
pub struct SieveConfig {
    pub model: Option<PathBuf>,          // ONNX classifier path
    pub confidence_threshold: f32,       // Default: 0.85
    pub feedback_dir: Option<PathBuf>,
    pub cache_size: usize,               // Default: 10000
}
```

### `HammerConfig` (subset)

```rust
pub struct HammerConfig {
    pub providers: ProvidersConfig,
    pub budget: BudgetConfig,
    pub cache: CacheConfig,
}
```

## Public API

```rust
// Load from default path (~/.gristmill/config.yaml)
let config = GristMillConfig::load()?;

// Load from explicit path
let config = GristMillConfig::load_from(&path)?;

// Overlay GRISTMILL_* environment variables
config.apply_env();
```

`apply_env()` is always called after loading to allow secrets and overrides via environment variables without editing the YAML file.

## Environment Variable Overrides

Environment variables follow the pattern `GRISTMILL_<SECTION>_<KEY>` (uppercased, underscored):

| Variable | Overrides |
|----------|-----------|
| `ANTHROPIC_API_KEY` | `hammer.providers.anthropic.api_key` |
| `SLACK_WEBHOOK_URL` | `bell_tower.channels.slack.webhook_url` |
| `EMAIL_USER` | `bell_tower.channels.email.username` |
| `EMAIL_PASS` | `bell_tower.channels.email.password` |
| `GRISTMILL_CONFIG` | Config file path (resolved before loading) |
| `GRISTMILL_SOCK` | IPC socket path (used by daemon, not config) |

## Full Config Reference

```yaml
core:
  workspace: ~/.gristmill
  log_level: info
  mode: daemon

sieve:
  model: ~/.gristmill/models/sieve-v1.onnx
  confidence_threshold: 0.85
  feedback_dir: ~/.gristmill/feedback/
  exact_cache_size: 10000
  semantic_similarity_threshold: 0.92

grinders:
  worker_threads: 0      # 0 = CPU cores - 1
  queue_depth: 1024
  batch_window_ms: 5
  max_batch_size: 32
  models:
    - model_id: minilm-l6-v2
      path: ~/.gristmill/models/minilm-l6-v2.onnx
      runtime: onnx
      warm: true

hammer:
  providers:
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
      default_model: claude-sonnet-4-6
      fallback_model: claude-haiku-4-5-20251001
    ollama:
      base_url: http://localhost:11434
      model: llama3.1:8b
  budget:
    daily_tokens: 500000
    monthly_tokens: 10000000
  cache:
    enabled: true
    similarity_threshold: 0.92
    max_entries: 50000
  batch:
    enabled: true
    window_ms: 5000
    max_batch_size: 10

millwright:
  max_concurrency: 8
  default_timeout_ms: 30000
  checkpoint_dir: ~/.gristmill/checkpoints/

ledger:
  hot:
    max_size_mb: 512
  warm:
    db_path: ~/.gristmill/memory/warm.db
    vector_index_path: ~/.gristmill/memory/vectors.usearch
  cold:
    archive_dir: ~/.gristmill/memory/cold/
    compress_level: 3
  compaction:
    interval_hours: 6
    similarity_threshold: 0.90
    stale_days: 90

bell_tower:
  channels:
    slack:
      webhook_url: ${SLACK_WEBHOOK_URL}
    email:
      smtp_host: smtp.gmail.com
      smtp_port: 587
      username: ${EMAIL_USER}
      password: ${EMAIL_PASS}
  quiet_hours:
    start: "22:00"
    end: "07:00"
    timezone: UTC
    override_for: [critical]
  digest:
    enabled: true
    interval_minutes: 60

integrations:
  dashboard:
    enabled: true
    port: 3000
  plugins_dir: ~/.gristmill/plugins/
```

## Dependencies

```toml
serde       = "1"
serde_yaml  = "0.9"
thiserror   = "1"
tracing     = "0.1"
```
