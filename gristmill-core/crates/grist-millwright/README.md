# grist-millwright

DAG pipeline orchestrator for GristMill. Executes directed acyclic graphs of steps in parallel with checkpoints, approval gates, retry, and configurable failure policies.

## Purpose

`grist-millwright` is the task execution engine. When the Sieve routes an event to a pipeline, Millwright:

1. Resolves the step dependency graph (topological sort)
2. Executes independent steps in parallel (bounded by `max_concurrency`)
3. Persists checkpoints so interrupted pipelines can resume
4. Publishes `pipeline.completed` / `pipeline.failed` events on the bus

## Key Types

```rust
pub struct Millwright { /* opaque */ }

pub struct Pipeline {
    pub id: String,
    pub steps: Vec<Step>,
    pub default_mode: ExecutionMode,
    pub max_concurrency: usize,              // Default: num_cpus - 1
    pub checkpoint_strategy: CheckpointStrategy,
    pub timeout_ms: u64,
    pub on_failure: FailurePolicy,
}

pub struct Step {
    pub id: String,
    pub step_type: StepType,
    pub depends_on: Vec<String>,             // DAG edges (step IDs)
    pub prefer_local: bool,                  // Default: true
    pub fallback_to_llm: bool,
    pub requires_approval: bool,
    pub approval_channel: Option<String>,
    pub timeout_ms: Option<u64>,
    pub retry: Option<RetryPolicy>,
}

pub enum StepType {
    LocalML { model_id: String },
    Rule { rule_id: String },
    Llm { prompt_template: String, max_tokens: u32 },
    External { action: String, config: serde_json::Value },
    Gate { condition: String },
    PythonCall { module: String, function: String },
    TypeScriptCall { adapter: String, method: String },
}

pub enum FailurePolicy {
    FailFast,           // Abort on first step failure (default)
    ContinueOnError,    // Run all steps, collect errors
    SkipAndContinue,    // Skip failed steps, proceed with successors
}

pub struct RetryPolicy {
    pub max_attempts: u32,
    pub backoff_ms: u64,           // Base backoff (doubles each retry)
    pub max_backoff_ms: u64,
}

pub struct PipelineResult {
    pub run_id: Ulid,
    pub pipeline_id: String,
    pub succeeded: bool,
    pub elapsed_ms: u64,
    pub output: serde_json::Value, // Merged outputs of all steps
    pub step_results: Vec<StepResult>,
}
```

## Public API

```rust
// Create orchestrator
let millwright = Millwright::new(config, bus).await?;

// Register a pipeline
millwright.register_pipeline(pipeline);

// Run a pipeline
let result: PipelineResult = millwright.run("my-pipeline", &event).await?;

// Builder API for Pipeline
let pipeline = Pipeline::new("content-digest")
    .with_step(
        Step::new("extract", StepType::LocalML { model_id: "ner-v1".into() })
    )
    .with_step(
        Step::new("summarise", StepType::Llm {
            prompt_template: "Summarise: {{extract.output}}".into(),
            max_tokens: 256,
        }).depends_on(["extract"])
    );
```

## Pipeline Execution

```
Pipeline.run(event)
  │
  ├─ Topological sort of steps
  ├─ Spawn ready steps concurrently (semaphore: max_concurrency)
  │
  ├─ For each step:
  │   ├─ Check requires_approval → wait for approval signal
  │   ├─ Execute StepType handler
  │   ├─ On failure: apply RetryPolicy → exponential backoff
  │   └─ Checkpoint write (atomic JSON to checkpoints/)
  │
  ├─ Apply FailurePolicy on step errors
  ├─ Collect outputs → merge into PipelineResult
  │
  └─ Publish to bus:
      pipeline.completed { run_id, pipeline_id, output }
      pipeline.failed    { run_id, pipeline_id, error }
```

## Checkpointing

Checkpoints are written atomically to `~/.gristmill/checkpoints/{pipeline_id}-{timestamp}.json`. If the daemon crashes mid-execution, a resumed run can skip already-completed steps.

```yaml
# ~/.gristmill/config.yaml
millwright:
  max_concurrency: 8
  default_timeout_ms: 30000
  checkpoint_dir: ~/.gristmill/checkpoints/
```

## Bus Events Published

| Topic | Payload |
|-------|---------|
| `pipeline.completed` | `{ run_id, pipeline_id, elapsed_ms, output }` |
| `pipeline.failed` | `{ run_id, pipeline_id, error, step_id }` |

Bell Tower watches can subscribe to these topics to trigger notifications.

## Dependencies

```toml
grist-event  = { path = "../grist-event" }
grist-bus    = { path = "../grist-bus" }
tokio        # async step execution
rayon        # CPU-parallel step dispatching
dashmap      # concurrent pipeline registry
metrics      # step timing histograms
tracing      # pipeline trace logging
```
