# GristMill
### Personal AI Distillation Platform

**Product Requirements & Architecture Specification**  
Version 1.1 · March 2026 · Confidential

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Vision & Strategic Position](#3-vision--strategic-position)
4. [System Architecture](#4-system-architecture)
5. [Data Architecture & Privacy](#5-data-architecture--privacy)
6. [User Experience](#6-user-experience)
7. [Phased Roadmap](#7-phased-roadmap)
8. [Risks & Mitigations](#8-risks--mitigations)
9. [Technical Stack](#9-technical-stack)
10. [Success Metrics](#10-success-metrics)
11. [Open Questions for v2 Research](#11-open-questions-for-v2-research)

---

# 1. Executive Summary

GristMill is a personal AI distillation platform that progressively trains small, local AI models on a user's own usage patterns, guided by a larger open-source teacher model. Over time, the local model absorbs enough domain-specific knowledge to handle the user's everyday workload without calling the teacher — delivering zero-latency, private, personalized AI that belongs entirely to the user.

> **The Core Promise**
>
> - Day 1: The teacher model handles everything while the local grinder observes.
> - Week 4: The grinder handles 40–60% of queries autonomously.
> - Month 6+: The grinder handles 80–90% of the user's workload with no API calls required.

The user never configures training pipelines, labels data, or thinks about model weights. GristMill's distillation loop is entirely automatic. The experience is simply an AI assistant that visibly gets better at the user's specific needs over time.

---

# 2. Problem Statement

Current AI assistants suffer from three structural problems that GristMill directly addresses.

## 2.1  Generality at the expense of fit

Frontier LLMs are optimized for breadth across millions of possible tasks. For any individual user with a stable set of recurring needs — a developer who always works in the same codebase, a writer with a consistent style, a support agent answering the same 200 question types — general capability is wasteful. The model carries enormous knowledge the user will never need, while offering no specialization for what they actually do every day.

## 2.2  Persistent external dependency

Every interaction with an LLM API sends user data to a third-party server, incurs network latency (100–500ms round-trips), depends on upstream availability, and costs money on a per-token basis. Users who could be served by a well-trained local model continue paying these costs indefinitely because no mechanism exists to move knowledge from the cloud to the edge over time.

## 2.3  No ownership of the AI relationship

Users build habits and workflows around LLMs but own nothing. When the provider changes pricing, degrades quality, or discontinues access, the user's accumulated context is lost. There is no flywheel — more usage does not produce a more personalized or capable model for that user. GristMill's distillation loop inverts this: more usage produces a better, more specialized local model that the user owns outright.

---

# 3. Vision & Strategic Position

GristMill's long-term vision is a world where every user has a personal AI model that is meaningfully distinct from everyone else's — shaped by their domain, their language, their judgment — and that runs entirely on their hardware. The LLM teacher is a bootstrap mechanism, not a permanent dependency.

The strategic position this creates is not competing with OpenAI on their terms. It is making the LLM API model irrelevant for the user's everyday workload. GristMill's moat is the personalization flywheel:

> **The Personalization Flywheel**
>
> - More usage generates richer training signal from real query patterns.
> - Richer signal produces a more specialized, accurate grinder.
> - A better grinder earns more user trust and handles more query types.
> - Increased trust drives more usage — the cycle accelerates.

The longer a user stays on GristMill, the more specialized their grinder becomes, and the harder it is to replicate that personalization by switching to a raw LLM. This is a durable advantage that LLM API cost deflation cannot erode.

---

# 4. System Architecture

## 4.1  Architecture Overview

GristMill is composed of two independently deployable processes connected by a shared filesystem contract. The Inference Stack handles all real-time user interactions. The `gristmill-trainer` service handles all model training asynchronously. Neither process needs to know the internals of the other — the versioned adapter checkpoint directory is the only shared interface.

> **Two-Process Architecture**
>
> - **Inference Stack:** Chat interface + Sieve + Grinder + Teacher. Always running. Owns the user-facing experience.
> - **gristmill-trainer:** Distillation Engine + LoRA training + checkpoint management + validation. Runs independently. Can be on a different machine.
> - **Contract boundary:** `/gristmill/checkpoints/` directory. Trainer writes versioned adapter checkpoints. Inference Stack file-watches and hot-loads.

This separation means a training crash never kills a chat session, GPU memory is never contended between inference and training simultaneously, and advanced users can run the trainer on a more powerful home server while the inference stack runs on a laptop.

| Process | Component | Language | Responsibility |
|---|---|---|---|
| Inference Stack | Local Grinder | Rust | Serves personalized responses. Hot-loads updated LoRA adapters on checkpoint change. |
| Inference Stack | The Sieve | Rust | Evaluates grinder confidence. Escalates to teacher. Writes training records to buffer. |
| Inference Stack | Teacher Client | Rust | Calls locally-hosted teacher via HTTP. Streams responses. Never logs to external endpoints. |
| gristmill-trainer | Distillation Engine | Python | Reads training buffer. Runs LoRA training cycles. Writes versioned checkpoints. |
| gristmill-trainer | Checkpoint Manager | Python | Versions, validates, and prunes adapter checkpoints. Exposes health/status API. |
| gristmill-trainer | Rollback Controller | Python | Detects quality regression via validation set. Promotes or discards each new checkpoint. |
| Shared | Training Buffer | SQLite | Written by Inference Stack (Sieve). Read by gristmill-trainer. Append-only with WAL mode. |

## 4.2  The Grinder (Local Model)

The grinder is a small (1B–7B parameter) open-source base model, selected based on the user's hardware profile. All domain specialization is applied through LoRA (Low-Rank Adaptation) adapters rather than full fine-tuning, which provides three key properties:

- **Parameter isolation:** New domain knowledge lives in the adapter, not the base weights, preventing interference between domains.
- **Swappability:** Different adapter sets can be loaded for different contexts (coding vs. writing vs. support) from a single base model.
- **Lightweight updates:** Only the adapter weights are updated during distillation cycles, making continuous learning computationally feasible on consumer hardware.

> **Default Models:** Qwen2.5-3B-Instruct or Gemma-3-4B for CPU-only deployments. Llama-3.2-7B for GPU-equipped devices. All under permissive open-source licenses.

## 4.3  The Sieve (Confidence Router)

The Sieve is GristMill's routing layer — the component that decides, for each query, whether the grinder can handle it confidently or whether the teacher must be called. This is the most critical component for both user experience and training signal quality.

### 4.3.1  Confidence Estimation

The primary confidence mechanism is Self-REF (Self-Reflective Confidence Tokens), trained into the grinder from the start. The grinder is fine-tuned to emit a structured confidence token alongside every response:

- `HIGH` — Grinder is confident. Response returned to the user immediately. No teacher call.
- `MED` — Grinder is uncertain. Response is held. Teacher is called. The better response is returned. Both are logged.
- `LOW` — Grinder does not attempt. Teacher handles the query. Query and response logged as priority training data.

A lightweight calibration probe (a small classifier trained on grinder hidden states) runs periodically to recalibrate confidence thresholds based on observed accuracy. This prevents the grinder from becoming either over-confident (returning wrong answers) or under-confident (escalating unnecessarily).

### 4.3.2  Sieve Thresholds

The Sieve threshold is configurable and shifts over time as the grinder improves. In early operation, the threshold is conservative (more escalation). As distillation cycles accumulate, the threshold relaxes automatically. Users can also manually adjust via a simple slider: 'Prefer speed' (lower threshold, more local) vs. 'Prefer accuracy' (higher threshold, more escalation).

## 4.4  The Teacher Model

The teacher is a mid-size open-source model (7B–32B parameter range), selected from the following approved options:

| Model | Size | Strengths | License |
|---|---|---|---|
| Qwen2.5-32B-Instruct | 32B | Reasoning, code, multilingual | Apache 2.0 |
| Llama-3.3-70B-Instruct | 70B | General capability, reasoning | Llama License |
| Mistral-Small-24B | 24B | Instruction following, efficiency | Apache 2.0 |
| Gemma-3-27B-IT | 27B | Safety, multimodal | Gemma ToS |

> **Critical Design Constraint:** GristMill will never use commercial LLM API outputs (OpenAI, Anthropic, etc.) as training data. Doing so violates their Terms of Service and was the basis for API revocations against DeepSeek and others in 2025–2026. All teacher models must be open-source with explicit training use rights.

> **Teacher Size Note:** The Law of Capacity Gap shows that using the largest available teacher is often suboptimal for small student models. A 32B teacher typically produces better 3B–7B students than a 70B+ teacher, due to distribution mismatch. Teacher selection should be matched to the grinder's parameter count.

## 4.5  The Distillation Engine (Inside gristmill-trainer)

The Distillation Engine is the core learning component of the `gristmill-trainer` service. It reads from the shared training buffer, runs LoRA training cycles, and writes versioned adapter checkpoints. It has no direct connection to the inference stack at runtime — all communication is through the filesystem contract described in Section 4.6.

### 4.5.1  Training Signal Collection

Every teacher call in the Inference Stack produces a training record written to the shared SQLite buffer in WAL (Write-Ahead Logging) mode, which allows the Inference Stack to write and `gristmill-trainer` to read concurrently without locking. Records accumulate with status `PENDING`. The Distillation Engine polls for `PENDING` records above the trigger threshold, marks them `IN_TRAINING` during a cycle, and `CONSUMED` on completion.

### 4.5.2  Distillation Technique

GristMill uses output-based distillation (black-box distillation) since open-source teachers are accessed locally via llama.cpp or vLLM, but their internal logits may not be exposed in all deployment modes. The primary loss functions are:

- **Reverse KL divergence** between teacher and student output distributions (where teacher logits are available). Reverse KL concentrates the student on the teacher's major modes rather than forcing coverage of low-probability tails.
- **Cross-entropy** on teacher text outputs (where logits are unavailable — i.e., black-box mode).
- **Chain-of-thought rationale matching** for reasoning tasks, following the Orca 2 methodology.

### 4.5.3  Catastrophic Forgetting Prevention

Each distillation cycle applies a three-part mitigation strategy:

1. **LoRA adapter isolation:** New task knowledge updates domain-specific adapter weights, not base model weights, preventing inter-task interference.
2. **Experience replay:** Each training batch includes a random 15–20% sample from a curated retention buffer — a fixed set of diverse examples covering core capabilities the grinder must not lose.
3. **Functional distillation from prior checkpoint:** The grinder's own previous checkpoint acts as a secondary teacher, penalizing large deviations from prior behavior on non-target-domain inputs.

### 4.5.4  Training Cycle Cadence

Training cycles are triggered by the `gristmill-trainer` scheduler, independently of the Inference Stack's operation:

- **Trigger condition:** 500+ `PENDING` training records in the buffer, OR 7 days elapsed since last cycle.
- **Resource gate:** Trainer checks GPU memory availability before starting. If inference stack is actively serving (detected via lock file heartbeat), trainer defers and retries in 5 minutes.
- **Cycle duration:** 30–120 minutes depending on hardware. Designed to run during idle periods but will not forcibly interrupt active inference.
- **Post-cycle:** Checkpoint written to staging area. Rollback Controller runs validation before promotion.

---

# 4.6  gristmill-trainer Service

`gristmill-trainer` is a standalone Python service that manages the full training lifecycle independently of the Inference Stack. It is the component that makes the long-term distillation vision operational — without it, GristMill is only a smart routing proxy. This section defines its complete specification: process lifecycle, resource negotiation, filesystem contract, IPC protocol, health API, and rollback mechanism.

## 4.6.1  Process Lifecycle

`gristmill-trainer` runs as a background system service (systemd unit on Linux, launchd plist on macOS, Windows Service on Windows). It is installed and enabled at GristMill setup time and persists across reboots independently of the chat interface.

| State | Behaviour |
|---|---|
| `IDLE` | Service is running. Scheduler polls buffer every 60 seconds for trigger condition. No GPU resources held. |
| `WAITING` | Trigger condition met but resource gate blocked (inference stack active or insufficient GPU memory). Retries every 5 minutes. Emits `status: waiting_for_resources`. |
| `TRAINING` | Active distillation cycle. GPU memory claimed via lock file. Emits progress heartbeat every 30 seconds to status socket. UI shows 'Your model is learning.' |
| `VALIDATING` | Training complete. Running post-cycle evaluation against held-out validation set. GPU memory still held. Emits `status: validating`. |
| `PROMOTING` | Validation passed. Moving checkpoint from staging to active. Notifying Inference Stack via socket. Emits `status: checkpoint_promoted, version: N`. |
| `ROLLING_BACK` | Validation failed. Discarding staged checkpoint. Retaining previous active checkpoint. Emits `status: rollback, reason: <metric_delta>`. |
| `PAUSED` | User has toggled 'Pause learning'. Service remains alive, buffer writes continue, but no training cycles are initiated. |

## 4.6.2  Resource Negotiation with Inference Stack

Both the Inference Stack and `gristmill-trainer` may compete for GPU memory on single-machine deployments. Resource negotiation is managed through two mechanisms.

#### Lock File Heartbeat

The Inference Stack writes a heartbeat timestamp to `/gristmill/run/inference.lock` every 10 seconds while actively serving requests. `gristmill-trainer` reads this file before claiming GPU memory. If the heartbeat is stale by more than 30 seconds, the Inference Stack is considered idle and the trainer may proceed. This is a cooperative protocol — neither process forcibly preempts the other.

#### GPU Memory Reservation

`gristmill-trainer` estimates required VRAM before starting a cycle (base model size + adapter parameters + optimizer state, approximately 2× model size in float32). If the estimated requirement exceeds available VRAM minus a 512MB safety margin, the cycle defers. On Apple Silicon (unified memory), the same logic applies to system RAM. The trainer never OOM-kills the inference process.

> **Multi-Machine Deployment:** When `gristmill-trainer` runs on a separate machine (e.g., a home server with a GPU), resource negotiation is not needed — the trainer claims full GPU resources freely. The filesystem contract (Section 4.6.3) works identically over a mounted network share or rsync-based sync.

## 4.6.3  Filesystem Contract

The shared filesystem is the only coupling between `gristmill-trainer` and the Inference Stack.

| Path | Description |
|---|---|
| `/gristmill/checkpoints/active/` | The currently loaded LoRA adapter. Inference Stack file-watches this directory. On change, hot-loads the new adapter between requests. |
| `/gristmill/checkpoints/staging/` | Checkpoint written by trainer after a completed cycle, awaiting validation. Never directly read by Inference Stack. |
| `/gristmill/checkpoints/history/v{N}/` | Versioned archive of previous checkpoints. Retention policy: last 5 versions. Used for rollback. |
| `/gristmill/checkpoints/manifest.json` | Monotonically-versioned JSON. Contains: `current_version`, `promoted_at`, `validation_score`, `record_count_at_promotion`. Atomic write via rename. |
| `/gristmill/db/training_buffer.sqlite` | Shared training buffer in WAL mode. Written by Inference Stack, read by `gristmill-trainer`. |
| `/gristmill/db/retention_buffer.sqlite` | Curated retention set. Managed exclusively by `gristmill-trainer`. |
| `/gristmill/run/inference.lock` | Heartbeat file written by Inference Stack every 10s. Timestamp format: Unix epoch. |
| `/gristmill/run/trainer.status` | JSON status file written by `gristmill-trainer` every 30s during active cycles. Read by UI layer for dashboard display. |
| `/gristmill/run/trainer.sock` | Unix domain socket. `gristmill-trainer` binds. Inference Stack connects to receive checkpoint promotion notifications. |

## 4.6.4  IPC Protocol (trainer.sock)

The Unix domain socket carries a minimal newline-delimited JSON message protocol from `gristmill-trainer` to the Inference Stack. The Inference Stack subscribes on startup. Messages are one-directional: trainer notifies, inference reacts.

| Message Type | Payload & Inference Stack Action |
|---|---|
| `checkpoint_promoted` | `{ version: N, validation_score: 0.87, record_count: 1240 }` — Inference Stack hot-loads adapter from `/active/` between the next two requests. No in-flight request is interrupted. |
| `checkpoint_rolled_back` | `{ version: N, reason: "validation_score_delta: -0.04" }` — Inference Stack continues serving current adapter. Logs event to status file. |
| `training_started` | `{ estimated_duration_minutes: 45, record_count: 620 }` — UI layer updates status indicator to 'Your model is learning.' |
| `training_progress` | `{ pct_complete: 0.42, elapsed_minutes: 19 }` — Emitted every 30 seconds during active cycle. |
| `trainer_paused` | `{ reason: "user_request" \| "low_battery" \| "resource_contention" }` — UI updates status indicator. |

## 4.6.5  Health & Status API

`gristmill-trainer` exposes a lightweight HTTP API on `localhost:7432` for the UI layer and administrative tooling. All endpoints return JSON.

| Endpoint | Response |
|---|---|
| `GET /status` | `{ state, current_version, last_cycle_at, next_trigger_at, buffer_pending_count, autonomy_pct_7d }` |
| `GET /history` | Array of past cycle summaries: `[{ version, promoted_at, validation_score, record_count, duration_minutes, rolled_back }]` |
| `GET /validation/latest` | Full validation report: per-domain accuracy, confidence calibration delta, retention score. |
| `GET /health` | `{ ok: true, uptime_seconds, last_heartbeat_seen }` |
| `POST /pause` | `{ paused: true }` — Suspends scheduling. Current active cycle completes if in progress. |
| `POST /resume` | `{ paused: false }` — Re-enables scheduling. Checks trigger condition immediately. |
| `POST /rollback/{version}` | Promotes a specific historical checkpoint version to active. Notifies Inference Stack via socket. |

## 4.6.6  Checkpoint Validation & Rollback

Every cycle produces a staged checkpoint that must pass validation before being promoted to active. The Rollback Controller runs a two-stage evaluation.

#### Stage 1 — Held-Out Validation Set

A fixed validation set of 200 examples (50 per domain, stratified by confidence score) is created on first setup and never updated. Promotion requires all three thresholds pass:

- Validation accuracy delta >= −0.01 (no more than 1% regression on the held-out set).
- Per-domain accuracy delta >= −0.03 on any single domain.
- Confidence calibration drift <= 0.05 (Expected Calibration Error delta).

#### Stage 2 — Retention Score

The retention buffer is run against the staged adapter. Promotion requires retention score >= 0.90 relative to the prior checkpoint's retention score.

If either stage fails, the staged checkpoint is archived to `/gristmill/checkpoints/history/` with `status: rolled_back`, a `checkpoint_rolled_back` message is emitted on the IPC socket, and the Inference Stack continues serving the last promoted adapter unchanged. Full failure metrics are available via `GET /validation/latest`.

> **Manual Rollback:** `POST /rollback/{version}` promotes any historical checkpoint. Exposed in the Progress Dashboard under 'Advanced' — not shown in the default UI.

---

# 5. Data Architecture & Privacy

## 5.1  Data Residency Principles

GristMill's data architecture is built on a single non-negotiable principle: all user data stays on the user's device. This includes:

- All query inputs and teacher responses stored in the training buffer.
- All model weights, adapter checkpoints, and validation sets.
- All confidence scoring logs and routing decisions.
- The retention buffer and its contents.

No telemetry, no cloud sync of training data, no model weights transmitted. The only network calls are to the locally-hosted teacher model (which may be accessed on a local server or private network endpoint the user controls) and to fetch base model weights on first install from official model repositories.

## 5.2  Training Buffer Schema

| Field | Type | Description |
|---|---|---|
| `record_id` | UUID | Unique identifier for deduplication across distillation cycles. |
| `timestamp` | ISO 8601 | When the query was processed. |
| `query_text` | String | The user's input, after PII scrubbing (see 5.3). |
| `teacher_response` | String | The teacher model's output — the learning target. |
| `grinder_response` | String \| null | The grinder's held response, if confidence was `MED`. Null for `LOW`. |
| `confidence_score` | Float 0–1 | The calibrated confidence score that triggered escalation. |
| `domain_tag` | Enum | Auto-classified: `code`, `writing`, `reasoning`, `qa`, `creative`, `other`. |
| `teacher_logits` | Float[] \| null | Top-k teacher token probabilities, if available in local inference mode. |
| `status` | Enum | Record lifecycle: `PENDING` → `IN_TRAINING` → `CONSUMED`. |
| `in_retention` | Boolean | Whether this record has been selected for the retention buffer. |

## 5.3  PII Scrubbing

Before any query is written to the training buffer, a lightweight local PII scrubber runs to detect and redact identifiable information. This is a hygiene measure — not a privacy guarantee — to prevent obviously sensitive content from appearing in training data the user may later inspect or export. The scrubber detects and redacts: email addresses, phone numbers, credit card numbers, Social Security Numbers, and common name patterns adjacent to sensitive context keywords.

## 5.4  Retention Buffer Curation

The retention buffer is a curated fixed-size set (default 2,000 examples) of diverse, high-quality training records included in every distillation cycle to prevent capability degradation. Curation selects records by:

- **Domain diversity:** Representative coverage of all `domain_tag` categories the grinder has encountered.
- **Difficulty stratification:** Examples spanning the full confidence score range, not only easy queries.
- **Temporal spread:** Records sampled across the full training history, not only recent data.
- **Quality filter:** Teacher responses that scored highly on a local automated quality metric (length-normalized perplexity under the teacher model).

---

# 6. User Experience

## 6.1  Design Principles

GristMill's UX is governed by a single directive: the distillation pipeline must be completely invisible to the user. Every interface element that exposes training mechanics, model weights, or adapter management is a design failure. The user should only ever experience:

- An AI that answers their questions.
- A visible (but passive) indicator that their model is improving over time.
- Simple controls for the few choices that are genuinely theirs to make.

## 6.2  Core Interface States

| State | User Experience |
|---|---|
| Grinder answers | Response appears with normal latency indicator. No distinction between grinder and teacher responses in the default view. |
| Sieve escalates | Brief 'thinking...' indicator while teacher is called. Response appears. No indication of escalation unless user opts into advanced view. |
| Training cycle running | Small pulsing indicator in the status bar: 'Your model is learning.' No interruption to use. |
| Grinder improves | Periodic notification (no more than weekly): 'Your model now handles X% of your queries locally.' Graph shows trajectory. |
| Grinder cold start | First 2 weeks: All responses handled by teacher. User sees: 'Building your model — this takes a few weeks to personalize.' No degraded experience. |

## 6.3  The Progress Dashboard

The one place where GristMill's mechanics are surfaced is an optional Progress Dashboard, accessible from settings:

- Local vs. cloud query split over time (area chart showing grinder autonomy growing).
- Domain coverage: which topics the grinder handles confidently vs. still escalates.
- Estimated cost savings since setup.
- Last training cycle completion date and improvement delta.

This dashboard is not shown by default and is not in the primary navigation. It exists for users who want transparency, not as a core workflow.

## 6.4  User Controls

- **Speed vs. Accuracy slider:** Controls the Sieve threshold. Left = more local (faster, slightly less accurate). Right = more escalation (slower, more reliable). Defaults to center.
- **Pause learning toggle:** Stops the distillation engine. All queries route to teacher. Data collection continues but no training cycles run.
- **Export my model:** Downloads the user's grinder + adapters as a portable bundle. The user owns this and can run it elsewhere.
- **Clear training data:** Deletes the training buffer. Adapters are not affected — only future improvement is paused until new data accumulates.

---

# 7. Phased Roadmap

## Phase 1 — Foundation (Months 1–3): The Sieve

Build the routing layer with confidence estimation. At this phase, GristMill is a smart proxy — it routes every query to the teacher model, scores the response, and logs training data. No grinder training runs yet. The value is establishing clean data collection and validating the confidence calibration mechanism.

> **Phase 1 Deliverables**
>
> - Local teacher model integration (llama.cpp or Ollama backend).
> - Self-REF confidence token training on base grinder model.
> - Sieve routing logic with configurable threshold.
> - Training buffer with PII scrubbing and domain classification.
> - Basic chat interface with status indicator.

## Phase 2 — Distillation Loop (Months 4–6): Grinder Training

Activate the Distillation Engine. First training cycles run on accumulated Phase 1 data. Implement LoRA adapter management, retention buffer curation, and post-cycle validation. At the end of this phase, grinders should be handling 30–50% of queries autonomously for users with 4+ weeks of data.

> **Phase 2 Deliverables**
>
> - `gristmill-trainer` as a standalone system service (systemd / launchd / Windows Service).
> - LoRA adapter training pipeline (PEFT + TRL).
> - Retention buffer curation algorithm.
> - Catastrophic forgetting mitigations: EWC-LoRA + replay.
> - Post-cycle validation, rollback controller, and checkpoint promotion via IPC.
> - `gristmill-trainer` health API on `localhost:7432`.
> - Progress Dashboard v1.

## Phase 3 — Domain Specialization (Months 7–9): Multi-Adapter

Extend the adapter system to support per-domain adapters that can be swapped based on detected query domain. A user's coding grinder and writing grinder become distinct, with specialized knowledge in each. Implement the speculative cascade pattern — for escalated queries, the grinder generates a draft that the teacher verifies rather than generating from scratch, reducing teacher compute cost by 30–50%.

> **Phase 3 Deliverables**
>
> - Multi-domain adapter management.
> - Automatic domain detection and adapter hot-swap.
> - Speculative cascade integration for escalated queries.
> - Teacher compute cost tracking and reporting.

## Phase 4 — Ecosystem (Months 10–12): Portability & Sharing

Enable model export, import, and optional anonymous sharing of adapter weights. Users who work in similar domains may benefit from adapter bundles bootstrapped from community-shared weights — dramatically reducing cold start time. Implement federated learning scaffolding for users who opt into contributing aggregate gradient updates (not raw data) to improve shared base adapters.

> **Phase 4 Deliverables**
>
> - Grinder + adapter portable export format.
> - Community adapter repository (opt-in, anonymized).
> - Cold start acceleration via domain adapter bootstrapping.
> - Federated learning scaffolding (privacy-preserving).

---

# 8. Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Cold start — poor UX before grinder is trained | High | Teacher handles all queries during cold start. No degraded experience. Target: grinder handles 30%+ within 4 weeks for a 5-query-per-day user. |
| Catastrophic forgetting degrades grinder over time | High | Three-layer mitigation: LoRA isolation + replay buffer + functional distillation from prior checkpoint. Automated rollback if post-cycle validation fails. |
| Hallucination propagation from teacher to grinder | High | Teacher responses undergo automated quality scoring before entering training buffer. Low-quality responses flagged and excluded. |
| Sieve over-confidence — wrong answers returned confidently | Medium | Calibration probe recalibrates thresholds monthly. User feedback ('This answer seems wrong') triggers immediate re-escalation and flags the record. |
| Domain drift — grinder degrades on rarely-used topics | Medium | Retention buffer ensures diverse domain coverage in every training cycle. Validation set includes examples from all historical domains. |
| Hardware requirements exclude low-end users | Medium | Phase 1 (Sieve only) runs on any device. Grinder training requires GPU or Apple Silicon. CPU-only path with 1B models available but slower. |
| ToS violations from commercial LLM outputs used as training data | Critical | Architectural constraint enforced in code: only approved open-source teacher models permitted as training data sources. Commercial API calls never logged to training buffer. |

---

# 9. Technical Stack

## 9.1  Inference Stack Runtime

- **Model inference:** llama.cpp (CPU) or llama.cpp with CUDA/Metal (GPU). Ollama as a higher-level wrapper for multi-model management.
- **Teacher model serving:** vLLM for GPU deployments, Ollama for unified local serving.
- **Adapter hot-swap:** inotify (Linux) / FSEvents (macOS) / ReadDirectoryChangesW (Windows) watching `/gristmill/checkpoints/active/`. Zero-downtime reload between requests.
- **Lock file heartbeat:** Written every 10s by Inference Stack to `/gristmill/run/inference.lock`. Read by `gristmill-trainer` for resource negotiation.
- **IPC subscription:** Inference Stack connects to `trainer.sock` on startup. Reconnects with exponential backoff if trainer is not yet running.
- **Orchestration:** Lobster DSL for Sieve routing workflow sequencing.

## 9.2  gristmill-trainer Stack

- **Service management:** systemd (Linux), launchd (macOS), Windows Service. Installed at GristMill setup. Auto-restart on crash.
- **LoRA training:** Hugging Face PEFT + TRL. Trainer reads base model weights from a shared read-only path; only adapter weights are written.
- **Validation runner:** lm-evaluation-harness for held-out set scoring.
- **Health API:** FastAPI on `localhost:7432`. Localhost-only binding.
- **Status socket:** Python asyncio Unix domain socket server on `trainer.sock`.
- **Scheduler:** APScheduler for trigger-condition polling and deferred retry logic.

## 9.3  Data Layer

- **Training buffer:** SQLite in WAL mode at `/gristmill/db/training_buffer.sqlite`. Written by Inference Stack (Rust), read by `gristmill-trainer` (Python). Record lifecycle: `PENDING` → `IN_TRAINING` → `CONSUMED`.
- **Retention buffer:** `/gristmill/db/retention_buffer.sqlite`. Managed exclusively by `gristmill-trainer`.
- **Checkpoint store:** `/gristmill/checkpoints/` per Section 4.6.3. Atomic promotion via filesystem rename.
- **Manifest:** `/gristmill/checkpoints/manifest.json`. Atomic write. Read by both Inference Stack and UI layer.

## 9.4  Language Allocation

Following GristMill's tri-language architecture principle:

- **Rust:** Full Inference Stack — Sieve routing, confidence scoring, grinder inference, teacher client, training buffer writes, adapter hot-swap, lock file heartbeat, IPC socket client.
- **Python:** Full `gristmill-trainer` service — Distillation Engine, LoRA training pipeline, checkpoint validation, rollback controller, retention buffer curation, health API, IPC socket server.
- **TypeScript:** UI layer, Progress Dashboard, advanced controls, manifest polling, trainer health display.

---

# 10. Success Metrics

| Metric | Target | Measurement Method |
|---|---|---|
| Grinder autonomy at 4 weeks | >= 40% | Queries handled without escalation / total queries |
| Grinder autonomy at 3 months | >= 70% | Same |
| Sieve calibration accuracy | >= 90% | % of `HIGH` confidence answers correct per held-out eval |
| Post-cycle rollback rate | <= 5% | % of distillation cycles rolled back due to validation failure |
| Training cycle completion time | <= 90 min (7B grinder) | Wall clock on reference hardware (RTX 3080 or Apple M2) |
| User-reported quality regression | <= 2% | Monthly feedback survey: 'Has your model gotten worse?' |
| Cold start to first 30% autonomy | <= 21 days | Days from first query to grinder handling 30% locally |

---

# 11. Open Questions for v2 Research

The following are known unknowns that require empirical investigation before Phase 3 design is finalized:

1. **Optimal distillation cycle frequency:** Is 500 records the right trigger threshold? Does more frequent cycling improve grinder growth rate, or does it introduce instability?
2. **Retention buffer sizing:** How does retention buffer size (currently 2,000 examples) interact with grinder parameter count? Is 2,000 sufficient for a 7B grinder?
3. **Cross-domain adapter interference:** When a user's queries span many domains, does a unified adapter outperform domain-specific adapters, or does specialization always win?
4. **Federated adapter sharing:** Can anonymous gradient aggregation across users produce a 'community base adapter' that meaningfully reduces cold start without leaking individual data?
5. **Teacher-grinder scale matching:** The Law of Capacity Gap predicts a 32B teacher is optimal for a 7B grinder. Does this hold empirically across GristMill's specific task distributions?

---

*GristMill · Personal AI Distillation Platform · v1.1 Specification*  
*Confidential — Internal Use Only*
