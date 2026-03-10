# gristmill-integrations

TypeScript shell for GristMill. Handles all I/O: inbound event adapters (hoppers), outbound notifications (Bell Tower), a REST API dashboard, and a dynamic plugin system.

> **Rule**: All processing logic delegates to Rust via `GristMillBridge`. TypeScript handles I/O only.

## Structure

```
src/
├── index.ts                  # Re-exports for npm consumers
├── main.ts                   # Entry point (bridge + plugins + dashboard)
├── cli.ts                    # Interactive CLI tool
│
├── core/
│   ├── bridge.ts             # IBridge + NativeBridge + MockBridge + GristMillBridge
│   └── ipc-bridge.ts         # Unix socket client (IpcBridge)
│
├── hopper/                   # Inbound channel adapters
│   ├── index.ts
│   └── http.ts               # HTTP + WebSocket adapter (HttpHopper, WebSocketHopper)
│
├── bell-tower/               # Outbound notifications
│   ├── dispatcher.ts         # NotificationDispatcher
│   ├── watch.ts              # WatchEngine + Watch store (JSON persistence)
│   └── channels/
│       ├── slack.ts          # Slack webhook channel
│       └── email.ts          # SMTP email channel
│
├── dashboard/
│   ├── server.ts             # Fastify app + static SPA hosting
│   └── routes/
│       ├── pipelines.ts      # GET/POST /api/pipelines, POST /api/pipelines/:id/run
│       ├── memory.ts         # POST /api/memory/remember|recall, GET /api/memory/:id
│       ├── metrics.ts        # GET /api/metrics/health|budget
│       ├── triage.ts         # POST /api/triage
│       ├── watches.ts        # CRUD /api/watches
│       └── plugins.ts        # GET /api/plugins
│
└── plugins/
    ├── registry.ts           # Plugin loader (dynamic import from ~/.gristmill/plugins/)
    ├── types.ts              # GristMillPlugin interface
    └── sdk.ts                # Plugin development helpers
```

## Quick Start

```bash
cd gristmill-integrations
pnpm install

# Mock mode — no daemon required
GRISTMILL_MOCK_BRIDGE=1 pnpm dev
# → http://127.0.0.1:3000

# Full mode — requires running gristmill-daemon
pnpm dev

# Interactive CLI
pnpm cli
```

## API Reference

All endpoints are served on port 3000 by default.

### Event Intake

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `POST` | `/events` | `{ channel, payload, priority?, correlationId?, tags? }` | `RouteDecision` |

```bash
curl -X POST http://localhost:3000/events \
  -H "Content-Type: application/json" \
  -d '{"channel":"http","payload":{"text":"disk usage at 94%"},"priority":"high"}'
# → {"route":"LOCAL_ML","confidence":0.9,"modelId":"mock-model"}
```

### Memory

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `POST` | `/api/memory/remember` | `{ content, tags? }` | `{ id }` |
| `POST` | `/api/memory/recall` | `{ query, limit? }` | `{ results: RankedMemory[] }` |
| `GET` | `/api/memory/:id` | — | `Memory` or 404 |

```bash
curl -X POST http://localhost:3000/api/memory/remember \
  -H "Content-Type: application/json" \
  -d '{"content":"prod-db-01 disk alert resolved","tags":["infra","postgres"]}'

curl -X POST http://localhost:3000/api/memory/recall \
  -H "Content-Type: application/json" \
  -d '{"query":"disk postgres","limit":5}'
```

### Triage

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `POST` | `/api/triage` | `{ text }` | `RouteDecision` |

### Pipelines

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `GET` | `/api/pipelines` | — | `{ pipelines: string[] }` |
| `POST` | `/api/pipelines` | Pipeline JSON | `{ id }` |
| `POST` | `/api/pipelines/:id/run` | `{ event? }` | `PipelineResult` |

### Metrics

| Method | Path | Returns |
|--------|------|---------|
| `GET` | `/api/metrics/health` | `{ status, uptime, timestamp }` |
| `GET` | `/api/metrics/budget` | `{ daily_used, daily_limit, pct_used, ... }` or `{ status: "no_data" }` |

### Watches

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `GET` | `/api/watches` | — | `Watch[]` |
| `POST` | `/api/watches` | Watch config | Created `Watch` |
| `PATCH` | `/api/watches/:id` | Partial Watch | Updated `Watch` |
| `DELETE` | `/api/watches/:id` | — | 204 |

### Plugins

| Method | Path | Returns |
|--------|------|---------|
| `GET` | `/api/plugins` | `{ count, plugins, adapters, channels, stepTypes }` |
| `GET` | `/api/plugins/:name` | Plugin detail or 404 |

## Bridge Selection

Resolved at startup in priority order:

| Condition | Bridge Used |
|-----------|-------------|
| `GRISTMILL_MOCK_BRIDGE=1` | `MockBridge` — in-memory, no daemon |
| `GRISTMILL_SOCK` set | `IpcBridge` → custom socket path |
| Default | `IpcBridge` → `~/.gristmill/gristmill.sock` |
| `@gristmill/core` npm package installed | `NativeBridge` — napi-rs in-process FFI |

## MockBridge Behaviour

Used when `GRISTMILL_MOCK_BRIDGE=1`. Suitable for development and testing without building the Rust FFI.

- `triage()` → always returns `{ route: "LOCAL_ML", confidence: 0.9 }`
- `remember()` → stores in-memory (lost on restart)
- `recall()` → word-level keyword match against in-memory store (including tags)
- `escalate()` → returns `[mock] Response to: <prompt>`
- `subscribe()` → async iterable that waits for `emit()` calls

## Plugin System

Plugins extend GristMill with custom adapters, notification channels, and pipeline step types. Place a `.js` or `.mjs` file in `~/.gristmill/plugins/`:

```typescript
// ~/.gristmill/plugins/my-plugin.mjs
export default class MyPlugin {
  id = "my-plugin";
  name = "My Plugin";
  version = "1.0.0";

  async initialize(ctx) {
    // Register a custom event adapter
    ctx.registerAdapter("github-webhook", (payload) => ({
      channel: "webhook",
      payload,
      tags: { provider: "github" },
    }));

    // Register a notification channel
    ctx.registerChannel("pagerduty", {
      send: async (notification) => { /* ... */ },
    });

    // Register a pipeline step type
    ctx.registerStepType("github-dispatch", {
      execute: async (stepCtx) => ({
        succeeded: true,
        output: { triggered: true },
      }),
    });
  }

  async destroy() {}
}
```

## Bell Tower — Notifications

```typescript
// Create a watch
POST /api/watches
{
  "name": "Pipeline failures",
  "condition": "topic == 'pipeline.failed'",
  "channels": ["slack"],
  "priority": "high"
}
```

Watches are evaluated against every bus event. When the condition matches, notifications are dispatched to the listed channels.

**Quiet hours** (configurable in `config.yaml`): low/normal/high notifications are suppressed; `critical` always fires.

**Digest mode**: batch low-priority notifications into periodic summaries.

Watches are persisted to `~/.gristmill/watches.json` and survive restarts.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GRISTMILL_MOCK_BRIDGE` | — | Set to `1` for MockBridge |
| `GRISTMILL_SOCK` | `~/.gristmill/gristmill.sock` | Daemon socket path |
| `GRISTMILL_CONFIG` | `~/.gristmill/config.yaml` | Config file path |
| `GRISTMILL_PLUGINS_DIR` | `~/.gristmill/plugins/` | Plugin directory |
| `GRISTMILL_WATCHES_FILE` | `~/.gristmill/watches.json` | Watch persistence |
| `PORT` | `3000` | Dashboard port |
| `HOST` | `127.0.0.1` | Dashboard host |
| `ANTHROPIC_API_KEY` | — | Required for LLM escalation |
| `SLACK_WEBHOOK_URL` | — | Slack notifications |

## Build & Test

```bash
pnpm install          # Install dependencies
pnpm lint             # Type check (tsc --noEmit)
pnpm build            # Compile to dist/
pnpm test             # Vitest unit tests
pnpm test:watch       # Watch mode
pnpm start            # Run compiled dist/main.js
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastify` | HTTP API server |
| `@fastify/cors` | CORS support |
| `@fastify/static` | React SPA static file serving |
| `ws` | WebSocket hopper |
| `nodemailer` | Email notifications |
| `yaml` | Config file parsing |
| `zod` | Schema validation |
| `ulid` | Sortable unique IDs (MockBridge) |
| `@msgpack/msgpack` | IpcBridge framing |
| `@gristmill/core` *(optional)* | napi-rs native bridge |
