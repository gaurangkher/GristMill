# grist-ffi

FFI bridge layer for GristMill. Exposes the Rust core to Python (via PyO3) and Node.js (via napi-rs). This crate is intentionally thin — all business logic stays in the underlying crates.

## Purpose

`grist-ffi` is the only place where cross-language bindings are defined. It delegates every call to `grist-core` and handles only serialization (JSON) and type conversion. No business logic belongs here.

## Cargo Features

| Feature | Builds | Use |
|---------|--------|-----|
| `python` | PyO3 `.whl` wheel | Python shell + training callbacks |
| `node` | napi-rs `.node` binary | TypeScript/Node.js shell |

Both features can be built from the same crate but not simultaneously (different exported symbols).

## Python Bridge (`pyo3_bridge.rs`)

Build the wheel:

```bash
cd gristmill-core/crates/grist-ffi
maturin build --release --features python
pip install target/wheels/gristmill_core-*.whl
```

### Exposed Python Classes

```python
from gristmill_core import GristMillBridge, GristEvent, RouteDecision, Memory

# Create bridge (loads config from path or uses defaults)
bridge = GristMillBridge(config_path="~/.gristmill/config.yaml")

# Triage an event
event = GristEvent(channel="python", payload={"text": "schedule meeting"})
decision: RouteDecision = bridge.triage(event)
print(decision.route, decision.confidence)

# Memory
memory_id: str = bridge.remember("resolved disk alert", tags=["infra"])
results = bridge.recall("disk alert", limit=10)

# Pipelines
bridge.register_pipeline({"id": "content-digest", "steps": [...]})
result = bridge.run_pipeline("content-digest", event)

# Register a Python-side model (callback from Rust inference pool)
bridge.register_python_model("my-model", lambda features: logits)
```

### Python Stub Classes (when wheel not installed)

`gristmill_ml/core.py` provides pure-Python stubs for development without the compiled extension:

```python
from gristmill_ml.core import HAS_NATIVE, PyGristMill

if HAS_NATIVE:
    # Real Rust bridge
else:
    # Stub raises RuntimeError on actual calls
```

## Node.js Bridge (`napi_bridge.rs`)

Build the binary:

```bash
npm install -g @napi-rs/cli

cd gristmill-core/crates/grist-ffi
napi build --release --features node
# Produces: gristmill_core.linux-x64-gnu.node  (or platform equivalent)
```

### Exposed TypeScript/JavaScript Interface

```typescript
// Auto-generated TypeScript types from napi-rs
import { GristMillBridge } from '@gristmill/core';

const bridge = new GristMillBridge(configPath);

// All methods are async (bridge async operations to tokio runtime)
const decision = await bridge.triage(eventJson);
const memoryId = await bridge.remember(content, tags);
const results  = await bridge.recall(query, limit);
const response = await bridge.escalate(prompt, maxTokens);

bridge.registerPipeline(pipelineJson);
const result = await bridge.runPipeline(pipelineId, eventJson);
const ids    = bridge.pipelineIds();

const eventJson = bridge.buildEvent(channel, payloadJson);
const sub = bridge.subscribe(topic);
const next = await sub.nextJson();  // null when ended
```

The TypeScript wrapper in `gristmill-integrations/src/core/bridge.ts` wraps this interface into the `IBridge` type used throughout the TypeScript shell.

## IPC Alternative

If the FFI build is not available (no compiled `.node` or `.whl`), both shells fall back to the **IPC bridge** which communicates with the running `gristmill-daemon` over a Unix socket (`~/.gristmill/gristmill.sock`). This has ~5% more latency but requires no FFI compilation.

## Design Constraints

- **No business logic** in this crate. Every function is `fn foo(args) -> serialize(core.foo(deserialize(args)))`.
- All values cross the boundary as **JSON strings** for transparency and debuggability.
- Async operations use `pyo3-asyncio` (Python) and `napi` async (Node.js) to bridge to the Tokio runtime inside Rust.

## Dependencies

```toml
grist-core       = { path = "../grist-core" }
grist-event      = { path = "../grist-event" }
grist-millwright = { path = "../grist-millwright" }
grist-bus        = { path = "../grist-bus" }
serde_json       = "1"
tokio            = { workspace = true }

# Feature: python
pyo3             = { version = "0.21", features = ["extension-module"] }
pyo3-asyncio     = "0.21"

# Feature: node
napi             = { version = "2", features = ["async"] }
napi-derive      = "2"
```
