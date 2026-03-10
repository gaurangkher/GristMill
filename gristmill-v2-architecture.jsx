import { useState, useEffect } from "react";

const RUST_COLOR = "#ce422b";
const PYTHON_COLOR = "#3776ab";
const TS_COLOR = "#3178c6";
const ACCENT_WARM = "#f0a030";
const ACCENT_GREEN = "#4ade80";
const ACCENT_PINK = "#f472b6";
const BG_DEEP = "#09090f";
const BG_CARD = "#0e0e18";
const BG_CARD_HOVER = "#131320";
const BORDER = "#1a1a2f";
const TEXT_DIM = "#6b7094";
const TEXT_MED = "#9ca0be";
const TEXT_BRIGHT = "#e8e9f0";

const CRATES = [
  {
    id: "grist-event",
    name: "grist-event",
    desc: "Universal GristEvent schema, ULID generation, serialization. Every message in the system passes through this type.",
    lang: "rust",
    layer: "shared",
    deps: [],
  },
  {
    id: "grist-sieve",
    name: "grist-sieve",
    desc: "Triage classifier. ONNX inference (<5ms), feature extraction, cost oracle, confidence scoring, feedback logging.",
    lang: "rust",
    layer: "core",
    deps: ["grist-event"],
    stats: { latency: "<5ms", model: "~50MB ONNX", routes: "4 classes" },
  },
  {
    id: "grist-grinders",
    name: "grist-grinders",
    desc: "Parallel worker pool for local ML inference. ONNX Runtime, llama.cpp (GGUF), TFLite. Zero-copy tensor ops via ndarray.",
    lang: "rust",
    layer: "core",
    deps: ["grist-event"],
    stats: { workers: "CPU-1", runtimes: "ONNX/GGUF/TFLite", batching: "dynamic" },
  },
  {
    id: "grist-millwright",
    name: "grist-millwright",
    desc: "DAG-based orchestrator. Parallel by default. Tokio for async I/O + Rayon for CPU work. Checkpoints, approval gates, retry policies.",
    lang: "rust",
    layer: "core",
    deps: ["grist-event", "grist-sieve", "grist-grinders", "grist-hammer"],
    stats: { concurrency: "8 default", scheduler: "work-stealing", resume: "checkpoint tokens" },
  },
  {
    id: "grist-ledger",
    name: "grist-ledger",
    desc: "Three-tier memory. Hot (sled LRU), Warm (SQLite FTS5 + usearch vectors), Cold (zstd JSONL). Auto-compaction with local summarizer.",
    lang: "rust",
    layer: "core",
    deps: ["grist-event"],
    stats: { hot: "<1ms", warm: "<10ms", cold: "<500ms" },
  },
  {
    id: "grist-hammer",
    name: "grist-hammer",
    desc: "LLM escalation gateway. Token budget manager, semantic cache (35% hit rate), batch aggregator, multi-provider router.",
    lang: "rust",
    layer: "core",
    deps: ["grist-event", "grist-ledger"],
    stats: { cache: "35% hit", providers: "4+", budget: "configurable" },
  },
  {
    id: "grist-bus",
    name: "grist-bus",
    desc: "Internal typed pub/sub event bus. Connects Rust core to TypeScript Bell Tower for reactive notifications.",
    lang: "rust",
    layer: "core",
    deps: [],
  },
  {
    id: "grist-ffi",
    name: "grist-ffi",
    desc: "Foreign function interface. PyO3 bridge for Python, napi-rs bridge for Node.js. Async-compatible with both runtimes.",
    lang: "rust",
    layer: "bridge",
    deps: ["grist-sieve", "grist-grinders", "grist-millwright", "grist-ledger", "grist-hammer", "grist-bus"],
  },
];

const PYTHON_MODULES = [
  { id: "py-core", name: "core.py", desc: "PyO3 bridge re-export. Wraps Rust core for Pythonic access." },
  { id: "py-training", name: "training/", desc: "Sieve trainer, NER trainer, embedder fine-tuning, anomaly detector training." },
  { id: "py-datasets", name: "datasets/", desc: "Feedback log import, synthetic augmentation, data loaders." },
  { id: "py-export", name: "export/", desc: "PyTorch → ONNX conversion, INT8/FP16 quantization, cross-validation." },
  { id: "py-experiments", name: "experiments/", desc: "MLflow/W&B tracking, A/B model comparison." },
  { id: "py-pipelines", name: "pipelines/", desc: "End-to-end retraining, custom model framework." },
];

const TS_MODULES = [
  { id: "ts-bridge", name: "core/bridge.ts", desc: "napi-rs wrapper. All processing delegates to Rust core." },
  { id: "ts-hopper", name: "hopper/", desc: "HTTP, WebSocket, webhook, cron, MQ, and FS watch adapters." },
  { id: "ts-belltower", name: "bell-tower/", desc: "Priority dispatcher, digest batching, quiet hours, watch engine." },
  { id: "ts-channels", name: "channels/", desc: "Slack, Telegram, Discord, email, SMS, push, outbound webhooks." },
  { id: "ts-dashboard", name: "dashboard/", desc: "Fastify API + React SPA. Pipeline status, memory browser, metrics." },
  { id: "ts-plugins", name: "plugins/", desc: "Dynamic plugin loader, registry, SDK for community extensions." },
];

const FLOW_DATA = {
  phases: [
    {
      name: "Event Intake",
      lang: "ts",
      steps: ["HTTP adapter receives webhook", "Normalizes to GristEvent", "Sends to Rust core via napi-rs"],
    },
    {
      name: "Triage",
      lang: "rust",
      steps: ["Sieve extracts features (<2ms)", "ONNX classifier routes (confidence 0.91)", "Cost Oracle approves LOCAL_ML route"],
    },
    {
      name: "Execution",
      lang: "rust",
      steps: ["Millwright builds DAG (6 steps, 4 parallel)", "Grinders run NER + classifier + embedder", "Only 1 step escalates to Hammer (LLM)"],
    },
    {
      name: "Memory",
      lang: "rust",
      steps: ["Ledger stores result in warm tier", "Embedding indexed in usearch", "Compactor scheduled for overnight"],
    },
    {
      name: "Notify",
      lang: "ts",
      steps: ["Bus emits pipeline.completed", "Bell Tower evaluates 3 watches", "Dispatches Telegram alert (critical)"],
    },
    {
      name: "Learn",
      lang: "python",
      steps: ["Feedback log appended", "Weekly: retrain Sieve on new data", "Export ONNX → hot-reload in Rust"],
    },
  ],
};

function LangBadge({ lang, size = "sm" }) {
  const colors = { rust: RUST_COLOR, python: PYTHON_COLOR, typescript: TS_COLOR, ts: TS_COLOR };
  const labels = { rust: "Rust", python: "Python", typescript: "TypeScript", ts: "TypeScript" };
  const sz = size === "sm" ? { font: "9px", pad: "2px 7px" } : { font: "10px", pad: "3px 10px" };
  return (
    <span
      style={{
        background: (colors[lang] || "#555") + "20",
        color: colors[lang] || "#aaa",
        fontSize: sz.font,
        fontWeight: 700,
        padding: sz.pad,
        borderRadius: "4px",
        fontFamily: "'IBM Plex Mono', monospace",
        letterSpacing: "0.06em",
        textTransform: "uppercase",
        border: `1px solid ${(colors[lang] || "#555")}33`,
      }}
    >
      {labels[lang] || lang}
    </span>
  );
}

function CrateCard({ crate: c, isActive, onClick }) {
  const layerColors = {
    shared: "#888",
    core: RUST_COLOR,
    bridge: ACCENT_WARM,
  };
  return (
    <div
      onClick={onClick}
      style={{
        background: isActive ? BG_CARD_HOVER : BG_CARD,
        border: `1px solid ${isActive ? (layerColors[c.layer] || BORDER) + "66" : BORDER}`,
        borderRadius: "10px",
        padding: "14px 16px",
        cursor: "pointer",
        transition: "all 0.2s ease",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: isActive ? "10px" : 0 }}>
        <span
          style={{
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: "13px",
            fontWeight: 600,
            color: TEXT_BRIGHT,
          }}
        >
          {c.name}
        </span>
        <LangBadge lang={c.lang} />
        {c.layer === "bridge" && (
          <span
            style={{
              fontSize: "8px",
              color: ACCENT_WARM,
              fontWeight: 700,
              fontFamily: "'IBM Plex Mono', monospace",
              letterSpacing: "0.1em",
            }}
          >
            FFI
          </span>
        )}
        <span
          style={{
            marginLeft: "auto",
            color: TEXT_DIM,
            fontSize: "14px",
            transition: "transform 0.2s",
            transform: isActive ? "rotate(90deg)" : "none",
          }}
        >
          ›
        </span>
      </div>
      {isActive && (
        <div style={{ animation: "fadeSlide 0.25s ease-out" }}>
          <p style={{ fontSize: "12px", color: TEXT_MED, lineHeight: 1.6, margin: "0 0 10px 0", fontFamily: "'IBM Plex Sans', sans-serif" }}>
            {c.desc}
          </p>
          {c.stats && (
            <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
              {Object.entries(c.stats).map(([k, v]) => (
                <div
                  key={k}
                  style={{
                    background: RUST_COLOR + "10",
                    border: `1px solid ${RUST_COLOR}22`,
                    borderRadius: "6px",
                    padding: "4px 8px",
                  }}
                >
                  <span style={{ fontSize: "10px", color: TEXT_DIM, fontFamily: "'IBM Plex Mono', monospace" }}>{k}: </span>
                  <span style={{ fontSize: "10px", color: RUST_COLOR, fontWeight: 600, fontFamily: "'IBM Plex Mono', monospace" }}>{v}</span>
                </div>
              ))}
            </div>
          )}
          {c.deps.length > 0 && (
            <div style={{ marginTop: "8px", display: "flex", alignItems: "center", gap: "4px", flexWrap: "wrap" }}>
              <span style={{ fontSize: "9px", color: TEXT_DIM, fontFamily: "'IBM Plex Mono', monospace" }}>deps:</span>
              {c.deps.map((d) => (
                <span key={d} style={{ fontSize: "9px", color: TEXT_DIM, fontFamily: "'IBM Plex Mono', monospace", opacity: 0.7 }}>
                  {d}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ModuleRow({ mod, color }) {
  return (
    <div style={{ display: "flex", gap: "10px", padding: "8px 0", borderBottom: `1px solid ${BORDER}` }}>
      <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "12px", color, minWidth: "130px", fontWeight: 600 }}>
        {mod.name}
      </span>
      <span style={{ fontFamily: "'IBM Plex Sans', sans-serif", fontSize: "12px", color: TEXT_MED, lineHeight: 1.5 }}>
        {mod.desc}
      </span>
    </div>
  );
}

function FlowTimeline() {
  const [activePhase, setActivePhase] = useState(null);
  const langColors = { rust: RUST_COLOR, ts: TS_COLOR, python: PYTHON_COLOR };

  return (
    <div style={{ position: "relative" }}>
      {/* Vertical line */}
      <div
        style={{
          position: "absolute",
          left: "15px",
          top: "0",
          bottom: "0",
          width: "2px",
          background: `linear-gradient(to bottom, ${RUST_COLOR}44, ${TS_COLOR}44, ${PYTHON_COLOR}44)`,
        }}
      />
      {FLOW_DATA.phases.map((phase, i) => (
        <div
          key={i}
          onClick={() => setActivePhase(activePhase === i ? null : i)}
          style={{ position: "relative", paddingLeft: "40px", marginBottom: "16px", cursor: "pointer" }}
        >
          {/* Dot */}
          <div
            style={{
              position: "absolute",
              left: "8px",
              top: "4px",
              width: "16px",
              height: "16px",
              borderRadius: "50%",
              background: langColors[phase.lang],
              border: `2px solid ${BG_DEEP}`,
              boxShadow: `0 0 8px ${langColors[phase.lang]}44`,
            }}
          />
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
            <span
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: "12px",
                fontWeight: 700,
                color: langColors[phase.lang],
                letterSpacing: "0.04em",
              }}
            >
              {phase.name}
            </span>
            <LangBadge lang={phase.lang} />
          </div>
          {activePhase === i && (
            <div style={{ animation: "fadeSlide 0.2s ease-out" }}>
              {phase.steps.map((step, si) => (
                <div
                  key={si}
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: "8px",
                    padding: "4px 0",
                  }}
                >
                  <span style={{ color: TEXT_DIM, fontSize: "11px", fontFamily: "'IBM Plex Mono', monospace", minWidth: "14px" }}>
                    {si + 1}.
                  </span>
                  <span style={{ fontSize: "12px", color: TEXT_MED, fontFamily: "'IBM Plex Sans', sans-serif", lineHeight: 1.5 }}>
                    {step}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function GristMillV2() {
  const [tab, setTab] = useState("overview");
  const [activeCrate, setActiveCrate] = useState("grist-sieve");

  return (
    <div
      style={{
        minHeight: "100vh",
        background: BG_DEEP,
        color: TEXT_BRIGHT,
        fontFamily: "'IBM Plex Sans', sans-serif",
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
        @keyframes fadeSlide { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 3px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #ffffff15; border-radius: 3px; }
      `}</style>

      <div style={{ maxWidth: "700px", margin: "0 auto", padding: "36px 20px" }}>
        {/* Header */}
        <div style={{ marginBottom: "36px" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: "10px", marginBottom: "6px" }}>
            <h1
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: "32px",
                fontWeight: 700,
                letterSpacing: "-0.03em",
                color: TEXT_BRIGHT,
              }}
            >
              GristMill
            </h1>
            <span
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: "12px",
                color: ACCENT_WARM,
                fontWeight: 600,
                padding: "2px 8px",
                border: `1px solid ${ACCENT_WARM}33`,
                borderRadius: "4px",
              }}
            >
              v2
            </span>
          </div>
          <p style={{ fontSize: "14px", color: TEXT_DIM, fontStyle: "italic", marginBottom: "16px" }}>
            Rust grinds. Python trains. TypeScript connects.
          </p>

          {/* Language breakdown */}
          <div style={{ display: "flex", gap: "2px", borderRadius: "6px", overflow: "hidden", height: "28px", marginBottom: "8px" }}>
            {[
              { lang: "Rust Core", pct: 65, color: RUST_COLOR },
              { lang: "TypeScript", pct: 25, color: TS_COLOR },
              { lang: "Python ML", pct: 10, color: PYTHON_COLOR },
            ].map((s) => (
              <div
                key={s.lang}
                style={{
                  flex: s.pct,
                  background: s.color + "33",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "10px",
                  fontWeight: 600,
                  color: s.color,
                  fontFamily: "'IBM Plex Mono', monospace",
                  borderRight: `1px solid ${BG_DEEP}`,
                }}
              >
                {s.lang} {s.pct}%
              </div>
            ))}
          </div>
          <div style={{ display: "flex", gap: "16px" }}>
            {[
              { label: "Sieve + Grinders + DAG + Memory", color: RUST_COLOR },
              { label: "Adapters + Notifications + Dashboard", color: TS_COLOR },
              { label: "Training + Export + Experiments", color: PYTHON_COLOR },
            ].map((s) => (
              <div key={s.label} style={{ display: "flex", alignItems: "center", gap: "5px" }}>
                <div style={{ width: "8px", height: "8px", borderRadius: "2px", background: s.color }} />
                <span style={{ fontSize: "9px", color: TEXT_DIM, fontFamily: "'IBM Plex Mono', monospace" }}>{s.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Tabs */}
        <div
          style={{
            display: "flex",
            gap: "2px",
            background: "#ffffff06",
            borderRadius: "8px",
            padding: "3px",
            marginBottom: "24px",
          }}
        >
          {[
            { id: "overview", label: "Architecture" },
            { id: "rust", label: "Rust Crates" },
            { id: "shells", label: "Python & TS" },
            { id: "flow", label: "Data Flow" },
          ].map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                flex: 1,
                padding: "9px 4px",
                border: "none",
                borderRadius: "6px",
                background: tab === t.id ? "#ffffff0e" : "transparent",
                color: tab === t.id ? TEXT_BRIGHT : TEXT_DIM,
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: "11px",
                fontWeight: 600,
                cursor: "pointer",
                transition: "all 0.2s",
                letterSpacing: "0.03em",
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Overview Tab */}
        {tab === "overview" && (
          <div style={{ animation: "fadeSlide 0.3s ease-out" }}>
            {/* ASCII-style architecture diagram */}
            <div
              style={{
                background: BG_CARD,
                border: `1px solid ${BORDER}`,
                borderRadius: "12px",
                padding: "24px",
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: "11px",
                lineHeight: 1.8,
                overflowX: "auto",
                marginBottom: "20px",
              }}
            >
              <div style={{ color: TEXT_DIM, marginBottom: "12px", fontSize: "9px", letterSpacing: "0.15em", textTransform: "uppercase" }}>
                System Architecture — Inter-Process Communication
              </div>
              <pre style={{ color: TEXT_MED, margin: 0, whiteSpace: "pre", fontSize: "10.5px" }}>
{`┌─────────────────────────────────────────────────────┐
│           TYPESCRIPT SHELL  (Node.js)               │
│                                                     │
│  Hopper       Bell Tower     Dashboard    Plugins   │
│  (adapters)   (notify)       (web UI)     (npm)     │
└───────────────────────┬─────────────────────────────┘
                        │  napi-rs (zero-copy FFI)
                        ▼
┌─────────────────────────────────────────────────────┐
│              RUST CORE  (single binary)             │
│                                                     │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌───────┐ │
│  │  Sieve  │ │ Grinders │ │Millwright │ │Hammer │ │
│  │ (triage)│ │ (ML pool)│ │ (DAG exec)│ │ (LLM) │ │
│  └────┬────┘ └────┬─────┘ └─────┬─────┘ └───┬───┘ │
│       └───────────┴─────────────┴────────────┘     │
│                    │          │                     │
│          ┌─────────┘    ┌────┘                     │
│          ▼              ▼                           │
│  ┌───────────┐  ┌────────────┐  ┌──────────────┐  │
│  │  Ledger   │  │  Event Bus │  │    Config     │  │
│  │ (3 tiers) │  │  (pub/sub) │  │  (YAML+env)  │  │
│  └───────────┘  └────────────┘  └──────────────┘  │
└───────────────────────┬─────────────────────────────┘
                        │  PyO3 (async FFI)
                        ▼
┌─────────────────────────────────────────────────────┐
│             PYTHON SHELL  (pip package)             │
│                                                     │
│  Training    Datasets    Export    Experiments       │
│  (PyTorch)   (pandas)    (ONNX)   (MLflow)         │
└─────────────────────────────────────────────────────┘`}
              </pre>
            </div>

            {/* Key principles */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
              {[
                {
                  title: "Rust owns all hot paths",
                  desc: "Every event triage, ML inference, DAG decision, and memory lookup runs in Rust. No GC pauses, zero-copy tensors, deterministic latency.",
                  color: RUST_COLOR,
                },
                {
                  title: "Python trains, Rust infers",
                  desc: "Python fine-tunes models on feedback data and exports ONNX. Rust hot-reloads the new model without restart. Closed learning loop.",
                  color: PYTHON_COLOR,
                },
                {
                  title: "TypeScript is the nervous system",
                  desc: "All external I/O flows through TypeScript adapters. Channel SDKs, webhook handling, notification dispatch, web dashboard.",
                  color: TS_COLOR,
                },
                {
                  title: "FFI is the contract layer",
                  desc: "napi-rs and PyO3 bridges are the only cross-language surface. Typed, versioned, zero-copy where possible. IPC mode available.",
                  color: ACCENT_WARM,
                },
              ].map((p) => (
                <div
                  key={p.title}
                  style={{
                    background: BG_CARD,
                    border: `1px solid ${BORDER}`,
                    borderRadius: "10px",
                    padding: "16px",
                    borderTop: `2px solid ${p.color}44`,
                  }}
                >
                  <div
                    style={{
                      fontFamily: "'IBM Plex Mono', monospace",
                      fontSize: "12px",
                      fontWeight: 700,
                      color: p.color,
                      marginBottom: "6px",
                    }}
                  >
                    {p.title}
                  </div>
                  <p style={{ fontSize: "11px", color: TEXT_DIM, lineHeight: 1.6 }}>{p.desc}</p>
                </div>
              ))}
            </div>

            {/* Distribution */}
            <div
              style={{
                marginTop: "20px",
                background: BG_CARD,
                border: `1px solid ${BORDER}`,
                borderRadius: "10px",
                padding: "16px",
              }}
            >
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "10px", color: TEXT_DIM, letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: "10px" }}>
                Distribution
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "8px" }}>
                {[
                  { label: "CLI Binary", value: "~15MB static", sub: "cargo build --release", color: RUST_COLOR },
                  { label: "npm Package", value: "@gristmill/core", sub: "napi prebuilt .node", color: TS_COLOR },
                  { label: "pip Wheel", value: "gristmill_core", sub: "maturin build", color: PYTHON_COLOR },
                ].map((d) => (
                  <div key={d.label} style={{ textAlign: "center", padding: "10px", background: "#ffffff04", borderRadius: "8px" }}>
                    <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "13px", fontWeight: 700, color: d.color }}>{d.value}</div>
                    <div style={{ fontSize: "10px", color: TEXT_DIM, marginTop: "2px" }}>{d.label}</div>
                    <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "9px", color: TEXT_DIM, marginTop: "4px", opacity: 0.6 }}>{d.sub}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Rust Crates Tab */}
        {tab === "rust" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "8px", animation: "fadeSlide 0.3s ease-out" }}>
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "10px", color: TEXT_DIM, letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: "4px" }}>
              Workspace Crates — gristmill-core/crates/
            </div>
            {CRATES.map((c) => (
              <CrateCard
                key={c.id}
                crate={c}
                isActive={activeCrate === c.id}
                onClick={() => setActiveCrate(activeCrate === c.id ? null : c.id)}
              />
            ))}
            {/* Key deps */}
            <div
              style={{
                marginTop: "12px",
                background: BG_CARD,
                border: `1px solid ${BORDER}`,
                borderRadius: "10px",
                padding: "16px",
              }}
            >
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "10px", color: TEXT_DIM, letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: "10px" }}>
                Key Dependencies
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                {[
                  "tokio", "rayon", "ort (ONNX)", "llama-cpp-2", "ndarray",
                  "usearch", "rusqlite+FTS5", "sled", "zstd", "pyo3", "napi-rs",
                  "serde", "tracing", "clap", "dashmap",
                ].map((dep) => (
                  <span
                    key={dep}
                    style={{
                      background: RUST_COLOR + "10",
                      border: `1px solid ${RUST_COLOR}22`,
                      color: RUST_COLOR,
                      fontSize: "10px",
                      fontWeight: 500,
                      padding: "3px 8px",
                      borderRadius: "4px",
                      fontFamily: "'IBM Plex Mono', monospace",
                    }}
                  >
                    {dep}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Python & TypeScript Tab */}
        {tab === "shells" && (
          <div style={{ animation: "fadeSlide 0.3s ease-out" }}>
            {/* Python */}
            <div
              style={{
                background: BG_CARD,
                border: `1px solid ${BORDER}`,
                borderRadius: "12px",
                padding: "20px",
                marginBottom: "16px",
                borderLeft: `3px solid ${PYTHON_COLOR}`,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "12px" }}>
                <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "14px", fontWeight: 700, color: TEXT_BRIGHT }}>
                  gristmill-ml
                </span>
                <LangBadge lang="python" size="md" />
                <span style={{ fontSize: "10px", color: TEXT_DIM, marginLeft: "auto", fontFamily: "'IBM Plex Mono', monospace" }}>
                  pip install gristmill-ml
                </span>
              </div>
              <p style={{ fontSize: "12px", color: TEXT_DIM, marginBottom: "12px", lineHeight: 1.6 }}>
                Owns the model lifecycle: training, fine-tuning, experiment tracking, and ONNX export.
                Does NOT run production inference — that's Rust's job.
              </p>
              {PYTHON_MODULES.map((m) => (
                <ModuleRow key={m.id} mod={m} color={PYTHON_COLOR} />
              ))}
              <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginTop: "12px" }}>
                {["PyTorch", "HuggingFace", "scikit-learn", "pandas", "MLflow", "onnxruntime-tools", "optimum"].map((dep) => (
                  <span
                    key={dep}
                    style={{
                      background: PYTHON_COLOR + "10",
                      border: `1px solid ${PYTHON_COLOR}22`,
                      color: PYTHON_COLOR,
                      fontSize: "10px",
                      padding: "2px 7px",
                      borderRadius: "4px",
                      fontFamily: "'IBM Plex Mono', monospace",
                    }}
                  >
                    {dep}
                  </span>
                ))}
              </div>
            </div>

            {/* TypeScript */}
            <div
              style={{
                background: BG_CARD,
                border: `1px solid ${BORDER}`,
                borderRadius: "12px",
                padding: "20px",
                borderLeft: `3px solid ${TS_COLOR}`,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "12px" }}>
                <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "14px", fontWeight: 700, color: TEXT_BRIGHT }}>
                  gristmill-integrations
                </span>
                <LangBadge lang="typescript" size="md" />
                <span style={{ fontSize: "10px", color: TEXT_DIM, marginLeft: "auto", fontFamily: "'IBM Plex Mono', monospace" }}>
                  pnpm add @gristmill/integrations
                </span>
              </div>
              <p style={{ fontSize: "12px", color: TEXT_DIM, marginBottom: "12px", lineHeight: 1.6 }}>
                Owns external communication: channel adapters, notifications, web dashboard, REST API, and the community plugin system.
              </p>
              {TS_MODULES.map((m) => (
                <ModuleRow key={m.id} mod={m} color={TS_COLOR} />
              ))}
              <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginTop: "12px" }}>
                {["Fastify", "Slack SDK", "Telegram Bot API", "Nodemailer", "chokidar", "Handlebars", "React (dashboard)"].map((dep) => (
                  <span
                    key={dep}
                    style={{
                      background: TS_COLOR + "10",
                      border: `1px solid ${TS_COLOR}22`,
                      color: TS_COLOR,
                      fontSize: "10px",
                      padding: "2px 7px",
                      borderRadius: "4px",
                      fontFamily: "'IBM Plex Mono', monospace",
                    }}
                  >
                    {dep}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Data Flow Tab */}
        {tab === "flow" && (
          <div style={{ animation: "fadeSlide 0.3s ease-out" }}>
            <div
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: "10px",
                color: TEXT_DIM,
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                marginBottom: "16px",
              }}
            >
              Event Lifecycle — Click each phase for details
            </div>
            <div
              style={{
                background: BG_CARD,
                border: `1px solid ${BORDER}`,
                borderRadius: "12px",
                padding: "24px",
              }}
            >
              <FlowTimeline />
            </div>

            {/* The retraining loop callout */}
            <div
              style={{
                marginTop: "20px",
                background: `linear-gradient(135deg, ${RUST_COLOR}08, ${PYTHON_COLOR}08, ${TS_COLOR}08)`,
                border: `1px solid ${BORDER}`,
                borderRadius: "12px",
                padding: "20px",
              }}
            >
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "11px", fontWeight: 700, color: ACCENT_WARM, marginBottom: "8px" }}>
                THE CLOSED LOOP
              </div>
              <p style={{ fontSize: "12px", color: TEXT_MED, lineHeight: 1.7 }}>
                Every Sieve routing decision is logged with its confidence score and outcome.
                Weekly, Python ingests these feedback logs, fine-tunes the classifier, quantizes to INT8, and exports a new ONNX model.
                The Rust daemon hot-reloads it without restart. Over time, fewer and fewer tasks need the LLM — the system literally trains itself to be cheaper.
              </p>
              <div style={{ display: "flex", gap: "6px", marginTop: "12px", alignItems: "center" }}>
                <LangBadge lang="rust" />
                <span style={{ color: TEXT_DIM, fontSize: "12px" }}>→</span>
                <span style={{ fontSize: "10px", color: TEXT_DIM, fontFamily: "'IBM Plex Mono', monospace" }}>feedback.jsonl</span>
                <span style={{ color: TEXT_DIM, fontSize: "12px" }}>→</span>
                <LangBadge lang="python" />
                <span style={{ color: TEXT_DIM, fontSize: "12px" }}>→</span>
                <span style={{ fontSize: "10px", color: TEXT_DIM, fontFamily: "'IBM Plex Mono', monospace" }}>sieve-v{"{n}"}.onnx</span>
                <span style={{ color: TEXT_DIM, fontSize: "12px" }}>→</span>
                <LangBadge lang="rust" />
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div style={{ textAlign: "center", marginTop: "40px", paddingTop: "20px", borderTop: `1px solid ${BORDER}` }}>
          <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "9px", color: TEXT_DIM, letterSpacing: "0.15em" }}>
            GRISTMILL v2 — RUST GRINDS · PYTHON TRAINS · TYPESCRIPT CONNECTS
          </span>
        </div>
      </div>
    </div>
  );
}
