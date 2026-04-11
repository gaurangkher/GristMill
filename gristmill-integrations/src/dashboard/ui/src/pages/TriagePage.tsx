import { useState } from "react";
import { api, type RouteDecision } from "../api.js";

const CHANNELS = ["cli", "http", "slack", "webhook", "websocket", "cron", "internal"];

const ROUTE_META: Record<RouteDecision["route"], { label: string; color: string; desc: string }> = {
  LOCAL_ML:   { label: "Local ML",    color: "var(--green)",  desc: "Handled by local ONNX/GGUF model — no LLM call" },
  RULES:      { label: "Rules",       color: "var(--accent)", desc: "Matched a deterministic rule — instant, no model" },
  HYBRID:     { label: "Hybrid",      color: "var(--yellow)", desc: "Local pre-processing + LLM refinement" },
  LLM_NEEDED: { label: "LLM Needed",  color: "var(--red)",    desc: "Confidence too low — escalating to LLM gateway" },
};

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = value >= 0.85 ? "var(--green)" : value >= 0.5 ? "var(--yellow)" : "var(--red)";
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--text-muted)", marginBottom: 4 }}>
        <span>Confidence</span>
        <span style={{ color, fontWeight: 600 }}>{pct}%</span>
      </div>
      <div style={{ height: 8, background: "var(--surface2)", borderRadius: 4, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, transition: "width 0.4s" }} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-muted)", marginTop: 3 }}>
        <span>threshold: 0.85</span>
        <span>{value >= 0.85 ? "✓ above threshold" : "✗ below threshold — escalates"}</span>
      </div>
    </div>
  );
}

function ResultRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "6px 0", borderBottom: "1px solid var(--border)" }}>
      <span style={{ color: "var(--text-muted)", fontSize: 13 }}>{label}</span>
      <span style={{ fontSize: 13 }}>{value}</span>
    </div>
  );
}

const EXAMPLE_INPUTS = [
  "Schedule a meeting with Alice tomorrow at 3pm",
  "What is the capital of France?",
  "Review this pull request and check for security vulnerabilities",
  "Send a Slack message to #ops saying the deployment succeeded",
  "Summarise the last 10 support tickets and identify common themes",
];

export default function TriagePage() {
  const [text, setText] = useState("");
  const [channel, setChannel] = useState("cli");
  const [result, setResult] = useState<RouteDecision | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    const t0 = performance.now();
    try {
      const decision = await api.triage(text.trim(), channel);
      setLatencyMs(Math.round(performance.now() - t0));
      setResult(decision);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const meta = result ? ROUTE_META[result.route] : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20, maxWidth: 720 }}>

      {/* Input panel */}
      <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: 20 }}>
        <div style={{ color: "var(--text-muted)", fontSize: 12, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 16 }}>
          Triage Playground
        </div>
        <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 16 }}>
          Run any text through the Sieve classifier to see how GristMill would route it.
          Useful for testing routing logic and validating model updates.
        </p>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void run(); }}
          placeholder="Enter text to classify…"
          rows={4}
          style={{
            width: "100%", boxSizing: "border-box",
            background: "var(--surface2)", border: "1px solid var(--border)",
            borderRadius: "var(--radius)", padding: "10px 12px",
            color: "var(--text)", fontSize: 14, resize: "vertical",
            fontFamily: "inherit",
          }}
        />

        <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 12, flexWrap: "wrap" }}>
          <label style={{ fontSize: 13, color: "var(--text-muted)" }}>
            Channel:&nbsp;
            <select
              value={channel}
              onChange={(e) => setChannel(e.target.value)}
              style={{
                background: "var(--surface2)", border: "1px solid var(--border)",
                borderRadius: "var(--radius)", padding: "4px 8px",
                color: "var(--text)", fontSize: 13,
              }}
            >
              {CHANNELS.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>

          <button
            onClick={() => void run()}
            disabled={loading || !text.trim()}
            style={{ marginLeft: "auto" }}
          >
            {loading ? "Classifying…" : "Triage  ⌘↵"}
          </button>
        </div>

        {/* Example prompts */}
        <div style={{ marginTop: 14, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Examples: </span>
          {EXAMPLE_INPUTS.map((ex) => (
            <button
              key={ex}
              onClick={() => setText(ex)}
              style={{
                background: "transparent", border: "1px solid var(--border)",
                borderRadius: "var(--radius)", padding: "2px 8px",
                fontSize: 12, color: "var(--text-muted)", marginLeft: 6, marginTop: 4,
                cursor: "pointer",
              }}
            >
              {ex.length > 40 ? ex.slice(0, 40) + "…" : ex}
            </button>
          ))}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{ background: "var(--surface)", border: "1px solid var(--red)", borderRadius: "var(--radius)", padding: 16, color: "var(--red)", fontSize: 13 }}>
          {error}
        </div>
      )}

      {/* Result */}
      {result && meta && (
        <div style={{ background: "var(--surface)", border: `2px solid ${meta.color}`, borderRadius: "var(--radius)", padding: 20 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
            <span style={{
              background: meta.color, color: "#fff",
              borderRadius: "var(--radius)", padding: "4px 14px",
              fontWeight: 700, fontSize: 15, letterSpacing: "0.02em",
            }}>
              {meta.label}
            </span>
            <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{meta.desc}</span>
            {latencyMs !== null && (
              <span style={{ marginLeft: "auto", fontSize: 12, color: "var(--text-muted)", fontFamily: "monospace" }}>
                {latencyMs} ms (round-trip)
              </span>
            )}
          </div>

          <ConfidenceBar value={result.confidence} />

          <div style={{ marginTop: 16 }}>
            {result.modelId && (
              <ResultRow label="Model" value={<code style={{ fontSize: 12 }}>{result.modelId}</code>} />
            )}
            {result.reason && (
              <ResultRow label="Reason" value={<span style={{ maxWidth: 400, textAlign: "right" }}>{result.reason}</span>} />
            )}
            {result.estimatedTokens != null && result.estimatedTokens > 0 && (
              <ResultRow label="Est. tokens" value={result.estimatedTokens.toLocaleString()} />
            )}
            <ResultRow label="Route" value={<code style={{ fontSize: 12 }}>{result.route}</code>} />
          </div>
        </div>
      )}

      {/* Legend */}
      <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: 16 }}>
        <div style={{ color: "var(--text-muted)", fontSize: 12, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 12 }}>
          Route Labels
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 24px" }}>
          {(Object.entries(ROUTE_META) as [RouteDecision["route"], typeof ROUTE_META[RouteDecision["route"]]][]).map(([key, m]) => (
            <div key={key} style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
              <span style={{
                background: m.color, color: "#fff", borderRadius: 4,
                padding: "1px 8px", fontSize: 11, fontWeight: 700, whiteSpace: "nowrap", marginTop: 1,
              }}>
                {m.label}
              </span>
              <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{m.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
