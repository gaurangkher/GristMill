import { useEffect, useState, useCallback } from "react";
import { api, type PipelineRun } from "../api.js";

const DEFAULT_PIPELINE = JSON.stringify({
  id: "my-pipeline",
  name: "My Pipeline",
  steps: [
    {
      id: "classify",
      type: { LocalMl: { model_id: "intent-classifier-v1" } },
      prefer_local: true,
      timeout_ms: 5000,
    },
  ],
}, null, 2);

const DEFAULT_PAYLOAD = JSON.stringify({ text: "Hello from the dashboard" }, null, 2);

function fmt(iso: string) {
  return new Date(iso).toLocaleString();
}

function statusBadge(status: string) {
  const cls: Record<string, string> = {
    completed: "badge-green", running: "badge-blue", failed: "badge-red", pending: "badge-gray",
  };
  return <span className={`badge ${cls[status] ?? "badge-gray"}`}>{status}</span>;
}

function SectionHeader({ title }: { title: string }) {
  return (
    <div style={{ color: "var(--text-muted)", fontSize: 12, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 16 }}>
      {title}
    </div>
  );
}

// ── Register panel ────────────────────────────────────────────────────────────

function RegisterPanel({ onRegistered }: { onRegistered: () => void }) {
  const [json, setJson] = useState(DEFAULT_PIPELINE);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);

  const validate = (v: string) => {
    try { JSON.parse(v); setParseError(null); }
    catch (e) { setParseError(e instanceof Error ? e.message : "Invalid JSON"); }
  };

  const register = async () => {
    let parsed: Record<string, unknown>;
    try { parsed = JSON.parse(json) as Record<string, unknown>; }
    catch { return; }
    setBusy(true);
    setMsg(null);
    try {
      const res = await api.pipelineRegister(parsed);
      setMsg({ ok: true, text: `Registered pipeline "${res.id}"` });
      onRegistered();
    } catch (e: unknown) {
      setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: 20 }}>
      <SectionHeader title="Register Pipeline" />
      <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 14 }}>
        Paste a pipeline definition as JSON. The <code>id</code> field is required.
        Registered pipelines persist until the daemon restarts.
      </p>
      <textarea
        value={json}
        onChange={(e) => { setJson(e.target.value); validate(e.target.value); }}
        rows={12}
        spellCheck={false}
        style={{
          width: "100%", boxSizing: "border-box",
          background: "var(--surface2)", border: `1px solid ${parseError ? "var(--red)" : "var(--border)"}`,
          borderRadius: "var(--radius)", padding: "10px 12px",
          color: "var(--text)", fontSize: 13, fontFamily: "monospace",
          resize: "vertical",
        }}
      />
      {parseError && <p style={{ color: "var(--red)", fontSize: 12, marginTop: 4 }}>{parseError}</p>}
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 12 }}>
        <button onClick={() => void register()} disabled={busy || !!parseError}>
          {busy ? "Registering…" : "Register"}
        </button>
        <button
          onClick={() => { setJson(DEFAULT_PIPELINE); setParseError(null); }}
          style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--text-muted)" }}
        >
          Reset example
        </button>
        {msg && (
          <span style={{ fontSize: 13, color: msg.ok ? "var(--green)" : "var(--red)" }}>{msg.text}</span>
        )}
      </div>
    </div>
  );
}

// ── Run panel ─────────────────────────────────────────────────────────────────

function RunPanel({ pipelineIds }: { pipelineIds: string[] }) {
  const [selectedId, setSelectedId] = useState("");
  const [payload, setPayload] = useState(DEFAULT_PAYLOAD);
  const [parseError, setParseError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (pipelineIds.length > 0 && !selectedId) setSelectedId(pipelineIds[0]);
  }, [pipelineIds, selectedId]);

  const validate = (v: string) => {
    try { JSON.parse(v); setParseError(null); }
    catch (e) { setParseError(e instanceof Error ? e.message : "Invalid JSON"); }
  };

  const run = async () => {
    if (!selectedId) return;
    let parsedPayload: unknown = {};
    try { parsedPayload = JSON.parse(payload); }
    catch { return; }
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.pipelineRun(selectedId, parsedPayload);
      setResult(res.result);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: 20 }}>
      <SectionHeader title="Run Pipeline" />

      {pipelineIds.length === 0 ? (
        <p style={{ color: "var(--text-muted)", fontSize: 13 }}>No pipelines registered yet. Register one above first.</p>
      ) : (
        <>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14, flexWrap: "wrap" }}>
            <label style={{ fontSize: 13, color: "var(--text-muted)" }}>
              Pipeline:&nbsp;
              <select
                value={selectedId}
                onChange={(e) => setSelectedId(e.target.value)}
                style={{
                  background: "var(--surface2)", border: "1px solid var(--border)",
                  borderRadius: "var(--radius)", padding: "4px 8px",
                  color: "var(--text)", fontSize: 13,
                }}
              >
                {pipelineIds.map((id) => <option key={id} value={id}>{id}</option>)}
              </select>
            </label>
          </div>

          <label style={{ fontSize: 13, color: "var(--text-muted)", display: "block", marginBottom: 6 }}>
            Event payload (JSON):
          </label>
          <textarea
            value={payload}
            onChange={(e) => { setPayload(e.target.value); validate(e.target.value); }}
            rows={5}
            spellCheck={false}
            style={{
              width: "100%", boxSizing: "border-box",
              background: "var(--surface2)", border: `1px solid ${parseError ? "var(--red)" : "var(--border)"}`,
              borderRadius: "var(--radius)", padding: "10px 12px",
              color: "var(--text)", fontSize: 13, fontFamily: "monospace",
              resize: "vertical",
            }}
          />
          {parseError && <p style={{ color: "var(--red)", fontSize: 12, marginTop: 4 }}>{parseError}</p>}

          <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 12 }}>
            <button onClick={() => void run()} disabled={busy || !!parseError || !selectedId}>
              {busy ? "Running…" : "Run"}
            </button>
            {error && <span style={{ fontSize: 13, color: "var(--red)" }}>{error}</span>}
          </div>

          {result !== null && (
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 12, color: "var(--text-muted)", fontWeight: 600, marginBottom: 6 }}>RESULT</div>
              <pre style={{
                background: "var(--surface2)", border: "1px solid var(--border)",
                borderRadius: "var(--radius)", padding: 12,
                fontSize: 12, fontFamily: "monospace", overflowX: "auto",
                margin: 0, color: "var(--text)",
              }}>
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ── Runs table ────────────────────────────────────────────────────────────────

function RunsTable({ runs, loading, error }: { runs: PipelineRun[]; loading: boolean; error: string | null }) {
  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: 20 }}>
      <SectionHeader title="Recent Runs" />
      {loading && <p style={{ color: "var(--text-muted)" }}>Loading…</p>}
      {error && <p style={{ color: "var(--red)" }}>Error: {error}</p>}
      {!loading && !error && runs.length === 0 && (
        <p style={{ color: "var(--text-muted)", fontSize: 13 }}>No pipeline runs yet.</p>
      )}
      {runs.length > 0 && (
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ color: "var(--text-muted)", textAlign: "left" }}>
              {["ID", "Status", "Progress", "Started"].map((h) => (
                <th key={h} style={{ padding: "6px 16px 6px 0", borderBottom: "1px solid var(--border)", fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id} style={{ borderBottom: "1px solid var(--border)" }}>
                <td style={{ padding: "10px 16px 10px 0", fontFamily: "monospace", fontSize: 12, color: "var(--text-muted)" }}>
                  {r.id.slice(0, 20)}…
                </td>
                <td style={{ padding: "10px 16px 10px 0" }}>{statusBadge(r.status)}</td>
                <td style={{ padding: "10px 16px 10px 0" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{ width: 100, height: 6, background: "var(--surface2)", borderRadius: 3, overflow: "hidden" }}>
                      <div style={{
                        width: `${r.steps_total > 0 ? (r.steps_completed / r.steps_total) * 100 : 0}%`,
                        height: "100%",
                        background: r.status === "failed" ? "var(--red)" : "var(--accent)",
                        transition: "width 0.3s",
                      }} />
                    </div>
                    <span style={{ color: "var(--text-muted)", fontSize: 12 }}>{r.steps_completed}/{r.steps_total}</span>
                  </div>
                </td>
                <td style={{ padding: "10px 0", color: "var(--text-muted)" }}>{fmt(r.started_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function PipelinesPage() {
  const [pipelineIds, setPipelineIds] = useState<string[]>([]);
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [runsLoading, setRunsLoading] = useState(true);

  const loadIds = useCallback(() => {
    api.pipelineIds()
      .then((r) => setPipelineIds(r.pipelines))
      .catch(() => {});
  }, []);

  useEffect(() => {
    loadIds();
    // Try to load runs (bridge may return IDs instead — handled gracefully)
    api.pipelines()
      .then((data) => {
        // bridge returns { pipelines: string[] } shape — not run objects
        if (Array.isArray(data)) setRuns(data);
      })
      .catch((e: unknown) => setRunsError(e instanceof Error ? e.message : String(e)))
      .finally(() => setRunsLoading(false));

    const t = setInterval(() => {
      loadIds();
      api.pipelines().then((d) => { if (Array.isArray(d)) setRuns(d); }).catch(() => {});
    }, 10_000);
    return () => clearInterval(t);
  }, [loadIds]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <RegisterPanel onRegistered={loadIds} />
      <RunPanel pipelineIds={pipelineIds} />
      <RunsTable runs={runs} loading={runsLoading} error={runsError} />
    </div>
  );
}
