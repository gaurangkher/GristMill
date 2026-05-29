import { useEffect, useState, useCallback } from "react";
import { api, type TrainerStatus, type CycleSummary, type HealthInfo } from "../api.js";

function fmt(iso: string | null) {
  if (!iso) return "—";
  return new Date(iso).toLocaleString();
}

function stateBadge(state: string) {
  const cls: Record<string, string> = {
    IDLE: "badge-gray", TRAINING: "badge-blue", VALIDATING: "badge-blue",
    PROMOTING: "badge-green", ROLLING_BACK: "badge-red", PAUSED: "badge-yellow", WAITING: "badge-yellow",
  };
  return <span className={`badge ${cls[state] ?? "badge-gray"}`}>{state}</span>;
}

// ── Validation report panel ───────────────────────────────────────────────────

function ValidationReport({ available }: { available: boolean }) {
  const [report, setReport] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  const load = () => {
    setLoading(true);
    setError(null);
    api.trainerValidation()
      .then(setReport)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  };

  const toggle = () => {
    if (!open && !report) load();
    setOpen((v) => !v);
  };

  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: 20 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ color: "var(--text-muted)", fontSize: 12, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em" }}>
          Latest Validation Report
        </div>
        <button
          onClick={toggle}
          disabled={!available}
          style={{ fontSize: 12 }}
        >
          {open ? "Hide" : "Show"}
        </button>
      </div>

      {!available && (
        <p style={{ color: "var(--text-muted)", fontSize: 13, marginTop: 10 }}>
          Trainer offline — start the trainer to view validation reports.
        </p>
      )}

      {available && open && (
        <div style={{ marginTop: 14 }}>
          {loading && <p style={{ color: "var(--text-muted)", fontSize: 13 }}>Loading…</p>}
          {error && (
            <p style={{ color: "var(--red)", fontSize: 13 }}>
              {error.includes("404") ? "No validation results yet — run a training cycle first." : error}
            </p>
          )}
          {report && !loading && (
            <>
              {/* Summary rows for well-known fields */}
              {typeof report.validation_score === "number" && (
                <div style={{ marginBottom: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4 }}>
                    <span style={{ color: "var(--text-muted)" }}>Score</span>
                    <strong style={{ color: report.validation_score >= 0.8 ? "var(--green)" : "var(--red)" }}>
                      {(report.validation_score as number * 100).toFixed(1)}%
                    </strong>
                  </div>
                  <div style={{ height: 8, background: "var(--surface2)", borderRadius: 4, overflow: "hidden" }}>
                    <div style={{
                      width: `${(report.validation_score as number) * 100}%`,
                      height: "100%",
                      background: (report.validation_score as number) >= 0.8 ? "var(--green)" : "var(--red)",
                      transition: "width 0.4s",
                    }} />
                  </div>
                  <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 3 }}>
                    Promotion threshold: 80%
                    {(report.validation_score as number) >= 0.8 ? " ✓ would be promoted" : " ✗ would be rolled back"}
                  </p>
                </div>
              )}
              {typeof report.domain === "string" && (
                <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid var(--border)", fontSize: 13 }}>
                  <span style={{ color: "var(--text-muted)" }}>Domain</span><strong>{report.domain as string}</strong>
                </div>
              )}
              {typeof report.version === "number" && (
                <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid var(--border)", fontSize: 13 }}>
                  <span style={{ color: "var(--text-muted)" }}>Version</span><strong>v{report.version as number}</strong>
                </div>
              )}
              {typeof report.completed_at === "string" && (
                <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid var(--border)", fontSize: 13 }}>
                  <span style={{ color: "var(--text-muted)" }}>Completed</span>
                  <span>{new Date(report.completed_at as string).toLocaleString()}</span>
                </div>
              )}
              {/* Full JSON dump */}
              <div style={{ marginTop: 12 }}>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
                  Full report
                </div>
                <pre style={{
                  background: "var(--surface2)", border: "1px solid var(--border)",
                  borderRadius: "var(--radius)", padding: 12, fontSize: 11,
                  fontFamily: "monospace", overflowX: "auto", margin: 0,
                  color: "var(--text)", maxHeight: 300, overflowY: "auto",
                }}>
                  {JSON.stringify(report, null, 2)}
                </pre>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function TrainerPage() {
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [trainerChecked, setTrainerChecked] = useState(false);
  const [status, setStatus] = useState<TrainerStatus | null>(null);
  const [history, setHistory] = useState<CycleSummary[]>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [rollingBack, setRollingBack] = useState<number | null>(null);

  const load = useCallback(() => {
    api.trainerHealth()
      .then((h) => { setHealth(h); setTrainerChecked(true); })
      .catch(() => { setHealth(null); setTrainerChecked(true); });
    api.trainerStatus().then(setStatus).catch(() => {});
    api.trainerHistory().then(setHistory).catch(() => {});
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 15_000);
    return () => clearInterval(t);
  }, [load]);

  const action = async (fn: () => Promise<unknown>, label: string) => {
    setBusy(true);
    setMsg(null);
    try {
      await fn();
      setMsg(`${label} succeeded`);
      load();
    } catch (e: unknown) {
      setMsg(`Error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  const rollback = async (version: number) => {
    if (!window.confirm(`Roll back to v${version}? The current model will be replaced.`)) return;
    setRollingBack(version);
    setMsg(null);
    try {
      const res = await api.trainerRollback(version);
      setMsg(`Rolled back to v${res.rolled_back_to}`);
      load();
    } catch (e: unknown) {
      setMsg(`Error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setRollingBack(null);
    }
  };

  const trainerAvailable = health?.ok ?? false;
  const currentVersion = status?.current_version ?? -1;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

      {/* Controls */}
      <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: 20 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <span style={{ fontWeight: 600, marginRight: 8 }}>Trainer controls</span>
          {status && stateBadge(status.state)}
          <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
            <button disabled={busy || !trainerAvailable} onClick={() => void action(api.trainerPause, "Pause")}>Pause</button>
            <button disabled={busy || !trainerAvailable} onClick={() => void action(api.trainerResume, "Resume")}>Resume</button>
          </div>
        </div>
        {msg && (
          <p style={{ marginTop: 12, color: msg.startsWith("Error") ? "var(--red)" : "var(--green)", fontSize: 13 }}>{msg}</p>
        )}
        {!trainerChecked && (
          <p style={{ marginTop: 12, color: "var(--text-muted)", fontSize: 13 }}>Checking trainer status…</p>
        )}
        {trainerChecked && !trainerAvailable && (
          <p style={{ marginTop: 12, color: "var(--text-muted)", fontSize: 13 }}>
            Trainer service offline — run{" "}
            <code>docker compose --profile trainer up -d</code>{" "}
            or <code>gristmill-trainer</code> locally to enable.
          </p>
        )}
      </div>

      {/* Status */}
      {status && (
        <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: 20 }}>
          <div style={{ color: "var(--text-muted)", fontSize: 12, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 12 }}>Status</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 24px" }}>
            {[
              ["Current version", `v${status.current_version}`],
              ["Pending records", status.buffer_pending_count.toLocaleString()],
              ["Last cycle", fmt(status.last_cycle_at)],
              ["Teacher cost", `$${status.teacher_cost_usd_total.toFixed(4)}`],
            ].map(([k, v]) => (
              <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid var(--border)" }}>
                <span style={{ color: "var(--text-muted)" }}>{k}</span>
                <strong>{v}</strong>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Distillation pipeline */}
      {status && (
        <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: 20 }}>
          <div style={{ color: "var(--text-muted)", fontSize: 12, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 16 }}>
            Distillation Pipeline
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>

            {/* Commercial LLM */}
            <div style={{ flex: 1, minWidth: 140, background: "var(--bg)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: "12px 16px" }}>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-muted)", marginBottom: 6 }}>
                Commercial LLM
              </div>
              <div style={{ fontWeight: 600, fontSize: 13, wordBreak: "break-all" }}>{status.commercial_llm}</div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>Escalation fallback</div>
            </div>

            <div style={{ color: "var(--text-muted)", fontSize: 18, userSelect: "none" }}>↓</div>

            {/* Teacher */}
            <div style={{ flex: 1, minWidth: 140, background: "var(--bg)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: "12px 16px" }}>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-muted)", marginBottom: 6 }}>
                Teacher
              </div>
              <div style={{ fontWeight: 600, fontSize: 13, wordBreak: "break-all" }}>{status.teacher_model}</div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>Generates training data (Ollama)</div>
            </div>

            <div style={{ color: "var(--accent)", fontSize: 18, userSelect: "none", fontWeight: 700 }}>→</div>

            {/* Student */}
            <div style={{ flex: 1, minWidth: 140, background: "var(--bg)", border: "1px solid var(--accent)", borderRadius: "var(--radius)", padding: "12px 16px" }}>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--accent)", marginBottom: 6 }}>
                Student
              </div>
              <div style={{ fontWeight: 600, fontSize: 13, wordBreak: "break-all" }}>{status.student_model}</div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>Being distilled · v{status.current_version}</div>
            </div>

          </div>
          <div style={{ marginTop: 12, fontSize: 11, color: "var(--text-muted)" }}>
            Teacher responses are logged to the training buffer → Student is fine-tuned via LoRA distillation. Commercial LLM is used only for low-confidence escalation and never enters the training buffer.
          </div>
        </div>
      )}

      {/* Validation report */}
      <ValidationReport available={trainerAvailable} />

      {/* Cycle History */}
      <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: 20 }}>
        <div style={{ color: "var(--text-muted)", fontSize: 12, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 12 }}>
          Cycle History
        </div>
        {history.length === 0 ? (
          <p style={{ color: "var(--text-muted)" }}>No cycles yet.</p>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr style={{ color: "var(--text-muted)", textAlign: "left" }}>
                {["Version", "Domain", "Score", "Records", "Duration", "Cost", "Status", "Completed", ""].map((h) => (
                  <th key={h} style={{ padding: "6px 12px 6px 0", borderBottom: "1px solid var(--border)", fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {history.map((c) => {
                const canRollback = trainerAvailable && !c.rolled_back && !c.error && c.version !== currentVersion;
                return (
                  <tr key={`${c.version}-${c.domain}`} style={{ borderBottom: "1px solid var(--border)" }}>
                    <td style={{ padding: "8px 12px 8px 0" }}>
                      v{c.version}
                      {c.version === currentVersion && (
                        <span style={{ marginLeft: 6, fontSize: 10, color: "var(--accent)", fontWeight: 700 }}>ACTIVE</span>
                      )}
                    </td>
                    <td style={{ padding: "8px 12px 8px 0" }}>{c.domain}</td>
                    <td style={{ padding: "8px 12px 8px 0", color: c.validation_score >= 0.8 ? "var(--green)" : "var(--red)", fontWeight: 600 }}>
                      {(c.validation_score * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: "8px 12px 8px 0" }}>{c.record_count}</td>
                    <td style={{ padding: "8px 12px 8px 0" }}>{c.duration_minutes.toFixed(1)} min</td>
                    <td style={{ padding: "8px 12px 8px 0" }}>${c.teacher_cost_usd.toFixed(4)}</td>
                    <td style={{ padding: "8px 12px 8px 0" }}>
                      {c.rolled_back
                        ? <span className="badge badge-red">Rolled back</span>
                        : c.error
                        ? <span className="badge badge-red">Failed</span>
                        : <span className="badge badge-green">OK</span>}
                    </td>
                    <td style={{ padding: "8px 12px 8px 0", color: "var(--text-muted)" }}>{fmt(c.completed_at)}</td>
                    <td style={{ padding: "8px 0" }}>
                      {canRollback && (
                        <button
                          onClick={() => void rollback(c.version)}
                          disabled={rollingBack !== null}
                          style={{
                            fontSize: 11, padding: "3px 10px",
                            background: "transparent", border: "1px solid var(--border)",
                            color: rollingBack === c.version ? "var(--text-muted)" : "var(--red)",
                          }}
                        >
                          {rollingBack === c.version ? "Rolling back…" : "Rollback"}
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
