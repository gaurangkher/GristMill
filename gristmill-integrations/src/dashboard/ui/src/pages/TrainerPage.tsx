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

export default function TrainerPage() {
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [status, setStatus] = useState<TrainerStatus | null>(null);
  const [history, setHistory] = useState<CycleSummary[]>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const load = useCallback(() => {
    api.trainerHealth().then(setHealth).catch(() => {});
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

  const trainerAvailable = health?.ok ?? false;

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
        {!trainerAvailable && (
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

      {/* History */}
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
                {["Version", "Domain", "Score", "Records", "Duration", "Cost", "Status", "Completed"].map((h) => (
                  <th key={h} style={{ padding: "6px 12px 6px 0", borderBottom: "1px solid var(--border)", fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {history.map((c) => (
                <tr key={`${c.version}-${c.domain}`} style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "8px 12px 8px 0" }}>v{c.version}</td>
                  <td style={{ padding: "8px 12px 8px 0" }}>{c.domain}</td>
                  <td style={{ padding: "8px 12px 8px 0" }}>{(c.validation_score * 100).toFixed(1)}%</td>
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
                  <td style={{ padding: "8px 0", color: "var(--text-muted)" }}>{fmt(c.completed_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
