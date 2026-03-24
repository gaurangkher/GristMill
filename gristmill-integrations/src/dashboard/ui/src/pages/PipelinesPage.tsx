import { useEffect, useState } from "react";
import { api, type PipelineRun } from "../api.js";

function fmt(iso: string) {
  return new Date(iso).toLocaleString();
}

function statusBadge(status: string) {
  const cls: Record<string, string> = {
    completed: "badge-green", running: "badge-blue", failed: "badge-red", pending: "badge-gray",
  };
  return <span className={`badge ${cls[status] ?? "badge-gray"}`}>{status}</span>;
}

export default function PipelinesPage() {
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.pipelines()
      .then(setRuns)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));

    const t = setInterval(() => {
      api.pipelines().then(setRuns).catch(() => {});
    }, 10_000);
    return () => clearInterval(t);
  }, []);

  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: 20 }}>
      <div style={{ color: "var(--text-muted)", fontSize: 12, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 16 }}>
        Pipeline Runs
      </div>

      {loading && <p style={{ color: "var(--text-muted)" }}>Loading…</p>}
      {error && <p style={{ color: "var(--red)" }}>Error: {error}</p>}

      {!loading && !error && runs.length === 0 && (
        <p style={{ color: "var(--text-muted)" }}>No pipeline runs yet.</p>
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
                    <span style={{ color: "var(--text-muted)", fontSize: 12 }}>
                      {r.steps_completed}/{r.steps_total}
                    </span>
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
