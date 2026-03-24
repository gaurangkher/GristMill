import { useEffect, useState } from "react";
import { api, type MetricsBudget, type MetricsHealth, type TrainerStatus } from "../api.js";

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{
      background: "var(--surface)",
      border: "1px solid var(--border)",
      borderRadius: "var(--radius)",
      padding: 20,
    }}>
      <div style={{ color: "var(--text-muted)", fontSize: 12, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 12 }}>
        {title}
      </div>
      {children}
    </div>
  );
}

function StatRow({ label, value, badge }: { label: string; value: string; badge?: React.ReactNode }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "6px 0", borderBottom: "1px solid var(--border)" }}>
      <span style={{ color: "var(--text-muted)" }}>{label}</span>
      <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
        {badge}
        <strong>{value}</strong>
      </span>
    </div>
  );
}

function statusBadge(ok: boolean) {
  return <span className={`badge ${ok ? "badge-green" : "badge-red"}`}>{ok ? "OK" : "ERR"}</span>;
}

function trainerStateBadge(state: string) {
  const map: Record<string, string> = {
    IDLE: "badge-gray", TRAINING: "badge-blue", VALIDATING: "badge-blue",
    PROMOTING: "badge-green", ROLLING_BACK: "badge-red", PAUSED: "badge-yellow", WAITING: "badge-yellow",
  };
  return <span className={`badge ${map[state] ?? "badge-gray"}`}>{state}</span>;
}

export default function OverviewPage() {
  const [health, setHealth] = useState<MetricsHealth | null>(null);
  const [budget, setBudget] = useState<MetricsBudget | null>(null);
  const [trainer, setTrainer] = useState<TrainerStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api.metricsHealth().then(setHealth),
      api.metricsBudget().then(setBudget),
      api.trainerStatus().then(setTrainer),
    ]).catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));

    const interval = setInterval(() => {
      api.trainerStatus().then(setTrainer).catch(() => {});
    }, 10_000);
    return () => clearInterval(interval);
  }, []);

  if (error) return <p style={{ color: "var(--red)" }}>Failed to load: {error}</p>;

  return (
    <div style={{ display: "grid", gap: 20, gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))" }}>
      <Card title="Subsystem Health">
        {health ? (
          <>
            <StatRow label="Sieve (triage)" value="" badge={statusBadge(health.sieve_ok)} />
            <StatRow label="Grinders (inference)" value="" badge={statusBadge(health.grinders_ok)} />
            <StatRow label="Hammer (LLM)" value="" badge={statusBadge(health.hammer_ok)} />
            <StatRow label="Ledger (memory)" value="" badge={statusBadge(health.ledger_ok)} />
          </>
        ) : <span style={{ color: "var(--text-muted)" }}>Loading…</span>}
      </Card>

      <Card title="LLM Budget">
        {budget ? (
          <>
            <StatRow label="Tokens used" value={budget.tokens_used.toLocaleString()} />
            <StatRow label="Budget remaining" value={`$${budget.budget_remaining.toFixed(4)}`} />
            <StatRow label="Requests today" value={budget.requests_today.toLocaleString()} />
          </>
        ) : <span style={{ color: "var(--text-muted)" }}>Loading…</span>}
      </Card>

      <Card title="Trainer">
        {trainer ? (
          <>
            <StatRow label="State" value="" badge={trainerStateBadge(trainer.state)} />
            <StatRow label="Active version" value={`v${trainer.current_version}`} />
            <StatRow label="Pending records" value={trainer.buffer_pending_count.toLocaleString()} />
            <StatRow label="Teacher cost" value={`$${trainer.teacher_cost_usd_total.toFixed(4)}`} />
          </>
        ) : <span style={{ color: "var(--text-muted)" }}>Loading…</span>}
      </Card>

      {trainer && Object.keys(trainer.domains).length > 0 && (
        <Card title="Domain Adapters">
          {Object.entries(trainer.domains).map(([domain, info]) => (
            <StatRow
              key={domain}
              label={domain}
              value={`v${info.version} · ${(info.validation_score * 100).toFixed(1)}%`}
            />
          ))}
        </Card>
      )}
    </div>
  );
}
