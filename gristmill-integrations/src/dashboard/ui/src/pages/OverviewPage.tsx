import { useEffect, useState } from "react";
import { api, type MetricsBudget, type MetricsHealth, type TrainerStatus, type RoutingMetrics } from "../api.js";

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
  const [routing, setRouting] = useState<RoutingMetrics | null>(null);
  const [trainer, setTrainer] = useState<TrainerStatus | null>(null);
  const [trainerChecked, setTrainerChecked] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Core metrics — surface errors if these fail
    Promise.all([
      api.metricsHealth().then(setHealth),
      api.metricsBudget().then(setBudget),
      api.metricsRouting().then(setRouting),
    ]).catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));

    // Trainer is optional — 503 must not block the rest of the page
    api.trainerStatus().then(setTrainer).catch(() => {}).finally(() => setTrainerChecked(true));

    const interval = setInterval(() => {
      api.metricsHealth().then(setHealth).catch(() => {});
      api.metricsBudget().then(setBudget).catch(() => {});
      api.metricsRouting().then(setRouting).catch(() => {});
      api.trainerStatus().then(setTrainer).catch(() => {});
    }, 15_000);
    return () => clearInterval(interval);
  }, []);

  if (error) return <p style={{ color: "var(--red)" }}>Failed to load: {error}</p>;

  const budgetPct = budget ? Math.min(100, budget.pct_used) : 0;
  const budgetBarColor = budgetPct >= 90 ? "var(--red, #ef4444)" : budgetPct >= 70 ? "#f59e0b" : "var(--accent)";

  return (
    <div style={{ display: "grid", gap: 20, gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))" }}>
      <Card title="System Health">
        {health ? (
          <>
            <StatRow label="Status" value={health.status.toUpperCase()} badge={statusBadge(health.status === "ok")} />
            <StatRow label="Uptime" value={`${Math.floor(health.uptime / 60)}m ${Math.floor(health.uptime % 60)}s`} />
            {routing && !routing._stub && (
              <>
                <StatRow label="Cache hit rate" value={`${(routing.sieve.routing_cache.hit_rate * 100).toFixed(1)}%`} />
                <StatRow label="Pipelines" value={routing.pipelines_registered.toLocaleString()} />
              </>
            )}
          </>
        ) : <span style={{ color: "var(--text-muted)" }}>Loading…</span>}
      </Card>

      <Card title="LLM Budget">
        {budget ? (
          <>
            <StatRow label="Daily used" value={`${budget.daily_used.toLocaleString()} tokens`} />
            <StatRow label="Daily limit" value={budget.daily_limit > 0 ? budget.daily_limit.toLocaleString() : "—"} />
            <StatRow label="Used" value={`${budget.pct_used.toFixed(1)}%`} />
            <div style={{ marginTop: 8 }}>
              <div style={{ background: "var(--border)", borderRadius: 4, height: 6, overflow: "hidden" }}>
                <div style={{ width: `${budgetPct}%`, height: "100%", background: budgetBarColor, transition: "width 0.3s ease" }} />
              </div>
            </div>
            {budget.status === "no_data" && (
              <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 6 }}>No LLM calls yet</p>
            )}
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
        ) : trainerChecked ? (
          <span style={{ color: "var(--text-muted)", fontSize: 13 }}>
            Trainer offline — start <code>gristmill-trainer</code> to enable
          </span>
        ) : (
          <span style={{ color: "var(--text-muted)" }}>Loading…</span>
        )}
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
