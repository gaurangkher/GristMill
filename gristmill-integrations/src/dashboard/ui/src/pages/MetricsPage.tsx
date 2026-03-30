import { useEffect, useState } from "react";
import { api, type RoutingMetrics } from "../api.js";

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{
      background: "var(--surface)",
      border: "1px solid var(--border)",
      borderRadius: "var(--radius)",
      padding: 20,
    }}>
      <div style={{
        color: "var(--text-muted)",
        fontSize: 12,
        fontWeight: 600,
        textTransform: "uppercase",
        letterSpacing: "0.08em",
        marginBottom: 12,
      }}>
        {title}
      </div>
      {children}
    </div>
  );
}

function StatRow({ label, value, badge }: { label: string; value: string; badge?: React.ReactNode }) {
  return (
    <div style={{
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      padding: "6px 0",
      borderBottom: "1px solid var(--border)",
    }}>
      <span style={{ color: "var(--text-muted)" }}>{label}</span>
      <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
        {badge}
        <strong>{value}</strong>
      </span>
    </div>
  );
}

function ProgressBar({ pct, color = "var(--accent)" }: { pct: number; color?: string }) {
  const clamped = Math.min(100, Math.max(0, pct));
  return (
    <div style={{
      background: "var(--border)",
      borderRadius: 4,
      height: 6,
      overflow: "hidden",
      marginTop: 4,
    }}>
      <div style={{
        width: `${clamped}%`,
        height: "100%",
        background: clamped >= 90 ? "var(--red, #ef4444)" : clamped >= 70 ? "#f59e0b" : color,
        transition: "width 0.3s ease",
      }} />
    </div>
  );
}

export default function MetricsPage() {
  const [data, setData] = useState<RoutingMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  async function load() {
    try {
      const m = await api.metricsRouting();
      setData(m);
      setLastRefresh(new Date());
      setError(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  useEffect(() => {
    void load();
    const interval = setInterval(() => { void load(); }, 10_000);
    return () => clearInterval(interval);
  }, []);

  if (error) return <p style={{ color: "var(--red)" }}>Failed to load metrics: {error}</p>;
  if (!data) return <p style={{ color: "var(--text-muted)" }}>Loading…</p>;

  const cache = data.sieve.routing_cache;
  const hitRatePct = cache.hit_rate * 100;
  const dailyUsedPct = data.hammer.daily_token_limit > 0
    ? (data.hammer.daily_tokens_used / data.hammer.daily_token_limit) * 100
    : 0;
  const monthlyUsedPct = data.hammer.monthly_token_limit > 0
    ? (data.hammer.monthly_tokens_used / data.hammer.monthly_token_limit) * 100
    : 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {data._stub && (
        <div style={{
          background: "#fef3c7",
          border: "1px solid #f59e0b",
          borderRadius: "var(--radius)",
          padding: "8px 14px",
          fontSize: 12,
          color: "#92400e",
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          alignSelf: "flex-start",
        }}>
          Stub data — daemon not connected
        </div>
      )}

      <div style={{ display: "grid", gap: 20, gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))" }}>
        <Card title="Sieve (Routing Engine)">
          <StatRow
            label="Confidence threshold"
            value={data.sieve.confidence_threshold.toFixed(2)}
          />
          <StatRow
            label="Cache exact hits"
            value={cache.exact_hits.toLocaleString()}
          />
          <StatRow
            label="Cache semantic hits"
            value={cache.semantic_hits.toLocaleString()}
          />
          <StatRow
            label="Cache misses"
            value={cache.misses.toLocaleString()}
          />
          <div style={{ padding: "8px 0 4px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
              <span style={{ color: "var(--text-muted)", fontSize: 13 }}>Hit rate</span>
              <strong>{hitRatePct.toFixed(1)}%</strong>
            </div>
            <ProgressBar pct={hitRatePct} color="#22c55e" />
          </div>
          <StatRow
            label="Feedback records sent"
            value={data.sieve.feedback_records_sent.toLocaleString()}
          />
        </Card>

        <Card title="LLM Gateway (Hammer)">
          <StatRow
            label="Semantic cache size"
            value={data.hammer.cache_size.toLocaleString()}
          />
          <div style={{ padding: "8px 0 4px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
              <span style={{ color: "var(--text-muted)", fontSize: 13 }}>Daily tokens</span>
              <strong>
                {data.hammer.daily_tokens_used.toLocaleString()} / {data.hammer.daily_token_limit.toLocaleString()}
              </strong>
            </div>
            <ProgressBar pct={dailyUsedPct} />
          </div>
          <StatRow
            label="Daily tokens remaining"
            value={data.hammer.daily_tokens_remaining.toLocaleString()}
          />
          <div style={{ padding: "8px 0 4px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
              <span style={{ color: "var(--text-muted)", fontSize: 13 }}>Monthly tokens</span>
              <strong>
                {data.hammer.monthly_tokens_used.toLocaleString()} / {data.hammer.monthly_token_limit.toLocaleString()}
              </strong>
            </div>
            <ProgressBar pct={monthlyUsedPct} />
          </div>
        </Card>

        <Card title="Pipelines">
          <StatRow
            label="Registered pipelines"
            value={data.pipelines_registered.toLocaleString()}
          />
        </Card>
      </div>

      {lastRefresh && (
        <p style={{ color: "var(--text-muted)", fontSize: 11, margin: 0 }}>
          Last refreshed: {lastRefresh.toLocaleTimeString()} · auto-refreshes every 10s
        </p>
      )}
    </div>
  );
}
