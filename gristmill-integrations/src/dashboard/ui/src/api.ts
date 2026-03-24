/** Typed API client for the GristMill dashboard backend. */

const BASE = "";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({})) as { error?: string; detail?: string };
    throw new Error(err.detail ?? err.error ?? `${path} → ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface HealthInfo {
  ok: boolean;
  uptime_seconds: number;
  last_heartbeat_seen: string | null;
}

export interface TrainerStatus {
  state: string;
  current_version: number;
  last_cycle_at: string | null;
  buffer_pending_count: number;
  teacher_cost_usd_total: number;
  domains: Record<string, { version: number; validation_score: number; promoted_at: string }>;
}

export interface CycleSummary {
  version: number;
  started_at: string;
  completed_at: string;
  record_count: number;
  duration_minutes: number;
  validation_score: number;
  rolled_back: boolean;
  domain: string;
  teacher_cost_usd: number;
  error: string | null;
}

export interface MetricsBudget {
  budget_remaining: number;
  tokens_used: number;
  requests_today: number;
}

export interface MetricsHealth {
  sieve_ok: boolean;
  grinders_ok: boolean;
  hammer_ok: boolean;
  ledger_ok: boolean;
}

export interface PipelineRun {
  id: string;
  started_at: string;
  status: string;
  steps_completed: number;
  steps_total: number;
}

export interface EcosystemStatus {
  community: { enabled: boolean; endpoint: string };
  federated: {
    enabled: boolean;
    privacy_budget: {
      epsilon_used: number;
      epsilon_budget: number;
      remaining: number;
      exhausted: boolean;
      cycles_contributed: number;
    };
  };
}

// ── API calls ─────────────────────────────────────────────────────────────────

export const api = {
  trainerHealth: () => get<HealthInfo>("/api/trainer/health"),
  trainerStatus: () => get<TrainerStatus>("/api/trainer/status"),
  trainerHistory: () => get<CycleSummary[]>("/api/trainer/history"),
  trainerPause: () => post<{ paused: boolean }>("/api/trainer/pause"),
  trainerResume: () => post<{ paused: boolean }>("/api/trainer/resume"),
  trainerRollback: (version: number) =>
    post<{ rolled_back_to: number }>(`/api/trainer/rollback/${version}`),

  metricsBudget: () => get<MetricsBudget>("/api/metrics/budget"),
  metricsHealth: () => get<MetricsHealth>("/api/metrics/health"),

  pipelines: () => get<PipelineRun[]>("/api/pipelines"),

  ecosystemStatus: () => get<EcosystemStatus>("/api/ecosystem/status"),
  ecosystemExport: (domain: string) =>
    post<{ ok: boolean; gmpack_path: string }>(`/api/ecosystem/export/${domain}`),
  ecosystemBootstrap: (domain: string, force = false) =>
    post<{ ok: boolean; bootstrapped: boolean }>(`/api/ecosystem/community/bootstrap/${domain}?force=${force}`),
  ecosystemPush: (domain: string) =>
    post<{ ok: boolean; adapter_id: string }>(`/api/ecosystem/community/push/${domain}`),
};
