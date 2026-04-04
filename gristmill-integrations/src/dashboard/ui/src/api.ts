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

async function patch<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "PATCH",
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

// Matches GET /api/metrics/budget
export interface MetricsBudget {
  daily_used: number;
  daily_limit: number;
  window_start_ms: number;
  pct_used: number;
  cached_at: string | null;
  status?: string;
}

// Matches GET /api/metrics/health
export interface MetricsHealth {
  status: string;
  uptime: number;
  timestamp: string;
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

// ── Memory types ──────────────────────────────────────────────────────────────

export interface MemoryItem {
  id: string;
  content: string;
  tags: string[];
  createdAtMs: number;
  tier: string;
}

export interface RankedMemoryItem {
  memory: MemoryItem;
  score: number;
}

// ── Watch type ────────────────────────────────────────────────────────────────

export interface WatchItem {
  id: string;
  name: string;
  topic: string;
  condition: string;
  channelIds: string[];
  enabled: boolean;
  cooldownMs: number;
  createdAt: string;
}

// ── Routing metrics type ──────────────────────────────────────────────────────

export interface RoutingMetrics {
  sieve: {
    confidence_threshold: number;
    routing_cache: {
      exact_hits: number;
      semantic_hits: number;
      misses: number;
      hit_rate: number;
    };
    feedback_records_sent: number;
  };
  hammer: {
    cache_size: number;
    daily_tokens_used: number;
    daily_token_limit: number;
    daily_tokens_remaining: number;
    monthly_tokens_used: number;
    monthly_token_limit: number;
  };
  pipelines_registered: number;
  _stub?: boolean;
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

  // ── Memory ─────────────────────────────────────────────────────────────────

  memoryRecall: (query: string, limit = 10) =>
    post<RankedMemoryItem[]>("/api/memory/recall", { query, limit }),
  memoryRemember: (content: string, tags: string[] = []) =>
    post<{ id: string }>("/api/memory/remember", { content, tags }),
  memoryGet: (id: string) => get<MemoryItem | null>(`/api/memory/${id}`),

  // ── Watches ────────────────────────────────────────────────────────────────

  watchesList: () => get<WatchItem[]>("/api/watches"),
  watchCreate: (w: Omit<WatchItem, "id" | "createdAt">) =>
    post<WatchItem>("/api/watches", w),
  watchUpdate: (id: string, patchBody: Partial<Omit<WatchItem, "id">>) =>
    patch<WatchItem>(`/api/watches/${id}`, patchBody),
  watchDelete: (id: string) => fetch(`/api/watches/${id}`, { method: "DELETE" }),

  // ── Metrics ────────────────────────────────────────────────────────────────

  metricsRouting: () => get<RoutingMetrics>("/api/metrics/routing"),
};
