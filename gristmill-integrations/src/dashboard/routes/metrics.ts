/**
 * Metrics routes — budget snapshot and health check.
 *
 * GET /api/metrics/budget  → latest cached hammer.budget bus event
 * GET /api/metrics/health  → { status: "ok", uptime: number }
 */

import type { FastifyInstance, FastifyPluginOptions } from "fastify";
import type { IBridge } from "../../core/bridge.js";

interface PluginOpts extends FastifyPluginOptions {
  bridge: IBridge;
}

interface BudgetSnapshot {
  daily_used: number;
  daily_limit: number;
  window_start_ms: number;
  pct_used: number;
  cached_at: string;
}

export async function metricsRoutes(
  app: FastifyInstance,
  opts: PluginOpts,
): Promise<void> {
  const { bridge } = opts;

  // Keep the latest hammer.budget event cached in memory
  let budgetCache: BudgetSnapshot | null = null;

  // Subscribe once in background; updates the cache on each event
  (async () => {
    try {
      for await (const raw of bridge.subscribe("hammer.budget")) {
        const p = raw as {
          daily_used?: number;
          daily_limit?: number;
          window_start_ms?: number;
        };
        const used = p.daily_used ?? 0;
        const limit = p.daily_limit ?? 1;
        budgetCache = {
          daily_used: used,
          daily_limit: limit,
          window_start_ms: p.window_start_ms ?? Date.now(),
          pct_used: Math.round((used / limit) * 10000) / 100,
          cached_at: new Date().toISOString(),
        };
      }
    } catch {
      // subscription ended or bridge shut down — ignore
    }
  })();

  app.get("/budget", async (_req, reply) => {
    if (!budgetCache) {
      return reply.send({
        daily_used: 0,
        daily_limit: 0,
        window_start_ms: Date.now(),
        pct_used: 0,
        cached_at: null,
        status: "no_data",
      });
    }
    return reply.send(budgetCache);
  });

  app.get("/health", async (_req, reply) => {
    return reply.send({
      status: "ok",
      uptime: process.uptime(),
      timestamp: new Date().toISOString(),
    });
  });
}
