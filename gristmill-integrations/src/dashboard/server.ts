/**
 * Dashboard server — Fastify API + optional React SPA static files.
 *
 * Mounts:
 *   /api/pipelines  → pipelinesRoutes
 *   /api/memory     → memoryRoutes
 *   /api/metrics    → metricsRoutes
 *   /api/triage     → triageRoutes
 *   /api/watches    → watchesRoutes  (when watchEngine is provided)
 *   /api/plugins    → pluginsRoutes  (when pluginRegistry is provided)
 *
 * If `dashboard/ui/dist/` exists at runtime, also serves the SPA at `/`.
 */

import { readFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import Fastify, { type FastifyInstance } from "fastify";
import cors from "@fastify/cors";

import type { IBridge, GristEventInit } from "../core/bridge.js";
import type { PluginRegistry } from "../plugins/registry.js";
import type { WatchEngine } from "../bell-tower/watch.js";
import { pipelinesRoutes } from "./routes/pipelines.js";
import { memoryRoutes } from "./routes/memory.js";
import { metricsRoutes } from "./routes/metrics.js";
import { triageRoutes } from "./routes/triage.js";
import { trainerRoutes } from "./routes/trainer.js";
import { watchesRoutes } from "./routes/watches.js";
import { pluginsRoutes } from "./routes/plugins.js";
import { slackWebhookRoutes } from "./routes/slack-webhook.js";
import { ecosystemRoutes } from "./routes/ecosystem.js";

export interface DashboardConfig {
  /** Port to listen on. Default: 4000 */
  port?: number;
  /** Host to bind. Default: "127.0.0.1" */
  host?: string;
  /** Allow CORS from these origins. Default: ["http://localhost:5173"] (Vite dev) */
  corsOrigins?: string[];
  /** Path to the built React SPA `dist/` directory. */
  uiDistPath?: string;
  /**
   * Slack signing secret for inbound webhook verification.
   * When set, mounts POST /webhooks/slack with HMAC-SHA256 verification.
   * Source: your Slack app → Basic Information → App Credentials → Signing Secret.
   * Pass via SLACK_SIGNING_SECRET env var — never hardcode.
   */
  slackSigningSecret?: string;
  /**
   * When provided, mounts the Watch CRUD API at `/api/watches`.
   * Should be the same `WatchEngine` instance given to `NotificationDispatcher`.
   */
  watchEngine?: WatchEngine;
  /**
   * If provided alongside `watchEngine`, the watch list is saved to this file
   * after every mutation and can be reloaded at startup via
   * `WatchEngine.loadFromFile()`.
   */
  watchPersistPath?: string;
  /**
   * When provided, mounts a read-only plugin status API at `/api/plugins`.
   * Should be the `PluginRegistry` instance after `load()` + `register()`.
   */
  pluginRegistry?: PluginRegistry;
}

export function createDashboardServer(
  bridge: IBridge,
  config: DashboardConfig = {},
): FastifyInstance {
  const app = Fastify({ logger: { level: "info" } });

  // CORS
  app.register(cors, {
    origin: config.corsOrigins ?? ["http://localhost:5173"],
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  });

  // Event intake — POST /events (HttpHopper-compatible ingest endpoint)
  app.post<{ Body: unknown }>(
    "/events",
    { schema: { body: { type: "object" } } },
    async (req, reply) => {
      const body = req.body as Record<string, unknown>;
      const event: GristEventInit = {
        channel: String(body["channel"] ?? "http"),
        payload: body["payload"] ?? body,
        priority: (body["priority"] as GristEventInit["priority"]) ?? "normal",
        correlationId: body["correlationId"] as string | undefined,
        tags: (body["tags"] as Record<string, string>) ?? {},
      };
      try {
        const decision = await bridge.triage(event);
        return reply.status(200).send(decision);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        return reply.status(500).send({ error: msg });
      }
    },
  );

  // API routes — core
  app.register(pipelinesRoutes, { bridge, prefix: "/api/pipelines" });
  app.register(memoryRoutes,    { bridge, prefix: "/api/memory" });
  app.register(metricsRoutes,   { bridge, prefix: "/api/metrics" });
  app.register(triageRoutes,    { bridge, prefix: "/api/triage" });
  // API routes — distillation trainer (proxies to localhost:7432)
  app.register(trainerRoutes,   { prefix: "/api/trainer" });
  // API routes — ecosystem (Phase 4: portability & sharing, proxies to localhost:7432)
  app.register(ecosystemRoutes, { prefix: "/api/ecosystem" });

  // API routes — watches (optional, requires watchEngine)
  if (config.watchEngine) {
    app.register(watchesRoutes, {
      watchEngine: config.watchEngine,
      persistPath: config.watchPersistPath,
      prefix: "/api/watches",
    });
  }

  // API routes — plugins (optional, requires pluginRegistry)
  if (config.pluginRegistry) {
    app.register(pluginsRoutes, {
      registry: config.pluginRegistry,
      prefix: "/api/plugins",
    });
  }

  // Slack inbound webhook — POST /webhooks/slack
  // Always mounted; signature verification activates when slackSigningSecret is set.
  app.register(slackWebhookRoutes, {
    bridge,
    signingSecret: config.slackSigningSecret,
    prefix: "/webhooks/slack",
  });

  // SPA static files (optional — only if the build output exists)
  const uiDist =
    config.uiDistPath ??
    _resolveUiDist();

  if (uiDist && existsSync(uiDist)) {
    // Dynamically import @fastify/static so the package remains optional
    import("@fastify/static").then(({ default: staticPlugin }) => {
      app.register(staticPlugin, {
        root: uiDist,
        prefix: "/",
        // For SPA routing: serve index.html for unknown paths
        wildcard: false,
      });

      // Catch-all for client-side routing
      app.setNotFoundHandler(async (_req, reply) => {
        const indexPath = join(uiDist, "index.html");
        if (existsSync(indexPath)) {
          const html = readFileSync(indexPath, "utf8");
          return reply.type("text/html").send(html);
        }
        return reply.code(404).send({ error: "Not found" });
      });
    }).catch(() => {
      app.log.warn(
        "@fastify/static not installed — dashboard SPA will not be served",
      );
    });
  } else {
    // Root health endpoint when no SPA is present
    app.get("/", async (_req, reply) => {
      const api = [
        "/events",
        "/api/pipelines",
        "/api/memory/remember",
        "/api/memory/recall",
        "/api/metrics/budget",
        "/api/metrics/health",
        "/api/triage",
        "/api/ecosystem/status",
      ];
      if (config.watchEngine) api.push("/api/watches");
      if (config.pluginRegistry) api.push("/api/plugins");
      return reply.send({ name: "gristmill-dashboard", status: "ok", api });
    });
  }

  return app;
}

/**
 * Start the dashboard server and listen on the configured port.
 * Returns the listening URL.
 */
export async function startDashboard(
  bridge: IBridge,
  config: DashboardConfig = {},
): Promise<string> {
  const app = createDashboardServer(bridge, config);
  const host = config.host ?? "127.0.0.1";
  const port = config.port ?? 4000;
  const address = await app.listen({ port, host });
  return address;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _resolveUiDist(): string | null {
  try {
    const __dirname = dirname(fileURLToPath(import.meta.url));
    // Compiled output: dist/dashboard/server.js → navigate to ui/dist
    const candidate = join(__dirname, "..", "dashboard", "ui", "dist");
    return candidate;
  } catch {
    return null;
  }
}
