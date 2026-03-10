/**
 * HttpHopper — normalises HTTP and WebSocket events into GristEvents
 * and submits them to the Rust core via the bridge.
 *
 * Architecture note: TypeScript handles I/O; all routing decisions come
 * from `bridge.triage()` which delegates to `grist-sieve`.
 */

import type { FastifyInstance, FastifyRequest, FastifyReply } from "fastify";
import fastify from "fastify";
import cors from "@fastify/cors";
import { WebSocketServer, WebSocket } from "ws";
import type { IncomingMessage } from "node:http";
import type { IBridge, GristEventInit, RouteDecision } from "../core/bridge.js";

// ── Types ─────────────────────────────────────────────────────────────────────

/**
 * A function that transforms a raw incoming payload into a normalised
 * `GristEventInit`.  Structurally identical to `AdapterHandler` from
 * plugins/types — typed locally to avoid a cross-package import cycle.
 */
export type AdapterHandler = (
  raw: unknown,
) => GristEventInit | Promise<GristEventInit>;

export interface HopperConfig {
  port?: number;
  host?: string;
  /** If provided, CORS is enabled for these origins. Pass `["*"]` to allow all. */
  corsOrigins?: string[];
  /** Maximum incoming body size in bytes (default: 1 MB). */
  bodyLimit?: number;
  /**
   * Plugin-registered adapter handlers keyed by channel name.
   *
   * When a POST /events request arrives with `channel` matching a key here,
   * the adapter is called to normalise the raw payload into a `GristEventInit`
   * before the event is triaged.  This lets plugins define custom ingestion
   * formats (e.g. GitHub webhooks, PagerDuty alerts) without modifying core.
   *
   * Populated at startup by calling `PluginRegistry.adapters` after plugins
   * have been loaded.
   */
  pluginAdapters?: ReadonlyMap<string, AdapterHandler>;
}

// ── HttpHopper ────────────────────────────────────────────────────────────────

/**
 * Fastify-based HTTP adapter.
 *
 * Endpoints:
 *   POST /events             — submit an event; returns the RouteDecision
 *   GET  /events/triage      — same as POST (for simple GET clients)
 *   GET  /health             — liveness check
 */
export class HttpHopper {
  private app: FastifyInstance;
  private wsServer?: WebSocketServer;

  constructor(
    private readonly bridge: IBridge,
    private readonly config: HopperConfig = {}
  ) {
    this.app = fastify({
      logger: false,
      bodyLimit: config.bodyLimit ?? 1_048_576,
    });
    this._registerRoutes();
  }

  // ── Setup ──────────────────────────────────────────────────────────────────

  private _registerRoutes(): void {
    const { app, bridge } = this;

    // CORS
    if (this.config.corsOrigins) {
      app.register(cors, { origin: this.config.corsOrigins });
    }

    // Health check
    app.get("/health", async (_req: FastifyRequest, _reply: FastifyReply) => {
      return { status: "ok", ts: Date.now() };
    });

    // Event intake — POST /events
    app.post<{ Body: unknown }>(
      "/events",
      { schema: { body: { type: "object" } } },
      async (req: FastifyRequest<{ Body: unknown }>, reply: FastifyReply) => {
        const body = req.body as Record<string, unknown>;
        const channel = String(body["channel"] ?? "http");

        let event: GristEventInit;

        // If a plugin has registered an adapter for this channel name, use it
        // to normalise the raw payload before triaging.
        const adapterFn = this.config.pluginAdapters?.get(channel);
        if (adapterFn) {
          try {
            event = await adapterFn(body["payload"] ?? body);
            // Ensure the channel label is preserved from the request
            if (!event.channel) event = { ...event, channel };
          } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : String(err);
            return reply
              .status(400)
              .send({ error: `Adapter "${channel}" failed: ${msg}` });
          }
        } else {
          event = {
            channel,
            payload: body["payload"] ?? body,
            priority: (body["priority"] as GristEventInit["priority"]) ?? "normal",
            correlationId: body["correlationId"] as string | undefined,
            tags: (body["tags"] as Record<string, string>) ?? {},
          };
        }

        try {
          const decision: RouteDecision = await bridge.triage(event);
          return reply.status(200).send(decision);
        } catch (err: unknown) {
          const msg = err instanceof Error ? err.message : String(err);
          return reply.status(500).send({ error: msg });
        }
      }
    );

    // Memory — POST /memory/remember
    app.post<{ Body: { content: string; tags?: string[] } }>(
      "/memory/remember",
      async (req, reply) => {
        const { content, tags = [] } = req.body;
        if (!content) return reply.status(400).send({ error: "content required" });
        const id = await bridge.remember(content, tags);
        return reply.status(201).send({ id });
      }
    );

    // Memory — POST /memory/recall
    app.post<{ Body: { query: string; limit?: number } }>(
      "/memory/recall",
      async (req, reply) => {
        const { query, limit = 10 } = req.body;
        if (!query) return reply.status(400).send({ error: "query required" });
        const results = await bridge.recall(query, limit);
        return reply.status(200).send(results);
      }
    );
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  async start(): Promise<void> {
    const port = this.config.port ?? 3000;
    const host = this.config.host ?? "0.0.0.0";
    await this.app.listen({ port, host });
    console.log(`HttpHopper listening on http://${host}:${port}`);
  }

  async stop(): Promise<void> {
    this.wsServer?.close();
    await this.app.close();
  }

  /** Expose the raw Fastify instance (e.g. for attaching WebSocket). */
  get server(): FastifyInstance {
    return this.app;
  }
}

// ── WebSocketHopper ───────────────────────────────────────────────────────────

/**
 * WebSocket adapter.  Attaches to an existing `http.Server` (typically the
 * Fastify server's underlying server).
 *
 * Protocol:
 *   Client sends: JSON object `{ channel, payload, priority?, ... }`
 *   Server replies: JSON `RouteDecision` or `{ error: string }`
 */
export class WebSocketHopper {
  private wss?: WebSocketServer;

  constructor(private readonly bridge: IBridge) {}

  attach(httpServer: import("node:http").Server): void {
    this.wss = new WebSocketServer({ server: httpServer });

    this.wss.on("connection", (ws: WebSocket, _req: IncomingMessage) => {
      ws.on("message", async (raw: Buffer | string) => {
        let event: GristEventInit;

        try {
          const msg =
            typeof raw === "string" ? raw : raw.toString("utf-8");
          const parsed = JSON.parse(msg) as Record<string, unknown>;
          event = {
            channel: String(parsed["channel"] ?? "websocket"),
            payload: parsed["payload"] ?? parsed,
            priority:
              (parsed["priority"] as GristEventInit["priority"]) ?? "normal",
          };
        } catch {
          ws.send(JSON.stringify({ error: "invalid JSON" }));
          return;
        }

        try {
          const decision = await this.bridge.triage(event);
          ws.send(JSON.stringify(decision));
        } catch (err: unknown) {
          const msg = err instanceof Error ? err.message : String(err);
          ws.send(JSON.stringify({ error: msg }));
        }
      });
    });
  }

  close(): void {
    this.wss?.close();
  }
}
