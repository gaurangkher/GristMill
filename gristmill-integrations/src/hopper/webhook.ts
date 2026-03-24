/**
 * WebhookHopper — receives inbound HTTP webhooks, verifies HMAC-SHA256
 * signatures, normalises the payload into a GristEvent, and submits it
 * to the Rust core via the bridge.
 *
 * Supports any service that signs payloads with an HMAC-SHA256 secret
 * in a request header (GitHub, GitLab, Linear, Stripe, etc.).
 *
 * Architecture: TypeScript handles I/O and signature verification only.
 * All routing decisions come from `bridge.triage()`.
 *
 * Usage:
 *   const wh = new WebhookHopper(bridge, { port: 3002 });
 *   wh.register("github", { secret: "...", headerName: "x-hub-signature-256" });
 *   await wh.start();
 *
 * Config via config.yaml (integrations.hoppers.webhook):
 *   port: 3002
 *   channels:
 *     github:
 *       secret: "${GITHUB_WEBHOOK_SECRET}"   # use env vars — never hardcode
 *       header: "x-hub-signature-256"
 *     stripe:
 *       secret: "${STRIPE_WEBHOOK_SECRET}"
 *       header: "stripe-signature"
 */

import { createHmac, timingSafeEqual } from "node:crypto";
import fastify, { type FastifyInstance } from "fastify";
import type { IBridge, GristEventInit } from "../core/bridge.js";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface WebhookChannelConfig {
  /** HMAC-SHA256 secret for this channel.  Set via env var — never hardcode. */
  secret: string;
  /**
   * Header that carries the signature (default: `"x-hub-signature-256"`).
   * GitHub format: `sha256=<hex>`
   * Raw hex signatures are also accepted.
   */
  headerName?: string;
}

export interface WebhookHopperConfig {
  port?: number;
  host?: string;
  /** Pre-configured channel secrets.  More can be added via `register()`. */
  channels?: Record<string, WebhookChannelConfig>;
}

// ── WebhookHopper ─────────────────────────────────────────────────────────────

/**
 * Fastify-based generic webhook receiver.
 *
 * Endpoint: POST /webhooks/:channel
 *
 * For each registered channel the hopper:
 *   1. Reads the raw body (needed for HMAC computation).
 *   2. Verifies the signature in the configured header using timing-safe compare.
 *   3. Parses JSON and wraps as GristEvent { channel, payload, tags }.
 *   4. Calls bridge.triage() and returns the RouteDecision as JSON.
 *
 * Unregistered channels always return 404.
 * Invalid signatures return 401.
 */
export class WebhookHopper {
  private app: FastifyInstance;
  private channels = new Map<string, WebhookChannelConfig>();

  constructor(
    private readonly bridge: IBridge,
    private readonly config: WebhookHopperConfig = {},
  ) {
    this.app = fastify({ logger: false });
    for (const [ch, cfg] of Object.entries(config.channels ?? {})) {
      this.channels.set(ch, cfg);
    }
    this._registerRoutes();
  }

  /** Register (or update) a webhook channel at runtime. */
  register(channel: string, cfg: WebhookChannelConfig): void {
    this.channels.set(channel, cfg);
  }

  // ── Routes ──────────────────────────────────────────────────────────────────

  private _registerRoutes(): void {
    const { app, bridge } = this;

    app.get("/health", async () => ({ status: "ok", ts: Date.now() }));

    // Raw body is required for HMAC verification — disable JSON auto-parse
    app.addContentTypeParser(
      "application/json",
      { parseAs: "buffer" },
      (_req, body, done) => done(null, body),
    );

    app.post<{ Params: { channel: string } }>(
      "/webhooks/:channel",
      async (req, reply) => {
        const { channel } = req.params;
        const cfg = this.channels.get(channel);
        if (!cfg) {
          return reply.status(404).send({ error: `Unknown webhook channel: ${channel}` });
        }

        const rawBody = req.body as Buffer;

        // Signature verification
        const headerName = cfg.headerName ?? "x-hub-signature-256";
        const sigHeader = req.headers[headerName] as string | undefined;
        if (!sigHeader) {
          return reply.status(401).send({ error: `Missing signature header: ${headerName}` });
        }

        if (!_verifySignature(rawBody, cfg.secret, sigHeader)) {
          return reply.status(401).send({ error: "Signature verification failed" });
        }

        // Parse payload
        let payload: unknown;
        try {
          payload = JSON.parse(rawBody.toString("utf-8"));
        } catch {
          payload = rawBody.toString("utf-8");
        }

        const event: GristEventInit = {
          channel,
          payload,
          priority: "normal",
          tags: {
            source: "webhook",
            webhook_channel: channel,
          },
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
  }

  // ── Lifecycle ───────────────────────────────────────────────────────────────

  async start(): Promise<void> {
    const port = this.config.port ?? 3002;
    const host = this.config.host ?? "0.0.0.0";
    await this.app.listen({ port, host });
    console.log(`WebhookHopper listening on http://${host}:${port}/webhooks/:channel`);
  }

  async stop(): Promise<void> {
    await this.app.close();
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Verify an HMAC-SHA256 signature using timing-safe comparison.
 *
 * Accepts both `sha256=<hex>` (GitHub style) and plain hex signatures.
 */
function _verifySignature(body: Buffer, secret: string, sigHeader: string): boolean {
  try {
    const expected = createHmac("sha256", secret).update(body).digest("hex");
    // Strip "sha256=" prefix if present
    const provided = sigHeader.startsWith("sha256=") ? sigHeader.slice(7) : sigHeader;
    const expectedBuf = Buffer.from(expected, "hex");
    const providedBuf = Buffer.from(provided, "hex");
    if (expectedBuf.length !== providedBuf.length) return false;
    return timingSafeEqual(expectedBuf, providedBuf);
  } catch {
    return false;
  }
}
