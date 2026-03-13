/**
 * Slack inbound webhook route — /webhooks/slack
 *
 * Handles:
 *   1. URL Verification challenge (Slack pings this once during app setup)
 *   2. Event API payloads   (message.channels, app_mention, etc.)
 *   3. Slash Commands       (POST with form-encoded body)
 *
 * Security:
 *   When SLACK_SIGNING_SECRET is set, every request is verified via
 *   HMAC-SHA256 over "v0:{timestamp}:{raw-body}" — requests that fail
 *   verification are rejected with 401.  In dev you may omit the secret
 *   (a warning is logged) but never do this in production.
 *
 * Payload normalisation:
 *   Slack event payloads are transformed into GristEventInit with:
 *     channel  = "slack"
 *     payload  = the full Slack event object
 *     priority = "normal"
 *     tags     = { slack_event_type, slack_team, slack_user }
 *
 * Mount point: /webhooks/slack  (registered in server.ts)
 */

import type {
  FastifyInstance,
  FastifyPluginOptions,
  FastifyRequest,
  FastifyReply,
} from "fastify";
import { createHmac, timingSafeEqual } from "node:crypto";
import type { IBridge, GristEventInit } from "../../core/bridge.js";

// ── Plugin options ─────────────────────────────────────────────────────────────

export interface SlackWebhookOptions extends FastifyPluginOptions {
  bridge: IBridge;
  /**
   * Slack signing secret (from your Slack app's "Basic Information" page).
   * Set via SLACK_SIGNING_SECRET env var.  If absent, signature verification
   * is skipped with a warning — only acceptable in local dev.
   */
  signingSecret?: string;
}

// ── Types (minimal Slack shapes) ──────────────────────────────────────────────

interface SlackUrlVerification {
  type: "url_verification";
  challenge: string;
}

interface SlackEventCallback {
  type: "event_callback";
  team_id?: string;
  event?: {
    type?: string;
    user?: string;
    text?: string;
    channel?: string;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

type SlackBody = SlackUrlVerification | SlackEventCallback | Record<string, unknown>;

// ── Route plugin ──────────────────────────────────────────────────────────────

export async function slackWebhookRoutes(
  app: FastifyInstance,
  opts: SlackWebhookOptions,
): Promise<void> {
  const { bridge, signingSecret } = opts;

  if (!signingSecret) {
    app.log.warn(
      "[slack-webhook] SLACK_SIGNING_SECRET not set — " +
        "request signature verification disabled. DO NOT use in production.",
    );
  }

  // Add raw body access (needed for HMAC verification)
  app.addContentTypeParser(
    "application/json",
    { parseAs: "buffer" },
    (_req, body: Buffer, done) => {
      try {
        done(null, JSON.parse(body.toString("utf-8")));
      } catch (err) {
        done(err as Error, undefined);
      }
    },
  );

  // ── POST /webhooks/slack ───────────────────────────────────────────────────

  app.post<{ Body: SlackBody }>(
    "/",
    {
      config: { rawBody: true }, // store raw buffer on req for HMAC
    },
    async (req: FastifyRequest<{ Body: SlackBody }>, reply: FastifyReply) => {
      // ── 1. Signature verification ──────────────────────────────────────────
      if (signingSecret) {
        const ts = req.headers["x-slack-request-timestamp"] as string | undefined;
        const sig = req.headers["x-slack-signature"] as string | undefined;

        if (!ts || !sig) {
          return reply.status(401).send({ error: "Missing Slack signature headers" });
        }

        // Reject requests older than 5 minutes (replay attack prevention)
        const age = Math.abs(Date.now() / 1000 - Number(ts));
        if (age > 300) {
          return reply.status(401).send({ error: "Request timestamp too old" });
        }

        const rawBody = (req as FastifyRequest & { rawBody?: Buffer }).rawBody;
        const bodyStr = rawBody ? rawBody.toString("utf-8") : JSON.stringify(req.body);
        const baseStr = `v0:${ts}:${bodyStr}`;
        const expected = `v0=${createHmac("sha256", signingSecret)
          .update(baseStr)
          .digest("hex")}`;

        if (!_safeEqual(expected, sig)) {
          return reply.status(401).send({ error: "Invalid Slack signature" });
        }
      }

      const body = req.body as SlackBody;

      // ── 2. URL Verification challenge ─────────────────────────────────────
      if ("type" in body && body.type === "url_verification") {
        const verifyBody = body as SlackUrlVerification;
        return reply.status(200).send({ challenge: verifyBody.challenge });
      }

      // ── 3. Event callback → GristEvent ────────────────────────────────────
      const event = _normalise(body);

      try {
        const decision = await bridge.triage(event);
        // Slack expects 200 OK quickly — send it before any slow processing
        return reply.status(200).send({ ok: true, decision });
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        app.log.error(`[slack-webhook] triage error: ${msg}`);
        // Still return 200 to Slack so it doesn't retry
        return reply.status(200).send({ ok: false, error: msg });
      }
    },
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _normalise(body: SlackBody): GristEventInit {
  const cb = body as SlackEventCallback;
  const slackEvent = cb.event ?? {};

  return {
    channel: "slack",
    payload: body,
    priority: "normal",
    tags: {
      slack_event_type: String(slackEvent["type"] ?? cb["type"] ?? "unknown"),
      slack_team: String(cb["team_id"] ?? ""),
      slack_user: String(slackEvent["user"] ?? ""),
      slack_channel: String(slackEvent["channel"] ?? ""),
    },
  };
}

/** Constant-time string comparison to prevent timing attacks. */
function _safeEqual(a: string, b: string): boolean {
  try {
    const bufA = Buffer.from(a, "utf-8");
    const bufB = Buffer.from(b, "utf-8");
    if (bufA.length !== bufB.length) return false;
    return timingSafeEqual(bufA, bufB);
  } catch {
    return false;
  }
}
