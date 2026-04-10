/**
 * Ecosystem routes — Phase 4: Portability & Sharing.
 *
 * All routes proxy to the gristmill-trainer health API on localhost:7432.
 * TypeScript handles only I/O; all model operations stay in Python/Rust.
 *
 * Mounts under /api/ecosystem:
 *   GET  /api/ecosystem/status
 *   POST /api/ecosystem/export/:domain
 *   POST /api/ecosystem/import
 *   GET  /api/ecosystem/community/adapters
 *   POST /api/ecosystem/community/push/:domain
 *   POST /api/ecosystem/community/bootstrap/:domain
 */

import type { FastifyInstance, FastifyPluginOptions } from "fastify";

const TRAINER_BASE = process.env["TRAINER_URL"] ?? "http://127.0.0.1:7432";
const TIMEOUT_MS = 10_000;
// Export/import operations may take longer (file I/O + potential download)
const LONG_TIMEOUT_MS = 120_000;

// ── Proxy helpers ─────────────────────────────────────────────────────────────

async function proxyGet(path: string, timeoutMs = TIMEOUT_MS): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${TRAINER_BASE}${path}`, {
      method: "GET",
      signal: controller.signal,
      headers: { Accept: "application/json" },
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw Object.assign(
        new Error((body as { detail?: string }).detail ?? `Trainer error ${res.status}`),
        { statusCode: res.status },
      );
    }
    return await res.json();
  } catch (err: unknown) {
    if ((err as { statusCode?: number }).statusCode) throw err;
    const msg = err instanceof Error ? err.message : String(err);
    throw Object.assign(new Error(`Trainer unavailable: ${msg}`), { statusCode: 503 });
  } finally {
    clearTimeout(timer);
  }
}

async function proxyPost(
  path: string,
  body?: unknown,
  timeoutMs = TIMEOUT_MS,
): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${TRAINER_BASE}${path}`, {
      method: "POST",
      signal: controller.signal,
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw Object.assign(
        new Error((errBody as { detail?: string }).detail ?? `Trainer error ${res.status}`),
        { statusCode: res.status },
      );
    }
    return await res.json();
  } catch (err: unknown) {
    if ((err as { statusCode?: number }).statusCode) throw err;
    const msg = err instanceof Error ? err.message : String(err);
    throw Object.assign(new Error(`Trainer unavailable: ${msg}`), { statusCode: 503 });
  } finally {
    clearTimeout(timer);
  }
}

function trainerOffline(err: unknown): { error: string; trainer_available: false } {
  const msg = err instanceof Error ? err.message : String(err);
  return { error: msg, trainer_available: false };
}

// ── Route plugin ──────────────────────────────────────────────────────────────

export async function ecosystemRoutes(
  fastify: FastifyInstance,
  _opts: FastifyPluginOptions,
): Promise<void> {
  /**
   * GET /api/ecosystem/status
   * Returns community opt-in status, federation enabled flag, and privacy budget.
   */
  fastify.get("/status", async (_req, reply) => {
    try {
      return reply.send(await proxyGet("/ecosystem/status"));
    } catch (err) {
      const code = (err as { statusCode?: number }).statusCode ?? 503;
      return reply.status(code).send(trainerOffline(err));
    }
  });

  /**
   * POST /api/ecosystem/export/:domain
   * Pack the active adapter for :domain into a .gmpack bundle.
   * Returns { ok, gmpack_path, domain }.
   */
  fastify.post<{ Params: { domain: string } }>(
    "/export/:domain",
    async (req, reply) => {
      const { domain } = req.params;
      try {
        return reply.send(
          await proxyPost(`/ecosystem/export/${encodeURIComponent(domain)}`, undefined, LONG_TIMEOUT_MS),
        );
      } catch (err) {
        const code = (err as { statusCode?: number }).statusCode ?? 503;
        return reply.status(code).send(trainerOffline(err));
      }
    },
  );

  /**
   * POST /api/ecosystem/import
   * Body: { gmpack_path: string, domain?: string }
   * Unpack, stage, and promote a .gmpack bundle.
   */
  fastify.post<{ Body: { gmpack_path: string; domain?: string } }>(
    "/import",
    {
      schema: {
        body: {
          type: "object",
          required: ["gmpack_path"],
          properties: {
            gmpack_path: { type: "string" },
            domain: { type: "string" },
          },
        },
      },
    },
    async (req, reply) => {
      try {
        return reply.send(
          await proxyPost("/ecosystem/import", req.body, LONG_TIMEOUT_MS),
        );
      } catch (err) {
        const code = (err as { statusCode?: number }).statusCode ?? 503;
        return reply.status(code).send(trainerOffline(err));
      }
    },
  );

  /**
   * GET /api/ecosystem/community/adapters?domain=code&min_score=0.7&limit=20
   * List available community adapters for a domain.
   */
  fastify.get<{
    Querystring: { domain: string; min_score?: string; limit?: string };
  }>(
    "/community/adapters",
    {
      schema: {
        querystring: {
          type: "object",
          required: ["domain"],
          properties: {
            domain: { type: "string" },
            min_score: { type: "string" },
            limit: { type: "string" },
          },
        },
      },
    },
    async (req, reply) => {
      const { domain, min_score = "0", limit = "20" } = req.query;
      const qs = new URLSearchParams({ domain, min_score, limit }).toString();
      try {
        return reply.send(await proxyGet(`/ecosystem/community/adapters?${qs}`));
      } catch (err) {
        const code = (err as { statusCode?: number }).statusCode ?? 503;
        return reply.status(code).send(trainerOffline(err));
      }
    },
  );

  /**
   * POST /api/ecosystem/community/push/:domain
   * Export the active adapter for :domain and push it to the community repo.
   */
  fastify.post<{ Params: { domain: string } }>(
    "/community/push/:domain",
    async (req, reply) => {
      const { domain } = req.params;
      try {
        return reply.send(
          await proxyPost(
            `/ecosystem/community/push/${encodeURIComponent(domain)}`,
            undefined,
            LONG_TIMEOUT_MS,
          ),
        );
      } catch (err) {
        const code = (err as { statusCode?: number }).statusCode ?? 503;
        return reply.status(code).send(trainerOffline(err));
      }
    },
  );

  /**
   * POST /api/ecosystem/community/bootstrap/:domain?force=true
   * Trigger cold-start bootstrapping for :domain from the community repo.
   */
  fastify.post<{
    Params: { domain: string };
    Querystring: { force?: string };
  }>(
    "/community/bootstrap/:domain",
    async (req, reply) => {
      const { domain } = req.params;
      const force = req.query.force === "true" ? "true" : "false";
      try {
        return reply.send(
          await proxyPost(
            `/ecosystem/community/bootstrap/${encodeURIComponent(domain)}?force=${force}`,
            undefined,
            LONG_TIMEOUT_MS,
          ),
        );
      } catch (err) {
        const code = (err as { statusCode?: number }).statusCode ?? 503;
        return reply.status(code).send(trainerOffline(err));
      }
    },
  );
}
