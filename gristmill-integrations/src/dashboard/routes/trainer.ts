/**
 * Trainer routes — proxy to the gristmill-trainer health API on localhost:7432.
 *
 * Mounts under /api/trainer:
 *   GET  /api/trainer/health
 *   GET  /api/trainer/status
 *   GET  /api/trainer/history
 *   GET  /api/trainer/validation/latest
 *   POST /api/trainer/pause
 *   POST /api/trainer/resume
 *   POST /api/trainer/rollback/:version
 *
 * All requests are forwarded to http://127.0.0.1:7432 with a 5-second timeout.
 * If the trainer service is not running a 503 is returned rather than crashing.
 */

import type { FastifyInstance, FastifyPluginOptions } from "fastify";

const TRAINER_BASE = process.env["TRAINER_URL"] ?? "http://127.0.0.1:7432";
const TIMEOUT_MS = 5_000;

export async function trainerRoutes(
  fastify: FastifyInstance,
  _opts: FastifyPluginOptions,
): Promise<void> {
  // ── Proxy helper ────────────────────────────────────────────────────────────

  async function proxy(
    trainerPath: string,
    method: "GET" | "POST" = "GET",
  ): Promise<unknown> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
    try {
      const res = await fetch(`${TRAINER_BASE}${trainerPath}`, {
        method,
        signal: controller.signal,
        headers: { "Content-Type": "application/json" },
      });
      return await res.json();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      throw Object.assign(new Error(`Trainer unavailable: ${msg}`), { statusCode: 503 });
    } finally {
      clearTimeout(timer);
    }
  }

  // ── Routes ──────────────────────────────────────────────────────────────────

  fastify.get("/health", async (_req, reply) => {
    try {
      return reply.send(await proxy("/health"));
    } catch (err: unknown) {
      return reply.status(503).send(trainerOffline(err));
    }
  });

  fastify.get("/status", async (_req, reply) => {
    try {
      return reply.send(await proxy("/status"));
    } catch (err: unknown) {
      return reply.status(503).send(trainerOffline(err));
    }
  });

  fastify.get("/history", async (_req, reply) => {
    try {
      return reply.send(await proxy("/history"));
    } catch (err: unknown) {
      return reply.status(503).send(trainerOffline(err));
    }
  });

  fastify.get("/validation/latest", async (_req, reply) => {
    try {
      return reply.send(await proxy("/validation/latest"));
    } catch (err: unknown) {
      return reply.status(503).send(trainerOffline(err));
    }
  });

  fastify.post("/pause", async (_req, reply) => {
    try {
      return reply.send(await proxy("/pause", "POST"));
    } catch (err: unknown) {
      return reply.status(503).send(trainerOffline(err));
    }
  });

  fastify.post("/resume", async (_req, reply) => {
    try {
      return reply.send(await proxy("/resume", "POST"));
    } catch (err: unknown) {
      return reply.status(503).send(trainerOffline(err));
    }
  });

  fastify.post<{ Params: { version: string } }>(
    "/rollback/:version",
    async (req, reply) => {
      const version = parseInt(req.params.version, 10);
      if (isNaN(version) || version < 1) {
        return reply.status(400).send({ error: "version must be a positive integer" });
      }
      try {
        return reply.send(await proxy(`/rollback/${version}`, "POST"));
      } catch (err: unknown) {
        return reply.status(503).send(trainerOffline(err));
      }
    },
  );
}

function trainerOffline(err: unknown): { error: string; trainer_available: false } {
  const msg = err instanceof Error ? err.message : String(err);
  return { error: msg, trainer_available: false };
}
