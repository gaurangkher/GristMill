/**
 * Watch CRUD routes — manage WatchEngine rules over HTTP.
 *
 * Endpoints:
 *   GET    /api/watches          — list all watches
 *   POST   /api/watches          — create a watch
 *   GET    /api/watches/:id      — get a single watch
 *   PATCH  /api/watches/:id      — update a watch (partial)
 *   DELETE /api/watches/:id      — delete a watch
 *
 * Watches are optionally persisted to disk after each mutation.
 * Pass `persistPath` in the options to enable persistence.
 */

import type { FastifyInstance, FastifyPluginOptions } from "fastify";
import { WatchEngine, createWatch } from "../../bell-tower/watch.js";
import type { Watch } from "../../bell-tower/watch.js";

// ── Plugin options ─────────────────────────────────────────────────────────────

export interface WatchesRoutesOptions extends FastifyPluginOptions {
  /** Shared WatchEngine instance (same one passed to NotificationDispatcher). */
  watchEngine: WatchEngine;
  /**
   * If set, the full watch list is persisted to this file path after every
   * create / update / delete operation.  Supports round-trip with
   * `WatchEngine.loadFromFile()` at startup.
   */
  persistPath?: string;
}

// ── Routes ────────────────────────────────────────────────────────────────────

export async function watchesRoutes(
  app: FastifyInstance,
  opts: WatchesRoutesOptions,
): Promise<void> {
  const { watchEngine, persistPath } = opts;

  /** Persist the current watch list, swallowing errors with a log. */
  async function persist(): Promise<void> {
    if (!persistPath) return;
    await watchEngine.saveToFile(persistPath).catch((err: unknown) => {
      app.log.error({ err }, "[watches] Failed to persist watch list");
    });
  }

  // ── GET /api/watches ──────────────────────────────────────────────────────

  app.get("/", async (_req, reply) => {
    return reply.send(watchEngine.listWatches());
  });

  // ── POST /api/watches ─────────────────────────────────────────────────────

  app.post<{
    Body: Omit<Watch, "id"> & Partial<Pick<Watch, "id">>;
  }>("/", async (req, reply) => {
    const body = req.body;

    if (!body.name || typeof body.name !== "string") {
      return reply.status(400).send({ error: "'name' (string) is required" });
    }
    if (!body.topic || typeof body.topic !== "string") {
      return reply.status(400).send({ error: "'topic' (string) is required" });
    }
    if (!body.condition || typeof body.condition !== "string") {
      return reply.status(400).send({ error: "'condition' (string) is required" });
    }
    if (!Array.isArray(body.channelIds)) {
      return reply
        .status(400)
        .send({ error: "'channelIds' (string[]) is required" });
    }

    const watch = createWatch({
      id: body.id,
      name: body.name,
      topic: body.topic,
      condition: body.condition,
      channelIds: body.channelIds as string[],
      cooldownMs: typeof body.cooldownMs === "number" ? body.cooldownMs : undefined,
      enabled: typeof body.enabled === "boolean" ? body.enabled : undefined,
    });

    watchEngine.addWatch(watch);
    await persist();

    return reply.status(201).send(watch);
  });

  // ── GET /api/watches/:id ──────────────────────────────────────────────────

  app.get<{ Params: { id: string } }>("/:id", async (req, reply) => {
    const watch = watchEngine.getWatch(req.params.id);
    if (!watch) {
      return reply.status(404).send({ error: `Watch "${req.params.id}" not found` });
    }
    return reply.send(watch);
  });

  // ── PATCH /api/watches/:id ────────────────────────────────────────────────

  app.patch<{
    Params: { id: string };
    Body: Partial<Omit<Watch, "id">>;
  }>("/:id", async (req, reply) => {
    const updated = watchEngine.updateWatch(req.params.id, req.body);
    if (!updated) {
      return reply.status(404).send({ error: `Watch "${req.params.id}" not found` });
    }
    await persist();
    return reply.send(updated);
  });

  // ── DELETE /api/watches/:id ───────────────────────────────────────────────

  app.delete<{ Params: { id: string } }>("/:id", async (req, reply) => {
    const watch = watchEngine.getWatch(req.params.id);
    if (!watch) {
      return reply.status(404).send({ error: `Watch "${req.params.id}" not found` });
    }
    watchEngine.removeWatch(req.params.id);
    await persist();
    return reply.status(204).send();
  });
}
